"""Main training script."""

import concurrent.futures
import dataclasses
import os
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from itertools import chain

import torch
import torch._dynamo
import zmq
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from plynder.core.config import Config
from plynder.train.data.batch_server import BatchServer
from plynder.train.data.collate_fn import collate_fn, reshape_padded_buffers
from plynder.train.data.dataset import LMDBDataset
from plynder.train.data.prefetch_loader import PrefetchLoader
from plynder.train.init_training import init_training
from plynder.train.losses import build_loss_kwargs, compute_cispo_loss, compute_rules_loss
from plynder.train.model_sender import ModelSender
from plynder.train.stats_collector import StatsCollector

logger = get_logger(__name__)

#: Hidden-state layer tapped by the rules-distillation head. This is the layer
#: the released model was trained with; changing it changes the head's input.
RULES_HEAD_HIDDEN_LAYER = 6


def main(cfg: Config) -> None:
    accelerator = init_training(cfg)
    _log_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    if accelerator.is_main_process:
        ctx = zmq.Context()

        model_sender = ModelSender(
            cfg.networking.model_pub,
            ctx,
            checkpoint_dir=cfg.paths.checkpoints,
            archive_dir=os.path.join(cfg.paths.checkpoints, "archive"),
            max_keep=3,
            archive_steps=cfg.training.checkpoint.archive_steps,
        )
        model_sender.start()

        stats_collector = StatsCollector(cfg.networking.stats_trainer_sub, ctx)
        stats_collector.start()

        server = BatchServer(
            networking=cfg.networking,
            buffers=cfg.buffers,
            lmdb=cfg.lmdb,
            batching=cfg.training.batching,
            num_replicas=accelerator.num_processes,
            num_workers=cfg.parallelism.dataloader_num_workers,
        )
        server.start()

        control_router = ctx.socket(zmq.ROUTER)
        control_router.bind(cfg.networking.control_router)

    accelerator.wait_for_everyone()

    global_step = 0

    logger.info(f"Loading config {cfg.paths.model_config}...")
    config = AutoConfig.from_pretrained(cfg.paths.model_config)
    model = AutoModelForCausalLM.from_config(config)
    model.requires_grad_(True)

    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=model.dtype)
    head.requires_grad_(True)

    muon_params = []
    adamw_params = []

    for name, param in chain(model.named_parameters(), head.named_parameters()):
        if not param.requires_grad:
            continue

        if "embed" in name or "lm_head" in name or "norm" in name.lower() or "bias" in name:
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    # Muon optimizer (matrix weights only)
    optimizer_muon = torch.optim.Muon(
        [
            {
                "params": muon_params,
                "lr": cfg.training.optimizer.learning_rate,
            }
        ],
        weight_decay=cfg.training.optimizer.adam_weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )

    # AdamW optimizer (biases, norms, embeddings, etc.)
    optimizer_adam = torch.optim.AdamW(
        [
            {
                "params": adamw_params,
                "lr": cfg.training.optimizer.learning_rate,
            }
        ],
        betas=(
            cfg.training.optimizer.adam_beta1,
            cfg.training.optimizer.adam_beta2,
        ),
        weight_decay=cfg.training.optimizer.adam_weight_decay,
        eps=cfg.training.optimizer.adam_epsilon,
        fused=True,
    )

    def lr_lambda(step):
        if (
            cfg.training.optimizer.lr_decay_steps is None
            or cfg.training.optimizer.start_learning_rate is None
        ):
            return 1.0

        if step < cfg.training.optimizer.lr_decay_steps:
            alpha = step / cfg.training.optimizer.lr_decay_steps
            return alpha + (1 - alpha) * (
                cfg.training.optimizer.start_learning_rate / cfg.training.optimizer.learning_rate
            )
        return 1.0

    scheduler_adam = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda)
    scheduler_muon = torch.optim.lr_scheduler.LambdaLR(optimizer_muon, lr_lambda)

    model, optimizer_muon, optimizer_adam = accelerator.prepare(
        model, optimizer_muon, optimizer_adam
    )
    head = accelerator.prepare(head)

    @contextmanager
    def no_sync_context():
        with (
            model.no_sync() if hasattr(model, "no_sync") else nullcontext(),
            head.no_sync() if hasattr(head, "no_sync") else nullcontext(),
        ):
            yield

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.assume_static_by_default = False
    model.train()
    model.compile(dynamic=True)
    head.train()
    head.compile(dynamic=True)

    # ── Datasets & DataLoaders ──────────────────────────────
    dataloader = DataLoader(
        dataset=LMDBDataset(
            rank=accelerator.process_index,
            db_path=cfg.lmdb.db_path,
            db_map_size=cfg.lmdb.map_size,
            sampler_address=cfg.networking.microbatch_assignment_dealer,
        ),
        batch_size=None,  # BatchServer owns all grouping logic
        num_workers=cfg.parallelism.dataloader_num_workers,
        collate_fn=lambda x: collate_fn(
            *x,
            microbatch_max_tokens=cfg.training.batching.microbatch_max_tokens,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
        ),
        pin_memory=True,
    )

    dataloader = PrefetchLoader(dataloader, accelerator.device)

    if accelerator.is_main_process:
        os.environ["MLFLOW_TRACKING_URI"] = cfg.paths.mlflow_tracking_uri
        _log_executor.submit(
            accelerator.init_trackers,
            cfg.experience_name,
            config=dataclasses.asdict(cfg),
            init_kwargs={
                "mlflow": {
                    "run_name": f"{cfg.experience_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            },
        )

    if cfg.networking.synchronize and accelerator.is_main_process:
        identities = {}

        logger.info("Waiting for rollout and evaluation to be ready...")
        while len(identities) < 2:
            identity, raw_topic = control_router.recv_multipart()
            topic = raw_topic.decode()
            assert topic in (cfg.networking.rollout_topic, cfg.networking.eval_topic)
            identities[topic] = identity

        logger.info("Sending model to rollout and evaluation...")
        model_sender.send(
            accelerator.unwrap_model(model), cfg.networking.rollout_topic, global_step
        )
        model_sender.send(accelerator.unwrap_model(model), cfg.networking.eval_topic, global_step)

    accelerator.wait_for_everyone()

    logger.info("Running warmup to trigger torch.compile...")
    torch._dynamo.config.cache_size_limit = 64
    from plynder.train.warmup import run_warmup

    run_warmup(
        model,
        head,
        optimizer_muon,
        optimizer_adam,
        accelerator,
        cfg,
        config.vocab_size,
        no_sync_context,
        hidden_layer_index=6,
    )
    logger.info("Warmup done.")

    # ── Profiler setup (if enabled) ────────────────────────
    if cfg.profiler.enabled:
        profiler_output_dir = cfg.profiler.output_dir or cfg.paths.logs
        os.makedirs(profiler_output_dir, exist_ok=True)

        profiler_activities = []
        for act in cfg.profiler.activities:
            profiler_activities.append(getattr(ProfilerActivity, act.upper()))

        profiler_schedule = schedule(
            wait=cfg.profiler.wait_steps,
            warmup=cfg.profiler.warmup_steps,
            active=cfg.profiler.active_steps,
            repeat=1,
        )

        profiler_ctx = profile(
            activities=profiler_activities,
            schedule=profiler_schedule,
            record_shapes=cfg.profiler.record_shapes,
            profile_memory=cfg.profiler.profile_memory,
            with_stack=cfg.profiler.with_stack,
            with_flops=cfg.profiler.with_flops,
        )
        logger.info(
            f"Profiler enabled. Waiting {cfg.profiler.wait_steps}, "
            f"warming up {cfg.profiler.warmup_steps}, "
            f"recording {cfg.profiler.active_steps} steps. "
            f"Output dir: {profiler_output_dir}"
        )
    else:
        profiler_ctx = nullcontext()

    time_start_training = None

    with profiler_ctx as prof:
        for batch, sync_step in (progress_bar := tqdm(dataloader, desc="Train", disable=True)):
            time_start_training = time_start_training or time.time()

            batch = reshape_padded_buffers(batch) # Avoid memory fragmentation
            context = nullcontext if sync_step else no_sync_context

            with context():
                input_ids = batch["token_ids"][:, :-1]  # We remove the final prediction
                output_ids = batch["token_ids"][:, 1:]  # We remove the bos token

                output = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )

                # ── rules loss (auxiliary head on an intermediate layer) ─────────
                logits_rules = head(output.hidden_states[RULES_HEAD_HIDDEN_LAYER])
                loss_kl_div = compute_rules_loss(
                    logits_rules,
                    batch["logits_mask"],
                    batch["rules_informative_tokens"],
                )

                # ── Policy loss (CISPO) ──────────────────────────────────────────
                loss_kwargs = build_loss_kwargs(cfg, batch, output_ids, output.logits)
                loss, clipped_ratio = compute_cispo_loss(**loss_kwargs)

                total_loss = loss + loss_kl_div

                scaled_loss = total_loss * (
                    batch["token_ids"].shape[0]
                    / cfg.training.batching.total_batch_size
                    * accelerator.num_processes
                )  # For grad acc
                accelerator.backward(scaled_loss)

                if sync_step:
                    if cfg.training.optimizer.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(
                            model.parameters(), cfg.training.optimizer.gradient_clipping
                        )
                    optimizer_muon.step()
                    optimizer_adam.step()
                    scheduler_muon.step()
                    scheduler_adam.step()
                    optimizer_muon.zero_grad()
                    optimizer_adam.zero_grad()

                    global_step += 1

                    if (
                        global_step % cfg.training.send_model_frequency == 0
                        and accelerator.is_main_process
                    ):
                        model_sender.send(
                            accelerator.unwrap_model(model),
                            cfg.networking.rollout_topic,
                            global_step,
                        )

                    if accelerator.is_main_process and (
                        eval_stats := stats_collector.get_evaluation()
                    ):
                        model_sender.send(
                            accelerator.unwrap_model(model), cfg.networking.eval_topic, global_step
                        )
                        step = int(eval_stats.pop("step"))
                        _log_executor.submit(accelerator.log, eval_stats, step=step)

                    if (
                        global_step % cfg.training.checkpoint.save_steps == 0
                        and accelerator.is_main_process
                    ):
                        model_sender.save(
                            global_step,
                            model,
                            head,
                            [optimizer_adam, optimizer_muon],
                            accelerator,
                        )

                    stop = torch.tensor(False, device=accelerator.device)
                    if accelerator.is_main_process:
                        stop = torch.tensor(
                            (time.time() - time_start_training) > cfg.training.training_time,
                            device=accelerator.device,
                        )
                    stop = broadcast(stop)

                    if stop:
                        break

                    if cfg.profiler.enabled:
                        prof.step()
                        # Early exit after profiling is complete
                        if prof.current_action == torch.profiler.ProfilerAction.NONE:
                            logger.info(
                                f"Profiler completed after step {global_step}. "
                                "Exiting training early."
                            )
                            break

                    logs = {"policy_loss": loss.detach().item()}
                    if global_step > 0:
                        logs["lr"] = scheduler_adam.get_last_lr()[0]
                    logs["rules_loss"] = loss_kl_div.detach().item()

                    progress_bar.set_postfix(**logs, step=global_step)
                    _log_executor.submit(
                        accelerator.log,
                        dict(
                            **logs,
                            clipped_ratio=clipped_ratio.detach().item(),
                            **(
                                stats_collector.get_rollout() if accelerator.is_main_process else {}
                            ),
                        ),
                        step=global_step,
                    )

        if cfg.profiler.enabled and accelerator.is_main_process:
            # Export Chrome trace
            prof_output_dir = cfg.profiler.output_dir or cfg.paths.logs
            os.makedirs(prof_output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = os.path.join(
                prof_output_dir,
                f"trace_{timestamp}.json",
            )
            prof.export_chrome_trace(trace_path)
            logger.info(f"Profiler trace exported to {trace_path}")

    if accelerator.is_main_process:
        logger.info("Training Loop ended. Sending model to evaluation and saving to disk")
        model_sender.send(accelerator.unwrap_model(model), cfg.networking.eval_topic, global_step)
        model_sender.save(global_step, model, head, [optimizer_adam, optimizer_muon], accelerator)

        logger.info("Waiting last evaluation stats")
        eval_stats, step = None, -1
        while step != global_step:
            while not eval_stats:
                time.sleep(0.1)
                eval_stats = stats_collector.get_evaluation()

            step = int(eval_stats.pop("step"))
            _log_executor.submit(accelerator.log, eval_stats, step=step)
            eval_stats = None

        if cfg.networking.synchronize:
            logger.info("Shutting down evaluation and rollout")
            control_router.send_multipart([identities[cfg.networking.eval_topic], b""])
            control_router.send_multipart([identities[cfg.networking.rollout_topic], b""])

        server.shutdown()
        stats_collector.shutdown()
        model_sender.shutdown()
        control_router.close()
        ctx.term()
        logger.info("Shutting down all internal services")

    accelerator.end_training()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    from plynder.core import setup_logging
    from plynder.core.config.hydra import from_hydra_config

    @hydra.main(
        version_base=None,
        config_path="../../../configs",
        config_name="config",
    )
    def train_app(cfg: DictConfig) -> None:
        """Main training entry point with Hydra."""
        # Convert Hydra config to dataclass
        config = from_hydra_config(cfg)

        # Setup logging
        if config.log_file is not None:
            config.log_file = (
                config.log_file + "_train_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
            )
        setup_logging(config.log_file)

        main(config)

    train_app()
