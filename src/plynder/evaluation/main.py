"""
Eval worker pool using multiprocessing.

Architecture
------------
  main process
    ├── ZMQ subscriber: receives model weights each training step
    ├── saves weights to a temp file (avoids large-payload queue issues)
    ├── sends (step, weights_path, n_games, sf_temp) to each worker
    │   via a per-worker mp.Queue
    └── collects results, computes ranking, publishes via ZMQ

  worker process  (evaluation.num_workers)
    ├── owns its own model instance (loaded from AutoConfig at startup)
    ├── owns a persistent EnginePool of eval.concurrent_per_worker Stockfish engines
    ├── each step: loads new weights from temp file, plays n_games asynchronously
    └── pushes individual game result dicts to the shared result_queue

Shutdown contract
-----------------
  Two-layer signal handling in each worker:

  1. Outer scope (signal.signal):  fires when the worker is blocked on
     job_queue.get().  The handler raises SystemExit, which unwinds the
     try/finally and lets the worker exit without leaving zombie engines.

  2. Inner scope (loop.add_signal_handler):  fires while the asyncio event
     loop is running a game batch.  It cancels the current task so that
     CancelledError propagates through every `finally` block in the async
     stack (EnginePool.shutdown in particular), then re-raises so the outer
     scope sees it as a clean exit.

  The main process catches (SystemExit, KeyboardInterrupt) in its own
  try/finally, which calls the single _cleanup() function that sends SIGTERM
  to workers, joins them with a timeout, hard-kills stragglers, and releases
  all ZMQ and Manager resources.
"""

import asyncio
import contextlib
import io
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import struct
import sys
import tempfile
import time
from typing import Any

import chess
import chess.engine
import choix
import numpy as np
import torch
import zmq
from plynder_rs import SequenceStatesAsync, TerminalTokens
from transformers import AutoConfig, AutoModelForCausalLM

from plynder.core import TOKEN_ID_TO_UCI, SpecialTokens, setup_logging
from plynder.core.config import Config
from plynder.evaluation.chess_utils import (
    UCI_TO_TOKEN_ID,
    EnginePool,
    safe_uci,
)

logger = logging.getLogger(__name__)

#: Stockfish search depth for evaluation games.
EVAL_SEARCH_DEPTH = 5

#: Maximum number of moves to analyse with multipv (top-K candidate moves).
EVAL_MULTIPV_CAP = 20


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def load_ranking_map(ranking_json_path: str) -> dict[float, float]:
    """Load the Stockfish temperature → Elo ranking calibration map."""
    with open(ranking_json_path) as f:
        data = json.load(f)
    return {float(k): float(v) for k, v in data.items()}


def ranking_to_temp(target_ranking: float, ranking_map: dict[float, float]) -> tuple[float, float]:
    """Return (temp, ranking) pair whose ranking is closest to target_ranking."""
    best_temp = min(ranking_map, key=lambda t: abs(ranking_map[t] - target_ranking))
    return best_temp, ranking_map[best_temp]


def compute_model_ranking(
    wins: int,
    losses: int,
    draws: int,
    anchor_ranking: float,
) -> float:
    """Bradley-Terry ranking estimate anchored to a known Stockfish strength."""
    game_data = [(0, 1)] * wins + [(1, 0)] * losses
    if not game_data:
        logger.warning("No decisive games; returning anchor ranking unchanged.")
        return anchor_ranking
    params = choix.ilsr_pairwise(2, game_data, alpha=0.01)
    return float(anchor_ranking + 400.0 * (params[0] - params[1]) / np.log(10))


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------


async def play_game(
    token_ids: list[int],
    temperature: float,
    engine_pool: EnginePool,
    stockfish_temperature: float,
    model: torch.nn.Module,
    uci_to_token_id: dict[str, int],
    token_id_to_uci: dict[int, str],
) -> dict[str, Any]:
    """Play a single game between the model and Stockfish.

    The model's color and the opening position are chosen at random.  The
    Rust ``SequenceStatesAsync`` enforces legal-move masks for both players.
    Returns a dict with ``model_color``, ``result``, ``termination``, and
    ``num_moves``.
    """
    sequence_state = SequenceStatesAsync(
        vocab_size=model.config.vocab_size,
        terminal_tokens=TerminalTokens(
            draw_id=SpecialTokens.DRAW,
            terminal_token_win_0=SpecialTokens.WHITE_WIN,
            terminal_token_win_1=SpecialTokens.BLACK_WIN,
        ),
        device=None,
        rollout_address=None,
        global_engine_id=None,
    )

    model_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
    token_ids[0] = SpecialTokens.BOS

    board = chess.Board()

    for tok in token_ids[1:]:
        uci = token_id_to_uci[tok]
        board.push_uci(uci)

    sequence_state.add_sequence("seq_id", token_ids)
    past_key_values = None
    turn = 0

    async with engine_pool.acquire() as sf:
        while True:
            sequence_state.spawn_valid_tokens_vec(["seq_id"])
            valid_moves = sequence_state.join_get_valid_tokens_vec()[0]

            if valid_moves[0] in (
                SpecialTokens.DRAW,
                SpecialTokens.WHITE_WIN,
                SpecialTokens.BLACK_WIN,
            ):
                break

            if board.turn == model_color:
                with torch.inference_mode():
                    input_tensor = (
                        torch.tensor([token_ids], dtype=torch.long)
                        if past_key_values is None
                        else torch.tensor([token_ids[-2:]], dtype=torch.long)
                    )
                    output = model(
                        input_ids=input_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
                    last_logits = output.logits[0, -1, :]
                    valid_move_tensor = torch.tensor(valid_moves, dtype=torch.long)
                    valid_logits = last_logits.index_select(0, valid_move_tensor)
                    scaled = valid_logits / temperature - valid_logits.max() / temperature
                    new_token_id = valid_moves[torch.multinomial(torch.exp(scaled), 1).item()]

                sequence_state.apply_sampled("seq_id", [new_token_id], 0)
                token_ids.append(new_token_id)
                new_uci = token_id_to_uci[new_token_id]
                board.push_uci(new_uci)

            else:
                multipv = min(EVAL_MULTIPV_CAP, len(valid_moves))
                infos = await sf.analyse(
                    board,
                    chess.engine.Limit(depth=EVAL_SEARCH_DEPTH),
                    multipv=multipv,
                )
                moves = [info["pv"][0] for info in infos]
                scores = np.array(
                    [info["score"].relative.score(mate_score=100_000) for info in infos],
                    dtype=np.float64,
                )
                scaled = scores / stockfish_temperature
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
                move = np.random.choice(moves, p=probs)
                new_uci = safe_uci(move, board)
                new_token_id = uci_to_token_id[new_uci]
                board.push_uci(new_uci)
                sequence_state.apply_sampled("seq_id", [new_token_id], 0)
                token_ids.append(new_token_id)

            turn += 1

    outcome = board.outcome(claim_draw=True)
    return {
        "model_color": "White" if model_color == chess.WHITE else "Black",
        "result": outcome.result() if outcome else "*",
        "termination": str(outcome.termination) if outcome else "unknown",
        "num_moves": turn,
    }


# ---------------------------------------------------------------------------
# Async batch runs inside one worker's event loop for a single eval step
# ---------------------------------------------------------------------------


async def _async_worker_batch(
    model: torch.nn.Module,
    openings: list[list[int]],
    stockfish_path: str,
    sf_temp: float,
    eval_temperature: float,
    concurrent_per_worker: int,
    num_games: int,
    result_queue: mp.Queue,
) -> None:
    """Play ``num_games`` concurrent games within one worker's asyncio event loop."""
    pool = EnginePool(
        stockfish_path,
        size=concurrent_per_worker,
        sf_config={"Threads": "1", "Hash": "16"},
    )
    await pool.start()

    # While the event loop owns the thread, replace the outer sys.exit signal
    # handlers with ones that cancel this coroutine.  CancelledError then
    # flows through every finally block in the async call stack, guaranteeing
    # pool.shutdown() below always runs.
    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()

    def _request_cancel():
        current_task.cancel()

    loop.add_signal_handler(signal.SIGTERM, _request_cancel)
    loop.add_signal_handler(signal.SIGINT, _request_cancel)

    sem = asyncio.Semaphore(concurrent_per_worker)

    try:

        async def guarded() -> None:
            async with sem:
                try:
                    r = await play_game(
                        token_ids=random.choice(openings)[:],
                        temperature=eval_temperature,
                        engine_pool=pool,
                        stockfish_temperature=sf_temp,
                        model=model,
                        uci_to_token_id=UCI_TO_TOKEN_ID,
                        token_id_to_uci=TOKEN_ID_TO_UCI,
                    )
                except asyncio.CancelledError:
                    raise  # don't swallow cancellation
                except Exception as exc:
                    r = exc
                result_queue.put(r)

        await asyncio.gather(*[guarded() for _ in range(num_games)])

    finally:
        # Remove our handlers so the process-level handlers are restored
        # once asyncio.run() returns.
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)
        # Always shut down engines; this runs whether we finished, crashed, or
        # were cancelled.
        await pool.shutdown()


# ---------------------------------------------------------------------------
# Worker process, persistent across steps, one process per worker
# ---------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    model_config_path: str,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    stockfish_path: str,
    concurrent_per_worker: int,
    eval_temperature: float,
    openings_jsonl: str,
) -> None:
    """Persistent worker process: loads weights per step and plays game batches."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[worker-{worker_id}] %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    openings: list[list[int]] = []
    with open(openings_jsonl) as f:
        for line in f:
            if line.strip():
                openings.append(json.loads(line.strip()))

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    config = AutoConfig.from_pretrained(model_config_path)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    try:
        while True:
            job = job_queue.get()  # blocks; interrupted by SystemExit on signal
            step, weights_path, n_games, sf_temp = job
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            asyncio.run(
                _async_worker_batch(
                    model=model,
                    openings=openings,
                    stockfish_path=stockfish_path,
                    sf_temp=sf_temp,
                    eval_temperature=eval_temperature,
                    concurrent_per_worker=concurrent_per_worker,
                    num_games=n_games,
                    result_queue=result_queue,
                )
            )

    except (SystemExit, KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker %d received shutdown signal, exiting.", worker_id)
    except Exception:
        logger.exception("Worker %d crashed.", worker_id)
        raise
    finally:
        logger.info("Worker %d done.", worker_id)


# ---------------------------------------------------------------------------
# Centralised cleanup, called from the main process finally block only
# ---------------------------------------------------------------------------


def _cleanup(
    workers: list[mp.Process],
    zmq_sockets: list[zmq.Socket],
    ctx: zmq.Context,
    worker_join_timeout: float = 30.0,
) -> None:
    """
    Single authority for all teardown.  Safe to call from a finally block.

    Order:
      1. Ask each worker to stop via SIGTERM (they handle it themselves).
      2. Join with a shared deadline; hard-kill any that overstay.
      3. Close ZMQ sockets (linger=0 so they don't block on unsent messages).
      4. Terminate the ZMQ context.
    """
    logger.info("Cleanup: sending SIGTERM to %d workers.", len(workers))
    for w in workers:
        if w.is_alive():
            with contextlib.suppress(ProcessLookupError):
                os.kill(w.pid, signal.SIGTERM)

    logger.info("Cleanup: joining %d workers.", len(workers))
    deadline = time.monotonic() + worker_join_timeout
    for w in workers:
        remaining = max(0.0, deadline - time.monotonic())
        w.join(timeout=remaining)
        if w.is_alive():
            logger.warning("Worker %d (pid %d) did not stop in time; killing.", w.pid, w.pid)
            w.kill()
            w.join()

    for sock in zmq_sockets:
        with contextlib.suppress(Exception):
            sock.close(linger=0)

    with contextlib.suppress(Exception):
        ctx.term()

    logger.info("Cleanup complete.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def global_main(cfg: Config) -> None:
    mp.set_start_method("spawn", force=True)

    num_workers = cfg.evaluation.num_workers
    concurrent_per_worker = cfg.evaluation.concurrent_per_worker
    num_matches = cfg.evaluation.num_matches

    ranking_map = load_ranking_map(cfg.paths.rankings_json)
    logger.info(
        "Loaded ranking map: %d temperature entries, range [%.1f, %.1f]",
        len(ranking_map),
        min(ranking_map.values()),
        max(ranking_map.values()),
    )

    job_queues = [mp.Queue() for _ in range(num_workers)]
    result_queue = mp.Queue()

    # Non-daemon: workers must be joined so they can run their own cleanup.
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=_worker_main,
            args=(
                wid,
                cfg.paths.model_config,
                job_queues[wid],
                result_queue,
                cfg.paths.stockfish_path,
                concurrent_per_worker,
                cfg.evaluation.temperature,
                cfg.paths.openings_jsonl,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    logger.info("Started %d eval worker processes.", num_workers)

    ctx = zmq.Context()
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(cfg.networking.model_sub)
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, cfg.networking.eval_topic)

    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.connect(cfg.networking.stats_trainer_pub)

    control_dealer = ctx.socket(zmq.DEALER)
    control_dealer.connect(cfg.networking.control_dealer)
    control_dealer.send(cfg.networking.eval_topic.encode())
    logger.info("Evaluation started, topic sent to control dealer.")

    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)
    poller.register(control_dealer, zmq.POLLIN)

    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    current_ranking = cfg.evaluation.start_ranking

    def _collect_game_results(expected: int) -> tuple[int, int, int, int, int]:
        """
        Collect exactly `expected` results from the shared queue.
        """
        wins = losses = draws = errors = total_moves = 0
        for _ in range(expected):
            r = result_queue.get()
            if isinstance(r, Exception):
                logger.error("Game error: %s", r)
                errors += 1
                continue
            white = r["model_color"] == "White"
            total_moves += r["num_moves"]
            res = r["result"]
            if res == "1-0":
                if white:
                    wins += 1
                else:
                    losses += 1
            elif res == "0-1":
                if white:
                    losses += 1
                else:
                    wins += 1
            elif "1/2" in res:
                draws += 1
            else:
                errors += 1
        return wins, losses, draws, errors, total_moves

    try:
        while True:
            ready = dict(poller.poll())
            if not ready:
                continue

            if control_dealer in ready:
                logger.info("Received shutdown signal via control socket.")
                control_dealer.recv()
                break

            if sub_socket not in ready:
                continue

            topic, step_bytes, msg = sub_socket.recv_multipart()
            step = struct.unpack("<i", step_bytes)[0]

            sf_temp, anchor_ranking = ranking_to_temp(current_ranking, ranking_map)

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                weights_path = f.name
            state_dict = torch.load(io.BytesIO(msg), map_location="cpu")
            torch.save(state_dict, weights_path)

            base, remainder = divmod(num_matches, num_workers)
            game_counts = [base + (1 if i < remainder else 0) for i in range(num_workers)]
            total_games = sum(game_counts)

            t0 = time.perf_counter()
            for wid, n in enumerate(game_counts):
                job_queues[wid].put((step, weights_path, n, sf_temp))

            try:
                wins, losses, draws, errors, total_moves = _collect_game_results(total_games)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(weights_path)

            elapsed = time.perf_counter() - t0

            total = wins + losses + draws
            model_ranking = max(0, compute_model_ranking(wins, losses, draws, anchor_ranking))
            current_ranking = model_ranking

            logger.info(
                "Step %d | sf_temp=%.1f anchor=%.1f | W:%d L:%d D:%d Err:%d"
                " | WinRate=%.1f%% Score=%.3f | model_ranking=%.1f"
                " | %.0f moves in %.1fs (%.1f g/s)",
                step,
                sf_temp,
                anchor_ranking,
                wins,
                losses,
                draws,
                errors,
                100 * wins / max(total, 1),
                (2 * wins + draws) / max(2 * total, 1),
                model_ranking,
                total_moves,
                elapsed,
                total_games / elapsed if elapsed > 0 else 0,
            )

            pub_socket.send_json(
                {
                    "step": step,
                    "eval_wins": wins,
                    "eval_draws": draws,
                    "eval_losses": losses,
                    "model_ranking": model_ranking,
                    "stockfish_ranking": anchor_ranking,
                }
            )

    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutdown requested.")
    finally:
        _cleanup(
            workers=workers,
            zmq_sockets=[sub_socket, pub_socket, control_dealer],
            ctx=ctx,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    import hydra
    from omegaconf import DictConfig

    from plynder.core.config.hydra import from_hydra_config

    @hydra.main(
        version_base=None,
        config_path="../../../configs",
        config_name="config",
    )
    def eval_app(cfg: DictConfig) -> None:
        config = from_hydra_config(cfg)
        if config.log_file is not None:
            config.log_file = (
                config.log_file + "_eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
            )
        setup_logging(config.log_file)
        global_main(config)

    eval_app()
