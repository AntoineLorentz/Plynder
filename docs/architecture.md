# Architecture

Detailed module-level reference for Plynder's distributed chess RL system.
For a high-level overview, see the [README](../README.md).

## System design

Three long-running processes communicate over ZeroMQ:

- **PUB/SUB**: model weight broadcasts (trainer → rollout/eval), stats
- **PUSH/PULL**: trajectory data (rollout → trainer)
- **ROUTER/DEALER**: microbatch task distribution (trainer → data workers)

All inter-process payloads use PyArrow IPC serialization. Trajectory data is
persisted in an LMDB ring buffer on `/dev/shm` for low-latency access.

## Entry points

All three entry points use Hydra with `config_path="../../../configs"`
(resolved relative to the entry-point file) and `config_name="config"`. The
Hydra `DictConfig` is converted to a typed dataclass hierarchy via
`core/config/hydra.py:from_hydra_config()`.

| Entry | File | How to run |
|-------|------|------------|
| Training | `src/plynder/train/main.py` | `uv run accelerate launch --config_file configs/accelerate.yaml src/plynder/train/main.py` |
| Rollout | `src/plynder/rollout/main.py` | `uv run python src/plynder/rollout/main.py` |
| Evaluation | `src/plynder/evaluation/main.py` | `uv run python src/plynder/evaluation/main.py` |

## Core Python modules

### Configuration (`core/config/`)

A dataclass hierarchy mirroring the Hydra YAML structure:

- `base.py`: Root `Config` dataclass with `validate()`
- `networking.py`: ZMQ ports, topics, trainer address
- `paths.py`: Filesystem paths (model config, tokenizer, checkpoints, logs)
- `infrastructure.py`: Buffer sizes, LMDB config, parallelism
- `training.py`: Optimizer, batching, checkpoint, and CISPO hyperparameters
- `rollout.py`: vLLM engine config, sampling parameters
- `eval.py`: Evaluation config (match count, concurrency, Stockfish settings)
- `profiler.py`: `torch.profiler` configuration

### Utilities (`core/utils.py`, `core/special_tokens.py`)

- `TOKEN_ID_TO_UCI`: Dict mapping token IDs → UCI move strings
- `generate_all_moves()`: Deterministic sorted list of all possible chess moves
- `SpecialTokens` IntEnum: BOS, DRAW, WHITE_WIN, BLACK_WIN
- `to_pgn()`: Convert a token-ID sequence to a `chess.pgn.Game`

## Rollout (`rollout/`)

A multi-process architecture where one main process spawns:

1. **`model_receiver.py`**: ZMQ SUB for model updates from the trainer. Writes
   weights to shared memory (3-slot circular buffer) and publishes shm metadata
   via ZMQ so vLLM workers can reload without restarting.
2. **`sender.py`**: Pulls trajectory data from workers and the Rust backend,
   serializes to PyArrow IPC, pushes to the trainer via ZMQ.
3. **`Worker` (`async_worker.py`)**: One per vLLM engine. Runs vLLM `AsyncLLM`,
   spawns `Group` objects for group-relative sampling, and listens for model updates.
4. **`async_group.py`**: `Group` wraps a vLLM request: samples an opening
   position, generates `group_size` continuations, and sends results to the
   Sender.
5. **`stats_pub.py`**: Accumulates and publishes win rates, latencies, and
   throughput via ZMQ.

### vLLM plugin (`rollout/vllm_plugin/`)

Version-specific patches (currently `0_18_1/`) that hook into vLLM internals
(model runner, GPU worker, sampler) to inject the Rust `SequenceStates` legal-
move mask during sampling. The plugin version is selected at runtime based on
`vllm.__version__`.

## Training (`train/`)

### Training loop (`main.py`)

1. Loads `Qwen3ForCausalLM` + a rules-distillation head network
2. Two optimizers: **Muon** (2D matrix weights) + **AdamW** (biases, norms,
   embeddings)
3. Linear warmup + constant LR scheduler
4. CISPO policy loss with per-group advantage normalization (white/black
   separately)
5. KL rules distillation: the head learns to mask illegal moves from an
   intermediate hidden state (`RULES_HEAD_HIDDEN_LAYER = 6`). Rules distillation
   reaches the same strength target with 7.7x less elapsed training time in the
   matched comparison.
6. `ModelSender` broadcasts weights to rollout; `StatsCollector` aggregates
   rollout stats and evaluation results

### Loss functions (`losses.py`)

- **`compute_cispo_loss`**: CISPO (Clipped Importance-Sampling weight
  Policy Optimization, MiniMax-M1 Eq.4-5): REINFORCE-style objective where the
  importance-sampling ratio is clipped, detached (stop-gradient) and used as a
  constant multiplier. With `cispo_use_mask` (the released model's recipe), the
  Eq.7 IS mask zeroes the gradient on over-confident good tokens (A>0,
  r > 1+eps_high) and abandoned bad tokens (A≤0, r < 1-eps_low). Gradients
  flow through the policy logprobs only.
- **`compute_rules_loss`**: KL divergence between the auxiliary rules head
  and the rules-derived teacher (legal-move masks).
- **`build_loss_kwargs`**: assembles the CISPO kwargs from a collated batch
  and the training config.

### Data pipeline (`train/data/`)

- `receiver.py`: ZMQ PULL from rollout → stores in LMDB ring buffer
- `batch_server.py`: Coordinates the Receiver + `BatchSampler`. Builds
  microbatches by sorting slots by length (minimizing padding) and distributes
  them via ZMQ ROUTER.
- `dataset.py`: `LMDBDataset` (IterableDataset) reads from LMDB via a ZeroMQ
  sampler with zero-copy PyArrow deserialization.
- `collate_fn.py`: Pads examples, computes group-relative advantages, constructs logits
  masks for illegal-move prohibition, handles attention masks and rules-
  informative tokens.
- `prefetch_loader.py`: One-batch-ahead prefetch with async GPU transfer.

## Evaluation (`evaluation/`)

- `play_game()`: A single game between the model and Stockfish. Uses the Rust
  `SequenceStatesAsync` for valid-move masks. Color and opening are randomized.
- `_worker_main()` / `_async_worker_batch()`: Persistent worker processes;
  each owns a pool of Stockfish engines and plays batches of concurrent games
  (asyncio semaphore-limited) for every received checkpoint.
- `global_main()`: Subscribes to model updates, distributes games to workers,
  computes the Bradley-Terry `model_ranking` anchored on the Stockfish
  temperature calibration (`rankings.json`), and publishes results.

## Rust extension (`plynder_rs`)

The Rust library at `rust_extension/` provides the high-performance chess rules
engine, exposed to Python via PyO3:

- **`lib.rs`**: PyO3 entry point. Exports `TerminalTokens` and
  `SequenceStatesAsync` (an async wrapper around `SequenceStatesPool` with a
  tokio runtime).
- **`sequences_states_pool.rs`**: Core `SequenceStatesPool` with
  `add_sequence()`, `valid_tokens()`, `all_valid_tokens_mask()`,
  `apply_sampled()`. Maintains an `AHashMap` of chess boards per sequence,
  handles legal-move computation and repetition detection. Constructs a mask in
  pinned memory and transfers it to the GPU.
- **`repetition_tracker.rs`**: Tracks threefold repetition per board hash
  (`u64 → u8`).
- **`writer.rs`**: Serializes sequence data (info header + logprobs + allowed
  tokens) for ZMQ sending.
- **`utils.rs`**: Token ID ↔ UCI move tables, insufficient-material detection,
  mask construction.

The extension is built automatically by `uv` (via `maturin`) when a source file
changes.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/bench_rollout.py` | vLLM benchmark suite comparing engine/concurrency configs |
| `scripts/generate_openings.py` | Regenerates the committed opening book when needed |
| `scripts/stockfish_tournament.py` | Runs a Stockfish tournament to calibrate Elo rankings → `rankings.json` |
| `scripts/create/` | Model and tokenizer creation utilities |
