"""
Stockfish temperature tournament using multiprocessing.

Architecture
------------
  main process
    ├── splits all (opening, t0, t1) game specs across N worker processes
    ├── streams completed-game results through a mp.Queue
    ├── prints live throughput (games/s, ETA, progress bar)
    └── logs every experiment run to experiments.jsonl

  worker process  (x num_workers)
    ├── owns its own asyncio event loop
    ├── owns a pool of `concurrent_per_worker` Stockfish engines (never spawned/quit per game)
    └── pushes each result to the shared queue as soon as the game finishes

Tuning guide
------------
  Total Stockfish threads ≈ num_workers x concurrent_per_worker
  Target this to equal your machine's logical CPU count for saturation.
  e.g. 16-core machine → num_workers=4, concurrent_per_worker=4  (16 engines total)

  Lower depth / multipv_cap for fast hypothesis testing; raise for publication runs.
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.engine
import choix
import numpy as np
import torch  # noqa: F401
from plynder_rs import SequenceStatesAsync, TerminalTokens

from plynder.core import TOKEN_ID_TO_UCI, SpecialTokens
from plynder.evaluation.chess_utils import (
    UCI_TO_TOKEN_ID,
    EnginePool,
    safe_uci,
)

logger = logging.getLogger(__name__)

DEFAULT_MULTIPV_CAP = 20  # top-K moves to analyse; tail moves have ~0 prob


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------


async def play_game(
    token_ids: list[int],
    engine_pool: EnginePool,
    temperature_player_0: float,
    temperature_player_1: float,
    depth: int,
    multipv_cap: int,
) -> dict[str, Any]:
    sequence_state = SequenceStatesAsync(
        vocab_size=1972,
        terminal_tokens=TerminalTokens(
            draw_id=SpecialTokens.DRAW,
            terminal_token_win_0=SpecialTokens.WHITE_WIN,
            terminal_token_win_1=SpecialTokens.BLACK_WIN,
        ),
        device=None,
        rollout_address=None,
        global_engine_id=None,
    )

    board = chess.Board()
    for tok in token_ids[1:]:
        board.push_uci(TOKEN_ID_TO_UCI[tok])

    sequence_state.add_sequence("seq_id", token_ids)

    turn = 0
    temp = temperature_player_0

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

            # Materialise legal moves once; cap multipv to top-K.
            multipv = min(multipv_cap, len(valid_moves))

            infos = await sf.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv,
            )

            moves = [info["pv"][0] for info in infos]
            scores = np.array(
                [info["score"].relative.score(mate_score=100_000) for info in infos],
                dtype=np.float64,
            )

            scaled = scores / temp
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()

            move = np.random.choice(moves, p=probs)
            new_uci = safe_uci(move, board)
            new_token_id = UCI_TO_TOKEN_ID[new_uci]
            board.push_uci(new_uci)

            sequence_state.apply_sampled("seq_id", [new_token_id], 0)
            token_ids.append(new_token_id)
            turn += 1
            temp = temperature_player_0 if turn % 2 == 0 else temperature_player_1

    outcome = board.outcome(claim_draw=True)
    return {
        "temperature_player_0": temperature_player_0,
        "temperature_player_1": temperature_player_1,
        "result": outcome.result() if outcome else "*",
        "termination": str(outcome.termination) if outcome else "unknown",
        "num_moves": turn,
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

GameSpec = tuple[list[int], float, float]  # (token_ids, temp_0, temp_1)


def _worker(
    game_specs: list[GameSpec],
    stockfish_path: str,
    depth: int,
    multipv_cap: int,
    concurrent_per_worker: int,
    result_queue: mp.Queue,
) -> int:
    """Entry point for each subprocess. Returns number of games completed."""
    return asyncio.run(
        _async_worker(
            game_specs,
            stockfish_path,
            depth,
            multipv_cap,
            concurrent_per_worker,
            result_queue,
        )
    )


async def _async_worker(
    game_specs: list[GameSpec],
    stockfish_path: str,
    depth: int,
    multipv_cap: int,
    concurrent_per_worker: int,
    result_queue: mp.Queue,
) -> int:
    pool = EnginePool(
        stockfish_path,
        size=concurrent_per_worker,
        sf_config={"Threads": "1", "Hash": "16"},
    )
    await pool.start()
    sem = asyncio.Semaphore(concurrent_per_worker)
    completed = 0

    async def play_one(spec: GameSpec):
        nonlocal completed
        token_ids, t0, t1 = spec
        async with sem:
            try:
                result = await play_game(
                    token_ids=token_ids[:],  # copy; each game mutates the list
                    engine_pool=pool,
                    temperature_player_0=t0,
                    temperature_player_1=t1,
                    depth=depth,
                    multipv_cap=multipv_cap,
                )
            except Exception as exc:
                result = exc
            result_queue.put(result)  # mp.Queue.put is safe from async ctx
            completed += 1
            return result

    await asyncio.gather(*[play_one(s) for s in game_specs], return_exceptions=True)
    await pool.shutdown()
    return completed


# ---------------------------------------------------------------------------
# Live throughput display
# ---------------------------------------------------------------------------


def _collect_with_throughput(queue: mp.Queue, total: int) -> list[Any]:
    """
    Drain *total* items from *queue*, printing a live progress/throughput line.
    Returns all items in arrival order (Exception instances included).
    """
    results: list = []
    start = time.perf_counter()
    prev_count = 0
    prev_time = start
    last_print = -1.0

    while len(results) < total:
        try:
            item = queue.get(timeout=0.15)
            results.append(item)
        except Exception:
            pass  # timeout; refresh the display

        now = time.perf_counter()
        if now - last_print >= 1.0:
            count = len(results)
            elapsed = now - start
            window = now - prev_time
            instant = (count - prev_count) / window if window > 0 else 0.0
            avg = count / elapsed if elapsed > 0 else 0.0
            eta = (total - count) / avg if avg > 0 else float("inf")

            bar_w = 24
            filled = int(bar_w * count / total) if total else 0
            bar = "█" * filled + "░" * (bar_w - filled)

            print(
                f"\r  [{bar}] {count:4d}/{total}"
                f"  {instant:5.1f} g/s"
                f"  avg {avg:5.1f}"
                f"  ETA {eta:4.0f}s"
                f"  {int(elapsed):4d}s elapsed   ",
                end="",
                flush=True,
            )
            prev_count = count
            prev_time = now
            last_print = now

    elapsed = time.perf_counter() - start
    avg = total / elapsed if elapsed > 0 else 0.0
    errors = sum(1 for r in results if isinstance(r, Exception))
    print(
        f"\r  ✓ {total} games in {elapsed:.1f}s"
        f"  |  avg {avg:.1f} g/s"
        f"  |  {errors} errors" + " " * 20
    )
    return results


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def compute_elo(results: list, initial: float = 1500) -> dict[float, float]:
    """
    Replacement using Bradley-Terry (choix).
    Returns comparable scalar ratings per temperature.
    """

    temps_to_idx = {}
    idx_to_temps = {}
    game_data = []

    def get_idx(t):
        if t not in temps_to_idx:
            i = len(temps_to_idx)
            temps_to_idx[t] = i
            idx_to_temps[i] = t
        return temps_to_idx[t]

    # ------------------------------------------------------------
    # Build pairwise dataset
    # ------------------------------------------------------------
    for r in results:
        if isinstance(r, Exception):
            continue

        t0 = r["temperature_player_0"]
        t1 = r["temperature_player_1"]
        res = r["result"]

        a = get_idx(t0)
        b = get_idx(t1)

        if res == "1-0":
            game_data.append((a, b))
        elif res == "0-1":
            game_data.append((b, a))
        elif res == "1/2-1/2":
            # Option 1: ignore draws (cleanest statistically)
            continue

    n = len(temps_to_idx)

    # regularization helps connectivity stability (important in chess datasets)
    params = choix.ilsr_pairwise(n, game_data, alpha=0.01)

    # ------------------------------------------------------------
    # Normalize to Elo-like scale (optional but useful)
    # ------------------------------------------------------------
    params = np.array(params)

    # center (identifiability: only differences matter)
    params -= params.mean()

    # convert log-strength → Elo-like scale
    elo_like = 1500 + 400 * params / np.log(10)

    return {idx_to_temps[i]: float(elo_like[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Experiment log  (append-only JSONL, one line per run)
# ---------------------------------------------------------------------------


def _log_experiment(
    log_path: Path,
    config: dict,
    elos: dict,
    wall_time: float,
    num_games: int,
    errors: int,
):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "wall_time_s": round(wall_time, 2),
        "games_per_sec": round(num_games / wall_time, 2) if wall_time > 0 else 0,
        "num_games": num_games,
        "errors": errors,
        "config": config,
        "elos": {str(k): round(v, 1) for k, v in elos.items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Logged → {log_path}")


# ---------------------------------------------------------------------------
# Main tournament runner
# ---------------------------------------------------------------------------


def run_tournament(
    num_matches: int,
    num_workers: int,
    concurrent_per_worker: int,
    temps: list[float],
    openings_jsonl: str,
    stockfish_path: str,
    depth: int,
    multipv_cap: int,
    experiment_log: str = "experiments.jsonl",
) -> dict[str, Any]:
    # Load openings
    openings: list[list[int]] = []
    with open(openings_jsonl) as f:
        for line in f:
            if line.strip():
                openings.append(json.loads(line.strip()))

    # Build all (opening, t0, t1) specs
    game_specs: list[GameSpec] = [
        (random.choice(openings), t0, t1)
        for t0 in temps
        for t1 in temps
        if t0 != t1
        for _ in range(num_matches // 2)
    ]
    random.shuffle(game_specs)  # even load distribution across workers
    total_games = len(game_specs)

    # Partition specs across workers
    chunk_size = max(1, (total_games + num_workers - 1) // num_workers)
    chunks = [game_specs[i : i + chunk_size] for i in range(0, total_games, chunk_size)]
    actual_workers = len(chunks)  # may be < num_workers if very few games

    total_engines = actual_workers * concurrent_per_worker
    print(
        f"\n  Tournament: {total_games} games"
        f"  |  {actual_workers} workers × {concurrent_per_worker} engines"
        f"  =  {total_engines} Stockfish instances"
        f"  |  depth={depth}  multipv_cap={multipv_cap}\n"
    )

    # result_queue: mp.Queue = mp.Queue()
    manager = mp.Manager()
    result_queue = manager.Queue()
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = [
            executor.submit(
                _worker,
                chunk,
                stockfish_path,
                depth,
                multipv_cap,
                concurrent_per_worker,
                result_queue,
            )
            for chunk in chunks
        ]

        time.sleep(1)  # give workers time to start (pragmatic fix)

        # Stream results from the queue while workers run
        all_results = _collect_with_throughput(result_queue, total_games)

        # Catch any worker-process-level crashes
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                logger.error("Worker process crashed: %s", exc)

    wall_time = time.perf_counter() - t_start
    errors = sum(1 for r in all_results if isinstance(r, Exception))
    elos = compute_elo(all_results)

    config_snapshot = {
        "num_matches": num_matches,
        "num_workers": num_workers,
        "concurrent_per_worker": concurrent_per_worker,
        "depth": depth,
        "multipv_cap": multipv_cap,
        "temps": [round(t, 2) for t in temps],
    }
    _log_experiment(
        Path(experiment_log),
        config=config_snapshot,
        elos=elos,
        wall_time=wall_time,
        num_games=total_games - errors,
        errors=errors,
    )

    return elos


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from hydra import compose, initialize_config_dir

    from plynder.core.config.hydra import from_hydra_config

    mp.set_start_method("spawn", force=True)

    cpu_count = mp.cpu_count()

    parser = argparse.ArgumentParser(
        description="Stockfish temperature tournament",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_name", default="test_config")
    parser.add_argument(
        "--num_matches",
        type=int,
        default=1024,
        help="Games per temperature pair (each pair plays both sides)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, cpu_count // 2),
        help="Worker processes. Rule of thumb: cpu_count // concurrent_per_worker",
    )
    parser.add_argument(
        "--concurrent_per_worker",
        type=int,
        default=8,
        help="Async-concurrent games per worker (= Stockfish engines per worker)",
    )
    parser.add_argument("--start_temp", type=float, default=500)
    parser.add_argument("--end_temp", type=float, default=25)
    parser.add_argument("--num_temps", type=int, default=32)
    parser.add_argument(
        "--p", type=float, default=-0.25, help="Temperature spacing: p=0 → logspace, p=1 → linear"
    )
    parser.add_argument(
        "--depth", type=int, default=5, help="Stockfish search depth. Try 2–3 for fast sweeps."
    )
    parser.add_argument(
        "--multipv_cap",
        type=int,
        default=DEFAULT_MULTIPV_CAP,
        help="Max moves analysed per turn. 10 is fine for high temps.",
    )
    parser.add_argument(
        "--experiment_log",
        default="experiments.jsonl",
        help="Append-only JSONL file, one line per tournament run",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo_root / "configs")):
        hydra_cfg = compose(config_name=args.config_name)
    config = from_hydra_config(hydra_cfg)

    temps = np.exp(
        np.exp(
            np.exp(
                np.linspace(
                    np.log(np.log(np.log(args.start_temp))),
                    np.log(np.log(np.log(args.end_temp))),
                    args.num_temps,
                )
            )
        )
    ).tolist()
    print("Temperatures:", [f"{t:.1f}" for t in temps])

    elos = run_tournament(
        num_matches=args.num_matches,
        num_workers=args.num_workers,
        concurrent_per_worker=args.concurrent_per_worker,
        temps=temps,
        openings_jsonl=config.paths.openings_jsonl,
        stockfish_path=config.paths.stockfish_path,
        depth=args.depth,
        multipv_cap=args.multipv_cap,
        experiment_log=args.experiment_log,
    )

    print("\nElo ratings:")
    last_elo = None
    print(f"  elos={sorted(elos.values())}")
    for temp, rating in sorted(elos.items(), key=lambda x: x[1], reverse=True):
        print(
            f"  temp={temp:8.1f}  elo={rating:.1f} delta={rating - last_elo:.1f}"
            if last_elo is not None
            else f"  temp={temp:8.1f}  elo={rating:.1f}"
        )
        last_elo = rating
