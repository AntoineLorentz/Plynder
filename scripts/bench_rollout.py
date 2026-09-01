#!/usr/bin/env python3
"""
Benchmark vllm_server across different configurations.

Usage:
    python bench.py
    python bench.py --config-name test_config.yaml --warmup 30 --duration 90
"""

import argparse
import contextlib
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import zmq
from omegaconf import DictConfig, OmegaConf

# ─── Benchmark matrix ────────────────────────────────────────────────────────
# Keys are dot-separated Hydra override paths (relative to config root).
# Add/remove rows freely.

BENCH_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "rollout.vllm.num_engines_per_gpu": 3,
        "rollout.vllm.max_concurrent_groups": 72,
    },
    {
        "name": "56cg",
        "rollout.vllm.num_engines_per_gpu": 3,
        "rollout.vllm.max_concurrent_groups": 56,
    },
    {
        "name": "64cg",
        "rollout.vllm.num_engines_per_gpu": 3,
        "rollout.vllm.max_concurrent_groups": 64,
    },
    {
        "name": "80cg",
        "rollout.vllm.num_engines_per_gpu": 3,
        "rollout.vllm.max_concurrent_groups": 80,
    },
    {
        "name": "88cg",
        "rollout.vllm.num_engines_per_gpu": 3,
        "rollout.vllm.max_concurrent_groups": 88,
    },
]

# ─── Timing ──────────────────────────────────────────────────────────────────

WARMUP_SEC = 30  # seconds to discard after first stat
BENCH_SEC = 90  # seconds to collect for measurement

# ─── Stats keys to aggregate (mean over all received windows) ────────────────

STAT_KEYS = [
    "throughput_inference",
    "throughput_total",
    "throughput_sequences",
    "p50_group_latency",
    "p90_group_latency",
    "p99_group_latency",
    "kept_ratio",
    "white_wins",
    "black_wins",
    "draws",
    "timeout_per_sec",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

VLLM_SCRIPT = str(REPO_ROOT / "src" / "plynder" / "rollout" / "main.py")

# ─── Colors / ANSI ───────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
WHITE = "\033[97m"


def c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    n_windows: int
    aggregated: dict[str, float] = field(default_factory=dict)
    error: str | None = None


# ─── Config helpers ───────────────────────────────────────────────────────────


def _zmq_addr(host: str, port: int, bind: bool = False) -> str:
    """Build a tcp ZMQ address. bind=True uses '*' instead of the host."""
    return f"tcp://{'*' if bind else host}:{port}"


def load_config(config_name: str) -> DictConfig:
    config_dir = str(REPO_ROOT / "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return OmegaConf.load(f"{config_dir}/{config_name}")


# ─── ZMQ background services ─────────────────────────────────────────────────


class SinkThread(threading.Thread):
    """Drains the PULL socket that rollout workers push samples to."""

    def __init__(self, ctx: zmq.Context, bind_addr: str):
        super().__init__(daemon=True, name="sink")
        self._ctx = ctx
        self._bind_addr = bind_addr
        self._stop_event = threading.Event()
        self.bytes_received = 0
        self.msgs_received = 0

    def run(self):
        sock = self._ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.bind(self._bind_addr)
        try:
            while not self._stop_event.is_set():
                try:
                    parts = sock.recv_multipart()
                    self.bytes_received += sum(len(p) for p in parts)
                    self.msgs_received += 1
                except zmq.Again:
                    pass
        finally:
            sock.close()

    def stop(self):
        self._stop_event.set()


class ModelPubThread(threading.Thread):
    """Holds the trainer model PUB socket open (workers load from disk at startup)."""

    def __init__(self, ctx: zmq.Context, bind_addr: str):
        super().__init__(daemon=True, name="model-pub")
        self._ctx = ctx
        self._bind_addr = bind_addr
        self._stop_event = threading.Event()

    def run(self):
        sock = self._ctx.socket(zmq.PUB)
        sock.bind(self._bind_addr)
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        finally:
            sock.close()

    def stop(self):
        self._stop_event.set()


def collect_stats(
    ctx: zmq.Context,
    stats_bind_addr: str,
    warmup_sec: int,
    bench_sec: int,
    stop_event: threading.Event,
) -> tuple | None:
    """
    Bind to the stats PUB socket, discard warmup_sec worth of windows,
    then collect for bench_sec.  Returns (aggregated_dict, n_windows) or None.
    """
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(stats_bind_addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    windows: list[dict[str, float]] = []

    # Phase 1: wait for first stat (up to 5 minutes)
    t0 = time.monotonic()
    print(f"    {c('Waiting for first stat…', DIM)}", end="", flush=True)
    while True:
        if stop_event.is_set() or time.monotonic() - t0 > 300:
            sock.close()
            return None
        try:
            sock.recv_json()
            break
        except zmq.Again:
            pass
    print(f" {c('got it', GREEN)}")

    # Phase 2: discard warmup samples
    print(f"    {c(f'Warmup {warmup_sec}s…', DIM)}", end="", flush=True)
    warmup_end = time.monotonic() + warmup_sec
    while time.monotonic() < warmup_end:
        if stop_event.is_set():
            sock.close()
            return None
        with contextlib.suppress(zmq.Again):
            sock.recv_json()
    print(f" {c('done', GREEN)}")

    # Phase 3: measure
    print(f"    {c(f'Measuring {bench_sec}s…', DIM)}", end="", flush=True)
    bench_end = time.monotonic() + bench_sec
    while time.monotonic() < bench_end:
        if stop_event.is_set():
            break
        try:
            data = sock.recv_json()
            windows.append(data)
        except zmq.Again:
            pass
    print(f" {c(f'{len(windows)} windows', GREEN)}")
    sock.close()

    if not windows:
        return None

    aggregated = {}
    for key in STAT_KEYS:
        vals = [w[key] for w in windows if key in w]
        aggregated[key] = statistics.mean(vals) if vals else float("nan")

    return aggregated, len(windows)


# ─── Process launcher ─────────────────────────────────────────────────────────


def launch(config_name: str, bench_cfg: dict[str, Any], vllm_script: str) -> subprocess.Popen:
    cmd = [sys.executable, VLLM_SCRIPT, f"--config-name={config_name}"]
    for k, v in bench_cfg.items():
        if k == "name":
            continue
        cmd.append(f"{k}={v}")
    print(f"    {c('CMD:', DIM)} {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        preexec_fn=os.setsid,
    )


def kill(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    except Exception:
        pass


# ─── Pretty table ─────────────────────────────────────────────────────────────

COL_SPECS = [
    # (header, key, fmt, unit, scale)
    ("Inf. tput", "throughput_inference", ".1f", "K tok/s", 1 / 1000),
    ("Tput", "throughput_total", ".1f", "K tok/s", 1 / 1000),
    ("Seq/s", "throughput_sequences", ".2f", "seq/s", 1),
    ("p50 lat", "p50_group_latency", ".2f", "s", 1),
    ("p90 lat", "p90_group_latency", ".2f", "s", 1),
    ("p99 lat", "p99_group_latency", ".2f", "s", 1),
    ("Kept", "kept_ratio", ".1%", "", 1),
    ("W%", "white_wins", ".1%", "", 1),
    ("B%", "black_wins", ".1%", "", 1),
    ("D%", "draws", ".1%", "", 1),
    ("T/out/s", "timeout_per_sec", ".2f", "", 1),
]


def _fmt(val: float, fmt: str, scale: float) -> str:
    if val != val:  # nan
        return "N/A"
    return format(val * scale, fmt)


def print_table(results: list[BenchResult]):
    if not results:
        return

    name_w = max(len("Config"), max(len(r.name) for r in results))
    win_w = 6
    col_w = [max(len(h), 8) for h, *_ in COL_SPECS]

    rows = []
    for r in results:
        if r.error:
            row = [r.name, "ERR"] + ["N/A"] * len(COL_SPECS)
        else:
            row = [r.name, str(r.n_windows)]
            for _, key, fmt, _, scale in COL_SPECS:
                row.append(_fmt(r.aggregated.get(key, float("nan")), fmt, scale))
        rows.append(row)

    def col_vals(idx):
        return [
            float(rows[i][idx + 2].replace("N/A", "nan").replace("%", "").replace("K", ""))
            for i in range(len(rows))
        ]

    def best_indices(idx, higher_is_better: bool):
        vals = col_vals(idx)
        valid = [v for v in vals if v == v]
        if not valid:
            return set()
        target = max(valid) if higher_is_better else min(valid)
        return {i for i, v in enumerate(vals) if v == target}

    hib_map = {
        0: True,
        1: True,
        2: True,
        3: False,
        4: False,
        5: False,
        6: True,
        7: None,
        8: None,
        9: None,
        10: False,
    }
    highlights = {ci: best_indices(ci, h) for ci, h in hib_map.items() if h is not None}

    all_w = [name_w, win_w] + col_w
    headers = ["Config", "W"] + [h for h, *_ in COL_SPECS]
    units = ["", ""] + [u for _, _, _, u, _ in COL_SPECS]

    def _sep(left, mid, right):
        parts = ["─" * (w + 2) for w in all_w]
        return left + mid.join(parts) + right

    print()
    print(c("  ┌" + _sep("", "┬", "") + "┐", DIM))

    hdr_cells = [c(h.center(w), BOLD, CYAN) for h, w in zip(headers, all_w, strict=False)]
    print("  │" + "│".join(f" {cell} " for cell in hdr_cells) + "│")

    unit_cells = [c(u.center(w), DIM) for u, w in zip(units, all_w, strict=False)]
    print("  │" + "│".join(f" {cell} " for cell in unit_cells) + "│")
    print("  ├" + _sep("", "┼", "") + "┤")

    for ri, row in enumerate(rows):
        cells = []
        for ci, (cell, w) in enumerate(zip(row, all_w, strict=False)):
            if ci == 0:
                cells.append(c(cell.ljust(w), BOLD, WHITE))
            elif ci == 1:
                cells.append(c(cell.rjust(w), DIM))
            else:
                key_ci = ci - 2
                cells.append(
                    c(cell.rjust(w), GREEN, BOLD)
                    if ri in highlights.get(key_ci, set())
                    else cell.rjust(w)
                )
        print("  │" + "│".join(f" {cell} " for cell in cells) + "│")

    print("  └" + _sep("", "┴", "") + "┘")
    print()
    print(c("  Green = best value in column", DIM))


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark vllm_server via Hydra config")
    p.add_argument(
        "--config-name",
        default="test_config.yaml",
        help="Hydra config file name (default: %(default)s)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_SEC,
        help="Seconds to discard after first stat (default: %(default)s)",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=BENCH_SEC,
        help="Seconds to collect stats (default: %(default)s)",
    )
    p.add_argument("--configs", nargs="+", help="Names of bench configs to run (default: all)")
    p.add_argument(
        "--vllm-script", default=VLLM_SCRIPT, help="Path to main.py (default: %(default)s)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load shared Hydra config ──────────────────────────────────────────────
    cfg = load_config(args.config_name)
    net = cfg.networking
    host = net.train_ip

    sink_addr = _zmq_addr(host, net.ports.samples, bind=True)
    model_addr = _zmq_addr(host, net.ports.model, bind=True)
    stats_addr = _zmq_addr(host, net.ports.stats, bind=True)

    # ── Select bench configs ──────────────────────────────────────────────────
    bench_configs = BENCH_CONFIGS
    if args.configs:
        bench_configs = [bc for bc in BENCH_CONFIGS if bc["name"] in args.configs]
        if not bench_configs:
            print(f"{RED}No matching configs found.{RESET}")
            sys.exit(1)

    print()
    print(c("  ╔═══════════════════════════════════╗", CYAN))
    print(c("  ║    vllm_server benchmark suite    ║", CYAN, BOLD))
    print(c("  ╚═══════════════════════════════════╝", CYAN))
    print(f"\n  Config file: {args.config_name}")
    print(f"  Configs:     {len(bench_configs)}")
    print(f"  Warmup:      {args.warmup}s per config")
    print(f"  Duration:    {args.duration}s per config")
    estimated = len(bench_configs) * (args.warmup + args.duration + 40)
    print(f"  Estimated:   ~{estimated // 60}m{estimated % 60:02d}s total\n")

    ctx = zmq.Context()

    sink = SinkThread(ctx, sink_addr)
    model_pub = ModelPubThread(ctx, model_addr)
    sink.start()
    model_pub.start()

    results: list[BenchResult] = []

    try:
        for i, bench_cfg in enumerate(bench_configs):
            name = bench_cfg.get("name", f"config_{i}")
            print(c(f"\n  [{i + 1}/{len(bench_configs)}] {name}", BOLD, YELLOW))
            print(c("  " + "─" * 40, DIM))

            proc = None
            stop_event = threading.Event()

            try:
                proc = launch(args.config_name, bench_cfg, args.vllm_script)

                agg_result, n_windows = None, 0
                try:
                    result = collect_stats(ctx, stats_addr, args.warmup, args.duration, stop_event)
                    if result is not None:
                        agg_result, n_windows = result
                except Exception as e:
                    print(f"    {c('Stats collection error: ' + str(e), RED)}")

                if agg_result:
                    results.append(
                        BenchResult(name=name, n_windows=n_windows, aggregated=agg_result)
                    )
                    tput_k = agg_result.get("throughput_inference", 0) / 1000
                    print(
                        f"    {c('Result:', BOLD)} inf_tput={tput_k:.1f}K tok/s  "
                        f"p50={agg_result.get('p50_group_latency', 0):.2f}s"
                    )
                else:
                    results.append(BenchResult(name=name, n_windows=0, error="no stats received"))
                    print(f"    {c('No stats received; skipping.', RED)}")

            except KeyboardInterrupt:
                print(f"\n  {c('Interrupted; stopping current run.', YELLOW)}")
                stop_event.set()
                raise
            except Exception as e:
                results.append(BenchResult(name=name, n_windows=0, error=str(e)))
                print(f"    {c('Error: ' + str(e), RED)}")
            finally:
                stop_event.set()
                if proc is not None:
                    print(f"    {c('Stopping server…', DIM)}", end="", flush=True)
                    kill(proc)
                    print(f" {c('done', GREEN)}")
                time.sleep(3)  # let ports fully release

    except KeyboardInterrupt:
        print(f"\n  {c('Benchmark aborted by user.', YELLOW)}")
    finally:
        sink.stop()
        model_pub.stop()
        ctx.term()

    print_table(results)


if __name__ == "__main__":
    main()
