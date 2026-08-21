#!/usr/bin/env python3
"""Eigenflow benchmark runner (quick / default / paper / fair).

Suites
------
quick   — smoke test
default — small local run
paper   — full AI Feynman + specials, coarse budgets, noise {0, 0.1}
fair    — Yu-style architecture sweep (params/FLOPs envelopes) + noise

Examples:
  python experiments/run_benchmark.py --quick
  python experiments/run_benchmark.py --suite fair --list-jobs
  python experiments/run_benchmark.py --suite fair --shard 3/64 --platform cpu \\
      --out results/shards/fair_cpu_3.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

from src.datasets import FEYNMAN_EQUATIONS, SPECIAL_FUNCTIONS, make_dataset
from src.layers import FourierKAN, MLP, QKAN, QNN, QuIRK, SplineKAN
from src.training import train_model
from src.utils.flops import estimate_flops

QUICK_TASKS = ["I.12.1", "j0"]
DEFAULT_TASKS = ["I.12.1", "I.15.3t", "I.6.20a", "j0", "erf", "sinc"]
PAPER_TASKS = sorted(FEYNMAN_EQUATIONS.keys()) + sorted(SPECIAL_FUNCTIONS.keys())
# Envelope sweep tasks (Yu-style dense arch grid); full table uses PAPER_TASKS.
FAIR_SWEEP_TASKS = list(DEFAULT_TASKS)

# Main paper comparison + classical sideline.
PAPER_MODELS = ["spline", "fourier", "qkan", "mlp"]
DEFAULT_MODELS = ["mlp", "spline", "fourier", "qnn", "qkan", "quirk"]
GPU_MODELS = ["mlp", "spline", "fourier", "qkan", "quirk"]

# Named coarse budgets (quick / default / paper).
BUDGETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(depth=1, h=4, grid=3, n_modes=3, n_reps=1, n_layers=1, n_qubits_cap=2, order=2),
    "small": dict(depth=1, h=8, grid=5, n_modes=5, n_reps=2, n_layers=2, n_qubits_cap=3, order=3),
    "medium": dict(depth=2, h=16, grid=8, n_modes=8, n_reps=3, n_layers=3, n_qubits_cap=4, order=3),
}

# Fair-suite architecture grids (Yu et al. spirit; see docs/paper_claim.md).
FAIR_DEPTHS = (1, 2)
FAIR_WIDTHS_SPLINE = (4, 8, 16)
FAIR_WIDTHS_FOURIER = (4, 8, 16)
FAIR_WIDTHS_QKAN = (2, 4, 8)
FAIR_WIDTHS_MLP = (16, 32, 64, 128)
FAIR_GRIDS = (3, 5, 8, 16)
FAIR_MODES = (3, 5, 8, 16)
FAIR_ORDERS = (2, 3)
FAIR_REPS = (1, 2, 3)

_ARCH_RE = re.compile(
    r"^d(?P<depth>\d+)_h(?P<h>\d+)"
    r"(?:_G(?P<G>\d+))?(?:_K(?P<K>\d+))?(?:_M(?P<M>\d+))?(?:_R(?P<R>\d+))?$"
)


@dataclass(frozen=True)
class Job:
    task: str
    model: str
    budget: str
    noise: float
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}|{self.task}|{self.budget}|n{self.noise}|s{self.seed}"


def _configure_jax(platform_name: str) -> str:
    plat = platform_name.lower()
    if plat == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    elif plat == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        devs = jax.devices()
        return f"{devs[0].platform}:{devs[0]}"
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def _layer_sizes(n_features: int, depth: int, h: int) -> List[int]:
    """depth=1 → [n, h, 1]; depth=2 → [n, h, h, 1]."""
    if depth < 1:
        raise ValueError(depth)
    hidden = [h] * depth
    return [n_features, *hidden, 1]


def _parse_arch_budget(budget: str) -> Optional[Dict[str, int]]:
    m = _ARCH_RE.match(budget)
    if not m:
        return None
    d = {k: int(v) for k, v in m.groupdict().items() if v is not None}
    return d


def _resolve_cfg(budget: str) -> Dict[str, Any]:
    if budget in BUDGETS:
        return dict(BUDGETS[budget])
    arch = _parse_arch_budget(budget)
    if arch is None:
        raise ValueError(f"Unknown budget/arch key {budget!r}")
    cfg = dict(
        depth=arch["depth"],
        h=arch["h"],
        grid=arch.get("G", 5),
        n_modes=arch.get("M", 5),
        n_reps=arch.get("R", 2),
        order=arch.get("K", 3),
        n_layers=2,
        n_qubits_cap=3,
    )
    return cfg


def _build_model(name: str, n_features: int, budget: str, qnn_device: str):
    cfg = _resolve_cfg(budget)
    depth = int(cfg["depth"])
    h = int(cfg["h"])
    sizes = _layer_sizes(n_features, depth, h)

    if name == "mlp":
        return MLP(sizes)
    if name == "spline":
        return SplineKAN(
            sizes,
            grid_size=int(cfg["grid"]),
            spline_order=int(cfg["order"]),
        )
    if name == "fourier":
        return FourierKAN(sizes, n_modes=int(cfg["n_modes"]))
    if name == "qnn":
        nq = min(n_features, int(cfg["n_qubits_cap"]))
        return QNN(
            n_features=n_features,
            n_qubits=nq,
            n_layers=int(cfg["n_layers"]),
            device=qnn_device,
        )
    if name == "qkan":
        # Match fair widths directly (no h//2); coarse budgets still use h.
        qh = h if _parse_arch_budget(budget) else max(2, h // 2)
        qs = _layer_sizes(n_features, depth, qh)
        return QKAN(qs, n_reps=int(cfg["n_reps"]))
    if name == "quirk":
        qh = h if _parse_arch_budget(budget) else max(2, h // 2)
        qs = _layer_sizes(n_features, depth, qh)
        return QuIRK(qs, n_reps=int(cfg["n_reps"]))
    raise ValueError(name)


def _arch_key_spline(depth: int, h: int, g: int, k: int) -> str:
    return f"d{depth}_h{h}_G{g}_K{k}"


def _arch_key_fourier(depth: int, h: int, m: int) -> str:
    return f"d{depth}_h{h}_M{m}"


def _arch_key_qkan(depth: int, h: int, r: int) -> str:
    return f"d{depth}_h{h}_R{r}"


def _arch_key_mlp(depth: int, h: int) -> str:
    return f"d{depth}_h{h}"


def fair_budgets_for_model(model: str) -> List[str]:
    """Architecture sweep keys for one model (Yu-style grid)."""
    keys: List[str] = []
    if model == "spline":
        for d in FAIR_DEPTHS:
            for h in FAIR_WIDTHS_SPLINE:
                for g in FAIR_GRIDS:
                    for k in FAIR_ORDERS:
                        keys.append(_arch_key_spline(d, h, g, k))
    elif model == "fourier":
        for d in FAIR_DEPTHS:
            for h in FAIR_WIDTHS_FOURIER:
                for m in FAIR_MODES:
                    keys.append(_arch_key_fourier(d, h, m))
    elif model == "qkan":
        for d in FAIR_DEPTHS:
            for h in FAIR_WIDTHS_QKAN:
                for r in FAIR_REPS:
                    keys.append(_arch_key_qkan(d, h, r))
    elif model == "mlp":
        for d in FAIR_DEPTHS:
            for h in FAIR_WIDTHS_MLP:
                keys.append(_arch_key_mlp(d, h))
    else:
        raise ValueError(
            f"fair suite supports spline,fourier,qkan,mlp — got {model!r}"
        )
    return keys


def _suite_tasks(suite: str) -> List[str]:
    if suite == "quick":
        return list(QUICK_TASKS)
    if suite == "default":
        return list(DEFAULT_TASKS)
    if suite == "paper":
        return list(PAPER_TASKS)
    if suite == "fair":
        return list(FAIR_SWEEP_TASKS)
    raise ValueError(suite)


def enumerate_jobs(
    *,
    suite: str,
    models: Sequence[str],
    budgets: Sequence[str],
    noises: Sequence[float],
    seeds: int,
    tasks: Optional[Sequence[str]] = None,
) -> List[Job]:
    task_list = list(tasks) if tasks else _suite_tasks(suite)
    jobs: List[Job] = []

    if suite == "fair" and not budgets:
        # Per-model architecture sweep.
        for task in task_list:
            for model in models:
                for budget in fair_budgets_for_model(model):
                    for noise in noises:
                        for seed in range(seeds):
                            jobs.append(
                                Job(
                                    task=task,
                                    model=model,
                                    budget=budget,
                                    noise=float(noise),
                                    seed=int(seed),
                                )
                            )
        return jobs

    for task in task_list:
        for model in models:
            for budget in budgets:
                for noise in noises:
                    for seed in range(seeds):
                        jobs.append(
                            Job(
                                task=task,
                                model=model,
                                budget=budget,
                                noise=float(noise),
                                seed=int(seed),
                            )
                        )
    return jobs


def shard_jobs(jobs: Sequence[Job], shard: str) -> List[Job]:
    if not shard:
        return list(jobs)
    i_str, n_str = shard.split("/")
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"shard index must satisfy 0 <= i < n, got {shard}")
    return [j for k, j in enumerate(jobs) if k % n == i]


def run_jobs(
    jobs: Sequence[Job],
    *,
    steps: int,
    n_samples: int,
    batch_size: int,
    lr: float,
    platform_name: str,
    qnn_device: str,
    out_path: Path,
) -> List[dict]:
    device_str = _configure_jax(platform_name)
    host = socket.gethostname()
    rows: List[dict] = []

    for job in jobs:
        print(f"=== {job.key} | platform={platform_name} | {device_str} ===", flush=True)
        key = jax.random.PRNGKey(
            job.seed + 17 * (abs(hash(job.task + job.model + job.budget)) % 10_000)
        )
        k_data, k_train = jax.random.split(key)
        data = make_dataset(
            job.task,
            k_data,
            n_samples=n_samples,
            noise_level=job.noise,
        )
        model = _build_model(job.model, data["n_features"], job.budget, qnn_device)
        flops = estimate_flops(model)
        cfg = _resolve_cfg(job.budget)
        result = train_model(
            model,
            data,
            k_train,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
        )
        row = {
            "model": job.model,
            "task": job.task,
            "budget": job.budget,
            "depth": cfg["depth"],
            "width": cfg["h"],
            "grid": cfg.get("grid", ""),
            "n_modes": cfg.get("n_modes", ""),
            "spline_order": cfg.get("order", ""),
            "n_reps": cfg.get("n_reps", ""),
            "noise": job.noise,
            "seed": job.seed,
            "n_params": result["n_params"],
            "flops": result.get("flops", flops),
            "val_rmse": result["val_rmse"],
            "test_rmse": result["test_rmse"],
            "train_time_s": result["train_time_s"],
            "infer_time_s": result["infer_time_s"],
            "steps": steps,
            "n_samples": n_samples,
            "platform": platform_name,
            "jax_device": device_str,
            "host": host,
            "python": platform.python_version(),
            "qnn_device": qnn_device if job.model == "qnn" else "",
        }
        rows.append(row)
        print(
            f"  params={row['n_params']} flops={row['flops']} "
            f"test_rmse={row['test_rmse']:.4e} train_s={row['train_time_s']:.2f}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_path.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path} ({len(rows)} rows)", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="Eigenflow benchmark runner")
    p.add_argument("--quick", action="store_true", help="Alias for --suite quick")
    p.add_argument(
        "--suite",
        choices=["quick", "default", "paper", "fair"],
        default="default",
        help="Preset task/budget/noise scale",
    )
    p.add_argument("--models", type=str, default="")
    p.add_argument("--tasks", type=str, default="")
    p.add_argument(
        "--budgets",
        type=str,
        default="",
        help="Comma list of named budgets or arch keys; empty + fair → full arch sweep",
    )
    p.add_argument("--noises", type=str, default="")
    p.add_argument("--seeds", type=int, default=-1)
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--n-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--platform",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Tag + JAX preference (set JAX_PLATFORMS before launch for GPU)",
    )
    p.add_argument(
        "--qnn-device",
        type=str,
        default="default.qubit",
        help="PennyLane device for QNN only",
    )
    p.add_argument("--shard", type=str, default="", help="Array shard i/n (0-based)")
    p.add_argument("--list-jobs", action="store_true", help="Print job count and exit")
    p.add_argument("--out", type=str, default="results/benchmark.csv")
    args = p.parse_args()

    suite = "quick" if args.quick else args.suite

    if suite == "quick":
        default_budgets, default_noises, default_seeds = ["tiny"], [0.0], 1
        default_steps, default_samples = 100, 400
        default_models = PAPER_MODELS
    elif suite == "default":
        default_budgets, default_noises, default_seeds = ["small"], [0.0, 0.1], 1
        default_steps, default_samples = 1500, 2000
        default_models = PAPER_MODELS
    elif suite == "paper":
        # Full-task matched coarse budgets + noise (table companion to fair envelopes).
        default_budgets, default_noises, default_seeds = (
            ["small", "medium"],
            [0.0, 0.1],
            3,
        )
        default_steps, default_samples = 3000, 4000
        default_models = PAPER_MODELS
    else:  # fair
        # Empty budgets → per-model arch sweep; noise included.
        default_budgets, default_noises, default_seeds = [], [0.0, 0.1], 3
        default_steps, default_samples = 2000, 3000
        default_models = PAPER_MODELS

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(default_models)
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    if args.budgets.strip():
        budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]
    else:
        budgets = list(default_budgets)
    noises = (
        [float(x) for x in args.noises.split(",") if x.strip()]
        if args.noises
        else list(default_noises)
    )
    seeds = args.seeds if args.seeds > 0 else default_seeds
    steps = args.steps if args.steps > 0 else default_steps
    n_samples = args.n_samples if args.n_samples > 0 else default_samples

    platform_name = args.platform
    if platform_name == "auto":
        platform_name = (
            "gpu"
            if any(
                "gpu" in str(d).lower() or d.platform == "gpu" for d in jax.devices()
            )
            else "cpu"
        )

    jobs = enumerate_jobs(
        suite=suite,
        models=models,
        budgets=budgets,
        noises=noises,
        seeds=seeds,
        tasks=tasks,
    )
    jobs = shard_jobs(jobs, args.shard)

    if args.list_jobs:
        full = enumerate_jobs(
            suite=suite,
            models=models,
            budgets=budgets,
            noises=noises,
            seeds=seeds,
            tasks=tasks,
        )
        print(
            f"suite={suite} full_jobs={len(full)} shard={args.shard or 'all'} "
            f"selected={len(jobs)}"
        )
        print(
            f"models={models} budgets={budgets or '(fair arch sweep)'} "
            f"noises={noises} seeds={seeds} steps={steps} n_samples={n_samples}"
        )
        if suite == "fair" and not budgets:
            for m in models:
                print(f"  {m}: {len(fair_budgets_for_model(m))} arch configs")
        return

    if not jobs:
        print("No jobs selected.", file=sys.stderr)
        sys.exit(1)

    run_jobs(
        jobs,
        steps=steps,
        n_samples=n_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        platform_name=platform_name,
        qnn_device=args.qnn_device,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
