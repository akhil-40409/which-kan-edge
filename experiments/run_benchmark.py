#!/usr/bin/env python3
"""One-command benchmark across models and tasks.

Examples:
  python experiments/run_benchmark.py --quick
  python experiments/run_benchmark.py --models mlp,spline --tasks I.15.3t,j0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

from eigenflow.datasets import make_dataset
from eigenflow.layers import MLP, QKAN, QNN, QuIRK, SplineKAN
from eigenflow.training import train_model

QUICK_TASKS = ["I.12.1", "j0"]
DEFAULT_TASKS = ["I.12.1", "I.15.3t", "I.6.20a", "j0", "erf", "sinc"]
DEFAULT_MODELS = ["mlp", "spline", "qnn", "qkan", "quirk"]


def _build_model(name: str, n_features: int, quick: bool):
    h = 4 if quick else 8
    if name == "mlp":
        return MLP([n_features, h, h, 1])
    if name == "spline":
        return SplineKAN([n_features, h, 1], grid_size=3 if quick else 5)
    if name == "qnn":
        nq = min(n_features, 2 if quick else 3)
        return QNN(n_features=n_features, n_qubits=nq, n_layers=1 if quick else 2)
    if name == "qkan":
        return QKAN([n_features, 2 if quick else 4, 1], n_reps=1 if quick else 2)
    if name == "quirk":
        return QuIRK([n_features, 2 if quick else 4, 1], n_reps=1 if quick else 2)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description="Eigenflow benchmark runner")
    p.add_argument("--quick", action="store_true", help="Smoke run (few steps)")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--tasks", type=str, default="")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out", type=str, default="results/benchmark.csv")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = QUICK_TASKS if args.quick else DEFAULT_TASKS

    steps = 100 if args.quick else 1500
    n_samples = 400 if args.quick else 2000
    rows = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for model_name in models:
            for seed in range(args.seeds):
                key = jax.random.PRNGKey(seed + 17 * hash(task + model_name) % 10_000)
                k_data, k_train = jax.random.split(key)
                print(f"=== {model_name} | {task} | seed={seed} ===")
                data = make_dataset(task, k_data, n_samples=n_samples)
                model = _build_model(model_name, data["n_features"], args.quick)
                result = train_model(
                    model, data, k_train, steps=steps, batch_size=64, lr=1e-3
                )
                row = {
                    "model": model_name,
                    "task": task,
                    "seed": seed,
                    "n_params": result["n_params"],
                    "val_rmse": result["val_rmse"],
                    "test_rmse": result["test_rmse"],
                    "train_time_s": result["train_time_s"],
                    "infer_time_s": result["infer_time_s"],
                    "steps": steps,
                }
                rows.append(row)
                print(
                    f"  params={row['n_params']} test_rmse={row['test_rmse']:.4e} "
                    f"time={row['train_time_s']:.2f}s"
                )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path} and {json_path}")


if __name__ == "__main__":
    main()
