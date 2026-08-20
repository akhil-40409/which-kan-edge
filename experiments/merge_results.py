#!/usr/bin/env python3
"""Merge shard CSVs from Sol array jobs into one table (+ optional CPU/GPU compare)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", type=str, default="results/shards/*.csv")
    p.add_argument("--out", type=str, default="results/paper_benchmark.csv")
    args = p.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {args.glob}")

    rows = []
    fieldnames = None
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows.extend(list(reader))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(paths)} files → {out} ({len(rows)} rows)")

    by_key: dict = {}
    for r in rows:
        key = (r["model"], r["task"], r["budget"], r["noise"], r["seed"])
        by_key.setdefault(key, {})[r.get("platform", "")] = r

    cmp_path = out.with_name(out.stem + "_cpu_gpu.csv")
    cmp_rows = []
    for key, plats in by_key.items():
        if "cpu" in plats and "gpu" in plats:
            cpu, gpu = plats["cpu"], plats["gpu"]
            ct, gt = float(cpu["train_time_s"]), float(gpu["train_time_s"])
            cmp_rows.append(
                {
                    "model": key[0],
                    "task": key[1],
                    "budget": key[2],
                    "noise": key[3],
                    "seed": key[4],
                    "cpu_train_s": ct,
                    "gpu_train_s": gt,
                    "speedup": (ct / gt) if gt > 0 else "",
                    "cpu_test_rmse": cpu["test_rmse"],
                    "gpu_test_rmse": gpu["test_rmse"],
                }
            )
    if cmp_rows:
        with cmp_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"Wrote CPU/GPU compare → {cmp_path} ({len(cmp_rows)} matched runs)")
    else:
        print("No overlapping CPU+GPU keys yet (run both arrays, then merge).")


if __name__ == "__main__":
    main()
