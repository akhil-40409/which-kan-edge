# Benchmark results

```bash
# local smoke
python experiments/run_benchmark.py --quick

# fair arch sweep + noise (envelope plots) — see docs/paper_claim.md
python experiments/run_benchmark.py --suite fair --list-jobs

# Sol overnight on Grace Hopper (after jobs/sol_setup_gh200.sh on -p arm):
GPU=gh200 bash jobs/submit_overnight.sh
python experiments/merge_results.py --glob 'results/shards/*.csv' --out results/fair_benchmark.csv
```

Artifacts land in `results/` and `results/shards/` (`*_gh200_*.csv` from Grace Hopper). CSV rows include `n_params`, `flops`, `test_rmse`, `train_time_s`, `noise`.
