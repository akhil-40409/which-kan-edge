# Running eigenflow on ASU Sol (first-timer guide)

This walkthrough gets you from zero Sol experience to an **overnight paper-scale**
CPU + GPU benchmark, then a merged results table.

Official ASU docs (keep these handy):

- [New user guide](https://docs.rc.asu.edu/new-user-guide/)
- [Connecting / VPN](https://docs.rc.asu.edu/connecting/)
- [Hardware / Sol specs](https://docs.rc.asu.edu/supercomputer-hardware/) (incl. **GraceHopper**)
- [Requesting resources](https://docs.rc.asu.edu/requesting-resources/) (Grace Hopper / GH200 → `-p arm`)
- [Partitions / QoS](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908867081) (`arm` = aarch64 Grace + GH200)
- Web portal: [https://sol.asu.edu](https://sol.asu.edu)

## 0. Account + VPN

1. Request Sol access if you have not: [Getting access](https://docs.rc.asu.edu/getting-access).
2. Connect to **ASU Cisco AnyConnect SSL VPN** before SSH / portal.
3. Prefer the **web portal** (`sol.asu.edu`) the first time — file browser + terminal.

Never run heavy jobs on the **login node**. Use `interactive` or `sbatch`.

## 1. Get the code on Sol

On Sol Shell Access (portal) or SSH (`ssh ASURITE@sol.asu.edu`):

```bash
cd ~/   # or your scratch project dir
git clone https://github.com/akhil-40409/eigenflow.git
cd eigenflow
```

## 2. One-time environment (interactive node)

```bash
interactive -c 4 -t 60 -p general -q public
bash jobs/sol_setup.sh
```

Smoke test (still in the interactive job):

```bash
source activate eigenflow
python experiments/run_benchmark.py --quick --models mlp,spline --tasks I.12.1
```

### GPU JAX (needed for the A100 array)

On a GPU interactive session:

```bash
interactive -c 4 -G a100:1 -t 60 -p general -q public
module load mamba/latest
module load cuda/12.2   # version may differ; try `module avail cuda`
source activate eigenflow
python -m pip install --upgrade "jax[cuda12]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -c "import jax; print(jax.devices())"   # expect GpuDevice
```

### Grace Hopper / GH200 (recommended for this paper)

Sol GraceHopper nodes ([specs](https://docs.rc.asu.edu/supercomputer-hardware/)): **72× NVIDIA Grace (aarch64)** + **1× GH200 480GB**. They live in the **`arm`** partition — not `general`.

**Critical:** your normal x86 `eigenflow` mamba env will **not** run on GH200. Build a separate aarch64 env **on an arm node**:

```bash
interactive -c 8 -G 1 -t 90 -p arm -q public --mem=64G
bash jobs/sol_setup_gh200.sh
# creates env eigenflow-gh200 + installs CUDA JAX + quick smoke
```

Interactive smoke later:

```bash
interactive -c 8 -G 1 -t 60 -p arm -q public --mem=64G
module load mamba/latest && source activate eigenflow-gh200
python -c "import platform, jax; print(platform.machine(), jax.devices())"
python experiments/run_benchmark.py --quick --models mlp,spline,fourier,qkan \
  --platform gpu --out /tmp/gh200_quick.csv
```

## 3. What “paper-scale” means here

Preprint claim and protocol: [`docs/paper_claim.md`](paper_claim.md).

### Fair suite (architecture sweep + noise) — run this for envelope plots

```bash
python experiments/run_benchmark.py --suite fair --list-jobs
```

| Axis | Values |
|------|--------|
| Tasks | 6 core (`DEFAULT_TASKS`) |
| Models | spline, fourier, qkan, mlp |
| Arch | Yu-style depth/width/basis grid (per model) |
| Noise | 0.0, 0.1 |
| Seeds | 3 |
| Steps / samples | 2000 / 3000 |

### Paper suite (full-task coarse table + noise)

```bash
python experiments/run_benchmark.py --suite paper --list-jobs
```

| Axis | Values |
|------|--------|
| Tasks | all Feynman + specials (**20**) |
| Models | spline, fourier, qkan, mlp (**4**) |
| Budgets | small, medium (**2**) |
| Noise | 0.0, 0.1 (**2**) |
| Seeds | **3** |
| Steps / samples | **3000** / **4000** |
| **Total jobs** | **20 × 4 × 2 × 2 × 3 = 960** |

GPU array uses the same JAX models (no QNN in the default paper set).

Each Slurm array task runs a **shard** of that list so many nodes work in parallel overnight.

Default overnight submit uses **`--suite fair`** (see `jobs/submit_overnight.sh`). Override with `SUITE=paper` for the full-task table.

## 4. Overnight submit (CPU + GPU / Grace Hopper)

From the **login node** (after setup), repo root:

```bash
chmod +x jobs/submit_overnight.sh jobs/sol_setup.sh jobs/sol_setup_gh200.sh

# Grace Hopper only (after sol_setup_gh200.sh on an arm node):
GPU=gh200 bash jobs/submit_overnight.sh

# Alternatives:
bash jobs/submit_overnight.sh                 # CPU + A100
GPU=both bash jobs/submit_overnight.sh        # CPU + A100 + GH200
SUITE=paper GPU=gh200 bash jobs/submit_overnight.sh
```

| `GPU=` | What gets submitted |
|--------|---------------------|
| `gh200` | CPU array + `jobs/sol_gh200_array.sbatch` (`-p arm -G 1`) |
| `a100` | CPU array + A100 array (default) |
| `both` | CPU + A100 + GH200 |
| `none` | CPU only |

Shard counts: `CPU_SHARDS=64`, `GPU_SHARDS=32`, `GH200_SHARDS=32` (override as env vars). GH200 env name: `GH200_ENV=eigenflow-gh200`.

Monitor:

```bash
squeue -u $USER
tail -f logs/gh200_<jobid>_0.out
```

GH200 shards write `results/shards/${SUITE}_gh200_*.csv`.

## 5. Merge + CPU vs GPU table

When both arrays finish:

```bash
source activate eigenflow
python experiments/merge_results.py \
  --glob 'results/shards/*.csv' \
  --out results/paper_benchmark.csv
```

You get:

- `results/paper_benchmark.csv` — all rows (accuracy + `train_time_s` + `platform`)
- `results/paper_benchmark_cpu_gpu.csv` — matched keys with **speedup**

## 6. Time estimate (why overnight is right)

Rough per-run wall time on Sol (order-of-magnitude):

| Model | CPU (EPYC) | A100 | GH200 |
|-------|------------|------|-------|
| mlp / spline / fourier | seconds–~1 min | often faster | often fastest |
| qkan | ~0.5–3 min | faster | faster |

**Serial** full fair/paper runs would be many days. **Sharded** across 64 CPU + 32 GH200 tasks overnight is the comfortable window.

If you need it faster: raise shards (`GH200_SHARDS=64`) or restrict models/tasks.

Or shorten steps:

```bash
python experiments/run_benchmark.py --suite fair --steps 1500 --n-samples 2000 ...
```

## 7. Etiquette / pitfalls

- Do **not** `pip install` on the login node for long builds — use `interactive`.
- Use system **mamba** (`module load mamba/latest`), not a home-brew conda.
- **A100:** `-G a100:1` on `general` / `htc`. **GH200:** `-p arm -G 1` (aarch64-only software).
- Never reuse the x86 `eigenflow` env on `arm` — use `eigenflow-gh200`.
- Request only the GPUs you use.
- Pull results off Sol when done (`scp` / portal download).
- Cancel with `scancel <jobid>` if something is wrong.

## 8. Minimal mental model

```text
You (laptop)
  → VPN
  → sol.asu.edu login / portal
  → sbatch array  (scheduler)
  → EPYC CPU nodes  +  (A100 | GraceHopper GH200 on arm)
  → results/shards/*.csv
  → merge_results.py
  → paper table
```

Scripts live in [`jobs/`](../jobs/). Runner: [`experiments/run_benchmark.py`](../experiments/run_benchmark.py).
