# Rebuttal Ablation Suite

**Paper:** Causal Discovery from Unseen Environments — NeurIPS 2026
**Purpose:** Addresses reviewer comment requesting sensitivity analysis to assumption violations.

---

## Scientific Objective

The paper's method relies on mean-shift interventions in a linear Gaussian SCM:

    X = W X + E,   X = (I - W)^{-1} E

where interventions stochastically shift the exogenous noise mean. The rebuttal experiments test sensitivity to four departures from the baseline design, one at a time.

---

## Repository Audit

Before reading this file, see the audit comment at the top of
`ablation/ablation_simulator.py`. Key facts:

| Question | Answer |
|----------|--------|
| `datasets_interventional` key | Tuple `(0,1,0,...,0)`: one unique intervention-pattern environment |
| `p_intervention` semantics | Per-node Bernoulli probability in `bernoulli` mode |
| `delta` application | `delta_per_node[i] = delta * sqrt(var(X_obs[:,i]))` |
| `build_env_labels` | Integer label per sample, ordered by `datasets_interventional.keys()` |
| `split_subsample` | obs_sub = full obs + all-zeros pattern rows; ivs_list per non-zero env |
| Edge weights | Signed, magnitudes ∈ [0.5, 2] |
| `n_samples` | Interventional samples only (excludes `n_observational`) |
| ERDagsComparison pool size | Uses `n_samples=1e6`; metadata incorrectly records `1e8` |

This ablation uses `n_samples=2000` directly — no subsampling.

---

## Design

### Baseline (shared across all ablations)

| Parameter | Value |
|-----------|-------|
| Graph | Erdős–Rényi, 10 nodes, 10 edges |
| Edge weights | Signed, magnitudes ∈ [0.5, 2] |
| Seeds | 0–99 |
| Interventional samples | 2000 |
| Observational samples | 100 |
| Intervention mode | Bernoulli, p_i = 0.5 ∀i |
| Shift strength | δ = 1.0 (in units of per-node obs std) |
| Off-target effects | None |

### Ablation A — Unequal Environment Sizes

**Mechanism:** Heterogeneous per-node Bernoulli probabilities.

    p_i(h) = 0.5 + h · z_perm[i]

where `z = linspace(-1, 1, d)` and `perm` is a seed-specific permutation fixed across h values.

**Why this analog?** The simulator uses Bernoulli mode where environment sizes emerge from unique pattern counts. There are no pre-specified explicit sizes, so direct analogue is heterogeneous marginal probabilities. Mean p_i = 0.5 at all h values (z is symmetric).

**Normalization:** Mean intervention probability fixed at 0.5 across all h. This prevents the ablation from changing the total expected intervention count.

| Severity (h) | p range |
|---|---|
| 0.00 | [0.50, 0.50] (baseline) |
| 0.15 | [0.35, 0.65] |
| 0.30 | [0.20, 0.80] |
| 0.45 | [0.05, 0.95] |

### Ablation B — Heterogeneous Intervention Strengths

**Mechanism:** Node-specific delta multipliers with fixed RMS.

    raw_i = exp(τ · z_perm[i])
    δ_i   = raw_i / sqrt(mean_j raw_j²)

Guarantees `mean(δ_i²) = 1` at all τ. At τ=0, δ_i = 1 ∀i.

**Normalization rationale:** Fixing the RMS shift strength prevents the ablation from merely increasing or decreasing total intervention signal energy. Differences in performance are attributable to the *spread* of strengths, not total energy.

| τ | δ min | δ max |
|---|---|---|
| 0.00 | 1.000 | 1.000 |
| 0.35 | ~0.67 | ~1.35 |
| 0.70 | ~0.41 | ~1.67 |
| 1.00 | ~0.26 | ~1.91 |

### Ablation C — Uniformly Weak Interventions

**Mechanism:** Reduce the common shift strength δ. Observational noise std = 1.0.

| δ | SNR interpretation |
|---|---|
| 1.00 | 100% of obs std (baseline) |
| 0.75 | 75% |
| 0.50 | 50% |
| 0.25 | 25% |
| 0.10 | 10% |

No normalization needed: δ directly sets the signal-to-noise ratio of the mean shift.

### Ablation D — Correlated Off-Target Effects

**Mechanism:** Random derangement π (π(i) ≠ i ∀i) applied once per seed. When node i is intervened, node π(i) also receives a shift of amplitude ρ · δ_i, sharing the same Bernoulli indicator.

    shift_i     += c(ρ) · δ_i · B_i
    shift_{π(i)} += c(ρ) · ρ · δ_i · B_i
    c(ρ) = 1 / sqrt(1 + ρ²)

In vectorized form:
    U = [V + ρ P V] / sqrt(1 + ρ²),    V_i = δ_i · B_i

**Normalization rationale:** `c(ρ)` prevents the ablation from increasing total shift energy per intervention event. At ρ=1, intended and off-target effects have equal amplitude but total shifted energy is the same as ρ=0.

**Target information for methods:**
- Environment-aware methods (IGSP, GIES, UT-IGSP, JCI-PC) receive the **actual** targets (intended + off-target) in the primary comparison. This is the fair comparison — methods are not silently handicapped.
- LiNGAM methods ignore labels and are unaffected.

| ρ | Interpretation |
|---|---|
| 0.00 | No off-target (baseline) |
| 0.10 | Weak leakage |
| 0.25 | Mild leakage |
| 0.50 | Moderate leakage |
| 1.00 | Equal intended/off-target amplitude |

---

## Commands

### Smoke test (runs in <5 min without a cluster)

```bash
cd synthetic_experiments

python experiments/run_rebuttal_ablations.py \
    --config configs/rebuttal_ablations.yaml \
    --quick \
    --results-dir results/rebuttal_smoke \
    --n-jobs 1
```

### Full experiment

```bash
python experiments/run_rebuttal_ablations.py \
    --config configs/rebuttal_ablations.yaml \
    --results-dir /path/to/scratch/rebuttal_ablations \
    --n-jobs 8 \
    --resume
```

### Resume after interruption

```bash
python experiments/run_rebuttal_ablations.py \
    --config configs/rebuttal_ablations.yaml \
    --results-dir /path/to/scratch/rebuttal_ablations \
    --n-jobs 8 \
    --resume
```

### Aggregation

```bash
python analysis/aggregate_rebuttal_ablations.py \
    --results-dir /path/to/scratch/rebuttal_ablations
```

### Plotting

```bash
python analysis/plot_rebuttal_ablations.py \
    --results-dir /path/to/scratch/rebuttal_ablations \
    --output-dir figures/rebuttal_ablations
```

### Tests

```bash
# From the synthetic_experiments directory:
pytest tests/test_rebuttal_ablations.py -v
```

---

## SLURM Example (HPC cluster)

```bash
#!/bin/bash
#SBATCH --job-name=rebuttal_ablations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rebuttal_%j.out

RESULTS=/path/to/scratch/rebuttal_ablations
conda activate cd-intervention-venv

cd /path/to/NEURIPS-2026-Causal-Discovery-from-Unknown-Interventions/synthetic_experiments

python experiments/run_rebuttal_ablations.py \
    --config configs/rebuttal_ablations.yaml \
    --results-dir "$RESULTS" \
    --n-jobs 16 \
    --resume
```

---

## Output Files

| File | Description |
|------|-------------|
| `results_dir/manifest.json` | Experiment metadata, git hash, package versions |
| `results_dir/ckpt_*.json` | Atomic checkpoints (one per condition×seed) |
| `results_dir/results_combined.json` | All rows combined |
| `results_dir/results.parquet` | Long-form table (one row per seed×condition×method) |
| `results_dir/summary.parquet` | Per-condition×method summary statistics |
| `results_dir/paired_diffs.parquet` | Paired SHD/F1 differences vs baseline |
| `results_dir/bootstrap_ci.parquet` | 95% paired bootstrap CIs |
| `figures/fig_main_shd.{pdf,png}` | Main 4-panel SHD figure |
| `figures/fig_full_shd.{pdf,png}` | Full-method SHD figure |
| `figures/fig_main_f1.{pdf,png}` | Main 4-panel F1 figure |
| `figures/fig_diagnostics.{pdf,png}` | Source diagnostics vs severity |

---

## Files Created

```
synthetic_experiments/
├── ablation/
│   └── ablation_simulator.py     Core simulation logic for all ablations
├── configs/
│   └── rebuttal_ablations.yaml   Experiment configuration
├── experiments/
│   └── run_rebuttal_ablations.py Experiment runner with CLI
├── analysis/
│   ├── aggregate_rebuttal_ablations.py  Result aggregation to Parquet
│   └── plot_rebuttal_ablations.py       Figure generation
├── tests/
│   └── test_rebuttal_ablations.py  Validation tests (pytest)
└── README_REBUTTAL_ABLATIONS.md    This file
```

---

## Load Results in Python

```python
import pandas as pd
import json

# Long-form table
df = pd.read_parquet('results/rebuttal_ablations/results.parquet')

# Summary statistics
summary = pd.read_parquet('results/rebuttal_ablations/summary.parquet')

# Paired differences vs baseline
diffs = pd.read_parquet('results/rebuttal_ablations/paired_diffs.parquet')

# Experiment metadata
with open('results/rebuttal_ablations/manifest.json') as f:
    meta = json.load(f)
    print(meta['git_hash'], meta['timestamp'])
```

---

## Acceptance Checklist

- [x] Smoke test runs without cluster
- [x] All four ablations implemented (A: unequal sizes, B: heterogeneous strengths, C: weak, D: off-target)
- [x] Each ablation changes exactly one feature relative to baseline
- [x] Zero-severity baseline is identical across ablations
- [x] Randomness is seeded and reproducible (same seed → same results)
- [x] Resume mode skips already-completed (condition, seed) pairs
- [x] Failures are recorded (failed, exception_type, exception_message columns)
- [x] Four-panel figure generated from saved results
- [x] Metadata includes enough to reconstruct every condition
- [x] README explains each normalization rationale
