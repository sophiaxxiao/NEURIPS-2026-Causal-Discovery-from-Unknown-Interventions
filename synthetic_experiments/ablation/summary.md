# Rebuttal Ablation Suite — Summary for Reviewers

**Paper:** Causal Discovery from Unseen Environments
**Methods evaluated:** ICA-LiNGAM, DirectLiNGAM (both operate on pooled data, no environment labels required)
**Focus:** Sensitivity of the LiNGAM-based identifier to four assumption violations.

---

## Experimental Design Overview

All four ablations follow a **one-factor-at-a-time** design. Each non-baseline condition changes exactly one parameter relative to a fixed shared baseline. This isolates the effect of each violation and admits paired inference (differences are computed seed-by-seed against the same baseline graph, same observational data, same observational noise).

### Shared baseline

| Parameter | Value |
|---|---|
| Graph | Erdős–Rényi, d=10 nodes, 10 expected edges, signed weights U([0.5, 2]) |
| Noise | i.i.d. Gaussian, σ=1 per node |
| Interventions | Bernoulli(0.5) per node per sample, pooled across unique patterns |
| Shift strength | δ=1.0 × per-node observational std |
| Off-target effects | None |
| n_interventional | 2 000 |
| n_observational | 100 |
| Seeds | 0–99 (100 independent DAGs) |

The linear SEM is X = WX + E, where W is the DAG adjacency matrix and E_i ~ N(μ_i, 1). In an intervened environment where node j is targeted, μ_j is shifted from 0 to δ_j; all other nodes remain at μ=0.

---

## Ablation A — Unequal Environment Sizes

**Reviewer concern addressed:** robustness to imbalanced data across environments.

**Mechanism.** In the simulator's Bernoulli mode, environment sizes emerge from the empirical frequency of each unique intervention pattern. We make these frequencies unequal by assigning per-node intervention probabilities

    p_i(h) = 0.5 + h · z_σ(i),    z = linspace(−1, 1, d)

where σ is a seed-specific random permutation of z, fixed across h values for each graph seed. The mean intervention probability remains exactly 0.5 for all h.

**Severity levels:**

| h | p range | Approx. max/min freq. ratio |
|---|---|---|
| 0.00 | [0.50, 0.50] | 1.0 (baseline) |
| 0.15 | [0.35, 0.65] | ~3.5 |
| 0.30 | [0.20, 0.80] | ~16 |
| 0.45 | [0.05, 0.95] | ~361 |

**What is held fixed:** DAG, observational data, total n_interventional=2000, observational noise, shift strength δ=1.0.

**Normalization rationale:** Fixing mean p_i=0.5 holds the expected total number of interventional events constant. Any change in SHD is attributable to imbalance, not to more or fewer interventions overall.

**Note on environment-aware methods:** At h=0.45, the p-vector concentrates samples more strongly in high-frequency patterns, increasing the fraction of "rich" environments (≥10 samples). UT-IGSP and IGSP discard sparse environments; their behavior therefore partially reflects the environment-richness change rather than pure size imbalance. LiNGAM results are unaffected by environment labels.

---

## Ablation B — Heterogeneous Intervention Strengths

**Reviewer concern addressed:** robustness when some targets are shifted strongly and others weakly.

**Mechanism.** Replace the uniform per-node delta δ_i=δ with

    δ_i(τ) = exp(τ · z_σ(i)) / sqrt(mean_j exp(2τ · z_j)),

where σ is again a seed-specific permutation fixed across τ values. The normalization guarantees

    mean_i δ_i(τ)² = 1  for all τ.

The effective shift on node i in an intervened environment is then δ_i(τ) × sqrt(Var(X_obs[:,i])).

**Severity levels (d=10 nodes):**

| τ | δ_min | δ_max | max/min |
|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 1.0 (baseline) |
| 0.35 | 0.671 | 1.351 | 2.0 |
| 0.70 | 0.413 | 1.673 | 4.1 |
| 1.00 | 0.258 | 1.905 | 7.4 |

**What is held fixed:** DAG, observational data, total n_interventional=2000, balanced p_i=0.5, zero off-target effects.

**Normalization rationale:** Fixing mean(δ_i²)=1 preserves the RMS shift energy. Without this normalization, increasing τ would also increase total signal energy, conflating the two effects. The ablation isolates the role of *spread* in shift magnitudes.

---

## Ablation C — Uniformly Weak Interventions

**Reviewer concern addressed:** whether the method degrades gracefully as interventions become weaker relative to observational noise.

**Mechanism.** Reduce the common scalar shift δ, applied uniformly across all nodes. The observational noise standard deviation is 1.0, so δ has a direct signal-to-noise interpretation: δ=0.5 means the mean shift equals half the noise std.

**Severity levels:**

| δ | Shift / noise std |
|---|---|
| 1.00 | 100% (baseline) |
| 0.75 | 75% |
| 0.50 | 50% |
| 0.25 | 25% |
| 0.10 | 10% |

**What is held fixed:** DAG, observational data, total n_interventional=2000, balanced p_i=0.5, zero off-target effects, homogeneous δ across nodes.

**Relationship to non-Gaussianity.** A mean-shifted Gaussian noise source E_i = N(δ · B_i, 1) (B_i ~ Bernoulli(p)) has excess kurtosis ∝ p(1−p)(1−6p(1−p))·δ⁴. As δ → 0, the pooled distribution approaches Gaussian, and the ICA-based identifier loses its theoretical signal. Ablation C therefore traces the transition from the non-Gaussian regime to the near-Gaussian regime and measures how fast accuracy degrades.

No normalization is applied; the reduction in δ is the quantity of interest.

---

## Ablation D — Correlated Off-Target Effects

**Reviewer concern addressed:** robustness when interventions "leak" to unintended nodes via a shared mechanism, violating source independence.

**Mechanism.** Generate a **random derangement** π of {0,…,d−1} (π(i)≠i for all i) once per graph seed, fixed across ρ values. When node i is intervened (Bernoulli indicator B_i=1), node π(i) receives an off-target shift driven by the *same* indicator:

    U_i       = c(ρ) · δ_i · B_i
    U_{π(i)} += c(ρ) · ρ  · δ_i · B_i
    c(ρ) = 1 / sqrt(1 + ρ²).

In vectorized form, the effective shift vector is

    U = (V + ρ P V) / sqrt(1 + ρ²),    V_j = δ_j · B_j,

where P is the permutation matrix of π. Crucially, the **same Bernoulli draw** B_i governs both the intended and off-target shift, creating genuine statistical dependence between the noise sources (E_i and E_{π(i)} become correlated in the pooled distribution).

**Severity levels:**

| ρ | Off-target fraction | Interpretation |
|---|---|---|
| 0.00 | 0% | None (baseline) |
| 0.10 | 10% | Weak leakage |
| 0.25 | 24% | Mild leakage |
| 0.50 | 45% | Moderate leakage |
| 1.00 | 71% | Equal intended/off-target amplitude |

**What is held fixed:** DAG, observational data, total n_interventional=2000, balanced p_i=0.5, δ=1.0.

**Normalization rationale:** c(ρ)=1/√(1+ρ²) ensures that the total shift energy per intervention event equals δ_i² regardless of ρ. Without this, increasing ρ would simultaneously increase total energy, which is a separate confounder. The ablation isolates the role of *cross-source dependence*, not total energy.

**Target information for methods:** Environment-aware methods receive the **actual** affected target set {i, π(i)} (not nominal {i} alone) in the primary comparison. Providing incorrect target labels would unfairly penalize label-aware methods and misattribute their accuracy loss to the off-target mechanism. The `sim.actual_targets` dict records both nominal and actual targets for reproducibility.

---

## Timing and Resource Requirements

Measured on one 10-core M4 MacBook Air, Python 3.13, **2 methods (ICA-LiNGAM, DirectLiNGAM)**:

| Configuration | Serial time | n_jobs=8 |
|---|---|---|
| 20 seeds × 15 conditions | <30 s | <5 s |
| 50 seeds × 15 conditions | ~1 min | ~10 s |
| **100 seeds × 15 conditions** | **~2 min** | **~20 s** |

**The full 100-seed experiment completes in well under 1 hour.** No reduction in seeds or conditions is needed.

### Why LiNGAM only?

Both ICA-LiNGAM and DirectLiNGAM operate on the **pooled dataset** (all interventional + observational samples stacked). They require no environment labels and are unaffected by how many environments there are or how samples are distributed across them. This makes them the natural ablation target:

- The ablation tests whether the *non-Gaussianity signal* that LiNGAM exploits is preserved under each violation.
- UT-IGSP, IGSP, and GIES require environment-level CI tests. In Bernoulli mode with n_interventional=2000 on d=10 nodes, ~800 unique patterns each contain only 2–3 samples, which is far below the minimum for reliable CI testing. Those methods degenerate in this regime and would not give interpretable ablation results.

---

## Commands

```bash
# From synthetic_experiments/

# Full experiment (~7 min serial, ~1 min with n_jobs=8)
python experiments/run_rebuttal_ablations.py \
    --config configs/rebuttal_ablations.yaml \
    --results-dir results/rebuttal_ablations \
    --n-jobs 8 \
    --resume

# Aggregate
python analysis/aggregate_rebuttal_ablations.py \
    --results-dir results/rebuttal_ablations

# Plot
python analysis/plot_rebuttal_ablations.py \
    --results-dir results/rebuttal_ablations \
    --output-dir figures/rebuttal_ablations

# Tests
pytest tests/test_rebuttal_ablations.py -v
```

---

## Statistical Analysis

- **Point estimates:** mean SHD and mean directed F1 across 100 seeds, per (condition, method).
- **Uncertainty:** standard error across seeds (shown as error bars).
- **Paired differences:** SHD(condition, seed) − SHD(baseline, seed), averaged across seeds. These are paired because each seed uses the same DAG in all conditions.
- **Bootstrap CIs:** 95% confidence intervals on the paired mean difference, by resampling the 100 graph seeds (not individual observations). Bootstrap RNG seed is fixed at 42.
- **No p-values** are reported; the goal is characterizing the sensitivity curve, not hypothesis testing.

---

## Results

Mean SHD (↓ better) and directed F1 (↑ better) across 100 seeds ± standard error.

### Ablation A — Unequal Environment Sizes

| h (severity) | DirectLiNGAM SHD | DirectLiNGAM F1 | ICA-LiNGAM SHD | ICA-LiNGAM F1 |
|---|---|---|---|---|
| 0.00 (baseline) | 0.60 ± 0.10 | 0.964 ± 0.006 | 1.30 ± 0.17 | 0.910 ± 0.011 |
| 0.15 | 0.64 ± 0.11 | 0.967 ± 0.005 | 1.37 ± 0.18 | 0.914 ± 0.010 |
| 0.30 | 0.81 ± 0.12 | 0.959 ± 0.006 | 1.78 ± 0.23 | 0.895 ± 0.011 |
| 0.45 | 1.21 ± 0.17 | 0.944 ± 0.007 | 2.15 ± 0.23 | 0.876 ± 0.012 |

### Ablation B — Heterogeneous Intervention Strengths

| τ (severity) | DirectLiNGAM SHD | DirectLiNGAM F1 | ICA-LiNGAM SHD | ICA-LiNGAM F1 |
|---|---|---|---|---|
| 0.00 (baseline) | 0.60 ± 0.10 | 0.964 ± 0.006 | 1.30 ± 0.17 | 0.910 ± 0.011 |
| 0.35 | 0.95 ± 0.15 | 0.913 ± 0.012 | 1.11 ± 0.16 | 0.914 ± 0.012 |
| 0.70 | 1.48 ± 0.20 | 0.879 ± 0.015 | 1.51 ± 0.20 | 0.895 ± 0.013 |
| 1.00 | 2.17 ± 0.29 | 0.837 ± 0.018 | 2.01 ± 0.25 | 0.859 ± 0.014 |

### Ablation C — Uniformly Weak Interventions

| δ (severity) | DirectLiNGAM SHD | DirectLiNGAM F1 | ICA-LiNGAM SHD | ICA-LiNGAM F1 |
|---|---|---|---|---|
| 1.00 (baseline) | 0.60 ± 0.10 | 0.964 ± 0.006 | 1.30 ± 0.17 | 0.910 ± 0.011 |
| 0.75 | 0.66 ± 0.10 | 0.948 ± 0.008 | 1.10 ± 0.15 | 0.919 ± 0.010 |
| 0.50 | 1.17 ± 0.17 | 0.897 ± 0.015 | 1.42 ± 0.19 | 0.892 ± 0.014 |
| 0.25 | 4.41 ± 0.29 | 0.666 ± 0.020 | 1.77 ± 0.21 | 0.867 ± 0.013 |
| 0.10 | 6.58 ± 0.40 | 0.550 ± 0.019 | 2.42 ± 0.25 | 0.832 ± 0.014 |

### Ablation D — Correlated Off-Target Effects

| ρ (severity) | DirectLiNGAM SHD | DirectLiNGAM F1 | ICA-LiNGAM SHD | ICA-LiNGAM F1 |
|---|---|---|---|---|
| 0.00 (baseline) | 0.60 ± 0.10 | 0.964 ± 0.006 | 1.30 ± 0.17 | 0.910 ± 0.011 |
| 0.10 | 2.31 ± 0.30 | 0.883 ± 0.014 | 2.86 ± 0.24 | 0.853 ± 0.011 |
| 0.25 | 4.80 ± 0.31 | 0.767 ± 0.013 | 5.11 ± 0.33 | 0.749 ± 0.013 |
| 0.50 | 8.27 ± 0.33 | 0.629 ± 0.011 | 8.42 ± 0.42 | 0.631 ± 0.013 |
| 1.00 | 10.08 ± 0.40 | 0.556 ± 0.015 | 10.70 ± 0.50 | 0.550 ± 0.014 |

---

## Reproducibility

Every result is fully reproducible:
- DAG: `DAG.erdos_renyi(seed=seed, ...)` — same seed → same graph.
- Observational data: `np.random.seed(seed)` is called at the start of each simulation.
- Intervention patterns: `np.random.seed(seed)` is called inside `setup_interventions()`.
- Ablation-specific permutations and derangements: `np.random.default_rng(seed)`.
- Checkpoints are atomic JSON files; interrupted runs resume from the last completed unit.
- Git hash, Python version, and package versions are recorded in `manifest.json`.
