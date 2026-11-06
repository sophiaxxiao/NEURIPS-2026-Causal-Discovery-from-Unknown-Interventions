#!/usr/bin/env python3
# coding=utf-8
"""
ResidualDependenceSHDAblation.py

Relates intervention-induced source dependence to ICA-LiNGAM SHD under
single-node atomic interventions with controlled environment imbalance.

Scientific objective
--------------------
Quantify how oracle total correlation (TC) of pooled effective disturbances,
oracle pairwise dependence diagnostics, and held-out fitted-residual dependence
relate to ICA-LiNGAM SHD, as environment imbalance increases.

Primary question
----------------
As intervention-induced source dependence increases under mutually exclusive
single-node interventions, does ICA-LiNGAM graph recovery deteriorate, after
accounting descriptively for intervention-induced non-Gaussianity?

Adjacency convention (verified at startup)
------------------------------------------
W[i,j] = weight of edge i -> j.
SEM:   X = X @ W + E   =>   E = X @ (I - W).
Source: IIDLinearSimulation.py lines 178-194.

Fixed parameters (resolved from repository)
--------------------------------------------
Source: configs/rebuttal_ablations.yaml
    TRAINING_SAMPLE_SIZE = 2000   (n_interventional_total)
    DELTA                = 1.0    (baseline_delta)
    N_OBSERVATIONAL      = 100    (n_observational)
Source: DependentSource.py
    intervention_mode = 'single_node'
    include_no_intervention = False  (obs block prepended separately, matching
        ERDagsComparison.py subsample = vstack([X_obs, X_interv]))
Source: ERDagsComparison.py
    n_nodes, n_edges = 10, 10
    weight_range     = [0.5, 2.0]
    seeds            = range(100)

Scientific distinction
----------------------
oracle effective disturbances  E_oracle = X_diag @ (I - W_true)
    Primary dependence object. Independent of estimation error.
    Used for oracle TC, oracle pairwise MI, oracle distance correlation.

fitted residuals  E_hat = X_diag @ (I - W_hat)
    Model-checking diagnostic. Partly confounded by graph-estimation error.
    Labelled 'fitted_residual_*' throughout. Never used as primary evidence.
"""

import argparse
import datetime
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from castle.algorithms import ICALiNGAM
from castle.datasets import DAG
from castle.metrics import MetricsDAG

from MakeDataset import ShiftSimulation

# ---------------------------------------------------------------------------
# Fixed simulation parameters
# ---------------------------------------------------------------------------

# Source: configs/rebuttal_ablations.yaml
TRAINING_SAMPLE_SIZE   = 2000
N_OBSERVATIONAL        = 100

# δ sweep: spans oracle TC ≈ 0.13 nats (δ=1) → ≈ 2.0 nats (δ=10) at balanced r=1
# δ=1.0 preserves existing checkpoints (backward-compatible).
DELTA_VALUES           = [1.0, 2.0, 5.0, 10.0]

# Source: DependentSource.py (single_node convention)
INTERVENTION_MODE      = "single_node"
INCLUDE_NO_INTERVENTION = False   # obs block prepended instead

# Source: ERDagsComparison.py
N_NODES                = 10
N_EDGES                = 10
WEIGHT_RANGE           = [0.5, 2.0]
SEEDS                  = list(range(100))

# This ablation
DIAGNOSTIC_SAMPLE_SIZE = 5000
IMBALANCE_RATIOS       = [1, 2, 5, 10]
N_KSG_NEIGHBORS        = 5        # KSG k-NN, fixed across all trials
N_BOOTSTRAP            = 2000     # cluster-bootstrap repetitions for CIs
BOOTSTRAP_SEED         = 42
DIAG_SEED_OFFSET       = 10_000   # diagnostic RNG seed = seed + DIAG_SEED_OFFSET
DCOR_SUBSAMPLE         = 1000     # rows used for distance-correlation (O(n²))
N_JOBS_DEFAULT         = 8


# ---------------------------------------------------------------------------
# Startup: print resolved parameters
# ---------------------------------------------------------------------------

def _print_resolved_params() -> None:
    print("=" * 64)
    print("  Resolved simulation parameters")
    print("=" * 64)
    print(f"  Training sample size   : {TRAINING_SAMPLE_SIZE}")
    print(f"    Source : configs/rebuttal_ablations.yaml (n_interventional_total)")
    print(f"  Diagnostic sample size : {DIAGNOSTIC_SAMPLE_SIZE}")
    print(f"    Source : this script (oracle, never used for fitting)")
    print(f"  Intervention shift δ   : {DELTA_VALUES}  (sweep)")
    print(f"    Source : extension of configs/rebuttal_ablations.yaml (baseline_delta=1.0)")
    print(f"    Applied: delta_per_node[j] = δ * var(X_obs[:,j])")
    print(f"  Observational count    : {N_OBSERVATIONAL}")
    print(f"    Source : configs/rebuttal_ablations.yaml (n_observational)")
    print(f"  No-intervention prob   : 0  (obs block prepended, as ERDagsComparison.py)")
    print(f"    Source : ERDagsComparison.py subsample = vstack([X_obs, X_interv])")
    print(f"  Intervention mode      : {INTERVENTION_MODE}, include_no_intervention={INCLUDE_NO_INTERVENTION}")
    print(f"    Source : DependentSource.py")
    print(f"  Bernoulli arm          : p_intervention = 1/{N_NODES} = {1/N_NODES:.4f} per node")
    print(f"    TC = 0 exactly (independent sources); same δ, same true DAG per seed")
    print(f"  DAG                    : ER({N_NODES} nodes, {N_EDGES} edges), W in {WEIGHT_RANGE}")
    print(f"    Source : ERDagsComparison.py")
    print(f"  Seeds                  : 0..{SEEDS[-1]}  ({len(SEEDS)} graphs)")
    print(f"  Imbalance ratios       : {IMBALANCE_RATIOS}  (single_node arm only)")
    print(f"    Mechanism: geomspace(1, r, {N_NODES}) permuted per seed, renormalised")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Adjacency convention verification
# ---------------------------------------------------------------------------

def _verify_convention() -> None:
    """
    Verify E = X @ (I - W) against manually simulated data.

    Hand-constructs a 3-node DAG (0->2, 1->2) with known noise and confirms
    that the formula recovers those noise values to numerical precision.
    This matches the SEM in IIDLinearSimulation.py:
        X[:, j] = X[:, parents(j)] @ W[parents(j), j] + noise_j
    """
    W = np.array([[0.0, 0.0, 0.7],
                  [0.0, 0.0, 0.5],
                  [0.0, 0.0, 0.0]])   # W[i,j] = weight of i->j
    d, n = 3, 3000
    rng = np.random.default_rng(1_234_567)
    noise = rng.standard_normal((n, d))    # known true noise

    # Simulate in topological order (0, 1, 2)
    X = np.zeros((n, d))
    X[:, 0] = noise[:, 0]                              # root
    X[:, 1] = noise[:, 1]                              # root
    X[:, 2] = W[0, 2] * X[:, 0] + W[1, 2] * X[:, 1] + noise[:, 2]

    E_hat = X @ (np.eye(d) - W)
    rel_err = np.max(np.abs(E_hat - noise)) / (noise.std() + 1e-12)
    if rel_err > 1e-9:
        raise AssertionError(
            f"Convention verification FAILED: E = X @ (I-W) has rel_err = {rel_err:.3e}.\n"
            "Inspect IIDLinearSimulation.py for the correct adjacency convention."
        )
    print(f"  [OK] E = X @ (I-W)  max_rel_err = {rel_err:.2e}")


# ---------------------------------------------------------------------------
# Mixture weights
# ---------------------------------------------------------------------------

def build_mixture_weights(imbalance_ratio: float, d: int, seed: int) -> np.ndarray:
    """
    Build probabilities for the d single-node environments.

    A fixed, seed-specific permutation is used so that which nodes are
    favoured stays consistent across imbalance ratios for a given seed.
    The permutation RNG is independent of the simulation seed.

    Parameters
    ----------
    imbalance_ratio : float  max/min frequency ratio r >= 1
    d               : int    number of nodes
    seed            : int    graph seed (permutation derived from this)

    Returns
    -------
    target_probs : (d,)  probabilities summing to 1
    """
    perm_rng = np.random.default_rng(seed * 7919 + 13)   # independent RNG
    perm = perm_rng.permutation(d)

    raw = np.geomspace(1.0, float(imbalance_ratio), d)    # ascending weights
    raw_perm = raw[perm]                                   # permute per seed
    target_probs = raw_perm / raw_perm.sum()

    assert np.isclose(target_probs.sum(), 1.0, atol=1e-9), \
        "Mixture weights do not sum to 1"
    return target_probs


# ---------------------------------------------------------------------------
# Oracle total correlation (exact Gaussian mixture, MC estimate)
# ---------------------------------------------------------------------------

def compute_oracle_tc(
    E_diag: np.ndarray,
    mixture_weights: np.ndarray,
    delta_per_node: np.ndarray,
) -> tuple:
    """
    Monte Carlo estimate of oracle total correlation.

    Under single-node atomic interventions (include_no_intervention=False),
    the pooled effective disturbances E = X(I-W) form a Gaussian mixture:

        p(e) = sum_k  w_k  *  prod_j  N(e_j ; mu_{k,j}, 1)

    where:
        w_k         = mixture_weights[k]   (prob. of environment k)
        mu_{k,j}    = delta_per_node[k]    if j == k
                    = 0                    otherwise
        noise std   = 1.0  (hardcoded in MakeDataset.simulate_interventions)

    Marginal of E_j:
        p(e_j) = w_j * N(e_j; delta_j, 1)  +  (1 - w_j) * N(e_j; 0, 1)

    TC = E[ log p(E) - sum_j log p_j(E_j) ]
       ~ (1/M) sum_m [ log p(e^m) - sum_j log p_j(e_j^m) ]

    Parameters
    ----------
    E_diag         : (M, d)  oracle effective disturbances (diagnostic sample)
    mixture_weights: (d,)    intervention-environment probabilities
    delta_per_node : (d,)    mean shifts (from training sim)

    Returns
    -------
    tc    : float  MC estimate (nats)
    tc_se : float  Monte Carlo standard error
    """
    M, d = E_diag.shape
    K = len(mixture_weights)
    assert K == d, "Expected one environment per node"
    LOG2PI_HALF = 0.5 * np.log(2.0 * np.pi)

    # ── Joint log-density ─────────────────────────────────────────────────
    # log p(e^m) = logsumexp_k [ log w_k + sum_j log N(e_j; mu_{k,j}, 1) ]
    log_comps = np.empty((M, K))
    for k in range(K):
        mu_k = np.zeros(d)
        mu_k[k] = delta_per_node[k]
        sq = np.sum((E_diag - mu_k[np.newaxis, :]) ** 2, axis=1)
        log_comps[:, k] = np.log(mixture_weights[k] + 1e-300) - 0.5 * sq - d * LOG2PI_HALF

    log_p_joint = logsumexp(log_comps, axis=1)    # (M,)

    # ── Marginal log-densities ─────────────────────────────────────────────
    log_p_marg = np.zeros(M)
    for j in range(d):
        wj = mixture_weights[j]
        ej = E_diag[:, j]
        dj = delta_per_node[j]

        log_shifted = np.log(wj + 1e-300) - 0.5 * (ej - dj) ** 2 - LOG2PI_HALF
        log_null    = np.log(1.0 - wj + 1e-300) - 0.5 * ej ** 2 - LOG2PI_HALF
        log_p_marg += logsumexp(
            np.column_stack([log_shifted, log_null]), axis=1
        )

    tc_vals = log_p_joint - log_p_marg     # (M,)
    return float(tc_vals.mean()), float(tc_vals.std(ddof=1) / np.sqrt(M))


# ---------------------------------------------------------------------------
# Pairwise KSG mutual information
# ---------------------------------------------------------------------------

def compute_pairwise_ksg_mi(
    E: np.ndarray, n_neighbors: int = N_KSG_NEIGHBORS
) -> tuple:
    """
    All pairwise KSG mutual information via sklearn.

    Returns (mean_mi, median_mi, max_mi) over the d*(d-1)/2 unique pairs.
    """
    d = E.shape[1]
    mi_vals = []
    for j in range(d):
        mi_row = mutual_info_regression(
            E, E[:, j], n_neighbors=n_neighbors, random_state=0
        )
        for k in range(j):           # lower triangle only, avoid diagonal
            mi_vals.append(float(mi_row[k]))

    arr = np.asarray(mi_vals)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


# ---------------------------------------------------------------------------
# Pairwise distance correlation (manual; dcor not available)
# ---------------------------------------------------------------------------

def _dcor_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Biased distance correlation between two 1-D arrays."""
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])

    def _dc(M):
        rm = M.mean(axis=1, keepdims=True)
        cm = M.mean(axis=0, keepdims=True)
        return M - rm - cm + M.mean()

    Ac, Bc = _dc(A), _dc(B)
    dcov2 = (Ac * Bc).mean()
    dvar_x = (Ac * Ac).mean()
    dvar_y = (Bc * Bc).mean()
    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_x * dvar_y)))


def compute_pairwise_dcor(
    E: np.ndarray, max_n: int = DCOR_SUBSAMPLE, seed: int = 0
) -> float:
    """Mean pairwise distance correlation, subsampled to max_n rows."""
    M, d = E.shape
    if M > max_n:
        idx = np.random.default_rng(seed).choice(M, size=max_n, replace=False)
        E = E[idx]
    vals = [_dcor_pair(E[:, i], E[:, j])
            for i in range(d) for j in range(i + 1, d)]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Absolute excess kurtosis
# ---------------------------------------------------------------------------

def compute_abs_excess_kurtosis(E: np.ndarray) -> tuple:
    """Mean and median absolute excess kurtosis across columns."""
    from scipy.stats import kurtosis as _kurtosis
    k = np.array([abs(float(_kurtosis(E[:, j], fisher=True)))
                  for j in range(E.shape[1])])
    return float(k.mean()), float(np.median(k))


# ---------------------------------------------------------------------------
# Oracle diagnostic sample generator
# ---------------------------------------------------------------------------

def generate_diagnostic_data(
    mixture_weights: np.ndarray,
    delta_per_node: np.ndarray,
    true_W: np.ndarray,
    n_diag: int,
    rng_seed: int,
) -> tuple:
    """
    Generate an independent diagnostic sample from the known mixture.

    E_oracle = oracle effective disturbances (no estimation error).
    X_diag   = corresponding observed data, used only for fitted residuals.

    SEM forward pass: X(I-W) = E  =>  X = E @ inv(I-W)
    Computed via solve((I-W)^T, E^T)^T  (numerically stable).

    Parameters
    ----------
    mixture_weights : (K,)  environment probabilities  (K == d)
    delta_per_node  : (d,)  mean shifts
    true_W          : (d,d) true adjacency  W[i,j] = weight of i->j
    n_diag, rng_seed: int

    Returns
    -------
    E_oracle : (n_diag, d)
    X_diag   : (n_diag, d)
    """
    d = len(delta_per_node)
    K = len(mixture_weights)
    rng = np.random.default_rng(rng_seed)

    # Sample component assignments
    components = rng.choice(K, size=n_diag, p=mixture_weights)

    # i.i.d. N(0,1) base noise + mean shifts
    E = rng.standard_normal((n_diag, d))
    for k in range(K):
        mask = components == k
        if mask.any():
            E[mask, k] += delta_per_node[k]

    # Reconstruct X: solve (I-W)^T X^T = E^T
    I_d = np.eye(d)
    X = np.linalg.solve((I_d - true_W).T, E.T).T

    return E, X


def generate_bernoulli_diagnostic_data(
    p_intervention: float,
    delta_per_node: np.ndarray,
    true_W: np.ndarray,
    n_diag: int,
    rng_seed: int,
) -> tuple:
    """
    Generate an independent diagnostic sample for the Bernoulli arm.

    Each node j is independently shifted with probability p_intervention,
    independently of all other nodes. Sources are independent across nodes
    by construction, so oracle TC = 0 exactly.

    Parameters
    ----------
    p_intervention : float   probability each node is shifted (= 1/N_NODES)
    delta_per_node : (d,)    mean shifts (same as training arm)
    true_W         : (d,d)   true adjacency
    n_diag, rng_seed : int

    Returns
    -------
    E_oracle : (n_diag, d)   independent effective disturbances
    X_diag   : (n_diag, d)   corresponding observed data
    """
    d = len(delta_per_node)
    rng = np.random.default_rng(rng_seed)

    interventions = rng.binomial(1, p_intervention, size=(n_diag, d))
    E = rng.standard_normal((n_diag, d))
    E += interventions * delta_per_node[np.newaxis, :]

    I_d = np.eye(d)
    X = np.linalg.solve((I_d - true_W).T, E.T).T
    return E, X


# ---------------------------------------------------------------------------
# TC validation tests (run at startup)
# ---------------------------------------------------------------------------

def _validate_tc() -> None:
    """
    Minimal validation checks for compute_oracle_tc.

    1. TC >= 0 up to MC noise.
    2. TC > 0 for strong single-node shifts (mutual exclusivity creates dependence).
    3. Weights summing to 1 requirement.
    4. Marginal density agrees with numerical marginalisation.
    """
    d = 4
    w = np.ones(d) / d
    delta = np.ones(d) * 2.0   # large shift for clear signal

    rng = np.random.default_rng(77)
    n = 20_000
    comps = rng.choice(d, size=n, p=w)
    E = rng.standard_normal((n, d))
    for k in range(d):
        E[comps == k, k] += delta[k]

    tc, tc_se = compute_oracle_tc(E, w, delta)

    # TC must be non-negative (it is a KL divergence)
    assert tc >= -4 * tc_se, \
        f"TC negative beyond 4-sigma MC noise: {tc:.4f} ± {tc_se:.4f}"

    # TC must be clearly positive for strong mutual-exclusion shifts
    assert tc > 0.05, \
        f"TC suspiciously small for large delta={delta[0]}: {tc:.4f}"

    # Verify marginal density: numerical marginalisation of E[:,0]
    j = 0
    from scipy.stats import norm as _norm
    e0_sample = E[:, j]
    # analytical marginal at a few test points
    test_pts = np.array([-2.0, 0.0, delta[0], 2.0])
    for tp in test_pts:
        p_analytic = (w[j] * _norm.pdf(tp, loc=delta[j], scale=1.0)
                      + (1 - w[j]) * _norm.pdf(tp, loc=0.0, scale=1.0))
        # numerical marginalisation over all components
        p_num = sum(w[k] * _norm.pdf(tp, loc=(delta[k] if k == j else 0.0), scale=1.0)
                    for k in range(d))
        assert abs(p_analytic - p_num) < 1e-10, \
            f"Marginal mismatch at {tp}: {p_analytic:.6f} vs {p_num:.6f}"

    print(f"  [OK] TC validation: balanced single-node TC = {tc:.4f} ± {tc_se:.4f}")


# ---------------------------------------------------------------------------
# SHD helper
# ---------------------------------------------------------------------------

def _safe_shd(adj, true_dag) -> float:
    if adj is None:
        return float("nan")
    return float(MetricsDAG(adj, true_dag).metrics["shd"])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(seed: int, delta: float, ir: float, arm: str,
               results_dir: str) -> str:
    """New checkpoint naming (includes δ and arm)."""
    if arm == "bernoulli":
        return os.path.join(
            results_dir, f"ckpt_rdsa_s{seed:04d}_d{delta:.4f}_bern.json"
        )
    return os.path.join(
        results_dir, f"ckpt_rdsa_s{seed:04d}_d{delta:.4f}_r{ir:.0f}_sn.json"
    )


def _ckpt_path_legacy(seed: int, ir: float, results_dir: str) -> str:
    """Legacy naming used before the δ-sweep extension (δ=1.0, single_node)."""
    return os.path.join(results_dir, f"ckpt_rdsa_s{seed:04d}_r{ir:.0f}.json")


def _load_ckpt(seed: int, delta: float, ir: float, arm: str,
               results_dir: str):
    # New-format checkpoint
    p = _ckpt_path(seed, delta, ir, arm, results_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    # Backward-compat: legacy naming for δ=1.0 single_node trials already done
    if arm == "single_node" and abs(delta - 1.0) < 1e-9:
        p_leg = _ckpt_path_legacy(seed, ir, results_dir)
        if os.path.exists(p_leg):
            with open(p_leg) as f:
                row = json.load(f)
            row.setdefault("delta", 1.0)
            row.setdefault("arm", "single_node")
            return row
    return None


def _save_ckpt(row: dict, results_dir: str) -> None:
    ir = row.get("imbalance_ratio", 0.0)
    if ir is None or (isinstance(ir, float) and np.isnan(ir)):
        ir = 0.0
    p = _ckpt_path(row["seed"], row["delta"], ir, row["arm"], results_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f, default=float)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Core trial function
# ---------------------------------------------------------------------------

def run_trial(seed: int, delta: float, imbalance_ratio: float, arm: str,
              results_dir: str, resume: bool = True) -> dict:
    """
    One experimental unit: (seed, delta, imbalance_ratio, arm).

    arm = "single_node" : mutually exclusive single-node interventions (TC > 0).
    arm = "bernoulli"   : independent Bernoulli(1/d) interventions per node (TC = 0).
                          imbalance_ratio is ignored for the Bernoulli arm.

    Returns a dict with all output columns.
    """
    t0 = time.perf_counter()

    # ── Resume check ────────────────────────────────────────────────────────
    if resume:
        cached = _load_ckpt(seed, delta, imbalance_ratio, arm, results_dir)
        if cached is not None:
            return cached

    row: dict = {
        "seed":                      seed,
        "n_nodes":                   N_NODES,
        "n_edges":                   N_EDGES,
        "training_sample_size":      TRAINING_SAMPLE_SIZE,
        "diagnostic_sample_size":    DIAGNOSTIC_SAMPLE_SIZE,
        "delta":                     delta,
        "arm":                       arm,
        "imbalance_ratio":           imbalance_ratio if arm == "single_node"
                                     else float("nan"),
        "observational_probability": 0.0,
        "fit_failed":                False,
        "failure_message":           "",
    }

    try:
        # ── 1. True DAG (identical to ERDagsComparison.py) ──────────────────
        true_dag = DAG.erdos_renyi(
            n_nodes=N_NODES, n_edges=N_EDGES, seed=seed,
            weight_range=WEIGHT_RANGE,
        )
        W = true_dag    # W[i,j] = weight of i->j

        # ── 2. Training simulation ───────────────────────────────────────────
        if arm == "single_node":
            target_probs = build_mixture_weights(imbalance_ratio, N_NODES, seed)
            row["target_probabilities"] = str(target_probs.tolist())

            sim = ShiftSimulation(
                n_nodes=N_NODES, n_edges=N_EDGES,
                n_samples=TRAINING_SAMPLE_SIZE,
                n_observational=N_OBSERVATIONAL,
                weighted_random_dag=true_dag,
                delta=delta,
                seed=seed,
                intervention_mode=INTERVENTION_MODE,
                include_no_intervention=INCLUDE_NO_INTERVENTION,
                single_node_probs=target_probs,
            )
            sim.simulate_observational()
            sim.compute_per_node_delta_from_observational(scalar=delta)
            sim.setup_interventions()
            sim.simulate_interventions()

            delta_per_node = sim.delta_per_node.copy()

            # Assert single-node contract
            for pattern in sim.datasets_interventional:
                assert sum(pattern) <= 1, \
                    f"Non-single-node pattern in interventional dict: {pattern}"
            assert np.all(sim.intervention_matrix.sum(axis=1) <= 1), \
                "intervention_matrix has rows with more than one active target"

            counts = np.array([sim.datasets_interventional[p]["X"].shape[0]
                               for p in sorted(sim.datasets_interventional,
                                               key=lambda pp: np.argmax(pp)
                                               if any(pp) else N_NODES)])
            row["realised_mixture_weights"] = str(
                (counts / counts.sum()).tolist()
            )

            X_obs    = sim.dataset_observational.X
            X_interv = np.vstack([d["X"] for d in sim.datasets_interventional.values()])
            X_train  = np.vstack([X_obs, X_interv])

        else:  # arm == "bernoulli"
            row["target_probabilities"]    = f"bernoulli(p={1/N_NODES:.4f})"
            row["realised_mixture_weights"] = "N/A"

            sim_bern = ShiftSimulation(
                n_nodes=N_NODES, n_edges=N_EDGES,
                n_samples=TRAINING_SAMPLE_SIZE,
                n_observational=N_OBSERVATIONAL,
                weighted_random_dag=true_dag,
                delta=delta,
                seed=seed,
                intervention_mode="bernoulli",
                p_intervention=1.0 / N_NODES,
            )
            sim_bern.simulate_observational()
            sim_bern.compute_per_node_delta_from_observational(scalar=delta)
            sim_bern.setup_interventions()
            sim_bern.simulate_interventions()

            delta_per_node = sim_bern.delta_per_node.copy()

            X_obs    = sim_bern.dataset_observational.X
            X_interv = np.vstack([d["X"] for d in sim_bern.datasets_interventional.values()])
            X_train  = np.vstack([X_obs, X_interv])

        # ── 3. Fit ICA-LiNGAM ────────────────────────────────────────────────
        estimated_binary_dag   = None
        estimated_weighted_dag = None
        shd_icalingam          = float("nan")

        try:
            model = ICALiNGAM()
            model.learn(X_train)
            estimated_binary_dag   = model.causal_matrix
            estimated_weighted_dag = model.weight_causal_matrix
            shd_icalingam          = _safe_shd(estimated_binary_dag, true_dag)
        except Exception as exc:
            row["fit_failed"]      = True
            row["failure_message"] = repr(exc)

        row["shd_icalingam"] = shd_icalingam

        # ── 4. Oracle diagnostic sample ──────────────────────────────────────
        # Different rng_seed offsets ensure training and diagnostic data are
        # independent within each arm, and the two arms don't share RNG paths.
        if arm == "single_node":
            E_oracle, X_diag = generate_diagnostic_data(
                mixture_weights=target_probs,
                delta_per_node=delta_per_node,
                true_W=W,
                n_diag=DIAGNOSTIC_SAMPLE_SIZE,
                rng_seed=seed + DIAG_SEED_OFFSET,
            )
            # ── 5a. Oracle TC (exact Gaussian mixture MC estimate) ───────────
            tc, tc_se = compute_oracle_tc(E_oracle, target_probs, delta_per_node)
        else:
            E_oracle, X_diag = generate_bernoulli_diagnostic_data(
                p_intervention=1.0 / N_NODES,
                delta_per_node=delta_per_node,
                true_W=W,
                n_diag=DIAGNOSTIC_SAMPLE_SIZE,
                rng_seed=seed + DIAG_SEED_OFFSET + 1,
            )
            # ── 5b. Bernoulli arm: TC = 0 exactly (independent sources) ──────
            tc, tc_se = 0.0, 0.0

        row["oracle_total_correlation"]       = tc
        row["oracle_total_correlation_mc_se"] = tc_se

        # ── 6. Oracle pairwise KSG MI ─────────────────────────────────────────
        mean_mi, med_mi, max_mi = compute_pairwise_ksg_mi(E_oracle)
        row["oracle_mean_pairwise_mi"]   = mean_mi
        row["oracle_median_pairwise_mi"] = med_mi
        row["oracle_max_pairwise_mi"]    = max_mi

        # ── 7. Oracle distance correlation ────────────────────────────────────
        row["oracle_mean_distance_corr"] = compute_pairwise_dcor(
            E_oracle, seed=seed
        )

        # ── 8. Oracle absolute excess kurtosis ───────────────────────────────
        mean_kurt, med_kurt = compute_abs_excess_kurtosis(E_oracle)
        row["oracle_mean_abs_excess_kurtosis"]   = mean_kurt
        row["oracle_median_abs_excess_kurtosis"] = med_kurt

        # ── 9. Fitted-residual dependence ────────────────────────────────────
        # NOTE: E_hat = X_diag @ (I - W_hat) is partly confounded by
        # graph-estimation error. Labelled 'fitted_residual_*'. Not primary evidence.
        if estimated_weighted_dag is not None:
            E_hat = X_diag @ (np.eye(N_NODES) - estimated_weighted_dag)
            fr_mean_mi, fr_med_mi, fr_max_mi = compute_pairwise_ksg_mi(E_hat)
            fr_dcor = compute_pairwise_dcor(E_hat, seed=seed + 1)
        else:
            fr_mean_mi = fr_med_mi = fr_max_mi = fr_dcor = float("nan")

        row["fitted_residual_mean_pairwise_mi"]   = fr_mean_mi
        row["fitted_residual_median_pairwise_mi"] = fr_med_mi
        row["fitted_residual_max_pairwise_mi"]    = fr_max_mi
        row["fitted_residual_mean_distance_corr"] = fr_dcor

    except Exception as exc:
        row["fit_failed"]      = True
        row["failure_message"] = repr(exc)

    row["runtime_seconds"] = time.perf_counter() - t0
    _save_ckpt(row, results_dir)
    return row


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def _cluster_bootstrap_mean(
    x: np.ndarray, seed_labels: np.ndarray,
    n_boot: int = N_BOOTSTRAP, rng_seed: int = BOOTSTRAP_SEED,
) -> tuple:
    """
    95% CI on mean(x) via cluster bootstrap, clustering by seed_labels.
    Returns (point_mean, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(rng_seed)
    unique = np.unique(seed_labels)
    n_seeds = len(unique)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(unique, size=n_seeds, replace=True)
        idx = np.concatenate([np.where(seed_labels == s)[0] for s in sel])
        boot[b] = x[idx].mean() if len(idx) > 0 else float("nan")
    return float(x.mean()), \
           float(np.nanpercentile(boot, 2.5)), \
           float(np.nanpercentile(boot, 97.5))


def _cluster_bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, seed_labels: np.ndarray,
    n_boot: int = N_BOOTSTRAP, rng_seed: int = BOOTSTRAP_SEED,
) -> tuple:
    """
    Spearman ρ with 95% CI via cluster bootstrap (resamples DAG seeds).

    The four imbalance conditions per DAG are repeated measures on the same
    graph; resampling by seed preserves this structure.

    Returns (point_r, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(rng_seed)
    unique = np.unique(seed_labels)
    n_seeds = len(unique)

    boot_r = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(unique, size=n_seeds, replace=True)
        idx = np.concatenate([np.where(seed_labels == s)[0] for s in sel])
        x_b, y_b = x[idx], y[idx]
        boot_r[b] = spearmanr(x_b, y_b)[0] if len(x_b) >= 2 else float("nan")

    point_r = spearmanr(x, y)[0]
    return float(point_r), float(np.nanpercentile(boot_r, 2.5)), \
           float(np.nanpercentile(boot_r, 97.5))


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute all statistical summaries; return structured dict."""
    # Patch legacy rows that pre-date the δ-sweep extension
    if "arm" not in df.columns:
        df = df.copy()
        df["arm"] = "single_node"
    if "delta" not in df.columns:
        df = df.copy()
        df["delta"] = 1.0

    ok = df[~df["fit_failed"].fillna(False)].copy()
    stats: dict = {}

    # ── Restrict Spearman analysis to single_node arm ────────────────────────
    ok_sn = ok[ok["arm"] == "single_node"].copy()

    pairs = [
        ("oracle_total_correlation",       "shd_icalingam", "oracle_TC_vs_SHD"),
        ("oracle_mean_pairwise_mi",         "shd_icalingam", "oracle_mean_MI_vs_SHD"),
        ("fitted_residual_mean_pairwise_mi","shd_icalingam", "fitted_residual_MI_vs_SHD"),
        ("oracle_mean_abs_excess_kurtosis", "shd_icalingam", "oracle_kurtosis_vs_SHD"),
    ]
    stats["spearman"] = {}
    seeds_arr = ok_sn["seed"].values
    for xcol, ycol, name in pairs:
        x = ok_sn[xcol].values.astype(float)
        y = ok_sn[ycol].values.astype(float)
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 4:
            stats["spearman"][name] = {
                "r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")
            }
            continue
        r, lo, hi = _cluster_bootstrap_spearman(
            x[valid], y[valid], seeds_arr[valid]
        )
        stats["spearman"][name] = {"r": r, "ci_lo": lo, "ci_hi": hi}
        print(f"  Spearman  {name}:  r={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ── Within-DAG OLS (seed-demeaned, single_node arm) ─────────────────────
    wda_cols = ["shd_icalingam", "oracle_total_correlation",
                "oracle_mean_abs_excess_kurtosis", "seed"]
    wda = ok_sn[wda_cols].dropna().copy()
    if len(wda) > N_NODES:
        for col in wda_cols[:-1]:
            wda[col + "_dm"] = wda[col] - wda.groupby("seed")[col].transform("mean")
        X_reg = np.column_stack([wda["oracle_total_correlation_dm"].values,
                                  wda["oracle_mean_abs_excess_kurtosis_dm"].values])
        y_reg = wda["shd_icalingam_dm"].values
        beta, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
        stats["within_dag_ols"] = {
            "beta_tc": float(beta[0]),
            "beta_ng": float(beta[1]),
            "note": (
                "Descriptive within-DAG OLS (seed fixed effects, single_node arm). "
                "beta_tc = change in demeaned SHD per 1-nat increase in demeaned "
                "oracle TC, adjusted descriptively for non-Gaussianity. "
                "Not a causal estimate."
            ),
        }
        print(f"  Within-DAG OLS (single_node):  "
              f"β_TC={beta[0]:+.3f}  β_NG={beta[1]:+.3f}")
    else:
        stats["within_dag_ols"] = {}

    # ── Condition summaries (single_node arm, δ × r) ─────────────────────────
    metric_cols = {
        "shd_icalingam":                   "SHD",
        "oracle_total_correlation":         "oracle TC",
        "oracle_mean_pairwise_mi":          "oracle MI",
        "fitted_residual_mean_pairwise_mi": "fit-resid MI",
        "oracle_mean_abs_excess_kurtosis":  "|kurtosis|",
    }
    cs = {}
    sn_grouped = ok_sn.groupby(["delta", "imbalance_ratio"], dropna=False)
    for (dv, ir), grp in sn_grouped:
        key = (float(dv), float(ir))
        fail_mask = (df["arm"] == "single_node") & \
                    (df["delta"] == dv) & \
                    (df["imbalance_ratio"].eq(ir) | (df["imbalance_ratio"].isna() & np.isnan(ir)))
        cs[key] = {"n_failed": int(df[fail_mask]["fit_failed"].fillna(False).sum())}
        for col in metric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values.astype(float)
            cs[key][col + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
            cs[key][col + "_se"]   = (float(vals.std(ddof=1) / np.sqrt(len(vals)))
                                      if len(vals) > 1 else float("nan"))
        shd_m  = cs[key].get("shd_icalingam_mean", float("nan"))
        shd_se = cs[key].get("shd_icalingam_se",   float("nan"))
        tc_m   = cs[key].get("oracle_total_correlation_mean", float("nan"))
        tc_se2 = cs[key].get("oracle_total_correlation_se",   float("nan"))
        print(f"  δ={dv:.1f} r={ir:2.0f}:  "
              f"SHD={shd_m:.2f}±{shd_se:.2f}  TC={tc_m:.3f}±{tc_se2:.3f}")
    stats["condition_summary"] = cs

    # ── TC calibration by δ (single_node, r=1, balanced) ────────────────────
    sn_r1 = ok_sn[ok_sn["imbalance_ratio"] == 1.0].copy()
    tc_calib = {}
    for dv, grp in sn_r1.groupby("delta"):
        tc_vals = grp["oracle_total_correlation"].dropna().values.astype(float)
        tc_calib[float(dv)] = {
            "tc_mean": float(tc_vals.mean()) if len(tc_vals) else float("nan"),
            "tc_se":   (float(tc_vals.std(ddof=1) / np.sqrt(len(tc_vals)))
                        if len(tc_vals) > 1 else float("nan")),
        }
        print(f"  TC calibration:  δ={dv:.1f}  TC={tc_calib[float(dv)]['tc_mean']:.4f}")
    stats["tc_calibration"] = tc_calib

    # ── ΔSHD threshold analysis ───────────────────────────────────────────────
    # ΔSHD = SHD_single(r=1) − SHD_bernoulli, paired by (seed, δ)
    ok_bern = ok[ok["arm"] == "bernoulli"].copy()
    delta_shd_stats = {}

    if len(ok_bern) > 0 and len(sn_r1) > 0:
        merged = sn_r1[["seed", "delta", "shd_icalingam"]].merge(
            ok_bern[["seed", "delta", "shd_icalingam"]],
            on=["seed", "delta"], suffixes=("_sn", "_bern")
        )
        merged["delta_shd"] = (merged["shd_icalingam_sn"].astype(float) -
                                merged["shd_icalingam_bern"].astype(float))
        merged = merged.dropna(subset=["delta_shd"])

        print("\n  ΔSHD = SHD_single(r=1) − SHD_bernoulli by δ:")
        for dv, grp in merged.groupby("delta"):
            dshd = grp["delta_shd"].values
            seeds_g = grp["seed"].values
            mean_d, lo, hi = _cluster_bootstrap_mean(dshd, seeds_g)
            harmful = "YES" if lo > 0 else ("borderline" if hi > 0 else "no")
            delta_shd_stats[float(dv)] = {
                "mean_delta_shd": float(mean_d),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_pairs": int(len(dshd)),
                "harmful_95pct": harmful,
            }
            print(f"    δ={dv:.1f}:  ΔSHD={mean_d:+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  harmful={harmful}")

    stats["delta_shd_threshold"] = delta_shd_stats

    return stats


def _summary_table(stats: dict) -> str:
    """Return formatted markdown condition-summary table (single_node arm, δ × r)."""
    cs = stats.get("condition_summary", {})
    hdr = (f"| {'δ':>5} | {'r':>4} | {'SHD':>10} | {'Oracle TC':>12} | "
           f"{'Oracle MI':>12} | {'Fit-res MI':>12} | "
           f"{'|kurt|':>10} | {'n_fail':>6} |")
    sep = "|" + "|".join(["-" * (len(h) - 2) + "--"
                           for h in hdr.split("|")[1:-1]]) + "|"
    rows = [hdr, sep]

    def _f(c, k):
        m = c.get(k + "_mean", float("nan"))
        s = c.get(k + "_se",   float("nan"))
        return f"{m:.2f}±{s:.2f}" if not (np.isnan(m) or np.isnan(s)) else "  N/A  "

    for (dv, ir) in sorted(cs.keys()):
        c = cs[(dv, ir)]
        rows.append(
            f"| {dv:5.1f} | {ir:4.0f} | {_f(c, 'shd_icalingam'):>10} | "
            f"{_f(c, 'oracle_total_correlation'):>12} | "
            f"{_f(c, 'oracle_mean_pairwise_mi'):>12} | "
            f"{_f(c, 'fitted_residual_mean_pairwise_mi'):>12} | "
            f"{_f(c, 'oracle_mean_abs_excess_kurtosis'):>10} | "
            f"{c.get('n_failed', 0):>6} |"
        )
    return "\n".join(rows)


def _delta_shd_table(stats: dict) -> str:
    """Return ΔSHD threshold table: SHD_single(r=1) − SHD_bern, by δ."""
    dss = stats.get("delta_shd_threshold", {})
    tc_cal = stats.get("tc_calibration", {})
    if not dss:
        return "(No Bernoulli arm data available.)"
    hdr = (f"| {'δ':>5} | {'Oracle TC':>11} | "
           f"{'SHD_sn':>8} | {'SHD_bern':>10} | "
           f"{'ΔSHD':>8} | {'95% CI':>16} | {'harmful?':>10} |")
    sep = "|" + "|".join(["-" * (len(h) - 2) + "--"
                           for h in hdr.split("|")[1:-1]]) + "|"
    rows = [hdr, sep]
    cs = stats.get("condition_summary", {})
    for dv in sorted(dss.keys()):
        d = dss[dv]
        tc_info = tc_cal.get(dv, {})
        tc_str = (f"{tc_info['tc_mean']:.3f}±{tc_info['tc_se']:.3f}"
                  if tc_info else "N/A")
        # SHD values from condition summary at r=1
        key_sn = (dv, 1.0)
        shd_sn = cs.get(key_sn, {}).get("shd_icalingam_mean", float("nan"))
        shd_sn_se = cs.get(key_sn, {}).get("shd_icalingam_se", float("nan"))
        shd_sn_str = f"{shd_sn:.2f}±{shd_sn_se:.2f}" \
                     if not (np.isnan(shd_sn) or np.isnan(shd_sn_se)) else "N/A"
        shd_bern_approx = shd_sn - d["mean_delta_shd"]
        rows.append(
            f"| {dv:5.1f} | {tc_str:>11} | "
            f"{shd_sn_str:>8} | {shd_bern_approx:>10.2f} | "
            f"{d['mean_delta_shd']:>+8.3f} | "
            f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | "
            f"{d['harmful_95pct']:>10} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Rebuttal figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, stats: dict, output_dir: str) -> None:
    """Three-panel PDF + PNG figure."""
    os.makedirs(output_dir, exist_ok=True)

    # Patch legacy rows
    if "arm" not in df.columns:
        df = df.copy()
        df["arm"] = "single_node"
    if "delta" not in df.columns:
        df = df.copy()
        df["delta"] = 1.0

    ok = df[~df["fit_failed"].fillna(False)].copy()
    for c in ["shd_icalingam", "oracle_total_correlation",
              "fitted_residual_mean_pairwise_mi", "delta"]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    ok_sn   = ok[ok["arm"] == "single_node"].copy()
    ok_bern = ok[ok["arm"] == "bernoulli"].copy()

    # Color by δ level, marker by imbalance ratio (single_node arm)
    delta_palette = {
        1.0: "#2196F3", 2.0: "#FF9800",
        5.0: "#4CAF50", 10.0: "#F44336",
    }
    ir_markers = {1: "o", 2: "s", 5: "^", 10: "D"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel A: oracle TC vs SHD (single_node arm, colour=δ, marker=r) ─────
    ax = axes[0]
    for dv in sorted(ok_sn["delta"].dropna().unique()):
        sub_d = ok_sn[ok_sn["delta"] == dv]
        for ir in IMBALANCE_RATIOS:
            sub = sub_d[sub_d["imbalance_ratio"] == ir]
            if len(sub) == 0:
                continue
            clr = delta_palette.get(float(dv), "#888888")
            mkr = ir_markers.get(ir, "o")
            label = f"δ={dv:.0f}, r={ir}" if ir == IMBALANCE_RATIOS[0] else None
            ax.scatter(sub["oracle_total_correlation"], sub["shd_icalingam"],
                       color=clr, marker=mkr, alpha=0.4, s=22, label=label)

    # Binned mean ± SE across all single_node conditions
    tc  = ok_sn["oracle_total_correlation"].values.astype(float)
    shd = ok_sn["shd_icalingam"].values.astype(float)
    valid = ~(np.isnan(tc) | np.isnan(shd))
    if valid.sum() > 10:
        q_edges = np.unique(np.percentile(tc[valid], np.linspace(0, 100, 8)))
        bids = np.digitize(tc[valid], q_edges)
        bx, by, be = [], [], []
        for bid in np.unique(bids):
            m = bids == bid
            if m.sum() < 2:
                continue
            bx.append(tc[valid][m].mean())
            by.append(shd[valid][m].mean())
            be.append(shd[valid][m].std(ddof=1) / np.sqrt(m.sum()))
        ax.errorbar(bx, by, yerr=be, fmt="k-o", lw=2, ms=7,
                    zorder=5, capsize=4, label="Binned mean ± SE")

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(shd[valid], tc[valid], frac=0.5)
        ax.plot(sm[:, 0], sm[:, 1], "k--", lw=1.5, alpha=0.55, label="LOWESS")
    except ImportError:
        pass

    ax.set_xlabel("Oracle total correlation (nats)", fontsize=10)
    ax.set_ylabel("ICA-LiNGAM SHD ↓", fontsize=10)
    ax.set_title("A: Oracle TC vs SHD (single-node arm)", fontsize=11)
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.97, "Colour = δ level; marker = r",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.9))

    # ── Panel B: ΔSHD vs δ (error bars = 95% bootstrap CI) ──────────────────
    ax = axes[1]
    dss = stats.get("delta_shd_threshold", {})
    if dss:
        delta_pts = sorted(dss.keys())
        means = [dss[d]["mean_delta_shd"] for d in delta_pts]
        lows  = [dss[d]["mean_delta_shd"] - dss[d]["ci_lo"] for d in delta_pts]
        highs = [dss[d]["ci_hi"] - dss[d]["mean_delta_shd"] for d in delta_pts]
        colors = [delta_palette.get(d, "#888888") for d in delta_pts]
        ax.errorbar(delta_pts, means,
                    yerr=[lows, highs],
                    fmt="none", ecolor="gray", elinewidth=1.5, capsize=5, zorder=2)
        ax.scatter(delta_pts, means, c=colors, s=80, zorder=3)
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
        # Shade harmful region (above 0)
        ax.fill_between(delta_pts,
                        [dss[d]["ci_lo"] for d in delta_pts],
                        [dss[d]["ci_hi"] for d in delta_pts],
                        alpha=0.12, color="red", label="95% CI band")
        # Mark first harmful δ
        first_harmful = next(
            (d for d in delta_pts if dss[d]["harmful_95pct"] == "YES"), None
        )
        if first_harmful is not None:
            ax.axvline(first_harmful, color="red", lw=1, ls=":",
                       label=f"First harmful δ = {first_harmful:.0f}")
        ax.set_xlabel("δ (shift magnitude)", fontsize=10)
        ax.set_ylabel("ΔSHD = SHD_single(r=1) − SHD_Bernoulli", fontsize=10)
        ax.set_title("B: When does dependence become harmful?", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.97,
                "ΔSHD > 0 ⟹ single-node worse than Bernoulli.\n"
                "Error bars = 95% cluster-bootstrap CI.",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No Bernoulli arm data.\nRe-run with Bernoulli arm.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B: ΔSHD vs δ (no data)", fontsize=11)

    # ── Panel C: TC calibration curve vs δ ───────────────────────────────────
    ax = axes[2]
    tc_cal = stats.get("tc_calibration", {})
    if tc_cal:
        delta_pts_c = sorted(tc_cal.keys())
        tc_means = [tc_cal[d]["tc_mean"] for d in delta_pts_c]
        tc_ses   = [tc_cal[d]["tc_se"]   for d in delta_pts_c]
        colors_c = [delta_palette.get(d, "#888888") for d in delta_pts_c]
        ax.errorbar(delta_pts_c, tc_means, yerr=tc_ses,
                    fmt="-", color="steelblue", lw=2, capsize=5,
                    label="Oracle TC (r=1 balanced)")
        ax.scatter(delta_pts_c, tc_means, c=colors_c, s=70, zorder=4)
        tc_max_balanced = float((N_NODES - 1) * np.log(N_NODES / (N_NODES - 1)))
        ax.axhline(tc_max_balanced, color="black", lw=1.2, ls="--", alpha=0.7,
                   label=f"Theoretical max (d−1)·log(d/(d−1))={tc_max_balanced:.3f} nats")
        ax.set_xlabel("δ (shift magnitude)", fontsize=10)
        ax.set_ylabel("Oracle total correlation (nats)", fontsize=10)
        ax.set_title("C: TC calibration vs δ", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.97,
                "TC → log(d) as δ → ∞ (balanced r=1).\n"
                "Colour = δ level.",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No TC calibration data.", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("C: TC calibration (no data)", fontsize=11)

    fig.suptitle(
        "Rebuttal: When does intervention-induced source dependence harm ICA-LiNGAM?\n"
        f"Single-node vs Bernoulli, δ sweep {DELTA_VALUES}, "
        f"r ∈ {IMBALANCE_RATIOS}, {N_NODES}-node ER DAGs, 100 seeds",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"residual_dependence_vs_shd.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Git hash
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Residual dependence vs SHD ablation — extended with δ sweep and "
            "Bernoulli arm to answer 'when does dependence become harmful?'."
        )
    )
    parser.add_argument("--results-dir",      default="results",
                        help="Root directory for checkpoints and output files.")
    parser.add_argument("--output-dir",       default="figures",
                        help="Directory for figures.")
    parser.add_argument("--n-jobs",           type=int, default=N_JOBS_DEFAULT)
    parser.add_argument("--seeds",            type=int, nargs="+", default=SEEDS)
    parser.add_argument("--delta-values",     type=float, nargs="+",
                        default=DELTA_VALUES)
    parser.add_argument("--imbalance-ratios", type=float, nargs="+",
                        default=IMBALANCE_RATIOS)
    parser.add_argument("--no-bernoulli",     action="store_true",
                        help="Skip Bernoulli arm (single_node only).")
    parser.add_argument("--resume",           action="store_true", default=True)
    parser.add_argument("--quick",            action="store_true",
                        help="Smoke test: 2 seeds × 2 δ × 2 r.")
    args = parser.parse_args()

    seeds        = [0, 1]         if args.quick else args.seeds
    delta_values = [1.0, 5.0]    if args.quick else args.delta_values
    irs          = [1, 10]        if args.quick else args.imbalance_ratios
    run_bern     = not args.no_bernoulli

    results_dir = os.path.join(args.results_dir,
                               "residual_dependence_shd_ablation")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Startup checks ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  ResidualDependenceSHDAblation  (δ-sweep + Bernoulli arm extension)")
    print("=" * 64)
    _print_resolved_params()
    print("\nVerifying adjacency convention ...")
    _verify_convention()
    print("\nValidating TC estimator ...")
    _validate_tc()

    # ── Build task list ──────────────────────────────────────────────────────
    tasks = []
    for s in seeds:
        for dv in delta_values:
            for ir in irs:
                tasks.append((s, dv, ir, "single_node"))
            if run_bern:
                tasks.append((s, dv, 1.0, "bernoulli"))

    n_sn   = sum(1 for t in tasks if t[3] == "single_node")
    n_bern = sum(1 for t in tasks if t[3] == "bernoulli")
    total  = len(tasks)
    print(f"\nRunning {total} trials  "
          f"({len(seeds)} seeds × {len(delta_values)} δ levels × "
          f"{len(irs)} r levels = {n_sn} single_node  +  "
          f"{n_bern} bernoulli)  "
          f"n_jobs={args.n_jobs}  resume={args.resume}")

    def _worker(s, dv, ir, arm):
        try:
            return run_trial(s, dv, ir, arm, results_dir, resume=args.resume)
        except Exception as exc:
            return {
                "seed": s, "delta": dv, "imbalance_ratio": ir, "arm": arm,
                "fit_failed": True, "failure_message": repr(exc),
                "shd_icalingam": float("nan"),
            }

    rows = Parallel(n_jobs=args.n_jobs, return_as="list")(
        delayed(_worker)(s, dv, ir, arm) for s, dv, ir, arm in tasks
    )
    print(f"  Completed {len(rows)}/{total}")

    df = pd.DataFrame(rows)
    n_fail = int(df["fit_failed"].fillna(False).sum())
    print(f"  ICA-LiNGAM failures: {n_fail}/{total}")

    # ── Statistics ───────────────────────────────────────────────────────────
    print("\nComputing statistics ...")
    stats = compute_statistics(df)

    table_str      = _summary_table(stats)
    threshold_str  = _delta_shd_table(stats)
    print("\nCondition summary (single_node arm):")
    print(table_str)
    print("\nΔSHD threshold analysis:")
    print(threshold_str)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    make_figure(df, stats, args.output_dir)

    # ── Save results ─────────────────────────────────────────────────────────
    csv_p  = os.path.join(results_dir, "residual_dependence_shd_ablation.csv")
    pkl_p  = os.path.join(results_dir, "residual_dependence_shd_ablation.pkl")
    meta_p = os.path.join(results_dir, "residual_dependence_shd_metadata.json")

    df.to_csv(csv_p, index=False)
    df.to_pickle(pkl_p)

    metadata = {
        "description":            (
            "Residual-dependence vs SHD ablation — extended with δ sweep and "
            "Bernoulli arm to answer 'when does dependence become harmful?'"
        ),
        "timestamp":              datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash":               _git_hash(),
        "python_version":         sys.version,
        "training_sample_size":   TRAINING_SAMPLE_SIZE,
        "diagnostic_sample_size": DIAGNOSTIC_SAMPLE_SIZE,
        "delta_values":           delta_values,
        "n_observational":        N_OBSERVATIONAL,
        "n_nodes":                N_NODES,
        "n_edges":                N_EDGES,
        "weight_range":           WEIGHT_RANGE,
        "seeds":                  seeds,
        "imbalance_ratios":       irs,
        "run_bernoulli_arm":      run_bern,
        "include_no_intervention": INCLUDE_NO_INTERVENTION,
        "n_ksg_neighbors":        N_KSG_NEIGHBORS,
        "n_bootstrap":            N_BOOTSTRAP,
        "bootstrap_seed":         BOOTSTRAP_SEED,
        "diag_seed_offset":       DIAG_SEED_OFFSET,
        "statistics":             stats,
        "summary_table":          table_str,
        "threshold_table":        threshold_str,
    }
    # Convert tuple keys in condition_summary for JSON serialization
    meta_stats = stats.copy()
    if "condition_summary" in meta_stats:
        meta_stats["condition_summary"] = {
            f"d{k[0]:.1f}_r{k[1]:.0f}": v
            for k, v in meta_stats["condition_summary"].items()
        }
    metadata["statistics"] = meta_stats

    with open(meta_p, "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"\nResults saved to {results_dir}/")
    print(f"  {os.path.basename(csv_p)}")
    print(f"  {os.path.basename(pkl_p)}")
    print(f"  {os.path.basename(meta_p)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
