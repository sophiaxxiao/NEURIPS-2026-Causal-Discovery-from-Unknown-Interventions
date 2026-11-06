#!/usr/bin/env python3
# coding=utf-8
"""
Rebuttal ablation experiment runner.
"Causal Discovery from Unseen Environments" — NeurIPS 2026.

Usage
-----
Smoke test (2 seeds, 200 samples, ICA-LiNGAM only):
    python experiments/run_rebuttal_ablations.py \\
        --config configs/rebuttal_ablations.yaml \\
        --quick \\
        --results-dir results/rebuttal_smoke \\
        --n-jobs 1

Full experiment:
    python experiments/run_rebuttal_ablations.py \\
        --config configs/rebuttal_ablations.yaml \\
        --results-dir /path/to/scratch/rebuttal_ablations \\
        --n-jobs 8 \\
        --resume

Aggregation (run after the experiment):
    python analysis/aggregate_rebuttal_ablations.py \\
        --results-dir /path/to/scratch/rebuttal_ablations

See README_REBUTTAL_ABLATIONS.md for full documentation.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml
from joblib import Parallel, delayed

# ---- path setup -------------------------------------------------------
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYNTH_DIR = os.path.dirname(_EXPERIMENTS_DIR)
_SRC_DIR = os.path.join(_SYNTH_DIR, "src")
_ABLATION_DIR = os.path.join(_SYNTH_DIR, "ablation")
for _p in [_SRC_DIR, _ABLATION_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from castle.algorithms import DirectLiNGAM, ICALiNGAM
from castle.datasets import DAG
from castle.metrics import MetricsDAG

from ablation_simulator import (
    AblationShiftSimulation,
    compute_delta_stats,
    compute_environment_size_stats,
    compute_p_intervention_stats,
    compute_source_diagnostics,
    heterogeneous_delta_vector_from_z,
    heterogeneous_p_vector_from_z,
    random_derangement,
)
from helpers import compute_graph_metrics
from method_wrappers import (
    build_env_labels,
    run_gies,
    run_igsp,
    run_jci_pc,
    run_ut_igsp,
    run_ut_igsp_oracle,
    split_subsample,
)


# -----------------------------------------------------------------------
# Condition dataclass
# -----------------------------------------------------------------------

@dataclass
class AblationCondition:
    """One experimental condition: a single row of the design matrix."""
    condition_name: str
    ablation_name: str       # 'baseline' | 'unequal_sizes' | 'heterogeneous_strengths'
                             # | 'weak_interventions' | 'off_target'
    severity_name: str       # human-readable parameter name
    severity_value: float
    n_nodes: int = 10
    n_edges: int = 10
    n_observational: int = 100
    n_interventional: int = 2000
    p_intervention: float = 0.5   # scalar; overridden by p_vector for Ablation A
    delta_scalar: float = 1.0
    h_unequal: float = 0.0        # Ablation A severity
    tau_hetero: float = 0.0       # Ablation B severity
    off_target_rho: float = 0.0   # Ablation D severity
    methods: List[str] = field(default_factory=lambda: ["icalingam", "dlingam", "utigsp", "igsp", "gies"])
    alpha_ci: float = 0.05
    alpha_inv: float = 0.05


# -----------------------------------------------------------------------
# Build condition list
# -----------------------------------------------------------------------

def build_conditions(cfg: dict) -> List[AblationCondition]:
    """Construct the full ordered condition list from a config dict.

    The baseline appears first. Non-baseline conditions are ordered A, B, C, D.
    The baseline is deduplicated: it appears once and is reused for all ablations.
    """
    methods = cfg.get("methods", ["icalingam", "dlingam", "utigsp", "igsp", "gies"])
    baseline_delta = cfg.get("baseline_delta", 1.0)
    # Common parameters shared across all conditions (excluding per-ablation severities)
    common = dict(
        n_nodes=cfg.get("n_nodes", 10),
        n_edges=cfg.get("n_edges", 10),
        n_observational=cfg.get("n_observational", 100),
        n_interventional=cfg.get("n_interventional_total", 2000),
        p_intervention=cfg.get("baseline_p_intervention", 0.5),
        methods=methods,
        alpha_ci=cfg.get("alpha_ci", 0.05),
        alpha_inv=cfg.get("alpha_inv", 0.05),
    )

    conditions = []

    # Shared baseline
    conditions.append(AblationCondition(
        condition_name="baseline",
        ablation_name="baseline",
        severity_name="none",
        severity_value=0.0,
        delta_scalar=baseline_delta,
        h_unequal=0.0, tau_hetero=0.0, off_target_rho=0.0,
        **common,
    ))

    # Ablation A: unequal environment sizes
    for h in cfg.get("ablation_A", {}).get("values", [0.15, 0.30, 0.45]):
        if h == 0.0:
            continue  # baseline already included
        conditions.append(AblationCondition(
            condition_name=f"h={h:.2f}",
            ablation_name="unequal_sizes",
            severity_name="h_unequal",
            severity_value=float(h),
            delta_scalar=baseline_delta,
            h_unequal=float(h),
            tau_hetero=0.0, off_target_rho=0.0,
            **common,
        ))

    # Ablation B: heterogeneous strengths
    for tau in cfg.get("ablation_B", {}).get("values", [0.35, 0.70, 1.00]):
        if tau == 0.0:
            continue
        conditions.append(AblationCondition(
            condition_name=f"tau={tau:.2f}",
            ablation_name="heterogeneous_strengths",
            severity_name="tau_hetero",
            severity_value=float(tau),
            delta_scalar=baseline_delta,
            tau_hetero=float(tau),
            h_unequal=0.0, off_target_rho=0.0,
            **common,
        ))

    # Ablation C: weak interventions (delta varies, others baseline)
    for delta in cfg.get("ablation_C", {}).get("values", [0.75, 0.50, 0.25, 0.10]):
        if float(delta) == float(baseline_delta):
            continue  # already covered by baseline
        conditions.append(AblationCondition(
            condition_name=f"delta={delta:.2f}",
            ablation_name="weak_interventions",
            severity_name="delta_scalar",
            severity_value=float(delta),
            delta_scalar=float(delta),
            h_unequal=0.0, tau_hetero=0.0, off_target_rho=0.0,
            **common,
        ))

    # Ablation D: correlated off-target effects
    for rho in cfg.get("ablation_D", {}).get("values", [0.10, 0.25, 0.50, 1.00]):
        if rho == 0.0:
            continue
        conditions.append(AblationCondition(
            condition_name=f"rho={rho:.2f}",
            ablation_name="off_target",
            severity_name="off_target_rho",
            severity_value=float(rho),
            delta_scalar=baseline_delta,
            off_target_rho=float(rho),
            h_unequal=0.0, tau_hetero=0.0,
            **common,
        ))

    return conditions


# -----------------------------------------------------------------------
# Per-condition per-seed simulation
# -----------------------------------------------------------------------

def _build_sim(cond: AblationCondition, true_dag: np.ndarray, seed: int) -> AblationShiftSimulation:
    """Instantiate and run the correct ablation simulator for this condition."""
    rng = np.random.default_rng(seed)  # explicit RNG for ablation-specific draws

    # ---- Ablation A: per-node p-vector ----
    p_vector = None
    if cond.ablation_name == "unequal_sizes" and cond.h_unequal > 0:
        z = np.linspace(-1.0, 1.0, cond.n_nodes)
        perm = rng.permutation(cond.n_nodes)
        z_perm = z[perm]
        p_vector = heterogeneous_p_vector_from_z(z_perm, cond.h_unequal)

    # ---- Ablation B: per-node delta-vector ----
    delta_vector = None
    if cond.ablation_name == "heterogeneous_strengths" and cond.tau_hetero > 0:
        z = np.linspace(-1.0, 1.0, cond.n_nodes)
        perm = rng.permutation(cond.n_nodes)
        z_perm = z[perm]
        delta_vector = heterogeneous_delta_vector_from_z(z_perm, cond.tau_hetero)

    # ---- Ablation D: derangement ----
    derangement = None
    if cond.ablation_name == "off_target" and cond.off_target_rho > 0:
        derangement = random_derangement(cond.n_nodes, rng)

    sim = AblationShiftSimulation(
        n_nodes=cond.n_nodes,
        n_edges=cond.n_edges,
        n_samples=cond.n_interventional,
        n_observational=cond.n_observational,
        weighted_random_dag=true_dag,
        p_intervention=cond.p_intervention,
        delta=cond.delta_scalar,
        seed=seed,
        intervention_mode="bernoulli",
        p_vector=p_vector,
        delta_vector=delta_vector,
        off_target_rho=cond.off_target_rho,
        derangement=derangement,
    )
    sim.generate_ablation_datasets()
    return sim


def _safe_shd(adj, true_dag):
    if adj is None:
        return float("nan")
    return float(MetricsDAG(adj, true_dag).metrics["shd"])


def _safe_metrics(adj, true_dag):
    if adj is None:
        return None
    return compute_graph_metrics(adj, true_dag)


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _pkg_versions():
    pkgs = ["numpy", "scipy", "joblib", "castle", "gcastle", "causaldag", "gies"]
    versions = {}
    for p in pkgs:
        try:
            import importlib.metadata
            versions[p] = importlib.metadata.version(p)
        except Exception:
            versions[p] = "unknown"
    return versions


# -----------------------------------------------------------------------
# Core trial function (one condition × one seed)
# -----------------------------------------------------------------------

def run_trial(cond: AblationCondition, seed: int, git_hash: str) -> dict:
    """Run one (condition, seed) unit of work.

    Returns a flat dict suitable for appending to a result table.
    Never raises: exceptions are captured in the 'failed' / 'exception_*' fields.
    """
    ts_start = datetime.datetime.utcnow().isoformat()
    row_base = dict(
        seed=seed,
        condition_name=cond.condition_name,
        ablation_name=cond.ablation_name,
        severity_name=cond.severity_name,
        severity_value=cond.severity_value,
        n_nodes=cond.n_nodes,
        n_edges=cond.n_edges,
        n_observational=cond.n_observational,
        n_interventional=cond.n_interventional,
        off_target_rho=cond.off_target_rho,
        git_hash=git_hash,
        timestamp=ts_start,
    )

    # -- 1. Generate DAG (same across conditions for this seed) --
    try:
        true_dag = DAG.erdos_renyi(
            n_nodes=cond.n_nodes, n_edges=cond.n_edges,
            seed=seed, weight_range=[0.5, 2.0],
        )
    except Exception as exc:
        row_base.update(dict(
            failed=True, exception_type=type(exc).__name__,
            exception_message=str(exc)[:500],
        ))
        return [_fill_nan_row(row_base, m) for m in cond.methods]

    # -- 2. Simulate --
    try:
        sim = _build_sim(cond, true_dag, seed)
    except Exception as exc:
        row_base.update(dict(
            failed=True, exception_type=type(exc).__name__,
            exception_message=str(exc)[:500],
        ))
        return [_fill_nan_row(row_base, m) for m in cond.methods]

    # -- 3. Extract pooled data --
    interv_ds = sim.datasets_interventional
    obs_data = sim.dataset_observational.X
    all_data = np.vstack([interv_ds[k]["X"] for k in interv_ds])
    env_labels = build_env_labels(sim)

    # Diagnostics (computed on exogenous sources)
    diag = compute_source_diagnostics(all_data, true_dag)
    env_stats = compute_environment_size_stats(interv_ds)

    p_stat = compute_p_intervention_stats(
        sim.p_vector if sim.p_vector is not None else sim.p_intervention
    )
    delta_stat = compute_delta_stats(sim.delta_per_node)

    row_base.update({**diag, **env_stats, **p_stat, **delta_stat})

    # -- 4. Build splits for environment-aware methods --
    # For Ablation D: use actual targets (includes off-target nodes).
    actual_tgts = getattr(sim, "actual_targets", None)

    obs_sub, ivs_list, known_targets = split_subsample(all_data, env_labels, sim)

    # If off-target is active, give environment-aware methods the actual targets.
    if actual_tgts is not None and cond.off_target_rho > 0:
        known_targets_actual = []
        patterns = list(interv_ds.keys())
        for i, pat in enumerate(patterns):
            rows = all_data[env_labels == i]
            if rows.shape[0] == 0 or not any(pat):
                continue
            known_targets_actual.append(actual_tgts[pat])
        # Re-build ivs_list in the same pattern order
        ivs_list_rebuilt, known_actual_rebuilt = [], []
        for i, pat in enumerate(patterns):
            rows = all_data[env_labels == i]
            if rows.shape[0] == 0 or not any(pat):
                continue
            ivs_list_rebuilt.append(rows)
            known_actual_rebuilt.append(actual_tgts[pat])
        ivs_list = ivs_list_rebuilt
        known_targets = known_actual_rebuilt

    # Subsample = obs prepended (LiNGAM sees this full matrix)
    subsample = np.vstack([obs_data, all_data])

    rows_out = []
    for method in cond.methods:
        row = dict(row_base)
        row["method"] = method
        t0 = time.perf_counter()
        try:
            adj = _run_method(
                method, subsample, obs_sub, ivs_list, known_targets,
                cond.alpha_ci, cond.alpha_inv,
            )
            row["runtime_seconds"] = time.perf_counter() - t0
            row["failed"] = False
            row["exception_type"] = ""
            row["exception_message"] = ""
            metrics = _safe_metrics(adj, true_dag)
            if metrics is not None:
                row["shd"] = float(metrics["shd"])
                row["directed_f1"] = float(metrics["F1"]) if not np.isnan(metrics["F1"]) else float("nan")
                row["directed_precision"] = float(metrics["precision"]) if not np.isnan(metrics["precision"]) else float("nan")
                row["directed_recall"] = float(metrics["recall"]) if not np.isnan(metrics["recall"]) else float("nan")
            else:
                row["shd"] = float("nan")
                row["directed_f1"] = float("nan")
                row["directed_precision"] = float("nan")
                row["directed_recall"] = float("nan")
        except Exception as exc:
            row["runtime_seconds"] = time.perf_counter() - t0
            row["failed"] = True
            row["exception_type"] = type(exc).__name__
            row["exception_message"] = traceback.format_exc()[-500:]
            row["shd"] = float("nan")
            row["directed_f1"] = float("nan")
            row["directed_precision"] = float("nan")
            row["directed_recall"] = float("nan")
        rows_out.append(row)

    return rows_out


def _fill_nan_row(base: dict, method: str) -> dict:
    r = dict(base)
    r["method"] = method
    for col in ["shd", "directed_f1", "directed_precision", "directed_recall",
                "runtime_seconds", "p_intervention_mean", "p_intervention_std",
                "environment_size_min", "environment_size_max", "environment_size_ratio",
                "n_environments", "delta_min", "delta_max", "delta_mean", "delta_rms",
                "delta_std", "mean_abs_excess_kurtosis", "min_abs_excess_kurtosis",
                "mean_abs_offdiag_source_correlation"]:
        r.setdefault(col, float("nan"))
    r.setdefault("failed", True)
    r.setdefault("exception_type", "SimulationError")
    r.setdefault("exception_message", "Simulation failed before method dispatch.")
    return r


def _run_method(method, subsample, obs_sub, ivs_list, known_targets,
                alpha_ci=0.05, alpha_inv=0.05):
    """Dispatch to the appropriate method wrapper. Returns adjacency or None."""
    if method == "icalingam":
        m = ICALiNGAM()
        m.learn(subsample)
        return m.causal_matrix
    elif method == "dlingam":
        m = DirectLiNGAM()
        m.learn(subsample)
        return m.causal_matrix
    elif method == "utigsp":
        return run_ut_igsp(obs_sub, ivs_list, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    elif method == "igsp":
        return run_igsp(obs_sub, ivs_list, known_targets, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    elif method == "gies":
        return run_gies(obs_sub, ivs_list, known_targets)
    elif method == "jcipc":
        return run_jci_pc(obs_sub, ivs_list, known_targets)
    elif method == "ut_igsp_oracle":
        return run_ut_igsp_oracle(obs_sub, ivs_list, known_targets, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    else:
        raise ValueError(f"Unknown method: {method!r}")


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------

def _checkpoint_path(results_dir: str, condition_name: str, seed: int) -> str:
    safe_cond = condition_name.replace("=", "eq").replace(".", "p")
    return os.path.join(results_dir, f"ckpt_{safe_cond}_s{seed:04d}.json")


def _checkpoint_exists(results_dir: str, condition_name: str, seed: int) -> bool:
    return os.path.exists(_checkpoint_path(results_dir, condition_name, seed))


def _write_checkpoint(results_dir: str, condition_name: str, seed: int, rows: list):
    path = _checkpoint_path(results_dir, condition_name, seed)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, allow_nan=True)
    os.replace(tmp, path)


def _load_all_checkpoints(results_dir: str) -> list:
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("ckpt_") and fname.endswith(".json"):
            try:
                with open(os.path.join(results_dir, fname)) as f:
                    rows.extend(json.load(f))
            except Exception:
                pass
    return rows


# -----------------------------------------------------------------------
# Joblib worker wrapper
# -----------------------------------------------------------------------

def _worker(cond_dict: dict, seed: int, results_dir: str, git_hash: str,
            overwrite: bool) -> str:
    """Worker: run one (condition, seed), write checkpoint, return status string."""
    cond = AblationCondition(**cond_dict)
    if not overwrite and _checkpoint_exists(results_dir, cond.condition_name, seed):
        return f"SKIP  {cond.condition_name} seed={seed}"
    rows = run_trial(cond, seed, git_hash)
    _write_checkpoint(results_dir, cond.condition_name, seed, rows)
    n_failed = sum(1 for r in rows if r.get("failed", False))
    return f"DONE  {cond.condition_name} seed={seed} ({n_failed}/{len(rows)} failed)"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run rebuttal ablation experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--results-dir", default="results/rebuttal_ablations",
                        help="Directory for checkpoint files and results.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers (default: from config).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override method list.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 2 seeds, 200 samples, ICA-LiNGAM, n_jobs=1.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (condition, seed) pairs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing checkpoints.")
    args = parser.parse_args()

    # -- Load config --
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # -- Apply quick overrides --
    if args.quick:
        qcfg = cfg.get("quick", {})
        cfg["seeds_full"] = qcfg.get("seeds", [0, 1])
        cfg["n_interventional_total"] = qcfg.get("n_interventional_total", 200)
        cfg["methods"] = qcfg.get("methods", ["icalingam"])
        args.n_jobs = qcfg.get("n_jobs", 1)
        print("[QUICK MODE] 2 seeds, 200 samples, ICA-LiNGAM only.")

    # -- Resolve parameters --
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds_full", list(range(100)))
    if args.methods:
        cfg["methods"] = args.methods
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg.get("n_jobs", 8)
    overwrite = args.overwrite and not args.resume

    results_dir = args.results_dir
    # Support environment variable override
    results_dir = os.environ.get("ABLATION_RESULTS_DIR", results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # -- Build conditions --
    conditions = build_conditions(cfg)
    print(f"Conditions: {len(conditions)} | Seeds: {len(seeds)} | "
          f"Methods: {cfg.get('methods')} | n_jobs: {n_jobs}")
    for c in conditions:
        print(f"  {c.condition_name:25s}  ablation={c.ablation_name}")

    # -- Write experiment manifest --
    git_hash = _git_hash()
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "package_versions": _pkg_versions(),
        "platform": platform.platform(),
        "n_conditions": len(conditions),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "conditions": [asdict(c) for c in conditions],
        "config_path": os.path.abspath(args.config),
        "results_dir": os.path.abspath(results_dir),
        "quick_mode": args.quick,
        "n_jobs": n_jobs,
        "simulator_audit": (
            "Bernoulli mode: each sample draws per-node Bernoulli indicators; "
            "unique patterns become environments. n_samples=n_interventional (no subsampling). "
            "delta_per_node[i]=delta*sqrt(var(X_obs[:,i])). "
            "ERDagsComparison metadata bug: it records n_samples_pool=1e8 but uses 1e6; "
            "this script records actual n_interventional."
        ),
    }
    manifest_path = os.path.join(results_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=True)
    print(f"Manifest written to {manifest_path}")

    # -- Build work units --
    work = [
        (asdict(cond), seed)
        for cond in conditions
        for seed in seeds
    ]
    total = len(work)
    print(f"Total work units: {total}")

    # Skip already-completed units in resume mode
    if args.resume:
        pending = [
            (cd, s) for cd, s in work
            if not _checkpoint_exists(results_dir, cd["condition_name"], s)
        ]
        print(f"Resume mode: {total - len(pending)} already done, {len(pending)} pending.")
        work = pending

    if not work:
        print("Nothing to do. All units complete.")
    else:
        # -- Run in parallel --
        status_iter = Parallel(n_jobs=n_jobs, return_as="generator", verbose=0)(
            delayed(_worker)(cd, s, results_dir, git_hash, overwrite)
            for cd, s in work
        )
        for i, status in enumerate(status_iter, 1):
            print(f"  [{i}/{len(work)}] {status}")

    # -- Collect all results --
    all_rows = _load_all_checkpoints(results_dir)
    print(f"\nTotal result rows collected: {len(all_rows)}")

    # Save combined JSON (for quick inspection; Parquet is produced by aggregate script)
    combined_path = os.path.join(results_dir, "results_combined.json")
    tmp = combined_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_rows, f, allow_nan=True)
    os.replace(tmp, combined_path)
    print(f"Combined results written to {combined_path}")
    print(f"\nNext step: python analysis/aggregate_rebuttal_ablations.py --results-dir {results_dir}")


if __name__ == "__main__":
    main()
