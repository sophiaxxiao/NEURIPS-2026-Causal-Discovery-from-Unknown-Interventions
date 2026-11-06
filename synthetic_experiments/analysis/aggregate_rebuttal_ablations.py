#!/usr/bin/env python3
# coding=utf-8
"""
Aggregate raw checkpoint files from run_rebuttal_ablations.py into a
Parquet result table with paired-difference statistics.

Usage
-----
    python analysis/aggregate_rebuttal_ablations.py \\
        --results-dir results/rebuttal_ablations

Output
------
    <results-dir>/results.parquet         long-form table (one row per seed×condition×method)
    <results-dir>/summary.parquet         per-condition×method summary statistics
    <results-dir>/paired_diffs.parquet    paired differences vs. baseline per seed
    <results-dir>/aggregate_manifest.json metadata about this aggregation run
"""

import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all checkpoint JSON files and the combined JSON into a DataFrame."""
    rows = []

    # Prefer individual checkpoints (more robust to partial runs)
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("ckpt_") and fname.endswith(".json"):
            try:
                with open(os.path.join(results_dir, fname)) as f:
                    rows.extend(json.load(f))
            except Exception as e:
                print(f"  WARNING: could not load {fname}: {e}")

    if not rows:
        combined = os.path.join(results_dir, "results_combined.json")
        if os.path.exists(combined):
            with open(combined) as f:
                rows = json.load(f)

    if not rows:
        raise FileNotFoundError(f"No result data found in {results_dir!r}")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows from {results_dir!r}")
    return df


# -----------------------------------------------------------------------
# Clean and type-cast
# -----------------------------------------------------------------------

_FLOAT_COLS = [
    "severity_value", "p_intervention_mean", "p_intervention_std",
    "environment_size_ratio", "delta_min", "delta_max", "delta_mean",
    "delta_rms", "delta_std", "off_target_rho",
    "shd", "directed_f1", "directed_precision", "directed_recall",
    "runtime_seconds",
    "mean_abs_excess_kurtosis", "min_abs_excess_kurtosis",
    "mean_abs_offdiag_source_correlation",
]
_INT_COLS = [
    "seed", "n_nodes", "n_edges", "n_observational", "n_interventional",
    "environment_size_min", "environment_size_max", "n_environments",
]
_STR_COLS = [
    "condition_name", "ablation_name", "severity_name",
    "method", "git_hash", "timestamp", "exception_type", "exception_message",
]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in _STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")
    if "failed" in df.columns:
        df["failed"] = df["failed"].fillna(True).astype(bool)

    # Deduplicate: keep last entry for each (seed, condition_name, method)
    key = ["seed", "condition_name", "method"]
    existing_keys = [k for k in key if k in df.columns]
    if existing_keys:
        before = len(df)
        df = df.drop_duplicates(subset=existing_keys, keep="last").reset_index(drop=True)
        if len(df) < before:
            print(f"  Deduplicated {before - len(df)} duplicate rows.")
    return df


# -----------------------------------------------------------------------
# Paired differences
# -----------------------------------------------------------------------

def compute_paired_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seed paired differences vs the baseline condition.

    Returns a DataFrame with columns:
        seed, condition_name, ablation_name, severity_value, method,
        shd_diff, f1_diff
    """
    baseline = df[df["ablation_name"] == "baseline"][["seed", "method", "shd", "directed_f1"]].copy()
    baseline = baseline.rename(columns={"shd": "shd_baseline", "directed_f1": "f1_baseline"})

    non_base = df[df["ablation_name"] != "baseline"].copy()
    merged = non_base.merge(baseline, on=["seed", "method"], how="left")
    merged["shd_diff"] = merged["shd"] - merged["shd_baseline"]
    merged["f1_diff"] = merged["directed_f1"] - merged["f1_baseline"]
    return merged[[
        "seed", "condition_name", "ablation_name", "severity_name",
        "severity_value", "method", "shd", "directed_f1",
        "shd_diff", "f1_diff", "failed",
    ]]


# -----------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition × method summary with mean, SE, median, IQR, failure rate."""
    records = []
    group_cols = ["condition_name", "ablation_name", "severity_name", "severity_value", "method"]
    for keys, grp in df.groupby(group_cols, dropna=False):
        cond_name, ablation_name, severity_name, severity_value, method = keys
        valid = grp[~grp["failed"]]
        n_total = len(grp)
        n_failed = grp["failed"].sum()

        rec = {
            "condition_name": cond_name,
            "ablation_name": ablation_name,
            "severity_name": severity_name,
            "severity_value": float(severity_value) if pd.notna(severity_value) else float("nan"),
            "method": method,
            "n_seeds": n_total,
            "n_failed": int(n_failed),
            "failure_rate": float(n_failed / n_total) if n_total > 0 else float("nan"),
        }

        for metric in ["shd", "directed_f1", "directed_precision", "directed_recall"]:
            vals = valid[metric].dropna().values if metric in valid.columns else np.array([])
            rec[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            rec[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            rec[f"{metric}_median"] = float(np.median(vals)) if len(vals) > 0 else float("nan")
            q25, q75 = (np.percentile(vals, [25, 75]) if len(vals) > 0 else (float("nan"), float("nan")))
            rec[f"{metric}_iqr"] = float(q75 - q25)

        # Diagnostic summaries
        for diag in ["mean_abs_excess_kurtosis", "min_abs_excess_kurtosis",
                     "mean_abs_offdiag_source_correlation",
                     "environment_size_ratio", "delta_rms"]:
            vals = valid[diag].dropna().values if diag in valid.columns else np.array([])
            rec[f"{diag}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

        records.append(rec)

    return pd.DataFrame(records)


def compute_bootstrap_ci(df: pd.DataFrame, metric: str = "shd",
                         n_bootstrap: int = 2000, ci: float = 0.95,
                         rng_seed: int = 42) -> pd.DataFrame:
    """Paired bootstrap CIs for each (condition, method) by resampling seeds.

    Returns a DataFrame with columns:
        condition_name, method, ci_lower, ci_upper (for the mean metric)
    """
    rng = np.random.default_rng(rng_seed)
    records = []
    baseline_data = df[df["ablation_name"] == "baseline"].set_index(["seed", "method"])[metric]

    group_cols = ["condition_name", "ablation_name", "method"]
    for keys, grp in df[df["ablation_name"] != "baseline"].groupby(group_cols):
        cond_name, ablation_name, method = keys
        seeds = grp["seed"].unique()

        boot_means = []
        for _ in range(n_bootstrap):
            boot_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            diffs = []
            for s in boot_seeds:
                cval = grp[grp["seed"] == s][metric].values
                bval = baseline_data.get((s, method), np.nan)
                if len(cval) > 0 and not np.isnan(cval[0]) and not np.isnan(bval):
                    diffs.append(cval[0] - bval)
            if diffs:
                boot_means.append(np.mean(diffs))

        alpha = (1 - ci) / 2
        if len(boot_means) > 0:
            ci_lo = float(np.quantile(boot_means, alpha))
            ci_hi = float(np.quantile(boot_means, 1 - alpha))
        else:
            ci_lo = ci_hi = float("nan")

        records.append({
            "condition_name": cond_name,
            "ablation_name": ablation_name,
            "method": method,
            f"{metric}_paired_diff_ci_lower": ci_lo,
            f"{metric}_paired_diff_ci_upper": ci_hi,
        })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate rebuttal ablation results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    results_dir = args.results_dir
    print(f"Aggregating from {results_dir!r}")

    df = load_results(results_dir)
    df = clean_df(df)

    # Save long-form Parquet
    parquet_path = os.path.join(results_dir, "results.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} rows to {parquet_path}")

    # Summary
    summary = compute_summary(df)
    summary_path = os.path.join(results_dir, "summary.parquet")
    summary.to_parquet(summary_path, index=False)
    print(f"Saved summary ({len(summary)} rows) to {summary_path}")

    # Paired diffs
    diffs = compute_paired_diffs(df)
    diffs_path = os.path.join(results_dir, "paired_diffs.parquet")
    diffs.to_parquet(diffs_path, index=False)
    print(f"Saved paired diffs ({len(diffs)} rows) to {diffs_path}")

    # Bootstrap CIs
    try:
        boot_shd = compute_bootstrap_ci(df, "shd", n_bootstrap=args.n_bootstrap)
        boot_f1 = compute_bootstrap_ci(df, "directed_f1", n_bootstrap=args.n_bootstrap)
        boot = boot_shd.merge(boot_f1, on=["condition_name", "ablation_name", "method"], how="outer")
        boot_path = os.path.join(results_dir, "bootstrap_ci.parquet")
        boot.to_parquet(boot_path, index=False)
        print(f"Saved bootstrap CIs ({len(boot)} rows) to {boot_path}")
    except Exception as e:
        print(f"  WARNING: bootstrap CI failed: {e}")

    # Print a quick preview
    print("\n--- Summary preview (SHD mean ± SE) ---")
    pivot_cols = ["condition_name", "method", "shd_mean", "shd_se", "n_seeds", "failure_rate"]
    preview_cols = [c for c in pivot_cols if c in summary.columns]
    with pd.option_context("display.max_rows", 60, "display.width", 120):
        print(summary[preview_cols].to_string(index=False))

    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "results_dir": os.path.abspath(results_dir),
        "n_rows": len(df),
        "conditions": sorted(df["condition_name"].unique().tolist()),
        "methods": sorted(df["method"].unique().tolist()),
        "seeds": sorted(df["seed"].dropna().astype(int).unique().tolist()),
        "n_failed": int(df["failed"].sum()),
        "parquet_path": os.path.abspath(parquet_path),
        "summary_path": os.path.abspath(summary_path),
    }
    with open(os.path.join(results_dir, "aggregate_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAggregation complete. Load with:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{parquet_path}')")


if __name__ == "__main__":
    main()
