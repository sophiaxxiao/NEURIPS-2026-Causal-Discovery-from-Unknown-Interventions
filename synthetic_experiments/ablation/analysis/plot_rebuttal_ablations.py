#!/usr/bin/env python3
# coding=utf-8
"""
Generate rebuttal figures from aggregated ablation results.

Usage
-----
    python analysis/plot_rebuttal_ablations.py \\
        --results-dir results/rebuttal_ablations \\
        --output-dir figures/rebuttal_ablations

Output files
------------
  fig_main_shd.pdf          4-panel main rebuttal figure (ICA-LiNGAM, GIES, IGSP, UT-IGSP)
  fig_full_shd.pdf          4-panel with all methods
  fig_main_f1.pdf           4-panel F1 version
  fig_diagnostics.pdf       source kurtosis and correlation vs severity
  fig_main_shd.png          PNG copies of each
  ...
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Style / constants
# -----------------------------------------------------------------------

# Methods shown in the main rebuttal panel
MAIN_METHODS = ["icalingam", "dlingam"]

# Full method list (used if additional methods are present in results)
ALL_METHODS = ["icalingam", "dlingam", "utigsp", "igsp", "gies"]

# Display names
METHOD_LABELS = {
    "icalingam":      "ICA-LiNGAM",
    "dlingam":        "DirectLiNGAM",
    "utigsp":         "UT-IGSP",
    "igsp":           "IGSP",
    "gies":           "GIES",
    "ut_igsp_oracle": "UT-IGSP (oracle)",
    "jcipc":          "JCI-PC",
}

# Consistent colors and markers
_PALETTE = {
    "icalingam":      ("#2196F3", "o"),
    "dlingam":        ("#FF9800", "s"),
    "utigsp":         ("#4CAF50", "^"),
    "igsp":           ("#9C27B0", "D"),
    "gies":           ("#F44336", "v"),
    "ut_igsp_oracle": ("#795548", "P"),
    "jcipc":          ("#607D8B", "X"),
}

ABLATION_LABELS = {
    "unequal_sizes":           "Ablation A: Unequal environment sizes",
    "heterogeneous_strengths": "Ablation B: Heterogeneous strengths",
    "weak_interventions":      "Ablation C: Weak interventions",
    "off_target":              "Ablation D: Correlated off-target effects",
}

XAXIS_LABELS = {
    "unequal_sizes":           "Severity  h  (p range = [0.5−h, 0.5+h])",
    "heterogeneous_strengths": "Log-spread  τ",
    "weak_interventions":      "Shift magnitude  δ  (obs noise std = 1)",
    "off_target":              "Off-target amplitude  ρ",
}

SEVERITY_PARAMS = {
    "unequal_sizes":           "h_unequal",
    "heterogeneous_strengths": "tau_hetero",
    "weak_interventions":      "delta_scalar",
    "off_target":              "off_target_rho",
}


# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------

def load_summary(results_dir: str) -> pd.DataFrame:
    summary_path = os.path.join(results_dir, "summary.parquet")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"summary.parquet not found in {results_dir!r}. "
            "Run aggregate_rebuttal_ablations.py first."
        )
    return pd.read_parquet(summary_path)


def load_results(results_dir: str) -> pd.DataFrame:
    parquet_path = os.path.join(results_dir, "results.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"results.parquet not found in {results_dir!r}.")
    return pd.read_parquet(parquet_path)


# -----------------------------------------------------------------------
# Panel helpers
# -----------------------------------------------------------------------

def _get_baseline_value(summary: pd.DataFrame, method: str, metric: str = "shd_mean") -> float:
    row = summary[(summary["ablation_name"] == "baseline") & (summary["method"] == method)]
    if row.empty or metric not in row.columns:
        return float("nan")
    return float(row[metric].values[0])


def _plot_panel(ax, summary: pd.DataFrame, ablation_name: str, methods: list,
                metric: str = "shd_mean", err_metric: str = "shd_se",
                show_legend: bool = True):
    """Plot one ablation panel on the given axes."""
    ablation_data = summary[summary["ablation_name"] == ablation_name]

    for method in methods:
        color, marker = _PALETTE.get(method, ("#999", "o"))
        label = METHOD_LABELS.get(method, method)

        # Include baseline as the zero-severity point
        mdata = ablation_data[ablation_data["method"] == method].copy()

        # Get baseline
        bline_row = summary[
            (summary["ablation_name"] == "baseline") & (summary["method"] == method)
        ]
        if not bline_row.empty and metric in bline_row.columns:
            bline_sev = 0.0
            bline_val = float(bline_row[metric].values[0])
            bline_err = float(bline_row[err_metric].values[0]) if err_metric in bline_row.columns else 0.0
        else:
            bline_sev = bline_val = bline_err = float("nan")

        # Severity values from non-baseline
        if "severity_value" in mdata.columns:
            x = pd.to_numeric(mdata["severity_value"], errors="coerce").values
        else:
            x = np.array([])

        y = mdata[metric].values.astype(float) if metric in mdata.columns else np.array([])
        e = mdata[err_metric].values.astype(float) if err_metric in mdata.columns else np.zeros_like(y)

        # Prepend baseline
        x = np.concatenate([[bline_sev], x])
        y = np.concatenate([[bline_val], y])
        e = np.concatenate([[bline_err], e])

        # Sort by x
        order = np.argsort(x)
        x, y, e = x[order], y[order], e[order]

        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() == 0:
            continue

        ax.errorbar(
            x[valid], y[valid], yerr=e[valid],
            color=color, marker=marker, label=label,
            linewidth=1.5, markersize=6, capsize=3, capthick=1,
        )

    ax.set_title(ABLATION_LABELS.get(ablation_name, ablation_name), fontsize=10, pad=6)
    ax.set_xlabel(XAXIS_LABELS.get(ablation_name, "Severity"), fontsize=9)
    ax.set_ylabel("Mean SHD ↓", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)

    if show_legend:
        ax.legend(fontsize=8, framealpha=0.8, loc="best")


# -----------------------------------------------------------------------
# Four-panel SHD figure
# -----------------------------------------------------------------------

def plot_four_panel(summary: pd.DataFrame, methods: list,
                    metric: str = "shd_mean", err_metric: str = "shd_se",
                    output_path: str = None, title: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    ablations = ["unequal_sizes", "heterogeneous_strengths", "weak_interventions", "off_target"]
    for ax, abl in zip(axes, ablations):
        _plot_panel(ax, summary, abl, methods,
                    metric=metric, err_metric=err_metric,
                    show_legend=(abl == ablations[0]))

    # Shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].get_legend().remove()
        fig.legend(handles, labels, loc="lower center", ncol=len(methods),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    sup = title or "Rebuttal Ablation Suite — Mean SHD (lower is better)"
    fig.suptitle(sup, fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fig.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


# -----------------------------------------------------------------------
# Diagnostics figure
# -----------------------------------------------------------------------

def plot_diagnostics(summary: pd.DataFrame, output_path: str = None):
    """Plot source kurtosis and source dependence vs ablation severity."""
    diag_cols = {
        "mean_abs_excess_kurtosis_mean": "Mean |excess kurtosis| of sources",
        "min_abs_excess_kurtosis_mean": "Min |excess kurtosis| of sources",
        "mean_abs_offdiag_source_correlation_mean": "Mean |off-diag source correlation|",
    }
    available = [c for c in diag_cols if c in summary.columns]
    if not available:
        print("  WARNING: No diagnostic columns found in summary. Skipping diagnostic plot.")
        return None

    ablations = ["unequal_sizes", "heterogeneous_strengths", "weak_interventions", "off_target"]
    n_diag = len(available)
    n_abl = len(ablations)

    fig, axes = plt.subplots(n_diag, n_abl, figsize=(3.5 * n_abl, 3 * n_diag), squeeze=False)

    for col_idx, abl in enumerate(ablations):
        abl_data = summary[summary["ablation_name"] == abl]
        for row_idx, diag_col in enumerate(available):
            ax = axes[row_idx][col_idx]

            if diag_col not in abl_data.columns:
                ax.set_visible(False)
                continue

            # One curve per method (diagnostics are per-condition, not per-method,
            # but we show first method's data)
            methods_in = abl_data["method"].unique()
            for method in methods_in[:1]:  # diagnostics don't depend on method
                mdata = abl_data[abl_data["method"] == method]
                if "severity_value" in mdata.columns:
                    x = pd.to_numeric(mdata["severity_value"], errors="coerce").values
                else:
                    x = np.array([])

                # Add baseline
                brow = summary[
                    (summary["ablation_name"] == "baseline") & (summary["method"] == method)
                ]
                bx = np.array([0.0])
                by = brow[diag_col].values.astype(float) if (not brow.empty and diag_col in brow.columns) else np.array([float("nan")])
                y = mdata[diag_col].values.astype(float) if diag_col in mdata.columns else np.array([])
                x = np.concatenate([bx, x])
                y = np.concatenate([by, y])
                order = np.argsort(x)
                x, y = x[order], y[order]
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() > 0:
                    ax.plot(x[valid], y[valid], "o-", color="#333", linewidth=1.5, markersize=5)

            ax.set_xlabel(XAXIS_LABELS.get(abl, "Severity"), fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(diag_cols[diag_col], fontsize=8)
            if row_idx == 0:
                ax.set_title(ABLATION_LABELS.get(abl, abl).split(":")[1].strip(), fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=7)

    fig.suptitle("Source Diagnostics vs Ablation Severity", fontsize=12)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fig.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"  Saved {output_path}")
    return fig


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot rebuttal ablation results.")
    parser.add_argument("--results-dir", required=True, help="Directory with results.parquet / summary.parquet.")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures.")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading summary from {results_dir!r}")
    summary = load_summary(results_dir)

    print(f"Available methods: {sorted(summary['method'].unique().tolist())}")
    print(f"Available conditions: {sorted(summary['condition_name'].unique().tolist())}")

    # Determine which main methods are actually present
    present = set(summary["method"].unique())
    main_methods_present = [m for m in MAIN_METHODS if m in present]
    all_methods_present = [m for m in ALL_METHODS if m in present]

    if not main_methods_present:
        print("WARNING: None of the main methods found; using all available.")
        main_methods_present = sorted(present)

    # 1. Main rebuttal figure — SHD, main methods
    print("Generating main SHD figure (main methods)...")
    plot_four_panel(
        summary, main_methods_present,
        metric="shd_mean", err_metric="shd_se",
        output_path=os.path.join(output_dir, "fig_main_shd.pdf"),
        title="Rebuttal Ablations — Mean SHD (main methods)",
    )

    # 2. Full SHD figure — all methods
    print("Generating full SHD figure (all methods)...")
    plot_four_panel(
        summary, all_methods_present,
        metric="shd_mean", err_metric="shd_se",
        output_path=os.path.join(output_dir, "fig_full_shd.pdf"),
        title="Rebuttal Ablations — Mean SHD (all methods)",
    )

    # 3. Main rebuttal figure — F1
    print("Generating main F1 figure...")
    plot_four_panel(
        summary, main_methods_present,
        metric="directed_f1_mean", err_metric="directed_f1_se",
        output_path=os.path.join(output_dir, "fig_main_f1.pdf"),
        title="Rebuttal Ablations — Directed F1 (main methods)",
    )

    # 4. Diagnostics figure
    print("Generating diagnostics figure...")
    plot_diagnostics(
        summary,
        output_path=os.path.join(output_dir, "fig_diagnostics.pdf"),
    )

    print(f"\nAll figures saved to {output_dir!r}")


if __name__ == "__main__":
    main()
