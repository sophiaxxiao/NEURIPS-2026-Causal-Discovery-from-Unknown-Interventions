"""Regenerate the main-text Sachs figure (Fig. 4a) at print size, with readable text.

Reproduces the asymmetric pooling sweep of the student's notebook
(real_data_experiments/sachs_protein/notebooks/sachs_intervention_test.ipynb, cells 2, 3 and 20):
  * ICA-LiNGAM / DirectLiNGAM: n samples drawn from all 9 conditions (6456), no labels
  * GIES / IGSP: n samples drawn from the 5-condition Squires subset (4275), split by label,
    known targets [[], [1], [3], [4], [6], [8]]
  * random guesses with ICA-LiNGAM's edge count and with the true edge count
  * 100 draws per size from np.random.default_rng(1), in the notebook's order
Differences from the notebook: ICA-LiNGAM and the random-guess graphs were unseeded there and
are seeded here, so the figure is reproducible; their curves can differ slightly.

    .venv/bin/python make_fig4a_sachs.py --n-jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
SACHS = os.path.join(HERE, "..", "real_data_experiments", "sachs_protein")
TOTAL_SIZES = [200, 500, 1000, 2000, 4000, "all"]
N_BOOT = 100
SQUIRES_SUBSET = {"cd3cd28": [], "cd3cd28+u0126": [1], "cd3cd28+psitect": [3], "cd3cd28+ly": [4],
                  "cd3cd28+aktinhib": [6], "cd3cd28+g0076": [8]}
METHODS = ["dlingam", "icalingam", "gies", "igsp", "random_guess_ica", "random_guess_groundtruth"]


def load():
    sys.path.insert(0, os.path.join(SACHS, "src"))
    from load_sachs import PROTEINS, get_ground_truth_bnlearn, load_sachs
    all_data = load_sachs(xls_dir=os.path.join(SACHS, "data", "original_data"),
                          use_western_blot=False, transform=False)
    true_amat = get_ground_truth_bnlearn().values.astype(int)
    sq = all_data[all_data["condition"].isin(SQUIRES_SUBSET)].reset_index(drop=True)
    return all_data[PROTEINS].values.astype(float), sq, true_amat, PROTEINS


def draws():
    """Index draws in exactly the notebook's order (one rng across all sizes and draws)."""
    X_full, sq, _, _ = load()
    rng2 = np.random.default_rng(1)
    out = []
    for s in TOTAL_SIZES:
        for b in range(N_BOOT):
            n_full = len(X_full) if s == "all" else min(s, len(X_full))
            i_full = rng2.choice(len(X_full), n_full, replace=False)
            n_sq = len(sq) if s == "all" else min(s, len(sq))
            i_sq = rng2.choice(len(sq), n_sq, replace=False)
            out.append((s, b, i_full, i_sq))
    return out


def one(s, b, i_full, i_sq):
    warnings.filterwarnings("ignore")
    import gies
    from castle.algorithms import DirectLiNGAM, ICALiNGAM
    from castle.datasets import DAG
    from castle.metrics import MetricsDAG
    from causaldag import igsp
    from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test
    from causaldag.utils.invariance_tests import (MemoizedInvarianceTester, gauss_invariance_suffstat,
                                                  gauss_invariance_test)
    X_full, sq, true_amat, PROTEINS = load()
    shd = lambda G: MetricsDAG(np.asarray(G), true_amat).metrics["shd"]  # noqa: E731
    X_pool = X_full[i_full]
    sq_b = sq.iloc[i_sq]
    X_obs_b = sq_b[sq_b["condition"] == "cd3cd28"][PROTEINS].values.astype(float)
    X_int_b = [sq_b[sq_b["condition"] == c][PROTEINS].values.astype(float)
               for c in SQUIRES_SUBSET if c != "cd3cd28"]
    tg = [t for c, t in SQUIRES_SUBSET.items() if c != "cd3cd28"]
    nonempty = [(x, t) for x, t in zip(X_int_b, tg) if len(x) > 0]
    X_int_f = [x for x, _ in nonempty]
    tg_f = [t for _, t in nonempty]
    r = {"size": s, "boot": b}
    try:
        m = DirectLiNGAM(); m.learn(X_pool); r["dlingam"] = shd(m.causal_matrix)
    except Exception:  # noqa: BLE001
        r["dlingam"] = np.nan
    ica_edges = None
    try:
        m = ICALiNGAM(random_state=b); m.learn(X_pool); r["icalingam"] = shd(m.causal_matrix)
        ica_edges = int(np.sum(m.causal_matrix != 0))
    except Exception:  # noqa: BLE001
        r["icalingam"] = np.nan
    try:
        est, _ = gies.fit_bic([X_obs_b] + X_int_f, [[]] + tg_f); r["gies"] = shd(est)
    except Exception:  # noqa: BLE001
        r["gies"] = np.nan
    np.random.seed(10_000 + 7 * b + (0 if s == "all" else int(s)))
    try:
        r["random_guess_ica"] = shd(DAG.erdos_renyi(len(PROTEINS), ica_edges)) if ica_edges else np.nan
    except Exception:  # noqa: BLE001
        r["random_guess_ica"] = np.nan
    r["random_guess_groundtruth"] = shd(DAG.erdos_renyi(len(PROTEINS), int(true_amat.sum())))
    try:
        if len(X_obs_b) < 5 or not tg_f:
            raise ValueError("too few samples")
        ci = MemoizedCI_Tester(gauss_ci_test, gauss_ci_suffstat(X_obs_b), alpha=0.05)
        inv = MemoizedInvarianceTester(gauss_invariance_test, gauss_invariance_suffstat(X_obs_b, X_int_f), alpha=0.05)
        dag = igsp([dict(interventions=set(t)) for t in tg_f], set(range(len(PROTEINS))), ci, inv)
        r["igsp"] = shd(dag.to_amat()[0])
    except Exception:  # noqa: BLE001
        r["igsp"] = np.nan
    return r


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    labels = {"dlingam": "DirectLiNGAM (pooled)", "icalingam": "ICA-LiNGAM (pooled)",
              "gies": "GIES (known targets)", "igsp": "IGSP (known targets)",
              "random_guess_ica": "Random, ICA-LiNGAM's # edges",
              "random_guess_groundtruth": "Random, true # edges"}
    colors = {"dlingam": "#1f77b4", "icalingam": "#ff7f0e", "gies": "#8c564b", "igsp": "#d62728",
              "random_guess_ica": "#9467bd", "random_guess_groundtruth": "#7f7f7f"}
    markers = {"dlingam": "o", "icalingam": "s", "gies": "P", "igsp": "D",
               "random_guess_ica": "X", "random_guess_groundtruth": "x"}
    x = np.arange(len(TOTAL_SIZES))
    fig, ax = plt.subplots(figsize=(3.8, 2.15))
    for m in METHODS:
        g = df.groupby("size", sort=False)[m]
        mu = np.array([g.get_group(s).mean() for s in TOTAL_SIZES])
        sd = np.array([g.get_group(s).std(ddof=0) for s in TOTAL_SIZES])
        rnd = m.startswith("random")
        ax.plot(x, mu, color=colors[m], marker=markers[m], ms=4, lw=1.4, ls=":" if rnd else "-",
                label=labels[m])
        if not rnd:
            ax.fill_between(x, mu - sd, mu + sd, color=colors[m], alpha=0.12, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["200", "500", "1k", "2k", "4k", "all"])
    ax.set_xlabel("total samples drawn")
    ax.set_ylabel("SHD (mean ± 1 SD)")
    ax.set_ylim(0, 31)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.3, ncol=2, loc="lower left", frameon=False, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sachs_asymmetric_sweep.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "sachs_asymmetric_sweep.png"), dpi=200, bbox_inches="tight")


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(HERE, "results", "sachs_fig4a_raw.csv")
    if not a.plot_only:
        rows = Parallel(n_jobs=a.n_jobs, verbose=2)(delayed(one)(*d) for d in draws())
        pd.DataFrame(rows).to_csv(csv, index=False)
    df = pd.read_csv(csv, dtype={"size": str})
    df["size"] = df["size"].map(lambda v: v if v == "all" else int(v))
    print(df.groupby("size", sort=False)[METHODS].mean().round(2))
    plot(df, a.out)


if __name__ == "__main__":
    main()
