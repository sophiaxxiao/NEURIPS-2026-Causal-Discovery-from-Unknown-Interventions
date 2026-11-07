"""Information-ladder comparison on synthetic data (camera-ready, Reviewer AMBX).

Same data pipeline as the student's repo (ERDagsComparison.py / ResidualDependenceSHDAblation.py):
  * castle ER DAG, d=10, 10 edges, |w| in [0.5, 2]
  * ShiftSimulation, intervention_mode='single_node', include_no_intervention=False,
    delta_per_node = delta * var(X_obs) (student convention), 100 observational rows
    always prepended
  * per sample size ss: the first ss rows of one random permutation of the
    interventional pool (nested subsamples, as in ERDagsComparison.py)

Methods, ordered by the information they use:
  ICA-LiNGAM (castle)     neither environment labels nor targets  [the paper's method]
  JCI-PC (env labels)     environment labels, targets unknown     [ours: jci_pc.py]
  UT-IGSP                 environment labels, targets unknown
  JCI-PC (known targets)  labels + targets   [student's run_jci_pc]
  IGSP, GIES              labels + targets

Metrics: castle MetricsDAG SHD (the convention of Fig. 3b / Fig. 4; undirected edges whose
skeleton is right are not penalised) and strict SHD (causal_utils.shd: undirected = error).

    python run_jci_ladder.py --quick
    python run_jci_ladder.py --n-dags 100 --n-jobs 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
STUDENT_SRC = os.path.join(HERE, "..", "synthetic_experiments", "src")
sys.path.insert(0, HERE)


def one_dag(seed: int, args) -> list[dict]:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, STUDENT_SRC)
    from castle.algorithms import ICALiNGAM
    from castle.datasets import DAG
    from castle.metrics import MetricsDAG
    from MakeDataset import ShiftSimulation
    from method_wrappers import (build_env_labels, run_gies, run_igsp, run_jci_pc,
                                 run_ut_igsp, split_subsample)

    from causal_utils import all_metrics
    from jci_pc import jci_pc

    true_dag = DAG.erdos_renyi(n_nodes=args.d, n_edges=args.edges, seed=seed, weight_range=[0.5, 2])
    sim = ShiftSimulation(n_nodes=args.d, n_edges=args.edges, n_samples=max(args.sizes),
                          n_observational=args.n_obs, weighted_random_dag=true_dag,
                          delta=args.delta, seed=seed, intervention_mode="single_node",
                          include_no_intervention=False)
    sim._generate_datasets()
    obs = sim.dataset_observational.X
    interv = sim.datasets_interventional
    pool = np.vstack([interv[k]["X"] for k in interv])
    labels = build_env_labels(sim)               # 0..K-1 per interventional env
    perm = np.random.default_rng(seed).permutation(len(pool))
    G_true = (true_dag != 0).astype(int)

    rows = []
    for ss in args.sizes:
        X_int, lab = pool[perm[:ss]], labels[perm[:ss]]
        X = np.vstack([obs, X_int])
        env = np.concatenate([np.zeros(len(obs), int), lab + 1])   # 0 = observational
        obs_sub, ivs, tg = split_subsample(X_int, lab, sim)

        def rec(method, fn, **extra):
            t0 = time.time()
            try:
                G = fn()
            except Exception:  # noqa: BLE001
                G = None
            row = {"seed": seed, "n_int": ss, "n_total": len(X), "method": method,
                   "runtime_s": time.time() - t0, **extra}
            if G is None:
                row["shd_castle"] = np.nan
            else:
                G = (np.asarray(G) != 0).astype(int)
                row["shd_castle"] = MetricsDAG(G, G_true).metrics["shd"]
                row.update(all_metrics(G, G_true))
            rows.append(row)

        def ica():
            m = ICALiNGAM(); m.learn(X); return m.causal_matrix
        rec("ICA-LiNGAM", ica)
        for a in args.alphas:
            rec("JCI-PC (env labels)", lambda: jci_pc(X, env, alpha=a, reference=0), alpha=a)
        if not args.only_new:
            rec("UT-IGSP", lambda: run_ut_igsp(obs_sub, ivs))
            rec("JCI-PC (known targets)", lambda: run_jci_pc(obs_sub, ivs, tg))
            rec("IGSP", lambda: run_igsp(obs_sub, ivs, tg))
            rec("GIES", lambda: run_gies(obs_sub, ivs, tg))
    return rows


def summarize(df):
    df = df.copy()
    df["alpha"] = df.get("alpha", np.nan)
    df["alpha"] = df["alpha"].fillna(-1)
    g = df.groupby(["method", "alpha", "n_int"])
    out = g.agg(shd_castle=("shd_castle", "mean"), shd_castle_se=("shd_castle", "sem"),
                shd_strict=("shd", "mean"), shd_strict_se=("shd", "sem"),
                shd_strict_median=("shd", "median"), dir_f1=("dir_f1", "mean"),
                n_undirected=("n_undirected", "mean"), fails=("shd_castle", lambda s: s.isna().sum()),
                runs=("seed", "size")).reset_index()
    out["alpha"] = out["alpha"].replace(-1, np.nan)
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (col, se, title) in zip(axes, [("shd_castle", "shd_castle_se", "castle SHD (as Fig. 3b)"),
                                            ("shd_strict", "shd_strict_se", "strict SHD (undirected = error)")]):
        for (m, a), sub in summary.groupby(["method", "alpha"], dropna=False):
            sub = sub.sort_values("n_int")
            lab = m if pd.isna(a) else f"{m} α={a:g}"
            ax.errorbar(sub["n_int"], sub[col], yerr=sub[se], marker="o", capsize=3, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("interventional samples (+100 obs)"); ax.set_title(title)
    axes[0].set_ylabel("SHD (mean ± s.e.)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Single-node design, d=10")
    fig.tight_layout(); fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dags", type=int, default=100)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--edges", type=int, default=10)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.01])
    ap.add_argument("--only-new", action="store_true", help="skip the student's baselines")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="jci_ladder")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_dags, args.sizes, args.tag = 4, [500, 5000], args.tag + "_quick"
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_dag)(s, args) for s in range(args.n_dags))
    df = pd.DataFrame([r for rows in res for r in rows])
    raw = os.path.join(args.out_dir, f"{args.tag}_raw.csv"); df.to_csv(raw, index=False)
    summ = summarize(df); sp = os.path.join(args.out_dir, f"{args.tag}_summary.csv"); summ.to_csv(sp, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    plot(summ, os.path.join(args.out_dir, f"{args.tag}_shd.png"))
    print(f"\n{time.time() - t0:.0f}s; raw -> {raw}; summary -> {sp}")


if __name__ == "__main__":
    main()
