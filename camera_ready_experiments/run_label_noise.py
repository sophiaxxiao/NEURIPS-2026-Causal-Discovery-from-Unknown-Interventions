"""How much label noise erases the advantage of label-aware methods?

Same data pipeline and seeds as run_jci_ladder.py (single-node design, d=10, 10 edges,
|w| in [0.5, 2], delta=1 times the observational variance, 100 observational samples), so the
eps = 0 rows reproduce the ladder at n = 1000 and n = 10000. Two kinds of noise:

  env   : each interventional sample is assigned to a wrong environment (uniform over the
          other d-1 interventional environments) with probability eps. This corrupts the
          grouping that every label-aware method relies on. Observational samples stay labelled.
  target: each interventional environment's listed target is replaced by a uniformly random
          wrong node with probability eps. The grouping is correct; only methods that use
          targets (JCI with known targets, IGSP, GIES) are affected.

ICA-LiNGAM never sees labels and mislabelling does not change the data, so it is run once per
n and serves as the reference line.

Noise uses common random numbers: each sample (or environment) gets one uniform draw and one
replacement label, fixed across eps, and is corrupted when its draw is below eps. Corruption is
therefore nested in eps.

    .venv/bin/python run_label_noise.py --quick
    .venv/bin/python run_label_noise.py --n-dags 100 --n-jobs 10
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

EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def _wrong_choice(rng, true_vals, n_choices):
    """A uniformly random value in range(n_choices) different from each entry of true_vals."""
    r = rng.integers(0, n_choices - 1, size=len(true_vals))
    return r + (r >= true_vals)


def one_dag(seed: int, args) -> list[dict]:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, STUDENT_SRC)
    from castle.algorithms import ICALiNGAM
    from castle.datasets import DAG
    from castle.metrics import MetricsDAG
    from MakeDataset import ShiftSimulation
    from method_wrappers import (build_env_labels, run_gies, run_igsp, run_jci_pc,
                                 run_ut_igsp, split_subsample)

    from jci_pc import jci_pc

    d = args.d
    true_dag = DAG.erdos_renyi(n_nodes=d, n_edges=args.edges, seed=seed, weight_range=[0.5, 2])
    sim = ShiftSimulation(n_nodes=d, n_edges=args.edges, n_samples=max(args.sizes),
                          n_observational=args.n_obs, weighted_random_dag=true_dag,
                          delta=args.delta, seed=seed, intervention_mode="single_node",
                          include_no_intervention=False)
    sim._generate_datasets()
    obs = sim.dataset_observational.X
    interv = sim.datasets_interventional
    pool = np.vstack([interv[k]["X"] for k in interv])
    labels = build_env_labels(sim)
    K = len(interv)                                   # number of interventional environments
    perm = np.random.default_rng(seed).permutation(len(pool))   # same as run_jci_ladder.py
    G_true = (true_dag != 0).astype(int)

    rows = []

    def rec(n, method, noise, eps, fn):
        # IGSP / UT-IGSP draw from NumPy's global RNG; seed it so reruns are reproducible
        np.random.seed((seed * 1_000_003 + n * 101 + int(round(eps * 1000))) % 2**32)
        t0 = time.time()
        try:
            G = fn()
        except Exception:  # noqa: BLE001
            G = None
        row = {"seed": seed, "n_int": n, "method": method, "noise": noise, "eps": eps,
               "runtime_s": time.time() - t0}
        if G is None:
            row["shd_castle"] = np.nan
        else:
            G = (np.asarray(G) != 0).astype(int)
            row["shd_castle"] = MetricsDAG(G, G_true).metrics["shd"]
        rows.append(row)

    for n in args.sizes:
        X_int, lab = pool[perm[:n]], labels[perm[:n]]
        X = np.vstack([obs, X_int])
        crn = np.random.default_rng([seed, n, 7])
        u_env = crn.random(n)
        alt_env = _wrong_choice(crn, lab, K)
        u_tgt = crn.random(K)
        tgt_true = np.array([[j for j, v in enumerate(p) if v == 1][0] for p in interv.keys()])
        alt_tgt = _wrong_choice(crn, tgt_true, d)

        def ica():
            m = ICALiNGAM(); m.learn(X); return m.causal_matrix
        rec(n, "ICA-LiNGAM", "none", 0.0, ica)

        for eps in args.eps:
            # --- environment-label noise: corrupt the grouping itself ---
            lab_e = np.where(u_env < eps, alt_env, lab)
            env = np.concatenate([np.zeros(len(obs), int), lab_e + 1])
            obs_sub, ivs, tg = split_subsample(X_int, lab_e, sim)
            rec(n, "JCI-PC (env labels)", "env", eps,
                lambda: jci_pc(X, env, alpha=0.05, reference=0))
            rec(n, "UT-IGSP", "env", eps, lambda: run_ut_igsp(obs_sub, ivs))
            rec(n, "JCI-PC (known targets)", "env", eps, lambda: run_jci_pc(obs_sub, ivs, tg))
            rec(n, "IGSP", "env", eps, lambda: run_igsp(obs_sub, ivs, tg))
            rec(n, "GIES", "env", eps, lambda: run_gies(obs_sub, ivs, tg))

            # --- target noise: correct grouping, wrong listed targets ---
            obs_sub, ivs, tg = split_subsample(X_int, lab, sim)
            # split_subsample lists environments in pattern order, skipping empty ones
            present = [i for i in range(K) if np.any(lab == i)]
            tg_noisy = [[int(alt_tgt[i])] if u_tgt[i] < eps else [int(tgt_true[i])] for i in present]
            assert [[int(tgt_true[i])] for i in present] == tg, "target alignment check failed"
            rec(n, "JCI-PC (known targets)", "target", eps, lambda: run_jci_pc(obs_sub, ivs, tg_noisy))
            rec(n, "IGSP", "target", eps, lambda: run_igsp(obs_sub, ivs, tg_noisy))
            rec(n, "GIES", "target", eps, lambda: run_gies(obs_sub, ivs, tg_noisy))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dags", type=int, default=100)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--edges", type=int, default=10)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tag = "label_noise"
    if args.quick:
        args.n_dags, args.eps, tag = 3, [0.0, 0.2], tag + "_quick"
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_dag)(s, args) for s in range(args.n_dags))
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(args.out_dir, f"{tag}_raw.csv"), index=False)
    summ = (df.groupby(["noise", "method", "n_int", "eps"])
              .agg(shd=("shd_castle", "mean"), se=("shd_castle", "sem"),
                   fails=("shd_castle", lambda s: int(s.isna().sum())), runs=("seed", "size"))
              .reset_index())
    summ.to_csv(os.path.join(args.out_dir, f"{tag}_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.round(2).to_string(index=False))
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
