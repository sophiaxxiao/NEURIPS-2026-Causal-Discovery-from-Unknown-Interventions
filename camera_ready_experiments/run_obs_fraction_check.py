"""Does ICA-LiNGAM's SHD rise with n in the single-node ladder because of source dependence,
or because the fixed 100 observational samples become a shrinking share of the pool?

Same data pipeline as run_jci_ladder.py, ICA-LiNGAM only, two arms per DAG:
  fixed_count : 100 observational samples at every n (as in run_jci_ladder.py)
  fixed_share : round(n/10) observational samples, i.e. the observational environment is the
                same size as an average interventional environment (share 1/11) at every n
Both arms draw observational rows from one pool of 1000 samples generated with the DAG, and
use the same per-node shifts, delta * var over all 1000 observational rows (the ladder uses its
100 rows, so shifts differ slightly; the fixed_count arm checks that the ladder is reproduced).

    .venv/bin/python run_obs_fraction_check.py --n-dags 100 --n-jobs 10
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
STUDENT_SRC = os.path.join(HERE, "..", "synthetic_experiments", "src")


def one_dag(seed, sizes, d=10, edges=10, delta=1.0):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, STUDENT_SRC)
    from castle.algorithms import ICALiNGAM
    from castle.datasets import DAG
    from castle.metrics import MetricsDAG
    from MakeDataset import ShiftSimulation
    W = DAG.erdos_renyi(n_nodes=d, n_edges=edges, seed=seed, weight_range=[0.5, 2])
    sim = ShiftSimulation(n_nodes=d, n_edges=edges, n_samples=max(sizes), n_observational=1000,
                          weighted_random_dag=W, delta=delta, seed=seed,
                          intervention_mode="single_node", include_no_intervention=False)
    sim._generate_datasets()
    obs_pool = sim.dataset_observational.X
    pool = np.vstack([sim.datasets_interventional[k]["X"] for k in sim.datasets_interventional])
    perm = np.random.default_rng(seed).permutation(len(pool))
    G_true = (W != 0).astype(int)
    rows = []
    for n in sizes:
        for arm, n_obs in (("fixed_count", 100), ("fixed_share", int(round(n / 10)))):
            X = np.vstack([obs_pool[:n_obs], pool[perm[:n]]])
            m = ICALiNGAM(); m.learn(X)
            rows.append({"seed": seed, "n_int": n, "arm": arm, "n_obs": n_obs,
                         "shd": MetricsDAG((m.causal_matrix != 0).astype(int), G_true).metrics["shd"]})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dags", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 5000, 10000])
    a = ap.parse_args()
    res = Parallel(n_jobs=a.n_jobs)(delayed(one_dag)(s, a.sizes) for s in range(a.n_dags))
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(HERE, "results", "obs_fraction_check_raw.csv"), index=False)
    s = df.groupby(["arm", "n_int"]).shd.agg(["mean", "sem"]).unstack(0).round(2)
    s.to_csv(os.path.join(HERE, "results", "obs_fraction_check_summary.csv"))
    print(s)
