# Try ICALiNGAM on dependent data. Compare to observational baseline with matched sample size.
# Three conditions: (1) single-node pooled, (2) Bernoulli pooled (IID sources), (3) observational.
# Note: rescaled data to remove var-sortability.

import numpy as np
from joblib import Parallel, delayed
from MakeDataset import ShiftSimulation
from castle.datasets import DAG
from castle.algorithms import ICALiNGAM
from helpers import compute_graph_metrics

n_trials = 20
d_values = np.arange(10, 150, 5) # smoke test
shift = 2
N_JOBS = -1 # Server. Change to 8 on MacBook, -1 uses all available cores

# Metric names and indices
METRIC_NAMES = ['shd', 'sid_lower', 'skel_shd', 'f1', 'precision', 'recall']
N_METRICS = len(METRIC_NAMES)


def compute_skeleton_shd(est_dag, true_adj):
    est_skeleton = np.clip((est_dag != 0).astype(int) + (est_dag != 0).astype(int).T, 0, 1)
    true_skeleton = np.clip(true_adj + true_adj.T, 0, 1)
    np.fill_diagonal(est_skeleton, 0)
    np.fill_diagonal(true_skeleton, 0)
    assert np.allclose(est_skeleton.T, est_skeleton), "Estimated skeleton should be symmetric"
    assert np.allclose(true_skeleton.T, true_skeleton), "True skeleton should be symmetric"
    est_flat = est_skeleton.flatten()
    true_flat = true_skeleton.flatten()
    FP = np.sum((est_flat == 1) & (true_flat == 0)).item()
    FN = np.sum((est_flat == 0) & (true_flat == 1)).item()
    return (FP + FN) // 2


def fit_and_score(X, true_adj, random_state=None):
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    model = ICALiNGAM(max_iter=5000, random_state=random_state)
    model.learn(X_std)
    m = compute_graph_metrics(model.causal_matrix, true_adj)
    skel_shd = compute_skeleton_shd(model.causal_matrix, true_adj)
    return np.array([
        m['shd'],
        m['sid_lower'],
        skel_shd,
        m['F1']        if not np.isnan(m['F1'])        else 0.0,
        m['precision'] if not np.isnan(m['precision']) else 0.0,
        m['recall']    if not np.isnan(m['recall'])    else 0.0,
    ])


def run_one_d(d_idx, d, n_trials, shift):
    n_edges = d
    res_single = np.zeros((n_trials, N_METRICS))
    res_bern   = np.zeros((n_trials, N_METRICS))
    res_obs    = np.zeros((n_trials, N_METRICS))

    for trial in range(n_trials):
        true_dag = DAG.erdos_renyi(n_nodes=d, n_edges=n_edges, seed=trial, weight_range=(0.5, 2))
        true_adj = (true_dag != 0).astype(int)

        # --- Single-node interventions ---
        sim_single = ShiftSimulation(
            n_nodes=d, n_edges=n_edges, n_samples=min(d * int(1e4), 50_000),
            weighted_random_dag=true_dag, delta=10.0, seed=trial,
            intervention_mode='single_node', include_no_intervention=True,
        )
        sim_single.simulate_observational()
        sim_single.compute_per_node_delta_from_observational(scalar=shift)
        sim_single.setup_interventions()
        X_single = np.vstack([data['X'] for data in sim_single.simulate_interventions().values()])

        # --- Bernoulli interventions (p=1/d → same avg. rate as single-node, but independent) ---
        sim_bern = ShiftSimulation(
            n_nodes=d, n_edges=n_edges, n_samples=min(d * int(1e4), 50_000),
            weighted_random_dag=true_dag, delta=10.0, seed=trial,
            intervention_mode='bernoulli', p_intervention=1/d,
        )
        sim_bern.simulate_observational()
        sim_bern.compute_per_node_delta_from_observational(scalar=shift)
        sim_bern.setup_interventions()
        X_bern = np.vstack([data['X'] for data in sim_bern.simulate_interventions().values()])

        X_obs = sim_single.dataset_observational.X[:len(X_single)]

        res_single[trial] = fit_and_score(X_single, true_adj, random_state=trial)
        res_bern[trial]   = fit_and_score(X_bern,   true_adj, random_state=trial + 1000)
        res_obs[trial]    = fit_and_score(X_obs,     true_adj, random_state=trial + 2000)

        print(
            f"  d={d} trial={trial}: "
            f"single SHD={res_single[trial,0]:.0f} F1={res_single[trial,3]:.2f} | "
            f"bern SHD={res_bern[trial,0]:.0f} F1={res_bern[trial,3]:.2f} | "
            f"obs SHD={res_obs[trial,0]:.0f} F1={res_obs[trial,3]:.2f} | "
            f"true_edges={true_adj.sum()}",
            flush=True
        )

    return d_idx, res_single, res_bern, res_obs


outputs = Parallel(n_jobs=N_JOBS)(
    delayed(run_one_d)(d_idx, d, n_trials, shift)
    for d_idx, d in enumerate(d_values)
)

results_single = np.zeros((len(d_values), n_trials, N_METRICS))
results_bern   = np.zeros((len(d_values), n_trials, N_METRICS))
results_obs    = np.zeros((len(d_values), n_trials, N_METRICS))
for d_idx, res_single, res_bern, res_obs in outputs:
    results_single[d_idx] = res_single
    results_bern[d_idx]   = res_bern
    results_obs[d_idx]    = res_obs

np.save('icalingam_results_single.npy', results_single)
np.save('icalingam_results_bern.npy',   results_bern)
np.save('icalingam_results_obs.npy',    results_obs)
print("\nResults saved.")

import matplotlib.pyplot as plt

METRIC_LABELS = ['SHD', 'SID Lower Bound', 'Skeleton SHD', 'F1', 'Precision', 'Recall']
n_cols = 3
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()

for i, label in enumerate(METRIC_LABELS):
    ax = axes[i]
    ax.errorbar(d_values, results_single[:, :, i].mean(axis=1), yerr=results_single[:, :, i].std(axis=1), label='Single-node pooled',    marker='o')
    ax.errorbar(d_values, results_bern[:, :, i].mean(axis=1),   yerr=results_bern[:, :, i].std(axis=1),   label='Bernoulli pooled (IID)', marker='^')
    ax.errorbar(d_values, results_obs[:, :, i].mean(axis=1),    yerr=results_obs[:, :, i].std(axis=1),    label='Observational',          marker='s', linestyle='--')
    ax.set_xlabel('Number of Nodes (d)')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=8)

plt.suptitle(f'ICALiNGAM: shift={shift}, n_trials={n_trials}', y=1.01)
plt.tight_layout()
plt.savefig('icalingam_metrics.png', bbox_inches='tight')
plt.show()
