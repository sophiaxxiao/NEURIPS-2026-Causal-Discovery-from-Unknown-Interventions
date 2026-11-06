# Ablation to test the effect of unequal environment sizes on the performance of each method. 

import sys, os, joblib, pickle, datetime, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import datetime
import os
import pickle
import subprocess
import sys
from itertools import product

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kendalltau

from castle.algorithms import DirectLiNGAM, ICALiNGAM
from castle.datasets import DAG
from castle.metrics import MetricsDAG

from MakeDataset import ShiftSimulation
from helpers import compute_graph_metrics
from method_wrappers import build_env_labels, split_subsample, run_ut_igsp, run_ut_igsp_oracle, run_igsp, run_jci_pc, run_gies

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

seeds           = np.arange(0, 100).tolist()
deltas          = np.arange(0, 2, step=0.25).tolist()
p_interventions = np.round(np.arange(0, 1, step=0.1), 1).tolist()
n_nodes, n_edges = 10, 10
sample_sizes    = [32, 64, 128, 200, 300, 400, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000]

N_JOBS = 8   # safe for M4 MacBook Air (10 cores); leave 2 free for the OS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_shd(adj, true_dag):
    if adj is None:
        return np.nan
    return MetricsDAG(adj, true_dag).metrics['shd']


def _safe_metrics(adj, true_dag):
    if adj is None:
        return None
    return compute_graph_metrics(adj, true_dag)


def _git_hash():
# Return the short git hash of the current commit, or 'unknown' if it can't be determined.
# Track exactly which code version produced the results, but don't fail if git isn't available
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def _save(fname, metadata, results, dirname=None):
    """Atomic checkpoint write — never leaves a corrupt file on crash."""
    path = os.path.join(dirname, fname) if dirname is not None else fname
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump({'metadata': metadata, 'results': results}, f)
    os.replace(tmp, path)


def _make_fname(p_intervention):
    return (
        f"results_phase_shift"
        f"_n{n_nodes}e{n_edges}"
        f"_p{p_intervention:.2f}"
        f"_seeds{seeds[0]}-{seeds[-1]}"
        f"_d{len(deltas)}deltas"
        f"_ss{sample_sizes[0]}-{sample_sizes[-1]}"
        f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f".pkl"
    )


# ---------------------------------------------------------------------------
# Unit of work: one (p_intervention, delta, seed) triple
# Returns everything needed to slot into results[delta][seed].
# Each call is fully self-contained — safe to run in a worker process.
# ---------------------------------------------------------------------------

def run_trial(p_intervention, delta, seed):
    true_dag = DAG.erdos_renyi(
        n_nodes=n_nodes, n_edges=n_edges, seed=seed, weight_range=[0.5, 2]
    )
    sim = ShiftSimulation(
        n_nodes=n_nodes, n_edges=n_edges, n_samples=int(1e6),
        n_observational=100, weighted_random_dag=true_dag,
        p_intervention=p_intervention, delta=delta, seed=seed,
    )
    sim._generate_datasets()

    obs_ds_array = sim.dataset_observational.X
    interv_ds    = sim.datasets_interventional
    all_data_arr = np.vstack([interv_ds[key]['X'] for key in interv_ds.keys()])
    env_labels   = build_env_labels(sim)
    perm         = np.random.permutation(all_data_arr.shape[0])

    seed_results = {}

    for ss in sample_sizes:
        interv_sub = all_data_arr[perm[:ss], :]
        subsample  = np.vstack([obs_ds_array, interv_sub])
        seed_results[ss] = {}

        obs_sub, ivs_list, known_targets = split_subsample(
            interv_sub, env_labels[perm[:ss]], sim
        )

        # Direct LiNGAM
        try:
            dlingam = DirectLiNGAM()
            dlingam.learn(subsample)
            dlingam_adj = dlingam.causal_matrix
            dlingam_wadj = dlingam.weight_causal_matrix
        except Exception:
            dlingam_adj = dlingam_wadj = None
        seed_results[ss]['shd_dlingam']          = _safe_shd(dlingam_adj, true_dag)
        seed_results[ss]['est_dag_dlingam']      = dlingam_adj
        seed_results[ss]['weighted_dag_dlingam'] = dlingam_wadj
        seed_results[ss]['dag_metrics_dlingam']  = _safe_metrics(dlingam_adj, true_dag)

        # ICA-LiNGAM
        try:
            icalingam = ICALiNGAM()
            icalingam.learn(subsample)
            icalingam_adj = icalingam.causal_matrix
            icalingam_wadj = icalingam.weight_causal_matrix
        except Exception:
            icalingam_adj = icalingam_wadj = None
        seed_results[ss]['shd_icalingam']          = _safe_shd(icalingam_adj, true_dag)
        seed_results[ss]['est_dag_icalingam']      = icalingam_adj
        seed_results[ss]['weighted_dag_icalingam'] = icalingam_wadj
        seed_results[ss]['dag_metrics_icalingam']  = _safe_metrics(icalingam_adj, true_dag)

        # UT-IGSP (unknown targets)
        adj_utigsp = run_ut_igsp(obs_sub, ivs_list)
        seed_results[ss]['shd_utigsp']         = _safe_shd(adj_utigsp, true_dag)
        seed_results[ss]['est_dag_utigsp']     = adj_utigsp
        seed_results[ss]['dag_metrics_utigsp'] = _safe_metrics(adj_utigsp, true_dag)

        # IGSP (true known-target IGSP via causaldag.igsp)
        adj_igsp = run_igsp(obs_sub, ivs_list, known_targets)
        seed_results[ss]['shd_igsp']         = _safe_shd(adj_igsp, true_dag)
        seed_results[ss]['est_dag_igsp']     = adj_igsp
        seed_results[ss]['dag_metrics_igsp'] = _safe_metrics(adj_igsp, true_dag)

        # UT-IGSP with oracle targets (unknown_target_igsp + ground-truth known_interventions)
        adj_ut_igsp_oracle = run_ut_igsp_oracle(obs_sub, ivs_list, known_targets)
        seed_results[ss]['shd_ut_igsp_oracle']         = _safe_shd(adj_ut_igsp_oracle, true_dag)
        seed_results[ss]['est_dag_ut_igsp_oracle']     = adj_ut_igsp_oracle
        seed_results[ss]['dag_metrics_ut_igsp_oracle'] = _safe_metrics(adj_ut_igsp_oracle, true_dag)

        # JCI-PC
        adj_jcipc = run_jci_pc(obs_sub, ivs_list, known_targets)
        seed_results[ss]['shd_jcipc']         = _safe_shd(adj_jcipc, true_dag)
        seed_results[ss]['est_dag_jcipc']     = adj_jcipc
        seed_results[ss]['dag_metrics_jcipc'] = _safe_metrics(adj_jcipc, true_dag)

        # GIES (known targets)
        adj_gies = run_gies(obs_sub, ivs_list, known_targets)
        seed_results[ss]['shd_gies']         = _safe_shd(adj_gies, true_dag)
        seed_results[ss]['est_dag_gies']     = adj_gies
        seed_results[ss]['dag_metrics_gies'] = _safe_metrics(adj_gies, true_dag)

    return p_intervention, delta, seed, seed_results


# ---------------------------------------------------------------------------
# Outer loop: one pickle per p_intervention, parallel over (delta, seed)
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(
    "/dartfs-hpc/scratch/sophiax", "results",
    datetime.datetime.now().strftime('%Y-%m-%d')
) # Redirect results to the scratch space on the cluster, tagged by date. Each run gets its own timestamped file within that day's folder.
os.makedirs(RESULTS_DIR, exist_ok=True)

for p_intervention in p_interventions:

    print(f"\n{'#'*60}")
    print(f"  p_intervention = {p_intervention:.2f}")
    print(f"{'#'*60}")

    fname = _make_fname(p_intervention)

    metadata = {
        'description':     'Phase-shift experiment: mean SHD vs sample size and non-Gaussianity (delta)',
        'timestamp':       datetime.datetime.now().isoformat(timespec='seconds'),
        'git_hash':        _git_hash(),
        'python_version':  sys.version,
        'n_nodes':         n_nodes,
        'n_edges':         n_edges,
        'n_samples_pool':  int(1e8),
        'n_observational': 100,
        'p_intervention':  p_intervention,
        'weight_range':    [0.5, 2],
        'seeds':           seeds,
        'deltas':          deltas,
        'sample_sizes':    sample_sizes,
        'methods':         ['dlingam', 'icalingam', 'utigsp', 'igsp', 'ut_igsp_oracle', 'jcipc', 'gies'],
        'alpha_ci':        0.05,
        'alpha_inv':       0.05,
        'results_key':     'results[delta][seed][sample_size][metric_key]',
        'n_jobs':          N_JOBS,
    }

    results = {delta: {} for delta in deltas}

    triples = list(product([p_intervention], deltas, seeds))
    total   = len(triples)
    completed = 0

    # return_as="generator" lets us checkpoint as each job finishes
    # rather than waiting for the entire batch to complete.
    job_iter = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(run_trial)(p, d, s) for p, d, s in triples
    )

    for p_int, delta, seed, seed_results in job_iter:
        results[delta][seed] = seed_results
        completed += 1
        _save(fname, metadata, results, dirname=RESULTS_DIR)
        print(f"  [{completed}/{total}] p={p_int:.2f} | delta={delta:.2f} | seed={seed}  → saved")

    print(f"\nDone with p_intervention={p_intervention:.2f}. Results saved to {fname}")

print("\nAll p_intervention values complete.")
print("Load a file with:")
print("  data    = pickle.load(open(fname, 'rb'))")
print("  meta    = data['metadata']")
print("  results = data['results']")
