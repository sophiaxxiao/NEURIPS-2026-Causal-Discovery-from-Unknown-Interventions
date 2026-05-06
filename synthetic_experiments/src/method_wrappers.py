# coding=utf-8
"""
Wrapper functions for label-aware causal discovery methods (UT-IGSP, IGSP with known targets, JCI-PC, GIES).

Design
------
The existing experiment loop builds `all_data_arr` by stacking environments,
draws a single random permutation, and slices `all_data_arr[perm[:ss]]` for
each sample size ss.  To keep ALL methods on exactly the same data, the
wrappers here reuse that same permuted slice:

  1. Build `all_data_arr` from INTERVENTIONAL data only (do NOT append obs).
     Call `build_env_labels(sim)` once per seed — it returns an integer label
     array of the same length, where 0, 1, ... mark each interventional env.

         all_data_list = [interv_ds[key]['X'] for key in interv_ds.keys()]
         all_data_arr  = np.vstack(all_data_list)   # interventional only
         env_labels    = build_env_labels(sim)

  2. For each sample size ss, build the subsample from the interventional
     chunk and always prepend the full obs data (100 samples) so that ALL
     methods — LiNGAM included — see the same obs rows:

         interv_sub = all_data_arr[perm[:ss], :]
         subsample  = np.vstack([obs_ds_array, interv_sub])

  3. Call `split_subsample` to recover per-environment arrays for the
     environment-aware methods (obs is always taken from sim in full):

         obs_sub, ivs_list, known_targets = split_subsample(
             interv_sub, env_labels[perm[:ss]], sim
         )

  4. Pass those arrays directly to the thin wrappers.

"""

import numpy as np
from castle.algorithms import PC
from castle.common import priori_knowledge as pk_module
from utigsp import ut_igsp, igsp as _igsp, ut_igsp_oracle as _ut_igsp_oracle
_OBS_LABEL = -1


# ---------------------------------------------------------------------------
# Data-prep helpers
# ---------------------------------------------------------------------------

def build_env_labels(sim):
    """Build an integer label array parallel to the experiment's all_data_arr.

    Labels:  0  →  first interventional environment
             1  →  second interventional environment
             ...

    The order matches the notebook's construction (obs is kept separate):
        all_data_list = [interv_ds[key]['X'] for key in interv_ds.keys()]
        all_data_arr = np.vstack(all_data_list)   # interventional only

    Parameters
    ----------
    sim : ShiftSimulation

    Returns
    -------
    env_labels : np.ndarray, shape (N_interventional,), dtype int
    """
    interv_ds = sim.datasets_interventional
    return np.concatenate([
        np.full(interv_ds[key]['X'].shape[0], i)
        for i, key in enumerate(interv_ds.keys())
    ])

def split_subsample(interv_chunk, labels_chunk, sim):
    """Split a permuted interventional chunk into per-environment arrays.

    The observational data is always taken from sim in full (it is tiny, ~100
    samples, and must be present for IGSP/JCI-PC to function at any sample size).

    Parameters
    ----------
    interv_chunk : np.ndarray, shape (ss, n_nodes)
        A slice of the interventional-only all_data_arr after permutation,
        e.g. all_data_arr[perm[:ss]].
    labels_chunk : np.ndarray, shape (ss,)
        Corresponding slice of env_labels, e.g. env_labels[perm[:ss]].
    sim : ShiftSimulation
        Used to retrieve obs data and look up intervention patterns.

    Returns
    -------
    obs_sub : np.ndarray   — always the full observational dataset from sim
    ivs_list : list of np.ndarray
        One array per interventional environment present in this chunk.
    known_targets : list of list of int
        Intervention target node indices aligned with ivs_list.
    """
    obs_parts = [sim.dataset_observational.X]  # start with the real obs data

    patterns = list(sim.datasets_interventional.keys())
    ivs_list, known_targets = [], []
    for i, pattern in enumerate(patterns):
        rows = interv_chunk[labels_chunk == i]
        if rows.shape[0] == 0:
            continue
        if not any(pattern):  # all-zeros: no intervention, same distribution as obs
            obs_parts.append(rows)
        else:
            ivs_list.append(rows)
            known_targets.append([j for j, v in enumerate(pattern) if v == 1])

    obs_sub = np.vstack(obs_parts)
    return obs_sub, ivs_list, known_targets


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def run_ut_igsp(obs_sub, ivs_list, alpha_ci=0.05, alpha_inv=0.05):
    """Run UT-IGSP with unknown intervention targets.

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    alpha_ci, alpha_inv : float

    Returns
    -------
    adj : np.ndarray, shape (n_nodes, n_nodes)
    """
    try:
        adj, _, _ = ut_igsp(obs_sub, ivs_list, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    except Exception as e:
        print(f"  [UT-IGSP error] {type(e).__name__}: {e}")
        adj = None
    return adj

def run_ut_igsp_oracle(obs_sub, ivs_list, known_targets, alpha_ci=0.05, alpha_inv=0.05):
    """Run UT-IGSP with oracle (ground-truth) intervention targets.

    Uses unknown_target_igsp with the true targets supplied as
    ``known_interventions``, giving the algorithm oracle target access.

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    known_targets : list of list of int
    alpha_ci, alpha_inv : float

    Returns
    -------
    adj : np.ndarray, shape (n_nodes, n_nodes)
    """
    try:
        adj, _, _ = _ut_igsp_oracle(obs_sub, ivs_list, known_targets, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    except Exception as e:
        print(f"  [UT-IGSP (oracle) error] {type(e).__name__}: {e}")
        adj = None
    return adj

def run_igsp(obs_sub, ivs_list, known_targets, alpha_ci=0.05, alpha_inv=0.05):
    """Run true IGSP (Wang et al.) with known intervention targets.

    Uses causaldag.igsp, exploiting the
    known targets directly in the search procedure.

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    known_targets : list of list of int
    alpha_ci, alpha_inv : float

    Returns
    -------
    adj : np.ndarray, shape (n_nodes, n_nodes)
    """
    try:
        adj, _, _ = _igsp(obs_sub, ivs_list, known_targets, alpha_ci=alpha_ci, alpha_inv=alpha_inv)
    except Exception as e:
        print(f"  [IGSP error] {type(e).__name__}: {e}")
        adj = None
    return adj


def run_jci_pc(obs_sub, ivs_list, known_targets):
    """Run JCI-PC using Castle's PC with JCI prior-knowledge constraints.

    Augments data with binary context variables (one per system node) and
    encodes JCI constraints:
      - No edge from any system variable into a context variable.
      - No edge between any two context variables.
      - Required edge C_i -> X_i for each node i that is actually targeted
        in the current subsample.

    Fisher Z is used as the CI test (Castle's default). Although the context
    variables are binary, Fisher Z tests for zero partial correlation, which
    is asymptotically valid even for non-Gaussian data and is the standard
    choice in JCI-PC (Mooij et al. 2020). The chi2/g2 alternatives in Castle
    are designed for purely discrete data and are worse here.

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    known_targets : list of list of int

    Returns
    -------
    adj : np.ndarray, shape (n_nodes, n_nodes)
    """
    n_nodes = obs_sub.shape[1]

    # Nodes that are actually targeted at least once in this subsample.
    # We only require C_i -> X_i for these: a node never targeted has its
    # context column identically 0 (zero variance), so forcing its edge
    # corrupts the skeleton-phase CI tests for X_i.
    ever_targeted = sorted({t for targets in known_targets for t in targets})

    def _context_row(targets):
        row = np.zeros(n_nodes, dtype=float)
        for t in targets:
            row[t] = 1.0
        return row

    aug_obs = np.hstack([obs_sub, np.zeros((obs_sub.shape[0], n_nodes))])
    aug_ivs = [
        np.hstack([X, np.tile(_context_row(targets), (X.shape[0], 1))])
        for X, targets in zip(ivs_list, known_targets)
    ]
    all_data = np.vstack([aug_obs] + aug_ivs)

    pk = pk_module.PrioriKnowledge(2 * n_nodes)
    pk.add_forbidden_edges(
        [(i, j) for i in range(n_nodes) for j in range(n_nodes, 2 * n_nodes)] +
        [(i, j) for i in range(n_nodes, 2 * n_nodes)
                for j in range(n_nodes, 2 * n_nodes) if i != j]
    )
    pk.add_required_edges([(i + n_nodes, i) for i in ever_targeted])

    pc = PC(priori_knowledge=pk)
    try:
        pc.learn(all_data)
    except Exception:
        return None

    return pc.causal_matrix[:n_nodes, :n_nodes].astype(int)


def run_gies(obs_sub, ivs_list, known_targets):
    """Run GIES with the Gaussian BIC score (known intervention targets).

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    known_targets : list of list of int

    Returns
    -------
    adj : np.ndarray, shape (n_nodes, n_nodes)
    """
    import gies
    # Prepare data. This gies wrapper expects a list of arrays, one per environment, with the observational data first and the interventional datasets following. The interventions are specified as a parallel list of target lists, with an empty list for the observational data.
    data          = [obs_sub] + ivs_list   # obs first, then each interventional env
    interventions = [[]] + known_targets   # [] for obs; target lists for each env
    try:
        estimate, _ = gies.fit_bic(data, interventions) # Use the BIC score function (you can also try the Gaussian L0 score by calling gies.scores.GaussIntL0Pen instead)
        return estimate.astype(int) # GIES only estimates the adjacency matrix. 
    except Exception:
        return None
