from MakeDataset import ShiftSimulation
from castle.datasets import DAG
import numpy as np


def _pool_envs(ivs_samples_list):
    """Split environments into rich (full-rank, >n_nodes+1 samples) and discard the rest.
    Falls back to pooling all sparse envs into one context only when no rich envs exist.
    Returns (rich_samples, fallback_pooled_or_None).
    """
    n_nodes = ivs_samples_list[0].shape[1] if ivs_samples_list else 1
    threshold = n_nodes + 1
    rich, sparse = [], []
    for X in ivs_samples_list:
        cov = np.cov(X, rowvar=False)
        if X.shape[0] > threshold: #alg.matrix_rank(cov) == cov.shape[0]:
            rich.append(X)
        else:
            sparse.append(X)

    if rich:
        if sparse:
            print(f"Discarding {len(sparse)} sparse envs, keeping {len(rich)} rich envs.")
        return rich, None

    # Fallback: no rich envs at all — try pooling so we return something
    if sparse:
        pooled = np.vstack(sparse)
        cov_p = np.cov(pooled, rowvar=False)
        if pooled.shape[0] > threshold: # and np.linalg.matrix_rank(cov_p) == cov_p.shape[0]:
            print(f"No rich envs — pooling {len(sparse)} sparse envs as fallback (shape={pooled.shape}).")
            return [], pooled
        else:
            print(f"No rich envs and pooled sparse envs still degenerate — discarding all.")
    return [], None


def ut_igsp(obs_samples, ivs_samples_list, alpha_ci=0.05, alpha_inv=0.05, initial_permutations=None):
    '''Runs UT-IGSP with NO known intervention targets.
    Sparse/degenerate environments are discarded; pooling is used only as a
    last resort when no individual environment meets the richness threshold.

    Parameters
    ----------
    obs_samples : np.ndarray, shape (n_obs, n_nodes)
        Observational data.
    ivs_samples_list : list of np.ndarray
        List of interventional datasets, one array per environment.
    alpha_ci : float
        Significance level for the CI test.
    alpha_inv : float
        Significance level for the invariance test.

    Returns
    -------
    adj : np.ndarray, top_ord : list, est_targets_list : list
    '''
    from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
    from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
    from causaldag import unknown_target_igsp

    n_nodes = obs_samples.shape[1]
    rich, pooled = _pool_envs(ivs_samples_list)
    inv_list = rich + ([pooled] if pooled is not None else [])
    if not inv_list:
        print("No valid contexts — skipping UT-IGSP.")
        return None, None, None
    setting_list = [dict(known_interventions=[]) for _ in inv_list]

    print(f"Contexts used: {len(inv_list)}")
    rng = np.random.default_rng(0)
    jitter = 1e-5
    obs_j = obs_samples + rng.standard_normal(obs_samples.shape) * jitter
    inv_j = [X + rng.standard_normal(X.shape) * jitter for X in inv_list]

    obs_suffstat = gauss_ci_suffstat(obs_j)
    invariance_suffstat = gauss_invariance_suffstat(obs_j, inv_j)
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    est_dag, est_targets_list = unknown_target_igsp(setting_list, set(range(n_nodes)), ci_tester, invariance_tester)
    return est_dag.to_amat()[0], est_dag.to_amat()[1], est_targets_list


def igsp(obs_samples, ivs_samples_list, known_interventions_list, alpha_ci=0.05, alpha_inv=0.05):
    '''Runs true IGSP (Wang et al.) with known intervention targets.

    Uses causaldag.igsp (not unknown_target_igsp), which leverages the known
    targets directly in the search procedure.

    Parameters
    ----------
    obs_samples : np.ndarray, shape (n_obs, n_nodes)
        Observational data.
    ivs_samples_list : list of np.ndarray
        List of interventional datasets, one array per environment.
    known_interventions_list : list of list of int
        Known intervention targets for each environment (list of node indices).
    alpha_ci : float
        Significance level for the CI test.
    alpha_inv : float
        Significance level for the invariance test.

    Returns
    -------
    adj : np.ndarray, top_ord : list, est_targets_list : list
    '''
    from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
    from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
    from causaldag import igsp as causaldag_igsp

    n_nodes = obs_samples.shape[1]
    threshold = n_nodes + 1

    rich_samples, rich_targets, sparse_list = [], [], []
    for X, targets in zip(ivs_samples_list, known_interventions_list):
        cov = np.cov(X, rowvar=False)
        if X.shape[0] > threshold : #and np.linalg.matrix_rank(cov) == cov.shape[0]:
            rich_samples.append(X)
            rich_targets.append(targets)
        else:
            sparse_list.append(X)

    inv_samples_list = rich_samples.copy()
    setting_list = [dict(interventions=targets) for targets in rich_targets]

    if rich_samples:
        if sparse_list:
            print(f"Discarding {len(sparse_list)} sparse envs, keeping {len(rich_samples)} rich envs.")
    elif sparse_list:
        sparse_envs = np.vstack(sparse_list)
        cov_sparse = np.cov(sparse_envs, rowvar=False)
        if sparse_envs.shape[0] > threshold: # and np.linalg.matrix_rank(cov_sparse) == cov_sparse.shape[0]:
            inv_samples_list.append(sparse_envs)
            setting_list.append(dict(interventions=[]))
            print(f"No rich envs — pooling {len(sparse_list)} sparse envs as fallback (shape={sparse_envs.shape}).")
        else:
            print(f"No rich envs and pooled sparse envs still degenerate — discarding all.")

    print(f"Rich contexts: {len(rich_samples)}, Total contexts: {len(inv_samples_list)}")
    if not inv_samples_list:
        print("No valid contexts — skipping IGSP.")
        return None, None, None

    rng = np.random.default_rng(0)
    jitter = 0
    obs_j = obs_samples + rng.standard_normal(obs_samples.shape) * jitter
    inv_j = [X + rng.standard_normal(X.shape) * jitter for X in inv_samples_list]

    obs_suffstat = gauss_ci_suffstat(obs_j)
    invariance_suffstat = gauss_invariance_suffstat(obs_j, inv_j)
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    est_dag = causaldag_igsp(setting_list, set(range(n_nodes)), ci_tester, invariance_tester)
    return est_dag.to_amat()[0], est_dag.to_amat()[1], None


def ut_igsp_oracle(obs_samples, ivs_samples_list, known_interventions_list, alpha_ci=0.05, alpha_inv=0.05):
    '''Runs UT-IGSP with oracle (ground-truth) intervention targets.

    Identical to ut_igsp except that the true targets are passed as
    ``known_interventions`` in each setting dict, giving the algorithm oracle
    access to the intervention structure while still using the
    unknown_target_igsp search procedure.

    Parameters
    ----------
    obs_samples : np.ndarray, shape (n_obs, n_nodes)
    ivs_samples_list : list of np.ndarray
    known_interventions_list : list of list of int
        Ground-truth intervention targets for each environment.
    alpha_ci, alpha_inv : float

    Returns
    -------
    adj : np.ndarray, top_ord : list, est_targets_list : list
    '''
    from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
    from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
    from causaldag import unknown_target_igsp

    n_nodes = obs_samples.shape[1]
    threshold = n_nodes + 1

    rich_samples, rich_targets, sparse_list = [], [], []
    for X, targets in zip(ivs_samples_list, known_interventions_list):
        if X.shape[0] > threshold:
            rich_samples.append(X)
            rich_targets.append(targets)
        else:
            sparse_list.append((X, targets))

    inv_list = rich_samples
    setting_list = [dict(known_interventions=targets) for targets in rich_targets]

    if not rich_samples and sparse_list:
        pooled = np.vstack([x for x, _ in sparse_list])
        if pooled.shape[0] > threshold:
            inv_list = [pooled]
            # pool targets (union) for the fallback environment
            pooled_targets = sorted({t for _, tgts in sparse_list for t in tgts})
            setting_list = [dict(known_interventions=pooled_targets)]
            print(f"No rich envs — pooling {len(sparse_list)} sparse envs as fallback (shape={pooled.shape}).")
        else:
            print("No rich envs and pooled sparse envs still degenerate — discarding all.")

    if not inv_list:
        print("No valid contexts — skipping UT-IGSP (oracle).")
        return None, None, None

    print(f"Contexts used (oracle): {len(inv_list)}")
    rng = np.random.default_rng(0)
    jitter = 1e-5
    obs_j = obs_samples + rng.standard_normal(obs_samples.shape) * jitter
    inv_j = [X + rng.standard_normal(X.shape) * jitter for X in inv_list]

    obs_suffstat = gauss_ci_suffstat(obs_j)
    invariance_suffstat = gauss_invariance_suffstat(obs_j, inv_j)
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    est_dag, est_targets_list = unknown_target_igsp(setting_list, set(range(n_nodes)), ci_tester, invariance_tester)
    return est_dag.to_amat()[0], est_dag.to_amat()[1], est_targets_list


if __name__ == '__main__':
    # ------- Example Usage of UT-IGSP with my ShiftSimulation Module -------
    _n_nodes, _n_edges = 10, 10
    _seed = 12
    _p_intervention = 0.1
    _delta = 1

    _true_dag = DAG.erdos_renyi(n_nodes=_n_nodes, n_edges=_n_edges, seed=_seed, weight_range=[0.5, 2])
    _sim = ShiftSimulation(n_nodes=_n_nodes, n_edges=_n_edges, n_samples=int(1e4), n_observational=100,
                           weighted_random_dag=_true_dag, p_intervention=_p_intervention, delta=_delta, seed=_seed)
    _sim._generate_datasets()

    _interventions_list = _sim.datasets_interventional.keys()
    print("Interventions List (raw):", _interventions_list)
    _known_interventions_list = [
        [i for i, val in enumerate(intervention) if val == 1]
        for intervention in _interventions_list if _sim.datasets_interventional[intervention]['X'].shape[0] > 1
    ]
    print("Known Interventions List:", _known_interventions_list)
    _ivs_samples_list = [_sim.datasets_interventional[key]['X'] for key in _sim.datasets_interventional
                         if len(_sim.datasets_interventional[key]['X']) > 1]
    _obs_samples = _sim.dataset_observational.X
