# coding=utf-8
"""
Wrapper for DCDI (Differentiable Causal Discovery from Interventional Data).

Requires dcdi to be installed and importable:
  git clone https://github.com/slachapelle/dcdi
  pip install -e dcdi/   # or add dcdi/ to PYTHONPATH

Also requires CDT (Causal Discovery Toolbox) which dcdi imports internally:
  pip install cdt
"""

import contextlib
import io
import tempfile
import types
from pathlib import Path

import numpy as np


def run_dcdi(obs_sub, ivs_list, known_targets=None,
             model="DCDI-G",
             intervention_type="perfect",
             num_train_iter=30000,
             reg_coeff=0.5,
             random_seed=42):
    """Run DCDI (Differentiable Causal Discovery from Interventional Data).

    Writes data to a temporary directory in DCDI's expected file format, runs
    training via dcdi.main.main(), reads the learned DAG from the output, and
    cleans up the temporary files.

    Valid (intervention_type, known_targets) combinations
    (others raise ValueError inside DCDI):
      - "perfect"   + known_targets=None   → unknown-target discovery
      - "perfect"   + known_targets=[...]  → known hard/perfect interventions
      - "imperfect" + known_targets=[...]  → known soft/shift interventions
        (most faithful to ShiftSimulation data; imperfect+unknown not supported)

    Parameters
    ----------
    obs_sub : np.ndarray, shape (n_obs, n_nodes)
    ivs_list : list of np.ndarray
    known_targets : list of list of int, or None
        If None, runs in unknown-target mode (intervention_type is forced to
        "perfect", the only mode that supports unknown targets).
    model : str
        "DCDI-G" (Gaussian ANM, faster) or "DCDI-DSF" (deep sigmoidal flow).
    intervention_type : str
        "perfect" or "imperfect". Ignored and overridden to "perfect" when
        known_targets is None.
    num_train_iter : int
        Maximum augmented-Lagrangian iterations. Early stopping with
        train_patience=5 and stop_crit_win=100 typically terminates well
        before this limit for small graphs.
    reg_coeff : float
        L1 sparsity regularisation coefficient on the learned adjacency.
    random_seed : int

    Returns
    -------
    adj : np.ndarray of int, shape (n_nodes, n_nodes), or None on failure
        Binary adjacency matrix of the learned DAG (adj[i,j]=1 means i→j).
    """
    try:
        from dcdi.main import main as dcdi_main
    except ImportError:
        raise ImportError(
            "dcdi is not installed. Clone https://github.com/slachapelle/dcdi "
            "and add it to your PYTHONPATH."
        )

    has_known = known_targets is not None
    if not has_known:
        intervention_type = "perfect"   # the only mode that supports unknown targets
    intervention_knowledge = "known" if has_known else "unknown"

    n_nodes = obs_sub.shape[1]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            exp_dir  = Path(tmpdir) / "exp"
            data_dir.mkdir()
            exp_dir.mkdir()

            # --- build data arrays ---
            # Stack obs first, then each interventional environment
            all_data = np.vstack([obs_sub] + ivs_list)
            n_obs    = obs_sub.shape[0]
            n_total  = all_data.shape[0]

            # Integer regime label per row: 0 = obs, 1,2,... = each interv env
            regimes = np.zeros(n_total, dtype=int)
            cumsum  = n_obs
            for i, ivs in enumerate(ivs_list):
                regimes[cumsum:cumsum + len(ivs)] = i + 1
                cumsum += len(ivs)

            # Per-row intervention targets (comma-separated node indices).
            # Observational rows → empty string (no intervened variables).
            if has_known:
                interv_rows = [''] * n_obs
                for ivs, targets in zip(ivs_list, known_targets):
                    target_str = ','.join(str(t) for t in targets)
                    interv_rows.extend([target_str] * len(ivs))
            else:
                interv_rows = [''] * n_total   # unknown mode: leave all empty

            # --- write DCDI data files ---
            np.save(data_dir / 'data_interv1.npy', all_data.astype(np.float64))
            # DAG1.npy is required by DataManagerFile (for gt-adjacency logging);
            # we don't know the true graph, so we write zeros.
            np.save(data_dir / 'DAG1.npy', np.zeros((n_nodes, n_nodes), dtype=int))
            (data_dir / 'regime1.csv').write_text('\n'.join(str(r) for r in regimes))
            (data_dir / 'intervention1.csv').write_text('\n'.join(interv_rows))

            # --- build opt namespace ---
            opt = types.SimpleNamespace(
                # reproducibility
                random_seed=random_seed,
                # paths
                exp_path=str(exp_dir),
                data_path=str(data_dir),
                i_dataset=1,
                # hardware
                gpu=False,
                float=True,
                # data
                train_samples=0.8,
                test_samples=None,
                normalize_data=False,
                dcd=False,
                regimes_to_ignore=None,
                # intervention settings
                intervention=True,
                intervention_type=intervention_type,
                intervention_knowledge=intervention_knowledge,
                # model architecture
                model=model,
                num_vars=n_nodes,
                num_layers=2,
                hid_dim=16,
                nonlin="leaky-relu",
                flow_num_layers=2,   # only used by DCDI-DSF
                flow_hid_dim=16,     # only used by DCDI-DSF
                # run mode
                train=True,
                retrain=False,
                dag_for_retrain=None,
                test_on_new_regimes=False,
                # optimisation / augmented Lagrangian
                num_train_iter=num_train_iter,
                train_patience=5,
                train_patience_post=5,
                train_batch_size=64,
                reg_coeff=reg_coeff,
                coeff_interv_sparsity=1e-8,
                stop_crit_win=100,
                omega_gamma=1e-4,
                omega_mu=0.9,
                h_threshold=1e-8,
                mu_init=1e-8,
                gamma_init=0.0,
                mu_mult_factor=2.0,
                optimizer="rmsprop",
                lr=1e-3,
                lr_reinit=None,      # set equal to lr inside dcdi_main
                lr_schedule=None,
                # logging (disable plotting to avoid matplotlib overhead)
                plot_freq=num_train_iter + 1,
                plot_density=False,
                no_w_adjs_log=True,
            )

            # Run DCDI, suppressing its verbose stdout
            with contextlib.redirect_stdout(io.StringIO()):
                dcdi_main(opt,
                          metrics_callback=lambda *a, **kw: None,
                          plotting_callback=None)

            dag_path = exp_dir / 'DAG.npy'
            if not dag_path.exists():
                print("  [DCDI error] DAG.npy not found after training")
                return None
            return np.load(dag_path).astype(int)

    except Exception as e:
        print(f"  [DCDI error] {type(e).__name__}: {e}")
        return None