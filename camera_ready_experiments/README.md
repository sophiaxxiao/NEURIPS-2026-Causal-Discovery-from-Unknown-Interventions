# Camera-ready experiments

This folder holds the experiments added for the camera-ready version: the appendix "Comparison With Labels-Only Methods" (including its "Noisy Labels" subsection), a check on the observational share reported in the same appendix, and a corrected rerun of Ablation B in "Robustness Ablations". The scripts reuse the data generators and baseline wrappers in `../synthetic_experiments/src` and the Sachs loader in `../real_data_experiments/sachs_protein/src`.

The `results/` folder already holds every raw result behind the paper. Rerunning the scripts regenerates it.

## Setup

Use Python 3.11, because `pywhy-graphs` requires a version below 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`dodiscover`, which provides Ψ-FCI, is not on PyPI, so `requirements.txt` installs it from GitHub at the commit we used.

## What reproduces what

| Paper | Script | Output |
|---|---|---|
| Single-node comparison figure ("Comparison With Labels-Only Methods", synthetic part) | `run_jci_ladder.py --n-dags 100 --n-jobs 8` | `results/jci_ladder_*.csv` |
| Held-fixed observational share (same subsection, results paragraph) | `run_obs_fraction_check.py --n-dags 100 --n-jobs 8` | `results/obs_fraction_check_*.csv` |
| Noisy-labels figure (same appendix, "Noisy Labels" subsection) | `run_label_noise.py --n-dags 100 --n-jobs 8` | `results/label_noise_*.csv` |
| Sachs labels-only figure and table (JCI, Ψ-FCI, GIES, IGSP, ICA-LiNGAM) | `run_sachs_sweep.sh` | `results/sachs/*.csv` |
| "Robustness Ablations" table, Ablation B (heterogeneous strengths) | `PYTHONPATH=patches python run_ablation_B_variance.py --n-jobs 8` | `results/ablation_B_var/results_combined.json` |
| The three figures above and the Sachs table rows | `make_rebuttal_figures.py` | `figures/` |

Sachs inputs: `data/make_sachs_fig4.py` rebuilds `data/sachs_fig4/` from the original `.xls` files with the loader used for the main-text Sachs figure. That means raw values, 3-SD outlier removal per condition, and 6,456 samples across 9 conditions.

`python run_sachs.py --smoke-test` checks the Ψ-FCI setup on a 4-node example in a few seconds.

## Implementation notes

- **JCI (labels only)** is `jci_pc.py`. It runs PC with Fisher-z tests on the data augmented with one indicator per non-reference environment. It enforces the JCI background knowledge: contexts are exogenous and mutually adjacent, and every test between system variables conditions on all context variables. JCI with known targets uses the existing `run_jci_pc` wrapper in `synthetic_experiments/src/method_wrappers.py`.
- **Ψ-FCI** uses `dodiscover.PsiFCI`, with one F-node per non-reference environment. `run_sachs.py` swaps in two of its own tests:
  - **Fisher-z CI test.** The dodiscover Fisher-z test at the pinned commit can test the wrong pair of variables. On the smoke-test graph it returns p ≈ 0 for a true independence.
  - **Linear conditional-discrepancy test.** It is a partial-correlation test of the environment indicator against the target, given the conditioning set. It matches the mean-shift model. The default kernel test fails on an empty conditioning set and is too slow at Sachs sample sizes.
- **SHD.** `shd_castle` is gcastle's `MetricsDAG` SHD, the convention of the main-text synthetic and Sachs figures. It does not penalize an edge left undirected when the skeleton is correct. `shd` (from `causal_utils.py`) is strict: every edge without a correct definite orientation counts as an error. For Ψ-FCI, bidirected edges and edges with a circle endpoint are mapped to undirected edges.
- **Sachs sampling** uses the scheme of the notebook behind the main-text Sachs figure (`sachs_intervention_test.ipynb`): n samples drawn uniformly without replacement from the pool. We use 30 draws per n (the notebook uses 100), and the full pool for the last point.
- **Ablation B fix.** In `synthetic_experiments/ablation`, the shared baseline and Ablations A, C and D scale each shift by the target's observational variance (`compute_per_node_delta_from_observational`). Ablation B's τ > 0 conditions instead went through `compute_per_node_delta_vector`, which scales by the standard deviation. `run_ablation_B_variance.py` reruns Ablation B with variance scaling and otherwise leaves the original runner untouched. The patch lives in `patches/sitecustomize.py`, and `PYTHONPATH=patches` is what makes it apply inside the joblib worker processes. The camera-ready table reports the corrected rows.
