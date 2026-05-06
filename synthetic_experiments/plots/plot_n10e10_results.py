"""
plot_n10e10_results.py
----------------------
Plots for n=10, e=10 phase-shift experiment at p_intervention=0.5.

Figures (saved to n10e10_plots/)
-------
1  shd_vs_n         – mean SHD ± 1 SD vs sample size, one subplot per delta
2  kendall_tau_vs_n – mean Kendall's τ ± 1 SD vs sample size, one subplot per delta
3  sid_vs_n         – mean SID ± 1 SD vs sample size, one subplot per delta
"""
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau as scipy_kendalltau
from castle.datasets import DAG

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = '../results/2026-04-20'
P_TARGET    = 0.5
N_NODES     = 10
N_EDGES     = 10

METHODS = ['dlingam', 'icalingam', 'utigsp', 'igsp', 'ut_igsp_oracle', 'jcipc', 'gies']
METHOD_LABELS = {
    'dlingam':        'DirectLiNGAM',
    'icalingam':      'ICA-LiNGAM',
    'utigsp':         'UT-IGSP',
    'igsp':           'IGSP',
    'ut_igsp_oracle': 'UT-IGSP (oracle)',
    'jcipc':          'JCI-PC',
    'gies':           'GIES',
}
METHOD_COLORS = {
    'dlingam':        '#1f77b4',
    'icalingam':      '#ff7f0e',
    'utigsp':         '#2ca02c',
    'igsp':           '#d62728',
    'ut_igsp_oracle': '#8c564b',
    'jcipc':          '#9467bd',
    'gies':           '#17becf',
}

POSTER_RC = {
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family':     'sans-serif',
}

SUMMARY_CSV = 'n10e10_p05_summary.csv'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _topo_rank(adj):
    """
    Return a rank array for the topological ordering of adj.
    rank[node] = position in topological order (0 = first source).
    Returns None if the graph contains a cycle.
    """
    try:
        bin_adj = (np.array(adj, dtype=float) != 0).astype(int)
        order   = list(nx.topological_sort(nx.DiGraph(bin_adj)))
        rank    = np.zeros(len(order), dtype=int)
        for pos, node in enumerate(order):
            rank[node] = pos
        return rank
    except nx.NetworkXUnfeasible:
        return None


def _kendall_tau(est_adj, true_adj):
    """
    Kendall's tau between topological orderings of true and estimated DAGs.
    Returns NaN if est_adj is None or the estimated graph is cyclic.
    """
    if est_adj is None:
        return np.nan
    true_rank = _topo_rank(true_adj)
    est_rank  = _topo_rank(est_adj)
    if true_rank is None or est_rank is None:
        return np.nan
    tau, _ = scipy_kendalltau(true_rank, est_rank)
    return tau


def _safe_float(v):
    if v is None:
        return np.nan
    try:
        f = float(v)
        return np.nan if np.isnan(f) else f
    except (TypeError, ValueError):
        return np.nan


def save_fig(fig, name):
    os.makedirs('n10e10_plots', exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join('n10e10_plots', f'n10e10_{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=200 if ext == 'png' else None)
        print(f'  Saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build / load summary DataFrame
# ---------------------------------------------------------------------------
def build_summary():
    pattern = os.path.join(RESULTS_DIR,
                           f'results_phase_shift_n10e10_p{P_TARGET:.2f}_*.pkl')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No pkl file found matching {pattern}')
    fpath = files[-1]
    print(f'Loading {fpath}')
    data = pickle.load(open(fpath, 'rb'))
    meta = data['metadata']
    res  = data['results']
    deltas  = meta['deltas']
    ss_list = meta['sample_sizes']
    seeds   = meta['seeds']

    # Regenerate true DAG adjacency matrices once per seed (deterministic).
    # True DAG is independent of delta; same seed → same graph.
    print('Regenerating true DAGs from seeds...')
    true_adjs = {}
    for seed in seeds:
        true_adjs[seed] = np.array(
            DAG.erdos_renyi(n_nodes=N_NODES, n_edges=N_EDGES,
                            seed=seed, weight_range=[0.5, 2])
        )

    rows = []
    for delta in deltas:
        print(f'  delta = {delta:.2f}')
        for ss in ss_list:
            for method in METHODS:
                shd_vals, sid_vals, tau_vals = [], [], []
                for seed in seeds:
                    try:
                        entry   = res[delta][seed][ss]
                        metrics = entry.get(f'dag_metrics_{method}')
                        est_adj = entry.get(f'est_dag_{method}')

                        shd = _safe_float(metrics['shd']       if metrics else None)
                        sid = _safe_float(metrics['sid_lower'] if metrics else None)
                        tau = _kendall_tau(est_adj, true_adjs[seed])
                    except (KeyError, TypeError):
                        shd = sid = tau = np.nan

                    shd_vals.append(shd)
                    sid_vals.append(sid)
                    tau_vals.append(tau)

                def _agg(vals):
                    arr = np.array(vals, dtype=float)
                    v   = arr[~np.isnan(arr)]
                    return (
                        np.mean(v)        if len(v) else np.nan,
                        np.std(v, ddof=1) if len(v) > 1 else 0.0,
                        len(v),
                    )

                m_shd, s_shd, n_shd = _agg(shd_vals)
                m_sid, s_sid, n_sid = _agg(sid_vals)
                m_tau, s_tau, n_tau = _agg(tau_vals)

                rows.append(dict(
                    delta=delta, ss=ss, method=method,
                    mean_shd=m_shd, std_shd=s_shd, n_shd=n_shd,
                    mean_sid=m_sid, std_sid=s_sid, n_sid=n_sid,
                    mean_tau=m_tau, std_tau=s_tau, n_tau=n_tau,
                ))

    return pd.DataFrame(rows)


if os.path.exists(SUMMARY_CSV):
    print(f'Loading cached summary from {SUMMARY_CSV}')
    df = pd.read_csv(SUMMARY_CSV)
else:
    print('Building summary from pkl file...')
    df = build_summary()
    df.to_csv(SUMMARY_CSV, index=False)
    print(f'Saved to {SUMMARY_CSV}')

deltas_all = sorted(df.delta.unique())
ss_all     = sorted(df.ss.unique())
print(f'{len(deltas_all)} deltas: {deltas_all}')
print(f'{len(ss_all)} sample sizes: {ss_all}')


# ---------------------------------------------------------------------------
# Generic figure: one subplot per delta, mean ± 1 SD vs sample size
# ---------------------------------------------------------------------------
def fig_metric_vs_ss(metric_col, std_col, title, ylabel, name, ylim=None):
    """
    Creates a figure with one subplot per delta value.
    Each subplot shows mean ± 1 SD vs sample size for every method.
    """
    n_deltas = len(deltas_all)
    n_cols   = 4
    n_rows   = int(np.ceil(n_deltas / n_cols))

    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4.0 * n_rows),
            sharex=True, sharey=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f'{title}  (n=10, e=10, p_intervention={P_TARGET})',
            fontsize=15, fontweight='bold',
        )
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for di, d in enumerate(deltas_all):
            ax   = axes_flat[di]
            dsub = df[df.delta == d]

            for method in METHODS:
                msub  = dsub[dsub.method == method].sort_values('ss')
                if msub.empty:
                    continue
                means = msub[metric_col].values
                stds  = msub[std_col].values
                xs    = msub['ss'].values

                ax.errorbar(
                    xs, means, yerr=stds,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    fmt='-o', markersize=2,
                    lw=1.8, capsize=2, alpha=0.85,
                )

            ax.set_xscale('log')
            ax.grid(True, alpha=0.25)
            ax.set_title(f'δ = {d:.2f}', fontweight='bold')

            if ylim is not None:
                ax.set_ylim(ylim)

            row = di // n_cols
            col = di % n_cols
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == n_rows - 1:
                ax.set_xlabel('Sample size (n)')
            if di == 0:
                ax.legend(fontsize=7, loc='upper right')

        for ax in axes_flat[n_deltas:]:
            ax.set_visible(False)

    save_fig(fig, name)


# ---------------------------------------------------------------------------
# Run all figures
# ---------------------------------------------------------------------------
print('\nFigure 1: Mean SHD vs n...')
fig_metric_vs_ss(
    'mean_shd', 'std_shd',
    title='Mean SHD vs Sample Size',
    ylabel='Mean SHD',
    name='1_shd_vs_n',
)

print('\nFigure 2: Kendall tau vs n...')
fig_metric_vs_ss(
    'mean_tau', 'std_tau',
    title="Mean Kendall's τ (topological ordering) vs Sample Size",
    ylabel="Mean Kendall's τ",
    name='2_kendall_tau_vs_n',
    ylim=(-1.05, 1.05),
)

print('\nFigure 3: Mean SID vs n...')
fig_metric_vs_ss(
    'mean_sid', 'std_sid',
    title='Mean SID vs Sample Size',
    ylabel='Mean SID',
    name='3_sid_vs_n',
)

print('\nAll figures done.')