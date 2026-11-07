"""Shared utilities for the camera-ready JCI and Psi-FCI experiments.

Graph convention used everywhere in this package:
    adjacency matrix G (d x d, 0/1), G[i, j] = 1 means an edge i -> j.
    A CPDAG/PAG summary may also contain undirected edges, encoded as
    G[i, j] = G[j, i] = 1.

Data generation follows Section 5.1 of the paper: Erdos-Renyi DAGs with a
given expected edge count, edge weights uniform on [-2, -0.5] U [0.5, 2],
standard-Gaussian observational noise, and soft mean-shift interventions on
the exogenous noise.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def random_er_dag(d: int, expected_edges: float, rng: np.random.Generator,
                  w_low: float = 0.5, w_high: float = 2.0) -> np.ndarray:
    """Weighted ER DAG. Returns W with W[i, j] != 0 meaning i -> j.

    Nodes are placed in a random topological order, so the order is not
    the index order.
    """
    p_edge = min(1.0, expected_edges / (d * (d - 1) / 2))
    order = rng.permutation(d)
    W = np.zeros((d, d))
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p_edge:
                w = rng.uniform(w_low, w_high) * rng.choice([-1.0, 1.0])
                W[order[a], order[b]] = w
    return W


@dataclass
class InterventionalData:
    X: np.ndarray          # (n, d) pooled samples
    env: np.ndarray        # (n,) environment label; 0 = observational
    W: np.ndarray          # true weighted adjacency, W[i, j] != 0 means i -> j

    @property
    def true_graph(self) -> np.ndarray:
        return (self.W != 0).astype(int)


def sample_single_node(W: np.ndarray, n_total: int, rng: np.random.Generator,
                       shift: float = 1.0, noise_sd: float = 1.0,
                       include_observational: bool = True) -> InterventionalData:
    """Atomic (single-node) design: one environment per node plus, optionally,
    the observational environment, sampled uniformly. Environment l > 0
    shifts the exogenous noise of node l-1 by `shift` (in units of noise_sd).
    """
    d = W.shape[0]
    n_env = d + 1 if include_observational else d
    env = rng.integers(0, n_env, size=n_total)
    if not include_observational:
        env = env + 1
    E = rng.normal(0.0, noise_sd, size=(n_total, d))
    hit = env > 0
    E[np.where(hit)[0], env[hit] - 1] += shift * noise_sd
    # x = W^T x + e  (row-vector form: X = E (I - W)^{-1})
    X = E @ np.linalg.inv(np.eye(d) - W)
    return InterventionalData(X=X, env=env, W=W)


def sample_bernoulli(W: np.ndarray, n_total: int, rng: np.random.Generator,
                     prob: float = 0.5, shift: float = 1.0,
                     noise_sd: float = 1.0) -> tuple[InterventionalData, np.ndarray]:
    """Independent-occurrence design (Appendix B). Returns the data and the
    (n, d) 0/1 matrix of realized intervention indicators. Environments are
    not well defined here (each sample has its own target pattern), which is
    why JCI is evaluated in the single-node design instead."""
    d = W.shape[0]
    B = (rng.random((n_total, d)) < prob).astype(float)
    E = rng.normal(0.0, noise_sd, size=(n_total, d)) + shift * noise_sd * B
    X = E @ np.linalg.inv(np.eye(d) - W)
    return InterventionalData(X=X, env=np.zeros(n_total, dtype=int), W=W), B


# --------------------------------------------------------------------------
# Conditional independence test
# --------------------------------------------------------------------------

class FisherZ:
    """Fisher-z partial-correlation test built on a cached correlation matrix.

    Valid for linear models: d-separation implies zero partial correlation
    regardless of the noise distribution, and the soft mean shifts are linear
    in the context indicators.
    """

    def __init__(self, data: np.ndarray):
        self.n = data.shape[0]
        sd = data.std(axis=0)
        if np.any(sd == 0):
            raise ValueError("A column has zero variance; drop it before testing.")
        self.C = np.corrcoef(data, rowvar=False)

    def pvalue(self, i: int, j: int, S: tuple[int, ...] = ()) -> float:
        idx = [i, j, *S]
        P = np.linalg.pinv(self.C[np.ix_(idx, idx)])
        denom = math.sqrt(abs(P[0, 0] * P[1, 1]))
        r = -P[0, 1] / denom if denom > 0 else 0.0
        r = max(min(r, 1 - 1e-12), -1 + 1e-12)
        dof = self.n - len(S) - 3
        if dof <= 0:
            return 1.0
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
        return float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _pair_status(G: np.ndarray, i: int, j: int) -> str:
    a, b = G[i, j] != 0, G[j, i] != 0
    if a and b:
        return "undirected"
    if a:
        return "fwd"
    if b:
        return "bwd"
    return "none"


def shd(G_est: np.ndarray, G_true: np.ndarray) -> int:
    """Structural Hamming distance between an estimate (DAG or CPDAG summary)
    and the true DAG. Each node pair whose status differs counts once: a
    missing edge, an extra edge, a reversed edge, or an edge left undirected
    (an orientation error relative to the fully directed target, as argued in
    the rebuttal for GIES) all cost 1."""
    d = G_true.shape[0]
    total = 0
    for i in range(d):
        for j in range(i + 1, d):
            if _pair_status(G_est, i, j) != _pair_status(G_true, i, j):
                total += 1
    return total


def directed_f1(G_est: np.ndarray, G_true: np.ndarray) -> dict:
    """Precision/recall/F1 on directed edges. Undirected edges in the estimate
    are not counted as directed predictions."""
    d = G_true.shape[0]
    pred = {(i, j) for i in range(d) for j in range(d)
            if G_est[i, j] and not G_est[j, i]}
    true = {(i, j) for i in range(d) for j in range(d) if G_true[i, j]}
    tp = len(pred & true)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def skeleton_f1(G_est: np.ndarray, G_true: np.ndarray) -> float:
    d = G_true.shape[0]
    est = {(i, j) for i in range(d) for j in range(i + 1, d)
           if G_est[i, j] or G_est[j, i]}
    true = {(i, j) for i in range(d) for j in range(i + 1, d)
            if G_true[i, j] or G_true[j, i]}
    tp = len(est & true)
    prec = tp / len(est) if est else 0.0
    rec = tp / len(true) if true else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def all_metrics(G_est: np.ndarray, G_true: np.ndarray) -> dict:
    out = {"shd": shd(G_est, G_true), "skeleton_f1": skeleton_f1(G_est, G_true)}
    out.update({f"dir_{k}": v for k, v in directed_f1(G_est, G_true).items()})
    out["n_undirected"] = int(np.sum(np.triu((G_est != 0) & (G_est.T != 0), 1)))
    return out


# --------------------------------------------------------------------------
# ICA-LiNGAM (label-free reference)
# --------------------------------------------------------------------------

_warned_fallback = False


def ica_lingam_graph(X: np.ndarray, seed: int = 0, edge_threshold: float = 0.0) -> np.ndarray:
    """Run ICA-LiNGAM on pooled data and return a 0/1 DAG, G[i, j] = 1 for i -> j.

    Uses the `lingam` package (as in the paper) when available. Otherwise it
    falls back to a minimal reimplementation and warns: the fallback is for
    testing the pipeline only and should not be used for reported numbers.
    """
    global _warned_fallback
    try:
        import lingam  # type: ignore
        model = lingam.ICALiNGAM(random_state=seed)
        model.fit(X)
        B = model.adjacency_matrix_   # B[i, j] = coefficient of j -> i
        return (np.abs(B.T) > edge_threshold).astype(int)
    except ImportError:
        if not _warned_fallback:
            warnings.warn("`lingam` not installed: using the minimal ICA-LiNGAM fallback. "
                          "Install `lingam` for reported numbers.", stacklevel=2)
            _warned_fallback = True
        return _ica_lingam_fallback(X, seed, edge_threshold)


def _ica_lingam_fallback(X: np.ndarray, seed: int, edge_threshold: float) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import FastICA
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    d = X.shape[1]
    ica = FastICA(n_components=d, whiten="unit-variance", max_iter=1000, random_state=seed)
    ica.fit(X)
    Wm = ica.components_
    _, col = linear_sum_assignment(1.0 / np.abs(Wm))
    PW = np.zeros_like(Wm)
    PW[col] = Wm
    PW = PW / np.diag(PW)[:, None]
    B = np.eye(d) - PW
    # causal order: repeatedly take a variable with no remaining parents
    order, remaining = [], list(range(d))
    Bt = B.copy()
    while remaining:
        sub = np.abs(Bt[np.ix_(remaining, remaining)])
        k = remaining[int(np.argmin(sub.sum(axis=1)))]
        order.append(k)
        remaining.remove(k)
    # adaptive lasso of each variable on its predecessors
    G = np.zeros((d, d), dtype=int)
    Xc = X - X.mean(axis=0)
    for pos, j in enumerate(order[1:], start=1):
        pred = order[:pos]
        lr = LinearRegression().fit(Xc[:, pred], Xc[:, j])
        w = np.abs(lr.coef_)
        las = LassoLarsIC(criterion="bic").fit(Xc[:, pred] * w, Xc[:, j])
        coef = las.coef_ * w
        for p_i, c in zip(pred, coef):
            if abs(c) > edge_threshold and c != 0:
                G[p_i, j] = 1
    return G
