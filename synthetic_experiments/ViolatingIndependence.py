"""
Experiment 1: Isolating the effect of dependence on LiNGAM recovery.

Design: highly non-Gaussian source noise (Laplace) + single-node interventions.
Holding non-Gaussianity ~constant across d isolates the independence-violation
effect.

Recovery is measured by F1 score on directed edges, which normalizes naturally
across graph sizes.

This version uses both a histogram MI estimator and a k-NN-based estimator
(via sklearn's mutual_info_regression, which uses Kraskov et al.'s approach).

Requires:
    pip install lingam scikit-learn numpy scipy matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression

try:
    import lingam
    _HAS_LINGAM = True
except ImportError:
    _HAS_LINGAM = False
    print("WARNING: lingam package not installed; falling back to homegrown ICA-LiNGAM.")
    print("         Install with: pip install lingam")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_VALUES = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50]
N_TRIALS_PER_D = 50
SAMPLES_PER_ENV = 200
INTERVENTION_MEAN = 2.0
INTERVENTION_STD = 0.5
NOISE_KIND = "laplace"
EDGE_PROB_FN = lambda d: 2.0 / d
RESULTS_FILE = "experiment1_results.npz"
FIGURE_BASENAME = "experiment1_figure"


# ---------------------------------------------------------------------------
# DAG sampling and SCM generation
# ---------------------------------------------------------------------------

def sample_er_dag(d: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """ER DAG. A[j, i] != 0 iff i -> j."""
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                w = rng.uniform(0.25, 1.0) * (1 if rng.random() < 0.5 else -1)
                A[j, i] = w
    return A


def _sample_noise(samples: int, d: int, kind: str,
                  rng: np.random.Generator) -> np.ndarray:
    if kind == "laplace":
        return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(samples, d))
    elif kind == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(samples, d))
    else:
        raise ValueError(f"Unknown noise kind: {kind}")


def generate_data_single_node(
    A: np.ndarray, samples_per_env: int,
    intervention_mean: float, intervention_std: float,
    rng: np.random.Generator, noise_kind: str = "laplace",
) -> np.ndarray:
    d = A.shape[0]
    I_minus_A_inv = np.linalg.inv(np.eye(d) - A)
    X_list = []
    for env in range(d):
        E = _sample_noise(samples_per_env, d, noise_kind, rng)
        shifts = rng.normal(loc=intervention_mean, scale=intervention_std,
                            size=samples_per_env)
        E[:, env] += shifts
        X_env = E @ I_minus_A_inv.T
        X_list.append(X_env)
    return np.vstack(X_list)


def generate_data_observational(
    A: np.ndarray, n_samples: int,
    rng: np.random.Generator, noise_kind: str = "laplace",
) -> np.ndarray:
    d = A.shape[0]
    I_minus_A_inv = np.linalg.inv(np.eye(d) - A)
    E = _sample_noise(n_samples, d, noise_kind, rng)
    return E @ I_minus_A_inv.T


# ---------------------------------------------------------------------------
# ICA-LiNGAM
# ---------------------------------------------------------------------------

def fit_ica_lingam(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seed = int(rng.integers(0, 2**31 - 1))
    if _HAS_LINGAM:
        model = lingam.ICALiNGAM(random_state=seed)
        model.fit(X)
        return model.adjacency_matrix_
    else:
        return _fit_ica_lingam_fallback(X, rng)


def _fit_ica_lingam_fallback(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from sklearn.decomposition import FastICA
    from scipy.optimize import linear_sum_assignment

    d = X.shape[1]
    seed = int(rng.integers(0, 2**31 - 1))
    ica = FastICA(n_components=d, whiten="unit-variance", random_state=seed,
                  max_iter=2000, tol=1e-6, fun="cube")
    try:
        ica.fit(X)
    except Exception:
        return np.full((d, d), np.nan)
    W = ica.components_

    _, col_ind = linear_sum_assignment(-np.abs(W))
    W = W[col_ind]
    diag = np.diag(W)
    if np.any(np.abs(diag) < 1e-10):
        return np.full((d, d), np.nan)
    W = W / diag[:, None]

    B = np.eye(d) - W
    perm = _find_causal_order(B)
    P = np.eye(d)[perm]
    B_ordered = P @ B @ P.T
    A_recovered = np.tril(B_ordered, k=-1)
    return P.T @ A_recovered @ P


def _find_causal_order(B: np.ndarray) -> list[int]:
    d = B.shape[0]
    B_abs = np.abs(B.copy())
    np.fill_diagonal(B_abs, 0)
    remaining = list(range(d))
    order = []
    for _ in range(d):
        sub = B_abs[np.ix_(remaining, remaining)]
        idx = int(np.argmin(sub.sum(axis=1)))
        order.append(remaining[idx])
        remaining.pop(idx)
    return order


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(A_true: np.ndarray, A_est: np.ndarray,
                        threshold: float = 0.1) -> tuple[float, float, float]:
    """
    Precision, recall, and F1 on directed edges after thresholding.
    """
    if np.any(np.isnan(A_est)):
        return 0.0, 0.0, 0.0
    B_true = (np.abs(A_true) > threshold).astype(int)
    B_est = (np.abs(A_est) > threshold).astype(int)
    tp = int(np.sum((B_true == 1) & (B_est == 1)))
    fp = int(np.sum((B_true == 0) & (B_est == 1)))
    fn = int(np.sum((B_true == 1) & (B_est == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def _select_pairs(d: int, n_pairs: int,
                  rng: np.random.Generator) -> list[tuple[int, int]]:
    """Select up to n_pairs distinct unordered pairs of coordinate indices."""
    seen = set()
    pairs = []
    while len(pairs) < min(n_pairs, d * (d - 1) // 2):
        i, j = sorted(rng.integers(0, d, size=2))
        if i != j and (i, j) not in seen:
            seen.add((int(i), int(j)))
            pairs.append((int(i), int(j)))
    return pairs


def pairwise_mi_histogram(X: np.ndarray, pairs: list[tuple[int, int]],
                          n_bins: int = 12) -> float:
    return float(np.mean([
        _mi_histogram(X[:, i], X[:, j], n_bins) for i, j in pairs
    ]))


def _mi_histogram(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    h, _, _ = np.histogram2d(x, y, bins=n_bins)
    p = h / h.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))


def pairwise_mi_knn(X: np.ndarray, pairs: list[tuple[int, int]],
                    n_neighbors: int = 5,
                    rng: np.random.Generator | None = None) -> float:
    """
    Average pairwise MI using sklearn's k-NN-based estimator.
    sklearn's mutual_info_regression uses the Kraskov et al. (2004) k-NN
    approach internally. Returns nats.
    """
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    mis = []
    for i, j in pairs:
        mi = mutual_info_regression(
            X[:, [i]], X[:, j],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=seed,
        )[0]
        mis.append(float(mi))
    return float(np.mean(mis))


def average_excess_kurtosis(X: np.ndarray) -> float:
    standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    return float(np.mean(np.mean(standardized ** 4, axis=0) - 3.0))


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    d: int
    edge_prob: float
    seed: int
    f1_intervention: float
    f1_observational: float
    precision_intervention: float
    precision_observational: float
    recall_intervention: float
    recall_observational: float
    avg_pairwise_mi_hist: float
    avg_pairwise_mi_knn: float
    avg_kurtosis: float
    n_edges_true: int


def run_trial(d: int, edge_prob: float, samples_per_env: int,
              intervention_mean: float, intervention_std: float,
              seed: int, noise_kind: str) -> TrialResult:
    rng = np.random.default_rng(seed)
    A_canonical = sample_er_dag(d, edge_prob, rng)
    perm = rng.permutation(d)
    A_true = A_canonical[np.ix_(perm, perm)]

    X_int = generate_data_single_node(
        A_true, samples_per_env, intervention_mean, intervention_std,
        rng, noise_kind,
    )
    X_obs = generate_data_observational(
        A_true, d * samples_per_env, rng, noise_kind,
    )

    A_int_est = fit_ica_lingam(X_int, rng)
    A_obs_est = fit_ica_lingam(X_obs, rng)

    prec_int, rec_int, f1_int = precision_recall_f1(A_true, A_int_est)
    prec_obs, rec_obs, f1_obs = precision_recall_f1(A_true, A_obs_est)

    E_int = X_int @ (np.eye(d) - A_true).T
    pairs = _select_pairs(d, min(20, d * (d - 1) // 2), rng)
    avg_mi_hist = pairwise_mi_histogram(E_int, pairs)
    avg_mi_knn = pairwise_mi_knn(E_int, pairs, rng=rng)
    avg_kurt = average_excess_kurtosis(E_int)

    return TrialResult(
        d=d, edge_prob=edge_prob, seed=seed,
        f1_intervention=f1_int, f1_observational=f1_obs,
        precision_intervention=prec_int, precision_observational=prec_obs,
        recall_intervention=rec_int, recall_observational=rec_obs,
        avg_pairwise_mi_hist=avg_mi_hist,
        avg_pairwise_mi_knn=avg_mi_knn,
        avg_kurtosis=avg_kurt,
        n_edges_true=int((np.abs(A_true) > 0.1).sum()),
    )


def run_experiment() -> None:
    print(f"{'d':>4} {'p_edge':>8} {'F1_int':>8} {'F1_obs':>8} "
          f"{'MI_hist':>10} {'MI_knn':>10} {'kurt':>8} {'edges':>7}")
    print("-" * 80)

    all_results: list[TrialResult] = []
    for d in D_VALUES:
        edge_prob = EDGE_PROB_FN(d)
        results_d = []
        for trial in range(N_TRIALS_PER_D):
            seed = 1000 * d + trial
            r = run_trial(d, edge_prob, SAMPLES_PER_ENV,
                          INTERVENTION_MEAN, INTERVENTION_STD,
                          seed, NOISE_KIND)
            results_d.append(r)
            all_results.append(r)

        f1_int = np.mean([r.f1_intervention for r in results_d])
        f1_obs = np.mean([r.f1_observational for r in results_d])
        mi_hist = np.mean([r.avg_pairwise_mi_hist for r in results_d])
        mi_knn = np.mean([r.avg_pairwise_mi_knn for r in results_d])
        kurt = np.mean([r.avg_kurtosis for r in results_d])
        edges = np.mean([r.n_edges_true for r in results_d])
        print(f"{d:>4} {edge_prob:>8.3f} {f1_int:>8.3f} {f1_obs:>8.3f} "
              f"{mi_hist:>10.4f} {mi_knn:>10.4f} {kurt:>8.3f} {edges:>7.1f}")

    np.savez(
        RESULTS_FILE,
        d=np.array([r.d for r in all_results]),
        f1_int=np.array([r.f1_intervention for r in all_results]),
        f1_obs=np.array([r.f1_observational for r in all_results]),
        precision_int=np.array([r.precision_intervention for r in all_results]),
        precision_obs=np.array([r.precision_observational for r in all_results]),
        recall_int=np.array([r.recall_intervention for r in all_results]),
        recall_obs=np.array([r.recall_observational for r in all_results]),
        mi_hist=np.array([r.avg_pairwise_mi_hist for r in all_results]),
        mi_knn=np.array([r.avg_pairwise_mi_knn for r in all_results]),
        kurt=np.array([r.avg_kurtosis for r in all_results]),
        edges=np.array([r.n_edges_true for r in all_results]),
    )
    print(f"\nResults saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots():
    data = np.load(RESULTS_FILE)
    d_vals = data["d"]
    f1_int = data["f1_int"]
    f1_obs = data["f1_obs"]
    mi_hist = data["mi_hist"]
    mi_knn = data["mi_knn"]
    unique_d = np.unique(d_vals)

    def agg(arr, stat=np.mean):
        return np.array([stat(arr[d_vals == d]) for d in unique_d])

    def sem(arr):
        return np.array([
            np.std(arr[d_vals == d]) / np.sqrt(np.sum(d_vals == d))
            for d in unique_d
        ])

    mi_hist_mean = agg(mi_hist)
    mi_hist_err = sem(mi_hist)
    mi_knn_mean = agg(mi_knn)
    mi_knn_err = sem(mi_knn)
    f1_int_mean = agg(f1_int)
    f1_int_err = sem(f1_int)
    f1_obs_mean = agg(f1_obs)
    f1_obs_err = sem(f1_obs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: pairwise MI from both estimators with reference lines
    ax = axes[0]
    knn_pos = np.maximum(mi_knn_mean, 1e-6)
    ax.errorbar(unique_d, mi_hist_mean, yerr=mi_hist_err, marker="o", ms=5,
                color="#2c7fb8", lw=1.5, capsize=3,
                label="Histogram estimator")
    ax.errorbar(unique_d, knn_pos, yerr=mi_knn_err, marker="s", ms=5,
                color="#d95f02", lw=1.5, capsize=3,
                label="k-NN estimator (KSG)")
    ref_d = 10
    if ref_d in unique_d:
        anchor_hist = mi_hist_mean[unique_d == ref_d][0]
        ax.plot(unique_d, anchor_hist * (ref_d / unique_d), ":",
                color="#2c7fb8", alpha=0.5, lw=1.0, label=r"$\propto 1/d$")
    ax.set_xlabel("Number of nodes $d$")
    ax.set_ylabel("Pairwise mutual information (nats)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(a) Per-pair dependence vs. graph size")

    # Right: F1 score
    ax = axes[1]
    ax.errorbar(unique_d, f1_int_mean, yerr=f1_int_err, marker="o", ms=5,
                color="#1f77b4", lw=1.5, capsize=3,
                label="Interventional (single-node)")
    ax.errorbar(unique_d, f1_obs_mean, yerr=f1_obs_err, marker="s", ms=5,
                color="#7f7f7f", lw=1.5, capsize=3,
                label="Observational (ceiling)")
    ax.set_xlabel("Number of nodes $d$")
    ax.set_ylabel("F1 score on directed edges")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title("(b) ICA-LiNGAM recovery quality")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_BASENAME}.pdf", bbox_inches="tight")
    plt.savefig(f"{FIGURE_BASENAME}.png", dpi=150, bbox_inches="tight")
    print(f"Figures saved as {FIGURE_BASENAME}.{{pdf,png}}")
    plt.close()

    print("\nSummary:")
    print(f"{'d':>4} {'MI_hist':>10} {'hist*d':>10} "
          f"{'MI_knn':>10} {'knn*d^2':>10} {'knn*d^4':>10} "
          f"{'F1_int':>8} {'F1_obs':>8}")
    print("-" * 90)
    for i, d in enumerate(unique_d):
        knn = max(mi_knn_mean[i], 1e-9)
        print(f"{d:>4} {mi_hist_mean[i]:>10.4f} "
              f"{mi_hist_mean[i] * d:>10.4f} "
              f"{mi_knn_mean[i]:>10.5f} "
              f"{knn * d**2:>10.4f} "
              f"{knn * d**4:>10.2f} "
              f"{f1_int_mean[i]:>8.3f} "
              f"{f1_obs_mean[i]:>8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip running the experiment; just regenerate the plot.",
    )
    args = parser.parse_args()

    if not args.plot_only:
        run_experiment()
    make_plots()


if __name__ == "__main__":
    main()