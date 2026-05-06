"""
Experiment 2: Isolating the effect of non-Gaussianity loss on LiNGAM recovery.

Design: Gaussian source noise + independent per-node intervention firing.
Each node fires its intervention with probability pi = 1/d, independently across
nodes. The pooled noise is independent across coordinates by construction, so
the only effect of varying d is through per-coordinate non-Gaussianity.

Left panel: three theoretical standardized non-Gaussianity contrasts (kurtosis,
tanh-negentropy, Gaussian-negentropy) computed in closed form for the pooled
mixture, peak-normalized for visual comparison. All three are non-monotonic in
d, illustrating that the sweet-spot phenomenon is universal across standardized
contrasts.

Right panel: F1 score of ICA-LiNGAM (using the lingam package's default
configuration, which uses FastICA with the logcosh contrast) on pooled
intervention data.

Recovery is measured by F1 score on directed edges.

Requires:
    pip install lingam scikit-learn numpy scipy matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.stats import norm

try:
    import lingam
    _HAS_LINGAM = True
except ImportError:
    _HAS_LINGAM = False
    print("WARNING: lingam package not installed; falling back to homegrown ICA-LiNGAM.")
    print("         The homegrown fallback gives much worse recovery than the lingam package.")
    print("         Install with: pip install lingam")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_VALUES = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50]
N_TRIALS_PER_D = 50
TOTAL_SAMPLES_FN = lambda d: 200 * d
INTERVENTION_MEAN = 2.0
INTERVENTION_STD = 0.5
EDGE_PROB_FN = lambda d: 2.0 / d
RESULTS_FILE = "experiment2_results.npz"
FIGURE_BASENAME = "experiment2_figure"

# Three theoretical contrasts to plot in the left panel
CONTRAST_NAMES = ["kurtosis", "tanh_negentropy", "gauss_negentropy"]
CONTRAST_LABELS = {
    "kurtosis": "kurtosis",
    "tanh_negentropy": r"$\tanh$-negentropy",
    "gauss_negentropy": "Gaussian-negentropy",
}
CONTRAST_COLORS = {
    "kurtosis": "#d95f02",
    "tanh_negentropy": "#1b9e77",
    "gauss_negentropy": "#7570b3",
}


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


def generate_data_independent_interventions(
    A: np.ndarray, n_samples: int, pi: float,
    intervention_mean: float, intervention_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    d = A.shape[0]
    I_minus_A_inv = np.linalg.inv(np.eye(d) - A)
    E = rng.normal(0, 1, size=(n_samples, d))
    fire_mask = rng.random(size=(n_samples, d)) < pi
    shifts = rng.normal(loc=intervention_mean, scale=intervention_std,
                        size=(n_samples, d))
    E = E + fire_mask * shifts
    return E @ I_minus_A_inv.T


# ---------------------------------------------------------------------------
# ICA-LiNGAM
# ---------------------------------------------------------------------------

def fit_ica_lingam(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ICA-LiNGAM via the lingam package (default contrast: logcosh)."""
    seed = int(rng.integers(0, 2**31 - 1))
    if _HAS_LINGAM:
        model = lingam.ICALiNGAM(random_state=seed)
        model.fit(X)
        return model.adjacency_matrix_
    else:
        return _fit_ica_lingam_fallback(X, rng)


def _fit_ica_lingam_fallback(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fallback if lingam is not installed; gives worse results than the lingam package."""
    from sklearn.decomposition import FastICA
    from scipy.optimize import linear_sum_assignment

    d = X.shape[1]
    seed = int(rng.integers(0, 2**31 - 1))
    ica = FastICA(n_components=d, whiten="unit-variance", random_state=seed,
                  max_iter=5000, tol=1e-6, fun="logcosh")
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
    if np.any(np.isnan(A_est)):
        return 0.0, 0.0, 0.0
    B_true = (np.abs(A_true) > threshold).astype(int)
    B_est = (np.abs(A_est) > threshold).astype(int)
    tp = int(np.sum((B_true == 1) & (B_est == 1)))
    fp = int(np.sum((B_true == 0) & (B_est == 1)))
    fn = int(np.sum((B_true == 1) & (B_est == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Theoretical contrast values for the pooled mixture
# ---------------------------------------------------------------------------

def mixture_pdf(x: float, pi: float, mu: float, sigma_delta: float,
                sigma_base: float = 1.0) -> float:
    sd_int = np.sqrt(sigma_base**2 + sigma_delta**2)
    return ((1 - pi) * norm.pdf(x, 0, sigma_base)
            + pi * norm.pdf(x, mu, sd_int))


def mixture_mean_var(pi: float, mu: float, sigma_delta: float,
                     sigma_base: float = 1.0) -> tuple[float, float]:
    sigma2_base = sigma_base**2
    sigma2_int = sigma_base**2 + sigma_delta**2
    mean = pi * mu
    var = ((1 - pi) * sigma2_base + pi * sigma2_int
           + (1 - pi) * (0 - mean) ** 2 + pi * (mu - mean) ** 2)
    return mean, var


def theoretical_kurtosis(pi: float, mu: float, sigma_delta: float,
                         sigma_base: float = 1.0) -> float:
    """Excess kurtosis of the two-component Gaussian mixture, in closed form."""
    sigma2_base = sigma_base**2
    sigma2_int = sigma_base**2 + sigma_delta**2
    mean, var = mixture_mean_var(pi, mu, sigma_delta, sigma_base)
    m4_base = (0 - mean)**4 + 6 * (0 - mean)**2 * sigma2_base + 3 * sigma2_base**2
    m4_int = (mu - mean)**4 + 6 * (mu - mean)**2 * sigma2_int + 3 * sigma2_int**2
    m4 = (1 - pi) * m4_base + pi * m4_int
    return m4 / var**2 - 3.0


def theoretical_negentropy(pi: float, mu: float, sigma_delta: float,
                           sigma_base: float = 1.0) -> float:
    """Negentropy J(X) = H(Gaussian with same mean,var) - H(X), via numerical integration."""
    mean, var = mixture_mean_var(pi, mu, sigma_delta, sigma_base)
    sd = np.sqrt(var)

    def integrand(x):
        p = mixture_pdf(x, pi, mu, sigma_delta, sigma_base)
        if p < 1e-15:
            return 0.0
        return -p * np.log(p)

    H_X, _ = quad(integrand, mean - 12 * sd, mean + 12 * sd, limit=300)
    H_gaussian = 0.5 * np.log(2 * np.pi * np.e * var)
    return H_gaussian - H_X


def theoretical_tanh_contrast(pi: float, mu: float, sigma_delta: float,
                              sigma_base: float = 1.0) -> float:
    """tanh-based negentropy approximation: (E[log cosh(X_std)] - E[log cosh(Z)])^2."""
    mean, var = mixture_mean_var(pi, mu, sigma_delta, sigma_base)
    sd = np.sqrt(var)

    def integrand_X(x):
        p = mixture_pdf(x, pi, mu, sigma_delta, sigma_base)
        if p < 1e-15:
            return 0.0
        x_std = (x - mean) / sd
        return p * np.log(np.cosh(x_std))

    E_X, _ = quad(integrand_X, mean - 12 * sd, mean + 12 * sd, limit=300)

    def gauss_integrand(z):
        return norm.pdf(z, 0, 1) * np.log(np.cosh(z))
    E_Z, _ = quad(gauss_integrand, -10, 10, limit=200)

    return (E_X - E_Z) ** 2


def standardize_for_contrast(values: np.ndarray) -> np.ndarray:
    """Normalize so peak = 1 for cross-contrast comparison on a shared axis."""
    m = np.max(np.abs(values))
    return values / m if m > 0 else values


def compute_theoretical_contrast(name: str, pi: float, mu: float,
                                 sigma_delta: float) -> float:
    if name == "kurtosis":
        return abs(theoretical_kurtosis(pi, mu, sigma_delta))
    elif name == "tanh_negentropy":
        return theoretical_tanh_contrast(pi, mu, sigma_delta)
    elif name == "gauss_negentropy":
        return theoretical_negentropy(pi, mu, sigma_delta)
    else:
        raise ValueError(f"Unknown contrast: {name}")


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    d: int
    edge_prob: float
    seed: int
    f1: float
    precision: float
    recall: float
    n_edges_true: int


def run_trial(d: int, edge_prob: float, n_samples: int,
              intervention_mean: float, intervention_std: float,
              seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    A_canonical = sample_er_dag(d, edge_prob, rng)
    perm = rng.permutation(d)
    A_true = A_canonical[np.ix_(perm, perm)]

    pi = 1.0 / d
    X = generate_data_independent_interventions(
        A_true, n_samples, pi, intervention_mean, intervention_std, rng,
    )

    A_est = fit_ica_lingam(X, rng)
    p, r, f1 = precision_recall_f1(A_true, A_est)

    return TrialResult(
        d=d, edge_prob=edge_prob, seed=seed,
        f1=f1, precision=p, recall=r,
        n_edges_true=int((np.abs(A_true) > 0.1).sum()),
    )


def run_experiment() -> None:
    print(f"{'d':>4} {'p_edge':>8} {'pi':>8} {'samples':>9} "
          f"{'F1':>8} {'prec':>8} {'rec':>8} {'edges':>7}")
    print("-" * 70)

    all_results: list[TrialResult] = []
    for d in D_VALUES:
        edge_prob = EDGE_PROB_FN(d)
        n_samples = TOTAL_SAMPLES_FN(d)
        results_d = []
        for trial in range(N_TRIALS_PER_D):
            seed = 2000 * d + trial
            r = run_trial(d, edge_prob, n_samples,
                          INTERVENTION_MEAN, INTERVENTION_STD, seed)
            results_d.append(r)
            all_results.append(r)

        f1 = np.mean([r.f1 for r in results_d])
        prec = np.mean([r.precision for r in results_d])
        rec = np.mean([r.recall for r in results_d])
        edges = np.mean([r.n_edges_true for r in results_d])
        print(f"{d:>4} {edge_prob:>8.3f} {1.0/d:>8.3f} {n_samples:>9} "
              f"{f1:>8.3f} {prec:>8.3f} {rec:>8.3f} {edges:>7.1f}")

    np.savez(
        RESULTS_FILE,
        d=np.array([r.d for r in all_results]),
        f1=np.array([r.f1 for r in all_results]),
        precision=np.array([r.precision for r in all_results]),
        recall=np.array([r.recall for r in all_results]),
        edges=np.array([r.n_edges_true for r in all_results]),
    )
    print(f"\nResults saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots():
    data = np.load(RESULTS_FILE)
    d_vals = data["d"]
    f1 = data["f1"]
    unique_d = np.unique(d_vals)

    def agg(arr, stat=np.mean):
        return np.array([stat(arr[d_vals == d]) for d in unique_d])

    def sem(arr):
        return np.array([
            np.std(arr[d_vals == d]) / np.sqrt(np.sum(d_vals == d))
            for d in unique_d
        ])

    f1_mean = agg(f1)
    f1_err = sem(f1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ---- Left panel: theoretical contrast curves ----
    ax = axes[0]
    d_grid = np.array(D_VALUES)

    contrast_curves = {}
    for name in CONTRAST_NAMES:
        curve = np.array([
            compute_theoretical_contrast(name, 1.0 / d,
                                         INTERVENTION_MEAN, INTERVENTION_STD)
            for d in d_grid
        ])
        contrast_curves[name] = curve

    for name in CONTRAST_NAMES:
        normalized = standardize_for_contrast(contrast_curves[name])
        ax.plot(d_grid, normalized, marker="o", ms=4, lw=1.5,
                color=CONTRAST_COLORS[name], label=CONTRAST_LABELS[name])

    ax.set_xlabel("Number of nodes $d$")
    ax.set_ylabel("Standardized non-Gaussianity (peak-normalized)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(a) Per-coordinate non-Gaussianity contrasts vs. $d$")
    ax.set_ylim(0, 1.1)

    # ---- Right panel: ICA-LiNGAM F1 ----
    ax = axes[1]
    ax.errorbar(unique_d, f1_mean, yerr=f1_err, marker="o", ms=5,
                color="#1f77b4", lw=1.5, capsize=3,
                label="ICA-LiNGAM (lingam package)")
    ax.set_xlabel("Number of nodes $d$")
    ax.set_ylabel("F1 score on directed edges")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("(b) ICA-LiNGAM recovery quality")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_BASENAME}.pdf", bbox_inches="tight")
    plt.savefig(f"{FIGURE_BASENAME}.png", dpi=150, bbox_inches="tight")
    print(f"Figures saved as {FIGURE_BASENAME}.{{pdf,png}}")
    plt.close()

    print("\nContrast peak locations (theoretical):")
    for name in CONTRAST_NAMES:
        peak_d = d_grid[np.argmax(contrast_curves[name])]
        print(f"  {CONTRAST_LABELS[name]:>30}: peaks at d = {peak_d}")

    print("\nF1 by d:")
    print(f"{'d':>4} {'F1':>10}")
    print("-" * 16)
    for i, d in enumerate(unique_d):
        print(f"{d:>4} {f1_mean[i]:>10.3f}")


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