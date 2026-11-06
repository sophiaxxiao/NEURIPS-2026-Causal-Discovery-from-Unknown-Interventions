# coding=utf-8
"""
Ablation simulator for the rebuttal experiment:
    "Causal Discovery from Unseen Environments" — NeurIPS 2026.

AUDIT SUMMARY (read before modifying)
======================================
Based on inspection of MakeDataset.py, IIDLinearSimulation.py,
method_wrappers.py, helpers.py, and ERDagsComparison.py.

1. datasets_interventional keys: tuples such as (0,1,0,...,0). Each key is
   ONE unique intervention-pattern environment — NOT one per-sample pattern.
   Value dict: {'dataset_object', 'X': (count, d), 'B': adjacency, 'intervened_nodes'}.

2. p_intervention: per-NODE Bernoulli success probability in 'bernoulli' mode.
   Each of the n_samples rows independently draws a binary vector of length
   n_nodes, and unique rows become environments.

3. delta application: delta_per_node[i] = delta * sqrt(var(X_obs[:, i])).
   The noise for node i in an intervened environment is Normal(delta_per_node[i], 1).
   delta is a scalar multiplier of the per-node observational standard deviation.

4. build_env_labels: returns an integer array of length N_interventional,
   where environment k gets label k ordered by datasets_interventional.keys().

5. split_subsample: obs_sub = full obs dataset + any all-zeros-pattern rows;
   ivs_list = one array per non-zero-pattern environment;
   known_targets = node index lists aligned with ivs_list.

6. DAG edge weights: signed, magnitudes in [0.5, 2] via weight_range=[0.5, 2]
   in DAG.erdos_renyi.

7. n_samples in ShiftSimulation = INTERVENTIONAL samples only (excludes obs).
   n_observational = the 100 separate observational samples.

8. ERDagsComparison.py generates n_samples=1e6 and then subsamples. The
   recorded n_samples_pool metadata incorrectly says 1e8; the actual pool is 1e6.
   This ablation uses n_samples=n_interventional_total directly — no subsampling.

ABLATION DESIGN
================
All four ablations share a common baseline:
    Bernoulli mode, p_i=0.5 (uniform), delta=1.0, rho=0, tau=0, h=0.

Ablation A — Unequal environment sizes:
    Heterogeneous per-node Bernoulli probabilities p_i(h) = 0.5 + h * z_perm[i],
    where z is linearly spaced in [-1,1] and perm is a seed-specific permutation.
    This is the correct analog for the Bernoulli simulator: there are no
    pre-specified explicit environment counts (they emerge from unique patterns),
    so environment size imbalance is induced by unequal marginal probabilities.
    Severity: h in {0.00, 0.15, 0.30, 0.45}.

Ablation B — Heterogeneous intervention strengths:
    Node-specific delta_i = exp(tau * z_perm[i]) / sqrt(mean(exp(2*tau*z))),
    so mean(delta_i^2) = 1 at all tau. Strength ranks randomly permuted per seed.
    Severity: tau in {0.00, 0.35, 0.70, 1.00}.

Ablation C — Uniformly weak interventions:
    Scalar delta in {1.00, 0.75, 0.50, 0.25, 0.10}.

Ablation D — Correlated off-target effects:
    Random derangement pi. When node i is intervened (B_i=1), node pi(i)
    also receives a shift of amplitude rho * delta_i * B_i.
    Normalization c(rho) = 1/sqrt(1+rho^2) preserves total shift energy.
    Actual targets = {i, pi(i)} for each nominal target i.
    Severity: rho in {0.00, 0.10, 0.25, 0.50, 1.00}.

PAIRED RANDOMNESS
==================
For a fixed seed:
  * DAG.erdos_renyi(seed=seed) → identical DAG in all conditions.
  * ShiftSimulation calls np.random.seed(seed) in setup_interventions(),
    so the Bernoulli intervention matrix is reproducible for the same p_vector
    across runs. When p_vector changes (Ablation A), the matrix changes
    but is still reproducible.
  * Observational data is generated from the RNG state after DAG creation.
  * All condition-specific permutations and derangements are generated from
    numpy.random.default_rng(seed) so they are independent of the global RNG.
"""

import sys
import os
import numpy as np
from scipy import stats

# Allow running from the ablation directory or the synthetic_experiments root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MakeDataset import ShiftSimulation
from IIDLinearSimulation import IIDLinearSimulation


# ---------------------------------------------------------------------------
# Helper: random derangement
# ---------------------------------------------------------------------------

def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a uniform random derangement of {0, ..., n-1}.

    A derangement is a permutation with no fixed points: pi(i) != i for all i.

    Parameters
    ----------
    n : int
    rng : np.random.Generator

    Returns
    -------
    pi : np.ndarray of shape (n,), dtype int
    """
    if n < 2:
        raise ValueError("Cannot form a derangement of fewer than 2 elements.")
    for _ in range(10_000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError("Failed to generate derangement in 10,000 attempts.")


# ---------------------------------------------------------------------------
# Helper: per-node p-vector for Ablation A
# ---------------------------------------------------------------------------

def heterogeneous_p_vector(n_nodes: int, h: float, rng: np.random.Generator) -> np.ndarray:
    """Compute per-node Bernoulli probabilities for Ablation A.

    p_i(h) = 0.5 + h * z_perm[i],  where z is linspace(-1, 1, n_nodes)
    and perm is generated from rng (seed-specific, fixed across h values).

    The mean intervention probability equals exactly 0.5 for all h (since z
    is symmetric about 0 and its mean is 0).

    Parameters
    ----------
    n_nodes : int
    h : float
        Severity. h=0 → uniform p=0.5. h=0.45 → range [0.05, 0.95].
    rng : np.random.Generator
        Seeded generator; call once per seed and reuse across h values by
        drawing the permutation first, then applying different h externally.

    Returns
    -------
    p_vector : np.ndarray of shape (n_nodes,)
    """
    z = np.linspace(-1.0, 1.0, n_nodes)
    perm = rng.permutation(n_nodes)
    z_perm = z[perm]
    p_vector = 0.5 + h * z_perm
    p_vector = np.clip(p_vector, 1e-6, 1 - 1e-6)  # numerical safety
    return p_vector


def heterogeneous_p_vector_from_z(z_perm: np.ndarray, h: float) -> np.ndarray:
    """Apply severity h to a pre-permuted z vector (reuse the same permutation)."""
    p_vector = 0.5 + h * z_perm
    return np.clip(p_vector, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Helper: per-node delta-vector for Ablation B
# ---------------------------------------------------------------------------

def heterogeneous_delta_vector(n_nodes: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Compute normalized per-node delta multipliers for Ablation B.

    raw_delta_i = exp(tau * z_perm[i])
    delta_i     = raw_delta_i / sqrt(mean(raw_delta_j^2))

    Guarantees mean(delta_i^2) = 1 at all tau. At tau=0, delta_i = 1 for all i.

    Parameters
    ----------
    n_nodes : int
    tau : float
        Log-spread parameter. tau=0 → homogeneous delta=1.
    rng : np.random.Generator
        Seeded generator; call once per seed to get the permutation.

    Returns
    -------
    delta_vec : np.ndarray of shape (n_nodes,)
    """
    z = np.linspace(-1.0, 1.0, n_nodes)
    perm = rng.permutation(n_nodes)
    z_perm = z[perm]
    raw = np.exp(tau * z_perm)
    rms = np.sqrt(np.mean(raw ** 2))
    return raw / rms


def heterogeneous_delta_vector_from_z(z_perm: np.ndarray, tau: float) -> np.ndarray:
    """Apply severity tau to a pre-permuted z vector (reuse the same permutation)."""
    raw = np.exp(tau * z_perm)
    rms = np.sqrt(np.mean(raw ** 2))
    return raw / rms


# ---------------------------------------------------------------------------
# Extended simulator subclass
# ---------------------------------------------------------------------------

class AblationShiftSimulation(ShiftSimulation):
    """ShiftSimulation extended with ablation-specific simulation options.

    New parameters (all optional — defaults reproduce base ShiftSimulation):
    ---------------------------------------------------------------------------
    p_vector : np.ndarray or None
        Per-node Bernoulli intervention probabilities (Ablation A).
        If None, uses the scalar p_intervention for all nodes.
    delta_vector : np.ndarray or None
        Per-node delta multipliers BEFORE observational-std scaling (Ablation B).
        If None, uses the scalar delta for all nodes.
        Set after simulate_observational() via compute_per_node_delta_vector().
    off_target_rho : float
        Off-target effect amplitude (Ablation D). rho=0 disables off-targets.
    derangement : np.ndarray or None
        Permutation pi for Ablation D. Required if off_target_rho > 0.
    nominal_targets : dict or None
        Populated by simulate_interventions; maps pattern → [intended targets].
    actual_targets : dict or None
        Populated by simulate_interventions; maps pattern → [intended + off-targets].
    """

    def __init__(
        self,
        n_nodes=10,
        n_edges=10,
        n_samples=2000,
        n_observational=100,
        weighted_random_dag=None,
        p_intervention=0.5,
        delta=1.0,
        seed=42,
        intervention_mode="bernoulli",
        include_no_intervention=False,
        # Ablation-specific
        p_vector=None,
        delta_vector=None,
        off_target_rho=0.0,
        derangement=None,
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_samples=n_samples,
            n_observational=n_observational,
            weighted_random_dag=weighted_random_dag,
            p_intervention=p_intervention,
            delta=delta,
            seed=seed,
            intervention_mode=intervention_mode,
            include_no_intervention=include_no_intervention,
        )
        self.p_vector = p_vector
        self.delta_vector = delta_vector
        self.off_target_rho = off_target_rho
        self.derangement = derangement
        self.nominal_targets = None  # populated by simulate_interventions
        self.actual_targets = None   # populated by simulate_interventions

    # ------------------------------------------------------------------
    # Ablation A: heterogeneous per-node Bernoulli probabilities
    # ------------------------------------------------------------------

    def setup_interventions(self, verbose=False):
        """Override: use per-node p_vector when provided (Ablation A)."""
        if self.p_vector is None:
            # Standard behavior
            return super().setup_interventions(verbose=verbose)

        # Heterogeneous Bernoulli per node
        self._set_seed()
        p_vec = np.asarray(self.p_vector, dtype=float)
        if p_vec.shape != (self.n_nodes,):
            raise ValueError(f"p_vector must have shape ({self.n_nodes},)")

        # Draw n_samples rows with per-column probabilities
        U = np.random.uniform(size=(self.n_samples, self.n_nodes))
        intervention_matrix = (U < p_vec[np.newaxis, :]).astype(int)

        self.intervention_matrix = intervention_matrix

        intervention_sets, counts = np.unique(
            intervention_matrix, axis=0, return_counts=True
        )

        if verbose:
            print(
                f"[AblationA] p_vector range [{p_vec.min():.3f}, {p_vec.max():.3f}], "
                f"unique patterns: {len(intervention_sets)}"
            )

        self.k = len(intervention_sets)
        self.intervention_dict = {
            tuple(int(x) for x in intervention_sets[i]): int(counts[i])
            for i in range(len(intervention_sets))
        }
        self.intervened_nodes = [np.flatnonzero(row) for row in intervention_matrix]
        return self.intervention_dict

    # ------------------------------------------------------------------
    # Ablation B: heterogeneous per-node delta multipliers
    # ------------------------------------------------------------------

    def compute_per_node_delta_vector(self, scalar=1.0):
        """Set delta_per_node using the ablation's delta_vector (Ablation B).

        delta_per_node[i] = scalar * delta_vector[i] * sqrt(var(X_obs[:, i]))

        This replaces the standard compute_per_node_delta_from_observational
        with a per-node multiplier. Ensures mean(delta_vector^2) = 1, so
        the RMS shift matches the scalar-delta baseline.
        """
        if self.dataset_observational is None:
            self.simulate_observational()
        X_obs = self.dataset_observational.X
        obs_std = np.std(X_obs, axis=0)
        dv = np.asarray(self.delta_vector, dtype=float)
        self.delta_per_node = scalar * dv * obs_std
        return self.delta_per_node

    # ------------------------------------------------------------------
    # Ablation D: correlated off-target effects
    # ------------------------------------------------------------------

    def simulate_interventions(self, verbose=False):
        """Override: add correlated off-target effects (Ablation D).

        When off_target_rho > 0, for each pattern P with P[i]=1, node pi(i)
        also receives a mean shift of c(rho)*rho*delta_per_node[i] using
        the same Bernoulli indicator (deterministic within the environment).

        c(rho) = 1 / sqrt(1 + rho^2) normalizes total shift energy.

        Updates self.nominal_targets and self.actual_targets.
        """
        if self.off_target_rho == 0.0 or self.derangement is None:
            result = super().simulate_interventions(verbose=verbose)
            # Record targets for consistency
            self.nominal_targets = {
                pat: info["intervened_nodes"]
                for pat, info in self.datasets_interventional.items()
            }
            self.actual_targets = dict(self.nominal_targets)
            return result

        # --- off-target simulation ---
        if self.dataset_observational is None:
            if verbose:
                print("Observational dataset not found. Simulating first...")
            self.simulate_observational()

        if self.intervened_nodes is None:
            if verbose:
                print("Intervention sets not found. Setting up first...")
            self.setup_interventions()

        delta_per_node = getattr(self, "delta_per_node", None)
        if delta_per_node is None:
            raise ValueError(
                "delta_per_node not set. Call compute_per_node_delta_from_observational() first."
            )

        rho = self.off_target_rho
        c = 1.0 / np.sqrt(1.0 + rho ** 2)
        pi = np.asarray(self.derangement, dtype=int)
        pi_inv = np.empty_like(pi)
        pi_inv[pi] = np.arange(len(pi))  # inverse permutation

        self.datasets_interventional = {}
        self.nominal_targets = {}
        self.actual_targets = {}
        env_idx = 0

        for pattern, count in self.intervention_dict.items():
            pat_arr = np.array(pattern)
            nominal_nodes = np.flatnonzero(pat_arr).tolist()

            # Compute effective mean shift per node for this pattern.
            # U_i = c * (delta[i]*P[i] + rho * delta[pi_inv[i]]*P[pi_inv[i]])
            # Note: pi_inv[i] is the node whose off-target leaks into i.
            effective_mu = np.zeros(self.n_nodes)
            for i in range(self.n_nodes):
                intended_shift = c * delta_per_node[i] * pat_arr[i]
                # off-target contribution TO node i from node pi_inv[i]
                j = pi_inv[i]
                offtarget_shift = c * rho * delta_per_node[j] * pat_arr[j]
                effective_mu[i] = intended_shift + offtarget_shift

            # Build noise dict for this pattern
            noise_dict_i = {}
            for i in range(self.n_nodes):
                mu_i = effective_mu[i]
                noise_dict_i[i] = (
                    lambda mu=mu_i: (lambda size: np.random.normal(loc=mu, scale=1.0, size=size))
                )()

            if verbose:
                print(
                    f"Env {env_idx+1}: pattern={pattern}, "
                    f"mu_range=[{effective_mu.min():.3f},{effective_mu.max():.3f}]"
                )

            sim_i = IIDLinearSimulation(
                W=self.weighted_random_dag,
                noise_models=noise_dict_i,
                n=count,
            )

            # Actual targets: nodes with nonzero effective shift
            actual_nodes = sorted(
                set(nominal_nodes)
                | {int(pi[j]) for j in nominal_nodes}
            )

            self.datasets_interventional[pattern] = {
                "dataset_object": sim_i,
                "X": sim_i.X,
                "B": sim_i.B,
                "intervened_nodes": nominal_nodes,
            }
            self.nominal_targets[pattern] = nominal_nodes
            self.actual_targets[pattern] = actual_nodes

            env_idx += 1

        return self.datasets_interventional

    # ------------------------------------------------------------------
    # Convenience: combined generation pipeline
    # ------------------------------------------------------------------

    def generate_ablation_datasets(self):
        """Generate datasets for the configured ablation condition.

        Resets np.random.seed(self.seed) at the start so that observational data
        is generated from a known, deterministic RNG state regardless of the
        global state when this method is called. setup_interventions() resets
        the seed again internally (inherited from ShiftSimulation._set_seed),
        ensuring interventional data is also reproducible.

        Selects the right compute_per_node_delta method and simulation path
        automatically based on which ablation parameters are set.
        """
        # Reset global RNG so obs data is deterministic for this seed.
        np.random.seed(self.seed)

        self.simulate_observational()

        if self.delta_vector is not None:
            # Ablation B: heterogeneous strengths
            self.compute_per_node_delta_vector(scalar=self.delta)
        else:
            # Baseline, A, C, D: scalar delta
            self.compute_per_node_delta_from_observational(scalar=self.delta)

        self.setup_interventions()
        self.simulate_interventions()


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def compute_source_diagnostics(X_pool: np.ndarray, W: np.ndarray) -> dict:
    """Compute exogenous-source diagnostics from pooled interventional data.

    Diagnostics concern the exogenous noise E = X @ (I - W), since the
    paper's theoretical assumptions are about the sources, not the observables.

    Parameters
    ----------
    X_pool : np.ndarray, shape (N, d)
    W : np.ndarray, shape (d, d) — weighted DAG adjacency (W[i,j] = weight on edge i→j)

    Returns
    -------
    dict with keys:
        mean_abs_excess_kurtosis   float
        min_abs_excess_kurtosis    float
        mean_abs_offdiag_source_correlation  float
    """
    d = W.shape[0]
    E = X_pool @ (np.eye(d) - W)  # recover exogenous sources, shape (N, d)

    kurtoses = np.array(
        [float(stats.kurtosis(E[:, i], fisher=True)) for i in range(d)]
    )
    abs_kurt = np.abs(kurtoses)

    # Off-diagonal correlation
    if E.shape[0] > 1:
        corr = np.corrcoef(E.T)  # (d, d)
        mask = ~np.eye(d, dtype=bool)
        offdiag_corr = float(np.mean(np.abs(corr[mask])))
    else:
        offdiag_corr = float("nan")

    return {
        "mean_abs_excess_kurtosis": float(np.mean(abs_kurt)),
        "min_abs_excess_kurtosis": float(np.min(abs_kurt)),
        "mean_abs_offdiag_source_correlation": offdiag_corr,
    }


def compute_environment_size_stats(interv_ds: dict) -> dict:
    """Compute per-environment count statistics."""
    counts = np.array([interv_ds[k]["X"].shape[0] for k in interv_ds])
    if len(counts) == 0:
        return {
            "environment_size_min": 0,
            "environment_size_max": 0,
            "environment_size_ratio": float("nan"),
            "n_environments": 0,
        }
    return {
        "environment_size_min": int(counts.min()),
        "environment_size_max": int(counts.max()),
        "environment_size_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float("nan"),
        "n_environments": int(len(counts)),
    }


def compute_delta_stats(delta_per_node: np.ndarray) -> dict:
    """Compute summary statistics of the per-node delta vector."""
    return {
        "delta_min": float(np.min(delta_per_node)),
        "delta_max": float(np.max(delta_per_node)),
        "delta_mean": float(np.mean(delta_per_node)),
        "delta_rms": float(np.sqrt(np.mean(delta_per_node ** 2))),
        "delta_std": float(np.std(delta_per_node)),
    }


def compute_p_intervention_stats(p_vector) -> dict:
    """Compute summary statistics of the per-node intervention probability."""
    p = np.atleast_1d(p_vector)
    return {
        "p_intervention_mean": float(np.mean(p)),
        "p_intervention_std": float(np.std(p)),
    }
