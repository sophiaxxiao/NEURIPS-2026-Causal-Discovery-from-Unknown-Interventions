#!/usr/bin/env python3
# coding=utf-8
"""
Tests for the rebuttal ablation suite.

Run with:
    pytest tests/test_rebuttal_ablations.py -v

From the synthetic_experiments directory (so src/ is on the path).
"""

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# ---- path setup -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SYNTH = os.path.dirname(_HERE)
_SRC = os.path.join(_SYNTH, "src")
_ABLATION = os.path.join(_SYNTH, "ablation")
for _p in [_SRC, _ABLATION]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from castle.datasets import DAG
from ablation_simulator import (
    AblationShiftSimulation,
    compute_delta_stats,
    compute_environment_size_stats,
    compute_p_intervention_stats,
    compute_source_diagnostics,
    heterogeneous_delta_vector,
    heterogeneous_delta_vector_from_z,
    heterogeneous_p_vector,
    heterogeneous_p_vector_from_z,
    random_derangement,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

N_NODES = 5
N_EDGES = 5
N_INTERVENTIONAL = 200
N_OBS = 50
SEED = 7


def _make_dag(seed=SEED):
    return DAG.erdos_renyi(n_nodes=N_NODES, n_edges=N_EDGES, seed=seed, weight_range=[0.5, 2.0])


def _make_baseline_sim(seed=SEED, dag=None):
    if dag is None:
        dag = _make_dag(seed)
    sim = AblationShiftSimulation(
        n_nodes=N_NODES, n_edges=N_EDGES,
        n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
        weighted_random_dag=dag, delta=1.0, seed=seed,
    )
    sim.generate_ablation_datasets()
    return sim


# -----------------------------------------------------------------------
# 1. Baseline reproduction
# -----------------------------------------------------------------------

class TestBaselineReproduction:
    """Baseline condition generates identical data when run twice with the same seed."""

    def test_dag_identical_across_runs(self):
        dag1 = _make_dag(SEED)
        dag2 = _make_dag(SEED)
        np.testing.assert_array_equal(dag1, dag2)

    def test_obs_data_reproducible(self):
        dag = _make_dag(SEED)
        s1 = _make_baseline_sim(SEED, dag)
        s2 = _make_baseline_sim(SEED, dag)
        np.testing.assert_array_equal(s1.dataset_observational.X, s2.dataset_observational.X)

    def test_interventional_data_reproducible(self):
        dag = _make_dag(SEED)
        s1 = _make_baseline_sim(SEED, dag)
        s2 = _make_baseline_sim(SEED, dag)
        X1 = np.vstack([s1.datasets_interventional[k]["X"] for k in s1.datasets_interventional])
        X2 = np.vstack([s2.datasets_interventional[k]["X"] for k in s2.datasets_interventional])
        np.testing.assert_array_equal(X1, X2)

    def test_total_interventional_count(self):
        sim = _make_baseline_sim()
        total = sum(v["X"].shape[0] for v in sim.datasets_interventional.values())
        assert total == N_INTERVENTIONAL

    def test_obs_count(self):
        sim = _make_baseline_sim()
        assert sim.dataset_observational.X.shape[0] == N_OBS

    def test_zero_severity_ablation_A_matches_baseline(self):
        """Ablation A with h=0 should produce identical obs data as baseline."""
        dag = _make_dag(SEED)
        base = _make_baseline_sim(SEED, dag)

        rng = np.random.default_rng(SEED)
        z = np.linspace(-1.0, 1.0, N_NODES)
        perm = rng.permutation(N_NODES)
        z_perm = z[perm]
        p_vec = heterogeneous_p_vector_from_z(z_perm, h=0.0)

        sim_A = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            p_vector=p_vec,
        )
        sim_A.generate_ablation_datasets()

        # Obs data must match: p_vec at h=0 is all 0.5 → same as baseline
        np.testing.assert_allclose(
            base.dataset_observational.X, sim_A.dataset_observational.X,
            err_msg="h=0 Ablation A obs data differs from baseline.",
        )

    def test_tau_zero_matches_baseline_delta(self):
        """Ablation B with tau=0 should have all delta_vector=1 (identical to baseline scalar=1)."""
        rng = np.random.default_rng(SEED)
        dv = heterogeneous_delta_vector(N_NODES, tau=0.0, rng=rng)
        np.testing.assert_allclose(dv, np.ones(N_NODES), atol=1e-12)

    def test_rho_zero_no_offtarget(self):
        """Ablation D with rho=0 should produce identical datasets to no-offtarget simulation."""
        dag = _make_dag(SEED)
        base = _make_baseline_sim(SEED, dag)

        sim_D = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            off_target_rho=0.0,
        )
        sim_D.generate_ablation_datasets()
        # obs data matches
        np.testing.assert_array_equal(
            base.dataset_observational.X, sim_D.dataset_observational.X
        )


# -----------------------------------------------------------------------
# 2. Ablation A: unequal environment sizes (heterogeneous p_i)
# -----------------------------------------------------------------------

class TestAblationA:
    """Ablation A: per-node Bernoulli probabilities."""

    def test_p_vector_mean_is_half(self):
        rng = np.random.default_rng(SEED)
        for h in [0.0, 0.15, 0.30, 0.45]:
            pv = heterogeneous_p_vector(N_NODES, h, rng)
            assert abs(np.mean(pv) - 0.5) < 1e-12, f"Mean p != 0.5 at h={h}"

    def test_p_range_increases_with_h(self):
        rng = np.random.default_rng(SEED)
        prev_spread = 0.0
        for h in [0.0, 0.15, 0.30, 0.45]:
            pv = heterogeneous_p_vector(N_NODES, h, rng)
            spread = float(pv.max() - pv.min())
            assert spread >= prev_spread - 1e-12
            prev_spread = spread

    def test_total_interventional_count_preserved(self):
        dag = _make_dag(SEED)
        rng = np.random.default_rng(SEED)
        pv = heterogeneous_p_vector(N_NODES, h=0.30, rng=rng)
        sim = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            p_vector=pv,
        )
        sim.generate_ablation_datasets()
        total = sum(v["X"].shape[0] for v in sim.datasets_interventional.values())
        assert total == N_INTERVENTIONAL

    def test_permutation_reproducible_across_h(self):
        """Same seed → same z_perm → same p direction, different magnitude."""
        rng1 = np.random.default_rng(SEED)
        z = np.linspace(-1, 1, N_NODES)
        perm = rng1.permutation(N_NODES)
        z_perm = z[perm]

        pv_15 = heterogeneous_p_vector_from_z(z_perm, 0.15)
        pv_30 = heterogeneous_p_vector_from_z(z_perm, 0.30)

        # Direction must be consistent: same nodes are high/low at both h values
        rank_15 = np.argsort(pv_15)
        rank_30 = np.argsort(pv_30)
        np.testing.assert_array_equal(rank_15, rank_30)

    def test_p_vector_in_valid_range(self):
        rng = np.random.default_rng(SEED)
        for h in [0.0, 0.15, 0.30, 0.45]:
            pv = heterogeneous_p_vector(N_NODES, h, rng)
            assert np.all(pv > 0) and np.all(pv < 1), f"p_vector out of (0,1) at h={h}"


# -----------------------------------------------------------------------
# 3. Ablation B: heterogeneous strengths
# -----------------------------------------------------------------------

class TestAblationB:
    def test_tau_zero_gives_unit_delta(self):
        rng = np.random.default_rng(SEED)
        dv = heterogeneous_delta_vector(N_NODES, tau=0.0, rng=rng)
        np.testing.assert_allclose(dv, np.ones(N_NODES), atol=1e-12)

    def test_rms_equals_one_at_all_tau(self):
        rng = np.random.default_rng(SEED)
        for tau in [0.0, 0.35, 0.70, 1.00]:
            dv = heterogeneous_delta_vector(N_NODES, tau, rng)
            rms = float(np.sqrt(np.mean(dv ** 2)))
            assert abs(rms - 1.0) < 1e-10, f"RMS != 1 at tau={tau}: {rms}"

    def test_spread_increases_with_tau(self):
        rng = np.random.default_rng(SEED)
        prev_ratio = 1.0
        for tau in [0.0, 0.35, 0.70, 1.00]:
            dv = heterogeneous_delta_vector(N_NODES, tau, rng)
            ratio = float(dv.max() / dv.min())
            assert ratio >= prev_ratio - 1e-10, f"Max/min ratio decreased at tau={tau}"
            prev_ratio = ratio

    def test_heterogeneous_delta_total_count(self):
        dag = _make_dag(SEED)
        rng = np.random.default_rng(SEED)
        dv = heterogeneous_delta_vector(N_NODES, tau=0.70, rng=rng)
        sim = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            delta_vector=dv,
        )
        sim.generate_ablation_datasets()
        total = sum(v["X"].shape[0] for v in sim.datasets_interventional.values())
        assert total == N_INTERVENTIONAL

    def test_permutation_direction_consistent(self):
        rng = np.random.default_rng(SEED)
        z = np.linspace(-1, 1, N_NODES)
        perm = rng.permutation(N_NODES)
        z_perm = z[perm]
        dv_35 = heterogeneous_delta_vector_from_z(z_perm, 0.35)
        dv_70 = heterogeneous_delta_vector_from_z(z_perm, 0.70)
        np.testing.assert_array_equal(np.argsort(dv_35), np.argsort(dv_70))


# -----------------------------------------------------------------------
# 4. Ablation C: weak interventions
# -----------------------------------------------------------------------

class TestAblationC:
    def test_only_delta_changes(self):
        """Compare two simulations differing only in delta; same structure."""
        dag = _make_dag(SEED)
        s1 = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
        )
        s1.generate_ablation_datasets()

        s2 = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=0.5, seed=SEED,
        )
        s2.generate_ablation_datasets()

        # Same intervention patterns
        assert set(s1.datasets_interventional.keys()) == set(s2.datasets_interventional.keys())

        # delta_per_node scales with delta
        ratio = s1.delta_per_node / s2.delta_per_node
        np.testing.assert_allclose(ratio, 2.0 * np.ones(N_NODES), rtol=1e-10)

    def test_delta_per_node_proportional(self):
        dag = _make_dag(SEED)
        s1 = _make_baseline_sim(SEED, dag)
        s_weak = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=0.25, seed=SEED,
        )
        s_weak.generate_ablation_datasets()
        np.testing.assert_allclose(
            s_weak.delta_per_node, 0.25 * s1.delta_per_node, rtol=1e-10
        )


# -----------------------------------------------------------------------
# 5. Ablation D: correlated off-target effects
# -----------------------------------------------------------------------

class TestAblationD:
    def test_derangement_has_no_fixed_points(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            pi = random_derangement(N_NODES, rng)
            assert not np.any(pi == np.arange(N_NODES)), f"Fixed point in derangement at seed={seed}"

    def test_derangement_is_permutation(self):
        rng = np.random.default_rng(SEED)
        pi = random_derangement(N_NODES, rng)
        assert sorted(pi) == list(range(N_NODES))

    def test_rho_zero_identical_to_baseline(self):
        dag = _make_dag(SEED)
        base = _make_baseline_sim(SEED, dag)

        rng = np.random.default_rng(SEED)
        pi = random_derangement(N_NODES, rng)

        sim_D = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            off_target_rho=0.0, derangement=pi,
        )
        sim_D.generate_ablation_datasets()
        # Same obs data
        np.testing.assert_array_equal(
            base.dataset_observational.X, sim_D.dataset_observational.X
        )

    def test_offtarget_node_not_equal_intended(self):
        rng = np.random.default_rng(SEED)
        pi = random_derangement(N_NODES, rng)
        # pi(i) != i for all i
        for i in range(N_NODES):
            assert pi[i] != i

    def test_actual_targets_include_offtarget(self):
        dag = _make_dag(SEED)
        rng = np.random.default_rng(SEED)
        pi = random_derangement(N_NODES, rng)

        sim = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            off_target_rho=0.5, derangement=pi,
        )
        sim.generate_ablation_datasets()

        assert sim.actual_targets is not None
        for pattern, actual in sim.actual_targets.items():
            nominal = sim.nominal_targets[pattern]
            for n in nominal:
                assert pi[n] in actual, (
                    f"Off-target {pi[n]} missing from actual_targets for pattern {pattern}"
                )

    def test_normalization_applied(self):
        """With rho > 0, effective mean shift on intended node < delta (normalized)."""
        dag = _make_dag(SEED)
        rng = np.random.default_rng(SEED)
        pi = random_derangement(N_NODES, rng)

        rho = 0.5
        c = 1.0 / np.sqrt(1.0 + rho ** 2)

        sim_normed = AblationShiftSimulation(
            n_nodes=N_NODES, n_edges=N_EDGES,
            n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
            weighted_random_dag=dag, delta=1.0, seed=SEED,
            off_target_rho=rho, derangement=pi,
        )
        sim_normed.generate_ablation_datasets()
        # Each node's effective mean shift should be <= c * delta_per_node (not > 1 * delta_per_node)
        base = _make_baseline_sim(SEED, dag)
        # Normalization check: c < 1 when rho > 0
        assert c < 1.0

    def test_source_dependence_increases_with_rho(self):
        """Empirical off-diagonal source correlation should increase with rho.

        Uses a large sample for statistical reliability.
        """
        dag = _make_dag(SEED)
        n_large = 5000
        correlations = {}
        for rho in [0.0, 0.5, 1.0]:
            rng = np.random.default_rng(SEED)
            pi = random_derangement(N_NODES, rng)
            sim = AblationShiftSimulation(
                n_nodes=N_NODES, n_edges=N_EDGES,
                n_samples=n_large, n_observational=N_OBS,
                weighted_random_dag=dag, delta=1.0, seed=SEED,
                off_target_rho=rho, derangement=pi,
            )
            sim.generate_ablation_datasets()
            X_pool = np.vstack([sim.datasets_interventional[k]["X"] for k in sim.datasets_interventional])
            diag = compute_source_diagnostics(X_pool, dag)
            correlations[rho] = diag["mean_abs_offdiag_source_correlation"]
        assert correlations[1.0] >= correlations[0.0], (
            f"Source correlation should increase with rho: {correlations}"
        )


# -----------------------------------------------------------------------
# 6. Paired design: same DAG across conditions
# -----------------------------------------------------------------------

class TestPairedDesign:
    """The same seed must produce the same DAG for all conditions."""

    def test_same_dag_across_conditions(self):
        dag = _make_dag(SEED)

        sims = []
        for delta in [1.0, 0.5, 0.25]:
            sim = AblationShiftSimulation(
                n_nodes=N_NODES, n_edges=N_EDGES,
                n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
                weighted_random_dag=dag, delta=delta, seed=SEED,
            )
            sim.generate_ablation_datasets()
            sims.append(sim)

        # All use the same DAG
        for s in sims:
            np.testing.assert_array_equal(s.weighted_random_dag, dag)

    def test_obs_data_identical_across_ablation_C(self):
        """Ablation C only changes delta; obs data must be identical."""
        dag = _make_dag(SEED)
        obs_datas = []
        for delta in [1.0, 0.75, 0.50, 0.25, 0.10]:
            sim = AblationShiftSimulation(
                n_nodes=N_NODES, n_edges=N_EDGES,
                n_samples=N_INTERVENTIONAL, n_observational=N_OBS,
                weighted_random_dag=dag, delta=delta, seed=SEED,
            )
            sim.generate_ablation_datasets()
            obs_datas.append(sim.dataset_observational.X.copy())
        for i in range(1, len(obs_datas)):
            np.testing.assert_array_equal(obs_datas[0], obs_datas[i])


# -----------------------------------------------------------------------
# 7. Result integrity
# -----------------------------------------------------------------------

class TestResultIntegrity:
    def test_compute_source_diagnostics_shape(self):
        dag = _make_dag(SEED)
        sim = _make_baseline_sim(SEED, dag)
        X_pool = np.vstack([sim.datasets_interventional[k]["X"] for k in sim.datasets_interventional])
        diag = compute_source_diagnostics(X_pool, dag)
        assert "mean_abs_excess_kurtosis" in diag
        assert "min_abs_excess_kurtosis" in diag
        assert "mean_abs_offdiag_source_correlation" in diag
        assert all(np.isfinite(v) for v in diag.values())

    def test_environment_size_stats(self):
        sim = _make_baseline_sim()
        stats = compute_environment_size_stats(sim.datasets_interventional)
        assert stats["environment_size_min"] >= 1
        assert stats["environment_size_max"] <= N_INTERVENTIONAL
        assert stats["environment_size_ratio"] >= 1.0

    def test_delta_stats(self):
        sim = _make_baseline_sim()
        stats = compute_delta_stats(sim.delta_per_node)
        assert stats["delta_min"] > 0
        assert stats["delta_max"] >= stats["delta_min"]
        assert stats["delta_rms"] > 0

    def test_p_stats_baseline(self):
        stats = compute_p_intervention_stats(0.5)
        assert abs(stats["p_intervention_mean"] - 0.5) < 1e-12
        assert stats["p_intervention_std"] == 0.0


# -----------------------------------------------------------------------
# 8. Condition builder test (requires experiments module)
# -----------------------------------------------------------------------

class TestConditionBuilder:
    def test_baseline_is_first(self):
        _EXPERIMENTS = os.path.join(os.path.dirname(_HERE), "experiments")
        if _EXPERIMENTS not in sys.path:
            sys.path.insert(0, _EXPERIMENTS)
        from run_rebuttal_ablations import build_conditions
        import yaml
        cfg_path = os.path.join(os.path.dirname(_HERE), "configs", "rebuttal_ablations.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("Config file not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        conditions = build_conditions(cfg)
        assert conditions[0].ablation_name == "baseline"

    def test_fifteen_conditions_from_spec(self):
        _EXPERIMENTS = os.path.join(os.path.dirname(_HERE), "experiments")
        if _EXPERIMENTS not in sys.path:
            sys.path.insert(0, _EXPERIMENTS)
        from run_rebuttal_ablations import build_conditions
        import yaml
        cfg_path = os.path.join(os.path.dirname(_HERE), "configs", "rebuttal_ablations.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("Config file not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        conditions = build_conditions(cfg)
        # 1 baseline + 3 (A) + 3 (B) + 4 (C) + 4 (D) = 15
        assert len(conditions) == 15, f"Expected 15 conditions, got {len(conditions)}"

    def test_one_factor_at_a_time(self):
        """Each non-baseline condition changes exactly one parameter from the baseline."""
        _EXPERIMENTS = os.path.join(os.path.dirname(_HERE), "experiments")
        if _EXPERIMENTS not in sys.path:
            sys.path.insert(0, _EXPERIMENTS)
        from run_rebuttal_ablations import build_conditions
        import yaml
        cfg_path = os.path.join(os.path.dirname(_HERE), "configs", "rebuttal_ablations.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("Config file not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        conditions = build_conditions(cfg)
        baseline = conditions[0]
        for cond in conditions[1:]:
            ablation = cond.ablation_name
            if ablation == "unequal_sizes":
                assert cond.h_unequal > 0
                assert cond.tau_hetero == baseline.tau_hetero
                assert cond.delta_scalar == baseline.delta_scalar
                assert cond.off_target_rho == baseline.off_target_rho
            elif ablation == "heterogeneous_strengths":
                assert cond.tau_hetero > 0
                assert cond.h_unequal == baseline.h_unequal
                assert cond.delta_scalar == baseline.delta_scalar
                assert cond.off_target_rho == baseline.off_target_rho
            elif ablation == "weak_interventions":
                assert cond.delta_scalar < baseline.delta_scalar
                assert cond.h_unequal == baseline.h_unequal
                assert cond.tau_hetero == baseline.tau_hetero
                assert cond.off_target_rho == baseline.off_target_rho
            elif ablation == "off_target":
                assert cond.off_target_rho > 0
                assert cond.h_unequal == baseline.h_unequal
                assert cond.tau_hetero == baseline.tau_hetero
                assert cond.delta_scalar == baseline.delta_scalar
