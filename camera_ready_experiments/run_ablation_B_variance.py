"""Rerun Ablation B (heterogeneous intervention strengths) with the shift scaled by the
observational VARIANCE, consistent with the shared baseline and with Ablations A, C, D.

In the student's code, compute_per_node_delta_vector (used only for Ablation B, tau > 0)
scales by the observational SD, while the shared baseline (tau = 0) and every other
experiment use compute_per_node_delta_from_observational, which scales by the variance.
This wrapper patches that one method and otherwise runs the student's runner unchanged.

    PYTHONPATH=patches .venv/bin/python run_ablation_B_variance.py --n-jobs 8

The same patch is in patches/sitecustomize.py; putting that directory on PYTHONPATH is what
makes it apply inside joblib worker processes (a plain monkeypatch here only affects the
parent process, i.e. --n-jobs 1).
"""
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "..", "synthetic_experiments", "ablation")
sys.path.insert(0, ABL)
sys.path.insert(0, os.path.join(ABL, "..", "src"))
import ablation_simulator  # noqa: E402


def compute_per_node_delta_vector_var(self, scalar=1.0):
    X_obs = self.dataset_observational.X
    self.delta_per_node = scalar * np.asarray(self.delta_vector, dtype=float) * np.var(X_obs, axis=0)
    return self.delta_per_node


ablation_simulator.AblationShiftSimulation.compute_per_node_delta_vector = compute_per_node_delta_vector_var

if __name__ == "__main__":
    extra = sys.argv[1:]
    sys.argv = [os.path.join(ABL, "experiments", "run_rebuttal_ablations.py"),
                "--config", os.path.join(HERE, "configs", "ablation_B_only.yaml"),
                "--results-dir", os.path.join(HERE, "results", "ablation_B_var"), *extra]
    runpy.run_path(sys.argv[0], run_name="__main__")
