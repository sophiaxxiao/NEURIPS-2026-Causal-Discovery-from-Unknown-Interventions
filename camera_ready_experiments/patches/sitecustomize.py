# Loaded at interpreter start-up (including joblib/loky workers) when this directory is on
# PYTHONPATH. Applies the Ablation B variance-scaling patch; see run_ablation_B_variance.py.
import os, sys
_ABL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "synthetic_experiments", "ablation")
for _p in (_ABL, os.path.join(_ABL, "..", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import numpy as _np
    import ablation_simulator as _a

    def _delta_vector_var(self, scalar=1.0):
        if self.dataset_observational is None:
            self.simulate_observational()
        X_obs = self.dataset_observational.X
        self.delta_per_node = scalar * _np.asarray(self.delta_vector, dtype=float) * _np.var(X_obs, axis=0)
        return self.delta_per_node

    _a.AblationShiftSimulation.compute_per_node_delta_vector = _delta_vector_var
    os.environ["ABLATION_B_VARIANCE_PATCH"] = "1"
except Exception as _e:  # noqa: BLE001
    sys.stderr.write(f"sitecustomize: Ablation B patch NOT applied: {_e}\n")
