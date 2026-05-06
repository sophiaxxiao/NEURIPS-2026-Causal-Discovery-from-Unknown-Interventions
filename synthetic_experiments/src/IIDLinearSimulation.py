
# Code for IID SEM simulations, incorporating custom noise models (the noise for each variable can be have different distributions). 
# Includes function for generating zero-inflated Gaussian noise w/custom parameters. 
# Modified version of the IIDSimulation method from gcastle library.
import castle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import logging
from castle.datasets import DAG

class IIDLinearSimulation(object):
    '''
    Simulate IID datasets for causal structure learning using linear SEM. 

    This class simulates data by traversing the graph in topological order. 
    **Note:** Built-in noise distribution logic (controlled solely by `sem_type`) 
    has been removed. You must provide explicit `noise_models`.

    Parameters
    ----------
    W : np.ndarray
        The weighted adjacency matrix for the target causal graph of shape $[d, d]$.
    n : int, optional
        Number of samples for the standard training dataset. Default is 1000.
    sem_type : str or list of str, optional
        Legacy parameter for validation. 
        - If str, assumes all variables share this type.
        - If list, must be length $d$.
        Note: The actual noise generation logic is now handled by `noise_models`. 
    noise_scale : float, optional
        Scale parameter of noise distribution in linear SEM. Default is 1.0.
    noise_models : dict, list, or np.ndarray, required
        Per-variable custom noise model(s). A per-variable noise model must be supplied
        for every variable you intend to simulate. If `None`, simulation will raise a ValueError.
        
        The mapping may be a dict keyed by variable index, or a list-like of
        length $d$ where the $i$-th entry describes noise for variable $i$.
        
        Supported forms for each entry are:
        1. **Callable:** A function that accepts `size` (or a single positional
           argument `n`) and returns an array of shape $(n,)$.
           Example: ``lambda size: np.random.normal(scale=0.5, size=size)``
        2. **Tuple:** ``(callable, kwargs_dict)``, where the callable will be called
           with ``size=n`` and the provided kwargs.
        3. **Dict:** A distribution spec, e.g., ``{'dist': 'gauss', 'scale': 0.5}``.
           Supported 'dist' keys: 'gauss', 'exp', 'gumbel', 'uniform', 'logistic'.
    '''
    def __init__(self, W, n=1000,
                 sem_type='gauss', noise_scale=1.0,
                 noise_models=None, verbose = False):

        self.B = (W != 0).astype(int)
        self.X = IIDLinearSimulation._simulate_linear_sem(
            W, n, sem_type, noise_scale, noise_models=noise_models)
        if verbose:
            logging.info('Finished synthetic dataset')

    @staticmethod
    def _simulate_linear_sem(W, n, sem_type, noise_scale, noise_models):
        """
        Simulate samples from linear SEM with specified type(s) of noise based on noise_models

        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        n: int
            Number of samples, n=inf mimics population risk.
        sem_type: str or list of str
            If str, all variables follow this noise type, e.g., 'gauss', 'exp', 'gumbel', 'uniform', 'logistic'.
            If list of str, the ith noise variable follows the ith type in the list.
            The length of the list should be equal to the number of variables (i.e., d).
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        X: np.ndarray
            [n, d] sample matrix, [d, d] if n=inf
        """
        def _simulate_single_equation(X, w, scale, sem_type_single, noise_model=None):
            """
            Simulate a single equation in the SEM using only a custom noise model.

            The project no longer includes built-in noise-model dispatch. Callers
            must provide per-variable noise models via the `noise_models`
            argument to `_simulate_linear_sem`/`IIDLinearSimulation`. Supported forms
            for `noise_model`:
            - callable: fn(size=n) or fn(n) -> array of shape (n,)
            - tuple: (callable, kwargs_dict) -> fn(size=n, **kwargs_dict)
            - dict: distribution spec, e.g. {'dist': 'gauss', 'scale': 0.5}

            Returns
            -------
            x: np.ndarray
                [n] simulated variable values
            """
            # Require a custom noise_model; built-in models removed.
            if noise_model is None:
                raise ValueError(
                    "Built-in noise models have been removed. Provide a per-variable "
                    "noise_model (callable, (callable, kwargs), or dict spec) via the "
                    "noise_models argument.")

            # Callable noise model
            if callable(noise_model):
                try:
                    z = noise_model(size=n)
                except TypeError:
                    z = noise_model(n)
                return X @ w + z

            # Tuple form: (callable, kwargs)
            if isinstance(noise_model, tuple) and callable(noise_model[0]):
                fn = noise_model[0]
                kwargs = noise_model[1] if len(noise_model) > 1 else {}
                try:
                    z = fn(size=n, **(kwargs or {}))
                except TypeError:
                    z = fn(n, **(kwargs or {}))
                return X @ w + z

            # Dict spec: interpret a small set of distributions
            if isinstance(noise_model, dict):
                dist = noise_model.get('dist', None)
                s = noise_model.get('scale', scale)
                if dist == 'gauss':
                    z = np.random.normal(scale=s, size=n)
                    return X @ w + z
                if dist == 'exp':
                    z = np.random.exponential(scale=s, size=n)
                    return X @ w + z
                if dist == 'gumbel':
                    z = np.random.gumbel(scale=s, size=n)
                    return X @ w + z
                if dist == 'uniform':
                    low = noise_model.get('low', -scale)
                    high = noise_model.get('high', scale)
                    z = np.random.uniform(low=low, high=high, size=n)
                    return X @ w + z
                if dist == 'logistic':
                    return np.random.binomial(1, sigmoid(X @ w)) * 1.0
                raise ValueError(f"Unknown noise_model dict dist: {dist}")

            raise TypeError("noise_model must be a callable, tuple(callable, kwargs), or dict specifying distribution")

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx = nx.DiGraph(W)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')

        # check if sem_type is a single string
        if isinstance(sem_type, str):
            # If it is, make it a list of size d with the same value
            sem_type = [sem_type] * d
        elif isinstance(sem_type, list):
            # If it's a list, check if the length is equal to d
            if len(sem_type) != d:
                raise ValueError(f"The length of sem_type needs to be equal to {d} (the number of variables).")
            # ensure all elements in the list are strings
            if not all(isinstance(i, str) for i in sem_type):
                raise ValueError("All elements in the sem_type list must be strings.")
        else:
            raise TypeError("sem_type should be either a string or a list of strings")

        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            # Determine if there's a custom noise model for this variable
            noise_model_for_j = None
            if noise_models is not None:
                # noise_models can be a dict (index -> model) or a list-like of length d
                if isinstance(noise_models, dict):
                    noise_model_for_j = noise_models.get(j, None)
                else:
                    # try list-like
                    try:
                        if len(noise_models) == d:
                            noise_model_for_j = noise_models[j]
                    except Exception:
                        noise_model_for_j = None

            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], sem_type[j], noise_model=noise_model_for_j)
        return X

