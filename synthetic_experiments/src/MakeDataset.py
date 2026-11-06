# Generate data from a linear SEM with mean-shifted interventions.
# 
from IIDLinearSimulation import IIDLinearSimulation
import numpy as np
from castle.datasets import DAG
from functools import partial
import matplotlib.pyplot as plt

class ShiftSimulation:
    def compute_per_node_delta_from_dag(self, scalar=1.0, noise_variances=None):
        """
        Computes per-node delta as a scalar multiple of the analytical standard deviation of each node from the DAG and noise variances.
        Sigma_analytical = np.linalg.inv(I - W.T) @ D @ np.linalg.inv(I - W.T).T

        Args:
            scalar (float): Multiple of the standard deviation to use for the mean shift.
            noise_variances (array-like or None): Per-node noise variances. If None, all set to 1.
        Sets:
            self.delta_per_node: np.ndarray of shape [n_nodes]
        """
        W = self.weighted_random_dag
        n = W.shape[0]
        I = np.eye(n)
        if noise_variances is None:
            D = np.eye(n)
        else:
            D = np.diag(noise_variances)
        try:
            IW_inv = np.linalg.inv(I - W.T) 
        except np.linalg.LinAlgError:
            raise ValueError("I - W is singular; check that W is a valid DAG adjacency matrix.")
        Sigma = IW_inv @ D @ IW_inv.T
        variances = np.diag(Sigma)
        self.delta_per_node = scalar * np.sqrt(variances)
        return self.delta_per_node

    def compute_per_node_delta_from_observational(self, scalar=1.0):
        """
        Computes a per-node delta as a scalar multiple of the empirical variance of each node in the observational data.
        
        Stores the result in self.delta_per_node (np.ndarray of shape [n_nodes]).
        Args:
            scalar (float): The multiple of the variance to use for the mean shift.
        """
        if self.dataset_observational is None:
            self.simulate_observational() 
        X_obs = self.dataset_observational.X
        variances = np.var(X_obs, axis=0)
        assert variances.shape[0] == self.n_nodes, "Variance shape does not match number of nodes."
        self.delta_per_node = scalar * variances
        return self.delta_per_node
    
    def __init__(self, n_nodes=10, n_edges=10, n_samples=10000, n_observational = int(1e5), weighted_random_dag = None, p_intervention = 0.5, delta=1, seed=42, intervention_mode='bernoulli', include_no_intervention=False, single_node_probs=None):
        """
        Initialize simulation parameters.

        Args:
            n_nodes (int): Number of nodes in the graph.
            n_edges (int): Expected number of edges.
            n_samples (int): Number of total samples for the pooled interventional dataset.
            n_observational (int): Number of samples for the OBSERVATIONAL (un-intervened) dataset. If None, defaults to 100,000.
            weighted_random_dag (np.ndarray or None): Predefined weighted adjacency matrix for the DAG
            seed (int): Random seed for reproducibility; if no weighted_random_dag is provided, one will be generated based on this seed.
            p_intervention (float): Probability of intervention per node. Only used when intervention_mode='bernoulli'.
            delta (float): Magnitude of the mean shift for interventions.
            intervention_mode (str): 'bernoulli' (default) — each node independently intervened with
                probability p_intervention per sample; or 'single_node' — exactly one node intervened
                per environment, each drawn ~ Uniform({0,...,d-1}).
            include_no_intervention (bool): Only used when intervention_mode='single_node'. If True,
                draws from d+1 categories (d nodes + no intervention), each with probability 1/(d+1).
                The no-intervention samples are simulated from the observational distribution.
            single_node_probs (array-like or None): Only used when intervention_mode='single_node'.
                If provided, overrides the uniform categorical distribution with these probabilities.
                Must be a finite, non-negative array of length n_nodes (when include_no_intervention=False)
                or n_nodes+1 (when include_no_intervention=True), summing to 1.0.
                When None (default), the original uniform behaviour is preserved.
        """
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.k = None # Number of environments, to be determined by the number of unique randomly generated interventions.
        self.p_intervention = p_intervention
        self.delta = delta
        self.intervention_mode = intervention_mode
        self.include_no_intervention = include_no_intervention

        # Validate and store single_node_probs
        if single_node_probs is not None:
            probs = np.asarray(single_node_probs, dtype=float)
            n_cat = n_nodes + (1 if include_no_intervention else 0)
            if probs.shape != (n_cat,):
                raise ValueError(
                    f"single_node_probs must have length {n_cat} "
                    f"(n_nodes={n_nodes}, include_no_intervention={include_no_intervention}), "
                    f"got shape {probs.shape}."
                )
            if not np.all(np.isfinite(probs)):
                raise ValueError("single_node_probs contains non-finite values.")
            if np.any(probs < 0):
                raise ValueError("single_node_probs contains negative values.")
            if not np.isclose(probs.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"single_node_probs must sum to 1.0, got {probs.sum():.6f}."
                )
            self.single_node_probs = probs
        else:
            self.single_node_probs = None
        self.delta_per_node = None # To be computed later based on observational data or DAG structure.
        self.n_samples = n_samples
        self.seed = seed
        self.n_observational = n_observational
        # State containers
        self.weighted_random_dag = weighted_random_dag # If None, will be generated.
        if weighted_random_dag is not None and weighted_random_dag.shape != (n_nodes, n_nodes):
            raise ValueError(f"Provided weighted_random_dag has shape {weighted_random_dag.shape}, expected ({n_nodes}, {n_nodes}).")
        self.intervened_nodes = None
        self.alphas = None

        # Results containers
        self.dataset_observational = None
        self.datasets_interventional = []

        # Base noise configuration (observational)
        self.noise_dict_base = {}

    def _set_seed(self):
        """Sets the random seed."""
        np.random.seed(self.seed)
 
    def setup_interventions(self, verbose=False):
        """
        Sets up the intervention targets and their corresponding mean shifts.

        intervention_mode='bernoulli': each node independently intervened with probability
            p_intervention; unique patterns become environments.
        intervention_mode='single_node': one environment per node, each intervening on exactly
            that node. n_samples is split evenly across the n_nodes environments.
        """
        self._set_seed()

        if self.intervention_mode == 'single_node':
            # n_categories: d nodes, plus optionally a no-intervention category (all-zeros)
            n_categories = self.n_nodes + (1 if self.include_no_intervention else 0)
            if self.single_node_probs is not None:
                # Nonuniform categorical distribution (backward-compatible extension).
                node_assignments = np.random.choice(
                    n_categories, size=self.n_samples, p=self.single_node_probs
                )
            else:
                node_assignments = np.random.randint(0, n_categories, size=self.n_samples)
            # category d (== n_nodes) maps to the all-zeros pattern (no intervention)

            intervention_matrix = np.zeros((self.n_samples, self.n_nodes), dtype=int)
            interv_mask = node_assignments < self.n_nodes          # samples that DO intervene
            intervention_matrix[np.where(interv_mask), node_assignments[interv_mask]] = 1

            intervention_dict = {}
            for j in range(n_categories):
                count = int(np.sum(node_assignments == j))
                if count == 0:
                    continue
                if j < self.n_nodes:
                    pattern = tuple(1 if i == j else 0 for i in range(self.n_nodes))
                else:
                    pattern = tuple(0 for _ in range(self.n_nodes))  # no-intervention
                intervention_dict[pattern] = count

            self.intervention_matrix = intervention_matrix
            self.intervention_dict   = intervention_dict
            self.intervened_nodes    = [np.flatnonzero(row) for row in intervention_matrix]
            self.k                   = n_categories
            if verbose:
                print(f'Single-node interventions: {n_categories} environments '
                      f'({"incl." if self.include_no_intervention else "excl."} no-intervention), '
                      f'counts ~ Multinomial(n={self.n_samples}, p=1/{n_categories}): '
                      f'{list(intervention_dict.values())}')
            return intervention_dict

        # --- bernoulli (default) ---
        if verbose:
            print(f'Setting up interventions with p_intervention={self.p_intervention} on {self.n_nodes} nodes across {self.n_samples} samples...')

        # Generate k bernoulli random vectors of length n_nodes (one per environment)
        intervention_matrix = np.random.binomial(1, self.p_intervention, size=(self.n_samples, self.n_nodes))

        # store raw matrix for downstream use
        self.intervention_matrix = intervention_matrix

        # Count unique intervention patterns (rows) and their frequencies
        intervention_sets, counts = np.unique(intervention_matrix, axis=0, return_counts=True)

        if verbose:
            print(f'Unique intervention targets found: {len(intervention_sets)}')

        self.k = len(intervention_sets)

        intervention_dict = {tuple(int(x) for x in intervention_sets[i]): int(counts[i])
                              for i in range(len(intervention_sets))}

        self.intervention_dict = intervention_dict
        self.intervened_nodes  = [np.flatnonzero(row) for row in intervention_matrix]

        return intervention_dict 
    
    def simulate_observational(self, n_obs=None):
        """
        Simulates the base environment (Env 0) with NO interventions.
        Returns an IIDLinearSimulation dataset object.
        """
        if self.weighted_random_dag is None:
            raise ValueError("DAG not found. Please generate DAG first before simulating observational data.")

        # Define base noise: Normal(0, 1)
        self.noise_dict_base = {
            i: lambda size: np.random.normal(loc=0.0, scale=1.0, size=size)
            for i in range(self.n_nodes)
        }

        self.dataset_observational = IIDLinearSimulation(
            W=self.weighted_random_dag,
            n=self.n_observational if n_obs is None else n_obs, # use self.count instead to match total samples FOR EACH GIVEN environments.
            noise_models=self.noise_dict_base
        )
        return self.dataset_observational 

    def simulate_interventions(self, verbose=False):
        """
        Simulates the interventional environments based on setup, using per-node delta if available.
        Returns a dictionary mapping environment index to dataset object and metadata.
        """
        if self.dataset_observational is None:
            if verbose:
                print("Observational dataset not found. Simulating observational dataset first...")
            self.simulate_observational()

        if self.intervened_nodes is None:
            if verbose:
                print("Intervention sets not found. Setting up interventions first...")
            self.setup_interventions()

        # Use per-node delta ONLY if available. ELSE RAISE ERROR IF delta is None.
        delta_per_node = getattr(self, 'delta_per_node', None) 

        self.datasets_interventional = {}
        env_idx = 0

        for (intervention_pattern, count) in self.intervention_dict.items():
            nodes = np.flatnonzero(intervention_pattern).tolist() # Get indices of intervened nodes for this environment. Could be empty if no intervention
            pattern = np.array(intervention_pattern) # Convert tuple of intervention pattern back to numpy array for indexing. 

            if verbose:
                print(f"Processing intervention pattern {intervention_pattern} with count {count}... on nodes {nodes}")

            noise_dict_i = {
                i: lambda size: np.random.normal(loc=0, scale=1.0, size=size)
                for i in range(self.n_nodes)
            }

            if len(nodes) > 0:
                for node_idx in nodes:
                    if delta_per_node is not None:
                        node_delta = delta_per_node[int(node_idx)]
                    else:
                        raise ValueError("delta_per_node is not set. Please compute or set it before simulating interventions.")
                    #print(f"Setting intervention on node {node_idx} with delta {node_delta}")
                    noise_dict_i[int(node_idx)] = (lambda mu: (lambda size: np.random.normal(loc=mu, scale=1.0, size=size)))(node_delta)
            if verbose:
                print(f"Simulating Env {env_idx+1}: Intervention pattern {intervention_pattern}, nodes={list(nodes)}")

            sim_i = IIDLinearSimulation(
                    W=self.weighted_random_dag,
                    noise_models=noise_dict_i,
                    n = count  # Use the count of samples for this specific intervention pattern to ensure the total number of samples across all environments matches self.n_samples.
                )
            X, B = sim_i.X, sim_i.B
            self.datasets_interventional[intervention_pattern] = {
                    'dataset_object': sim_i,
                    'X': X,
                    'B': B,
                    'intervened_nodes': list(nodes)
                }
            
            X_mean = X.mean(axis=0)
            if verbose:
                print(f"Env {env_idx+1} mean: {X_mean}")
            del noise_dict_i, sim_i  # Clean up large objects to save memory.

            env_idx += 1

        return self.datasets_interventional # Dictionary mapping environment index to dataset object and metadata (intervened nodes, etc.)
    
    def _generate_datasets(self):
        """Convenience method to generate observational and interventional datasets in sequence."""
        self.simulate_observational()
        self.compute_per_node_delta_from_observational(scalar=self.delta)
        self.setup_interventions()
        self.simulate_interventions()

if __name__ == "__main__":
    dag = DAG.erdos_renyi(n_nodes=5, n_edges=7, seed=42, edges = [0.5, 2]) # Example DAG generation for testing. In practice, you would want to generate this based on the n_nodes and n_edges parameters, or allow it to be passed in as an argument.

    # Example usage
    sim = ShiftSimulation(
        n_nodes=5,
        n_edges=7,
        n_samples=10000,
        p_intervention=0.3,
        weighted_random_dag=dag,
        delta=2.0,
        seed=42
    )
    sim.simulate_observational()
    sim.compute_per_node_delta_from_observational(scalar=2.0)
    sim.setup_interventions()
    interventions = sim.simulate_interventions()
    for pattern, data in interventions.items():
        print(f"Intervention Pattern: {pattern}, Intervened Nodes: {data['intervened_nodes']}, Dataset Shape: {data['X'].shape}")    

    # Alternative usage
    sim._generate_datasets()
    for pattern, data in sim.datasets_interventional.items():
        print(f"Intervention Pattern: {pattern}, Intervened Nodes: {data['intervened_nodes']}, Dataset Shape: {data['X'].shape}")