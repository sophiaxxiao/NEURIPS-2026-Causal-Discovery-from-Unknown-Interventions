# Implementation of Structural Intervention Distance (SID) between DAGs and CPDAGs.
# Adapted from the R code of Jonas Peters and Peter Bühlmann (2015), with some optimizations.

import numpy as np
import networkx as nx


# ── private helpers (SID) ─────────────────────────────────────────────────────

def _is_cpdag(adj):
    """True if adj has any undirected edge (both adj[i,j] and adj[j,i] nonzero)."""
    return bool(np.any((adj != 0) & (adj.T != 0)))


def _compute_path_matrix(G):
    """ 
    Transitive closure + identity.
    PM[i,j] = True iff there is a directed path from i to j, or i == j.
    Uses repeated squaring (Peters & Bühlmann 2015, computePathMatrix.R).
    """
    p = G.shape[0]
    PM = (np.eye(p) + (G != 0).astype(float)) > 0
    k  = max(1, int(np.ceil(np.log2(p)))) if p > 1 else 1
    for _ in range(k):
        PM = (PM @ PM) > 0
    return PM


def _compute_path_matrix2(G, cond_set, PM1):
    """
    Like _compute_path_matrix but with all outgoing edges removed from cond_set nodes.
    Returns PM1 unchanged when cond_set is empty.
    """
    if not cond_set:
        return PM1
    G2 = G.copy().astype(float)
    for c in cond_set:
        G2[c, :] = 0.0
    return _compute_path_matrix(G2)


def _reachable_non_causal(adj_orig, i, cond_set, PM, PM2):
    """
    For fixed source i and conditioning set cond_set, return a boolean array
    of length p where result[j] = True iff j is reachable from i via a
    non-directed (backdoor) path not blocked by cond_set.

    Direct translation of dSepAdji() — Peters & Bühlmann (2015).
    Convention: adj[i,j] != 0  means i → j.
    """
    p        = adj_orig.shape[0]
    cond_set = set(cond_set)

    # Ancestors of cond_set nodes in true DAG
    if cond_set:
        anc_mask = np.zeros(p, dtype=bool)
        for c in cond_set:
            anc_mask |= PM[:, c].astype(bool)
        anc_cond = set(np.where(anc_mask)[0])
    else:
        anc_cond = set()

    A        = adj_orig.copy().astype(float)
    # Indices [0..p-1]: reached via incoming edge (→ node)
    # Indices [p..2p-1]: reached via outgoing edge (← node)
    reachable = np.zeros(2 * p, dtype=bool)
    reach_nc  = np.zeros(2 * p, dtype=bool)
    checked   = np.zeros(p,     dtype=bool)
    RM        = np.zeros((2 * p, 2 * p), dtype=bool)
    nc_later  = []   # (reachable_through, new_reachable) for 2nd activation

    to_check = []

    # Seed: children of i (forward edges i→ch, reached via incoming edge)
    ch_i = np.where(A[i, :] != 0)[0]
    reachable[ch_i] = True
    A[i, ch_i] = 0
    to_check.extend(ch_i.tolist())

    # Seed: parents of i (backdoor, non-causal, reached via outgoing edge)
    pa_i = np.where(A[:, i] != 0)[0]
    reachable[pa_i + p] = True
    reach_nc[pa_i + p]  = True
    A[pa_i, i] = 0
    to_check.extend(pa_i.tolist())

    k = 0
    while k < len(to_check):
        node = to_check[k]; k += 1
        if checked[node]:
            continue
        checked[node] = True

        Pa = np.where(A[:, node] != 0)[0].tolist()
        Ch = np.where(A[node, :] != 0)[0].tolist()

        # Parents of node
        Pa1 = [q for q in Pa if q not in cond_set]
        for q in Pa1:
            RM[q,     node] = True
            RM[q + p, node] = True

        if node in anc_cond:
            for q in Pa:
                RM[node, q + p] = True
            if PM2[i, node]:
                for q in Pa:
                    nc_later.append((node, q))
            for q in Pa:
                if not checked[q]:
                    to_check.append(q)

        if node not in cond_set:
            for q in Pa:
                RM[node + p, q + p] = True
            for q in Pa:
                if not checked[q]:
                    to_check.append(q)

        # Children of node
        Ch1 = [c for c in Ch if c not in cond_set]
        for c in Ch1:
            RM[c + p, node + p] = True

        Ch2 = [c for c in Ch if c in anc_cond]
        for c in Ch2:
            RM[c, node + p] = True
        for c in Ch2:
            if PM2[i, c]:
                nc_later.append((c, node))

        if node not in cond_set:
            for c in Ch:
                RM[node,     c] = True
                RM[node + p, c] = True
            for c in Ch:
                if not checked[c]:
                    to_check.append(c)

    # Transitive closure of reachability matrix
    RM = _compute_path_matrix(RM.astype(float))

    # Propagate reachable sets
    seeds = np.where(reachable)[0]
    if len(seeds):
        reachable |= np.any(RM[seeds, :], axis=0)

    seeds = np.where(reach_nc)[0]
    if len(seeds):
        reach_nc |= np.any(RM[seeds, :], axis=0)

    # Second activation step for specific non-causal path type
    for (rt, nr) in nc_later:
        reach_nc[nr + p] = True
        RM[nr,     rt]     = False
        RM[nr,     rt + p] = False
        RM[nr + p, rt]     = False
        RM[nr + p, rt + p] = False

    if nc_later:
        seeds = np.where(reach_nc)[0]
        if len(seeds):
            reach_nc |= np.any(RM[seeds, :], axis=0)

    return reach_nc[:p] | reach_nc[p:]


def _is_pair_incorrect(i, j, pa_gp, pa_g, true_dag, PM, reach_nc):
    """
    Return True if the estimated parent set pa_gp for node i fails the backdoor
    criterion for the causal effect of i on j in true_dag.
    """
    ij_g_null  = not bool(PM[i, j])
    ij_gp_null = j in pa_gp          # j is a parent of i in est → predicts no effect

    if ij_gp_null and ij_g_null:
        return False   # both agree: no causal effect
    if ij_gp_null and not ij_g_null:
        return True    # est misses a true causal path
    if pa_gp == pa_g:
        return False   # identical parent sets → valid adjustment

    p = true_dag.shape[0]
    if PM[i, j]:
        # Any descendant (or self) of a causal child of i in pa_gp?
        causal_ch = [k for k in range(p) if true_dag[i, k] and PM[k, j]]
        if causal_ch and pa_gp and np.any(PM[np.ix_(causal_ch, sorted(pa_gp))]):
            return True

    return bool(reach_nc[j])


def _all_dags_intern(gm, a, row_names, seen, result):
    """
    Recursively enumerate all DAG expansions of an undirected component.
    Translation of allDagsIntern() — Peters & Bühlmann (2015).

    gm        : full p×p adjacency matrix (copied at each recursive level)
    a         : symmetric submatrix for the undirected component
    row_names : numpy array of global indices corresponding to rows/cols of a
    seen      : set of flattened tuples for deduplication across calls
    result    : list accumulating valid p×p DAG matrices
    """
    if a.sum() == 0:
        key = tuple(gm.flatten().tolist())
        if key not in seen:
            seen.add(key)
            result.append(gm.copy())
        return

    sinks = np.where(a.sum(axis=0) > 0)[0]   # nodes with at least one neighbor

    for x in sinks:
        Adjx     = (a[x, :] != 0)
        un_local = np.where(Adjx)[0]

        # Clique check: all neighbors of x must be mutually adjacent (no new v-structures)
        if len(un_local) > 0:
            sub = a[np.ix_(un_local, un_local)].copy()
            np.fill_diagonal(sub, 1)
            if not sub.all():
                continue

        gm2 = gm.copy()
        if len(un_local) > 0:
            un_global = row_names[un_local]
            x_global  = row_names[x]
            gm2[un_global, x_global] = 1   # neighbors → x (x becomes a sink)
            gm2[x_global, un_global] = 0

        mask   = np.ones(len(a), dtype=bool); mask[x] = False
        _all_dags_intern(gm2, a[np.ix_(mask, mask)], row_names[mask], seen, result)


def _all_dags_jonas(est_graph, comp):
    """
    Expand the undirected component `comp` of est_graph into all DAGs in its MEC.
    Returns a list of p×p matrices, or None if mixed directed/undirected edges found.
    Translation of allDagsJonas() — Peters & Bühlmann (2015).
    """
    comp = np.array(sorted(comp))
    a    = est_graph[np.ix_(comp, comp)].copy().astype(float)

    if np.any((a + a.T) == 1):   # mixed directed + undirected edges in component
        return None

    result = []; seen = set()
    _all_dags_intern(est_graph.copy(), a, comp, seen, result)
    return result




def compute_SID_DAG(est_dag, true_dag):
    """
    Exact Structural Intervention Distance between two DAGs.
    Inputs: adjacency matrices (numpy arrays). Convention: adj[i,j] != 0 means i → j.
    Output: integer SID.
    Peters & Bühlmann (2015).
    """
    p   = true_dag.shape[0]
    PM  = _compute_path_matrix(true_dag)
    sid = 0

    for i in range(p):
        pa_gp    = set(np.where(est_dag[:, i]  != 0)[0])
        pa_g     = set(np.where(true_dag[:, i] != 0)[0])
        PM2      = _compute_path_matrix2(true_dag, pa_gp, PM)
        reach_nc = _reachable_non_causal(true_dag, i, pa_gp, PM, PM2)

        for j in range(p):
            if i != j and _is_pair_incorrect(i, j, pa_gp, pa_g, true_dag, PM, reach_nc):
                sid += 1

    return sid


def compute_SID_CPDAG(est_cpdag, true_dag):
    """
    SID bounds when est_cpdag is a CPDAG (or DAG) and true_dag is a DAG.
    Returns (lower_bound, upper_bound); both equal when est_cpdag is a DAG.

    Convention: adj[i,j] != 0 means i → j; undirected edge: adj[i,j] = adj[j,i] = 1.
    Peters & Bühlmann (2015).
    """
    p             = true_dag.shape[0]
    PM            = _compute_path_matrix(true_dag)
    incorrect_int = np.zeros((p, p), dtype=bool)
    min_total     = 0
    max_total     = 0

    # Undirected part of est_cpdag
    undir   = (est_cpdag != 0) & (est_cpdag.T != 0)
    G_undir = nx.Graph()
    G_undir.add_nodes_from(range(p))
    for r, c in zip(*np.where(np.triu(undir))):
        G_undir.add_edge(int(r), int(c))
    conn_comps = list(nx.connected_components(G_undir))

    # Check validity: chordal undirected components, each with ≤ 8 nodes
    is_valid = all(
        len(comp) <= 1 or (len(comp) <= 8 and nx.is_chordal(G_undir.subgraph(comp)))
        for comp in conn_comps
    )

    for comp in conn_comps:
        comp = sorted(comp)

        # Expand the undirected component into DAGs
        if is_valid and len(comp) > 1:
            expansions = _all_dags_jonas(est_cpdag, comp)
            if expansions is None:
                is_valid = False
        if not is_valid or len(comp) == 1:
            expansions = [est_cpdag.copy()]

        if is_valid:
            # ── Valid CPDAG: accumulate incorrectness across the whole component ──
            comp_incorrect = np.zeros(len(expansions))

            for i in comp:
                pa_g = set(np.where(true_dag[:, i] != 0)[0])

                # Group expansion indices by the parent set they assign to i
                groups: dict = {}
                for idx, exp in enumerate(expansions):
                    key = frozenset(int(q) for q in np.where(exp[:, i] != 0)[0])
                    groups.setdefault(key, []).append(idx)

                for pa_gp_frozen, exp_idxs in groups.items():
                    pa_gp    = set(pa_gp_frozen)
                    PM2      = _compute_path_matrix2(true_dag, pa_gp, PM)
                    reach_nc = _reachable_non_causal(true_dag, i, pa_gp, PM, PM2)

                    for j in range(p):
                        if i == j:
                            continue
                        if _is_pair_incorrect(i, j, pa_gp, pa_g, true_dag, PM, reach_nc):
                            incorrect_int[i, j] = True
                            for idx in exp_idxs:
                                comp_incorrect[idx] += 1

            min_total += comp_incorrect.min()
            max_total += comp_incorrect.max()

        else:
            # ── Invalid/large component: local expansion per node independently ──
            for i in comp:
                pa_g    = set(np.where(true_dag[:, i] != 0)[0])
                certain = set(int(q) for q in np.where(
                    (est_cpdag[:, i] != 0) & (est_cpdag[i, :] == 0)
                )[0])
                possible = [int(q) for q in np.where(
                    (est_cpdag[:, i] != 0) & (est_cpdag[i, :] != 0)
                )[0]]

                n_local   = 2 ** len(possible)
                local_inc = np.zeros(n_local)

                for idx in range(n_local):
                    extra    = {possible[k] for k in range(len(possible)) if (idx >> k) & 1}
                    pa_gp    = certain | extra
                    PM2      = _compute_path_matrix2(true_dag, pa_gp, PM)
                    reach_nc = _reachable_non_causal(true_dag, i, pa_gp, PM, PM2)

                    for j in range(p):
                        if i == j:
                            continue
                        if _is_pair_incorrect(i, j, pa_gp, pa_g, true_dag, PM, reach_nc):
                            incorrect_int[i, j] = True
                            local_inc[idx] += 1

                min_total += local_inc.min()
                max_total += local_inc.max()

    return int(min_total), int(max_total)


def compute_sid(est, true_dag):
    """
    Auto-detects graph types and computes SID.
    Returns (lower_bound, upper_bound); equal when both inputs are DAGs.

    Raises ValueError if:
      - true_dag has undirected edges (SID requires the reference to be a DAG)
      - est has undirected edges but true_dag also does (not well-defined)
    """
    if _is_cpdag(true_dag):
        raise ValueError(
            "true_dag must be a DAG (no undirected edges). "
            "SID is only defined when the reference graph is a DAG."
        )

    if _is_cpdag(est):
        lower, upper = compute_SID_CPDAG(est, true_dag)
    else:
        sid   = compute_SID_DAG(est, true_dag)
        lower = upper = sid

    return lower, upper
