"""Joint Causal Inference (Mooij, Magliacane & Claassen, 2020) on top of PC.

Regime: environment labels are known, intervention targets are not. Each
non-reference environment becomes a binary context indicator C_k that is
appended to the system variables X_1..X_d, and a constraint-based search is
run on the joint data with the JCI background assumptions:

  JCI Assumption 1 (exogeneity): no system variable causes a context
      variable, so every context-system edge is oriented C_k -> X_j.
  JCI Assumption 3 (generic context model): context variables are treated
      as mutually adjacent (confounded) and are never tested against each
      other.

One-hot indicators share a common cause (the environment selector), so a
test of C_k against X_j can be distorted by paths C_k <-> C_m -> ... that are
opened when conditioning on a system descendant. With
condition_on_contexts=True (default), every CI test conditions on all context
indicators other than those being tested, i.e. the environment enters as a
fixed effect. Contexts are exogenous roots, so conditioning on them only
blocks paths; true separations are preserved, and the faithfulness problems
from deterministic one-hot relations noted in the rebuttal are avoided. The
plain variant (condition_on_contexts=False) is kept for comparison.

We use PC rather than FCI because the paper assumes causal sufficiency among
the system variables; this makes the output a CPDAG over the system variables
and keeps the comparison like-for-like with ICA-LiNGAM. JCI Assumption 2
(no confounding between context and system variables) holds by construction
in the synthetic design.

Usage:
    G = jci_pc(X, env, alpha=0.01)   # G[i, j] = 1 means i -> j; both = undirected
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from causal_utils import FisherZ


def context_indicators(env: np.ndarray, reference: int | None = None) -> tuple[np.ndarray, list]:
    """One indicator column per environment, dropping the reference
    environment (default: the most frequent label, typically observational)."""
    labels, counts = np.unique(env, return_counts=True)
    if len(labels) < 2:
        raise ValueError("JCI needs at least two environments.")
    if reference is None:
        reference = 0 if 0 in labels else labels[int(np.argmax(counts))]
    keep = [lab for lab in labels if lab != reference]
    C = np.stack([(env == lab).astype(float) for lab in keep], axis=1)
    return C, keep


def jci_pc(X: np.ndarray, env: np.ndarray, alpha: float = 0.01,
           max_depth: int | None = None, reference: int | None = None,
           condition_on_contexts: bool = True, return_full: bool = False):
    d = X.shape[1]
    C, _ = context_indicators(env, reference)
    K = C.shape[1]
    D = np.hstack([X, C])
    p = d + K
    is_ctx = np.array([False] * d + [True] * K)
    ctx_idx = list(range(d, p))
    test = FisherZ(D)

    def base_set(i, j):
        if not condition_on_contexts:
            return ()
        return tuple(c for c in ctx_idx if c not in (i, j))

    # ---------------- skeleton (PC-stable) ----------------
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)
    sepset: dict[tuple[int, int], tuple[int, ...]] = {}
    depth = 0
    while True:
        snapshot = adj.copy()
        any_testable = False
        for i in range(p):
            for j in range(i + 1, p):
                if not adj[i, j] or (is_ctx[i] and is_ctx[j]):
                    continue
                base = base_set(i, j)
                for a, b in ((i, j), (j, i)):
                    cand = [k for k in np.where(snapshot[a])[0]
                            if k != b and k not in base]
                    if len(cand) < depth:
                        continue
                    any_testable = True
                    removed = False
                    for S in combinations(cand, depth):
                        if test.pvalue(i, j, S + base) > alpha:
                            adj[i, j] = adj[j, i] = False
                            sepset[(i, j)] = sepset[(j, i)] = S + base
                            removed = True
                            break
                    if removed:
                        break
        depth += 1
        if not any_testable or (max_depth is not None and depth > max_depth):
            break

    # ---------------- orientation ----------------
    # O[i, j] = True means an arrowhead at j on edge i - j. An edge is
    # directed i -> j iff O[i, j] and not O[j, i]; undirected iff neither.
    O = np.zeros((p, p), dtype=bool)
    for c in np.where(is_ctx)[0]:
        for x in np.where(adj[c] & ~is_ctx)[0]:
            O[c, x] = True                       # JCI Assumption 1

    def undirected(a, b):
        return adj[a, b] and not O[a, b] and not O[b, a]

    def directed(a, b):
        return adj[a, b] and O[a, b] and not O[b, a]

    # v-structures a -> m <- b with a, b non-adjacent; m must be a system variable
    for m in np.where(~is_ctx)[0]:
        nbrs = np.where(adj[m])[0]
        for a, b in combinations(nbrs, 2):
            if adj[a, b] or (is_ctx[a] and is_ctx[b]):
                continue
            if m in sepset.get((a, b), ()):
                continue
            # orient only if it does not reverse an existing arrow
            if not O[m, a] and not O[m, b]:
                O[a, m] = True
                O[b, m] = True

    # Meek rules R1-R4 on the system part (context arrows act as known arrows)
    changed = True
    while changed:
        changed = False
        for a in range(p):
            for b in range(p):
                if is_ctx[b] or not undirected(a, b) or is_ctx[a]:
                    continue
                # R1: c -> a - b, c not adjacent b  =>  a -> b
                if any(directed(c, a) and not adj[c, b] and c != b for c in range(p)):
                    O[a, b] = True; changed = True; continue
                # R2: a -> c -> b and a - b  =>  a -> b
                if any(directed(a, c) and directed(c, b) for c in range(p)):
                    O[a, b] = True; changed = True; continue
                # R3: a - c1 -> b, a - c2 -> b, c1, c2 non-adjacent  =>  a -> b
                cs = [c for c in range(p) if undirected(a, c) and directed(c, b)]
                if any(not adj[c1, c2] for c1, c2 in combinations(cs, 2)):
                    O[a, b] = True; changed = True; continue
                # R4: a - c, c -> e -> b, a adjacent e, c not adjacent b  =>  a -> b
                hit = False
                for c in range(p):
                    if not (adj[a, c] and not is_ctx[c]) or adj[c, b]:
                        continue
                    for e in range(p):
                        if adj[a, e] and directed(c, e) and directed(e, b):
                            hit = True; break
                    if hit:
                        break
                if hit:
                    O[a, b] = True; changed = True

    G = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(d):
            if i != j and adj[i, j]:
                if directed(i, j):
                    G[i, j] = 1
                elif undirected(i, j) or (O[i, j] and O[j, i]):
                    # undirected, or a conflicting (bidirected) orientation:
                    # report as undirected (an orientation error vs. the DAG)
                    G[i, j] = 1
    if return_full:
        return G, {"adj": adj, "O": O, "is_ctx": is_ctx, "sepset": sepset}
    return G
