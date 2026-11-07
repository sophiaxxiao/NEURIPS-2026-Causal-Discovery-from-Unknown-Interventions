"""Sachs experiments for the camera-ready: Psi-FCI (rebuttal to Reviewer 8saU)
and JCI (Reviewer AMBX), both using environment labels but not targets.

Psi-FCI (Jaber et al., NeurIPS 2020) is run through the py-why `dodiscover`
package. The rebuttal says the student used "the DoWhy package"; Psi-FCI
lives in dodiscover (same py-why ecosystem), not in dowhy itself.

FIRST run the smoke test, which checks that the installed dodiscover version
matches the API used here:
    python run_sachs.py --smoke-test

Then, e.g.:
    python run_sachs.py --data-dir path/to/sachs_conditions --reference cd3cd28 \
        --methods jci psifci --sizes 500 1000 2000 4000 --n-boot 30

Data input (either):
    --data-dir DIR    one file per condition (.csv/.xls/.xlsx); the file stem is
                      the environment label
    --data-csv FILE   one file with an environment column (--env-col)
Column names are matched case-insensitively to the 11 Sachs proteins
(e.g. 'praf', 'p44/42', 'pakts473' are recognized).

Ground truth: by default the 17-edge consensus network (as in bnlearn's
`sachs`). CHECK that this is the same edge list used for Fig. 4; otherwise
pass --truth-csv with columns parent,child.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_utils import all_metrics, ica_lingam_graph
from jci_pc import jci_pc

PROTEINS = ["raf", "mek", "plcg", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk"]

ALIASES = {
    "raf": "raf", "praf": "raf",
    "mek": "mek", "pmek": "mek", "mek12": "mek",
    "plcg": "plcg", "plc": "plcg", "plcgamma": "plcg", "pplcg": "plcg",
    "pip2": "pip2", "pip3": "pip3",
    "erk": "erk", "perk": "erk", "p4442": "erk", "erk12": "erk",
    "akt": "akt", "pakt": "akt", "pakts473": "akt", "akt473": "akt",
    "pka": "pka", "pkc": "pkc",
    "p38": "p38", "pp38": "p38", "p38mapk": "p38",
    "jnk": "jnk", "pjnk": "jnk",
}

# 17-edge consensus network (Sachs et al., 2005), as in bnlearn's `sachs`.
CONSENSUS_EDGES = [
    ("pkc", "raf"), ("pka", "raf"), ("pkc", "mek"), ("pka", "mek"), ("raf", "mek"),
    ("mek", "erk"), ("pka", "erk"), ("erk", "akt"), ("pka", "akt"),
    ("plcg", "pip3"), ("plcg", "pip2"), ("pip3", "pip2"),
    ("pkc", "pka"), ("pkc", "jnk"), ("pka", "jnk"), ("pkc", "p38"), ("pka", "p38"),
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _canon(col: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", str(col).lower())
    return ALIASES.get(key)


def _standardize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    mapping = {c: _canon(c) for c in df.columns}
    found = {v for v in mapping.values() if v}
    missing = [p for p in PROTEINS if p not in found]
    if missing:
        raise ValueError(f"{source}: could not find columns for {missing}. "
                         f"Columns present: {list(df.columns)}")
    df = df.rename(columns={c: v for c, v in mapping.items() if v})
    return df[PROTEINS].astype(float)


def load_sachs(args) -> tuple[pd.DataFrame, np.ndarray]:
    frames, labels = [], []
    if args.data_dir:
        files = sorted(p for p in Path(args.data_dir).iterdir()
                       if p.suffix.lower() in (".csv", ".xls", ".xlsx"))
        if not files:
            raise SystemExit(f"No .csv/.xls/.xlsx files in {args.data_dir}")
        for f in files:
            df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
            frames.append(_standardize_columns(df, f.name))
            labels += [f.stem] * len(df)
    elif args.data_csv:
        df = pd.read_csv(args.data_csv)
        if args.env_col not in df.columns:
            raise SystemExit(f"--env-col '{args.env_col}' not in {list(df.columns)}")
        labels = df[args.env_col].astype(str).tolist()
        frames.append(_standardize_columns(df.drop(columns=[args.env_col]), args.data_csv))
    else:
        raise SystemExit("Give --data-dir or --data-csv (or --smoke-test).")
    X = pd.concat(frames, ignore_index=True)
    env = np.array(labels)
    if args.conditions:
        keep = np.isin(env, args.conditions)
        unknown = set(args.conditions) - set(env)
        if unknown:
            raise SystemExit(f"Unknown --conditions {sorted(unknown)}; available: {sorted(set(env))}")
        X, env = X[keep].reset_index(drop=True), env[keep]
    if args.log:
        if (X.values <= 0).any():
            raise SystemExit("--log requested but data has non-positive values.")
        X = np.log(X)
    return X, env


def true_graph(args) -> np.ndarray:
    edges = CONSENSUS_EDGES
    if args.truth_csv:
        t = pd.read_csv(args.truth_csv)
        edges = [(_canon(a), _canon(b)) for a, b in zip(t["parent"], t["child"])]
    idx = {p: i for i, p in enumerate(PROTEINS)}
    G = np.zeros((len(PROTEINS),) * 2, dtype=int)
    for a, b in edges:
        G[idx[a], idx[b]] = 1
    return G


def stratified_resample(env: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a with-replacement resample of total size n, keeping each
    environment's share of the pool."""
    labs, counts = np.unique(env, return_counts=True)
    shares = counts / counts.sum()
    sizes = np.maximum(1, np.round(shares * n).astype(int))
    out = []
    for lab, k in zip(labs, sizes):
        pool = np.where(env == lab)[0]
        out.append(rng.choice(pool, size=k, replace=True))
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Psi-FCI via dodiscover
# --------------------------------------------------------------------------

def _import_dodiscover():
    try:
        import dodiscover  # noqa: F401
        from dodiscover import PsiFCI, make_context
        from dodiscover.ci import FisherZCITest
        from dodiscover.cd import BregmanCDTest, KernelCDTest
    except ImportError as e:
        raise SystemExit(f"Could not import dodiscover ({e}). Install with:\n"
                         "    pip install dodiscover pywhy-graphs") from e
    try:
        from dodiscover import InterventionalContextBuilder
    except ImportError:
        try:
            from dodiscover.context_builder import InterventionalContextBuilder
        except ImportError as e:
            raise SystemExit("Found dodiscover but not InterventionalContextBuilder; "
                             "check the installed version's docs for building an "
                             "interventional context.") from e
    return PsiFCI, make_context, InterventionalContextBuilder, FisherZCITest, BregmanCDTest, KernelCDTest


def _edge_list(G, kind: str) -> list:
    """Edges of one type from a pywhy-graphs mixed-edge graph, across API versions."""
    for getter in (lambda: getattr(G, f"{kind}_edges"),
                   lambda: G.get_graphs(kind).edges,
                   lambda: getattr(G, f"sub_{kind}_graph")().edges):
        try:
            return list(getter())
        except Exception:  # noqa: BLE001 - try the next API
            continue
    return []


def pag_to_summary(G, names: list[str]) -> tuple[np.ndarray, dict]:
    """Definite edges a -> b (directed, with no other mark on that pair) become
    G[a, b] = 1. Every other adjacency (o->, o-o, <->, ---) is reported as
    undirected, i.e. skeleton credit but an orientation error vs. the DAG.
    F-nodes (augmented intervention nodes) are dropped."""
    idx = {n: i for i, n in enumerate(names)}
    kinds = {k: [(u, v) for u, v in _edge_list(G, k) if u in idx and v in idx]
             for k in ("directed", "bidirected", "circle", "undirected")}
    pairs_other = {frozenset(e) for k in ("bidirected", "circle", "undirected") for e in kinds[k]}
    S = np.zeros((len(names),) * 2, dtype=int)
    for u, v in kinds["directed"]:
        if frozenset((u, v)) in pairs_other or (v, u) in kinds["directed"]:
            S[idx[u], idx[v]] = S[idx[v], idx[u]] = 1
        else:
            S[idx[u], idx[v]] = 1
    for e in pairs_other:
        u, v = tuple(e)
        S[idx[u], idx[v]] = S[idx[v], idx[u]] = 1
    counts = {f"n_{k}": len(v) for k, v in kinds.items()}
    return S, counts


def _linear_cd_test():
    """Linear conditional-discrepancy test: H0 is P_i(Y | X) = P_j(Y | X) in mean,
    tested as zero partial correlation between the group indicator and Y given X
    (Fisher z). Matched to soft mean-shift interventions in a linear SCM and to the
    Fisher-z CI test. Added because dodiscover's KernelCDTest crashes on marginal
    tests (empty X) and its 1000-permutation null is too slow for 36 Sachs F-nodes."""
    import math

    from scipy import stats
    from dodiscover.cd.base import BaseConditionalDiscrepancyTest

    class LinearCDTest(BaseConditionalDiscrepancyTest):
        def __init__(self):
            self.n_jobs = None
            self.propensity_est = None
            self.propensity_model = None

        def _statistic(self, *a, **k):  # abstract in the base class; unused here
            raise NotImplementedError

        def test(self, df, group_col, y_vars, x_vars):
            self._check_test_input(df, y_vars, group_col, x_vars)
            g = df[list(group_col)[0]].to_numpy(float)
            y = df[list(y_vars)].to_numpy(float)[:, 0]
            Z = np.column_stack([np.ones(len(df))] + ([df[list(x_vars)].to_numpy(float)]
                                                       if x_vars else []))
            rg = g - Z @ np.linalg.lstsq(Z, g, rcond=None)[0]
            ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            denom = math.sqrt(float(rg @ rg) * float(ry @ ry))
            r = float(rg @ ry) / denom if denom > 0 else 0.0
            r = max(min(r, 1 - 1e-12), -1 + 1e-12)
            dof = len(df) - (Z.shape[1] - 1) - 3
            z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(dof, 1))
            return abs(z), float(2 * stats.norm.sf(abs(z)))

    return LinearCDTest()


def _fisherz_ci_test():
    """Fisher-z CI test for Psi-FCI. Replaces dodiscover's FisherZCITest, whose
    `fisherz` builds the variable list from a set and indexes columns in data order,
    so for some (x, y, Z) it returns the partial correlation of the wrong pair
    (e.g. a-b | c instead of a-c | b). Verified on the smoke-test SCM."""
    import math

    from scipy import stats
    from dodiscover.ci.base import BaseConditionalIndependenceTest

    class FisherZ(BaseConditionalIndependenceTest):
        def test(self, df, x_vars, y_vars, z_covariates=None):
            self._check_test_input(df, x_vars, y_vars, z_covariates)
            cols = [next(iter(x_vars)), next(iter(y_vars)), *sorted(z_covariates or (), key=str)]
            C = np.corrcoef(df[cols].to_numpy(float), rowvar=False)
            P = np.linalg.pinv(C)
            r = -P[0, 1] / math.sqrt(abs(P[0, 0] * P[1, 1]))
            r = max(min(r, 1 - 1e-12), -1 + 1e-12)
            dof = len(df) - (len(cols) - 2) - 3
            z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(dof, 1))
            return abs(z), float(2 * stats.norm.sf(abs(z)))

    return FisherZ()


def run_psifci(X: pd.DataFrame, env: np.ndarray, reference: str | None, alpha: float,
               cd_test: str, max_cond_set_size: int | None, n_jobs: int | None):
    PsiFCI, make_context, ICB, FisherZCITest, BregmanCDTest, KernelCDTest = _import_dodiscover()
    labs = list(dict.fromkeys(env))          # preserve order
    if reference is not None:
        if reference not in labs:
            raise SystemExit(f"--reference '{reference}' not among {labs}")
        labs = [reference] + [l for l in labs if l != reference]
    dfs = [X[env == lab].reset_index(drop=True) for lab in labs]
    context = (make_context(create_using=ICB)
               .variables(data=dfs[0])
               .num_distributions(len(dfs))
               .obs_distribution(reference is not None)
               .build())
    if cd_test == "kernel":
        cd = KernelCDTest()
    elif cd_test == "bregman":
        cd = BregmanCDTest()
    else:
        cd = _linear_cd_test()
    learner = PsiFCI(ci_estimator=_fisherz_ci_test(), cd_estimator=cd, alpha=alpha,
                     max_cond_set_size=max_cond_set_size, n_jobs=n_jobs)
    learner.learn_graph(dfs, context)
    return pag_to_summary(learner.graph_, list(X.columns))


# Known targets of the Squires (UT-IGSP) subset, as in the student's Fig. 4 notebook
# (0-based indices into raf, mek, plcg, pip2, pip3, erk, akt, pka, pkc, p38, jnk).
SQUIRES_TARGETS = {"cd3cd28+u0126": [1], "cd3cd28+psitect": [3], "cd3cd28+ly": [4],
                   "cd3cd28+aktinhib": [6], "cd3cd28+g0076": [8]}


def _split_known_targets(X: np.ndarray, env: np.ndarray, reference: str):
    """Observational block + non-empty interventional blocks with their known targets."""
    obs = X[env == reference]
    blocks, targets = [], []
    for cond, t in SQUIRES_TARGETS.items():
        Xi = X[env == cond]
        if len(Xi) > 0:
            blocks.append(Xi); targets.append(t)
    return obs, blocks, targets


def run_gies(X, env, reference):
    """Notebook call: gies.fit_bic([obs] + interventional blocks, [[]] + targets)."""
    import gies
    obs, blocks, targets = _split_known_targets(X, env, reference)
    est, _ = gies.fit_bic([obs] + blocks, [[]] + targets)
    return est


def run_igsp(X, env, reference, alpha):
    """Notebook call: causaldag IGSP with Gaussian CI and invariance tests."""
    from causaldag import igsp
    from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test
    from causaldag.utils.invariance_tests import (MemoizedInvarianceTester, gauss_invariance_suffstat,
                                                  gauss_invariance_test)
    obs, blocks, targets = _split_known_targets(X, env, reference)
    ci = MemoizedCI_Tester(gauss_ci_test, gauss_ci_suffstat(obs), alpha=alpha)
    inv = MemoizedInvarianceTester(gauss_invariance_test, gauss_invariance_suffstat(obs, blocks), alpha=alpha)
    dag = igsp([dict(interventions=set(t)) for t in targets], set(range(X.shape[1])), ci, inv)
    return dag.to_amat()[0]


def smoke_test(args):
    """Tiny 4-node linear SCM with an observational and two soft-shift
    environments; checks that the Psi-FCI wrapper runs and returns a graph."""
    rng = np.random.default_rng(0)
    names = ["a", "b", "c", "d"]
    W = np.zeros((4, 4)); W[0, 1] = 1.2; W[1, 2] = -1.0; W[3, 2] = 0.8
    frames, labels = [], []
    for k, target in enumerate([None, 0, 3]):
        E = rng.normal(size=(800, 4))
        if target is not None:
            E[:, target] += 1.5
        frames.append(pd.DataFrame(E @ np.linalg.inv(np.eye(4) - W), columns=names))
        labels += [f"env{k}"] * 800
    X = pd.concat(frames, ignore_index=True)
    env = np.array(labels)
    t0 = time.time()
    S, counts = run_psifci(X, env, "env0", 0.05, args.cd_test, None, 1)
    print(f"Psi-FCI ran in {time.time() - t0:.1f}s. Edge-type counts (system nodes): {counts}")
    print("Summary adjacency (rows -> cols, symmetric = not definitely oriented):")
    print(pd.DataFrame(S, index=names, columns=names))
    print("True edges: a -> b, b -> c, d -> c.  The skeleton should be a-b, b-c, c-d.")
    G_true = (W != 0).astype(int)
    m = all_metrics(S, G_true)
    print(f"skeleton F1 = {m['skeleton_f1']:.2f}, SHD = {m['shd']}")
    if m["skeleton_f1"] < 0.99:
        print("WARNING: skeleton not recovered on an easy example; check the dodiscover "
              "version / API before trusting Sachs numbers.")


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-dir"); ap.add_argument("--data-csv"); ap.add_argument("--env-col", default="env")
    ap.add_argument("--conditions", nargs="+", help="subset of environment labels to use")
    ap.add_argument("--reference", help="observational/reference condition label (e.g. cd3cd28)")
    ap.add_argument("--log", action="store_true", help="log-transform (match your Fig. 4 preprocessing)")
    ap.add_argument("--truth-csv")
    ap.add_argument("--methods", nargs="+", default=["jci", "psifci"],
                    choices=["jci", "psifci", "ica_lingam", "ica_lingam_castle", "gies", "igsp"])
    ap.add_argument("--sizes", nargs="+", default=None,
                    help="total sample sizes, or 'all'; default = the full pool only")
    ap.add_argument("--sampling", choices=["fig4", "stratified"], default="fig4",
                    help="fig4: uniform draw without replacement from the pool, as in the "
                         "student's Fig. 4 notebook; stratified: with replacement per condition")
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--cd-test", choices=["linear", "kernel", "bregman"], default="linear")
    ap.add_argument("--max-cond-set-size", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None, help="passed to Psi-FCI")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sachs_raw.csv")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test(args)
        return

    X, env = load_sachs(args)
    G_true = true_graph(args)
    labs, counts = np.unique(env, return_counts=True)
    print(f"Loaded {len(X)} samples, {len(labs)} conditions:")
    for l, c in zip(labs, counts):
        print(f"    {l:30s} {c}")
    if args.reference is None and "jci" in args.methods:
        print("NOTE: no --reference given; JCI uses the largest condition as reference.")

    sizes = [len(X) if str(s) == "all" else min(int(s), len(X)) for s in (args.sizes or ["all"])]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    def out_for(method):                # one file per method (Psi-FCI rows have extra columns)
        return args.out.replace(".csv", f"_{method}.csv")
    done = set()
    for method in args.methods:         # resume
        if os.path.exists(out_for(method)):
            prev = pd.read_csv(out_for(method))
            done |= set(zip(prev["method"], prev["n"], prev["boot"]))
    if done:
        print(f"Resuming: {len(done)} runs already done")

    for n in sizes:
        for b in range(args.n_boot):
            rng = np.random.default_rng([args.seed, n, b])
            if n >= len(X):
                idx = rng.permutation(len(X))
            elif args.sampling == "fig4":
                idx = rng.choice(len(X), size=n, replace=False)
            else:
                idx = stratified_resample(env, n, rng)
            Xb, eb = X.iloc[idx].reset_index(drop=True), env[idx]
            for method in args.methods:
                if (method, n, b) in done:
                    continue
                if n >= len(X) and b > 0 and method in ("jci", "psifci", "gies"):
                    continue                    # full pool: deterministic, one run suffices
                t0 = time.time()
                extra = {}
                if method == "jci":
                    ref = args.reference
                    ref_int = None
                    codes, uniq = pd.factorize(eb)
                    if ref is not None:
                        ref_int = int(np.where(uniq == ref)[0][0])
                    G = jci_pc(Xb.values, codes, alpha=args.alpha, reference=ref_int)
                elif method == "psifci":
                    G, extra = run_psifci(Xb, eb, args.reference, args.alpha, args.cd_test,
                                          args.max_cond_set_size, args.n_jobs)
                elif method == "gies":
                    G = run_gies(Xb.values, eb, args.reference)
                elif method == "igsp":
                    G = run_igsp(Xb.values, eb, args.reference, args.alpha)
                elif method == "ica_lingam_castle":
                    from castle.algorithms import ICALiNGAM
                    mdl = ICALiNGAM(); mdl.learn(Xb.values); G = mdl.causal_matrix
                else:
                    G = ica_lingam_graph(Xb.values, seed=b)
                from castle.metrics import MetricsDAG
                G = (np.asarray(G) != 0).astype(int)
                row = {"method": method, "n": n, "boot": b, "runtime_s": time.time() - t0,
                       "n_envs": len(set(eb)), "shd_castle": MetricsDAG(G, G_true).metrics["shd"],
                       **all_metrics(G, G_true), **extra}
                pd.DataFrame([row]).to_csv(out_for(method), mode="a", index=False,
                                           header=not os.path.exists(out_for(method)))
                print(f"n={n} boot={b} {method}: SHD castle={row['shd_castle']} strict={row['shd']} ({row['runtime_s']:.1f}s)")

    df = pd.concat([pd.read_csv(out_for(m)) for m in args.methods if os.path.exists(out_for(m))])
    summary = (df.groupby(["method", "n"])
                 .agg(shd_castle_mean=("shd_castle", "mean"), shd_castle_se=("shd_castle", "sem"),
                      shd_castle_sd=("shd_castle", "std"),
                      shd_mean=("shd", "mean"), shd_se=("shd", "sem"),
                      skel_f1=("skeleton_f1", "mean"), dir_f1=("dir_f1", "mean"),
                      runs=("shd", "size"))
                 .reset_index())
    summ_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summ_path, index=False)
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nsummary -> {summ_path}")


if __name__ == "__main__":
    sys.exit(main())
