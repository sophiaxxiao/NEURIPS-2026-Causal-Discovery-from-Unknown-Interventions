"""
Sachs et al. (2005) dataset loader for causal discovery benchmarking.

Preprocessing steps (all optional):
  1. Log-transform (log1p) — flow cytometry data is approximately log-normally distributed
  2. Remove outliers per column (values beyond k standard deviations), applied per condition
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Column name mapping: observed variants -> canonical names
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "Raf":       "Raf",
    "Mek":       "Mek",
    "Plcg":      "PLCg",
    "Pip2":      "PIP2",
    "Pip3":      "PIP3",
    "Erk":       "Erk",
    "Akt":       "Akt",
    "Pka":       "PKA",
    "Pkc":       "PKC",
    "P38":       "P38",
    "Jnk":       "Jnk",
    # xls file style
    "Praf":      "Raf",
    "Pmek":      "Mek",
    "P44/42":    "Erk",
    "Pakts473":  "Akt",
    "Pjnk":      "Jnk",
}

PROTEINS = ["Raf", "Mek", "PLCg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]

# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

# Intervention targets per condition (for reference)
INTERVENTION_TARGETS = {
    "observational":              set(),
    "cd3cd28":                    {"PKC", "PKA"},
    "cd3cd28icam2":               {"PKC", "PKA"},
    "cd3cd28+aktinhib":           {"PKC", "PKA", "Akt"},
    "cd3cd28+g0076":              {"PKC", "PKA"},
    "cd3cd28+psitect":            {"PKC", "PKA"},
    "cd3cd28+u0126":              {"PKC", "PKA", "Mek"},
    "cd3cd28+ly":                 {"PKC", "PKA", "PIP3"},
    "pma":                        {"PKC"},
    "b2camp":                     {"PKA"},
    "cd3cd28icam2+aktinhib":      {"PKC", "PKA", "Akt"},
    "cd3cd28icam2+g0076":         {"PKC", "PKA"},
    "cd3cd28icam2+psit":          {"PKC", "PKA"},
    "cd3cd28icam2+u0126":         {"PKC", "PKA", "Mek"},
    "cd3cd28icam2+ly":            {"PKC", "PKA", "PIP3"},
}

FILENAME_TO_CONDITION = {
    "1__cd3cd28":                "cd3cd28",
    "2__cd3cd28icam2":           "cd3cd28icam2",
    "3__cd3cd28+aktinhib":       "cd3cd28+aktinhib",
    "4__cd3cd28+g0076":          "cd3cd28+g0076",
    "5__cd3cd28+psitect":        "cd3cd28+psitect",
    "6__cd3cd28+u0126":          "cd3cd28+u0126",
    "7__cd3cd28+ly":             "cd3cd28+ly",
    "8__pma":                    "pma",
    "9__b2camp":                 "b2camp",
    "10__cd3cd28icam2+aktinhib": "cd3cd28icam2+aktinhib",
    "11__cd3cd28icam2+g0076":    "cd3cd28icam2+g0076",
    "12__cd3cd28icam2+psit":     "cd3cd28icam2+psit",
    "13__cd3cd28icam2+u0126":    "cd3cd28icam2+u0126",
    "14__cd3cd28icam2+ly":       "cd3cd28icam2+ly",
    "1. cd3cd28":                "cd3cd28",
    "2. cd3cd28icam2":           "cd3cd28icam2",
    "3. cd3cd28+aktinhib":       "cd3cd28+aktinhib",
    "4. cd3cd28+g0076":          "cd3cd28+g0076",
    "5. cd3cd28+psitect":        "cd3cd28+psitect",
    "6. cd3cd28+u0126":          "cd3cd28+u0126",
    "7. cd3cd28+ly":             "cd3cd28+ly",
    "8. pma":                    "pma",
    "9. b2camp":                 "b2camp",
    "10. cd3cd28icam2+aktinhib": "cd3cd28icam2+aktinhib",
    "11. cd3cd28icam2+g0076":    "cd3cd28icam2+g0076",
    "12. cd3cd28icam2+psit":     "cd3cd28icam2+psit",
    "13. cd3cd28icam2+u0126":    "cd3cd28icam2+u0126",
    "14. cd3cd28icam2+ly":       "cd3cd28icam2+ly",
}

# Conditions 10-14 were held out for western blot validation only.
WESTERN_BLOT_CONDITIONS = {
    "cd3cd28icam2+aktinhib",
    "cd3cd28icam2+g0076",
    "cd3cd28icam2+psit",
    "cd3cd28icam2+u0126",
    "cd3cd28icam2+ly",
}

# Squires (2020) subset used in the UT-IGSP paper real-data analysis.
SQUIRES_SUBSET = {
    "cd3cd28":          [],
    "cd3cd28+u0126":    [1],
    "cd3cd28+psitect":  [3],
    "cd3cd28+ly":       [4],
    "cd3cd28+aktinhib": [6],
    "cd3cd28+g0076":    [8],
}

# ---------------------------------------------------------------------------
# Ground truth DAG
# ---------------------------------------------------------------------------

# Edges used in the original Sachs et al. (2005) paper (pre-bnlearn).
SACHS_GROUND_TRUTH_EDGES = [
    ("PKC", "Raf"),
    ("PKC", "Mek"),
    ("PKC", "Jnk"),
    ("PKC", "P38"),
    ("PKC", "PKA"),
    ("PKA", "Raf"),
    ("PKA", "Mek"),
    ("PKA", "Erk"),
    ("PKA", "Akt"),
    ("PKA", "Jnk"),
    ("PKA", "P38"),
    ("Raf", "Mek"),
    ("Mek", "Erk"),
    ("Erk", "Akt"),
    ("PLCg", "PIP2"),
    ("PLCg", "PIP3"),
    ("PIP3", "PIP2"),
    ("PIP3", "Akt"),
]

# bnlearn Sachs ground truth DAG (17 edges), sourced from the sachs.bif file.
# Node indices correspond to PROTEINS = ["Raf","Mek","PLCg","PIP2","PIP3","Erk","Akt","PKA","PKC","P38","Jnk"]
BNLEARN_SACHS_ARCS = {
    (0, 1),   # Raf  → Mek
    (1, 5),   # Mek  → Erk
    (2, 3),   # PLCg → PIP2
    (2, 4),   # PLCg → PIP3
    (4, 3),   # PIP3 → PIP2
    (5, 6),   # Erk  → Akt
    (7, 0),   # PKA  → Raf
    (7, 1),   # PKA  → Mek
    (7, 5),   # PKA  → Erk
    (7, 6),   # PKA  → Akt
    (7, 9),   # PKA  → P38
    (7, 10),  # PKA  → Jnk
    (8, 0),   # PKC  → Raf
    (8, 1),   # PKC  → Mek
    (8, 7),   # PKC  → PKA
    (8, 9),   # PKC  → P38
    (8, 10),  # PKC  → Jnk
}


def get_ground_truth_bnlearn(proteins=PROTEINS):
    """
    Return the bnlearn Sachs ground truth as an adjacency matrix aligned to
    `proteins` order. Entry [i,j]=1 means directed edge protein_i -> protein_j.
    """
    idx = {p: i for i, p in enumerate(proteins)}
    n = len(proteins)
    mat = np.zeros((n, n), dtype=int)
    for i, j in BNLEARN_SACHS_ARCS:
        mat[i, j] = 1
    return pd.DataFrame(mat, index=proteins, columns=proteins)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def harmonize_columns(df):
    """Rename columns to canonical protein names."""
    df = df.rename(columns=lambda c: c.capitalize())
    return df.rename(columns=COLUMN_MAP)[PROTEINS]


def remove_outliers(df, k=3.0):
    """Remove rows where any value is more than k SDs from the per-column mean."""
    z = (df - df.mean()) / df.std()
    keep = (z.abs() <= k).all(axis=1)
    return df[keep].reset_index(drop=True), keep, (~keep).sum()


def _stem_to_condition(stem):
    """Resolve a filename stem to a condition key."""
    if stem in FILENAME_TO_CONDITION:
        return FILENAME_TO_CONDITION[stem]
    parts = stem.split("__", 1)
    if len(parts) == 2:
        return parts[1]
    m = re.match(r"^\d+\.\s*(.+)$", stem)
    if m:
        return m.group(1)
    return stem

# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_sachs(xls_dir, transform=False, outlier_k=3.0, use_western_blot=False, proteins=PROTEINS):
    """
    Load all Sachs .xls files into a single DataFrame.

    Parameters
    ----------
    xls_dir : str or Path
    transform : bool
        Apply log1p transform before outlier removal. Default False.
    outlier_k : float or None
        Remove rows beyond this many SDs per protein per condition.
        Pass None to skip outlier removal. Default 3.0.
    use_western_blot : bool
        Include western blot validation conditions 10-14. Default False.
    proteins : list of str

    Returns
    -------
    pd.DataFrame with protein columns + 'condition' column.
    """
    xls_dir = Path(xls_dir)
    xls_files = sorted(xls_dir.glob("*.xls"))
    if not xls_files:
        raise FileNotFoundError(f"No .xls files found in {xls_dir}")

    dfs = []
    for fpath in xls_files:
        condition = _stem_to_condition(fpath.stem)
        if condition in WESTERN_BLOT_CONDITIONS and not use_western_blot:
            continue
        df = pd.read_excel(fpath, engine="xlrd", header=0)
        df = harmonize_columns(df)
        df["condition"] = condition
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if transform:
        df[proteins] = np.log1p(df[proteins])

    if outlier_k is not None:
        keep = pd.Series(True, index=df.index)
        for cond in df["condition"].unique():
            idx = df.index[df["condition"] == cond]
            _, cond_keep, _ = remove_outliers(df.loc[idx, proteins], k=outlier_k)
            keep.loc[idx[~cond_keep.values]] = False
        df = df[keep].reset_index(drop=True)

    return df # Dataframe with all conditions and protein columns, ready for analysis. 
