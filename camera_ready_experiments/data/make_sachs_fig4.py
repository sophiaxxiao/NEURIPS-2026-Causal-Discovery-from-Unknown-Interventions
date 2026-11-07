"""Per-condition Sachs CSVs with exactly the Fig. 4 preprocessing, using the
student's loader (real_data_experiments/sachs_protein/src/load_sachs.py):
original .xls files, conditions 1-9, raw values (transform=False), per-condition
3-SD outlier removal. Writes data/sachs_fig4/<condition>.csv (6456 rows total)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SACHS = os.path.join(HERE, "..", "..", "real_data_experiments", "sachs_protein")
sys.path.insert(0, os.path.join(SACHS, "src"))
from load_sachs import load_sachs  # noqa: E402

df = load_sachs(os.path.join(SACHS, "data", "original_data"), transform=False, use_western_blot=False)
out = os.path.join(HERE, "sachs_fig4"); os.makedirs(out, exist_ok=True)
for cond, sub in df.groupby("condition"):
    sub.drop(columns="condition").to_csv(os.path.join(out, f"{cond}.csv"), index=False)
    print(f"{cond:20s} {len(sub)}")
print("total", len(df))
