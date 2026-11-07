"""Figures + LaTeX table for the camera-ready additions (JCI comparison, Psi-FCI on Sachs).

    python make_rebuttal_figures.py            # writes figures/
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})

# fixed colour per method (Okabe-Ito order); line style encodes the information used
STYLE = {
    "ICA-LiNGAM":             ("#E69F00", "-",  "o", "no labels, no targets"),
    "JCI-PC (env labels)":    ("#0072B2", "--", "s", "labels only"),
    "UT-IGSP":                ("#009E73", "--", "^", "labels only"),
    "Psi-FCI":                ("#CC79A7", "--", "D", "labels only"),
    "JCI-PC (known targets)": ("#56B4E9", ":",  "s", "labels + targets"),
    "IGSP":                   ("#D55E00", ":",  "v", "labels + targets"),
    "GIES":                   ("#7F7F7F", ":",  "P", "labels + targets"),
}


def ladder(out):
    s = pd.read_csv(os.path.join(HERE, "results", "jci_ladder_summary.csv"))
    s = s[(s.method != "JCI-PC (env labels)") | (s.alpha == 0.05)]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in ["ICA-LiNGAM", "JCI-PC (env labels)", "UT-IGSP", "JCI-PC (known targets)", "IGSP", "GIES"]:
        sub = s[s.method == m].sort_values("n_int")
        c, ls, mk, info = STYLE[m]
        lab = {"JCI-PC (env labels)": "JCI-PC",
               "JCI-PC (known targets)": "JCI-PC"}.get(m, m)
        ax.errorbar(sub.n_int, sub.shd_castle, yerr=sub.shd_castle_se, color=c, ls=ls, marker=mk,
                    ms=4, lw=1.5, capsize=2, label=f"{lab} [{info}]")
    ax.set_xscale("log"); ax.set_ylim(bottom=0)
    ax.set_xlabel("interventional samples (+100 observational)")
    ax.set_ylabel("SHD (mean ± s.e., 100 DAGs)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(out, "jci_ladder.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "jci_ladder.png"), dpi=200, bbox_inches="tight")


def load_sachs():
    fs = [f for f in glob.glob(os.path.join(HERE, "results", "sachs", "*.csv")) if "summary" not in f]
    df = pd.concat([pd.read_csv(f).assign(pool=os.path.basename(f).split("_")[0]) for f in fs])
    df["method"] = df["method"].replace({"ica_lingam_castle": "ICA-LiNGAM", "jci": "JCI-PC (env labels)",
                                         "psifci": "Psi-FCI", "gies": "GIES", "igsp": "IGSP"})
    return df


def sachs(out):
    df = load_sachs()
    g = df.groupby(["pool", "method", "n"]).agg(
        c=("shd_castle", "mean"), csd=("shd_castle", "std"), s=("shd", "mean"), ssd=("shd", "std"),
        f=("dir_f1", "mean"), fsd=("dir_f1", "std"), k=("shd", "size")).reset_index().fillna(0)
    series = [("full9", "ICA-LiNGAM", "ICA-LiNGAM, 9 cond."),
              ("full9", "JCI-PC (env labels)", "JCI (labels only), 9 cond."),
              ("squires", "JCI-PC (env labels)", "JCI (labels only), 5-cond. subset"),
              ("full9", "Psi-FCI", "Ψ-FCI, 9 cond."),
              ("squires", "Psi-FCI", "Ψ-FCI, 5-cond. subset"),
              ("squires", "IGSP", "IGSP (known targets), 5-cond. subset"),
              ("squires", "GIES", "GIES (known targets), 5-cond. subset")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    panels = [("c", "csd", "SHD (gcastle, as Fig. 4)"), ("s", "ssd", "strict SHD"),
              ("f", "fsd", "directed-edge F1")]
    for ax, (col, sd, title) in zip(axes, panels):
        for pool, m, lab in series:
            sub = g[(g.pool == pool) & (g.method == m)].sort_values("n")
            c, _, mk, _ = STYLE[m]
            ls = "-" if pool == "full9" else ("--" if m in ("JCI-PC (env labels)", "Psi-FCI") else ":")
            x = np.arange(len(sub))
            ax.plot(x, sub[col], color=c, ls=ls, marker=mk, ms=3.5, lw=1.5, label=lab)
            ax.fill_between(x, sub[col] - sub[sd], sub[col] + sub[sd], color=c, alpha=0.12, lw=0)
        ax.set_xticks(range(6)); ax.set_xticklabels(["200", "500", "1k", "2k", "4k", "all"])
        ax.set_title(title, fontsize=8); ax.set_xlabel("samples drawn")
    axes[0].set_ylabel("mean ± 1 SD (30 draws)")
    axes[2].set_ylim(0, 0.5)
    axes[0].axhline(17, color="#999", lw=0.8, ls=":"); axes[1].axhline(17, color="#999", lw=0.8, ls=":")
    axes[1].annotate("empty graph", (0, 17.15), fontsize=6, color="#666")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "sachs_label_only.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "sachs_label_only.png"), dpi=200, bbox_inches="tight")

    # LaTeX table: n = 1000 and full pool
    rows = []
    for pool, m, lab in series:
        sub = g[(g.pool == pool) & (g.method == m)].sort_values("n")
        r1, rall = sub[sub.n == 1000].iloc[0], sub.iloc[-1]
        lab_tex = lab.replace("Ψ", r"$\Psi$")
        cells = []
        for r in (r1, rall):
            if r.k > 1:
                cells += [f"{r.c:.1f} $\\pm$ {r.csd:.1f}", f"{r.s:.1f} $\\pm$ {r.ssd:.1f}", f"{r.f:.2f}"]
            else:  # deterministic method on the full pool: a single run
                cells += [f"{r.c:.0f}", f"{r.s:.0f}", f"{r.f:.2f}"]
        rows.append(lab_tex + " & " + " & ".join(cells) + r" \\")
    with open(os.path.join(out, "sachs_table_rows.tex"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(g[g.n >= 4000][["pool", "method", "n", "k"]])


def label_noise(out):
    s = pd.read_csv(os.path.join(HERE, "results", "label_noise_summary.csv"))
    panels = [("env", 1000), ("env", 10000), ("target", 1000), ("target", 10000)]
    titles = {"env": "environment-label noise", "target": "target noise"}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), sharey=True)
    for ax, (noise, n) in zip(axes, panels):
        ica = s[(s.method == "ICA-LiNGAM") & (s.n_int == n)].iloc[0]
        ax.axhline(ica.shd, color=STYLE["ICA-LiNGAM"][0], lw=1.5, label="ICA-LiNGAM (no labels)")
        ax.axhspan(ica.shd - ica.se, ica.shd + ica.se, color=STYLE["ICA-LiNGAM"][0], alpha=0.15, lw=0)
        for m in ["JCI-PC (env labels)", "UT-IGSP", "JCI-PC (known targets)", "IGSP", "GIES"]:
            sub = s[(s.method == m) & (s.noise == noise) & (s.n_int == n)].sort_values("eps")
            if sub.empty:
                continue
            c, ls, mk, info = STYLE[m]
            lab = {"JCI-PC (env labels)": "JCI-PC", "JCI-PC (known targets)": "JCI-PC"}.get(m, m)
            ax.errorbar(sub.eps, sub.shd, yerr=sub.se, color=c, ls=ls, marker=mk, ms=3.5, lw=1.4,
                        capsize=2, label=f"{lab} [{info}]")
        ax.set_title(f"{titles[noise]}\n$n$ = {n:,}", fontsize=7.5)
        ax.set_xlabel("noise rate $\\varepsilon$")
    axes[0].set_ylim(0, 11.8)
    axes[0].set_ylabel("SHD (mean ± s.e.)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(out, "label_noise.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "label_noise.png"), dpi=200, bbox_inches="tight")


def label_noise_main(out):
    """One panel for the main text: environment-label noise at n = 10^4."""
    s = pd.read_csv(os.path.join(HERE, "results", "label_noise_summary.csv"))
    n = 10000
    fig, ax = plt.subplots(figsize=(3.0, 2.15))
    ica = s[(s.method == "ICA-LiNGAM") & (s.n_int == n)].iloc[0]
    ax.axhline(ica.shd, color=STYLE["ICA-LiNGAM"][0], lw=1.8, label="ICA-LiNGAM [no labels]")
    ax.axhspan(ica.shd - ica.se, ica.shd + ica.se, color=STYLE["ICA-LiNGAM"][0], alpha=0.15, lw=0)
    for m, lab in [("JCI-PC (env labels)", "JCI [labels only]"),
                   ("JCI-PC (known targets)", "JCI [labels + targets]")]:
        sub = s[(s.method == m) & (s.noise == "env") & (s.n_int == n)].sort_values("eps")
        c, ls, mk, _ = STYLE[m]
        ax.errorbar(sub.eps, sub.shd, yerr=sub.se, color=c, ls=ls, marker=mk, ms=4, lw=1.6,
                    capsize=2, label=lab)
    ax.set_xlabel("fraction of samples mislabeled $\\varepsilon$")
    ax.set_ylabel("SHD (mean ± s.e.)")
    ax.set_ylim(0, 9.8)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "label_noise_main.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "label_noise_main.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ladder(a.out); sachs(a.out); label_noise(a.out); label_noise_main(a.out)
