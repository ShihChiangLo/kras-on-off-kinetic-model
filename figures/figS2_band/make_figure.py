# -*- coding: utf-8 -*-
"""S2 Fig. Required catalysis fold change R against alpha_RAF, by state
selectivity, for KRAS G12V and G12D.

Input : data/required_catalysis_fold.csv
Output: output/figS2_band.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

PAGE_W, MIN_PT, MAX_PT = 5.2, 8.0, 12.0
plt.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.size": 8.0, "axes.titlesize": 8.0, "axes.titleweight": "bold",
    "axes.labelsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0, "legend.title_fontsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

S_RAMP = ["#c6d9ec", "#8ab4d8", "#4a86b8", "#1f4b73"]
ALLELE = {"G12V": "#1f4b73", "G12D": "#b5651d"}
INK = "#222222"
S_ORDER = ["S=inf", "S=182", "S=16", "S=7.7"]
A_ORDER = ["alpha=inf", "alpha=10", "alpha=3", "alpha=1"]


def audit_fonts(fig, name):
    bad = set()
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_text() or "").strip():
            continue
        pt = round(t.get_fontsize(), 2)
        if pt < MIN_PT - 1e-6 or pt > MAX_PT + 1e-6:
            bad.add((pt, (t.get_text() or "")[:40].replace("\n", " ")))
    if bad:
        for pt, s in sorted(bad):
            print(f"    FONT VIOLATION {name}: {pt} pt  {s!r}")
        raise SystemExit(f"{name}: text outside {MIN_PT}-{MAX_PT} pt")


d = pd.read_csv(DATA / "required_catalysis_fold.csv")
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 3.3), constrained_layout=True)
handles = None
for ax, v in zip(axes, ["G12V", "G12D"]):
    dv = d[d["variant"] == v]
    ends = {}
    for i, sl in enumerate(S_ORDER):
        y = [float(dv[(dv["S_label"] == sl) & (dv["alpha_label"] == al)]
                   ["R_required_fold"].iloc[0]) for al in A_ORDER]
        ax.plot(np.arange(len(A_ORDER)), y, "-o", color=S_RAMP[i], lw=1.6, ms=3.5,
                label=sl.replace("S=", "S = ").replace("inf", r"$\infty$"))
        ends[sl] = y[-1]
    # Draw the R = 1 line before placing the labels: it expands the y axis (the
    # G12D panel's data spans 0.28-0.38 but the line sits at 1.0), and the label
    # offsets are measured against the axis range.
    ax.axhline(1.0, color="#999999", lw=1, ls="--")
    values = [float(c) for c in dv["R_required_fold"]]
    ylo, yhi = min(values + [1.0]), max(values + [1.0])
    pad = 0.05 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    # Label only the two ends of the band: on the G12D panel the two lowest
    # differ by 0.004, which no offset separates legibly at this size.
    for sl in (max(ends, key=ends.get), min(ends, key=ends.get)):
        ax.annotate(f"{ends[sl]:.2f}", (len(A_ORDER) - 1, ends[sl]),
                    textcoords="offset points", xytext=(5, -2), color=INK)
    ax.set_xticks(np.arange(len(A_ORDER)))
    ax.set_xticklabels([a.replace("alpha=", "").replace("inf", r"$\infty$")
                        for a in A_ORDER])
    ax.set_xlabel(r"$\alpha_{\mathrm{RAF}}$")
    ax.set_xlim(-0.25, len(A_ORDER) - 0.30)
    ax.set_title(f"KRAS {v}", color=ALLELE[v])
    # Both panels carry the y label: they do not share a y axis (G12V spans
    # 1.6-3.4, G12D 0.28-0.38), and an unlabelled right panel reads as if they did.
    ax.set_ylabel("Required fold change $R$")
    if v == "G12V":
        handles = ax.get_legend_handles_labels()
fig.legend(*handles, loc="outside lower center", ncol=4, frameon=False,
           title="State selectivity", handlelength=1.6, columnspacing=1.2)

audit_fonts(fig, "S2 Fig")
fig.savefig(OUT / "figS2_band.png")
plt.close(fig)
print("wrote", OUT / "figS2_band.png")
