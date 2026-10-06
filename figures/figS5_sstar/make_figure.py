# -*- coding: utf-8 -*-
"""S5 Text figure. S*, the state selectivity above which raising the dose still
helps, against alpha_RAF.

Input : the CSVs listed in SSTAR_FILES and COMMON_PROBE_FILES, in data/
Output: output/figS5_sstar.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from two_arm import parameters as P

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

ALLELE = {"G12V": "#1f4b73", "G12D": "#b5651d"}
MUTED = "#6b6b6b"
A_ORDER = ["alpha=inf", "alpha=10", "alpha=3", "alpha=1"]
SPREAD_LO, SPREAD_HI = P.MEASURED_SELECTIVITY_SPREAD
S_MEDIAN = P.MEASURED_SELECTIVITY_MEDIAN

# Order matters: rows are de-duplicated on (variant, alpha, t_end, rescue,
# intrinsic scenario) keeping the last occurrence, so a later file overrides an
# earlier one for the same condition.
SSTAR_FILES = [
    "sstar_g12v_alpha_1_inf.csv",
    "sstar_g12v_alpha_3_10.csv",
    "sstar_g12d.csv",
    "sstar_g12v_late_readout.csv",
    "sstar_g12v_assay_intrinsic.csv",
    "sstar_g12v_rescue.csv",
]
COMMON_PROBE_FILES = [
    "sstar_g12v_common_probe_alpha1.csv",
    "sstar_g12v_common_probe_alphainf.csv",
]


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


df = pd.concat([pd.read_csv(DATA / f) for f in SSTAR_FILES], ignore_index=True)
df = df.drop_duplicates(subset=["variant", "alpha_label", "t_end_s", "rescue",
                                "intrinsic_scenario"], keep="last")
base = df[(df.t_end_s == 1e4) & (~df.rescue.astype(bool)) & (df.intrinsic_scenario == "model")]

common = {}
for f in COMMON_PROBE_FILES:
    for _, r in pd.read_csv(DATA / f).iterrows():
        common[r["alpha_label"]] = r

fig, ax = plt.subplots(figsize=(PAGE_W, 4.0), constrained_layout=True)
ax.axhspan(SPREAD_LO, SPREAD_HI, color="#9e9e9e", alpha=0.12, zorder=0)
ax.axhline(S_MEDIAN, color=MUTED, ls=":", lw=1.0, zorder=1)
x = np.arange(len(A_ORDER))
for v in ("G12V", "G12D"):
    sub = base[base.variant == v].set_index("alpha_label")
    ys, los, his, xs = [], [], [], []
    for i, a in enumerate(A_ORDER):
        if a not in sub.index:
            continue
        r = sub.loc[a]
        xs.append(i); ys.append(r.S_star)
        if v == "G12V" and a in common:
            c = common[a]
            ax.plot([i, i], [r.bracket_lo, r.bracket_hi], color=ALLELE[v],
                    lw=6, alpha=0.10, solid_capstyle="butt", zorder=1)
            los.append(c["bracket_lo"]); his.append(c["bracket_hi"])
        else:
            los.append(r.bracket_lo); his.append(r.bracket_hi)
    for xi, lo, hi in zip(xs, los, his):
        ax.plot([xi, xi], [lo, hi], color=ALLELE[v], lw=6, alpha=0.30,
                solid_capstyle="butt", zorder=2)
    dashed = (v == "G12D")
    ax.plot(xs, ys, "--" if dashed else "-", color=ALLELE[v], lw=1.2,
            zorder=3, alpha=0.85 if dashed else 1.0)
    ax.plot(xs, ys, "_", color=ALLELE[v], ms=13, mew=1.8, zorder=4,
            label=f"KRAS {v}")
    for xi, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}", (xi, y), textcoords="offset points",
                    xytext=(7, -11 if v == "G12D" else 4), color=ALLELE[v])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([a.replace("alpha=", "").replace("inf", r"$\infty$")
                    for a in A_ORDER])
ax.set_xlabel(r"$\alpha_{\mathrm{RAF}}$")
ax.set_ylabel("$S*$, the state selectivity above\nwhich raising the dose still helps")
h = [Patch(facecolor="#9e9e9e", alpha=0.12,
           label=f"measured $S$ range, {SPREAD_LO:g}-{SPREAD_HI:g}"),
     Line2D([], [], color=MUTED, ls=":", label=f"median $S$ = {S_MEDIAN:g}"),
     Patch(facecolor=ALLELE["G12V"], alpha=0.10, label="wider bracket, coarser $S$ grid")]
# Legend under the axes, not inside them: at 8 pt a five-entry legend covers
# the alpha = 1 brackets, which are the widest in the figure.
fig.legend(handles=ax.get_legend_handles_labels()[0] + h,
           loc="outside lower center", ncol=2, frameon=False,
           handlelength=1.6, labelspacing=0.35, columnspacing=1.4)
ax.set_xlim(-0.35, len(A_ORDER) - 0.55)
ax.set_ylim(top=1.5e3)

audit_fonts(fig, "S5 Text figure")
fig.savefig(OUT / "figS5_sstar.png")
plt.close(fig)
print("wrote", OUT / "figS5_sstar.png")
