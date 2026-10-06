# -*- coding: utf-8 -*-
"""Figure in S6 Text. Sensitivity of the covalent arm to K_I.

Input : data/ki_sensitivity.csv   one row per K_I (K_I_M) and intrinsic-hydrolysis
                               scenario (scenario)
Output: output/figS6_ki_sensitivity.png

KRAS G12C at [OFF] = 1e-6 M (ref_off_M). K_I is swept from 6.0e-8 to 6.0e-6 M
with k_inact/K_I held at 9900 M^-1 s^-1 (kinact_s / K_I_M). No simulation:
every point is a row of the CSV.

Two panels, one y scale each (no twin axis):
  A  time to 90% capture with the RAS(OFF) inhibitor alone (t90_off_alone_h), in
     both intrinsic-hydrolysis scenarios, and with the RAS(ON) inhibitor at the dose
     that minimises that time (t90_best_full_h);
  B  the ratio of the two times (time_speedup_x), with the 1.5-fold level.
The vertical line in both panels is the K_I the model uses, 2.20e-7 M.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import pandas as pd                  # noqa: E402

HERE = Path(__file__).resolve().parent
# The script is meant to be run from its own folder, like simulate.py, so the
# repository root goes on the import path for `two_arm`.
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from two_arm.parameters import K_I    # noqa: E402

SRC = HERE / "data" / "ki_sensitivity.csv"
OUT = HERE / "output" / "figS6_ki_sensitivity.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
SC = {"cuevas_dmso": ("DMSO arm", "#2a78d6"), "cuevas_cypa": ("CypA arm", "#eb6834")}
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

d = pd.read_csv(SRC, float_precision="round_trip").sort_values("K_I_M")
fig, (a, b) = plt.subplots(2, 1, figsize=(5.6, 5.2), sharex=True,
                           constrained_layout=True)
MK = {"cuevas_dmso": "o", "cuevas_cypa": "s"}
for sc, (lab, col) in SC.items():
    s = d[d.scenario == sc]
    mk = MK[sc]
    a.plot(s.K_I_M, s.t90_off_alone_h, "-", lw=1.6, color=col, marker=mk, ms=4,
           markevery=3, markerfacecolor="white", markeredgewidth=1.2,
           label=f"{lab}, RAS(OFF) agent alone")
    b.plot(s.K_I_M, s.time_speedup_x, "-", lw=1.6, color=col, marker=mk, ms=4,
           markevery=3, markerfacecolor="white", markeredgewidth=1.2, label=lab)
# The two scenarios' with-RAS(ON) curves differ by less than 0.01 h at every
# K_I in the sweep, so they are drawn once, in neutral ink, rather than as two
# lines one of which would simply hide the other.
_s = d[d.scenario == "cuevas_dmso"]
a.plot(_s.K_I_M, _s.t90_best_full_h, "--", lw=1.6, color=INK, marker="^", ms=4,
       markevery=(1, 3), markerfacecolor="white", markeredgewidth=1.2,
       label="with the RAS(ON) agent at its best dose\n(both scenarios; they differ by < 0.01 h)")

for ax in (a, b):
    ax.set_xscale("log")
    ax.axvline(K_I, color=MUTED, lw=0.9, ls=":")
    ax.grid(axis="y", color=GRID, lw=0.6)
a.set_ylim(3.2, 8.6)
a.set_ylabel("time to 90% capture (h)")
a.set_title("A  the time itself moves with $K_{I}$", loc="left", color=INK)
a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.04, 0.40), ncol=1)
b.axhline(1.5, color=MUTED, lw=0.9)
b.set_ylim(1.0, 2.1)
b.set_ylabel("speedup from the RAS(ON) agent (×)")
b.set_xlabel("$K_{I}$ (M); the dotted line is the value the model uses, "
             "$2.20\\times10^{-7}$ M")
b.set_title("B  the speedup does not", loc="left", color=INK)
b.text(6.5e-8, 1.53, "1.5-fold, the level this run was scored on", fontsize=7.5,
       color=MUTED)
b.legend(frameon=False, loc="lower right", ncol=2)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
