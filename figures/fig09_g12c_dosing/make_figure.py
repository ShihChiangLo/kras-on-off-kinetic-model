# -*- coding: utf-8 -*-
"""Figure 9. What a covalent RAS(OFF) inhibitor can reach on KRAS G12C, how
fast, and at what RAS(ON) dose.

Input : data/baseline.csv     drug-free steady state, one row per RAS pool
        data/dose_grid.csv     benefit and cost against [ON] and [OFF]
        data/on_dose_interval.csv      the [ON] interval shaded in panel D
        data/capture_timecourse.csv   covalent capture against time
Output: output/fig09_g12c_dosing.png

Every panel is KRAS G12C in the Cuevas DMSO intrinsic-hydrolysis scenario
(scenario == "cuevas_dmso"; mukhyd and K_I_M columns of data/baseline.csv).
Wild-type and mutant RAS are both present, 1:1 (pool_M column of
data/baseline.csv). Covalent capture is a percentage of the mutant pool,
wild-type occupancy a percentage of the wild-type pool.

A  Drug-free steady state: GDP- and GTP-loaded fraction of each pool. The
   covalent inhibitor binds GDP-loaded mutant RAS only, so the GDP bar is what it
   can reach at any instant.
B  Covalent capture against time at [OFF] = 1e-6 M: the RAS(OFF) inhibitor alone
   (cond "full", on_M 0), with the RAS(ON) inhibitor at 3e-7 M and its measured
   k_cat,TCI (cond "full"), and at the same dose with k_cat,TCI = 0
   (cond "shield"). Dashed line: 90 % capture. Dotted line: panel A's
   GDP-loaded fraction of the mutant pool.
C  The same axes for RAS(ON) doses from 3e-8 to 1e-5 M, measured k_cat,TCI.
D  Benefit (upper: mutant capture at 6 h) and cost (lower: wild-type RAS
   occupied by the RAS(ON) inhibitor at 24 h) against RAS(ON) dose, both with
   [OFF] = 1e-6 M present; the grey dashed line is the cost with [OFF] = 0.
   The band is the [ON] interval whose 24 h wild-type occupancy lies between
   1 % and 5 % (window_lo_M, window_hi_M of data/on_dose_interval.csv). The two
   thresholds define an interval; they are not clinical criteria.

The figure is authored at the width it is placed at in the manuscript (the
5.25 in text width) and saved at 600 dpi without a tight bounding box, so the
point sizes below are the printed point sizes. PLOS asks for 8-12 pt inside
figures at the published size; audit_fonts() fails the build if any text
leaves that window, and audit_canvas() fails it if any text falls outside the
saved canvas.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

HERE = Path(__file__).resolve().parent
# The script is meant to be run from its own folder, like simulate.py, so the
# repository root goes on the import path for `two_arm`.
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from two_arm.parameters import (CAPTURE_CRITERION,  # noqa: E402
                                OCCUPANCY_LIMITS)

DATA = HERE / "data"
OUT = HERE / "output" / "fig09_g12c_dosing.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

INK, MUTED, REF = "#0b0b0b", "#6b6a66", "#52514e"
C_OFF, C_FULL, C_SHIELD = "#8d8b86", "#2a78d6", "#eb6834"
C_COST, C_BAND = "#b3283c", "#cfe0f2"
C_GTP, C_GDP = "#c9c7c1", "#2a78d6"

PAGE_W, MIN_PT, MAX_PT = 5.25, 8.0, 12.0


# PLOS asks for Arial, Times or Symbol inside figures. Arial is not distributed
# with this repository (see README), so require_arial() stops the script if
# matplotlib cannot find it, rather than letting matplotlib substitute another
# typeface. Mathematical text is set in Arial as well.
def require_arial():
    for style, weight in (("normal", "normal"), ("normal", "bold"),
                          ("italic", "normal")):
        prop = font_manager.FontProperties(family="Arial", style=style,
                                           weight=weight)
        path = font_manager.findfont(prop, fallback_to_default=False)
        face = font_manager.get_font(path)
        if (face.family_name != "Arial"
                or ("Bold" in face.style_name) != (weight == "bold")
                or ("Italic" in face.style_name) != (style == "italic")):
            raise SystemExit(f"Arial ({style}, {weight}) not found; got {path}")


require_arial()
# Neither the sf nor the cal slot of mathematical text is drawn in these
# figures. Both are pointed at Arial so that every slot names a font that is
# present: left at its default, cal names a script typeface, and matplotlib
# prints a fallback warning when it cannot find one.
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
                     "mathtext.cal": "Arial"})

plt.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

MAIN = "cuevas_dmso"             # intrinsic-hydrolysis scenario drawn in every panel
OFF_REC = 1e-06                  # [OFF] in M; the recommended_off_M column of data/on_dose_interval.csv
WT_LEVELS = list(OCCUPANCY_LIMITS)   # wild-type occupancy (%) bounding the band in panel D
CRIT = CAPTURE_CRITERION         # capture criterion (%), dashed line in panels B and C
ON_B = [0.0, 3e-07]              # [ON] in M drawn in panel B
ON_C = [0.0, 3e-08, 1e-07, 3e-07, 1e-06, 1e-05]   # [ON] in M drawn in panel C
SHIELD_ON = 3e-07                # [ON] in M of the k_cat,TCI = 0 curve in panel B

# Layout at the placed width (inches and axes-fraction offsets).
FIG_H = 7.5
LEFT = 0.115
RIGHT = 0.965
TOP = 0.968
BOTTOM = 0.20
HSPACE = 0.62
WSPACE = 0.62
LEG_X = -0.30                    # left edge of the keys under panels B, C and D
LEG_A_Y = -0.27
LEG_B_Y = -0.22
LEG_C_Y = -0.19
LEG_D_Y = -0.45


def molar(x):
    """3e-07 -> '$3\\times10^{-7}$ M'; 1e-07 -> '$10^{-7}$ M' (no e notation)."""
    e = int(np.floor(np.log10(x) + 1e-9))
    m = x / 10.0 ** e
    if abs(m - round(m)) > 1e-6:
        raise ValueError(f"dose {x!r} is not a whole multiple of a power of ten")
    m = int(round(m))
    return (f"$10^{{{e}}}$ M" if m == 1 else f"${m}\\times10^{{{e}}}$ M")


KCAT = r"$k_{\mathrm{cat,TCI}}$"
OFF_LABEL = "1 \u00b5M"         # OFF_REC as the caption writes it (upright micro sign)


def _time_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(60.0, 86400.0)
    ax.set_xticks([60, 600, 3600, 21600, 86400])
    ax.set_xticklabels(["1 min", "10 min", "1 h", "6 h", "24 h"])
    ax.set_xlabel("time")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _on_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(8e-10, 1.3e-5)
    ax.set_xlabel("ON drug (M)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_a(ax, b):
    d = b[b.scenario == MAIN]
    pools = ["mutant G12C", "wild-type"]
    y = np.arange(len(pools))[::-1]
    gdp = [float(d[d.pool == p].gdp_pct.iloc[0]) for p in pools]
    gtp = [float(d[d.pool == p].gtp_pct.iloc[0]) for p in pools]
    ax.barh(y, gdp, color=C_GDP, height=0.52, label="GDP-loaded")
    ax.barh(y, gtp, left=gdp, color=C_GTP, height=0.52, label="GTP-loaded")
    ax.set_yticks(y)
    ax.set_yticklabels(["mutant\nG12C", "wild-type"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of that pool (no drug, steady state)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.47))
    ax.set_ylim(-0.75, 1.75)


def panel_b(ax, c, gdp_pct):
    d = c[(c.scenario == MAIN) & (c.off_M == OFF_REC)]
    ax.axhline(CRIT, color=REF, lw=0.9, ls="--", zorder=1)
    # Panel A's GDP-loaded fraction: the pool that exists before any
    # hydrolysis. Every curve crosses it well after t = 0 rather than at t = 0,
    # because the covalent step (1/k_obs of about 9 min at this dose) overlaps
    # the hydrolysis that refills the GDP-loaded pool. Where each one crosses
    # differs: about 13 min for the RAS(OFF) inhibitor alone, a little earlier
    # with the RAS(ON) inhibitor added, and about twice as late for the
    # k_cat,TCI = 0 curve, which has no catalytic resupply.
    ax.axhline(gdp_pct, color=C_GDP, lw=0.9, ls=":", zorder=1)
    on_hi = max(ON_B)
    for on, cond, col, ls, lab in [
            (0.0, "full", C_OFF, "-", "OFF drug alone"),
            (on_hi, "full", C_FULL, "-",
             f"+ ON {molar(on_hi)}, measured {KCAT}"),
            (SHIELD_ON, "shield", C_SHIELD, "--",
             f"+ ON {molar(SHIELD_ON)}, {KCAT} = 0")]:
        p = d[(d.on_M == on) & (d.cond == cond)].sort_values("t_s")
        ax.plot(p.t_s, p.captured_pct, ls, lw=1.4, color=col, label=lab,
                zorder=3)
    ax.set_ylim(0, 103)
    ax.set_ylabel("mutant RAS covalently captured (%)")
    _time_axis(ax)
    ax.legend(frameon=False, loc="upper left")


def panel_c(ax, c):
    d = c[(c.scenario == MAIN) & (c.off_M == OFF_REC) & (c.cond == "full")]
    ons = sorted(x for x in ON_C)
    cmap = plt.get_cmap("viridis")
    ax.axhline(CRIT, color=REF, lw=0.9, ls="--", zorder=1)
    for i, on in enumerate(ons):
        p = d[d.on_M == on].sort_values("t_s")
        col = C_OFF if on == 0.0 else cmap(0.12 + 0.72 * i / max(1, len(ons) - 1))
        lab = "OFF drug alone" if on == 0.0 else f"+ ON {molar(on)}"
        ax.plot(p.t_s, p.captured_pct, "-", lw=1.3, color=col, label=lab,
                zorder=3)
    ax.set_ylim(0, 103)
    ax.set_ylabel("mutant RAS covalently captured (%)")
    _time_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4,
              columnspacing=0.9)


b = pd.read_csv(DATA / "baseline.csv")
g = pd.read_csv(DATA / "dose_grid.csv")
v = pd.read_csv(DATA / "on_dose_interval.csv")
c = pd.read_csv(DATA / "capture_timecourse.csv")

# Every series drawn below must exist in the tables; an empty selection would
# otherwise leave a curve out of the figure without an error.
_required = {
    "baseline mutant pool": (b.scenario == MAIN) & (b.pool == "mutant G12C"),
    "baseline wild-type pool": (b.scenario == MAIN) & (b.pool == "wild-type"),
    "dose grid benefit/cost at [OFF]": (g.scenario == MAIN) & (g.off_M == OFF_REC) & (g.on_M > 0),
    "dose grid cost at [OFF] = 0": (g.scenario == MAIN) & (g.off_M == 0.0) & (g.on_M > 0),
    "[ON] interval": v.scenario == MAIN,
    "time course panel B, k_cat = 0": (c.scenario == MAIN) & (c.off_M == OFF_REC)
                              & (c.on_M == SHIELD_ON) & (c.cond == "shield"),
}
for on in sorted(set(ON_B) | set(ON_C)):
    _required[f"time course [ON] = {on:g} M"] = ((c.scenario == MAIN) & (c.off_M == OFF_REC)
                                         & (c.on_M == on) & (c.cond == "full"))
_missing = [name for name, mask in _required.items() if not mask.any()]
if _missing:
    raise SystemExit(f"rows missing from data/: {_missing}")
if float(v[v.scenario == MAIN].recommended_off_M.iloc[0]) != OFF_REC or OFF_REC != 1e-06:
    raise SystemExit("OFF_REC does not match recommended_off_M in data/on_dose_interval.csv")

gdp = float(b[(b.scenario == MAIN) & (b.pool == "mutant G12C")]
            .gdp_pct.iloc[0])

def audit_fonts(fig, name):
    bad = set()
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_text() or "").strip() or not t.get_visible():
            continue
        pt = round(t.get_fontsize(), 2)
        if (pt < MIN_PT - 1e-6 or pt > MAX_PT + 1e-6
                or t.get_fontname() != "Arial"):
            bad.add((pt, t.get_fontname(), (t.get_text() or "")[:40].replace("\n", " ")))
    if bad:
        for pt, fam, s in sorted(bad):
            print(f"    FONT VIOLATION {name}: {pt} pt  {fam}  {s!r}")
        raise SystemExit(f"{name}: text outside {MIN_PT}-{MAX_PT} pt or not in Arial")


def audit_canvas(fig, name):
    """Everything drawn must lie inside the saved canvas (no tight bbox here,
    so anything outside it would be cut off)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    tol = 0.5 / fig.dpi
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
        raise SystemExit(f"{name}: drawn content spans ({bb.x0:.3f}, {bb.y0:.3f}) to "
                         f"({bb.x1:.3f}, {bb.y1:.3f}) in, canvas is {W:.2f} x {H:.2f} in")


fig = plt.figure(figsize=(PAGE_W, FIG_H))
gs = fig.add_gridspec(2, 2, left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
                      height_ratios=[1.0, 1.25], hspace=HSPACE, wspace=WSPACE)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
sub = gs[1, 1].subgridspec(2, 1, hspace=0.12, height_ratios=[1.0, 1.0])
axD1 = fig.add_subplot(sub[0])
axD2 = fig.add_subplot(sub[1], sharex=axD1)

panel_a(axA, b)
axA.get_legend().remove()
axA.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, LEG_A_Y),
           ncol=2, handlelength=1.4, columnspacing=1.0)
panel_b(axB, c, gdp)
panel_c(axC, c)

# Panel D: benefit and cost share the x axis, each with its own y axis.
d = g[(g.scenario == MAIN) & (g.off_M == OFF_REC) & (g.on_M > 0)] \
    .sort_values("on_M")
z = g[(g.scenario == MAIN) & (g.off_M == 0.0) & (g.on_M > 0)] \
    .sort_values("on_M")
row = v[v.scenario == MAIN].iloc[0]
for ax in (axD1, axD2):
    ax.axvspan(float(row.window_lo_M), float(row.window_hi_M),
               color=C_BAND, lw=0, zorder=0)
    _on_axis(ax)
axD1.plot(d.on_M, d.benefit_capture_6h_pct, "-", lw=1.4, color=C_FULL,
          label=f"mutant capture at 6 h,\n[OFF] = {OFF_LABEL}", zorder=4)
axD1.set_ylim(80, 100)
axD1.set_ylabel("mutant RAS\ncaptured at 6 h (%)", color=C_FULL)
axD1.tick_params(axis="y", colors=C_FULL)
axD1.spines["left"].set_color(C_FULL)
axD1.set_xlabel("")
axD1.tick_params(axis="x", labelbottom=False)
for lvl in WT_LEVELS:
    axD2.axhline(lvl, color=REF, lw=0.8, zorder=1)
axD2.plot(d.on_M, d.cost_wt_occupancy_24h_pct, "-", lw=1.4, color=C_COST,
          label=f"wild-type occupancy at 24 h,\n[OFF] = {OFF_LABEL}",
          zorder=4)
axD2.plot(z.on_M, z.cost_wt_occupancy_24h_pct, "--", lw=1.2, color=MUTED,
          label="wild-type occupancy at 24 h,\n[OFF] = 0", zorder=3)
axD2.set_yscale("log")
axD2.set_ylim(0.01, 100)
axD2.set_ylabel("wild-type RAS\noccupied at 24 h (%)", color=C_COST)
axD2.tick_params(axis="y", colors=C_COST)
axD2.spines["left"].set_color(C_COST)
h1, l1 = axD1.get_legend_handles_labels()
h2, l2 = axD2.get_legend_handles_labels()
axD2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left",
            bbox_to_anchor=(LEG_X, LEG_D_Y), handlelength=1.6)

# Keys under panels B and C, and shorter y labels.
for ax in (axB, axC):
    ax.set_ylabel("mutant RAS captured (%)")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
axB.legend(frameon=False, loc="upper left", bbox_to_anchor=(LEG_X, LEG_B_Y),
           handlelength=1.6)
axC.legend(frameon=False, loc="upper left", bbox_to_anchor=(LEG_X, LEG_C_Y),
           ncol=2, handlelength=1.4, columnspacing=0.8)

for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C"), (axD1, "D")):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10, pad=5)
audit_fonts(fig, "Fig 9")
audit_canvas(fig, "Fig 9")
# No tight bbox: cropping changes the saved width, and the printed point size
# is (placed width / saved width) x the authored size.
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
print(f"  mutant G12C GDP-loaded at steady state: {gdp:.2f} %")
print(f"  shaded [ON] interval: {float(row.window_lo_M):.3g} - {float(row.window_hi_M):.3g} M")
