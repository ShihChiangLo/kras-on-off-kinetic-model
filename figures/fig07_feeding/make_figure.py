# -*- coding: utf-8 -*-
"""Figure 7. Accessible RAS-GDP, RAS(OFF) engagement, and the engagement gain
against RAS(OFF) dose, on KRAS G12D.

Input : data/reference_point.csv, data/matched_theta_grid.csv
Output: output/fig07_feeding.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Authored 5.2 in wide, while the manuscript places the figure at its 5.25 in
# text width, so everything is enlarged by about one per cent on the page: an
# 8 pt label prints at 8.1 pt. PLOS asks for 8-12 pt inside a figure at the
# published size, and audit_fonts() checks the authored sizes against that
# window, which the one per cent does not take anything out of.
PAGE_W, MIN_PT, MAX_PT = 5.2, 8.0, 12.0


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
    "font.size": 8.0, "axes.titlesize": 8.0, "axes.titleweight": "bold",
    "axes.labelsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0, "legend.title_fontsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

C_OFF, C_ON, C_BASE = "#4C6EF5", "#E8590C", "#868E96"
C_COMBO, C_K0 = "#0B7285", "#C2255C"

# Rotated rather than horizontal: at 8 pt in a 2.5 in panel the horizontal
# forms collide.
ARMS = ["vehicle", "ON, no cat.", "ON", "OFF", "ON+OFF, no cat.", "ON+OFF"]
ARM_KW = dict(rotation=40, ha="right", rotation_mode="anchor")


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


pil = pd.read_csv(DATA / "reference_point.csv").iloc[0]
md = pd.read_csv(DATA / "matched_theta_grid.csv")

# Two rows rather than three across: six arm labels do not fit under a 1.7 in
# panel at 8 pt.
fig = plt.figure(figsize=(PAGE_W, 5.1), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
a1 = fig.add_subplot(gs[0, 0])
a2 = fig.add_subplot(gs[0, 1])
a3 = fig.add_subplot(gs[1, :])
cols = [C_BASE, C_ON, C_ON, C_OFF, C_K0, C_COMBO]

dacc = [pil["D_access_end_vehicle_M"],
        pil["D_access_end_on_only_kcat0_M"], pil["D_access_end_on_only_M"],
        pil["D_access_end_off_only_M"], pil["D_access_end_combo_kcat0_M"],
        pil["D_access_end_combo_M"]]
# Plotted in units of 1e-8 M rather than with matplotlib's offset text, so the
# bars read 7.83 and 10.4, the same figures the Results quote in M.
dacc = [d / 1e-8 for d in dacc]
a1.bar(range(6), dacc, color=cols)
a1.axhline(dacc[0], color="k", ls=":", lw=1)
a1.set_xticks(range(6)); a1.set_xticklabels(ARMS, **ARM_KW)
a1.set_ylabel("Accessible mutant\nRAS-GDP ($10^{-8}$ M)")
a1.set_title("Accessible RAS-GDP")

lo = float(pil["off_bound_pct_off_only"]); hi = float(pil["off_bound_pct_combo"])
eng = [0.0, 0.0, 0.0, lo, float(pil["off_bound_pct_combo_kcat0"]), hi]
a2.bar(range(6), eng, color=cols)
a2.axhline(lo, color=C_OFF, ls=":", lw=1.1, zorder=0)
a2.annotate("", xy=(5.58, hi), xytext=(5.58, lo),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
a2.text(5.92, (hi + lo) / 2,
        f"+{pil['engagement_gain_vs_off_only_pp']:.2f}\npoints",
        va="center", ha="left", linespacing=1.2)
a2.set_xlim(-0.7, 8.2)
a2.set_xticks(range(6)); a2.set_xticklabels(ARMS, **ARM_KW)
a2.set_ylabel("RAS(OFF)-bound\nmutant RAS (%)")
a2.set_title("RAS(OFF) engagement")

for v, c in (("G12D", C_COMBO), ("G12V", C_K0)):
    s = md[(md.variant == v) & (np.isclose(md.theta_ON, 1.0))].sort_values("theta_OFF")
    a3.plot(s.theta_OFF, s.engagement_gain_vs_kcat0_pp, color=c, lw=1.6)
    a3.plot(s.theta_OFF, s.engagement_gain_vs_off_only_pp, color=c, lw=1.4, ls="--")
a3.axhline(0, color="k", lw=1)
a3.set_xscale("log")
a3.set_xlabel(r"$\theta_{\mathrm{OFF}}$")
a3.set_ylabel("RAS(OFF) engagement gain\n(percentage points)")
a3.legend(handles=[Line2D([], [], color=C_COMBO, lw=1.6, label="KRAS G12D"),
                   Line2D([], [], color=C_K0, lw=1.6, label="KRAS G12V")],
          frameon=False, loc="upper left")
a3.set_title("Engagement gain against RAS(OFF) dose")

audit_fonts(fig, "Fig 7")
# No tight bbox: cropping changes the saved width, and the printed point size
# is (placed width / saved width) x the authored size.
fig.savefig(OUT / "fig07_feeding.png")
plt.close(fig)
print("wrote", OUT / "fig07_feeding.png")
print(f"  arrow spans {hi - lo:.4f} pp; CSV gives "
      f"{float(pil['engagement_gain_vs_off_only_pp']):.4f} pp")
