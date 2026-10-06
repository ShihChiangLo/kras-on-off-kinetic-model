# -*- coding: utf-8 -*-
"""Figure 5. AMG 510 dose-response, with and without intrinsic GTP hydrolysis.

Input : data/dose_response.csv (written by ras_switch/simulate.py)
Output: output/fig05_dose_response.png
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "dose_response.csv"
OUT = HERE / "output" / "fig05_dose_response.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

COLORS = {
    "G12C": "#ff7f0e",
    "G12C without intrinsic GTP hydrolysis": "#ffbb78",
    "G12D": "#1f77b4",
    "G12D without intrinsic GTP hydrolysis": "#aec7e8",
    "G12V": "#2ca02c",
    "G12V without intrinsic GTP hydrolysis": "#98df8a",
}

# Drawn at 7 x 5 in, the size the TIFF is submitted at, so the point sizes
# below are the printed point sizes; audit_fonts() enforces the 8-12 pt that
# PLOS allows inside a figure. The figure carries no title (PLOS puts it in the
# caption); the legend names the curves without intrinsic GTP hydrolysis.
MIN_PT, MAX_PT = 8.0, 12.0

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

plot_df = pd.read_csv(SRC).rename(columns={
    "curve": "Model_label", "drug_M": "Drug_conc",
    "mean_signal_pct": "Mean_Relative_signal_percentage",
    "sd_signal_pct": "Std_Relative_signal_percentage"})
unknown = sorted(set(plot_df["Model_label"]) - set(COLORS))
if unknown:
    raise SystemExit(f"curves without a colour: {unknown}")

fig, ax = plt.subplots(figsize=(7, 5))
for label, subdf in plot_df.groupby("Model_label", sort=False):
    subdf = subdf.sort_values("Drug_conc")
    ax.errorbar(subdf["Drug_conc"], subdf["Mean_Relative_signal_percentage"],
                yerr=subdf["Std_Relative_signal_percentage"], marker="o",
                linestyle="-", capsize=4, color=COLORS.get(label), label=label)

ax.set_xscale("log")
ax.set_xlabel("AMG510 Concentration (M)", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Relative Signal (%)", fontsize=12, fontweight="bold")
ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=2)
ax.tick_params(axis="both", which="minor", length=4, width=1)
ax.set_ylim(30, 105)
ax.legend(fontsize=11)
fig.tight_layout()
audit_fonts(fig, "Fig 5")
fig.savefig(OUT, dpi=300)
plt.close(fig)
print("wrote", OUT)
for label, subdf in plot_df.groupby("Model_label", sort=False):
    sub = subdf.sort_values("Drug_conc")
    print(f"  {label:<42} n={len(sub):2d}  last={sub['Mean_Relative_signal_percentage'].iloc[-1]:8.4f}")
