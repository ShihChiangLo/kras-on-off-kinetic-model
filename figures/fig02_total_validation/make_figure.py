# -*- coding: utf-8 -*-
"""Figure 2. Total RAS-GTP: Michaelis-Menten model, non-MM model, experiment.

Input : data/ras_gtp_summary.csv (written by ras_switch/simulate.py)
Output: output/fig02_total_validation.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUM = HERE / 'data' / 'ras_gtp_summary.csv'
OUT = HERE / 'output' / 'fig02_total_validation.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

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

plt.rcParams.update({'figure.dpi': 600, 'savefig.dpi': 600, 'font.size': 8.0,
                     'axes.labelsize': 8.0, 'xtick.labelsize': 8.0,
                     'ytick.labelsize': 8.0, 'legend.fontsize': 8.0,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Stites et al., per-condition percentages over the nine-condition testing matrix.
STITES_TOTAL = {
    "WT":   [6.4, 7.9, 10.0, 3.4, 4.6, 6.5, 4.8, 6.3, 8.4],
    "G12D": [37.8, 43.1, 48.5, 36.3, 40.5, 45.5, 37.1, 43.3, 48.5],
    "G12V": [49.8, 53.0, 55.6, 49.2, 51.7, 53.9, 48.4, 52.4, 55.1],
}

# mean, spread, spread_is_published. Gibbs 1990 Table I reports mean +- S.E.
# (n = 3-5) for the two NIH3T3 values; Bollag 1996 reports 45.7 % for G12D with
# no uncertainty, so that bar carries none and has no spread to carry.
# S5 Table lists all three with their constructs and assay conditions.
EXPERIMENTAL_TOTAL = {
    "WT":   (7.0, 1.3, True),
    "G12D": (45.7, None, False),
    "G12V": (71.0, 3.8, True),
}

study = {}
for r in csv.DictReader(open(SUM)):
    study[r['variant']] = (float(r['mean_total_gtp_pct']),
                           float(r['sd_total_gtp_pct']))

DATASET_COLORS = {
    "Stites Model (Michaelis–Menten)": "#A0C4C7",
    "This Study (Non-MM Model)": "#F28E2B",
    "Reference Experimental Data": "#466A8F",
}
ORDER = list(DATASET_COLORS)
SPECIES = ("WT", "G12D", "G12V")


def series(ds, sp):
    if ds == ORDER[0]:
        v = STITES_TOTAL[sp]
        return float(np.mean(v)), float(np.std(v, ddof=1)), True
    if ds == ORDER[1]:
        m, s = study[sp]
        return m, s, True
    return EXPERIMENTAL_TOTAL[sp]


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


bar_width = 0.25
x = np.arange(len(SPECIES))
fig, ax = plt.subplots(figsize=(PAGE_W, 3.9), constrained_layout=True)
for di, ds in enumerate(ORDER):
    first = True
    for xi, sp in enumerate(SPECIES):
        m, s, has_spread = series(ds, sp)
        pos = x[xi] + di * bar_width
        ax.bar(pos, m, color=DATASET_COLORS[ds], width=bar_width, edgecolor='black',
               label=ds if first else "", yerr=(s if has_spread else None), capsize=0)
        ax.text(pos, 2, f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        first = False
ax.set_xlabel('Transfected RAS', fontweight='bold')
ax.set_ylabel('Total RAS-GTP (%)', fontweight='bold')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(SPECIES)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', frameon=False)
audit_fonts(fig, 'Fig 2')
# No tight bbox: cropping changes the saved width, and the printed point size
# is (placed width / saved width) x the authored size.
fig.savefig(OUT)
plt.close(fig)
print('wrote', OUT)
for ds in ORDER:
    for sp in SPECIES:
        m, s, has_spread = series(ds, sp)
        print(f'  {ds[:28]:<30} {sp:<5} mean={m:8.4f}  err={"none" if not has_spread else f"{s:.4f}"}')
