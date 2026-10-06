# -*- coding: utf-8 -*-
"""Figure 3. Transfected RAS-GTP: Michaelis-Menten model, non-MM model, experiment.

Input : data/ras_gtp_summary.csv (written by ras_switch/simulate.py)
Output: output/fig03_transfected_validation.png
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
OUT = HERE / 'output' / 'fig03_transfected_validation.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Same size and type rule as Figure 2: authored at the placed width, 8-12 pt
# enforced by audit_fonts().
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

# Stites et al., read from Table S3 of that article's supporting material at
# the same nine concentration sets. That table reports no KRAS G12C value, so
# the Stites series is absent for that variant.
STITES_TRANSFECTED = {"WT": [6.4, 7.9, 10.0, 3.4, 4.6, 6.5, 4.8, 6.3, 8.4],
                      "G12V": [75.9, 82.2, 87.0, 76.0, 81.5, 86.3, 74.7, 82.0, 86.4]}

# The GTP-bound fraction of the transfected subpopulation. Wild-type and G12V
# are transfected HRas in COS1 cells, Boykevisch 2006 Fig 2A; the G12C figure is
# the roughly 75 % that two reviews restate for KRAS G12C, which is a
# restatement rather than a primary measurement. None of the three is published
# with an uncertainty, so this series carries no error bars. S5 Table lists all
# three with their constructs, cell lines and assays.
EXPERIMENTAL_TRANSFECTED = {"WT": (5.3, None), "G12V": (81.6, None), "G12C": (75.0, None)}

study = {r['variant']: (float(r['mean_transfected_gtp_pct']),
                        float(r['sd_transfected_gtp_pct']))
         for r in csv.DictReader(open(SUM))}
COLORS = {"Stites Model (Michaelis–Menten)": "#A0C4C7", "This Study (Non-MM Model)": "#F28E2B",
          "Reference Experimental Data": "#466A8F"}
ORDER = list(COLORS)
SPECIES = ("WT", "G12V", "G12C")


def series(ds, sp):
    if ds == ORDER[0]:
        if sp not in STITES_TRANSFECTED:
            return None, None
        v = STITES_TRANSFECTED[sp]
        return float(np.mean(v)), float(np.std(v, ddof=1))
    if ds == ORDER[1]:
        return study[sp]
    return EXPERIMENTAL_TRANSFECTED[sp]


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


bw = 0.25
x = np.arange(len(SPECIES))
fig, ax = plt.subplots(figsize=(PAGE_W, 3.9), constrained_layout=True)
for di, ds in enumerate(ORDER):
    first = True
    for xi, sp in enumerate(SPECIES):
        m, s = series(ds, sp)
        if m is None:
            continue
        pos = x[xi] + di * bw
        ax.bar(pos, m, color=COLORS[ds], width=bw, edgecolor='black',
               label=ds if first else "", yerr=s, capsize=0)
        ax.text(pos, 2, f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        first = False
ax.set_xlabel('Transfected RAS', fontweight='bold')
ax.set_ylabel('Transfected RAS-GTP (%)', fontweight='bold')
ax.set_xticks(x + bw)
ax.set_xticklabels(SPECIES)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', frameon=False)
audit_fonts(fig, 'Fig 3')
fig.savefig(OUT)
plt.close(fig)
print('wrote', OUT)
for ds in ORDER:
    for sp in SPECIES:
        m, s = series(ds, sp)
        print(f'  {ds[:26]:<28}{sp:<5} mean={"skip" if m is None else f"{m:8.4f}"}  err={s}')
