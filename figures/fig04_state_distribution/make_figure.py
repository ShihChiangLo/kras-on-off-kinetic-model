# -*- coding: utf-8 -*-
"""Figure 4. RAS state distribution across the nine-condition testing matrix.

Input : data/state_distribution.csv (written by ras_switch/simulate.py)
Output: output/fig04_state_distribution.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "state_distribution.csv"
OUT = HERE / "output" / "fig04_state_distribution.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Drawn at 6.4 x 4.0 in, the size the TIFF is submitted at, so the point
# sizes below are the printed point sizes. PLOS allows at most 7.5 in of width
# and 8-12 pt text inside a figure; audit_fonts() enforces the type size.
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

STATE_COLS = ["total_free_pct", "total_gtp_pct", "total_gdp_pct"]
MODEL_ORDER = ["WT", "G12C", "G12D", "G12V"]

df = pd.read_csv(SRC)
if df.groupby("variant").size().ne(9).any():
    raise SystemExit("expected nine conditions per variant")
dfp = df.groupby("variant", as_index=False).agg(
    **{f"mean_{c}": (c, "mean") for c in STATE_COLS},
    **{f"sd_{c}": (c, "std") for c in STATE_COLS},
).rename(columns={"variant": "Model_label"})
dfp["Model_label"] = pd.Categorical(dfp["Model_label"], categories=MODEL_ORDER, ordered=True)
dfp = dfp.sort_values("Model_label")

mean_cols = [f"mean_{c}" for c in STATE_COLS]
std_cols = [f"sd_{c}" for c in STATE_COLS]
state_map = dict(zip(mean_cols, ["nucleotide free", "GTP bound", "GDP bound"]))

mean_long = dfp.melt(id_vars=["Model_label"], value_vars=mean_cols,
                     var_name="metric", value_name="mean")
std_long = dfp.melt(id_vars=["Model_label"], value_vars=std_cols,
                    var_name="std_metric", value_name="std")
mean_long["std"] = std_long["std"].to_numpy()
mean_long["state"] = mean_long["metric"].map(state_map)

state_order = ["nucleotide free", "GTP bound", "GDP bound"]
mean_long["state"] = pd.Categorical(mean_long["state"], categories=state_order, ordered=True)
mean_long = mean_long.sort_values(["state", "Model_label"])

x = np.arange(len(state_order))
bar_width = 0.18
offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * bar_width

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for index, model_label in enumerate(MODEL_ORDER):
    subdf = mean_long[mean_long["Model_label"] == model_label].sort_values("state")
    ax.bar(x + offsets[index], subdf["mean"].to_numpy(), width=bar_width,
           yerr=subdf["std"].to_numpy(), capsize=4, label=model_label,
           edgecolor="black", linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(state_order)
ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=2)
ax.tick_params(axis="both", which="minor", length=4, width=1)
ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(fontsize=12, loc="upper left")
fig.tight_layout()
audit_fonts(fig, "Fig 4")
fig.savefig(OUT, dpi=300)
plt.close(fig)
print("wrote", OUT)
for _, r in mean_long.iterrows():
    print(f"  {str(r['Model_label']):<5} {str(r['state']):<16} mean={r['mean']:8.4f}  sd={r['std']:8.4f}")
