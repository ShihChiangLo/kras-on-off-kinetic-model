# -*- coding: utf-8 -*-
"""Figure 8. KRAS G12D against G12V: Bliss excess maps, allele-against-allele
scatter, and the engagement gain against RAS(ON) dose.

Input : data/matched_theta_grid.csv, data/fixed_concentration_grid.csv
Output: output/fig08_variants.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

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

C_ON = "#E8590C"
C_COMBO, C_K0 = "#0B7285", "#C2255C"


def log_edges(v):
    lv = np.log10(v)
    mid = (lv[:-1] + lv[1:]) / 2
    return 10 ** np.concatenate(([lv[0] - (mid[0] - lv[0])], mid,
                                 [lv[-1] + (lv[-1] - mid[-1])]))


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


md = pd.read_csv(DATA / "matched_theta_grid.csv")
fx = pd.read_csv(DATA / "fixed_concentration_grid.csv")

fig, axs = plt.subplots(2, 2, figsize=(PAGE_W, 5.3), constrained_layout=True)
lim2 = float(md.dE_bliss_excess.abs().max())
n2 = TwoSlopeNorm(vmin=-lim2, vcenter=0.0, vmax=lim2)
for j, v in enumerate(("G12D", "G12V")):
    s = md[md.variant == v]
    yo = np.sort(s.theta_OFF.unique()); xo = np.sort(s.theta_ON.unique())
    z = s.pivot_table(index="theta_OFF", columns="theta_ON",
                      values="dE_bliss_excess").reindex(index=yo, columns=xo).to_numpy()
    pcm = axs[0, j].pcolormesh(log_edges(xo), log_edges(yo), z, cmap="RdBu_r",
                               norm=n2, shading="flat")
    axs[0, j].set_xscale("log"); axs[0, j].set_yscale("log")
    axs[0, j].set_xlabel(r"$\theta_{\mathrm{ON}}$")
    axs[0, j].set_ylabel(r"$\theta_{\mathrm{OFF}}$" if j == 0 else "")
    axs[0, j].set_title(f"KRAS {v}")
fig.colorbar(pcm, ax=axs[0, :], label=r"Bliss excess $\Delta E$", shrink=0.9)

gd = md[md.variant == "G12D"].set_index(["theta_ON", "theta_OFF"])
gv = md[md.variant == "G12V"].set_index(["theta_ON", "theta_OFF"])
ix = gd.index.intersection(gv.index)
inv_m = (gd.loc[ix, "dE_bliss_excess"] <= gv.loc[ix, "dE_bliss_excess"])
gdf = fx[fx.variant == "G12D"].set_index(["on_dose_M", "off_dose_M"])
gvf = fx[fx.variant == "G12V"].set_index(["on_dose_M", "off_dose_M"])
ixf = gdf.index.intersection(gvf.index)
inv_f = (gdf.loc[ixf, "dE_bliss_excess"] <= gvf.loc[ixf, "dE_bliss_excess"])
ax = axs[1, 0]
# Exceptions: open squares for the matched-dose cells, crosses for the
# fixed-concentration cells. The two sets overlap in a small region at low
# dE, so they are drawn again, magnified, in an inset placed in the empty
# corner below the diagonal; the main panel keeps the full range.
C_X = "#7A1F00"


def draw_points(a, big):
    k = 1.6 if big else 1.0
    a.scatter(gv.loc[ix, "dE_bliss_excess"], gd.loc[ix, "dE_bliss_excess"],
              s=7 * k, color=C_COMBO, label=r"matched $\theta$", zorder=2)
    a.scatter(gvf.loc[ixf, "dE_bliss_excess"], gdf.loc[ixf, "dE_bliss_excess"],
              s=7 * k, facecolors="none", edgecolors=C_ON, lw=0.6,
              label="fixed concentration", zorder=2)
    a.scatter(gv.loc[ix, "dE_bliss_excess"][inv_m.values],
              gd.loc[ix, "dE_bliss_excess"][inv_m.values],
              s=24 * k, marker="s", facecolors="none", edgecolors="k",
              lw=0.9, zorder=3)
    a.scatter(gvf.loc[ixf, "dE_bliss_excess"][inv_f.values],
              gdf.loc[ixf, "dE_bliss_excess"][inv_f.values],
              s=20 * k, marker="x", color=C_X, lw=1.1, zorder=4)


draw_points(ax, False)
mx = float(max(gd.loc[ix, "dE_bliss_excess"].max(),
               gdf.loc[ixf, "dE_bliss_excess"].max()))
ax.plot([0, mx], [0, mx], color="k", lw=1, ls=":", zorder=1)
ax.set_xlabel(r"KRAS G12V $\Delta E$"); ax.set_ylabel(r"KRAS G12D $\Delta E$")
# Legend in the empty corner below the diagonal, under the inset.
ax.legend(loc="lower right", bbox_to_anchor=(0.0738, -0.0030),
          bbox_transform=ax.transData, frameon=False, handletextpad=0.2,
          handlelength=0.8, labelspacing=0.3, borderpad=0.0,
          borderaxespad=0.0, markerscale=1.6, markerfirst=False)
# Inset: the twelve crosses and the squares beside them. Nine of the twelve
# crosses fall into three groups of three that agree to five decimal places,
# each group sitting on a matched-dose square, so no magnification separates
# them.
ZX, ZY = (0.0073, 0.0090), (0.0052, 0.0077)
axin = ax.inset_axes([0.039, 0.0135, 0.0345, 0.0195], transform=ax.transData)
draw_points(axin, True)
axin.plot([0, mx], [0, mx], color="k", lw=1, ls=":", zorder=1)
axin.set_xlim(*ZX); axin.set_ylim(*ZY)
axin.set_xticks([]); axin.set_yticks([])
for sp in axin.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.35")
_zoom = ax.indicate_inset_zoom(axin, edgecolor="0.35", lw=0.6)
# The legend and the inset sit inside the panel. They are kept out of the
# constrained-layout solve so that the panel grid is set by the data alone.
ax.get_legend().set_in_layout(False)
axin.set_in_layout(False)
# Keep only the upper-left connector: the lower ones would cross the legend.
for k, cl in enumerate(_zoom.connectors):
    cl.set_visible(k == 1)
    cl.set_in_layout(False)
_zoom.set_in_layout(False)
ax.set_title(r"$\Delta E$, allele against allele")

for v, c in (("G12D", C_COMBO), ("G12V", C_K0)):
    s = md[(md.variant == v) & (np.isclose(md.theta_OFF, 1.0))].sort_values("theta_ON")
    axs[1, 1].plot(s.theta_ON, s.engagement_gain_vs_off_only_pp, color=c, lw=1.6,
                   label=f"KRAS {v}")
axs[1, 1].axhline(0, color="k", lw=1)
axs[1, 1].set_xscale("log"); axs[1, 1].set_xlabel(r"$\theta_{\mathrm{ON}}$")
axs[1, 1].set_ylabel("Engagement gain over\nRAS(OFF) alone (points)")
axs[1, 1].legend(frameon=False, loc="upper left", handlelength=1.0,
                 borderaxespad=0.2)
axs[1, 1].set_title("Engagement gain")

audit_fonts(fig, "Fig 8")
fig.savefig(OUT / "fig08_variants.png")
plt.close(fig)
print("wrote", OUT / "fig08_variants.png")
print(f"  matched-theta cells where G12V >= G12D: {int(inv_m.sum())}/{len(ix)}; "
      f"fixed concentration: {int(inv_f.sum())}/{len(ixf)}")
