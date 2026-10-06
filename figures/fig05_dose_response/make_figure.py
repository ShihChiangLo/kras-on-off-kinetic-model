# -*- coding: utf-8 -*-
"""Figure 5. AMG 510 dose-response, with and without intrinsic GTP hydrolysis.

Input : data/dose_response.csv (written by ras_switch/simulate.py)
Output: output/fig05_dose_response.png
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

# The en dash and the spaces either side of the newline are part of the
# published title. Matplotlib centres each line on its own, so dropping either
# space moves that line sideways relative to the other.
TITLE = ("Dose–Response curve for different RAS mutants \n"
         " with/without intrinsic GTP hydrolysis")

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
ax.set_xlabel("AMG510 Concentration (M)", fontsize=15, fontweight="bold")
ax.set_ylabel("Mean Relative Signal (%)", fontsize=15, fontweight="bold")
ax.set_title(TITLE, fontsize=15, fontweight="bold")
ax.tick_params(axis="both", which="major", labelsize=15, length=6, width=2)
ax.tick_params(axis="both", which="minor", length=4, width=1)
ax.set_ylim(30, 105)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
plt.close(fig)
print("wrote", OUT)
for label, subdf in plot_df.groupby("Model_label", sort=False):
    sub = subdf.sort_values("Drug_conc")
    print(f"  {label:<42} n={len(sub):2d}  last={sub['Mean_Relative_signal_percentage'].iloc[-1]:8.4f}")
