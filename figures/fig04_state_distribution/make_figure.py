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
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "state_distribution.csv"
OUT = HERE / "output" / "fig04_state_distribution.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

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

fig, ax = plt.subplots(figsize=(8, 5))
for index, model_label in enumerate(MODEL_ORDER):
    subdf = mean_long[mean_long["Model_label"] == model_label].sort_values("state")
    ax.bar(x + offsets[index], subdf["mean"].to_numpy(), width=bar_width,
           yerr=subdf["std"].to_numpy(), capsize=4, label=model_label,
           edgecolor="black", linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(state_order)
ax.tick_params(axis="both", which="major", labelsize=15, length=6, width=2)
ax.tick_params(axis="both", which="minor", length=4, width=1)
ax.set_ylabel("Percentage (%)", fontsize=15, fontweight="bold")
ax.set_title("RAS state distribution (mean ± SD across testing matrix)",
             fontsize=15, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(fontsize=15, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, dpi=300)
plt.close(fig)
print("wrote", OUT)
for _, r in mean_long.iterrows():
    print(f"  {str(r['Model_label']):<5} {str(r['state']):<16} mean={r['mean']:8.4f}  sd={r['std']:8.4f}")
