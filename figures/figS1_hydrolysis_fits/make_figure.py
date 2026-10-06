"""S1 Fig: apparent CypA:RMC-7977-assisted GTP hydrolysis rate of each KRAS variant.

The phosphate-release time courses of Fig 2b of Cuevas-Navarro et al.
(Nature 637, 224-229, 2025) were digitized into data/. Each trace is fitted with
a one-rule PySB model,

    RAS-GTP -> RAS-GDP + Pi,   rate k_hyd_app,

so that Pi(t) = R0 (1 - exp(-k_hyd_app t)). The assay runs at a saturating
CypA:inhibitor concentration, so binding and catalysis are lumped into the one
apparent constant. R0, the hydrolysable RAS-GTP pool, is shared by all variants
and fixed at 1.07 uM: the plateau of the G12D trace, the only trace that reaches
its plateau within 90 min. The Pi baseline is fixed at 0, leaving k_hyd_app as the
only free parameter per variant. Its 95 % interval comes from a residual-resampling
bootstrap (200 replicates, fixed seed) and is conditional on the fixed R0.

The model runs in minutes and uM, the units of the source figure; rate constants
are reported in s^-1.

Writes output/figS1_hydrolysis_fits.png and output/fitted_rate_constants.csv.
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysb import Model, Monomer, Observable, Parameter, Rule
from pysb.simulator import ScipyOdeSimulator
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "RMC_7977_assisted_GTP_hydrolysis.csv")
OUT_DIR = os.path.join(HERE, "output")

SEC_PER_MIN = 60.0
R0_FIXED = 1.07          # uM
PI_BASELINE = 0.0        # uM
N_BOOTSTRAP = 200
RANDOM_SEED = 20260821
K_BOUNDS = (1e-5, 5.0)   # 1/min


def build_hydrolysis_model():
    model = Model("ras_gtp_hydrolysis", _export=False)
    ras = Monomer("RAS", ["nuc"], {"nuc": ["GTP", "GDP"]}, _export=False)
    pi = Monomer("Pi", _export=False)
    r0 = Parameter("R0", R0_FIXED, _export=False)
    pi0 = Parameter("Pi_0", PI_BASELINE, _export=False)
    k = Parameter("k_hyd_app", 0.05, _export=False)
    for component in (ras, pi, r0, pi0, k):
        model.add_component(component)
    model.add_component(Rule("gtp_hydrolysis", ras(nuc="GTP") >> ras(nuc="GDP") + pi(),
                             k, _export=False))
    model.add_component(Observable("PO4_release", pi(), _export=False))
    model.initial(ras(nuc="GTP"), r0)
    model.initial(pi(), pi0)
    return model


def per_second(k_per_min):
    return k_per_min / SEC_PER_MIN


def sci_tex(x, sig=3):
    """Mathtext scientific notation, e.g. 1.44\\times10^{-3}."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return f"{mant:.{sig - 1}f}\\times10^{{{exp}}}"


def load_digitized_data(csv_path):
    """Return {variant: (time_min, pi_uM)} from the CSV's VARIANT_x, VARIANT_y column pairs.

    The pairs have different lengths; blank cells are dropped pair-wise.
    """
    raw = pd.read_csv(csv_path)
    raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith("Unnamed")]]
    data = OrderedDict()
    for v in [c[:-2] for c in raw.columns if c.endswith("_x")]:
        block = raw[[f"{v}_x", f"{v}_y"]].apply(pd.to_numeric, errors="coerce").dropna()
        t = block[f"{v}_x"].to_numpy(float)
        y = block[f"{v}_y"].to_numpy(float)
        order = np.argsort(t)
        data[v] = (t[order], y[order])
    return data


class VariantSimulator:
    """One simulator per variant, evaluated at that variant's sampling times.

    The traces do not all start at t = 0, so t = 0 is prepended to the time grid
    and dropped from the output.
    """

    def __init__(self, t_data):
        self.t_data = np.asarray(t_data, float)
        self._prepended = self.t_data[0] > 0.0
        tspan = np.concatenate(([0.0], self.t_data)) if self._prepended else self.t_data
        self.sim = ScipyOdeSimulator(build_hydrolysis_model(), tspan=tspan,
                                     integrator_options={"rtol": 1e-10, "atol": 1e-12})

    def po4(self, k_hyd_app, R0):
        res = self.sim.run(param_values={"k_hyd_app": float(k_hyd_app), "R0": float(R0),
                                         "Pi_0": PI_BASELINE})
        y = np.asarray(res.observables["PO4_release"], float)
        return y[1:] if self._prepended else y


def _scan_k(vsim, y_obs, R0, k_grid):
    sse = [float(np.sum((vsim.po4(k, R0) - y_obs) ** 2)) for k in k_grid]
    return float(k_grid[int(np.argmin(sse))])


def fit_k(vsim, y_obs, R0, k_init=None, scan="wide"):
    """Least-squares k_hyd_app with R0 fixed, fitted in log space.

    A grid scan brackets the optimum before the local refinement; started from a
    poor guess, the refinement can stop early because the objective comes from
    an ODE solver. The bootstrap uses a narrow grid around the point estimate.
    """
    if scan == "wide":
        k_init = _scan_k(vsim, y_obs, R0, np.geomspace(K_BOUNDS[0], 1.0, 200))
    else:
        k_init = _scan_k(vsim, y_obs, R0, np.geomspace(k_init * 0.4, k_init * 2.5, 9))
    sol = least_squares(
        lambda theta: vsim.po4(np.exp(theta[0]), R0) - y_obs,
        x0=[np.log(k_init)],
        bounds=([np.log(K_BOUNDS[0])], [np.log(K_BOUNDS[1])]),
        method="trf", xtol=1e-12, ftol=1e-12,
    )
    return float(np.exp(sol.x[0]))


def goodness_of_fit(y_obs, y_fit):
    resid = y_obs - y_fit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2, float(np.sqrt(ss_res / len(y_obs)))


def bootstrap_ci(vsim, y_obs, y_fit, R0, k_hat, n, rng):
    """95 % percentile interval from refits to the fitted curve plus resampled residuals.

    Residuals are mean-centred first: with no intercept in the model they do not
    sum to zero, and resampling them raw would shift every replicate the same way.
    """
    resid = y_obs - y_fit
    resid = resid - resid.mean()
    draws = np.empty(n)
    for i in range(n):
        y_star = y_fit + rng.choice(resid, size=resid.size, replace=True)
        draws[i] = fit_k(vsim, y_star, R0, k_init=k_hat, scan="narrow")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def run_fits(data):
    rng = np.random.default_rng(RANDOM_SEED)
    rows, curves = [], {}
    for v, (t, y) in data.items():
        vsim = VariantSimulator(t)
        k_hat = fit_k(vsim, y, R0_FIXED)
        y_fit = vsim.po4(k_hat, R0_FIXED)
        r2, rmse = goodness_of_fit(y, y_fit)
        lo, hi = bootstrap_ci(vsim, y, y_fit, R0_FIXED, k_hat, N_BOOTSTRAP, rng)
        rows.append(OrderedDict(
            variant=v,
            k_hyd_app_per_s=per_second(k_hat),
            ci95_low_per_s=per_second(lo),
            ci95_high_per_s=per_second(hi),
            k_hyd_app_per_min=k_hat,
            half_life_min=np.log(2) / k_hat,
            R0_uM=R0_FIXED,
            R_squared=r2,
            RMSE_uM=rmse,
            n_points=len(t),
        ))
        curves[v] = (t, y, y_fit)
        print(f"{v:>5}: k = {per_second(k_hat):.4e} /s  "
              f"[{per_second(lo):.4e}, {per_second(hi):.4e}]   R2 = {r2:.4f}")
    df = pd.DataFrame(rows).sort_values("k_hyd_app_per_s", ascending=False)
    return df.reset_index(drop=True), curves


INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
ACCENT = "#2a78d6"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf",
             "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK_SOFT,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_panels(df, curves, R0_batch, out_path):
    """Small multiples: one panel per variant, plus a rate-comparison panel."""
    order = df["variant"].tolist()               # fastest -> slowest
    n = len(order)
    ncols, nrows = 4, 2
    # Axes are not shared: one cell holds the rate bar chart, which has its own
    # scales. The time-course panels get identical limits instead.
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 5.6))
    axes = axes.ravel()

    # The rate bar chart sits top right; the seven time courses fill the other cells.
    BAR_CELL = 3
    panel_cells = [i for i in range(nrows * ncols) if i != BAR_CELL]

    for cell, v in zip(panel_cells, order):
        ax = axes[cell]
        t, y, y_fit = curves[v]
        row = df.set_index("variant").loc[v]
        _style_axes(ax)

        t_dense = np.linspace(0, 90, 300)
        ax.plot(t_dense, R0_batch * (1 - np.exp(-row.k_hyd_app_per_min * t_dense)),
                color=ACCENT, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.plot(t, y, "o", markersize=4.2, markerfacecolor="white",
                markeredgecolor=INK, markeredgewidth=1.1, linestyle="none", zorder=4)

        ax.set_title(f"KRAS({v})", fontsize=10, color=INK, loc="left", pad=6)
        # Annotation goes in the corner the curve leaves empty.
        if R0_batch * (1 - np.exp(-row.k_hyd_app_per_min * 90)) > 0.6:
            xy, ha, va = (0.97, 0.06), "right", "bottom"
        else:
            xy, ha, va = (0.04, 0.94), "left", "top"
        ax.text(*xy,
                f"$k$ = ${sci_tex(row.k_hyd_app_per_s)}$ s$^{{-1}}$\n"
                f"$R^2$ = {row.R_squared:.3f}",
                transform=ax.transAxes, ha=ha, va=va,
                fontsize=8, color=INK_SOFT)
        ax.set_xlim(0, 92)
        ax.set_ylim(0, 1.2)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if cell % ncols != 0:
            ax.set_yticklabels([])
        if cell < ncols:
            ax.set_xticklabels([])

    # Rate ranking with 95 % bootstrap intervals.
    ax = axes[BAR_CELL]
    _style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ypos = np.arange(len(order))[::-1]
    # Plotted in units of 1e-3 s^-1.
    SCALE = 1e-3
    k_s = df["k_hyd_app_per_s"] / SCALE
    ax.barh(ypos, k_s, height=0.6, color=BLUE_RAMP[:len(order)], zorder=3)
    ax.errorbar(k_s, ypos,
                xerr=[k_s - df["ci95_low_per_s"] / SCALE,
                      df["ci95_high_per_s"] / SCALE - k_s],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=8.5, color=INK)
    ax.set_title("Apparent rate constant", fontsize=10, color=INK, loc="left", pad=6)
    ax.set_xlabel("$k_{hyd,app}$  ($\\times10^{-3}$ s$^{-1}$)", fontsize=8.5)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlim(0, df["ci95_high_per_s"].max() / SCALE * 1.15)

    for cell in range(nrows * ncols):
        if cell != BAR_CELL and cell not in panel_cells[:n]:
            axes[cell].set_visible(False)

    fig.supxlabel("Time (min)", fontsize=9.5, color=INK, x=0.40, y=0.035)
    fig.supylabel("PO$_4$ release (µM)", fontsize=9.5, color=INK, x=0.004)
    fig.suptitle(
        "CYPA–RMC-7977-assisted GTP hydrolysis by KRAS variants — PySB fit to digitized Fig. 2b",
        fontsize=11.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.925,
             f"Points, digitized data · Line, single-exponential PySB model with shared "
             f"$R_0$ = {R0_batch:.3f} µM "
             f"(declared, fixed)"
             f" and $Pi_0$ = 0 · "
             f"bars show 95% bootstrap CI",
             fontsize=8, color=INK_SOFT, ha="left")

    fig.tight_layout(rect=[0.02, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_digitized_data(CSV_PATH)
    df, curves = run_fits(data)
    df.to_csv(os.path.join(OUT_DIR, "fitted_rate_constants.csv"), index=False)
    out = os.path.join(OUT_DIR, "figS1_hydrolysis_fits.png")
    plot_panels(df, curves, R0_FIXED, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
