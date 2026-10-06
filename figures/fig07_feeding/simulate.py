# -*- coding: utf-8 -*-
"""Simulations behind Fig 7: how catalytic release feeds the RAS(OFF) arm.

    python3 simulate.py            both tables, into results/
    python3 simulate.py --help

Writes
    reference_point.csv       the reference design point on KRAS G12D reduced
                              to one row: Bliss bookkeeping, catalytic-release
                              output, accessible RAS-GDP and RAS(OFF)
                              engagement, arm by arm
    matched_theta_grid.csv    KRAS G12D and G12V over a 13 x 13 grid of doses,
                              each arm's dose expressed as a multiple of its
                              own dissociation constant

The dose grid is shared with Fig 8, which plots the two alleles against each
other, so `matched_theta_grid.csv` is written into both figures' data folders
rather than computed twice.

Each dose point is read at six arms. `vehicle`, `on_only`, `off_only` and
`combo` are the four an experiment would have. `combo_kcat0` and `on_only_kcat0`
are controls built from the same model: identical binding, identical dose, with
catalytic release set to exactly zero. Neither is a second compound, and the
difference between an arm and its control is what the RAS(ON) inhibitor's
catalytic activity contributes on top of its binding.

This script and `../fig08_variants/simulate.py` are the only two places the
reversible RAS(OFF) probe is driven over a dose grid; the readouts themselves
live in `two_arm.simulate` so that the two cannot drift apart.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from two_arm.model import build                                 # noqa: E402
from two_arm.parameters import (KCAT_TCI, KD2_DECIMAL, KD_OFF_GDP,  # noqa: E402
                                KOFF_OFF, KON_OFF, THETA_REFERENCE)
from two_arm.simulate import (ARMS, bliss, compile_model,       # noqa: E402
                              escape_rate, initial_overrides, integrate,
                              series)

#: Dose multiples both axes of the grid run over: three decades either side of
#: the dissociation constant, thirteen points, so the reference design point
#: (one times the dissociation constant on both arms) is a grid point.
THETA_DECADES = (-3.0, 3.0)
THETA_POINTS = 13

#: The two alleles the reversible probe is characterized on. Both are
#: stimulated by the RAS(ON) inhibitor, at rates that differ thirteenfold,
#: which is what makes the comparison informative.
VARIANTS = ("G12D", "G12V")

#: Cases per call to the integrator. A case's trajectory does not depend on
#: which batch it was in, but the lowest species concentration reported with it
#: is the lowest over its batch, so the batch size is fixed here rather than
#: left to the caller.
CHUNK = 48


def theta_grid():
    """The dose multiples of one axis, spaced evenly in log dose."""
    lo, hi = THETA_DECADES
    return np.logspace(lo, hi, THETA_POINTS)


def arm_doses(arm, on_dose, off_dose, catalytic_release):
    """(RAS(ON) dose, RAS(OFF) dose, catalytic release) for one arm.

    An arm named `..._kcat0` is the zero-catalytic-release control of the arm
    it is named after: same doses, catalytic release exactly zero.
    """
    on = on_dose if arm.startswith(("on_only", "combo")) else 0.0
    off = off_dose if arm.startswith(("off_only", "combo")) else 0.0
    release = 0.0 if arm.endswith("_kcat0") else catalytic_release
    return on, off, release


def case_overrides(arm, on_dose, off_dose, catalytic_release):
    """Parameter overrides for one arm at one dose point."""
    on, off, release = arm_doses(arm, on_dose, off_dose, catalytic_release)
    return {**initial_overrides(), "OnDrug_0": on, "OffDrug_0": off,
            "mukcat_TCI": release}


def readouts(g, tspan, catalytic_release, kon_off):
    """One trajectory reduced to the numbers a dose point is summarized by.

    Endpoints and time integrals are taken from the same trajectory, so a flux,
    its substrate and the engagement it produces can never come from different
    runs. Integrals are trapezoidal over the simulation's own time grid.
    """
    s = series(g, catalytic_release, kon_off)
    return {
        "release_integral_M": float(np.trapezoid(s["tri_flux"], tspan)),
        "accessible_end_M": float(s["accessible_gdp_M"][-1]),
        "accessible_auc_Ms": float(np.trapezoid(s["accessible_gdp_M"], tspan)),
        "assoc_integral_M": float(np.trapezoid(s["off_assoc_flux"], tspan)),
        "off_bound_pct_end": float(s["off_bound_pct"][-1]),
        "off_bound_pct_auc": float(np.trapezoid(s["off_bound_pct"], tspan)),
        "signalling_pct_end": float(s["signalling_pct"][-1]),
        "signalling_pct_auc": float(np.trapezoid(s["signalling_pct"], tspan)),
        "ras_gtp_pct_end": float(s["ras_gtp_pct"][-1]),
        "on_trapped_pct_end": float(s["on_trapped_pct"][-1]),
        "wt_on_trapped_pct_end": float(s["wt_on_trapped_pct"][-1]),
    }


def point_summary(veh, on, off, combo, combo_k0, on_k0):
    """One dose point: the six arms reduced to a single row.

    The Bliss excess is computed on the signalling readout, and again on its
    time integral, and again against the zero-catalytic-release controls. The
    engagement gain is the rise in RAS(OFF) engagement from adding the RAS(ON)
    inhibitor, reported both against the RAS(OFF) arm alone and against the
    control at the same dose with catalytic release switched off. The second is
    the one that isolates catalysis from binding.
    """
    e_on, e_off, e_combo, expected, excess = bliss(
        veh["signalling_pct_end"], on["signalling_pct_end"],
        off["signalling_pct_end"], combo["signalling_pct_end"])
    out = {
        "sig_vehicle_pct": veh["signalling_pct_end"],
        "sig_on_only_pct": on["signalling_pct_end"],
        "sig_off_only_pct": off["signalling_pct_end"],
        "sig_combo_pct": combo["signalling_pct_end"],
        "sig_combo_kcat0_pct": combo_k0["signalling_pct_end"],
        "E_ON": e_on, "E_OFF": e_off, "E_combo": e_combo,
        "E_bliss_expected": expected, "dE_bliss_excess": excess,
        # Catalytic release, arm by arm: the time-integrated RAS-GDP it has
        # produced by the end of the simulation.
        "C_TCI_on_only_M": on["release_integral_M"],
        "C_TCI_combo_M": combo["release_integral_M"],
        "C_TCI_combo_kcat0_M": combo_k0["release_integral_M"],
        # Accessible mutant RAS-GDP: GDP-loaded, nothing bound, which is
        # exactly the substrate pattern of the RAS(OFF) binding rule.
        "D_access_end_vehicle_M": veh["accessible_end_M"],
        "D_access_end_on_only_M": on["accessible_end_M"],
        "D_access_end_off_only_M": off["accessible_end_M"],
        "D_access_end_combo_M": combo["accessible_end_M"],
        "D_access_end_combo_kcat0_M": combo_k0["accessible_end_M"],
        "AUC_D_access_vehicle_Ms": veh["accessible_auc_Ms"],
        "AUC_D_access_on_only_Ms": on["accessible_auc_Ms"],
        "AUC_D_access_combo_Ms": combo["accessible_auc_Ms"],
        "AUC_D_access_combo_kcat0_Ms": combo_k0["accessible_auc_Ms"],
        "I_OFF_assoc_off_only_M": off["assoc_integral_M"],
        "I_OFF_assoc_combo_M": combo["assoc_integral_M"],
        "I_OFF_assoc_combo_kcat0_M": combo_k0["assoc_integral_M"],
        "off_bound_pct_off_only": off["off_bound_pct_end"],
        "off_bound_pct_combo": combo["off_bound_pct_end"],
        "off_bound_pct_combo_kcat0": combo_k0["off_bound_pct_end"],
        "AUC_off_bound_off_only": off["off_bound_pct_auc"],
        "AUC_off_bound_combo": combo["off_bound_pct_auc"],
        "AUC_off_bound_combo_kcat0": combo_k0["off_bound_pct_auc"],
        "on_trapped_pct_on_only": on["on_trapped_pct_end"],
        "on_trapped_pct_combo": combo["on_trapped_pct_end"],
        "ras_gtp_pct_vehicle": veh["ras_gtp_pct_end"],
        "ras_gtp_pct_combo": combo["ras_gtp_pct_end"],
    }
    out["engagement_gain_vs_off_only_pp"] = (
        combo["off_bound_pct_end"] - off["off_bound_pct_end"])
    out["engagement_gain_vs_kcat0_pp"] = (
        combo["off_bound_pct_end"] - combo_k0["off_bound_pct_end"])
    out["engagement_gain_vs_off_only_ratio"] = (
        combo["off_bound_pct_end"] / off["off_bound_pct_end"]
        if off["off_bound_pct_end"] > 0 else np.nan)
    out["engagement_gain_vs_kcat0_ratio"] = (
        combo["off_bound_pct_end"] / combo_k0["off_bound_pct_end"]
        if combo_k0["off_bound_pct_end"] > 0 else np.nan)
    out["assoc_gain_vs_kcat0_ratio"] = (
        combo["assoc_integral_M"] / combo_k0["assoc_integral_M"]
        if combo_k0["assoc_integral_M"] > 0 else np.nan)
    out["AUC_engagement_gain_vs_kcat0_ratio"] = (
        combo["off_bound_pct_auc"] / combo_k0["off_bound_pct_auc"]
        if combo_k0["off_bound_pct_auc"] > 0 else np.nan)
    # Bliss assumes the two arms act independently. They act on the same
    # protein pool, so the combination is also scored against the better of the
    # two single arms, a reference model that makes no independence
    # assumption.
    out["hsa_excess"] = e_combo - max(e_on, e_off)
    out["beats_best_single_agent"] = bool(
        combo["signalling_pct_end"] < min(on["signalling_pct_end"],
                                          off["signalling_pct_end"]))
    # The same excess in percentage points of the mutant pool rather than as a
    # fraction, which is the size a reader can weigh against the readout.
    out["sig_bliss_expected_pct"] = veh["signalling_pct_end"] * (1.0 - expected)
    out["sig_excess_pp"] = (veh["signalling_pct_end"] * (1.0 - expected)
                            - combo["signalling_pct_end"])
    auc = bliss(veh["signalling_pct_auc"], on["signalling_pct_auc"],
                off["signalling_pct_auc"], combo["signalling_pct_auc"])
    out.update({"E_ON_AUC": auc[0], "E_OFF_AUC": auc[1], "E_combo_AUC": auc[2],
                "dE_bliss_excess_AUC": auc[4]})
    # RAS(ON) occupancy in the arms that carry no RAS(ON) inhibitor, and on the
    # wild-type pool, which the RAS(ON) inhibitor also binds and the RAS(OFF)
    # probe cannot: a dose matched on the mutant allele is not matched on the
    # wild-type one, and that is a candidate confounder of any allele claim.
    out["on_trapped_pct_off_only"] = off["on_trapped_pct_end"]
    out["wt_on_trapped_pct_on_only"] = on["wt_on_trapped_pct_end"]
    out["wt_on_trapped_pct_combo"] = combo["wt_on_trapped_pct_end"]
    out["signalling_above_vehicle"] = bool(
        max(on["signalling_pct_end"], off["signalling_pct_end"],
            combo["signalling_pct_end"]) > veh["signalling_pct_end"])
    control = bliss(veh["signalling_pct_end"], on_k0["signalling_pct_end"],
                    off["signalling_pct_end"], combo_k0["signalling_pct_end"])
    out.update({"E_ON_kcat0": control[0], "E_combo_kcat0": control[2],
                "dE_bliss_excess_kcat0": control[4]})
    return out


def reference_point(variant="G12D"):
    """The six arms at the reference design point, reduced to one summary row.

    The dose of each arm is its own dissociation constant, which is a design
    point and not a concentration anyone has dosed.
    """
    theta_on, theta_off = THETA_REFERENCE
    catalytic_release = KCAT_TCI[variant]
    on_dose = theta_on * KD2_DECIMAL[variant]
    off_dose = theta_off * KD_OFF_GDP
    model = build(variant, catalytic_release, kon_off=KON_OFF,
                  koff_off=KOFF_OFF)
    tspan, simulator = compile_model(model)

    overrides = [case_overrides(arm, on_dose, off_dose, catalytic_release)
                 for arm in ARMS]
    trajectories = integrate(model, tspan, simulator, overrides, chunk=CHUNK)

    rows = {}
    for arm, g in zip(ARMS, trajectories):
        on, off, release = arm_doses(arm, on_dose, off_dose, catalytic_release)
        rows[arm] = readouts(g, tspan, release, KON_OFF)

    summary = point_summary(rows["vehicle"], rows["on_only"], rows["off_only"],
                            rows["combo"], rows["combo_kcat0"],
                            rows["on_only_kcat0"])
    summary.update({"variant": variant, "theta_ON": theta_on,
                    "theta_OFF": theta_off, "on_dose_M": on_dose,
                    "off_dose_M": off_dose, "kcat_TCI": catalytic_release,
                    # The RAS(ON) arm with catalytic release switched off:
                    # accessible RAS-GDP at the end of the run, the sixth bar
                    # of Fig 7's first panel. The other five arms carry the
                    # same quantity above; only the grid has no use for this
                    # one, so it is added here rather than in point_summary.
                    "D_access_end_on_only_kcat0_M":
                        rows["on_only_kcat0"]["accessible_end_M"]})
    return pd.DataFrame([summary])


def dose_grid(variant, on_doses, off_doses, design, theta_on=None,
              theta_off=None):
    """Every arm over a grid of RAS(ON) and RAS(OFF) doses, for one allele.

    `theta_on` and `theta_off` label the axes in multiples of the dissociation
    constant. They are passed in when the grid was built from them, rather than
    recovered by dividing the dose again, so that the two alleles share one
    axis exactly and a comparison between them can be made point by point.
    """
    catalytic_release = KCAT_TCI[variant]
    model = build(variant, catalytic_release, kon_off=KON_OFF,
                  koff_off=KOFF_OFF)
    tspan, simulator = compile_model(model)
    if theta_on is None:
        theta_on = [float(d) / KD2_DECIMAL[variant] for d in on_doses]
    if theta_off is None:
        theta_off = [float(d) / KD_OFF_GDP for d in off_doses]

    # The vehicle arm is shared by the whole grid, each single-arm dose is run
    # once, and only the combination arms need every pair, so the grid costs
    # 2 n^2 + 3 n + 1 integrations rather than 6 n^2.
    plan = [("vehicle", 0.0, 0.0)]
    plan += [(arm, float(on), 0.0) for on in on_doses
             for arm in ("on_only", "on_only_kcat0")]
    plan += [("off_only", 0.0, float(off)) for off in off_doses]
    plan += [(arm, float(on), float(off)) for on in on_doses
             for off in off_doses for arm in ("combo", "combo_kcat0")]

    overrides = [case_overrides(arm, on, off, catalytic_release)
                 for arm, on, off in plan]
    trajectories = integrate(model, tspan, simulator, overrides, chunk=CHUNK)

    rows = {}
    for (arm, on, off), g in zip(plan, trajectories):
        release = 0.0 if arm.endswith("_kcat0") else catalytic_release
        rows[(arm, on, off)] = readouts(g, tspan, release, KON_OFF)

    escape = escape_rate(model, tspan, simulator)["total"]
    points = []
    for i, on in enumerate(on_doses):
        for j, off in enumerate(off_doses):
            on, off = float(on), float(off)
            summary = point_summary(
                rows[("vehicle", 0.0, 0.0)], rows[("on_only", on, 0.0)],
                rows[("off_only", 0.0, off)], rows[("combo", on, off)],
                rows[("combo_kcat0", on, off)],
                rows[("on_only_kcat0", on, 0.0)])
            summary.update({
                "variant": variant, "design": design, "on_dose_M": on,
                "off_dose_M": off, "theta_ON": float(theta_on[i]),
                "theta_OFF": float(theta_off[j]),
                "kcat_TCI": catalytic_release,
                # Phi: the catalytic-release rate divided by the rate at which
                # mutant RAS-GTP leaves the GTP state unaided, measured on this
                # model at zero dose. Carried for reference only: matched on
                # Phi the two alleles do not collapse and their ordering
                # reverses, so the absolute catalytic-release rate, not this
                # ratio, is the coordinate that carries the result.
                "Phi": catalytic_release / escape,
                "k_escape": escape})
            points.append(summary)
    return pd.DataFrame(points)


def matched_theta_grid():
    """Both alleles over the shared grid of dose multiples."""
    theta = theta_grid()
    frames = []
    for variant in VARIANTS:
        started = time.perf_counter()
        frames.append(dose_grid(variant, theta * KD2_DECIMAL[variant],
                                theta * KD_OFF_GDP, "matched_theta",
                                theta_on=theta, theta_off=theta))
        print(f"  {variant}: {len(frames[-1])} dose points, "
              f"{time.perf_counter() - started:.1f} s")
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "results",
                        help="where to write the tables (default: results/)")
    parser.add_argument("--to-figures", action="store_true",
                        help="also copy them into the data/ folder each figure "
                             "is drawn from")
    parser.add_argument("--skip-grid", action="store_true",
                        help="only the reference design point, not the dose "
                             "grid, which is the slow part")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("reference design point on KRAS G12D, six arms")
    summary = reference_point()
    summary.to_csv(args.out / "reference_point.csv", index=False)
    print(f"  RAS(OFF) engagement {summary['off_bound_pct_off_only'][0]:.3f} % "
          f"alone, {summary['off_bound_pct_combo'][0]:.3f} % combined "
          f"(+{summary['engagement_gain_vs_off_only_pp'][0]:.3f} points)")
    print(f"  Bliss excess {summary['dE_bliss_excess'][0]:+.6f}")
    print(f"  wrote reference_point.csv, "
          f"{time.perf_counter() - started:.1f} s elapsed")

    if not args.skip_grid:
        print("dose grid, both alleles")
        grid = matched_theta_grid()
        grid.to_csv(args.out / "matched_theta_grid.csv", index=False)
        print(f"  wrote matched_theta_grid.csv, {len(grid)} rows")

    if args.to_figures:
        # Both figures read the matched-dose grid, so each keeps its own copy
        # next to the figure that uses it.
        here = HERE / "data"
        here.mkdir(parents=True, exist_ok=True)
        written = []
        for name in ("reference_point.csv", "matched_theta_grid.csv"):
            if (args.out / name).exists():
                shutil.copyfile(args.out / name, here / name)
                written.append(name)
        if (args.out / "matched_theta_grid.csv").exists():
            other = HERE.parent / "fig08_variants" / "data"
            other.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.out / "matched_theta_grid.csv",
                            other / "matched_theta_grid.csv")
            written.append("matched_theta_grid.csv (also for Fig 8)")
        print("  copied into data/: " + ", ".join(written))

    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
