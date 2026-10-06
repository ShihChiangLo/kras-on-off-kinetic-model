"""Simulations behind Fig 9: dosing the RAS(ON) and the covalent RAS(OFF) arm
together on KRAS G12C.

    python3 -m figures.fig09_g12c_dosing.simulate
    python3 figures/fig09_g12c_dosing/simulate.py --out some/other/folder

Writes
    baseline.csv            the drug-free GTP-loaded fraction of each pool, per
                            intrinsic-hydrolysis scenario, with the size of the
                            generated network
    dose_grid.csv           both axes over RAS(ON) dose by covalent dose
    interval_edges.csv      the two dose-interval edges at every covalent dose
    on_dose_interval.csv    the recommended RAS(ON) dose interval and the three
                            tests that would withdraw the recommendation
    capture_timecourse.csv  the curves the figure's time-course panels are
                            drawn from

Both inhibitors are present in every case computed here, which is what makes
the two axes of the figure what they are. Benefit is covalent capture, read at
6 h; cost is wild-type RAS(ON) occupancy, read at 24 h. Why benefit cannot be
an occupancy is explained at `readouts` below.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from two_arm import parameters as P
from two_arm.model import (CAPTURED, MUTANT_TOTAL, WILDTYPE_TOTAL, build,
                           check_catalysis, covalent_constants)
from two_arm.simulate import (at_time, bisect_log, compile_model, integrate,
                              initial_overrides, structure, time_at_level)

TABLES = ("baseline.csv", "dose_grid.csv", "interval_edges.csv",
          "on_dose_interval.csv", "capture_timecourse.csv")
#: The tables that ship in `data/`. The first four are the ones `make_figure.py`
#: reads; `interval_edges.csv` is not plotted and ships because S6 Text quotes
#: interval edges that appear in no other table.
FIGURE_TABLES = ("baseline.csv", "dose_grid.csv", "on_dose_interval.csv",
                 "capture_timecourse.csv", "interval_edges.csv")

#: The two declared intrinsic-hydrolysis scenarios for KRAS G12C, in the order
#: every table lists them.
SCENARIOS = ("cuevas_dmso", "cuevas_cypa")

#: Covalent doses, as a multiple of the 2e-7 M mutant RAS pool: none, half of
#: it, exactly it, five times it and fifty times it. Capture is stoichiometric,
#: so crossing the pool size is what this axis has to straddle.
COVALENT_DOSES = (0.0, 1.0e-7, 2.0e-7, 1.0e-6, 1.0e-5)
#: The covalent dose the recommendation is made at, five times the pool.
RECOMMENDED_COVALENT_DOSE = 1.0e-6
#: Fifty times the pool. The dose interval here is compared with the one at the
#: recommended dose, which is how the saturation test below is read.
SATURATING_COVALENT_DOSE = 1.0e-5

#: RAS(ON) doses: zero, then a dense log grid from 1 nM to 10 uM. Dense because
#: the interval edges are compared against this grid; the edges themselves are
#: solved for, so they are not a property of the grid.
ON_DOSES = (0.0,) + tuple(np.logspace(-9, -5, 25))

#: The two wild-type RAS(ON) occupancies that bound the dose interval, in % of
#: the wild-type pool, taken from `two_arm.parameters` so that this script and
#: `make_figure.py` cannot disagree about where the band is. They are limits
#: declared for this analysis, not clinical thresholds, and nothing in the
#: figure should read as though they were.
OCCUPANCY_LIMITS = P.OCCUPANCY_LIMITS

#: Bracket, tolerance and iteration cap for the RAS(ON) dose that gives a
#: declared wild-type occupancy. The bracket spans several decades either side
#: of the answer; the tolerance is on the ratio of its two ends, because the
#: bisection is in log dose.
INTERVAL_BRACKET = (1.0e-11, 1.0e-3)
INTERVAL_TOL_REL = 1.0e-6
INTERVAL_MAX_ITER = 60

# The three tests that would withdraw the recommendation, fixed before the
# simulations were run. Each names a condition under which the corresponding
# claim may not be made.
#   gain floor      the lower edge of the interval has to collect at least this
#                   fraction of the capture that a saturating RAS(ON) dose
#                   would add, or the interval is not where the benefit is
#   saturation tol  the interval has to stop moving between the recommended and
#                   the saturating covalent dose, to this relative tolerance,
#                   or the recommendation rests on the covalent dose rather
#                   than on capture being complete
#   lift ceiling    wild-type occupancy anywhere inside the interval has to
#                   stay below this multiple of its value without the covalent
#                   inhibitor, or capture of the mutant pool is being paid for
#                   with wild-type engagement
GAIN_FLOOR = 0.5
SATURATION_TOL_REL = 0.01
LIFT_CEILING_X = 3.0

#: Time-course doses: one covalent dose, a fan of RAS(ON) doses, and one case
#: with catalytic release switched off at the middle RAS(ON) dose.
TIMECOURSE_COVALENT_DOSE = 1.0e-6
TIMECOURSE_ON_DOSES = (0.0, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 1.0e-5)
TIMECOURSE_NO_RELEASE_ON_DOSE = 3.0e-7

#: The two catalytic-release settings the tables label cases by. Under
#: `shield` the tri-complex still withholds RAS-GTP from the effector but no
#: longer releases it as RAS-GDP, so the covalent arm loses its substrate
#: supply. It is the same model with that one rate set to exactly zero, not a
#: second compound.
RELEASE = {"full": P.KCAT_TCI[P.COVALENT_VARIANT], "shield": 0.0}

#: Both RAS pools are at this concentration, so it is both percentage
#: denominator and the stoichiometric reference for the covalent dose.
POOL_M = P.CONDITION[0]

_MODELS = {}


def model_for(scenario):
    """The covalent model for one intrinsic-hydrolysis scenario, with its grid.

    The association rate and the inactivation rate arrive together from one
    call to `covalent_constants` and are never overridden at simulation time:
    they are two faces of the published second-order efficiency, and moving one
    without the other would break that identity. Building takes about half a
    second, so the model is cached per scenario rather than rebuilt per case.
    """
    if scenario not in _MODELS:
        constants = covalent_constants(P.KINACT_OVER_KI, P.K_I, P.KOFF_COVALENT)
        model = build(P.COVALENT_VARIANT, RELEASE["full"],
                      kon_off=constants["kon"], koff_off=constants["koff"],
                      kinact=constants["kinact"], covalent=True,
                      hydrolysis=P.G12C_HYDROLYSIS[scenario])
        tspan, simulator = compile_model(model, P.T_END_COVALENT,
                                         P.N_POINTS_COVALENT)
        _MODELS[scenario] = (model, tspan, simulator)
    return _MODELS[scenario]


def readouts(scenario, cases, curves=False):
    """Both axes for each case, out of one integration per case.

    A case is {"on_M", "off_M", "cond"}. Benefit and cost come from the same
    trajectory, so they cannot drift apart, and the conservation and
    monotonicity columns are measured on that same trajectory rather than on a
    separate check run.

    `benefit_capture_6h_pct` is covalent capture: mutant RAS the RAS(OFF)
    inhibitor has reacted with, as a percentage of the mutant pool. It is
    deliberately not an occupancy. The covalent inhibitor consumes its target,
    so mutant RAS(ON) occupancy falls towards zero at 24 h with 1 uM of
    covalent inhibitor present -- not because the RAS(ON) arm stopped working
    but because its denominator has become an adduct. A capture level at a
    fixed time cannot be emptied that way. That occupancy is carried through
    all the same, in the `mu_on_occupancy_*` columns, because its collapse is
    itself a result.

    `cost_wt_occupancy_24h_pct` is wild-type RAS carrying the RAS(ON)
    inhibitor, as a percentage of the wild-type pool: the competing target the
    tri-complex also binds and the covalent arm cannot touch.
    """
    model, tspan, simulator = model_for(scenario)
    hydrolysis = P.G12C_HYDROLYSIS[scenario]
    overrides = []
    for case in cases:
        release = RELEASE[case["cond"]]
        check_catalysis([release], P.COVALENT_VARIANT, hydrolysis)
        overrides.append({**initial_overrides(),
                          "OnDrug_0": float(case["on_M"]),
                          "OffDrug_0": float(case["off_M"]),
                          "mukcat_TCI": release})
    rows, frames = [], []
    for case, g in zip(cases, integrate(model, tspan, simulator, overrides)):
        mutant = g[MUTANT_TOTAL[0]] + g[MUTANT_TOTAL[1]]
        wildtype = g[WILDTYPE_TOTAL[0]] + g[WILDTYPE_TOTAL[1]]
        captured = 100.0 * g[CAPTURED] / mutant
        mutant_occupancy = 100.0 * g["obsmuRAS_on_drug_any"] / mutant
        wildtype_occupancy = 100.0 * g["obsRAS_on_drug_any"] / wildtype
        mutant_gtp = 100.0 * g["obsmuRAS_T_total"] / mutant
        wildtype_gtp = 100.0 * g["obsRAS_T_total"] / wildtype
        off_dose, cond = float(case["off_M"]), case["cond"]
        # The covalent inhibitor is conserved: free plus bound, of which the
        # adduct is part, has to stay at the dose for the whole trajectory.
        covalent_balance = (g["obsOffDrug_free"] + g["obsOffDrug_bound_any"]
                            - off_dose)
        rows.append({
            "scenario": scenario, "variant": P.COVALENT_VARIANT,
            "K_I_M": P.K_I, "mukhyd": hydrolysis, "cond": cond,
            "kcat_TCI": RELEASE[cond],
            "on_M": float(case["on_M"]), "off_M": off_dose,
            "on_over_KD2": (float(case["on_M"])
                            / P.KD2_DECIMAL[P.COVALENT_VARIANT]),
            "off_over_RAS": off_dose / POOL_M,
            "t_end_s": float(tspan[-1]),
            "t_benefit_s": P.T_BENEFIT, "t_cost_s": P.T_COST,
            "benefit_capture_6h_pct": at_time(tspan, captured, P.T_BENEFIT),
            "cost_wt_occupancy_24h_pct": at_time(tspan, wildtype_occupancy,
                                                 P.T_COST),
            "mu_on_occupancy_24h_pct": at_time(tspan, mutant_occupancy,
                                               P.T_COST),
            "mu_on_occupancy_6h_pct": at_time(tspan, mutant_occupancy,
                                              P.T_BENEFIT),
            "cost_wt_occupancy_6h_pct": at_time(tspan, wildtype_occupancy,
                                                P.T_BENEFIT),
            "capture_24h_pct": float(captured[-1]),
            "mu_gtp_24h_pct": float(mutant_gtp[-1]),
            "wt_gtp_24h_pct": float(wildtype_gtp[-1]),
            "t90_s": time_at_level(tspan, captured, P.CAPTURE_CRITERION),
            "t90_h": time_at_level(tspan, captured,
                                   P.CAPTURE_CRITERION) / 3600.0,
            "cap_monotone_max_drop_pp": float(
                max(0.0, np.max(-np.diff(captured)))),
            "off_conservation_rel": float(
                np.max(np.abs(covalent_balance)) / off_dose if off_dose > 0
                else np.max(np.abs(covalent_balance))),
            "wt_pool_conservation_rel": float(
                np.max(np.abs(wildtype - POOL_M)) / POOL_M),
            "mu_pool_conservation_rel": float(
                np.max(np.abs(mutant - POOL_M)) / POOL_M),
        })
        # The integration either stayed finite or every readout above is
        # meaningless, so this is checked rather than reported.
        if not all(np.all(np.isfinite(v)) for v in g.values()):
            raise RuntimeError(
                f"non-finite values at [ON] = {float(case['on_M']):g} M, "
                f"[OFF] = {off_dose:g} M, {cond}")
        if curves:
            frames.append(pd.DataFrame({
                "scenario": scenario, "cond": cond,
                "on_M": float(case["on_M"]), "off_M": off_dose,
                "t_s": tspan, "t_h": tspan / 3600.0,
                "captured_pct": captured, "mu_gtp_pct": mutant_gtp,
                "mu_on_occupancy_pct": mutant_occupancy,
                "wt_on_occupancy_pct": wildtype_occupancy}))
    table = pd.DataFrame(rows)
    if curves:
        return table, pd.concat(frames, ignore_index=True)
    return table


def wildtype_occupancy_at(scenario, covalent_dose, on_dose):
    """Wild-type RAS(ON) occupancy at 24 h for one dose pair, %."""
    case = [{"on_M": float(on_dose), "off_M": float(covalent_dose),
             "cond": "full"}]
    return float(readouts(scenario, case)["cost_wt_occupancy_24h_pct"].iloc[0])


def on_dose_at_occupancy(scenario, covalent_dose, limit):
    """RAS(ON) dose whose 24 h wild-type occupancy equals `limit` %.

    Wild-type occupancy rises monotonically with RAS(ON) dose -- the RAS(ON)
    inhibitor is the only drug that binds that pool, and nothing consumes it
    -- so bisecting in log dose is well posed. Returns (dose, iterations), the
    dose being NaN if the bracket does not contain the limit, which is an
    answer about the dose range rather than a failure.
    """
    lo, hi = INTERVAL_BRACKET

    def excess(dose):
        return wildtype_occupancy_at(scenario, covalent_dose, dose) - limit

    dose, report = bisect_log(excess, lo, hi, excess(lo), excess(hi),
                              INTERVAL_TOL_REL, INTERVAL_MAX_ITER)
    return (float("nan") if dose is None else float(dose)), report["n_iter"]


def occupancy_without_covalent(grid, scenario, doses):
    """Wild-type occupancy without the covalent inhibitor, at `doses`.

    The reference the wild-type lift is measured against. Interpolated in log
    dose on the grid already computed, so it costs no extra integration; the
    grid's zero dose is moved far below the bracket rather than dropped, so the
    interpolation keeps a left-hand end.
    """
    reference = grid[(grid.scenario == scenario)
                     & (grid.off_M == 0.0)].sort_values("on_M")
    return np.interp(np.log10(doses),
                     np.log10(reference.on_M.replace(0, 1e-30)),
                     reference.cost_wt_occupancy_24h_pct)


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
def baseline_table():
    """The drug-free GTP-loaded fraction of each pool, per scenario."""
    rows = []
    for scenario in SCENARIOS:
        model, _, _ = model_for(scenario)
        network = structure(model)
        drug_free = readouts(scenario, [{"on_M": 0.0, "off_M": 0.0,
                                         "cond": "full"}])
        for pool, column in (("mutant G12C", "mu_gtp_24h_pct"),
                             ("wild-type", "wt_gtp_24h_pct")):
            gtp = float(drug_free[column].iloc[0])
            rows.append({"scenario": scenario, "K_I_M": P.K_I,
                         "mukhyd": P.G12C_HYDROLYSIS[scenario], "pool": pool,
                         "gtp_pct": gtp, "gdp_pct": 100.0 - gtp,
                         "pool_M": POOL_M, "n_rules": network["n_rules"],
                         "n_species": network["n_species"]})
    return pd.DataFrame(rows)


def dose_grid_table():
    """Both axes over the whole RAS(ON) dose by covalent dose grid."""
    frames = []
    for scenario in SCENARIOS:
        for covalent_dose in COVALENT_DOSES:
            frames.append(readouts(scenario, [
                {"on_M": on, "off_M": covalent_dose, "cond": "full"}
                for on in ON_DOSES]))
        print(f"    {scenario}: {len(COVALENT_DOSES)} x {len(ON_DOSES)} cases",
              flush=True)
    return pd.concat(frames, ignore_index=True)


def interval_edges_table(grid):
    """Both dose-interval edges at every covalent dose, and what they collect.

    One row per (scenario, covalent dose, occupancy limit). The capture at an
    edge needs its own integration, because an edge is not a grid point.
    """
    rows = []
    for scenario in SCENARIOS:
        for covalent_dose in COVALENT_DOSES:
            here = grid[(grid.scenario == scenario)
                        & (grid.off_M == covalent_dose)].sort_values("on_M")
            without_on_drug = float(here.loc[here.on_M == 0.0,
                                             "benefit_capture_6h_pct"].iloc[0])
            best_on_grid = float(here.benefit_capture_6h_pct.max())
            found = []
            for limit in OCCUPANCY_LIMITS:
                edge, iterations = on_dose_at_occupancy(scenario,
                                                        covalent_dose, limit)
                found.append(edge)
                if edge == edge:
                    at_edge = readouts(scenario, [{"on_M": edge,
                                                   "off_M": covalent_dose,
                                                   "cond": "full"}]).iloc[0]
                    capture = float(at_edge.benefit_capture_6h_pct)
                    occupancy = float(at_edge.cost_wt_occupancy_24h_pct)
                    lift = occupancy / float(
                        occupancy_without_covalent(grid, scenario, edge))
                else:
                    capture = occupancy = lift = float("nan")
                gain = capture - without_on_drug
                headroom = best_on_grid - without_on_drug
                rows.append({
                    "scenario": scenario, "off_M": covalent_dose,
                    "off_over_RAS": covalent_dose / POOL_M,
                    "wt_threshold_pct": limit, "on_edge_M": edge,
                    "on_edge_over_KD2": (edge
                                         / P.KD2_DECIMAL[P.COVALENT_VARIANT]),
                    "edge_found": bool(edge == edge),
                    "wt_occupancy_at_edge_pct": occupancy,
                    "benefit_at_edge_pct": capture,
                    "benefit_at_on0_pct": without_on_drug,
                    "benefit_max_on_grid_pct": best_on_grid,
                    "gain_at_edge_pct": gain, "gain_max_pct": headroom,
                    "gain_fraction": (gain / headroom if headroom > 0
                                      else float("nan")),
                    "benefit_ratio_absolute": (capture / best_on_grid
                                               if best_on_grid
                                               else float("nan")),
                    "wt_lift_vs_off0_x": lift,
                    "bisection_iters": iterations,
                    "t_benefit_s": P.T_BENEFIT, "t_cost_s": P.T_COST})
            print(f"    {scenario}: covalent {covalent_dose:9.3g} M, edges "
                  f"{found[0]:.4g} / {found[1]:.4g} M", flush=True)
    return pd.DataFrame(rows)


def on_dose_interval_table(grid, edges):
    """The recommended dose interval and the three tests, one row per scenario.
    """
    rows = []
    for scenario in SCENARIOS:
        recommended = edges[(edges.scenario == scenario)
                            & (edges.off_M == RECOMMENDED_COVALENT_DOSE)]
        lo = recommended[
            recommended.wt_threshold_pct == OCCUPANCY_LIMITS[0]].iloc[0]
        hi = recommended[
            recommended.wt_threshold_pct == OCCUPANCY_LIMITS[1]].iloc[0]

        gain_fails = bool(float(lo.gain_fraction) < GAIN_FLOOR)

        saturating = edges[(edges.scenario == scenario)
                           & (edges.off_M == SATURATING_COVALENT_DOSE)]
        drift = []
        for limit, edge in zip(OCCUPANCY_LIMITS, (lo, hi)):
            moved = saturating[
                saturating.wt_threshold_pct == limit].on_edge_M.iloc[0]
            drift.append(abs(moved - float(edge.on_edge_M))
                         / float(edge.on_edge_M))
        saturation_fails = bool(max(drift) > SATURATION_TOL_REL)

        # Every grid point inside the interval, plus the two edges themselves,
        # because the worst wild-type lift need not fall on a grid point.
        inside = grid[(grid.scenario == scenario)
                      & (grid.off_M == RECOMMENDED_COVALENT_DOSE)
                      & (grid.on_M >= float(lo.on_edge_M))
                      & (grid.on_M <= float(hi.on_edge_M))].sort_values("on_M")
        lifts = (inside.cost_wt_occupancy_24h_pct.to_numpy()
                 / occupancy_without_covalent(grid, scenario, inside.on_M))
        worst_lift = max(float(np.max(lifts)) if len(lifts) else 0.0,
                         float(lo.wt_lift_vs_off0_x),
                         float(hi.wt_lift_vs_off0_x))
        lift_fails = bool(worst_lift > LIFT_CEILING_X)

        rows.append({
            "scenario": scenario,
            "recommended_off_M": RECOMMENDED_COVALENT_DOSE,
            "window_lo_M": float(lo.on_edge_M),
            "window_hi_M": float(hi.on_edge_M),
            "window_lo_over_KD2": float(lo.on_edge_over_KD2),
            "window_hi_over_KD2": float(hi.on_edge_over_KD2),
            "window_width_x": float(hi.on_edge_M) / float(lo.on_edge_M),
            "benefit_at_on0_pct": float(lo.benefit_at_on0_pct),
            "benefit_at_lo_pct": float(lo.benefit_at_edge_pct),
            "benefit_at_hi_pct": float(hi.benefit_at_edge_pct),
            "benefit_max_pct": float(lo.benefit_max_on_grid_pct),
            "gain_fraction_at_lo": float(lo.gain_fraction),
            "gain_fraction_at_hi": float(hi.gain_fraction),
            "benefit_ratio_absolute_at_lo": float(lo.benefit_ratio_absolute),
            "window_drift_lo_rel": drift[0],
            "window_drift_hi_rel": drift[1],
            "worst_wt_lift_in_window_x": worst_lift,
            "fires_gain_below_50pct": gain_fails,
            "fires_window_not_saturated": saturation_fails,
            "fires_lift_above_3x": lift_fails,
            "any_falsifier_fires": bool(gain_fails or saturation_fails
                                        or lift_fails),
            "gain_floor": GAIN_FLOOR,
            "saturation_tol_rel": SATURATION_TOL_REL,
            "lift_ceiling_x": LIFT_CEILING_X})
    return pd.DataFrame(rows)


def capture_timecourse_table():
    """The curves the figure's time-course panels are drawn from.

    Nothing is read off these; they come from the same machinery as the tables
    above, so the panels cannot disagree with the numbers.
    """
    frames = []
    for scenario in SCENARIOS:
        cases = [{"on_M": float(x), "off_M": TIMECOURSE_COVALENT_DOSE,
                  "cond": "full"} for x in TIMECOURSE_ON_DOSES]
        cases.append({"on_M": TIMECOURSE_NO_RELEASE_ON_DOSE,
                      "off_M": TIMECOURSE_COVALENT_DOSE, "cond": "shield"})
        frames.append(readouts(scenario, cases, curves=True)[1])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\nWrites")[0],
        epilog="Writes " + ", ".join(TABLES) + ".",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help="folder the tables are written to")
    ap.add_argument("--to-figures", action="store_true",
                    help="also copy the shipped tables into the data/ folder "
                         "beside this script")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = mark = time.time()

    def write(table, name):
        nonlocal mark
        table.to_csv(args.out / name, index=False)
        print(f"  {len(table)} rows -> {name} ({time.time() - mark:.0f} s)",
              flush=True)
        mark = time.time()

    def read_back(name):
        """Read a table back from the file just written.

        Each derived table is built from what the earlier ones publish rather
        than from the values still in memory, so that every number in it can be
        recovered from the files in this folder. The two are not always the
        same: a decimal that is written and parsed again is whatever value the
        parser returns for that text, which need not be the bit pattern it was
        written from. `float_precision` is stated rather than left to the
        default so that this does not depend on the pandas version.
        """
        return pd.read_csv(args.out / name, float_precision="high")

    print("drug-free baselines", flush=True)
    write(baseline_table(), "baseline.csv")

    print("dose grid", flush=True)
    write(dose_grid_table(), "dose_grid.csv")
    grid = read_back("dose_grid.csv")

    print("RAS(ON) dose interval", flush=True)
    write(interval_edges_table(grid), "interval_edges.csv")
    write(on_dose_interval_table(grid, read_back("interval_edges.csv")),
          "on_dose_interval.csv")

    print("capture time courses", flush=True)
    write(capture_timecourse_table(), "capture_timecourse.csv")

    if args.to_figures:
        target = Path(__file__).resolve().parent / "data"
        for name in FIGURE_TABLES:
            shutil.copyfile(args.out / name, target / name)
        print(f"  copied {len(FIGURE_TABLES)} tables into {target}",
              flush=True)
    print(f"done in {time.time() - started:.0f} s; tables in {args.out}")


if __name__ == "__main__":
    main()
