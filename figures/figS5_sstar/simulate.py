"""Simulations behind the S5 Text figure: the dose-help boundary S*.

    python3 -m figures.figS5_sstar.simulate
    python3 figures/figS5_sstar/simulate.py --out some/other/folder
    python3 figures/figS5_sstar/simulate.py --table sstar_g12d.csv

Writes one table per panel of the figure:

    sstar_g12v_alpha_1_inf.csv          KRAS G12V, the two end points of the
                                        effector-attenuation axis
    sstar_g12v_alpha_3_10.csv           KRAS G12V, the two intermediate ones
    sstar_g12d.csv                      KRAS G12D, both end points
    sstar_g12v_late_readout.csv         the same at a hundredfold later readout
    sstar_g12v_assay_intrinsic.csv      with the assay intrinsic hydrolysis rate
    sstar_g12v_rescue.csv               with the effector rescue route open
    sstar_g12v_common_probe_alpha1.csv  the two end points re-bracketed on the
    sstar_g12v_common_probe_alphainf.csv  same probe as the intermediate ones

WHAT IS BEING SOLVED

Two nested solves, both bisections in log space.

The inner one finds the catalytic release at which the combination begins to
beat the RAS(OFF) arm alone: the engagement gain, the percentage points of the
mutant RAS pool that the combination puts the RAS(OFF) inhibitor on over and
above what that inhibitor achieves by itself, is negative at low catalytic
release and positive at high, and the threshold is where it is zero. Divided by
the measured catalytic release of that allele, the threshold is the required
fold change R: how far short of feeding the measured compound is.

The outer one finds the state selectivity at which raising both doses a
thousandfold stops lowering that requirement and starts raising it. Its
objective is the difference D(S) = R at the high dose minus R at the reference
dose. D < 0 means raising the dose still helps; D > 0 means it hurts. The state
selectivity where D crosses zero is the dose-help boundary S*.

HOW THE ANSWER IS REPORTED

The bracket is what may be quoted and the root inside it is conditional -- on
this readout time, this rescue state and this intrinsic hydrolysis scenario.
Both are in every table: `bracket_lo` and `bracket_hi` are the two adjacent
probe points that straddle the sign change, `D_at_bracket_lo` and
`D_at_bracket_hi` are D there, and `S_star` is the root between them. The
progress lines below print the bracket, never the root alone.

A probe with no sign change at all is a result, not a failure: the status is
`NO_CROSSING` and the verdict says that raising the dose still helps across the
whole probe. More than one sign change is not a boundary, and is reported as
`MULTIPLE_CROSSINGS` with the root left empty rather than resolved.

COST AND RESUMING

One boundary costs one D(S) per probe point plus one per bisection step, and one
D(S) is two of the inner solves, each of which integrates the network about
sixty times. The eight tables together are roughly 280 D(S) values and about 55
minutes of wall clock. Every D(S) is appended to `boundary_cache.jsonl` in the
output folder as soon as it is finished and is read back from there on the next
run, so an interrupted run resumes where it stopped and the tables can be
produced one at a time with `--table` without recomputing what they share.
Delete that file to force a clean recomputation.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from two_arm import parameters as P
from two_arm.model import (ATTENUATION_PARAMETER, SELECTIVITY_PARAMETER, build,
                           catalysis_bounds, check_catalysis)
from two_arm.simulate import (bisect_log, compile_model, dose_scales,
                              initial_overrides, integrate)

#: Mutant RAS carrying the RAS(OFF) inhibitor, as a percentage of the mutant
#: pool, read at the end of the time grid. The engagement gain is the difference
#: of this quantity between the combination and the RAS(OFF) arm alone.
ENGAGEMENT_OBSERVABLE = "obsmuRAS_off_any"

#: The two intrinsic-hydrolysis scenarios this figure sweeps for KRAS G12V.
#: `model` is the rate the switch model carries for that allele; `assay` is the
#: assay-scenario rate, carried as a declared scenario rather than replacing
#: it. `None` means the model's own value, and is never spelled as a
#: number here, so the two cannot drift apart.
INTRINSIC = {"model": None, "assay": P.G12V_HYDROLYSIS_ASSAY}

#: The structural scenario every case here is computed under: a RAS carrying
#: the RAS(OFF) inhibitor does not hydrolyse its GTP intrinsically. It is
#: stated in the figure caption rather than carried in a column.
DRUG_BOUND_HYDROLYSIS = False

#: The spread of state selectivity measured for one compound of this class
#: across seven KRAS alleles is `P.MEASURED_SELECTIVITY_SPREAD`, the one copy
#: of it in this work. Whether the boundary falls inside that spread is the
#: question the figure asks, so it is carried in a column and in the verdict.

#: The probes, one per panel. A probe is quoted as a bracket, so its points are
#: round numbers rather than a generated grid.
PROBE_TWO_POINT = (16.0, 182.0)
PROBE_WIDE = (182.0, 1.0e3, 1.0e4, 1.0e5)
PROBE_COMMON = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1.0e3)
PROBE_G12D = (7.7, 16.0, 64.0, 182.0, 512.0, 1.0e3)
PROBE_LATE = (1.0, 3.1623, 7.7, 16.0, 182.0)
PROBE_RESCUE = (16.0, 64.0, 182.0, 512.0, 1.0e3)
PROBE_ASSAY = (1.0, 3.1623, 7.7, 16.0, 64.0, 182.0, 1.0e3)

#: A hundredfold later readout than the horizon every other case uses. Eleven
#: days is beyond what a model without RAS turnover can describe, so results
#: there are read at that time and never called a steady state.
T_END_LATE = 1.0e6

#: The string the two re-bracketed tables carry, so that a reader of one row
#: knows which probe its bracket came from.
COMMON_PROBE_SOURCE = "seven-point common probe [16 .. 1000]"

_CACHE_FILE = "boundary_cache.jsonl"


def _case(variant, alpha, probe, t_end=P.T_END, rescue=False,
          intrinsic="model"):
    return {"variant": variant, "alpha": alpha, "probe": tuple(probe),
            "t_end": t_end, "rescue": rescue, "intrinsic": intrinsic}


#: The eight tables, each a list of cases in the order its rows appear. The
#: order of the tables themselves is cheapest first.
TABLES = {
    "sstar_g12v_late_readout.csv": [
        _case("G12V", 1.0, PROBE_LATE, T_END_LATE),
        _case("G12V", math.inf, PROBE_LATE, T_END_LATE)],
    "sstar_g12v_common_probe_alphainf.csv": [
        _case("G12V", math.inf, PROBE_COMMON)],
    "sstar_g12v_common_probe_alpha1.csv": [
        _case("G12V", 1.0, PROBE_COMMON)],
    "sstar_g12v_alpha_1_inf.csv": [
        _case("G12V", math.inf, PROBE_TWO_POINT),
        _case("G12V", 1.0, PROBE_WIDE)],
    "sstar_g12v_alpha_3_10.csv": [
        _case("G12V", 3.0, PROBE_COMMON),
        _case("G12V", 10.0, PROBE_COMMON)],
    "sstar_g12d.csv": [
        _case("G12D", 1.0, PROBE_G12D),
        _case("G12D", math.inf, PROBE_G12D)],
    "sstar_g12v_assay_intrinsic.csv": [
        _case("G12V", 1.0, PROBE_ASSAY, intrinsic="assay"),
        _case("G12V", math.inf, PROBE_ASSAY, intrinsic="assay")],
    "sstar_g12v_rescue.csv": [
        _case("G12V", 1.0, PROBE_RESCUE, rescue=True),
        _case("G12V", 3.0, PROBE_RESCUE, rescue=True),
        _case("G12V", 10.0, PROBE_RESCUE, rescue=True)],
}

#: Tables that carry the extra `bracket_source` column.
BRACKET_SOURCE_TABLES = ("sstar_g12v_common_probe_alpha1.csv",
                         "sstar_g12v_common_probe_alphainf.csv")

#: The tables the figure script reads: all of them.
FIGURE_TABLES = tuple(TABLES)


# ---------------------------------------------------------------------------
# labels and cache keys
# ---------------------------------------------------------------------------
def _number(value):
    """A design-axis value as a short, stable string.

    Used for labels and for cache keys, so that the same design point written
    two ways is the same point. Six significant figures is more than any axis
    of this figure is specified to.
    """
    if not math.isfinite(value):
        return "inf"
    if float(value).is_integer() and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.6g}"


def alpha_label(alpha):
    return f"alpha={_number(alpha)}"


def _key(case, selectivity):
    """The cache key of one D(S) value: every axis this figure varies.

    All six segments are regenerated from the numbers. The readout time, the
    rescue state and the intrinsic scenario are in the key because they change
    the answer; leaving any of them out would let one case silently return
    another's value.
    """
    return "|".join([case["variant"], alpha_label(case["alpha"]),
                     f"S={_number(selectivity)}",
                     f"t{float(case['t_end']):.0e}",
                     f"rescue{int(bool(case['rescue']))}", case["intrinsic"]])


# ---------------------------------------------------------------------------
# the inner solve: the catalytic release at which the combination starts to win
# ---------------------------------------------------------------------------
_MODELS = {}


def model_for(variant, selectivity, attenuation, rescue, hydrolysis, t_end):
    """The model and its simulator for one design point.

    A finite state selectivity or effector attenuation is built in at one -- the
    value that adds the rule but gives it no effect -- and the actual value is
    applied to its parameter at simulation time. Building is the expensive part,
    so one built model serves a whole sweep over either axis. Infinite values
    are the absence of the corresponding rule and so have to be built in.
    """
    built_selectivity = 1.0 if math.isfinite(selectivity) else selectivity
    built_attenuation = 1.0 if math.isfinite(attenuation) else attenuation
    key = (variant, built_selectivity, built_attenuation, rescue, hydrolysis,
           t_end)
    if key not in _MODELS:
        release, _ = catalysis_bounds(variant, hydrolysis)
        model = build(variant, release, kon_off=P.KON_OFF,
                      koff_off=P.KOFF_OFF, hydrolysis=hydrolysis,
                      state_selectivity=built_selectivity,
                      effector_attenuation=built_attenuation,
                      drug_bound_hydrolysis=DRUG_BOUND_HYDROLYSIS,
                      effector_rescue=rescue)
        tspan, simulator = compile_model(model, t_end, P.N_POINTS)
        _MODELS[key] = (model, tspan, simulator)
    return _MODELS[key]


def _axis_overrides(model, selectivity, attenuation):
    """The two design axes, as parameter values on the built model."""
    overrides = {}
    if math.isfinite(selectivity):
        overrides[SELECTIVITY_PARAMETER] = float(selectivity) * P.KOFF_OFF
    if math.isfinite(attenuation):
        overrides[ATTENUATION_PARAMETER] = (
            float(attenuation) * float(model.parameters["mukd_Eff"].value))
    return overrides


def _engagement(trajectory):
    """Mutant RAS carrying the RAS(OFF) inhibitor at the end of the grid, %."""
    pool = (trajectory["obsmuRAS_total"] + trajectory["obsmuRASn_total"])
    return float((100.0 * trajectory[ENGAGEMENT_OBSERVABLE] / pool)[-1])


def required_fold_change(case, selectivity, theta):
    """The required fold change R at one design point and one dose pair.

    `theta` is (RAS(ON) dose, RAS(OFF) dose), each as a multiple of its own
    dissociation constant. Both constants are read back off the built model
    rather than taken from a literal, so a dose cannot drift from the affinity
    the model was built with.

    Returns (R, report). The engagement gain is walked on a geometric grid
    spanning the whole window the model allows a catalytic release in, which
    reports how many sign changes the gain has on that grid; the bisection is
    then seeded from the two ends of the grid. R is NaN when the gain has the
    same sign at both ends of that window: the combination either never beats
    the RAS(OFF) arm alone there or always does, and either way no threshold
    exists inside it.
    """
    variant, rescue = case["variant"], case["rescue"]
    hydrolysis = INTRINSIC[case["intrinsic"]]
    model, tspan, simulator = model_for(variant, selectivity, case["alpha"],
                                        rescue, hydrolysis, case["t_end"])
    low, high = catalysis_bounds(variant, hydrolysis)
    on_scale, off_scale = dose_scales(model)
    on_dose, off_dose = theta[0] * on_scale, theta[1] * off_scale
    base = {**initial_overrides(),
            **_axis_overrides(model, selectivity, case["alpha"])}

    def combination(releases):
        """The engagement of the combination at each catalytic release."""
        check_catalysis(releases, variant, hydrolysis)
        rows = [{**base, "OnDrug_0": on_dose, "OffDrug_0": off_dose,
                 "mukcat_TCI": float(r)} for r in releases]
        return [_engagement(g)
                for g in integrate(model, tspan, simulator, rows, chunk=64)]

    # The RAS(OFF) arm alone has no catalytic release to vary, so it is one
    # integration for the whole solve rather than one per catalytic release.
    alone = _engagement(integrate(model, tspan, simulator, [
        {**base, "OnDrug_0": 0.0, "OffDrug_0": off_dose,
         "mukcat_TCI": 0.0}])[0])

    grid = np.geomspace(low, high, P.PROBE_POINTS)
    gains = np.array(combination(grid)) - alone
    sign = np.sign(gains)
    crossings = int(np.sum(sign[:-1] * sign[1:] < 0))

    def gain(release):
        return combination([release])[0] - alone

    threshold, report = bisect_log(gain, low, high, gains[0], gains[-1],
                                   P.ROOT_TOL_REL, P.ROOT_MAX_ITER)
    report = {"crossings": crossings,
              "monotone": bool(np.all(np.diff(gains) > 0)),
              "window": (low, high), "threshold": threshold, **report}
    if threshold is None:
        return float("nan"), report
    return float(threshold) / P.KCAT_TCI[variant], report


# ---------------------------------------------------------------------------
# the outer solve: the state selectivity at which raising the dose stops helping
# ---------------------------------------------------------------------------
class Boundary:
    """The outer objective D(S), with the resumable cache in front of it.

    D(S) is the required fold change at a thousandfold dose minus the one at the
    reference dose. Positive means raising both doses raises the requirement --
    the dose hurts. Negative means it still helps.

    The cache is keyed on S rounded to six significant figures, which is finer
    than the bracket tolerance, so the last step of the outer bisection is a
    cache hit and costs nothing.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.values = {}
        self.evaluations = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue            # a run cut off mid-line
                self.values[row["key"]] = row["D"]

    def __len__(self):
        return len(self.values)

    def __call__(self, case, selectivity, quiet=False):
        key = _key(case, selectivity)
        if key in self.values:
            return self.values[key]
        started = time.time()
        reference, _ = required_fold_change(case, selectivity,
                                            P.THETA_REFERENCE)
        high, _ = required_fold_change(case, selectivity, P.THETA_HIGH)
        difference = high - reference
        self.values[key] = difference
        self.evaluations += 1
        with self.path.open("a") as handle:
            handle.write(json.dumps({"key": key, "D": difference}) + "\n")
        if not quiet:
            print(f"      S = {_number(selectivity):>10}  R(reference) = "
                  f"{reference:10.6f}  R(high dose) = {high:11.6f}  "
                  f"D = {difference:+11.6f}  ({time.time() - started:.0f} s)",
                  flush=True)
        return difference


def solve_boundary(case, boundary):
    """One row of one table: probe, uniqueness check, then bisect.

    The probe comes first so that the row can say how many sign changes D(S)
    has on it. One is a boundary and is bisected for. None is a result about
    the whole probe. More than one is not a boundary and is reported as such.
    """
    label = (f"{case['variant']} {alpha_label(case['alpha'])} "
             f"readout {case['t_end']:.0e} s "
             f"rescue {'on' if case['rescue'] else 'off'} "
             f"intrinsic {case['intrinsic']}")
    print(f"    {label}", flush=True)
    probe = list(case["probe"])
    differences = [boundary(case, s) for s in probe]
    crossings = sum(1 for i in range(len(differences) - 1)
                    if differences[i] * differences[i + 1] < 0)

    row = {"variant": case["variant"],
           "alpha_label": alpha_label(case["alpha"]),
           "alpha_nominal": case["alpha"],
           "t_end_s": float(case["t_end"]),
           "rescue": bool(case["rescue"]),
           "intrinsic_scenario": case["intrinsic"],
           "probe": str([float(s) for s in probe]),
           "D_on_probe": str([round(float(d), 6) for d in differences]),
           "n_sign_changes": crossings}

    if crossings == 0:
        # A NaN is not a negative value: it means D does not exist at that
        # state selectivity, because the reference dose has no threshold
        # inside the window the model allows a catalytic release in. Counting
        # one as a failure would answer a different question.
        defined = [d for d in differences if d == d]
        undefined = len(differences) - len(defined)
        negative = bool(defined) and all(d < 0 for d in defined)
        positive = bool(defined) and all(d > 0 for d in defined)
        note = ("" if not undefined else
                f" ({undefined} probe point(s) UNDEFINED: at those S the "
                "feeding threshold lies outside the builder's kcat window at "
                "theta=(1,1), so D does not exist there -- those probe points "
                "are undefined, not negative)")
        verdict = (("D < 0 at every DEFINED probe point -- raising the dose "
                    "still helps everywhere it can be evaluated" if negative
                    else "D > 0 at every DEFINED probe point -- raising the "
                    "dose hurts everywhere it can be evaluated" if positive
                    else "mixed signs but no crossing") + note)
        row.update({"n_D_undefined": undefined,
                    "all_defined_negative": negative,
                    "status": "NO_CROSSING", "S_star": float("nan"),
                    "bracket_lo": float("nan"), "bracket_hi": float("nan"),
                    "probe_lo": float(min(probe)),
                    "probe_hi": float(max(probe)), "verdict": verdict,
                    "n_iter": 0, "residual_D": float("nan"),
                    "converged": False,
                    "inside_measured_spread_7p7_to_182": False})
        print(f"      no crossing on S in [{min(probe):g}, "
              f"{max(probe):g}]: {verdict}", flush=True)
        return row

    if crossings > 1:
        row.update({"status": "MULTIPLE_CROSSINGS", "S_star": float("nan"),
                    "bracket_lo": float("nan"), "bracket_hi": float("nan"),
                    "verdict": "two or more sign changes -- NOT a boundary. "
                               "STOP.",
                    "n_iter": 0, "residual_D": float("nan"),
                    "converged": False,
                    "inside_measured_spread_7p7_to_182": False})
        print("      two or more sign changes: this is not a boundary",
              flush=True)
        return row

    i = next(i for i in range(len(differences) - 1)
             if differences[i] * differences[i + 1] < 0)
    low, high = probe[i], probe[i + 1]
    d_low, d_high = differences[i], differences[i + 1]
    print(f"      bracket [{low:g}, {high:g}]  D {d_low:+.6f} -> "
          f"{d_high:+.6f}; bisecting", flush=True)
    root, report = bisect_log(lambda s: boundary(case, s), low, high,
                              d_low, d_high, P.BOUNDARY_TOL_REL,
                              P.BOUNDARY_MAX_ITER)
    residual = report.get("f_root", 0.0)
    spread_lo, spread_hi = P.MEASURED_SELECTIVITY_SPREAD
    inside = bool(spread_lo <= root <= spread_hi)
    row.update({"status": "ROOT", "S_star": float(root),
                "bracket_lo": float(low), "bracket_hi": float(high),
                "D_at_bracket_lo": float(d_low),
                "D_at_bracket_hi": float(d_high),
                "n_iter": report["n_iter"], "residual_D": residual,
                "converged": True,
                "inside_measured_spread_7p7_to_182": inside,
                "verdict": (f"S* = {root:.6f}, bracket [{low:g}, {high:g}] "
                            f"({'INSIDE' if inside else 'OUTSIDE'} the "
                            f"{spread_lo:g}-{spread_hi:g} "
                            "cross-allele spread)")})
    print(f"      bracket [{low:g}, {high:g}], root S* = {root:.6f} inside it "
          f"({report['n_iter']} steps, residual {residual:+.3e})", flush=True)
    return row


def table(name, boundary):
    """One table: one row per case, in the order the figure draws them."""
    rows = [solve_boundary(case, boundary) for case in TABLES[name]]
    frame = pd.DataFrame(rows)
    if name in BRACKET_SOURCE_TABLES:
        frame["bracket_source"] = COMMON_PROBE_SOURCE
    return frame


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\nWHAT IS BEING SOLVED")[0],
        epilog="Writes " + ", ".join(TABLES) + ".",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help="folder the tables are written to")
    ap.add_argument("--to-figures", action="store_true",
                    help="also copy them into the data/ folder the figure "
                         "script reads")
    ap.add_argument("--table", action="append", choices=sorted(TABLES),
                    metavar="NAME",
                    help="compute only this table; may be given more than "
                         "once. Finished D(S) values are shared through the "
                         "cache file, so the tables may be split across runs "
                         "or processes in any order")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [n for n in TABLES if args.table is None or n in args.table]
    started = mark = time.time()

    boundary = Boundary(args.out / _CACHE_FILE)
    if len(boundary):
        print(f"resuming: {len(boundary)} D(S) values already on disk",
              flush=True)

    for name in wanted:
        print(f"{name}", flush=True)
        frame = table(name, boundary)
        frame.to_csv(args.out / name, index=False)
        print(f"  {len(frame)} rows -> {name} ({time.time() - mark:.0f} s)",
              flush=True)
        mark = time.time()

    if args.to_figures:
        target = Path(__file__).resolve().parent / "data"
        for name in wanted:
            shutil.copyfile(args.out / name, target / name)
        print(f"  copied {len(wanted)} tables into {target}")
    print(f"done in {time.time() - started:.0f} s; "
          f"{boundary.evaluations} new D(S) evaluations; tables in {args.out}")


if __name__ == "__main__":
    main()
