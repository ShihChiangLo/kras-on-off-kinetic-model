"""Simulations behind S2 Fig: how much catalytic release the combination would
need, across the design space of the reversible RAS(OFF) probe.

    python3 -m figures.figS2_band.simulate
    python3 figures/figS2_band/simulate.py --out some/other/folder

Writes
    required_catalysis_fold.csv   the required fold change R at every corner of
                                  the state selectivity by effector
                                  attenuation grid, for KRAS G12D and G12V

R is the factor by which that allele's measured catalytic release would have to
rise before the combination beats the RAS(OFF) arm alone on RAS(OFF)
engagement. R below 1 is therefore a margin rather than a requirement: the
measured rate is already past the crossing.

The quantity solved for is the catalytic-release rate at which the engagement
gain changes sign, where the engagement gain is RAS(OFF) engagement of the
mutant RAS pool under both inhibitors minus the same quantity under the
RAS(OFF) inhibitor alone, both read at the end of the horizon. Dividing that
rate by the measured one gives R.

The effector rescue route is closed at every corner computed here, which is
what the figure caption states: a drug-bound RAS that is also holding the
effector does not hydrolyse its GTP. Both alleles are simulated with
drug-bound RAS-GTP not hydrolysing intrinsically either.
"""
import argparse
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
from two_arm.model import (MUTANT_TOTAL, OFF_BOUND, build, catalysis_bounds,
                           check_catalysis)
from two_arm.simulate import (bisect_log, compile_model, dose_scales,
                              initial_overrides, integrate)

TABLE = "required_catalysis_fold.csv"

#: The two alleles the figure has panels for, in the order the table lists
#: them. The RAS(OFF) probe is an idealized compound and has no measured
#: selectivity on either, which is why both are swept rather than fitted.
ALLELES = ("G12D", "G12V")

#: Dose of each arm, as a multiple of that arm's dissociation constant. One
#: for both arms is the reference design point, where each dose equals that
#: arm's dissociation constant. The inhibitors are depleted by binding, so this
#: is a dose coordinate and not an occupancy.
THETA_ON, THETA_OFF = P.THETA_REFERENCE

#: Catalytic-release rate at which the engagement gain is read without
#: solving: the rate measured for that allele, and the denominator of R.
MEASURED_RELEASE = P.KCAT_TCI


def label(prefix, value):
    """`S=182`, `alpha=10`, `S=inf`: the two label columns of the table.

    Whole numbers are written without a decimal point, which is what the
    published table has; the one non-integer value of either grid is S = 7.7.
    """
    if not math.isfinite(value):
        return f"{prefix}=inf"
    if float(value).is_integer() and abs(value) < 1e6:
        return f"{prefix}={int(value)}"
    return f"{prefix}={value:.6g}"


def corner(allele, selectivity, attenuation):
    """One corner's model, integration grid and the two doses, in M.

    A model is built per corner because state selectivity and effector
    attenuation are rate constants of it, not quantities a simulation can set:
    whether each is finite decides which rules exist, and the value then sets
    one rate. Both doses are read back off the built model so that a dose
    expressed as a multiple of a dissociation constant cannot drift from the
    constant the model carries.
    """
    release_floor, _ = catalysis_bounds(allele)
    model = build(allele, release_floor, kon_off=P.KON_OFF,
                  koff_off=P.KOFF_OFF, state_selectivity=selectivity,
                  effector_attenuation=attenuation,
                  drug_bound_hydrolysis=False, effector_rescue=False)
    tspan, simulator = compile_model(model, P.T_END, P.N_POINTS)
    kd2, kd_off = dose_scales(model)
    return model, tspan, simulator, THETA_ON * kd2, THETA_OFF * kd_off


def engagement_gain(allele, model, tspan, simulator, on_dose, off_dose):
    """Return a function of the catalytic-release rate: the engagement gain, pp.

    Every call integrates the same six arms of `two_arm.simulate.ARMS` at the
    same two doses, so that the arm the gain is measured against is never a
    separately dosed simulation. Only the combination and the RAS(OFF)-only arm
    enter the gain; the other four are computed because they come from the same
    batch and cost nothing to carry, and because dropping them would change
    which arms share a batch.
    """
    base = initial_overrides()

    def gain(release):
        check_catalysis([release], allele)
        arms = [(0.0, 0.0, 0.0),
                (on_dose, 0.0, release),
                (0.0, off_dose, 0.0),
                (on_dose, off_dose, release),
                (on_dose, off_dose, 0.0),
                (on_dose, 0.0, 0.0)]
        cases = [{**base, "OnDrug_0": on, "OffDrug_0": off,
                  "mukcat_TCI": kcat} for on, off, kcat in arms]
        trajectories = integrate(model, tspan, simulator, cases)

        def engaged_pct(g):
            pool = sum(g[name] for name in MUTANT_TOTAL)
            return float((100.0 * g[OFF_BOUND] / pool)[-1])

        return engaged_pct(trajectories[3]) - engaged_pct(trajectories[2])

    return gain


def required_fold(allele, selectivity, attenuation):
    """Solve for R at one corner; return it with the probe's own diagnostics.

    The window the rate is sought in is the one the model enforces on any
    non-zero catalytic release: that allele's intrinsic hydrolysis rate at the
    bottom and GAP-stimulated hydrolysis of wild-type RAS at the top. It is
    first walked on a log grid, which reports how many sign changes the gain
    has on that window, and the two ends of the grid then seed the bisection.
    A window with no crossing is an answer about the design point rather than a
    failure, and is reported as R = NaN.
    """
    model, tspan, simulator, on_dose, off_dose = corner(
        allele, selectivity, attenuation)
    gain = engagement_gain(allele, model, tspan, simulator, on_dose, off_dose)
    low, high = catalysis_bounds(allele)

    probe = np.geomspace(low, high, P.PROBE_POINTS)
    gains = np.array([gain(rate) for rate in probe], dtype=float)
    signs = np.sign(gains)
    crossings = int(np.sum(signs[:-1] * signs[1:] < 0))

    root, report = bisect_log(gain, low, high, gains[0], gains[-1],
                              P.ROOT_TOL_REL, P.ROOT_MAX_ITER)
    measured = MEASURED_RELEASE[allele]
    return {
        "R": float(root) / measured if root is not None else float("nan"),
        "release_at_crossing": float(root) if root is not None
        else float("nan"),
        "gain_at_window_floor": float(gains[0]),
        "gain_at_window_ceiling": float(gains[-1]),
        "n_crossings_on_probe": crossings,
        "monotone_on_probe": bool(np.all(np.diff(gains) > 0)),
        "n_iterations": report["n_iter"],
        "gain_at_root": report.get("f_root", float("nan")),
    }


def required_fold_table(verbose=True):
    """R at all sixteen combinations of state selectivity and effector
    attenuation, for each of the two alleles, in the order the figure reads
    them."""
    rows = []
    for allele in ALLELES:
        for selectivity in P.S_VALUES:
            for attenuation in P.ALPHA_VALUES:
                started = time.time()
                solved = required_fold(allele, selectivity, attenuation)
                rows.append({"variant": allele,
                             "S_label": label("S", selectivity),
                             "alpha_label": label("alpha", attenuation),
                             "R_required_fold": solved["R"]})
                if verbose:
                    note = ""
                    if solved["n_crossings_on_probe"] != 1:
                        note += (f"  [{solved['n_crossings_on_probe']} sign "
                                 f"changes on the probe]")
                    if not solved["monotone_on_probe"]:
                        note += "  [gain not increasing throughout the window]"
                    print(f"  {allele} {label('S', selectivity):>7} "
                          f"{label('alpha', attenuation):>11}  "
                          f"R = {solved['R']:.4f}  "
                          f"(gain {solved['gain_at_window_floor']:+.2f} to "
                          f"{solved['gain_at_window_ceiling']:+.2f} pp, "
                          f"{solved['n_iterations']} bisections, "
                          f"{time.time() - started:.0f} s){note}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\nWrites")[0],
        epilog="Writes " + TABLE + ". Takes about 26 minutes: 32 corners, "
               "each one a grid walk and a bisection over the "
               "catalytic-release window, about 370 integrations per corner.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help="folder the table is written to")
    ap.add_argument("--to-figures", action="store_true",
                    help="also copy it into the data/ folder the figure "
                         "script reads")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("required catalytic release, by state selectivity and effector "
          "attenuation", flush=True)
    table = required_fold_table()
    table.to_csv(args.out / TABLE, index=False)
    print(f"  {len(table)} rows -> {TABLE}", flush=True)

    if args.to_figures:
        target = Path(__file__).resolve().parent / "data"
        shutil.copyfile(args.out / TABLE, target / TABLE)
        print(f"  copied {TABLE} into {target}")
    print(f"done in {time.time() - started:.0f} s; table in {args.out}")


if __name__ == "__main__":
    main()
