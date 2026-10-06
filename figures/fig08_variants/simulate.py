# -*- coding: utf-8 -*-
"""Simulations behind Fig 8: KRAS G12D against G12V at equal concentrations.

    python3 simulate.py            the table, into results/
    python3 simulate.py --help

Writes
    fixed_concentration_grid.csv   both alleles over one 13 x 13 grid of
                                   absolute doses, each dose point read at the
                                   six arms

Fig 8 also reads `matched_theta_grid.csv`, where each allele's RAS(ON) dose is
a multiple of its own dissociation constant. That table is shared with Fig 7
and is written by `../fig07_feeding/simulate.py`, which also defines the model
setup, the per-arm readouts and the dose-grid runner this script uses.

The two grids answer different questions. Matching the dose multiple matches
concentration divided by dissociation constant, not occupancy: the RAS(ON)
inhibitor is depleted by binding, so G12V's RAS(ON) occupancy is 1.33 times
G12D's at the reference dose and rises to 3.19 at the top of the grid, falling
below parity only under theta_ON = 0.3, where both occupancies are under 2%.
The grid here holds the concentration itself equal instead, which is what a
single dose given to a mixed population would do, and the occupancy then
differs between the alleles because their affinities do. The dose range is
the one that covers low occupancy through saturation for both alleles and for
the RAS(OFF) arm: three decades either side of the weakest and the strongest
dissociation constant in play.
"""
import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

# Loaded by path rather than by name: the shared code lives in the other
# figure's script, whose file name is the same as this one's.
_shared_path = HERE.parent / "fig07_feeding" / "simulate.py"
_spec = importlib.util.spec_from_file_location("fig07_feeding_simulate",
                                               _shared_path)
feeding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feeding)

from two_arm.parameters import KD2_DECIMAL, KD_OFF_GDP  # noqa: E402

#: The absolute dose range of each arm, in M, as (low, high, points). Both ends
#: are derived from the dissociation constants rather than chosen: the RAS(ON)
#: axis runs from a thousandth of the tighter allele's constant to a thousand
#: times the weaker one's, and the RAS(OFF) axis three decades either side of
#: the probe's.
ON_DOSE_RANGE = (1.31e-10, 3.64e-4, 13)
OFF_DOSE_RANGE = (2.0e-10, 2.0e-4, 13)


def _check_dose_range():
    """Confirm the dose ranges are still the three decades they are meant to be.

    Written as decimals above so that the grid is reproducible to the last bit
    of a double; this checks that those decimals still bracket the constants
    they were derived from, in case one of the constants is ever revised.
    """
    tight = min(KD2_DECIMAL[v] for v in feeding.VARIANTS)
    weak = max(KD2_DECIMAL[v] for v in feeding.VARIANTS)
    for (lo, hi, _), (expected_lo, expected_hi) in (
            (ON_DOSE_RANGE, (1e-3 * tight, 1e3 * weak)),
            (OFF_DOSE_RANGE, (1e-3 * KD_OFF_GDP, 1e3 * KD_OFF_GDP))):
        if not (np.isclose(lo, expected_lo) and np.isclose(hi, expected_hi)):
            raise ValueError(
                f"dose range [{lo:g}, {hi:g}] M is no longer the three decades "
                f"either side of the dissociation constants it was derived "
                f"from, which are now [{expected_lo:g}, {expected_hi:g}] M")


def dose_axis(dose_range):
    """One axis of the grid, spaced evenly in log dose."""
    lo, hi, points = dose_range
    return np.logspace(np.log10(lo), np.log10(hi), points)


def fixed_concentration_grid():
    """Both alleles over the shared grid of absolute doses."""
    _check_dose_range()
    on_doses, off_doses = dose_axis(ON_DOSE_RANGE), dose_axis(OFF_DOSE_RANGE)
    frames = []
    for variant in feeding.VARIANTS:
        started = time.perf_counter()
        frames.append(feeding.dose_grid(variant, on_doses, off_doses,
                                        "fixed_concentration"))
        print(f"  {variant}: {len(frames[-1])} dose points, "
              f"{time.perf_counter() - started:.1f} s")
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "results",
                        help="where to write the table (default: results/)")
    parser.add_argument("--to-figures", action="store_true",
                        help="also copy it into the data/ folder this figure "
                             "is drawn from")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("dose grid at equal concentrations, both alleles")
    grid = fixed_concentration_grid()
    grid.to_csv(args.out / "fixed_concentration_grid.csv", index=False)
    print(f"  wrote fixed_concentration_grid.csv, {len(grid)} rows")
    if args.to_figures:
        data = HERE / "data"
        data.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out / "fixed_concentration_grid.csv",
                        data / "fixed_concentration_grid.csv")
        print("  copied into data/: fixed_concentration_grid.csv")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
