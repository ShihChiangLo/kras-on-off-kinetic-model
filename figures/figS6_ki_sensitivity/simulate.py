# -*- coding: utf-8 -*-
"""Simulations behind the figure in S6 Text: the covalent arm against K_I.

    python3 simulate.py                  writes both tables into results/
    python3 simulate.py --to-figures     also copies the figure's table into
                                         data/, where make_figure.py reads it

Writes
    ki_dose_scan.csv    the times to 50, 90 and 99% capture at every K_I, ON
                        dose and ON condition: one row per simulation
    ki_sensitivity.csv  the figure's table, one row per K_I and
                        intrinsic-hydrolysis scenario, summarised from the scan

K_I is the binding constant of the first, reversible step of the covalent
RAS(OFF) inhibitor on KRAS G12C. It is the one constant of the arm that a
different source would move, which is why it is swept: over two orders of
magnitude around the measured value, 6.0e-8 to 6.0e-6 M, with the published
second-order efficiency k_inact/K_I held fixed at 9900 M^-1 s^-1. Fixing the
efficiency means each K_I determines all three constants
(`two_arm.model.covalent_constants`), and sweeping K_I then sweeps exactly one
thing: how saturated the first step is at
the working dose, k_obs = k_inact [OFF] / (K_I + [OFF]).

At each K_I, in each of the two intrinsic-hydrolysis scenarios for this allele,
the table records

  * the time to 90% covalent capture of the mutant RAS pool with the RAS(OFF)
    inhibitor alone at 1e-6 M;
  * the shortest such time over a ten-point dose grid of the RAS(ON) inhibitor,
    and the ratio of the two times, which is the speedup the combination buys;
  * the same minimum for a shielding control, in which the RAS(ON) inhibitor
    binds exactly as before but its catalytic release is set to zero, so the
    tri-complex withholds RAS-GTP without producing the RAS-GDP the covalent
    arm needs;
  * whether the RAS(OFF) inhibitor on its own reaches 90% capture inside a 6 h
    window at any dose up to 1e-3 M.

The three `fires_*` columns are falsification conditions, fixed before the
sweep was run. True means the condition was met, which constrains what the
article may claim at that K_I; it is not a pass or a fail of this code. They
are, in order of the columns:

  speedup_below_1p5x   the combination speeds capture up by less than 1.5-fold,
                       so "the RAS(ON) inhibitor roughly halves the capture
                       time" may not be written without a K_I condition;
  shield_faster        shielding alone is faster than the RAS(OFF) inhibitor
                       alone, which would mean the split between catalysis and
                       shielding does not hold;
  off_alone_makes_6h   the RAS(OFF) inhibitor alone does reach 90% capture
                       within 6 h at some dose up to 1e-3 M, so the statement
                       that it does not needs a K_I condition.

Takes about 2.5 minutes: 792 integrations, over 72 compiled models, two per
scenario and K_I because the 24 h and the 6 h window need separate time grids.
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# The script is meant to be run from its own folder, like make_figure.py, so
# the repository root goes on the import path for `two_arm`.
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from two_arm.model import (CAPTURED, MUTANT_TOTAL, build,  # noqa: E402
                           check_catalysis, covalent_constants)
from two_arm.parameters import (CAPTURE_CRITERION,  # noqa: E402
                                COVALENT_VARIANT, G12C_HYDROLYSIS, K_I,
                                KCAT_TCI, KINACT_OVER_KI, KOFF_COVALENT,
                                N_POINTS_COVALENT, T_BENEFIT, T_END_COVALENT)
from two_arm.simulate import (compile_model, initial_overrides,  # noqa: E402
                              integrate, time_at_level)

# --- the sweep ------------------------------------------------------------
#: The band K_I is swept over, M, and how many log-spaced points in it.
K_I_BAND = (6.0e-8, 6.0e-6)
K_I_POINTS = 13
#: Values unioned into the log grid so that each is simulated by construction
#: rather than by luck. Six, in the order written: one value below the covalent
#: arm's working K_I; that working value, 2.20e-7 M; three values above it; and
#: the top of the band, where the first step is far from saturated and the
#: chemistry approaches a one-step description.
K_I_NAMED = (1.73e-7, 2.20e-7, 2.67e-7, 3.06e-7, 6.3030303030303030e-7, 6.0e-6)

#: Intrinsic hydrolysis of mutant RAS: the two arms of one joint fit, carried
#: through as declared scenarios rather than one of them being chosen.
SCENARIOS = tuple(G12C_HYDROLYSIS)

#: RAS(ON) inhibitor doses, M. Zero is the RAS(OFF)-alone reference.
ON_DOSES = (0.0, 1.0e-9, 3.0e-9, 1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6,
            3.0e-6, 1.0e-5)
#: `full` is the measured catalytic-release rate; `shield` is the same model
#: with that rate at exactly zero, which is binding without catalysis.
ON_CONDITIONS = ("full", "shield")
#: The single RAS(OFF) dose everything except the window test is read at, M.
#: Five times the mutant RAS pool, and above the 1.8e-7 M below which no dose
#: can reach 90% capture at all: the inhibitor is consumed, one molecule per
#: RAS, so a substoichiometric dose is a ceiling and not a slow approach.
REFERENCE_OFF = 1.0e-6

#: The window test's RAS(OFF) dose bracket, M. The upper end is the dose the
#: claim under test names; the lower end is far below the stoichiometric floor.
WINDOW_BRACKET = (1.0e-10, 1.0e-3)

#: Capture levels read off every curve, %. All three are in the dose scan; the
#: figure and all three falsification conditions use 90%.
CAPTURE_LEVELS = (50.0, 90.0, 99.0)
#: The speedup below which the first falsification condition is met.
SPEEDUP_FLOOR = 1.5

_MODELS: dict[tuple, tuple] = {}


def k_i_grid():
    """The K_I values to simulate, M, in increasing order.

    A log grid over the band with the named points unioned in, so that each of
    them is simulated by construction. A named point that coincides with a grid
    point is kept once: the top of the band is such a point, so the grid is one
    shorter than the two lists together.
    """
    grid = list(np.logspace(np.log10(K_I_BAND[0]), np.log10(K_I_BAND[1]),
                            K_I_POINTS))
    grid += [float(k) for k in K_I_NAMED]
    out = []
    for value in sorted(grid):
        if not out or abs(value - out[-1]) / value > 1e-9:
            out.append(float(value))
    for named in K_I_NAMED:
        if not any(abs(named - v) / named < 1e-9 for v in out):
            raise SystemExit(f"named K_I {named:g} M is not in the grid")
    return out


def simulator(scenario, k_i, t_end, n_points):
    """The compiled covalent model for one scenario and K_I, and its time grid.

    One model serves every dose and both ON conditions: the catalytic-release
    rule is always built, at rate zero if need be, so the shielding control is
    the same topology and can be set through a parameter override. The three
    covalent constants are not overridden that way. They are three faces of two
    measured numbers, so they are derived together and set when the model is
    built, where the builder's own bounds still see them.

    Models are kept rather than rebuilt: building and compiling one costs about
    three integrations, and each serves between two and twenty of them. The
    24 h and the 6 h window need separate models, because the time grid is
    fixed when the model is compiled.
    """
    key = (scenario, float(k_i), float(t_end), int(n_points))
    if key not in _MODELS:
        constants = covalent_constants(KINACT_OVER_KI, k_i, KOFF_COVALENT)
        model = build(COVALENT_VARIANT, KCAT_TCI[COVALENT_VARIANT],
                      kon_off=constants["kon"], koff_off=constants["koff"],
                      kinact=constants["kinact"], covalent=True,
                      hydrolysis=G12C_HYDROLYSIS[scenario])
        tspan, sim = compile_model(model, t_end, n_points)
        _MODELS[key] = (model, tspan, sim, constants)
    return _MODELS[key]


def catalytic_release(condition):
    """The mutant catalytic-release rate that defines one ON condition, s^-1."""
    if condition == "full":
        return KCAT_TCI[COVALENT_VARIANT]
    if condition == "shield":
        return 0.0
    raise ValueError(f"unknown ON condition {condition!r}; expected one of "
                     f"{ON_CONDITIONS}")


def capture(scenario, k_i, cases, t_end=T_END_COVALENT,
            n_points=N_POINTS_COVALENT):
    """Capture curves for a list of {on_M, off_M, cond} cases at one K_I.

    Returns (capture at the last time point, dict of level to time) per case.
    Capture is covalently bound inhibitor as a percentage of the instantaneous
    total mutant RAS, nucleotide-loaded and nucleotide-free: one inhibitor
    carries one RAS and the bond is terminal, so the drug-side observable is
    the captured-RAS count.
    """
    model, tspan, sim, _ = simulator(scenario, k_i, t_end, n_points)
    overrides = []
    for case in cases:
        kcat = catalytic_release(case["cond"])
        # A rate set through an override has not passed the builder's bounds,
        # so it is checked here instead.
        check_catalysis([kcat], COVALENT_VARIANT, G12C_HYDROLYSIS[scenario])
        overrides.append({**initial_overrides(), "OnDrug_0": case["on_M"],
                          "OffDrug_0": case["off_M"], "mukcat_TCI": kcat})
    out = []
    for g in integrate(model, tspan, sim, overrides):
        pool = g[MUTANT_TOTAL[0]] + g[MUTANT_TOTAL[1]]
        curve = 100.0 * g[CAPTURED] / pool
        out.append((float(curve[-1]),
                    {level: time_at_level(tspan, curve, level)
                     for level in CAPTURE_LEVELS}))
    return out


def dose_scan(grid):
    """Times to each capture level over the ON dose grid, at every K_I.

    One row per intrinsic-hydrolysis scenario, K_I, ON condition and ON dose,
    all at the reference RAS(OFF) dose. The figure's table is summarised from
    this one, and this one is written out beside it.
    """
    cases = [{"on_M": on, "off_M": REFERENCE_OFF, "cond": cond}
             for cond, on in itertools.product(ON_CONDITIONS, ON_DOSES)]
    rows, started = [], time.time()
    for scenario in SCENARIOS:
        for k_i in grid:
            constants = simulator(scenario, k_i, T_END_COVALENT,
                                  N_POINTS_COVALENT)[3]
            for case, (_, times) in zip(cases, capture(scenario, k_i, cases)):
                rows.append({"scenario": scenario, "K_I_M": k_i,
                             "kinact_s": constants["kinact"],
                             "kon_M": constants["kon"], **case,
                             **{f"t{level:g}_s": times[level]
                                for level in CAPTURE_LEVELS}})
        print(f"  dose scan, {scenario:12s} {len(grid)} K_I values "
              f"({time.time() - started:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def reaches_criterion_in_window(scenario, k_i):
    """Does the RAS(OFF) inhibitor alone reach 90% capture inside 6 h?

    True when some dose inside the bracket does: capture at the end of the
    window rises with dose, so this requires both that the bottom of the
    bracket falls short of the criterion and that the top of it clears the
    criterion. The dose itself is not needed here, so the bracket is not
    bisected.
    """
    cases = [{"on_M": 0.0, "off_M": off, "cond": "full"}
             for off in WINDOW_BRACKET]
    (low, _), (high, _) = capture(scenario, k_i, cases, t_end=T_BENEFIT)
    return bool(low - CAPTURE_CRITERION < 0.0
                and high - CAPTURE_CRITERION > 0.0)


def window_tests(grid):
    """Whether the RAS(OFF) inhibitor alone makes the 6 h window, by scenario
    and K_I."""
    out, started = {}, time.time()
    for scenario in SCENARIOS:
        for k_i in grid:
            out[(scenario, k_i)] = reaches_criterion_in_window(scenario, k_i)
        print(f"  6 h window, {scenario:12s} {len(grid)} K_I values "
              f"({time.time() - started:.0f} s)", flush=True)
    return out


def summarise(scan, in_window, grid):
    """One row per intrinsic-hydrolysis scenario and K_I, from the dose scan.

    K_I itself is taken from the grid and not from the scan, so the sweep's own
    values reach the table; scan rows are matched to it by relative closeness.
    """
    criterion = f"t{CAPTURE_CRITERION:g}_s"
    rows = []
    for scenario in SCENARIOS:
        for k_i in grid:
            at = scan[(scan.scenario == scenario)
                      & (np.isclose(scan.K_I_M, k_i, rtol=1e-9))]
            full = at[at.cond == "full"].sort_values("on_M")
            shield = at[at.cond == "shield"].sort_values("on_M")
            kinact = float(full.kinact_s.iloc[0])
            alone = float(full.loc[full.on_M == 0.0, criterion].iloc[0])
            dosed = full[full.on_M > 0.0]
            best = dosed.loc[dosed[criterion].idxmin()]
            speedup = alone / float(best[criterion])
            shielded = shield[shield.on_M > 0.0][criterion]
            shield_best = (float(shielded.min()) if shielded.notna().any()
                           else np.nan)
            shield_faster = bool(shield_best == shield_best
                                 and shield_best < alone)
            made_window = in_window[(scenario, k_i)]
            rows.append({
                "scenario": scenario,
                "K_I_M": k_i,
                "off_over_K_I": REFERENCE_OFF / k_i,
                "kinact_s": kinact,
                "kon_M": float(full.kon_M.iloc[0]),
                "k_obs_at_ref_off_s": (kinact * REFERENCE_OFF
                                       / (k_i + REFERENCE_OFF)),
                "ref_off_M": REFERENCE_OFF,
                "t90_off_alone_h": alone / 3600.0,
                "t90_best_full_h": float(best[criterion]) / 3600.0,
                "best_on_M": float(best["on_M"]),
                "time_speedup_x": speedup,
                "t90_min_shield_h": shield_best / 3600.0,
                "shield_min_over_off_alone_x": shield_best / alone,
                "shield_ever_reaches_criterion": bool(shielded.notna().any()),
                "off_alone_reaches_90pct_in_6h": made_window,
                "fires_speedup_below_1p5x": bool(speedup < SPEEDUP_FLOOR),
                "fires_shield_faster": shield_faster,
                "fires_off_alone_makes_6h": made_window})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep K_I for the figure in S6 Text.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"),
                    help="where to write the two tables "
                         "(default: results/)")
    ap.add_argument("--to-figures", action="store_true",
                    help="also copy the figure's table into data/, where "
                         "make_figure.py reads it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    grid = k_i_grid()
    print(f"K_I from {K_I_BAND[0]:g} to {K_I_BAND[1]:g} M, "
          f"{K_I_POINTS} log-spaced points plus {len(K_I_NAMED)} named ones, "
          f"{len(grid)} after coincidences; {len(SCENARIOS)} "
          f"intrinsic-hydrolysis scenarios", flush=True)
    scan = args.out / "ki_dose_scan.csv"
    dose_scan(grid).to_csv(scan, index=False)
    in_window = window_tests(grid)
    # The summary is computed from the scan as written, not from the values
    # still in memory. Writing a float and reading it back is not exact here:
    # pandas writes the shortest text that reads back to the same double but
    # does not read it back with that guarantee, so the two tables agree to
    # the last digit only if both are built from the same text. The parser is
    # named rather than left to the default, which it currently equals, so that
    # a change of default cannot move the last digit of the published table.
    table = summarise(pd.read_csv(scan, float_precision="high"),
                      in_window, grid)
    target = args.out / "ki_sensitivity.csv"
    table.to_csv(target, index=False)
    if args.to_figures:
        shutil.copyfile(target, HERE / "data" / "ki_sensitivity.csv")

    fired = {name: int(table[name].sum()) for name in table.columns
             if name.startswith("fires_")}
    print(f"speedup over the whole band: {table.time_speedup_x.min():.4f}x to "
          f"{table.time_speedup_x.max():.4f}x")
    at_measured = table[np.isclose(table.K_I_M, K_I, rtol=1e-9)]
    print("at the measured K_I = {:g} M: speedup ".format(K_I)
          + ", ".join(f"{r.scenario} {r.time_speedup_x:.4f}x"
                      for _, r in at_measured.iterrows()))
    for name, count in fired.items():
        print(f"{name}: condition met at {count}/{len(table)} points")
    print(f"done in {time.time() - started:.0f} s; table in {target}")


if __name__ == "__main__":
    main()
