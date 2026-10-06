"""Simulations behind Figs 2-5.

    python3 -m ras_switch.simulate                       all three tables, into results/
    python3 -m ras_switch.simulate --skip-dose-response  Figs 2-4 only
    python3 -m ras_switch.simulate --to-figures          also copy them into figures/*/data/

Writes
    ras_gtp_summary.csv   Figs 2 and 3: % of RAS that is GTP-bound at steady state,
                          mean and sample SD over the nine conditions
    state_distribution.csv
                          Fig 4: % nucleotide-free, GTP-bound and GDP-bound RAS,
                          one row per variant and condition
    dose_response.csv     Fig 5: active RAS after 60 min of AMG 510 as % of the
                          pre-drug value, mean and sample SD over the nine conditions
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pysb.simulator import ScipyOdeSimulator

from .model import ACTIVE, FREE, INACTIVE, build
from .parameters import (CONDITIONS, DRUG_DOSES, INCUBATION_S, N_STEADY, SOLVER,
                         T_STEADY)

VARIANT_ORDER = ["G12C", "G12D", "G12V", "WT"]


def _last(result, names):
    return float(sum(result.observables[n][-1] for n in names))


def steady_state(model):
    tspan = np.linspace(0, T_STEADY, N_STEADY)
    return ScipyOdeSimulator(model, tspan=tspan, **SOLVER).run()


def steady_state_readouts(variant, condition):
    """GTP-bound percentages (Figs 2, 3) and the three-state split (Fig 4) at t = 1e4 s."""
    r = steady_state(build(variant, condition))
    out = {}
    for pool, prefixes in (("total", ["", "mu"]), ("mutant", ["mu"])):
        groups = {state: [p + n for p in prefixes for n in names]
                  for state, names in (("gtp", ACTIVE), ("gdp", INACTIVE), ("free", FREE))}
        total = sum(_last(r, g) for g in groups.values())
        for state, g in groups.items():
            out[f"{pool}_{state}_pct"] = _last(r, g) / total * 100
    return out


def dose_response(variant, condition, intrinsic_hydrolysis):
    """Active RAS 60 min after adding AMG 510 at steady state, % of the pre-drug value.

    Returns one value per dose in DRUG_DOSES.
    """
    model = build(variant, condition, with_amg510=True,
                  intrinsic_hydrolysis=intrinsic_hydrolysis)
    before = steady_state(model)
    active = ACTIVE + ["mu" + n for n in ACTIVE]
    drug_index = [str(s) for s in model.species].index("AMG510(b=None)")
    tspan = np.linspace(T_STEADY, T_STEADY + INCUBATION_S, INCUBATION_S + 1)
    sim = ScipyOdeSimulator(model, tspan=tspan, **SOLVER)
    out = []
    for dose in DRUG_DOSES:
        start = np.array(before.species[-1], dtype=float)
        start[drug_index] = dose
        after = sim.run(initials=start)
        out.append(_last(after, active) / _last(before, active) * 100)
    return out


def _mean_sd(df, keys, cols):
    agg = {}
    for c in cols:
        agg[f"mean_{c}"] = (c, "mean")
        agg[f"sd_{c}"] = (c, lambda v: v.std(ddof=1))
    agg["n"] = (cols[0], "count")
    return df.groupby(keys, as_index=False, sort=False).agg(**agg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-dose-response", action="store_true",
                    help="skip Fig 5, which takes most of the run time")
    ap.add_argument("--to-figures", action="store_true",
                    help="copy the tables into the data/ folders the figure scripts read")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = [{"variant": v, "ras_M": c[0], "effector_M": c[1], "gap_M": c[2],
             **steady_state_readouts(v, c)} for v in VARIANT_ORDER for c in CONDITIONS]
    states = pd.DataFrame(rows)
    states.to_csv(args.out / "state_distribution.csv", index=False)
    summary = _mean_sd(states.rename(columns={"mutant_gtp_pct": "transfected_gtp_pct"}),
                       ["variant"], ["total_gtp_pct", "transfected_gtp_pct"])
    summary.to_csv(args.out / "ras_gtp_summary.csv", index=False)
    print(f"Figs 2-4 tables written ({time.time() - t0:.0f} s)", flush=True)

    if not args.skip_dose_response:
        rows = []
        for v in ["G12C", "G12D", "G12V"]:
            for hydrolysis in (True, False):
                curve = v if hydrolysis else f"{v} without intrinsic GTP hydrolysis"
                for c in CONDITIONS:
                    for dose, signal in zip(DRUG_DOSES, dose_response(v, c, hydrolysis)):
                        rows.append({"curve": curve, "drug_M": dose, "signal_pct": signal})
                print(f"  {curve} done ({time.time() - t0:.0f} s)", flush=True)
        _mean_sd(pd.DataFrame(rows), ["curve", "drug_M"], ["signal_pct"]).to_csv(
            args.out / "dose_response.csv", index=False)
    if args.to_figures:
        figures = Path(__file__).resolve().parent.parent / "figures"
        targets = {"ras_gtp_summary.csv": ["fig02_total_validation", "fig03_transfected_validation"],
                   "state_distribution.csv": ["fig04_state_distribution"],
                   "dose_response.csv": ["fig05_dose_response"]}
        for name, folders in targets.items():
            if name == "dose_response.csv" and args.skip_dose_response:
                continue
            for folder in folders:
                shutil.copyfile(args.out / name, figures / folder / "data" / name)
    print(f"done in {time.time() - t0:.0f} s; tables in {args.out}")


if __name__ == "__main__":
    main()
