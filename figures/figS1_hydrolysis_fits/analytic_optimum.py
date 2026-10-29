#!/usr/bin/env python3
"""analytic_optimum.py -- the exact least-squares optimum of the S1 Fig
hydrolysis fit, computed without any numerical ODE integration.

This script produces the rate constants reported in the article. `make_figure.py`
fits the same model with an ODE solver and is kept as an independent structural
check on it, not as the source of the reported values; the two agree to between
2.6e-8 and 2.3e-5 relative, variant by variant.

Why the reported values come from here
--------------------------------------
The model is one irreversible first-order step with the phosphate baseline fixed
at zero and a declared pool R0, so the observable is exactly

    Pi(t) = R0 (1 - exp(-k t)),

and the least-squares objective has an exact stationarity condition

    g(k) = sum_i [ R0 (1 - exp(-k t_i)) - y_i ] * t_i * exp(-k t_i) = 0,

whose root is bracketed and bisected here at 50 decimal digits. No solver
tolerance enters it.

The ODE route reaches the same optimum only to within its own finite-difference
gradient noise, so the value it returns depends on where the fit is started.
Swept from half to twice the optimum, that dependence measures 4e-8 to 4e-3
relative across the seven alleles. For KRAS G12A it exceeds the distance from
the reported value to its third-significant-figure rounding boundary (3.4e-4
relative), so over that sweep the ODE fit returns values printing as 5.15e-4,
5.16e-4 and 5.17e-4. Swept identically, the closed form returns one
three-significant-figure value for every allele -- and that is a property of the
model rather than an observation: the stationarity condition has a single sign
change over twelve decades in k, so any bracketing solver reaches the same root
from any start.

Usage
-----
    python3 analytic_optimum.py --csv data/RMC_7977_assisted_GTP_hydrolysis.csv \
        [--code-dir .] [--r0 1.07] [--dps 50] [--verify-closed-form] \
        [--compare output/fitted_rate_constants.csv ...] [--digit-scan]

`--code-dir` is the directory holding the module that reads the digitized CSV
and builds the simulator. In this repository that is `make_figure.py`; the name
is detected rather than assumed, so the script also runs against trees where the
same two functions live in `fit_hydrolysis.py`. Override with `--reader-module`.

`--verify-closed-form` evaluates the closed form against the repository's own
PySB simulator at the fitted value and refuses to report if they disagree. That
check is the reason this script imports the reader module instead of carrying
its own copy of the data reader and the model: a private copy would verify the
closed form against itself, which is no check at all.
"""
import argparse
import csv
import os
import sys


def import_reader(code_dir, explicit=None):
    """Import the module exposing load_digitized_data / VariantSimulator."""
    sys.path.insert(0, code_dir)
    names = [explicit] if explicit else ["make_figure", "fit_hydrolysis"]
    tried = []
    for name in names:
        if not name:
            continue
        if not os.path.exists(os.path.join(code_dir, name + ".py")):
            tried.append(f"{name}.py (absent)")
            continue
        mod = __import__(name)
        missing = [a for a in ("load_digitized_data", "VariantSimulator")
                   if not hasattr(mod, a)]
        if missing:
            tried.append(f"{name}.py (no {', '.join(missing)})")
            continue
        return mod, name
    raise SystemExit(
        f"no reader module found in {code_dir!r}; tried: {', '.join(tried) or 'nothing'}.\n"
        f"Pass --code-dir at the folder holding make_figure.py, or name the module "
        f"with --reader-module.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="digitized trace CSV (default: data/ beside this script)")
    ap.add_argument("--code-dir", default=None,
                    help="directory holding make_figure.py (default: beside this script)")
    ap.add_argument("--reader-module", default=None,
                    help="module basename to import instead of auto-detecting")
    ap.add_argument("--r0", default="1.07")
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--verify-closed-form", action="store_true",
                    help="check the closed form against the PySB simulator "
                         "(needs pysb and BNGPATH)")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="fitted_rate_constants.csv files to score against the optimum")
    ap.add_argument("--digit-scan", action="store_true",
                    help="sweep the solver's starting point over x0.5..x2 and report "
                         "how many distinct 3-significant-figure values come back")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = a.code_dir or here
    csv_path = a.csv or os.path.join(here, "data", "RMC_7977_assisted_GTP_hydrolysis.csv")

    import numpy as np
    from mpmath import mp, mpf, exp as mexp
    mp.dps = a.dps
    R0 = mpf(a.r0)

    reader, reader_name = import_reader(code_dir, a.reader_module)
    data = reader.load_digitized_data(csv_path)

    def dssr(k, t, y):
        k = mpf(k)
        s = mpf(0)
        for ti, yi in zip(t, y):
            ti = mpf(float(ti)); yi = mpf(float(yi)); e = mexp(-k * ti)
            s += 2 * (yi - R0 * (1 - e)) * (-R0 * ti * e)
        return s

    def ssr(k, t, y):
        k = mpf(k)
        return sum((mpf(float(yi)) - R0 * (1 - mexp(-k * mpf(float(ti))))) ** 2
                   for ti, yi in zip(t, y))

    def argmin(t, y, k0):
        """Bisection on dSSR/dk. Bracketed, so it cannot silently fail to
        converge the way a secant solver can."""
        lo, hi = mpf(repr(float(k0))) * mpf("0.5"), mpf(repr(float(k0))) * mpf("2")
        n = 0
        while dssr(lo, t, y) >= 0 and n < 60:
            lo /= 2; n += 1
        while dssr(hi, t, y) <= 0 and n < 120:
            hi *= 2; n += 1
        if not (dssr(lo, t, y) < 0 < dssr(hi, t, y)):
            raise SystemExit("could not bracket the minimum; check R0 and the data")
        for _ in range(4 * a.dps + 40):
            m = (lo + hi) / 2
            if dssr(m, t, y) < 0:
                lo = m
            else:
                hi = m
        return (lo + hi) / 2

    # each bisection is seeded from a coarse grid, so no previously stored value
    # is needed to start it
    truth = {}
    for v, (t, y) in data.items():
        grid = np.geomspace(1e-5, 1.0, 200)
        k0 = float(grid[int(np.argmin([float(ssr(k, t, y)) for k in grid]))])
        truth[v] = argmin(t, y, k0)

    print(f"# exact least-squares optimum, R0={a.r0}, Pi_0=0, mp.dps={a.dps}")
    print(f"# data: {os.path.relpath(csv_path, here)}   "
          f"reader: {reader_name}.load_digitized_data")
    print(f"{'variant':>8} {'k_exact /min':>26} {'k_exact /s':>26} {'n':>4}")
    for v, ks in truth.items():
        print(f"{v:>8} {mp.nstr(ks, 18):>26} {mp.nstr(ks / 60, 18):>26} "
              f"{len(data[v][0]):>4}")

    if a.verify_closed_form:
        print("\n# closed form vs PySB simulator at k_exact (max |diff|, uM)")
        ok = True
        for v, ks in truth.items():
            t, y = data[v]
            y_ode = reader.VariantSimulator(t).po4(float(ks), float(a.r0))
            y_cf = np.array([float(R0 * (1 - mexp(-ks * mpf(float(ti))))) for ti in t])
            d = float(np.max(np.abs(y_ode - y_cf)))
            flag = "" if d < 1e-6 else "  <-- CLOSED FORM DISAGREES"
            if d >= 1e-6:
                ok = False
            print(f"{v:>8} {d:>12.3e}{flag}")
        if not ok:
            raise SystemExit("closed form does not match the simulator; "
                             "the model is no longer a single irreversible "
                             "first-order step with Pi_0 = 0")

    if a.digit_scan:
        from decimal import Decimal, Context
        print("\n# starting-point sweep, x0.5 .. x2.0, 21 seeds")
        print(f"{'variant':>8} {'distinct 3 s.f.':>16}  values")
        for v, (t, y) in data.items():
            seen = set()
            for mu in np.geomspace(0.5, 2.0, 21):
                k = argmin(t, y, float(truth[v]) * float(mu)) / 60
                seen.add(str(Decimal(Context(prec=3).create_decimal(
                    Decimal(mp.nstr(k, 17))))))
            print(f"{v:>8} {len(seen):>16}  {sorted(seen)}")

    for path in a.compare:
        print(f"\n# {path} scored against the exact optimum")
        print(f"{'variant':>8} {'k_file /min':>24} {'rel err vs exact':>18}")
        with open(path) as fh:
            for r in csv.DictReader(fh):
                v = r["variant"]
                if v not in truth:
                    continue
                kf = mpf(repr(float(r["k_hyd_app_per_min"])))
                print(f"{v:>8} {float(kf):>24.17g} "
                      f"{float(abs(kf - truth[v]) / truth[v]):>18.3e}")


if __name__ == "__main__":
    main()
