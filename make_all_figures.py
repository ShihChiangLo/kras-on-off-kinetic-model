#!/usr/bin/env python3
"""Draw every figure from the tables in figures/*/data/.

    python3 make_all_figures.py                 all of them (S1 Fig takes about five minutes)
    python3 make_all_figures.py --skip-slow     all but S1 Fig
    python3 make_all_figures.py --only fig07 fig09

Each figures/<name>/make_figure.py reads its own data/ folder and writes to its
own output/ folder. The exit status is non-zero if any figure fails.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"

# folder, label, slow
JOBS = [
    ("fig02_total_validation", "Fig 2", False),
    ("fig03_transfected_validation", "Fig 3", False),
    ("fig04_state_distribution", "Fig 4", False),
    ("fig05_dose_response", "Fig 5", False),
    ("fig07_feeding", "Fig 7", False),
    ("fig08_variants", "Fig 8", False),
    ("fig09_g12c_dosing", "Fig 9", False),
    ("figS1_hydrolysis_fits", "S1 Fig", True),
    ("figS2_band", "S2 Fig", False),
    ("figS5_sstar", "S5 Text figure", False),
    ("figS6_ki_sensitivity", "S6 Text figure", False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-slow", action="store_true", help="skip S1 Fig, which fits a model")
    ap.add_argument("--only", nargs="+", metavar="PREFIX",
                    help="run only the figure folders that start with these prefixes")
    args = ap.parse_args()

    jobs = [j for j in JOBS if not (args.skip_slow and j[2])]
    if args.only:
        jobs = [j for j in jobs if any(j[0].startswith(p) for p in args.only)]
    if not jobs:
        sys.exit("no figure selected")

    failed = []
    for folder, label, _ in jobs:
        t0 = time.time()
        proc = subprocess.run([sys.executable, "make_figure.py"], cwd=FIGURES / folder,
                              text=True, capture_output=True)
        status = "ok" if proc.returncode == 0 else "FAILED"
        print(f"{status:<7}{label:<16}{time.time() - t0:7.1f} s   figures/{folder}/output/")
        if proc.returncode != 0:
            failed.append(label)
            print(proc.stdout + proc.stderr)
    print(f"{len(jobs) - len(failed)} of {len(jobs)} figures drawn")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
