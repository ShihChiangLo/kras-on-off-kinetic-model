# kras-on-off-kinetic-model

Code and data for the figures of *Kinetic modeling reveals distinct mechanisms of
RAS(ON)/RAS(OFF) inhibitor combinations*, by Shih-Chiang Lo and Yingkai Zhang.

## Contents

| Folder | What it holds |
|---|---|
| `ras_switch/` | The RAS nucleotide cycle, with a wild-type and a mutant RAS pool, and the simulations behind Figs 2-5 |
| `two_arm/` | The same cycle with a RAS(ON) and a RAS(OFF) inhibitor added, and the machinery the combination figures share |
| `figures/<figure>/` | For each figure: the table it is drawn from (`data/`), the script that draws it (`make_figure.py`) and, where the table is simulated here, the script that produces it (`simulate.py`) |
| `make_all_figures.py` | Draws every figure |

Figs 1 and 6 are schematics and are not included. The folders `figS5_sstar` and
`figS6_ki_sensitivity` hold the figures that appear inside S5 Text and S6 Text;
the article has no S5 or S6 Fig.

## Install

Python 3.11 was used. In a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PySB calls BioNetGen to generate the reaction network, and BioNetGen's own
driver is a Perl script, so Perl has to be on the machine as well. `BNGPATH`
has to point at the copy of BioNetGen that the `bionetgen` package installs,
which is `bng-linux`, `bng-mac` or `bng-win` inside that package depending on
the platform. In a POSIX shell:

```bash
export BNGPATH=$(python3 -c "import bionetgen,os;print(os.path.dirname(bionetgen.__file__)+'/bng-linux')")
```

On Windows the same path is set with `set` or `$env:BNGPATH` instead.

Every figure drawn here except S1 Fig is set in Arial, one of the typefaces
PLOS asks for, and Arial is not distributed here. It ships with macOS and
Windows; on Linux it has to be installed, for example from the Microsoft core
fonts package, whose Arial is an older release than the one named under
Reproducibility. If matplotlib was used on the machine before Arial was
installed, delete its font cache (the `fontlist-*.json` file in the directory
that `python3 -c "import matplotlib; print(matplotlib.get_cachedir())"` prints)
so that it finds the new font. The plotting scripts of those figures stop with
an error if they cannot find Arial, rather than drawing in another typeface.
S1 Fig is set in DejaVu Sans, which comes with matplotlib.

## Draw the figures

```bash
python3 make_all_figures.py              # every figure
python3 make_all_figures.py --skip-slow  # every figure except S1 Fig
python3 make_all_figures.py --only fig05
```

Each figure is written to `figures/<figure>/output/`. With the package versions
in `requirements.txt` and the Arial files named under Reproducibility, on Linux
x86-64 with Python 3.11, every PNG is byte-identical to the file used in the
article. On other platforms, anti-aliased edges can differ by a few pixels.

S1 Fig fits a one-reaction PySB model to each digitized hydrolysis trace, with a
200-replicate bootstrap, and takes about five minutes. The other figures only
read their tables.

## Re-run the simulations

Every table under `figures/<figure>/data/` can be recomputed from the models in
this repository, with one exception: the digitized trace of S1 Fig is data read
off a published figure rather than something this code computes (see Data
sources). Each simulation script writes into a `results/` folder under the
directory it is run from, so run it from its own figure's folder;
`--to-figures` also copies the tables into the `data/` folder the figure is
drawn from. A full rebuild is the simulation scripts followed by
`make_all_figures.py`.

### Figs 2-5, the RAS switch alone

```bash
python3 -m ras_switch.simulate --to-figures
```

Each simulation starts both RAS pools nucleotide-free and integrates to
t = 10<sup>4</sup> s, under each of the nine ([RAS], [effector], [GAP]) conditions of
S3 Table, with [GEF] = 2 × 10<sup>-10</sup> M. Figs 2-4 read the last time point.
For Fig 5, AMG 510 is then added to that state and the run continues for 60 min.
Figs 2-5 report the mean over the nine conditions, and the error bars are the
sample standard deviation (n − 1 in the denominator). Figs 2-4 take under two
minutes; Fig 5 adds about seven. `--skip-dose-response` leaves out Fig 5.

### Figs 7-9, S2 Fig and the S5 and S6 Text figures, both inhibitors

```bash
cd figures/fig09_g12c_dosing && python3 simulate.py --to-figures
```

and likewise in `fig07_feeding`, `fig08_variants`, `figS2_band`, `figS5_sstar`
and `figS6_ki_sensitivity`. Each script takes `--help`, and each prints what it
is doing and how long it took. Measured here, on two cores:

| Figure | What it computes | Time |
|---|---|---|
| Fig 7 | one design point, then a 13 × 13 dose grid for two alleles | 1.5 min |
| Fig 8 | the same grid at equal concentrations instead of equal dose multiples | 1.5 min |
| Fig 9 | a dose grid, a bisection for the dose interval, and the capture time courses | 2.5 min |
| S6 Text figure | a dose scan at each of eighteen values of K_I, in two scenarios | 2.5 min |
| S2 Fig | a threshold solved at each of the sixteen combinations of the two design axes, for each of the two alleles | 26 min |
| S5 Text figure | a boundary solved by bisection, each step of which is two thresholds | 55 min |

The last two are long because each answer is a root found by bisection, and for
the S5 Text figure each step of the outer bisection is itself two inner
bisections, so one number costs a few hundred simulations. `figS5_sstar` takes
`--table` so its eight tables can be split across processes, and appends to a
cache in its output folder so an interrupted run resumes.

## Solver

Every simulation of the RAS network uses

```python
ScipyOdeSimulator(model, tspan=tspan, integrator="lsoda",
                  integrator_options={"rtol": 1e-8, "atol": 1e-16})
```

The network is stiff (the GAP association constant is 1.3 × 10<sup>10</sup>
M<sup>-1</sup> s<sup>-1</sup> against a GAP pool of order 10<sup>-11</sup> M).
Under PySB's default integrator for it, `vode` with backward differentiation,
the simulations return NaN or wrong values without raising an error, so the
integrator is set explicitly.

The simulations behind Figs 2-5 additionally cap lsoda at 200,000 internal
steps (`"mxstep": 200000`). The two-inhibitor simulations leave that cap at the
PySB default of 2<sup>31</sup> − 1, which is where their results were computed.

The S1 Fig fit is a single first-order reaction and uses the default integrator
with `rtol` 10<sup>-10</sup> and `atol` 10<sup>-12</sup>.

## Reproducibility

Re-running the simulation scripts reproduces every table in `data/` to the last
digit, and redrawing then reproduces every PNG byte for byte. That was verified
by deleting the tables and the figures and rebuilding both from the models.

Three things that exactness depends on, stated because they are easy to break:

- The platform. Byte-identical output is a property of the code together with
  the package versions in `requirements.txt`, Python 3.11 and Linux x86-64.
  The same scripts on a different CPU architecture produce figures that differ
  in a few anti-aliased pixels, and a different BLAS or libm build can move a
  stiff endpoint by about 10<sup>-8</sup> relative, which is the solver's own
  error tolerance.
- The font. The PNGs used in the article were drawn with the Arial files that
  ship with macOS: `Arial.ttf` and `Arial Bold.ttf` version 5.01, and
  `Arial Italic.ttf` version 5.00. Another release of Arial can differ in glyph
  outlines or spacing, and then the redrawn figures can differ from the
  published ones in the pixels of their text.
- Three tables are built from an intermediate table on disk rather than from
  the values in memory: `interval_edges.csv` and `on_dose_interval.csv`, both
  built from the re-read `dose_grid.csv` of Fig 9, and `ki_sensitivity.csv`,
  summarised from the dose scan written beside it. That round trip is not
  exact: writing a float is exact but reading it back is correct only to about
  one part in 10<sup>16</sup>, so the last digit or two of those three tables
  carries it. The scripts therefore write and re-read deliberately, and say so
  at the line where they do it, rather than computing in memory and producing
  numbers that differ from the published ones in the last digit.

Nothing in this repository depends on a random seed except the bootstrap of
S1 Fig, which seeds its own generator.

## Data sources

`figures/figS1_hydrolysis_fits/data/RMC_7977_assisted_GTP_hydrolysis.csv` holds
points digitized from Fig 2b of Cuevas-Navarro et al., *Nature* 637, 224-229
(2025). Every other table here is written by the models in `ras_switch/` and
`two_arm/`.

## License

MIT; see `LICENSE`.
