"""Integration and readouts shared by the figures the two-arm model draws.

Every figure script in `figures/` builds its own grid of cases and calls into
this module, so the readouts cannot drift between figures. Nothing here writes
a file or chooses a dose; those belong to the figure scripts.

The time grid is zero followed by points spaced evenly in log time, because
every readout in this work is either an endpoint or a time at which a curve
crosses a level, and a log grid resolves both equally well at one minute and
at twenty-four hours.
"""
import warnings

import numpy as np
from pysb.bng import generate_equations
from pysb.simulator import ScipyOdeSimulator

from .parameters import (CONDITION, CYPA_TOTAL, GEF_TOTAL, N_POINTS, SOLVER,
                         T_END)

#: The six arms every dose point of the reversible-probe figures is read at, of
#: which the last two are zero-catalytic-release controls built from the same
#: model: identical binding, identical dose, catalytic release set to exactly
#: zero. Neither is a second compound.
ARMS = ("vehicle", "on_only", "off_only", "combo", "combo_kcat0",
        "on_only_kcat0")


def timespan(t_end=T_END, n_points=N_POINTS, t_start=1e-6):
    """Zero, then `n_points - 1` points spaced evenly in log time."""
    return np.concatenate(
        ([0.0], np.logspace(np.log10(t_start), np.log10(t_end), n_points - 1)))


def compile_model(model, t_end=T_END, n_points=N_POINTS):
    """Return (tspan, simulator) for one model. Compiling is about 0.01 s."""
    tspan = timespan(t_end, n_points)
    return tspan, ScipyOdeSimulator(model, tspan=tspan, **SOLVER)


def initial_overrides(condition=None):
    """The concentrations that define one cellular condition.

    Both RAS pools start nucleotide-free at [RAS]. Passed as overrides rather
    than built into the model so that one compiled model serves every
    condition.
    """
    ras0, effector0, gap0 = CONDITION if condition is None else condition
    return {"RASn_0": ras0, "muRASn_0": ras0, "Effector_0": effector0,
            "GAP_0": gap0}


def dose_scales(model):
    """(RAS(ON), RAS(OFF)) dissociation constants read off the built model, M.

    Reading them back rather than passing them in means a dose expressed as a
    multiple of a dissociation constant cannot drift from the constant the
    model was actually built with.
    """
    p = model.parameters
    return (p["muk2_off"].value / p["muk2_on"].value,
            p["koff_off"].value / p["koff_on"].value)


def integrate(model, tspan, simulator, overrides, chunk=48):
    """Integrate one case per entry of `overrides`; return the trajectories.

    Each entry is a dict of parameter name to value, applied on top of the
    model's own values. An unknown name raises rather than being ignored.
    Returns a list of dicts: every observable as an array over `tspan`, plus
    two solver diagnostics, `min_species` and `n_solver_warnings`. Both are
    properties of the whole batch rather than of the single case: the first is
    the smallest concentration reached by any species of any case in the batch,
    and a negative value there means the integration undershot zero somewhere.

    Cases are integrated in batches only to save Python overhead. A case's
    trajectory does not depend on which batch it was in or on its position in
    it; the two diagnostics above do, because they are batch-wide.
    """
    names = [p.name for p in model.parameters]
    base = np.array([p.value for p in model.parameters], dtype=float)
    observables = [o.name for o in model.observables]
    out = []
    for start in range(0, len(overrides), chunk):
        block = overrides[start:start + chunk]
        values = np.tile(base, (len(block), 1))
        for i, case in enumerate(block):
            for name, value in case.items():
                if name not in names:
                    raise ValueError(f"unknown parameter {name!r}")
                values[i, names.index(name)] = float(value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = simulator.run(param_values=values)
        series = np.asarray(result.observables)
        if series.ndim == 0 or np.asarray(result.species).ndim == 2:
            series = series[None, ...]
        worst = float(np.min(np.asarray(result.species)))
        for i in range(len(block)):
            case = {name: np.asarray(series[i][name], dtype=float)
                    for name in observables}
            case["min_species"] = worst
            case["n_solver_warnings"] = len(caught)
            out.append(case)
    return out


# ---------------------------------------------------------------------------
# percentages and fluxes, all against the pool they belong to
# ---------------------------------------------------------------------------
def mutant_total(g):
    return g["obsmuRAS_total"] + g["obsmuRASn_total"]


def wildtype_total(g):
    return g["obsRAS_total"] + g["obsRASn"]


def series(g, catalytic_release, kon_off):
    """The time-resolved readouts of one trajectory.

    off_bound_pct    mutant RAS carrying the RAS(OFF) inhibitor, % of the pool
    signalling_pct   mutant RAS-GTP holding the effector, % of the pool
    ras_gtp_pct      mutant RAS in the GTP state, % of the pool
    on_trapped_pct   mutant RAS carrying the RAS(ON) inhibitor, % of the pool
    wt_on_trapped_pct   the same for the wild-type pool, which the RAS(ON)
                     inhibitor also binds and the RAS(OFF) inhibitor cannot
    accessible_gdp_M GDP-loaded mutant RAS with nothing bound: exactly the
                     substrate pattern of the RAS(OFF) binding rule
    tri_flux         catalytic release, M/s: its rate times its substrate
    off_assoc_flux   RAS(OFF) association, M/s: its rate times both substrates
    """
    mu = mutant_total(g)
    return {
        "off_bound_pct": 100.0 * g["obsmuRAS_off_any"] / mu,
        "signalling_pct": 100.0 * g["obsmuRAS_GTP_Eff_any"] / mu,
        "ras_gtp_pct": 100.0 * g["obsmuRAS_T_total"] / mu,
        "on_trapped_pct": 100.0 * g["obsmuRAS_on_drug_any"] / mu,
        "wt_on_trapped_pct": 100.0 * g["obsRAS_on_drug_any"] / wildtype_total(g),
        "accessible_gdp_M": g["obsmuRAS_GDP"],
        "tri_flux": catalytic_release * g["obsmuRAS_GTP_Tri"],
        "off_assoc_flux": kon_off * g["obsmuRAS_GDP"] * g["obsOffDrug_free"],
        "free_off_M": g["obsOffDrug_free"],
        "tri_substrate_M": g["obsmuRAS_GTP_Tri"],
    }


def worst_conservation(g, condition=None):
    """Largest relative drift of any conserved total over the trajectory.

    Every moiety in this model is conserved: nothing is synthesised and nothing
    degrades. The largest drift is therefore a direct measure of how well the
    integration held, in the units the tolerances are set in.
    """
    ras0, effector0, gap0 = CONDITION if condition is None else condition
    off0 = float(g["obsOffDrug_free"][0] + g["obsOffDrug_bound_any"][0])
    on0 = float(g["obsOnDrug_total"][0])
    totals = {
        "mutant RAS": (mutant_total(g), ras0),
        "wild-type RAS": (wildtype_total(g), ras0),
        "effector": (g["obsEffector_free"] + g["obsRAS_GTP_Eff_any"]
                     + g["obsmuRAS_GTP_Eff_any"], effector0),
        "GEF": (g["obsGEF_free"] + g["obsRAS_GDP_GEF"] + g["obsRAS_GTP_GEF"]
                + g["obsmuRAS_GDP_GEF"] + g["obsmuRAS_GTP_GEF"], GEF_TOTAL),
        "GAP": (g["obsGAP_free"] + g["obsRAS_GTP_GAP"]
                + g["obsmuRAS_GTP_GAP"], gap0),
        "CypA": (g["obsCypA_total"], CYPA_TOTAL),
        "RAS(ON) inhibitor": (g["obsOnDrug_total"], on0),
        "RAS(OFF) inhibitor": (g["obsOffDrug_free"]
                               + g["obsOffDrug_bound_any"], off0),
    }
    worst, where = 0.0, ""
    for name, (trace, total) in totals.items():
        drift = float(np.max(np.abs(np.asarray(trace, dtype=float) - total)))
        relative = drift / total if total > 0 else drift
        if relative > worst:
            worst, where = relative, name
    return worst, where


# ---------------------------------------------------------------------------
# interaction scores
# ---------------------------------------------------------------------------
def bliss(vehicle, on_only, off_only, combination):
    """Bliss independence on the signalling readout.

    Each arm's effect is the fraction by which it lowers the signal from
    vehicle. Independence predicts the combination's effect from the two single
    arms; the excess is how much better the combination does than that.
    Returns (effect_on, effect_off, effect_combination, predicted, excess).
    """
    effect_on = 1.0 - on_only / vehicle
    effect_off = 1.0 - off_only / vehicle
    effect_combination = 1.0 - combination / vehicle
    predicted = effect_on + effect_off - effect_on * effect_off
    return (effect_on, effect_off, effect_combination, predicted,
            effect_combination - predicted)


def escape_rate(model, tspan, simulator, condition=None):
    """Total first-order rate at which mutant RAS-GTP leaves the GTP state.

    Measured on this model at zero dose and zero catalytic release, as the sum
    of the three routes out of the GTP state divided by the GTP-state pool:
    intrinsic hydrolysis, GEF-mediated exchange and GAP-stimulated hydrolysis.
    Returned as a dict of the three routes plus their total.
    """
    overrides = {**initial_overrides(condition), "OnDrug_0": 0.0,
                 "OffDrug_0": 0.0, "mukcat_TCI": 0.0}
    g = integrate(model, tspan, simulator, [overrides])[0]
    value = {p.name: p.value for p in model.parameters}
    gtp = float(g["obsmuRAS_T_total"][-1])
    routes = {
        "intrinsic": value["mukhyd"] * float(
            g["obsmuRAS_GTP"][-1] + g["obsmuRAS_GTP_Effector"][-1]) / gtp,
        "gef_exchange": value["mukcat_GTP"] * float(
            g["obsmuRAS_GTP_GEF"][-1]) / gtp,
        "gap": value["mukcat_GAP"] * float(g["obsmuRAS_GTP_GAP"][-1]) / gtp,
    }
    routes["total"] = sum(routes.values())
    return routes


# ---------------------------------------------------------------------------
# reading a curve
# ---------------------------------------------------------------------------
def at_time(tspan, curve, t):
    """Value of a curve at time `t`, interpolated in log time.

    The grid is spaced evenly in log time, so interpolating there has the same
    relative accuracy everywhere. The leading zero is dropped rather than
    taking its logarithm.
    """
    x = np.asarray(tspan, dtype=float)
    y = np.asarray(curve, dtype=float)
    if t >= x[-1]:
        return float(y[-1])
    keep = x > 0.0
    return float(np.interp(np.log10(t), np.log10(x[keep]), y[keep]))


def time_at_level(tspan, curve, level):
    """First time a curve reaches `level`, interpolated in log time.

    Not reaching it is a result, returned as NaN rather than raised.
    """
    y = np.asarray(curve, dtype=float)
    if not np.any(y >= level):
        return float("nan")
    i = int(np.argmax(y >= level))
    if i == 0:
        return 0.0
    t0, t1, y0, y1 = tspan[i - 1], tspan[i], y[i - 1], y[i]
    if y1 == y0 or t0 <= 0.0:
        return float(t1)
    f = (level - y0) / (y1 - y0)
    return float(10.0 ** (np.log10(t0) + f * (np.log10(t1) - np.log10(t0))))


# ---------------------------------------------------------------------------
# root finding
# ---------------------------------------------------------------------------
def bisect_log(f, lo, hi, f_lo, f_hi, tol_rel, max_iter, residual=True):
    """Bisect for f = 0 between `lo` and `hi`, halving the ratio each step.

    The quantities this package solves for span decades, so the midpoint is the
    geometric one and the tolerance is on the ratio of the two ends rather than
    on their difference. `f_lo` and `f_hi` are passed in because the caller has
    already evaluated them on its own probe.

    Returns (root, report). A root of None means the two ends have the same
    sign, which is an answer about the function and not a failure. The report
    carries `n_iter`, `converged`, `bracket_ratio` (the width of the last
    bracket as a ratio, whose excess over one is the remaining uncertainty; on
    the convergence path this is the bracket before the final midpoint) and
    `f_root`, the value of f at the returned root.

    `residual=False` skips that last evaluation of f, which costs a full
    simulation for the callers here and is only worth paying for if the number
    is reported.
    """
    if f_lo == 0.0:
        return float(lo), {"n_iter": 0, "converged": True,
                           "bracket_ratio": 1.0, "f_root": 0.0}
    if f_hi == 0.0:
        return float(hi), {"n_iter": 0, "converged": True,
                           "bracket_ratio": 1.0, "f_root": 0.0}
    if f_lo * f_hi > 0.0:
        return None, {"n_iter": 0, "converged": False, "f_root": float("nan"),
                      "bracket_ratio": float(hi) / float(lo),
                      "reason": "no sign change between the ends"}
    a, b, f_a = float(lo), float(hi), float(f_lo)
    n, ratio = 0, float(hi) / float(lo)
    for n in range(1, int(max_iter) + 1):
        mid = float(np.sqrt(a * b))
        f_mid = f(mid)
        ratio = b / a
        if f_mid == 0.0 or (b / a - 1.0) <= tol_rel:
            a = b = mid
            break
        if f_a * f_mid < 0.0:
            b = mid
        else:
            a, f_a = mid, f_mid
    root = float(np.sqrt(a * b))
    return root, {"n_iter": n, "converged": True, "bracket_ratio": ratio,
                  "f_root": float(f(root)) if residual else float("nan")}


def structure(model):
    """(rules, species, reactions, parameters) of the generated network."""
    generate_equations(model)
    return {"n_rules": len(model.rules), "n_species": len(model.species),
            "n_reactions": len(model.reactions),
            "n_parameters": len(model.parameters)}
