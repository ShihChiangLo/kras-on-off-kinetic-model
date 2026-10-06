"""PySB model of the RAS nucleotide cycle with a wild-type and a mutant RAS pool.

Reactions R1-R11 of S2 Table, written once for each pool. With
``with_amg510=True`` the model also carries AMG 510, which binds mutant
RAS-GDP irreversibly with a single second-order constant (main text, Fig 5).
"""
from pysb import Initial, Model, Monomer, Observable, Parameter, Rule
from pysb.core import SelfExporter

from .parameters import GEF_TOTAL, K_DRUG, VARIANTS


def _pool_rules(prefix, nucleotide_free, ras, effector, gef, gap, p):
    """Add R1-R11 for one RAS pool; ``prefix`` is "" (wild-type) or "mu" (mutant)."""
    k = lambda name: p[prefix + name]
    Rule(f"{prefix}RAS_GDP_binding", nucleotide_free(b=None) | ras(b=None, state="D"),
         k("ka_GDP"), k("kd_GDP"))
    Rule(f"{prefix}RAS_GTP_binding", nucleotide_free(b=None) | ras(b=None, state="T"),
         k("ka_GTP"), k("kd_GTP"))
    Rule(f"{prefix}RAS_intrinsic_hydrolysis", ras(b=None, state="T") >> ras(b=None, state="D"),
         k("khyd"))
    Rule(f"{prefix}RAS_effector_binding",
         ras(b=None, state="T") + effector(b=None) | ras(b=1, state="T") % effector(b=1),
         k("ka_Eff"), k("kd_Eff"))
    Rule(f"{prefix}RAS_effector_complex_hydrolysis",
         ras(b=1, state="T") % effector(b=1) >> ras(b=None, state="D") + effector(b=None),
         k("khyd"))
    Rule(f"{prefix}RAS_GDP_GEF_binding",
         ras(b=None, state="D") + gef(b=None) | ras(b=1, state="D") % gef(b=1),
         k("kDa_GEF"), k("kDd_GEF"))
    Rule(f"{prefix}RAS_GEF_exchange_GDP_to_GTP",
         ras(b=1, state="D") % gef(b=1) >> ras(b=None, state="T") + gef(b=None),
         k("kcat_GDP"))
    Rule(f"{prefix}RAS_GTP_GEF_binding",
         ras(b=None, state="T") + gef(b=None) | ras(b=1, state="T") % gef(b=1),
         k("kTa_GEF"), k("kTd_GEF"))
    Rule(f"{prefix}RAS_GEF_exchange_GTP_to_GDP",
         ras(b=1, state="T") % gef(b=1) >> ras(b=None, state="D") + gef(b=None),
         k("kcat_GTP"))
    Rule(f"{prefix}RAS_GTP_GAP_binding",
         ras(b=None, state="T") + gap(b=None) | ras(b=1, state="T") % gap(b=1),
         k("kTa_GAP"), k("kTd_GAP"))
    Rule(f"{prefix}RAS_GAP_hydrolysis",
         ras(b=1, state="T") % gap(b=1) >> ras(b=None, state="D") + gap(b=None),
         k("kcat_GAP"))


def build(variant, condition, with_amg510=False, intrinsic_hydrolysis=True):
    """Return the model for ``variant`` ("WT", "G12D", "G12V" or "G12C").

    ``condition`` is ([RAS], [effector], [GAP]) in M; both RAS pools start
    nucleotide-free at [RAS]. ``intrinsic_hydrolysis=False`` sets the mutant
    pool's intrinsic hydrolysis rate to zero (the lighter curves of Fig 5).
    """
    rates = dict(VARIANTS[variant])
    if not intrinsic_hydrolysis:
        rates["mukhyd"] = 0.0
    ras0, effector0, gap0 = condition

    SelfExporter.cleanup()
    model = Model(name=f"ras_switch_{variant}")
    GEF = Monomer("GEF", ["b"])
    GAP = Monomer("GAP", ["b"])
    Effector = Monomer("Effector", ["b"])
    RASn = Monomer("RASn", ["b"])
    RAS = Monomer("RAS", ["b", "state"], {"state": ["T", "D"]})
    muRASn = Monomer("muRASn", ["b"])
    muRAS = Monomer("muRAS", ["b", "state"], {"state": ["T", "D"]})
    p = {name: Parameter(name, value) for name, value in rates.items()}

    _pool_rules("", RASn, RAS, Effector, GEF, GAP, p)
    _pool_rules("mu", muRASn, muRAS, Effector, GEF, GAP, p)

    Initial(RASn(b=None), Parameter("RAS_0", ras0))
    Initial(muRASn(b=None), Parameter("muRAS_0", ras0))
    Initial(Effector(b=None), Parameter("Effector_0", effector0))
    Initial(GEF(b=None), Parameter("GEF_0", GEF_TOTAL))
    Initial(GAP(b=None), Parameter("GAP_0", gap0))

    if with_amg510:
        Drug = Monomer("AMG510", ["b"])
        Rule("AMG510_binds_mutant_RAS_GDP",
             muRAS(b=None, state="D") + Drug(b=None) >> muRAS(b=1, state="D") % Drug(b=1),
             Parameter("k_drug", K_DRUG))
        Initial(Drug(b=None), Parameter("AMG510_0", 0.0))

    for prefix, free, ras in (("", RASn, RAS), ("mu", muRASn, muRAS)):
        Observable(f"{prefix}RAS_free", free(b=None))
        Observable(f"{prefix}RAS_GTP", ras(b=None, state="T"))
        Observable(f"{prefix}RAS_GTP_Effector", ras(b=1, state="T") % Effector(b=1))
        Observable(f"{prefix}RAS_GTP_GEF", ras(b=1, state="T") % GEF(b=1))
        Observable(f"{prefix}RAS_GTP_GAP", ras(b=1, state="T") % GAP(b=1))
        Observable(f"{prefix}RAS_GDP", ras(b=None, state="D"))
        Observable(f"{prefix}RAS_GDP_GEF", ras(b=1, state="D") % GEF(b=1))
    return model


ACTIVE = ["RAS_GTP", "RAS_GTP_Effector", "RAS_GTP_GEF", "RAS_GTP_GAP"]
INACTIVE = ["RAS_GDP", "RAS_GDP_GEF"]
FREE = ["RAS_free"]
