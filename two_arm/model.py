"""PySB model of the RAS nucleotide cycle with a RAS(ON) and a RAS(OFF) arm.

Both RAS pools of `ras_switch.model` are here, with the same rules and the same
rate constants, and two inhibitors are added on top:

RAS(ON) arm, on both alleles
    The inhibitor binds cyclophilin A, and that binary complex binds RAS-GTP to
    form a tri-complex. The tri-complex withholds RAS-GTP from the effector,
    and it can also release it as RAS-GDP, which is drug-stimulated hydrolysis
    (`catalytic release` below). That second route is the one the combination
    depends on, because its product is what the RAS(OFF) arm binds.

RAS(OFF) arm, on the mutant allele only
    The inhibitor binds the switch-II pocket, which is open only on the
    GDP-loaded protein. A RAS carrying it is frozen there: no GTP loading, no
    GEF, no GAP, no effector and no nucleotide release. Two versions are built,
    `reversible` for the idealized probe of Figs 7 and 8 and `covalent` for
    AMG 510 on KRAS G12C in Fig 9.

No RAS can carry both inhibitors. That is structural, not a rate set to zero:
every RAS(ON) rule requires an empty switch-II pocket and every RAS(OFF) rule
requires no RAS(ON) inhibitor bound. The wild-type allele carries no switch-II
site at all, so the RAS(OFF) arm cannot touch it, and neither inhibitor binds
nucleotide-free RAS.

Two idealizations of the RAS(OFF) probe can be relaxed, which is what S2 Fig
and the S5 Text figure sweep:

    state_selectivity   S = KD(GTP state) / KD(GDP state). Infinite means the
                        probe binds the GDP state only, which is the idealized
                        probe and the only topology Figs 7 and 8 use. Finite S
                        adds a second binding rule, for the GTP state, whose
                        dissociation rate is S times the first one's.
    effector_attenuation
                        alpha_RAF = KD,RAF(drug-bound RAS) / KD,RAF(free RAS).
                        Infinite means a drug-bound RAS cannot bind the
                        effector at all. Finite alpha_RAF adds a rule by which
                        it can, with the dissociation rate multiplied by
                        alpha_RAF; the association rate is shared, so
                        alpha_RAF moves the off-rate only.

and two further conditions, both named in the S5 Text figure caption:

    drug_bound_hydrolysis
                        whether a drug-bound RAS-GTP still hydrolyses its GTP
                        intrinsically. False throughout this work, and the
                        default here for that reason.
    effector_rescue     whether a drug-bound RAS that is also holding the
                        effector can hydrolyse its GTP. False unless stated.

At infinite state selectivity none of those four has anything to act on: the
probe binds the GDP state, the effector binds the GTP state, so no RAS is both
drug-bound and GTP-loaded, and the reaction network and every trajectory come
out the same whichever way they are set. The builder refuses the two
combinations that would hide a contradiction rather than merely be moot.
"""
from pysb import ANY, Model, Monomer, Observable, Parameter, Rule

from .parameters import (CONDITION, CYPA_TOTAL, GEF_TOTAL, K1_ON, K2_ON,
                         KCAT_GAP_WT, KD1, KD2, VARIANTS)

#: Mutant RAS carrying the RAS(OFF) inhibitor, either nucleotide state.
OFF_BOUND = "obsmuRAS_off_any"
#: GDP-loaded mutant RAS with nothing bound: the RAS(OFF) arm's substrate.
ACCESSIBLE_GDP = "obsmuRAS_GDP"
#: Mutant RAS-GTP inside a tri-complex: catalytic release's substrate.
TRI_SUBSTRATE = "obsmuRAS_GTP_Tri"
#: Mutant RAS-GTP holding the effector: the signalling readout.
SIGNAL = "obsmuRAS_GTP_Eff_any"
#: Mutant RAS the covalent inhibitor has reacted with. One inhibitor carries
#: one RAS and the bond is terminal, so the drug-side count is the RAS count.
CAPTURED = "obsOffDrug_covalent"
#: The denominator of every mutant percentage.
MUTANT_TOTAL = ("obsmuRAS_total", "obsmuRASn_total")
#: The denominator of every wild-type percentage.
WILDTYPE_TOTAL = ("obsRAS_total", "obsRASn")

_EFFECTOR_BINDING = "muRAS_GTP_bind_unbind_Effector"
_EFFECTOR_HYDROLYSIS = "muRAS_GTPase_activity_Effector"
_INTRINSIC_HYDROLYSIS = "muRAS_GTPase_activity"
_GTP_OFF_BINDING = "muRAS_GTP_bind_unbind_OffDrug"
_EFFECTOR_BINDING_DRUG_BOUND = "muRAS_GTP_bind_unbind_Effector_drugbound"
#: Parameter the effector attenuation is written into.
ATTENUATION_PARAMETER = "mukd_Eff_drug"
#: Parameter the state selectivity is written into.
SELECTIVITY_PARAMETER = "koff_off_T"


def rates_for(variant, hydrolysis=None):
    """That allele's rate constants, optionally with a different intrinsic
    hydrolysis rate for the mutant pool.

    The mutant intrinsic hydrolysis rate is the only constant this work ever
    replaces, and only for a declared scenario. It is also the constant whose
    mutant values come from more than one source, which S1 Table, Note 2 sets
    out. Replacing it here rather than at simulation time keeps it inside the
    catalytic-release bounds that `catalysis_bounds` enforces.
    """
    rates = dict(VARIANTS[variant])
    if hydrolysis is not None:
        rates["mukhyd"] = float(hydrolysis)
    return rates


def catalysis_bounds(variant, hydrolysis=None):
    """(lower, upper) bound on a non-zero catalytic-release rate, s^-1.

    Lower: that allele's own intrinsic hydrolysis rate. The tri-complex is
    modelled as repairing switch II, and a repair cannot make hydrolysis slower
    than the unrepaired protein. Upper: GAP-stimulated hydrolysis of wild-type
    RAS, which is a ceiling this work sets rather than a reported limit on the
    drug-stimulated rate. It is the fastest hydrolysis anywhere in this model,
    so a swept rate above it is taken as out of range.
    """
    return float(rates_for(variant, hydrolysis)["mukhyd"]), float(KCAT_GAP_WT)


def check_catalysis(values, variant, hydrolysis=None):
    """Raise unless every value is either exactly zero or inside the bounds.

    Zero switches the mechanism off and is always allowed. The bounds are
    enforced when a model is built, so anything swept afterwards has to be
    checked here instead; without this a sweep through an impossible rate would
    produce a smooth and wrong surface with no error.
    """
    lo, hi = catalysis_bounds(variant, hydrolysis)
    for value in values:
        value = float(value)
        if value == 0.0:
            continue
        if not lo <= value <= hi:
            raise ValueError(
                f"catalytic release of {value:g} /s is outside [{lo:g}, "
                f"{hi:g}] for {variant}: it is either below that allele's own "
                f"intrinsic hydrolysis rate or above the ceiling this work "
                f"sets, GAP-stimulated hydrolysis of wild-type RAS. Use "
                f"exactly 0.0 to switch the mechanism off.")
    return True


def covalent_constants(kinact_over_ki, k_i, koff):
    """The covalent arm's rate constants, as a dict, from the efficiency and K_I.

    The inactivation rate and the association rate follow as identities once
    the published second-order efficiency and the binding constant K_I are both
    given. The dissociation rate is inherited rather than measured for this
    compound, so it is passed in rather than derived:

        kinact = (kinact/K_I) * K_I
        kon    = (koff + kinact) / K_I
        kon   >= kinact/K_I, because kinact / (koff + kinact) <= 1

    The last line is a bound rather than a choice: the published efficiency puts
    a floor under kon. Across the K_I range the S6 Text figure sweeps, kon moves
    by 0.12%, and koff is more than three orders of magnitude below kinact, so
    the efficiency and K_I are what the arm rests on.

    A dict is returned rather than a tuple so that two of the constants cannot
    be swapped by a caller.
    """
    efficiency, k_i, koff = float(kinact_over_ki), float(k_i), float(koff)
    if efficiency <= 0.0 or k_i <= 0.0:
        raise ValueError(
            f"kinact/K_I and K_I must both be positive; got {efficiency!r} "
            f"and {k_i!r}")
    kinact = efficiency * k_i
    kon = (koff + kinact) / k_i
    if kon < efficiency:
        raise ValueError(
            f"kon {kon:.6g} is below kinact/K_I {efficiency:.6g}, which the "
            f"two-step mechanism forbids")
    return {"kon": kon, "koff": koff, "kinact": kinact, "K_I": k_i,
            "kinact_over_K_I": kon * kinact / (koff + kinact)}


def _guard_switch_II(rule):
    """Require an empty switch-II pocket on every mutant RAS pattern of a rule.

    Both sides of the rule are patched, and the count is asserted, so a rule
    whose shape changes cannot be silently left half-guarded.
    """
    patched = 0
    for pattern in (rule.reactant_pattern, rule.product_pattern):
        for complex_pattern in pattern.complex_patterns:
            for monomer_pattern in complex_pattern.monomer_patterns:
                if monomer_pattern.monomer.name == "muRAS":
                    monomer_pattern.site_conditions["s"] = None
                    patched += 1
    if patched != 2:
        raise RuntimeError(
            f"guarding {rule.name}: expected 2 mutant RAS patterns, found "
            f"{patched}")


def build(variant, catalytic_release, *, kon_off, koff_off, kinact=None,
          covalent=False, hydrolysis=None, condition=None,
          state_selectivity=float("inf"),
          effector_attenuation=float("inf"),
          drug_bound_hydrolysis=False, effector_rescue=False):
    """Return the two-arm model.

    variant             "G12D", "G12V" or "G12C". The wild-type pool is always
                        present as well, as the competing target it is.
    catalytic_release   the mutant catalytic-release rate, s^-1. Wild-type
                        catalytic release is zero throughout: wild-type RAS was
                        measured not to be stimulated.
    kon_off, koff_off   the RAS(OFF) arm's association and dissociation rates.
    kinact, covalent    `covalent=True` adds the irreversible step at `kinact`
                        and is KRAS G12C only. `covalent=False` is the
                        reversible probe and the step is absent rather than set
                        to zero.
    hydrolysis          replaces the mutant intrinsic hydrolysis rate, for a
                        declared scenario only.
    condition           ([RAS], [effector], [GAP]) in M. Both RAS pools start
                        nucleotide-free at [RAS].

    Both inhibitor concentrations start at zero; a dose is set by overriding
    `OnDrug_0` and `OffDrug_0` at simulation time.
    """
    rates = rates_for(variant, hydrolysis)
    check_catalysis([catalytic_release], variant, hydrolysis)
    ras0, effector0, gap0 = CONDITION if condition is None else condition

    finite_selectivity = state_selectivity != float("inf")
    finite_attenuation = effector_attenuation != float("inf")
    if state_selectivity < 1.0:
        raise ValueError(
            f"state selectivity must be >= 1 (1 = no preference between the "
            f"two nucleotide states), got {state_selectivity}")
    if effector_attenuation < 1.0:
        raise ValueError(
            f"effector attenuation must be >= 1 (1 = the inhibitor does not "
            f"affect effector binding), got {effector_attenuation}")
    if effector_rescue and not finite_attenuation:
        raise ValueError(
            "effector_rescue is meaningless at infinite effector attenuation: "
            "no drug-bound RAS holding the effector exists for that route to "
            "act on")
    if covalent and variant != "G12C":
        raise ValueError(
            f"the covalent inhibitor bonds to Cys12 and has no target in "
            f"{variant}")
    if covalent and kinact is None:
        raise ValueError("covalent=True needs kinact")

    model = Model(_export=False)

    # -- molecules ---------------------------------------------------------
    # Nucleotide-free RAS carries no inhibitor site at all, on either allele:
    # there is nowhere for either bond to attach, so no rule can put a drug on
    # it. `d` is the RAS(ON) inhibitor's site and `s` the switch-II pocket;
    # the wild-type allele never has `s`.
    GEF = Monomer("GEF", ["b"], _export=False)
    GAP = Monomer("GAP", ["b"], _export=False)
    Effector = Monomer("Effector", ["b"], _export=False)
    CypA = Monomer("CypA", ["b"], _export=False)
    OnDrug = Monomer("OnDrug", ["c", "r"], _export=False)
    RASn = Monomer("RASn", ["b"], _export=False)
    RAS = Monomer("RAS", ["b", "d", "state"], {"state": ["T", "D"]},
                  _export=False)
    muRASn = Monomer("muRASn", ["b"], _export=False)
    muRAS = Monomer("muRAS", ["b", "d", "s", "state"],
                    {"state": ["T", "D"]}, _export=False)
    # `cov` separates the reversible encounter complex from the covalent
    # adduct. Only cov="n" can dissociate; cov="y" is terminal.
    OffDrug = Monomer("OffDrug", ["r", "cov"], {"cov": ["n", "y"]},
                      _export=False)
    for monomer in (GEF, GAP, Effector, CypA, OnDrug, RASn, RAS, muRASn,
                    muRAS, OffDrug):
        model.add_component(monomer)

    def parameter(name, value):
        p = Parameter(name, float(value), _export=False)
        model.add_component(p)
        return p

    def rule(name, *args):
        r = Rule(name, *args, _export=False)
        model.add_component(r)
        return r

    def observable(name, pattern):
        o = Observable(name, pattern, _export=False)
        model.add_component(o)
        return o

    # -- rate constants ----------------------------------------------------
    switch = {}
    for prefix in ("", "mu"):
        for name in ("kd_GDP", "ka_GDP", "kd_GTP", "ka_GTP", "kDa_GEF",
                     "kDd_GEF", "kcat_GDP", "kTa_GEF", "kTd_GEF", "kcat_GTP",
                     "kTa_GAP", "kTd_GAP", "kcat_GAP", "khyd", "ka_Eff",
                     "kd_Eff"):
            switch[prefix + name] = parameter(prefix + name,
                                              rates[prefix + name])

    k1_on = parameter("k1_on", K1_ON)
    k1_off = parameter("k1_off", KD1 * K1_ON)
    k2_on = parameter("k2_on", K2_ON)
    k2_off = parameter("k2_off", KD2["WT"] * K2_ON)
    muk2_on = parameter("muk2_on", K2_ON)
    muk2_off = parameter("muk2_off", KD2[variant] * K2_ON)

    off_on = parameter("koff_on", kon_off)
    off_off = parameter("koff_off", koff_off)
    # The irreversible step exists only for the covalent inhibitor. There is no
    # parameter for it otherwise, so a reversible probe cannot be turned
    # covalent by setting a number.
    off_inact = parameter("kinact", kinact) if covalent else None

    # -- site conditions ---------------------------------------------------
    # `d=None` or `s=None` on a rule means the rule does not fire on a RAS that
    # already carries that inhibitor. Omitting the site makes it a wildcard.
    no_on_drug = {"d": None}
    no_off_drug = {"s": None}
    # Finite effector attenuation is the one case where a drug-bound RAS may
    # bind the effector, so the guard comes off the effector rules and the
    # drug-bound branch gets a rule of its own below.
    effector_guard = {} if finite_attenuation else {"s": None}

    # -- the switch, wild-type allele -------------------------------------
    rule("RAS_bind_unbind_GDP",
         RASn(b=None) | RAS(b=None, d=None, state="D"),
         switch["ka_GDP"], switch["kd_GDP"])
    rule("RAS_bind_unbind_GTP",
         RASn(b=None) | RAS(b=None, d=None, state="T"),
         switch["ka_GTP"], switch["kd_GTP"])
    rule("RAS_GTPase_activity",
         RAS(b=None, state="T", d=None) >> RAS(b=None, state="D", d=None),
         switch["khyd"])
    rule("RAS_GTP_bind_unbind_Effector",
         RAS(b=None, state="T", d=None) + Effector(b=None)
         | RAS(b=1, state="T", d=None) % Effector(b=1),
         switch["ka_Eff"], switch["kd_Eff"])
    rule("RAS_GTPase_activity_Effector",
         RAS(b=1, state="T", d=None) % Effector(b=1)
         >> RAS(b=None, state="D", d=None) + Effector(b=None),
         switch["khyd"])
    rule("RAS_GDP_bind_unbind_GEF",
         RAS(b=None, state="D", d=None) + GEF(b=None)
         | RAS(b=1, state="D", d=None) % GEF(b=1),
         switch["kDa_GEF"], switch["kDd_GEF"])
    rule("RAS_GDP_to_GTP_by_GEF",
         RAS(b=1, state="D", d=None) % GEF(b=1)
         >> RAS(b=None, state="T", d=None) + GEF(b=None),
         switch["kcat_GDP"])
    rule("RAS_GTP_bind_unbind_GEF",
         RAS(b=None, state="T", d=None) + GEF(b=None)
         | RAS(b=1, state="T", d=None) % GEF(b=1),
         switch["kTa_GEF"], switch["kTd_GEF"])
    rule("RAS_GTP_to_GDP_by_GEF",
         RAS(b=1, state="T", d=None) % GEF(b=1)
         >> RAS(b=None, state="D", d=None) + GEF(b=None),
         switch["kcat_GTP"])
    rule("RAS_GTP_bind_unbind_GAP",
         RAS(b=None, state="T", d=None) + GAP(b=None)
         | RAS(b=1, state="T", d=None) % GAP(b=1),
         switch["kTa_GAP"], switch["kTd_GAP"])
    rule("RAS_GTP_to_GDP_by_GAP",
         RAS(b=1, state="T", d=None) % GAP(b=1)
         >> RAS(b=None, state="D", d=None) + GAP(b=None),
         switch["kcat_GAP"])

    # -- the switch, mutant allele ----------------------------------------
    # Nucleotide loading fires only on an unmodified RAS: `s=None` on these two
    # rules is what makes the RAS(OFF) inhibitor an inhibitor, because a RAS
    # frozen in the GDP state cannot reload with GTP.
    rule("muRAS_bind_unbind_GDP",
         muRASn(b=None) | muRAS(b=None, d=None, s=None, state="D"),
         switch["muka_GDP"], switch["mukd_GDP"])
    rule("muRAS_bind_unbind_GTP",
         muRASn(b=None) | muRAS(b=None, d=None, s=None, state="T"),
         switch["muka_GTP"], switch["mukd_GTP"])
    # Intrinsic hydrolysis carries no switch-II condition here: the adduct is
    # not assumed to touch the catalytic machinery. It is reachable only if a
    # drug-bound RAS can be GTP-loaded, which is what `drug_bound_hydrolysis`
    # decides below.
    rule(_INTRINSIC_HYDROLYSIS,
         muRAS(b=None, state="T", d=None)
         >> muRAS(b=None, state="D", d=None),
         switch["mukhyd"])
    rule(_EFFECTOR_BINDING,
         muRAS(b=None, state="T", d=None, **effector_guard)
         + Effector(b=None)
         | muRAS(b=1, state="T", d=None, **effector_guard) % Effector(b=1),
         switch["muka_Eff"], switch["mukd_Eff"])
    rule(_EFFECTOR_HYDROLYSIS,
         muRAS(b=1, state="T", d=None, **effector_guard) % Effector(b=1)
         >> muRAS(b=None, state="D", d=None, **effector_guard)
         + Effector(b=None),
         switch["mukhyd"])
    rule("muRAS_GDP_bind_unbind_GEF",
         muRAS(b=None, state="D", d=None, s=None) + GEF(b=None)
         | muRAS(b=1, state="D", d=None, s=None) % GEF(b=1),
         switch["mukDa_GEF"], switch["mukDd_GEF"])
    rule("muRAS_GDP_to_GTP_by_GEF",
         muRAS(b=1, state="D", d=None, s=None) % GEF(b=1)
         >> muRAS(b=None, state="T", d=None, s=None) + GEF(b=None),
         switch["mukcat_GDP"])
    rule("muRAS_GTP_bind_unbind_GEF",
         muRAS(b=None, state="T", d=None, s=None) + GEF(b=None)
         | muRAS(b=1, state="T", d=None, s=None) % GEF(b=1),
         switch["mukTa_GEF"], switch["mukTd_GEF"])
    rule("muRAS_GTP_to_GDP_by_GEF",
         muRAS(b=1, state="T", d=None, s=None) % GEF(b=1)
         >> muRAS(b=None, state="D", d=None, s=None) + GEF(b=None),
         switch["mukcat_GTP"])
    rule("muRAS_GTP_bind_unbind_GAP",
         muRAS(b=None, state="T", d=None, s=None) + GAP(b=None)
         | muRAS(b=1, state="T", d=None, s=None) % GAP(b=1),
         switch["mukTa_GAP"], switch["mukTd_GAP"])
    rule("muRAS_GTP_to_GDP_by_GAP",
         muRAS(b=1, state="T", d=None, s=None) % GAP(b=1)
         >> muRAS(b=None, state="D", d=None, s=None) + GAP(b=None),
         switch["mukcat_GAP"])

    # -- RAS(ON) arm -------------------------------------------------------
    rule("OnDrug_bind_unbind_CypA",
         OnDrug(c=None, r=None) + CypA(b=None)
         | OnDrug(c=1, r=None) % CypA(b=1), k1_on, k1_off)
    rule("Binary_bind_unbind_RAS",
         OnDrug(c=1, r=None) % CypA(b=1) + RAS(b=None, d=None, state="T")
         | OnDrug(c=1, r=2) % CypA(b=1) % RAS(b=None, d=2, state="T"),
         k2_on, k2_off)
    rule("Binary_bind_unbind_muRAS",
         OnDrug(c=1, r=None) % CypA(b=1)
         + muRAS(b=None, d=None, state="T", s=None)
         | OnDrug(c=1, r=2) % CypA(b=1)
         % muRAS(b=None, d=2, state="T", s=None), muk2_on, muk2_off)

    # Catalytic release: the tri-complex hydrolyses the RAS-GTP it holds and
    # releases RAS-GDP, keeping the binary complex intact. This is the only
    # route by which the RAS(ON) inhibitor produces RAS-GDP instead of merely
    # withholding RAS-GTP, and the product is exactly what the RAS(OFF) arm
    # binds, which is why the rate decides whether the two arms compete or
    # feed each other.
    #
    # Both rules are built even when the rate is zero. The mechanism is
    # reported in the literature and it is the rate that is unmeasured, so the
    # model says "rate unknown" rather than "mechanism absent", and a value
    # supplied later changes no rule and no species.
    #
    # The mutant rule carries the same switch-II guard as the binding rule
    # above, so a RAS already carrying the RAS(OFF) inhibitor is not a
    # substrate: the two arms stay exclusive through the catalytic step and not
    # only through binding.
    parameter("kcat_TCI", 0.0)
    rule("Tri_catalytic_release",
         OnDrug(c=1, r=2) % CypA(b=1) % RAS(b=None, d=2, state="T")
         >> OnDrug(c=1, r=None) % CypA(b=1)
         + RAS(b=None, d=None, state="D"),
         model.parameters["kcat_TCI"])
    parameter("mukcat_TCI", catalytic_release)
    rule("muTri_catalytic_release",
         OnDrug(c=1, r=2) % CypA(b=1)
         % muRAS(b=None, d=2, state="T", s=None)
         >> OnDrug(c=1, r=None) % CypA(b=1)
         + muRAS(b=None, d=None, state="D", s=None),
         model.parameters["mukcat_TCI"])

    # -- RAS(OFF) arm, mutant allele only ---------------------------------
    # `d=None` on every binding rule is the other half of the exclusivity: a
    # RAS already carrying the RAS(ON) inhibitor is not a substrate. It is what
    # keeps a finite state selectivity, which puts this inhibitor on the
    # RAS(ON) inhibitor's own substrate, from building a RAS with both.
    rule("muRAS_GDP_bind_unbind_OffDrug",
         muRAS(b=None, s=None, state="D", d=None) + OffDrug(r=None, cov="n")
         | muRAS(b=None, s=1, state="D", d=None) % OffDrug(r=1, cov="n"),
         off_on, off_off)
    if covalent:
        rule("muRAS_GDP_OffDrug_inactivation",
             muRAS(s=1, state="D") % OffDrug(r=1, cov="n")
             >> muRAS(s=1, state="D") % OffDrug(r=1, cov="y"), off_inact)
    if finite_selectivity:
        # The GTP-state branch. Its dissociation rate is replaced below by
        # `koff_off_T` = S x koff_off, so S = 1 means no preference between the
        # two nucleotide states and S -> inf recovers the GDP-only probe.
        rule(_GTP_OFF_BINDING,
             muRAS(b=None, s=None, state="T", d=None)
             + OffDrug(r=None, cov="n")
             | muRAS(b=None, s=1, state="T", d=None) % OffDrug(r=1, cov="n"),
             off_on, off_off)
        if covalent:
            rule("muRAS_GTP_OffDrug_inactivation",
                 muRAS(s=1, state="T") % OffDrug(r=1, cov="n")
                 >> muRAS(s=1, state="T") % OffDrug(r=1, cov="y"), off_inact)

    # -- initial conditions ------------------------------------------------
    for pattern, (name, value) in (
            (RASn(b=None), ("RASn_0", ras0)),
            (muRASn(b=None), ("muRASn_0", ras0)),
            (muRAS(b=None, d=None, s=None, state="D"), ("muRAS_GDP_0", 0.0)),
            (Effector(b=None), ("Effector_0", effector0)),
            (GAP(b=None), ("GAP_0", gap0)),
            (GEF(b=None), ("GEF_0", GEF_TOTAL)),
            (CypA(b=None), ("CypA_0", CYPA_TOTAL)),
            (OnDrug(c=None, r=None), ("OnDrug_0", 0.0)),
            (OffDrug(r=None, cov="n"), ("OffDrug_0", 0.0))):
        model.initial(pattern, parameter(name, value))

    # -- observables -------------------------------------------------------
    observable("obsRASn", RASn(b=None))
    observable("obsRAS_total", RAS())
    observable("obsRAS_T_total", RAS(state="T"))
    observable("obsRAS_D_total", RAS(state="D"))
    observable("obsRAS_GTP_Eff_any", RAS(b=1, state="T") % Effector(b=1))
    observable("obsRAS_on_drug_any", RAS(d=ANY))

    observable("obsmuRASn", muRASn(b=None))
    observable("obsmuRASn_total", muRASn())
    observable("obsmuRAS_total", muRAS())
    observable("obsmuRAS_T_total", muRAS(state="T"))
    observable("obsmuRAS_D_total", muRAS(state="D"))
    observable(SIGNAL, muRAS(b=1, state="T") % Effector(b=1))
    observable("obsmuRAS_on_drug_any", muRAS(d=ANY))

    observable("obsRAS_GTP", RAS(b=None, d=None, state="T"))
    observable("obsRAS_GDP", RAS(b=None, d=None, state="D"))
    observable("obsRAS_GTP_Effector",
               RAS(b=1, d=None, state="T") % Effector(b=1))
    observable("obsRAS_GDP_GEF", RAS(b=1, d=None, state="D") % GEF(b=1))
    observable("obsRAS_GTP_GEF", RAS(b=1, d=None, state="T") % GEF(b=1))
    observable("obsRAS_GTP_GAP", RAS(b=1, d=None, state="T") % GAP(b=1))
    observable("obsRAS_GTP_Tri",
               OnDrug(c=2, r=1) % CypA(b=2) % RAS(b=None, d=1, state="T"))

    observable("obsmuRAS_GTP", muRAS(b=None, d=None, s=None, state="T"))
    observable(ACCESSIBLE_GDP, muRAS(b=None, d=None, s=None, state="D"))
    observable("obsmuRAS_GTP_Effector",
               muRAS(b=1, d=None, s=None, state="T") % Effector(b=1))
    observable("obsmuRAS_GDP_GEF",
               muRAS(b=1, d=None, s=None, state="D") % GEF(b=1))
    observable("obsmuRAS_GTP_GEF",
               muRAS(b=1, d=None, s=None, state="T") % GEF(b=1))
    observable("obsmuRAS_GTP_GAP",
               muRAS(b=1, d=None, s=None, state="T") % GAP(b=1))
    observable(TRI_SUBSTRATE,
               OnDrug(c=2, r=1) % CypA(b=2)
               % muRAS(b=None, d=1, s=None, state="T"))

    observable("obsEffector_free", Effector(b=None))
    observable("obsGEF_free", GEF(b=None))
    observable("obsGAP_free", GAP(b=None))
    observable("obsCypA_free", CypA(b=None))
    observable("obsCypA_total", CypA())
    observable("obsOnDrug_free", OnDrug(c=None, r=None))
    observable("obsOnDrug_total", OnDrug())
    observable("obsOnBinary", OnDrug(c=1, r=None) % CypA(b=1))
    observable("obsOnDrug_on_RAS", OnDrug(r=ANY))

    observable("obsOffDrug_free", OffDrug(r=None, cov="n"))
    observable("obsOffDrug_total", OffDrug())
    observable("obsOffDrug_bound_any", OffDrug(r=ANY))
    observable(CAPTURED, OffDrug(r=ANY, cov="y"))
    observable("obsOffDrug_noncovalent", OffDrug(r=ANY, cov="n"))
    observable(OFF_BOUND, muRAS(s=ANY))

    # -- relaxing the two idealizations -----------------------------------
    # Done after the model is complete, and in this order, because each step
    # reads what the one before it produced.
    if finite_selectivity:
        selectivity = parameter(SELECTIVITY_PARAMETER,
                                float(state_selectivity) * float(koff_off))
        model.rules[_GTP_OFF_BINDING].rate_reverse = selectivity
    if finite_attenuation:
        # The guard came off both effector rules when the model was built, so
        # it goes back on the binding rule here: without that, the free and the
        # drug-bound branch would share one rate pair and only alpha_RAF = 1
        # would be expressible. The hydrolysis rule is the rescue route, and it
        # stays open only if that route is declared open.
        _guard_switch_II(model.rules[_EFFECTOR_BINDING])
        if not effector_rescue:
            _guard_switch_II(model.rules[_EFFECTOR_HYDROLYSIS])
        attenuated = parameter(
            ATTENUATION_PARAMETER,
            float(effector_attenuation) * float(rates["mukd_Eff"]))
        rule(_EFFECTOR_BINDING_DRUG_BOUND,
             muRAS(b=None, d=None, s=1, state="T") % OffDrug(r=1, cov="n")
             + Effector(b=None)
             | muRAS(b=2, d=None, s=1, state="T") % OffDrug(r=1, cov="n")
             % Effector(b=2),
             switch["muka_Eff"], attenuated)
    if not drug_bound_hydrolysis:
        _guard_switch_II(model.rules[_INTRINSIC_HYDROLYSIS])

    return model
