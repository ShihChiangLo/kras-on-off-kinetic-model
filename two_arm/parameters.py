"""Constants for the two-arm model: the RAS switch, the RAS(ON) inhibitor and
the RAS(OFF) inhibitor.

The switch rate constants are the ones in `ras_switch.parameters`, imported here
rather than repeated, so the two models in this repository cannot drift apart.
They are listed with their status and source in S1 Table.

The drug-side constants below are listed in S1 Table (the reversible RAS(OFF)
probe and the tri-complex affinities) and in S6 Text and S7 Text (the covalent
KRAS G12C arm). Concentrations are in M, first-order rates in s^-1 and
second-order rates in M^-1 s^-1.
"""
from ras_switch.parameters import CONDITIONS, GEF_TOTAL, VARIANTS

# The single ([RAS], [effector], [GAP]) condition every two-arm result is
# computed at: the middle row of the nine conditions of S3 Table. A drug
# titration is then one curve rather than nine.
CONDITION = tuple(CONDITIONS[4])

# ---------------------------------------------------------------------------
# RAS(ON) inhibitor: a tri-complex that needs cyclophilin A as a co-binder.
#
#   drug + CypA        <-> binary complex            KD1
#   binary + RAS-GTP   <-> tri-complex               KD2, per allele
#
# The two association constants are fixed at round values by this work and the
# dissociation constants carry the affinity. S1 Table lists the dissociation
# constants with their status and source; three of the four KD2 values are
# surface-plasmon-resonance measurements and the KRAS G12C one is not (see
# below).
# ---------------------------------------------------------------------------
# Cyclophilin A is far above the RAS pool here, so the cyclophilin step is not
# rate-limiting. The RAS(ON) inhibitor itself is depleted by binding, which is
# why a dose is reported as a multiple of a dissociation constant rather than
# as an occupancy. The value is the conservative low end of the reported
# cellular range, rounded (S1 Table).
CYPA_TOTAL = 10.0 * 1e-6
K1_ON = 1e7
K2_ON = 1e6
KD1 = 55 * 1e-9
#: The KRAS G12C value is not of a kind with the other three: it is
#: back-calculated from a disruption endpoint rather than measured by surface
#: plasmon resonance. Like the other three, it sets the tri-complex
#: dissociation rate of its allele (KD2 * K2_ON, `two_arm.model`), so it
#: enters every two-arm simulation of KRAS G12C. The KRAS G12C RAS(ON) doses
#: are set as absolute concentrations, never as multiples of this value; it
#: is used to express them as multiples only when they are reported.
KD2 = {"WT": 154 * 1e-9, "G12V": 131 * 1e-9, "G12D": 364 * 1e-9,
       "G12C": 18 * 1e-9}

# The same four constants written as a decimal in molar rather than in
# nanomolar. Two of the four pairs, G12D and G12C, are the same number to
# fifteen digits and differ in the last bit of a double; the other two are
# identical. Because the network is stiff, the solver can turn that last bit
# into a relative difference of about 1e-8 in an endpoint, which is its own
# error tolerance. Figs 7 and 8 scale their dose axes by the decimal spelling,
# so it is kept here: re-running this code then reproduces those tables
# exactly rather than only to within the solver's tolerance. S2 Fig and the
# S5 Text figure read their dose scale back off the built model instead
# (`dose_scales` in `two_arm.simulate`). Fig 9 and the S6 Text figure set
# absolute concentrations, but they report them as multiples of the constant
# as well, and those columns use the decimal spelling too.
KD2_DECIMAL = {"WT": 1.54e-7, "G12V": 1.31e-7, "G12D": 3.64e-7,
               "G12C": 1.8e-8}

# Catalytic release from the tri-complex: drug-stimulated hydrolysis of the
# bound RAS-GTP, after which the binary complex leaves intact. Fitted to the
# digitized phosphate-release traces of S1 Fig; wild-type RAS is not
# stimulated, which is a measured absence rather than an unknown.
KCAT_TCI = {"WT": 0.0, "G12D": 1.44e-3, "G12V": 1.10e-4, "G12C": 1.893e-4}

# Upper bound the model enforces on any catalytic-release rate. It is a ceiling
# this work sets, not a reported limit: GAP-stimulated hydrolysis of wild-type
# RAS is the fastest hydrolysis anywhere in this model, so a swept
# drug-stimulated rate above it is taken as out of range.
KCAT_GAP_WT = 5.4

# ---------------------------------------------------------------------------
# RAS(OFF) inhibitor, reversible probe (Figs 7 and 8, S2 Fig, S5 Text figure).
#
# An idealized compound rather than a real one: it binds the GDP-loaded mutant
# RAS only, reversibly, and freezes it there. Its affinity is set equal to the
# mutant RAS pool so that dose divided by affinity is an occupancy and not a
# depletion titration; the association constant supplies a scale only.
# ---------------------------------------------------------------------------
KD_OFF_GDP = 2.0e-7
KON_OFF = 2.3e7
KOFF_OFF = 4.6

# State selectivity S = KD(GTP state) / KD(GDP state) and effector attenuation
# alpha_RAF = KD,RAF(drug-bound RAS) / KD,RAF(free RAS). Both are design axes,
# swept over the values below; neither has been measured for the compound and
# allele modelled here. S = inf and alpha_RAF = inf are the idealized probe.
S_VALUES = (7.7, 16.0, 182.0, float("inf"))
ALPHA_VALUES = (1.0, 3.0, 10.0, float("inf"))

# ---------------------------------------------------------------------------
# RAS(OFF) inhibitor, covalent (Fig 9 and the S6 Text figure).
#
#   RAS-GDP + I  <-> RAS-GDP:I     kon / koff
#   RAS-GDP:I     -> RAS-GDP-I     kinact, irreversible
#
# The inactivation rate and the association rate follow as identities from the
# published second-order efficiency and the binding constant K_I. The
# dissociation rate is inherited rather than measured for this compound and is
# passed in. `covalent_constants` below does that arithmetic.
# ---------------------------------------------------------------------------
KINACT_OVER_KI = 9900.0
K_I = 2.20e-7
KOFF_COVALENT = 7.16e-7

# The covalent arm is KRAS G12C only: the bond is to Cys12, which no other
# allele here has, and the wild-type pool carries no binding site at all.
COVALENT_VARIANT = "G12C"

# Several intrinsic-hydrolysis values exist for KRAS G12C. The two below are
# the two drug-free arms of the study the catalytic constants are fitted to,
# Cuevas-Navarro et al., Nature 637, 224-229 (2025): `cuevas_dmso` is its
# vehicle control arm and `cuevas_cypa` its cyclophilin-A-only arm. In both of
# them the catalytic constant can exceed the drug-free rate, and both are
# carried through as declared scenarios rather than one of them being chosen.
# The KRAS G12C results of Fig 9 and the S6 Text figure are reported in both.
G12C_HYDROLYSIS = {"cuevas_dmso": 3.25e-5, "cuevas_cypa": 4.633e-5}

# The assay-scenario value of the mutant intrinsic hydrolysis rate of KRAS
# G12V, used for one of the alternative readings of the S5 Text figure. It is a
# declared scenario rather than the value the rest of this work uses, which is
# the one the switch model carries for that allele.
G12V_HYDROLYSIS_ASSAY = 1.033e-5

# ---------------------------------------------------------------------------
# Readouts and integration
# ---------------------------------------------------------------------------
# Dose divided by dissociation constant, for each arm. The reference design
# point is one for both; the high-dose point of the S5 Text figure is a
# thousandfold increase in both.
THETA_REFERENCE = (1.0, 1.0)
THETA_HIGH = (1.0e3, 1.0e3)

T_END = 1.0e4            # s, the horizon of every reversible-probe result
N_POINTS = 401
T_END_COVALENT = 86400.0  # s, 24 h; the covalent arm is read against the clock
N_POINTS_COVALENT = 1201
T_BENEFIT = 21600.0      # s, 6 h: when covalent capture is read
T_COST = 86400.0         # s, 24 h: when wild-type RAS(ON) occupancy is read
CAPTURE_CRITERION = 90.0  # % of the mutant pool

# The two wild-type RAS(ON) occupancies that bound the dose interval of Fig 9.
# They are declared limits for reading the figure, not clinical criteria, and
# every caption that shows them says so.
OCCUPANCY_LIMITS = (1.0, 5.0)

# The state selectivity one clinical-stage RAS(OFF) compound shows across the
# seven KRAS alleles for which both nucleotide states were measured, and the
# median of that spread. S1 Table names the compound and the source. It is a
# spread across alleles rather than a value measured for the allele any
# combination here is aimed at, which is why S is swept over it rather than
# fixed at it.
MEASURED_SELECTIVITY_SPREAD = (7.7, 182.0)
MEASURED_SELECTIVITY_MEDIAN = 16.0

# Root-finding, used for every threshold this package solves. Both solvers
# bisect in log space, so a bracket is a factor rather than a difference.
PROBE_POINTS = 25
ROOT_TOL_REL = 1.0e-9
ROOT_MAX_ITER = 80
BOUNDARY_TOL_REL = 1.0e-4
BOUNDARY_MAX_ITER = 40

# The network is stiff: the GAP association constant is 1.3e10 M^-1 s^-1
# against a GAP pool of order 1e-11 M. Under PySB's default integrator for it
# the simulations return NaN or wrong values without raising, so the integrator
# is set explicitly. The internal step cap is left at the PySB default here,
# which is where every result in this package was computed.
SOLVER = {
    "compiler": "python",
    "integrator": "lsoda",
    "integrator_options": {"rtol": 1e-8, "atol": 1e-16},
}

__all__ = [
    "ALPHA_VALUES", "BOUNDARY_MAX_ITER", "BOUNDARY_TOL_REL",
    "CAPTURE_CRITERION", "CONDITION", "CONDITIONS", "COVALENT_VARIANT",
    "MEASURED_SELECTIVITY_MEDIAN", "MEASURED_SELECTIVITY_SPREAD",
    "OCCUPANCY_LIMITS",
    "CYPA_TOTAL", "G12C_HYDROLYSIS", "G12V_HYDROLYSIS_ASSAY", "GEF_TOTAL",
    "K1_ON", "K2_ON", "KCAT_GAP_WT", "KCAT_TCI", "KD1", "KD2", "KD_OFF_GDP",
    "KINACT_OVER_KI", "KOFF_COVALENT", "KOFF_OFF", "KON_OFF", "K_I",
    "N_POINTS", "N_POINTS_COVALENT", "PROBE_POINTS", "ROOT_MAX_ITER",
    "ROOT_TOL_REL", "SOLVER", "S_VALUES", "T_BENEFIT", "T_COST", "T_END",
    "T_END_COVALENT", "THETA_HIGH", "THETA_REFERENCE", "VARIANTS",
]
