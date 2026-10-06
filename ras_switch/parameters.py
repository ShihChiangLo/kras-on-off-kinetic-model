"""Rate constants, initial conditions and solver settings for the RAS switch model.

Every cell carries two RAS pools: wild-type RAS and a transfected pool of the
variant being modelled (prefix ``mu``). Rate constants are in s^-1 or
M^-1 s^-1 and are listed with their status and source in S1 Table. The
nucleotide-association and GEF-exchange constants already include the
variant's kinetic scaling factor (S1 Text), and the GEF and GAP association
constants already include the membrane-localization factor D = 250 (S2 Text).
"""

WT = {
    "kd_GDP": 1.1e-4, "ka_GDP": 41.4, "kd_GTP": 2.5e-4, "ka_GTP": 396,
    "kDa_GEF": 8500000, "kDd_GEF": 9.224, "kcat_GDP": 3.9,
    "kTa_GEF": 750000, "kTd_GEF": 1.8e-1, "kcat_GTP": 7.2e-1,
    "kTa_GAP": 1.3e10, "kTd_GAP": 6.56, "kcat_GAP": 5.4,
    "khyd": 3.5e-4, "ka_Eff": 4.5e7, "kd_Eff": 3.6,
    "mukd_GDP": 1.1e-4, "muka_GDP": 41.4, "mukd_GTP": 2.5e-4, "muka_GTP": 396,
    "mukDa_GEF": 8500000, "mukDd_GEF": 9.224, "mukcat_GDP": 3.9,
    "mukTa_GEF": 750000, "mukTd_GEF": 1.8e-1, "mukcat_GTP": 7.2e-1,
    "mukTa_GAP": 1.3e10, "mukTd_GAP": 6.56, "mukcat_GAP": 5.4,
    "mukhyd": 3.5e-4, "muka_Eff": 4.5e7, "mukd_Eff": 3.6,
}

G12D = {
    "kd_GDP": 1.1e-4, "ka_GDP": 51.75, "kd_GTP": 2.5e-4, "ka_GTP": 495,
    "kDa_GEF": 8500000, "kDd_GEF": 9.224, "kcat_GDP": 4.875,
    "kTa_GEF": 750000, "kTd_GEF": 1.8e-1, "kcat_GTP": 0.8999999999999999,
    "kTa_GAP": 1.3e10, "kTd_GAP": 6.56, "kcat_GAP": 5.4,
    "khyd": 3.5e-4, "ka_Eff": 4.5e7, "kd_Eff": 3.6,
    "mukd_GDP": 5.238095238095238e-05, "muka_GDP": 71.02941176470587,
    "mukd_GTP": 0.00125, "muka_GTP": 1697.1428571428573,
    "mukDa_GEF": 8500000, "mukDd_GEF": 9.224, "mukcat_GDP": 4.875,
    "mukTa_GEF": 2672750.0000000005, "mukTd_GEF": 1.8e-1, "mukcat_GTP": 3.784125,
    "mukTa_GAP": 1.3e10, "mukTd_GAP": 11.959859999999999,
    "mukcat_GAP": 0.00014000000000000001,
    "mukhyd": 0.00014000000000000001, "muka_Eff": 4.5e7, "mukd_Eff": 3.6,
}

G12V = {
    "kd_GDP": 1.1e-4, "ka_GDP": 248.39999999999998, "kd_GTP": 2.5e-4, "ka_GTP": 2376,
    "kDa_GEF": 8500000, "kDd_GEF": 9.224, "kcat_GDP": 23.4,
    "kTa_GEF": 750000, "kTd_GEF": 1.8e-1, "kcat_GTP": 4.32,
    "kTa_GAP": 1.3e10, "kTd_GAP": 6.56, "kcat_GAP": 5.4,
    "khyd": 3.5e-4, "ka_Eff": 4.5e7, "kd_Eff": 3.6,
    "mukd_GDP": 3.404761904761905e-05, "muka_GDP": 564.9882352941177,
    "mukd_GTP": 0.0002, "muka_GTP": 9843.428571428572,
    "mukDa_GEF": 8500000, "mukDd_GEF": 9.224, "mukcat_GDP": 23.4,
    "mukTa_GEF": 1001666.6666666667, "mukTd_GEF": 1.8e-1, "mukcat_GTP": 6.132,
    "mukTa_GAP": 1.3e10, "mukTd_GAP": 11.959947499999998,
    "mukcat_GAP": 5.2499999999999995e-05,
    "mukhyd": 5.2499999999999995e-05, "muka_Eff": 4.5e7, "mukd_Eff": 1.5999999999999999,
}

G12C = {
    "kd_GDP": 1.1e-4, "ka_GDP": 140.76, "kd_GTP": 2.5e-4, "ka_GTP": 1346.3999999999999,
    "kDa_GEF": 8500000, "kDd_GEF": 9.224, "kcat_GDP": 13.26,
    "kTa_GEF": 750000, "kTd_GEF": 1.8e-1, "kcat_GTP": 2.448,
    "kTa_GAP": 1.3e10, "kTd_GAP": 6.56, "kcat_GAP": 5.4,
    "khyd": 3.5e-4, "ka_Eff": 4.5e7, "kd_Eff": 3.6,
    "mukd_GDP": 1.1e-4, "muka_GDP": 140.76,
    "mukd_GTP": 2.5e-4, "muka_GTP": 1346.3999999999999,
    "mukDa_GEF": 8500000, "mukDd_GEF": 9.224, "mukcat_GDP": 13.26,
    "mukTa_GEF": 750000, "mukTd_GEF": 1.8e-1, "mukcat_GTP": 2.448,
    "mukTa_GAP": 1.3e10, "mukTd_GAP": 119.59974779411765,
    "mukcat_GAP": 0.0002522058823529412,
    "mukhyd": 0.0002522058823529412, "muka_Eff": 4.5e7, "mukd_Eff": 4.307142857,
}

VARIANTS = {"WT": WT, "G12D": G12D, "G12V": G12V, "G12C": G12C}

# Total GEF, the same in every condition (M).
GEF_TOTAL = 2e-10

# The nine ([RAS], [effector], [GAP]) conditions of S3 Table, in M. [RAS] is the
# initial concentration of each of the two RAS pools.
CONDITIONS = [
    (4e-7, 2e-7, 6e-11),
    (4e-7, 4e-7, 6e-11),
    (4e-7, 8e-7, 6e-11),
    (2e-7, 1e-7, 4e-11),
    (2e-7, 2e-7, 4e-11),
    (2e-7, 4e-7, 4e-11),
    (8e-7, 4e-7, 1e-10),
    (8e-7, 8e-7, 1e-10),
    (8e-7, 1.6e-6, 1e-10),
]

# Pre-drug phase: integrate from nucleotide-free RAS to t = 1e4 s and read the
# last point.
T_STEADY = 10000
N_STEADY = 10001

# AMG 510 (Fig 5): second-order covalent inactivation of mutant RAS-GDP,
# k_inact/K_I in M^-1 s^-1, applied for 60 min after the pre-drug phase.
K_DRUG = 9900
DRUG_DOSES = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
INCUBATION_S = 3600

# The network is stiff. Under PySB's default integrator it returns NaN or wrong
# values without raising an error, so the integrator is set explicitly.
SOLVER = {
    "integrator": "lsoda",
    "integrator_options": {"rtol": 1e-8, "atol": 1e-16, "mxstep": 200000},
}
