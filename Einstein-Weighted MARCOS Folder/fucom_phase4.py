"""
================================================================================
PHASE 4: EINSTEIN-WEIGHTED MARCOS — FINAL CANDIDATE RANKING
================================================================================
Hybrid XAI Personnel Selection Framework — Q1 Journal Pipeline

----------------------------------------------------------------------
STEP A: FINAL CRITERIA WEIGHTS (user-confirmed mapping + global budget)
----------------------------------------------------------------------
14 FUCOM expert-criterion weights (Phase 2) are summed into 6 core pillars
per the user's definitive mapping, then scaled by a 60% human-expert budget.
The remaining 40% is allocated as fixed weights to the 3 system criteria
that FUCOM never rated (Smart_AI_Performance_Criteria, Salary_NORM,
PILLAR_Location_Logistics_NORM). The 9 final weights are asserted to sum
to exactly 1.0 before any MARCOS computation proceeds.

----------------------------------------------------------------------
STEP B: EINSTEIN-WEIGHTED MARCOS ON FERMATEAN FUZZY NUMBERS (FFNs)
----------------------------------------------------------------------
Standard MARCOS (Stevic et al., 2020) extended to Fermatean Fuzzy Sets with
Einstein t-norm/t-conorm-based aggregation (the natural cube generalization
of the Pythagorean-fuzzy Einstein operators, since FFS replace squares with
cubes throughout):

  Fermatean Fuzzy Einstein Sum of A=(mu_A,nu_A), B=(mu_B,nu_B):
      mu_{A(+)B} = cbrt( (mu_A^3 + mu_B^3) / (1 + mu_A^3 * mu_B^3) )
      nu_{A(+)B} = cbrt( (nu_A^3 * nu_B^3) / (1 + (1-nu_A^3)(1-nu_B^3)) )

  Fermatean Fuzzy Einstein Scalar Multiplication of A by lambda > 0:
      mu_{lambda.A} = cbrt( ((1+mu^3)^lambda - (1-mu^3)^lambda)
                              / ((1+mu^3)^lambda + (1-mu^3)^lambda) )
      nu_{lambda.A} = cbrt( 2*(nu^3)^lambda / ((2-nu^3)^lambda + (nu^3)^lambda) )

  Fermatean Fuzzy Einstein Weighted Average (FFEWA) of A_1..A_n with
  weights w_1..w_n (sum w_j = 1):
      FFEWA = (w_1.A_1) (+) (w_2.A_2) (+) ... (+) (w_n.A_n)
  computed by iterative Einstein summation (associative, so order-invariant).

Algorithm:
  1. For each of the 9 criteria, scan all 293 candidates' Score Function
     S_ij = mu_ij^3 - nu_ij^3 and select the ACTUAL candidate FFN with the
     maximum S as the Ideal (AI_j) and the ACTUAL candidate FFN with the
     minimum S as the Anti-Ideal (AAI_j) for that criterion — per explicit
     user mandate (real candidate values are used as reference points,
     nothing is invented).
  2. FFEWA-aggregate each candidate's 9 criterion FFNs (and the AI/AAI
     reference vectors) into one overall FFN using the Einstein operators
     above, weighted by the final 9 criteria weights.
  3. Defuzzify every aggregated FFN via the Score Function S = mu^3 - nu^3,
     then apply a monotonic +1 shift (S' = S + 1) to guarantee strict
     positivity for the ratio-based utility degrees below — this does not
     alter relative ranking since it is a constant shift applied uniformly.
  4. Utility degrees:
         K_i^+ = S_i' / S_AI'      (utility relative to the Ideal solution)
         K_i^- = S_i' / S_AAI'     (utility relative to the Anti-Ideal solution)
  5. Final utility function (closed-form simplification of the original
     Stevic et al. MARCOS formula — algebraically identical):
         f(K_i) = (K_i^+ + K_i^-) / (1 + K_i^+/K_i^- + K_i^-/K_i^+)
  6. Rank candidates by f(K_i) descending; Rank 1 = best candidate.
================================================================================
"""

import numpy as np
import pandas as pd

FUZZIFIED_PATH = "/mnt/user-data/outputs/Fermatean_Fuzzified_Matrix.csv"
FUCOM_PATH = "/mnt/user-data/outputs/FUCOM_Optimal_Weights.xlsx"
OUTPUT_PATH = "/mnt/user-data/outputs/Q1_Final_Candidate_Rankings.xlsx"

CRITERION_COLUMNS = [
    "PILLAR_Experience_NORM",
    "PILLAR_Education_NORM",
    "PILLAR_Technical_Skills_Certifications_NORM",
    "PILLAR_Psychological_Composite_NORM",
    "PILLAR_Adaptability_TimeManagement_NORM",
    "PILLAR_Cultural_Fit_Creativity_NORM",
    "PILLAR_Location_Logistics_NORM",
    "Salary_NORM",
    "Smart_AI_Performance_Criteria",
]

# ============================================================================
# STEP A.1 — LOAD FUCOM (14-CRITERION) WEIGHTS
# ============================================================================
print("=" * 78)
print("STEP A.1 — LOAD PHASE 2 FUCOM WEIGHTS (14 EXPERT CRITERIA)")
print("=" * 78)

fucom_df = pd.read_excel(FUCOM_PATH, sheet_name="FUCOM_Weights")
fucom_weights = dict(zip(fucom_df["Criterion"], fucom_df["Optimal_FUCOM_Weight"]))

assert np.isclose(sum(fucom_weights.values()), 1.0, atol=1e-8), \
    "FAIL: source FUCOM weights do not sum to 1.0"
print(f"PASS | 14 FUCOM weights loaded, sum = {sum(fucom_weights.values()):.10f}")

# ============================================================================
# STEP A.2 — USER-CONFIRMED 14 -> 6 PILLAR AGGREGATION (SUMMED SUB-WEIGHTS)
# ============================================================================
print("\n" + "=" * 78)
print("STEP A.2 — AGGREGATE FUCOM SUB-WEIGHTS INTO 6 CORE PILLARS")
print("=" * 78)

PILLAR_AGGREGATION_MAP = {
    "PILLAR_Experience_NORM": ["Experience"],
    "PILLAR_Education_NORM": ["Education"],
    "PILLAR_Technical_Skills_Certifications_NORM": ["Technical Skills", "Certifications"],
    "PILLAR_Adaptability_TimeManagement_NORM": ["Adaptability", "Time Management"],
    "PILLAR_Cultural_Fit_Creativity_NORM": ["Cultural Fit", "Creativity & Innovation", "Teamwork"],
    "PILLAR_Psychological_Composite_NORM": [
        "Communication Skills", "Leadership Ability", "Problem Solving",
        "Decision Making Ability", "Emotional Intelligence",
    ],
}

# Integrity check: every one of the 14 source criteria must be used exactly once.
all_mapped = [c for group in PILLAR_AGGREGATION_MAP.values() for c in group]
assert sorted(all_mapped) == sorted(fucom_weights.keys()), (
    "FAIL: 14-criterion aggregation map does not exactly cover the 14 FUCOM "
    f"criteria. Mapped: {sorted(all_mapped)} | Source: {sorted(fucom_weights.keys())}"
)
assert len(all_mapped) == 14 and len(set(all_mapped)) == 14, \
    "FAIL: duplicate or missing criterion in aggregation map"

pillar_summed_weights = {}
for pillar, sub_criteria in PILLAR_AGGREGATION_MAP.items():
    summed = sum(fucom_weights[c] for c in sub_criteria)
    pillar_summed_weights[pillar] = summed
    print(f"  {pillar:<46s} = {' + '.join(sub_criteria)}")
    print(f"  {'':<46s}   -> summed weight = {summed:.6f}")

summed_total = sum(pillar_summed_weights.values())
assert np.isclose(summed_total, 1.0, atol=1e-8), \
    f"FAIL: summed 6-pillar weights = {summed_total}, expected 1.0"
print(f"\nPASS | Sum of 6 aggregated pillar weights = {summed_total:.10f}")

# ============================================================================
# STEP A.3 — GLOBAL 60/40 BUDGET REDISTRIBUTION
# ============================================================================
print("\n" + "=" * 78)
print("STEP A.3 — GLOBAL BUDGET REDISTRIBUTION (60% Expert / 40% System)")
print("=" * 78)

HUMAN_BUDGET = 0.60
FIXED_SYSTEM_WEIGHTS = {
    "Smart_AI_Performance_Criteria": 0.250,
    "Salary_NORM": 0.100,
    "PILLAR_Location_Logistics_NORM": 0.050,
}
assert np.isclose(sum(FIXED_SYSTEM_WEIGHTS.values()), 0.40, atol=1e-12), \
    "FAIL: fixed system weights do not sum to the allocated 0.40 budget"

final_weights = {}
for pillar, w in pillar_summed_weights.items():
    scaled = w * HUMAN_BUDGET
    final_weights[pillar] = scaled
    print(f"  {pillar:<46s} : {w:.6f} x 0.60 = {scaled:.6f}")

for crit, w in FIXED_SYSTEM_WEIGHTS.items():
    final_weights[crit] = w
    print(f"  {crit:<46s} : fixed         = {w:.6f}")

# Re-order to match CRITERION_COLUMNS exactly
final_weights = {col: final_weights[col] for col in CRITERION_COLUMNS}
weight_sum = sum(final_weights.values())

print(f"\nFinal 9-criterion weight vector (in candidate-matrix column order):")
for col in CRITERION_COLUMNS:
    print(f"  {col:<46s} = {final_weights[col]:.6f}")

assert np.isclose(weight_sum, 1.0, atol=1e-9), (
    f"FAIL: final 9 criteria weights sum to {weight_sum}, not 1.0"
)
print(f"\nPASS | Final 9 criteria weights sum EXACTLY to 1.0  ({weight_sum:.12f})")

weight_vector = np.array([final_weights[col] for col in CRITERION_COLUMNS])

# ============================================================================
# STEP B.0 — LOAD & PIVOT THE FERMATEAN FUZZIFIED MATRIX
# ============================================================================
print("\n" + "=" * 78)
print("STEP B.0 — LOAD PHASE 3 OUTPUT AND PIVOT TO WIDE FORMAT")
print("=" * 78)

long_df = pd.read_csv(FUZZIFIED_PATH)
assert long_df.isnull().sum().sum() == 0, "FAIL: residual nulls in fuzzified matrix"

candidate_meta = (
    long_df[["Candidate_ID", "Recruitment_Code", "Department"]]
    .drop_duplicates()
    .set_index("Candidate_ID")
    .sort_index()
)
n_candidates = len(candidate_meta)
n_criteria = len(CRITERION_COLUMNS)
assert n_candidates == 293, f"FAIL: expected 293 candidates, found {n_candidates}"

mu_matrix = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Mu")[CRITERION_COLUMNS]
nu_matrix = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Nu")[CRITERION_COLUMNS]
pi_matrix = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Pi")[CRITERION_COLUMNS]

mu_matrix = mu_matrix.sort_index()
nu_matrix = nu_matrix.sort_index()
pi_matrix = pi_matrix.sort_index()

assert mu_matrix.shape == (n_candidates, n_criteria), "FAIL: Mu matrix shape mismatch"
print(f"PASS | Pivoted Mu/Nu/Pi matrices       : {mu_matrix.shape} (candidates x criteria)")

cubic_check = mu_matrix.values ** 3 + nu_matrix.values ** 3 + pi_matrix.values ** 3
max_cubic_dev = np.abs(cubic_check - 1.0).max()
assert max_cubic_dev < 1e-6, f"FAIL: cubic law violated in pivoted matrix, dev={max_cubic_dev}"
print(f"PASS | Cubic law re-verified post-pivot : max deviation = {max_cubic_dev:.2e}")

# ============================================================================
# FERMATEAN FUZZY EINSTEIN OPERATORS
# ============================================================================
EPS = 1e-12

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def einstein_sum(mu_a, nu_a, mu_b, nu_b):
    """Fermatean Fuzzy Einstein Sum: A (+) B."""
    mu_a3, mu_b3 = mu_a ** 3, mu_b ** 3
    nu_a3, nu_b3 = nu_a ** 3, nu_b ** 3

    mu_num = mu_a3 + mu_b3
    mu_den = 1.0 + mu_a3 * mu_b3
    mu_out = np.cbrt(mu_num / np.where(mu_den == 0, EPS, mu_den))

    nu_num = nu_a3 * nu_b3
    nu_den = 1.0 + (1.0 - nu_a3) * (1.0 - nu_b3)
    nu_out = np.cbrt(nu_num / np.where(nu_den == 0, EPS, nu_den))

    return _clip01(mu_out), _clip01(nu_out)

def einstein_scalar_multiply(mu, nu, lam):
    """Fermatean Fuzzy Einstein Scalar Multiplication: lambda . A."""
    mu3 = np.clip(mu ** 3, EPS, 1.0 - EPS)
    nu3 = np.clip(nu ** 3, EPS, 1.0 - EPS)

    a = (1.0 + mu3) ** lam
    b = (1.0 - mu3) ** lam
    mu_out = np.cbrt((a - b) / (a + b))

    c = (2.0 - nu3) ** lam
    d = nu3 ** lam
    nu_out = np.cbrt((2.0 * d) / (c + d))

    return _clip01(mu_out), _clip01(nu_out)

def ffewa_aggregate(mu_row, nu_row, weights):
    """
    Fermatean Fuzzy Einstein Weighted Average across n criteria for ONE
    candidate (or reference vector). mu_row, nu_row, weights are length-n
    1D arrays. Returns the aggregated (mu, nu) scalar pair.
    """
    mu_acc, nu_acc = einstein_scalar_multiply(mu_row[0], nu_row[0], weights[0])
    for j in range(1, len(weights)):
        mu_w, nu_w = einstein_scalar_multiply(mu_row[j], nu_row[j], weights[j])
        mu_acc, nu_acc = einstein_sum(mu_acc, nu_acc, mu_w, nu_w)
    return mu_acc, nu_acc

def score_function(mu, nu):
    """Fermatean Fuzzy Score Function: S = mu^3 - nu^3."""
    return mu ** 3 - nu ** 3

# ============================================================================
# STEP B.1 — IDEAL (AI) AND ANTI-IDEAL (AAI) SOLUTIONS PER CRITERION
# ============================================================================
print("\n" + "=" * 78)
print("STEP B.1 — IDEAL / ANTI-IDEAL SOLUTIONS (per-criterion, via Score Function)")
print("=" * 78)

score_matrix = score_function(mu_matrix.values, nu_matrix.values)  # (293, 9)

ai_mu, ai_nu, ai_idx = np.zeros(n_criteria), np.zeros(n_criteria), np.zeros(n_criteria, dtype=int)
aai_mu, aai_nu, aai_idx = np.zeros(n_criteria), np.zeros(n_criteria), np.zeros(n_criteria, dtype=int)

candidate_ids = mu_matrix.index.values
for j in range(n_criteria):
    best_row = int(np.argmax(score_matrix[:, j]))
    worst_row = int(np.argmin(score_matrix[:, j]))
    ai_mu[j], ai_nu[j], ai_idx[j] = mu_matrix.values[best_row, j], nu_matrix.values[best_row, j], candidate_ids[best_row]
    aai_mu[j], aai_nu[j], aai_idx[j] = mu_matrix.values[worst_row, j], nu_matrix.values[worst_row, j], candidate_ids[worst_row]
    print(
        f"  {CRITERION_COLUMNS[j]:<46s} AI=Candidate#{ai_idx[j]:<4d}(S={score_matrix[best_row,j]:+.4f})  "
        f"AAI=Candidate#{aai_idx[j]:<4d}(S={score_matrix[worst_row,j]:+.4f})"
    )

# ============================================================================
# STEP B.2 — FFEWA AGGREGATION: CANDIDATES, AI, AND AAI
# ============================================================================
print("\n" + "=" * 78)
print("STEP B.2 — EINSTEIN-WEIGHTED AGGREGATION (FFEWA)")
print("=" * 78)

agg_mu = np.zeros(n_candidates)
agg_nu = np.zeros(n_candidates)
for i in range(n_candidates):
    m, n_ = ffewa_aggregate(mu_matrix.values[i, :], nu_matrix.values[i, :], weight_vector)
    agg_mu[i], agg_nu[i] = m, n_

ai_agg_mu, ai_agg_nu = ffewa_aggregate(ai_mu, ai_nu, weight_vector)
aai_agg_mu, aai_agg_nu = ffewa_aggregate(aai_mu, aai_nu, weight_vector)

print(f"  Aggregated AI  (Ideal)      : mu={ai_agg_mu:.6f}, nu={ai_agg_nu:.6f}")
print(f"  Aggregated AAI (Anti-Ideal) : mu={aai_agg_mu:.6f}, nu={aai_agg_nu:.6f}")
print(f"  PASS | All 293 candidate FFNs aggregated via FFEWA")

# ============================================================================
# STEP B.3 — DEFUZZIFICATION (+1 SHIFT) AND UTILITY DEGREES
# ============================================================================
print("\n" + "=" * 78)
print("STEP B.3 — DEFUZZIFICATION & UTILITY DEGREES (K+, K-)")
print("=" * 78)

S_i = score_function(agg_mu, agg_nu)
S_AI = score_function(ai_agg_mu, ai_agg_nu)
S_AAI = score_function(aai_agg_mu, aai_agg_nu)

# Monotonic +1 shift -> strictly positive domain for ratio-based utility degrees.
S_i_shift = S_i + 1.0
S_AI_shift = S_AI + 1.0
S_AAI_shift = S_AAI + 1.0

assert S_AI_shift > 0 and S_AAI_shift > 0, "FAIL: shifted AI/AAI scores not strictly positive"

K_plus = S_i_shift / S_AI_shift     # utility relative to Ideal
K_minus = S_i_shift / S_AAI_shift   # utility relative to Anti-Ideal

print(f"  S_AI (shifted)  = {S_AI_shift:.6f}")
print(f"  S_AAI (shifted) = {S_AAI_shift:.6f}")
print(f"  PASS | K+ range = [{K_plus.min():.4f}, {K_plus.max():.4f}]")
print(f"  PASS | K- range = [{K_minus.min():.4f}, {K_minus.max():.4f}]")

# ============================================================================
# STEP B.4 — FINAL UTILITY FUNCTION f(K) AND RANKING
# ============================================================================
print("\n" + "=" * 78)
print("STEP B.4 — FINAL UTILITY FUNCTION f(K) AND RANKING")
print("=" * 78)

f_K = (K_plus + K_minus) / (1.0 + (K_plus / K_minus) + (K_minus / K_plus))

assert np.all(np.isfinite(f_K)), "FAIL: non-finite f(K) values detected"

results_df = pd.DataFrame({
    "Candidate_ID": candidate_ids,
    "Recruitment_Code": candidate_meta.loc[candidate_ids, "Recruitment_Code"].values,
    "Department": candidate_meta.loc[candidate_ids, "Department"].values,
    "Aggregated_Mu": agg_mu,
    "Aggregated_Nu": agg_nu,
    "Score_S": S_i,
    "K_plus": K_plus,
    "K_minus": K_minus,
    "f_K": f_K,
})

results_df = results_df.sort_values("f_K", ascending=False).reset_index(drop=True)
results_df.insert(0, "Rank", np.arange(1, n_candidates + 1))

assert results_df["Rank"].nunique() == n_candidates, "FAIL: duplicate ranks assigned"
assert results_df["Rank"].min() == 1 and results_df["Rank"].max() == n_candidates, \
    "FAIL: rank range invalid"
assert results_df["f_K"].is_monotonic_decreasing, "FAIL: ranking is not properly sorted"

print(f"PASS | {n_candidates} candidates ranked 1..{n_candidates}, no duplicate ranks")
print(f"PASS | f(K) range = [{f_K.min():.6f}, {f_K.max():.6f}]")
print(f"\nTop 5 candidates:")
print(results_df.head(5)[["Rank", "Candidate_ID", "Recruitment_Code", "Department", "f_K"]].to_string(index=False))
print(f"\nBottom 5 candidates:")
print(results_df.tail(5)[["Rank", "Candidate_ID", "Recruitment_Code", "Department", "f_K"]].to_string(index=False))

# ============================================================================
# STEP C — EXPORT MULTI-SHEET AUDITED WORKBOOK
# ============================================================================
print("\n" + "=" * 78)
print("STEP C — EXPORT Q1_Final_Candidate_Rankings.xlsx")
print("=" * 78)

weights_audit_df = pd.DataFrame({
    "Criterion_Column": CRITERION_COLUMNS,
    "Final_Weight": [final_weights[c] for c in CRITERION_COLUMNS],
    "Source": (
        ["FUCOM-aggregated x 0.60"] * 6 + ["Fixed system allocation (0.40 budget)"] * 3
    ),
})

ai_aai_df = pd.DataFrame({
    "Criterion_Column": CRITERION_COLUMNS,
    "AI_Candidate_ID": ai_idx,
    "AI_Mu": ai_mu, "AI_Nu": ai_nu,
    "AAI_Candidate_ID": aai_idx,
    "AAI_Mu": aai_mu, "AAI_Nu": aai_nu,
})

validation_df = pd.DataFrame({
    "Metric": [
        "Candidates ranked",
        "Sum of final 9 criteria weights",
        "Max cubic-law deviation (pivoted input)",
        "Aggregated AI score S (shifted)",
        "Aggregated AAI score S (shifted)",
        "f(K) range",
        "Duplicate ranks",
    ],
    "Value": [
        n_candidates,
        f"{weight_sum:.12f}",
        f"{max_cubic_dev:.2e}",
        f"{S_AI_shift:.6f}",
        f"{S_AAI_shift:.6f}",
        f"[{f_K.min():.6f}, {f_K.max():.6f}]",
        "0 (verified)",
    ],
})

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Final_Rankings", index=False)
    weights_audit_df.to_excel(writer, sheet_name="Criteria_Weights", index=False)
    ai_aai_df.to_excel(writer, sheet_name="AI_AAI_Reference", index=False)
    validation_df.to_excel(writer, sheet_name="Validation_Audit", index=False)

print(f"EXPORT COMPLETE -> {OUTPUT_PATH}")
print("Sheets: Final_Rankings | Criteria_Weights | AI_AAI_Reference | Validation_Audit")
print("=" * 78)
