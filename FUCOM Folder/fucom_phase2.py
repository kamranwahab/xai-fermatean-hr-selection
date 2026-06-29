"""
================================================================================
PHASE 2: FULL CONSISTENCY METHOD (FUCOM) — CRITERIA WEIGHTING
================================================================================
Hybrid XAI Personnel Selection Framework — Q1 Journal Pipeline

Mathematical Protocol (Pamučar, Stević & Sremac, 2018 — original FUCOM paper):

  Step 1: Rank criteria C1 > C2 > ... > Cn by expert-derived significance
          (here, the arithmetic mean of 18 expert scores per criterion).

  Step 2: Determine comparative priority (Phi) between adjacently ranked
          criteria:
              Phi_(k/k+1) = W_k / W_(k+1) ,  k = 1, 2, ..., n-1
          where W_k is the mean significance score of the k-th ranked
          criterion. Phi_(k/k+1) >= 1 by construction of the ranking.

  Step 3: Solve the non-linear program for final weights w_1...w_n that
          minimizes the maximum deviation (chi / DFC) from full consistency:

              min chi

              s.t.
              (C1) | w_k / w_(k+1) - Phi_(k/k+1) | <= chi   for all k=1..n-1
              (C2) | w_k / w_(k+2) - Phi_(k/k+1) * Phi_(k+1/k+2) | <= chi
                                                              for all k=1..n-2
              (C3) sum_j w_j = 1
              (C4) w_j > 0  for all j   (here bounded (0, 1])
              (C5) ranking order preserved: w_1 >= w_2 >= ... >= w_n

  The final chi (also reported in literature as DFC — Deviation from Full
  Consistency) must be as close to 0 as possible. chi -> 0 indicates the
  derived weights are in mathematically full consistency with the expert
  pairwise comparative priorities.

No fabricated values are used anywhere in this script — every numeric input
to the optimizer is derived directly from FINAL_Cleaned_Expert_Opinions.xlsx.
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint

INPUT_PATH = "/mnt/user-data/uploads/1782035590466_FINAL_Cleaned_Expert_Opinions.xlsx"
OUTPUT_PATH = "/mnt/user-data/outputs/FUCOM_Optimal_Weights.xlsx"

# ------------------------------------------------------------------------
# STEP 0: LOAD & VALIDATE
# ------------------------------------------------------------------------
df = pd.read_excel(INPUT_PATH)

n_experts, n_criteria = df.shape
criteria_names = df.columns.tolist()

assert n_experts == 18, f"FAIL: expected 18 experts, found {n_experts}"
assert n_criteria == 14, f"FAIL: expected 14 criteria, found {n_criteria}"
assert df.isnull().sum().sum() == 0, "FAIL: residual nulls detected in expert matrix"
assert (df.min().min() >= 1.0) and (df.max().max() <= 5.0), "FAIL: values outside Likert bounds [1,5]"

print("=" * 78)
print("STEP 0 — INPUT VALIDATION")
print("=" * 78)
print(f"PASS | Matrix shape confirmed       : {df.shape} (18 experts x 14 criteria)")
print(f"PASS | Residual nulls                : 0")
print(f"PASS | Value bounds                  : [{df.min().min():.2f}, {df.max().max():.2f}] within [1.0, 5.0]")

# ------------------------------------------------------------------------
# STEP 1: CRITERIA RANKING VIA ARITHMETIC MEAN OF 18 EXPERTS
# ------------------------------------------------------------------------
criteria_means = df.mean(axis=0)

ranked = criteria_means.sort_values(ascending=False)
ranked_names = ranked.index.tolist()
ranked_scores = ranked.values.astype(float)

print("\n" + "=" * 78)
print("STEP 1 — CRITERIA RANKING (Descending Mean Expert Significance)")
print("=" * 78)
for i, (name, score) in enumerate(zip(ranked_names, ranked_scores), start=1):
    print(f"  Rank {i:2d}: {name:<28s} mean = {score:.4f}")

# ------------------------------------------------------------------------
# STEP 2: COMPARATIVE PRIORITY (Phi) BETWEEN ADJACENT RANKED CRITERIA
# ------------------------------------------------------------------------
n = n_criteria
phi = np.array([ranked_scores[k] / ranked_scores[k + 1] for k in range(n - 1)])

print("\n" + "=" * 78)
print("STEP 2 — COMPARATIVE PRIORITY (Phi_k/(k+1) = W_k / W_(k+1))")
print("=" * 78)
for k in range(n - 1):
    print(f"  Phi_{k+1}/{k+2}  ({ranked_names[k]:<24s} / {ranked_names[k+1]:<24s}) = {phi[k]:.4f}")

# Precompute two-step priority products for the second consistency
# condition: Phi_(k/k+2) = Phi_(k/k+1) * Phi_(k+1/k+2)
phi2 = np.array([phi[k] * phi[k + 1] for k in range(n - 2)]) if n > 2 else np.array([])

# ------------------------------------------------------------------------
# STEP 3: NON-LINEAR OPTIMIZATION — MINIMIZE MAX DEVIATION (chi / DFC)
# ------------------------------------------------------------------------
# Decision vector x = [w_1, w_2, ..., w_n, chi]  (length n+1)
# Objective: minimize chi (the last element of x)

def objective(x):
    return x[-1]  # chi

def objective_grad(x):
    g = np.zeros_like(x)
    g[-1] = 1.0
    return g

# --- Constraint set C1: |w_k/w_(k+1) - Phi_k| <= chi  for k=1..n-1 ---
# Implemented as two smooth (non-abs) inequalities to keep the gradient
# well-defined everywhere for the SLSQP/trust-constr line search:
#   chi - (ratio - Phi_k) >= 0   AND   chi + (ratio - Phi_k) >= 0
def c1_constraints(x):
    w = x[:n]
    chi = x[-1]
    vals = []
    for k in range(n - 1):
        ratio = w[k] / w[k + 1]
        diff = ratio - phi[k]
        vals.append(chi - diff)
        vals.append(chi + diff)
    return np.array(vals)

# --- Constraint set C2: |w_k/w_(k+2) - Phi_k*Phi_(k+1)| <= chi for k=1..n-2 ---
def c2_constraints(x):
    w = x[:n]
    chi = x[-1]
    vals = []
    for k in range(n - 2):
        ratio = w[k] / w[k + 2]
        diff = ratio - phi2[k]
        vals.append(chi - diff)
        vals.append(chi + diff)
    return np.array(vals)

# --- Constraint C3: sum(w) = 1 ---
def c3_constraint(x):
    return np.sum(x[:n]) - 1.0

# --- Constraint C5: ranking order preserved, w_k >= w_(k+1) ---
def c5_constraints(x):
    w = x[:n]
    return np.array([w[k] - w[k + 1] for k in range(n - 1)])  # must be >= 0

nlc1 = NonlinearConstraint(c1_constraints, 0, np.inf)
nlc2 = NonlinearConstraint(c2_constraints, 0, np.inf)
nlc3 = NonlinearConstraint(c3_constraint, 0, 0)
nlc5 = NonlinearConstraint(c5_constraints, 0, np.inf)

constraints = [nlc1, nlc2, nlc3, nlc5]

# --- Bounds: weights strictly within (epsilon, 1]; chi within [0, 1] ---
epsilon = 1e-6
bounds = [(epsilon, 1.0)] * n + [(0.0, 1.0)]

# --- Initial guess: weights proportional to ranked mean scores (already
#     in correct descending order, satisfying C5 at the starting point),
#     normalized to sum to 1. chi initialized at a small positive value. ---
x0_weights = ranked_scores / ranked_scores.sum()
x0 = np.concatenate([x0_weights, [0.05]])

result = minimize(
    objective,
    x0,
    jac=objective_grad,
    method="trust-constr",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 5000, "gtol": 1e-10, "xtol": 1e-12, "verbose": 0},
)
solver_used = "trust-constr"

# Fallback to SLSQP from the trust-constr solution if needed
if not result.success:
    result = minimize(
        objective,
        result.x,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    solver_used = "SLSQP (fallback from trust-constr)"

assert result.success, f"FAIL: FUCOM optimization did not converge — {result.message}"

optimal_weights = result.x[:n]
chi_dfc = result.x[-1]

# Defensive re-normalization (guards against floating point drift only;
# does not alter the optimizer's solution materially)
optimal_weights = optimal_weights / optimal_weights.sum()

print("\n" + "=" * 78)
print("STEP 3 — NON-LINEAR OPTIMIZATION RESULT")
print("=" * 78)
print(f"Solver                         : {solver_used}")
print(f"Convergence status              : {result.success} ({result.message})")
print(f"Sum of optimal weights          : {optimal_weights.sum():.10f}  (PASS, target = 1.0)")
print(f"Deviation from Full Consistency : chi (DFC) = {chi_dfc:.8f}")
if chi_dfc < 0.05:
    print("Consistency interpretation       : EXCELLENT (chi < 0.05) — peer-review grade")
elif chi_dfc < 0.10:
    print("Consistency interpretation       : ACCEPTABLE (chi < 0.10)")
else:
    print("Consistency interpretation       : WARNING — chi >= 0.10, review expert inputs")

# ------------------------------------------------------------------------
# STEP 4: MAP OPTIMAL WEIGHTS BACK TO ORIGINAL CRITERIA NAMES
# ------------------------------------------------------------------------
fucom_result_df = pd.DataFrame({
    "Criterion": ranked_names,
    "Rank": range(1, n + 1),
    "Mean_Expert_Score": ranked_scores,
    "Optimal_FUCOM_Weight": optimal_weights,
})

# Re-sort to match original column order in the source workbook for
# downstream traceability against FINAL_Cleaned_Expert_Opinions.xlsx
fucom_result_original_order = fucom_result_df.set_index("Criterion").loc[criteria_names].reset_index()

print("\n" + "=" * 78)
print("STEP 4 — FINAL FUCOM WEIGHTS (Original Column Order)")
print("=" * 78)
print(fucom_result_original_order.to_string(index=False))

# Validation: weights must strictly fall within (0, 1]
assert (optimal_weights > 0).all() and (optimal_weights <= 1.0).all(), \
    "FAIL: one or more optimal weights violate the (0, 1] bound"
assert np.isclose(optimal_weights.sum(), 1.0, atol=1e-8), \
    "FAIL: optimal weights do not sum to 1.0"

print("\nPASS | All weights strictly within (0, 1] bound")
print(f"PASS | Sum of weights = {optimal_weights.sum():.10f} == 1.0")

# ------------------------------------------------------------------------
# STEP 5: EXPORT MULTI-SHEET AUDITED WORKBOOK
# ------------------------------------------------------------------------
phi_df = pd.DataFrame({
    "Comparison": [f"Phi_{k+1}/{k+2}" for k in range(n - 1)],
    "Criterion_k": ranked_names[:-1],
    "Criterion_k+1": ranked_names[1:],
    "W_k": ranked_scores[:-1],
    "W_k+1": ranked_scores[1:],
    "Phi_k_over_k+1": phi,
})

audit_df = pd.DataFrame({
    "Metric": [
        "Number of Experts",
        "Number of Criteria",
        "Solver",
        "Convergence Status",
        "Sum of Optimal Weights",
        "Chi (Deviation from Full Consistency / DFC)",
        "Consistency Interpretation",
    ],
    "Value": [
        n_experts,
        n_criteria,
        solver_used,
        str(result.success),
        f"{optimal_weights.sum():.10f}",
        f"{chi_dfc:.8f}",
        "EXCELLENT" if chi_dfc < 0.05 else ("ACCEPTABLE" if chi_dfc < 0.10 else "WARNING"),
    ],
})

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    fucom_result_original_order.to_excel(writer, sheet_name="FUCOM_Weights", index=False)
    fucom_result_df.to_excel(writer, sheet_name="Ranked_Criteria", index=False)
    phi_df.to_excel(writer, sheet_name="Comparative_Priority", index=False)
    audit_df.to_excel(writer, sheet_name="Validation_Audit", index=False)

print("\n" + "=" * 78)
print(f"EXPORT COMPLETE -> {OUTPUT_PATH}")
print("Sheets: FUCOM_Weights | Ranked_Criteria | Comparative_Priority | Validation_Audit")
print("=" * 78)
