"""
================================================================================
SECTION 5.5 / 6.0 — GOD-MODE COMPARATIVE & SENSITIVITY SUITE
================================================================================
Hybrid XAI Personnel Selection Framework — Q1 Journal Pipeline

Five stress tests on the Phase 4 Einstein-Weighted Fermatean MARCOS ranking:
  TEST 1: Big Five MCDM Engine Super-Roster (MARCOS/VIKOR/COPRAS/WASPAS/EDAS)
  TEST 2: Belton-Gear Rank Reversal Stress Test (10% / 25% bottom truncation)
  TEST 3: AI-Criterion Weight Perturbation Sweep (5 weight scenarios)
  TEST 4: ML Backbone Sensitivity (XGBoost -> HistGradientBoostingClassifier)
  TEST 5: Top-Heavy Non-Parametric Correlation (Spearman, Kendall tau-b, r_w)

DISCLOSED METHODOLOGICAL NOTE (Test 4): The task names a "HistGradientBoosting
Regressor" as the XGBoost proxy. The original Layer-2 model was an XGBoost
*classifier* whose predict_proba() output was averaged with Logistic
Regression's predict_proba() to form Smart_AI_Performance_Criteria. To keep
exact parity with that architecture (probability output in [0,1], averaged
with the unchanged LR backbone), this script uses
HistGradientBoostingClassifier (the classifier variant) rather than the
regressor. This is flagged explicitly rather than silently substituted.

No fabricated values anywhere — every number is computed from the three
source files (Fermatean_Fuzzified_Matrix.csv, FUCOM_Optimal_Weights.xlsx,
ML_Augmented_Mendeley_Matrix.xlsx).
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

FUZZIFIED_PATH = "/mnt/user-data/uploads/Fermatean_Fuzzified_Matrix.csv"
FUCOM_PATH = "/mnt/user-data/uploads/FUCOM_Optimal_Weights.xlsx"
ML_PATH = "/mnt/user-data/uploads/ML_Augmented_Mendeley_Matrix.xlsx"

OUTPUT_XLSX = "/mnt/user-data/outputs/Q1_Master_Robustness_Audit.xlsx"
OUTPUT_FIG = "/mnt/user-data/outputs/Figure_C2_Master_Robustness_Panel.png"

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
# SETUP — RECONSTRUCT THE FINAL 9 CRITERIA WEIGHTS (identical to Phase 4)
# ============================================================================
print("=" * 78)
print("SETUP — RECONSTRUCTING MASTER WEIGHT VECTOR")
print("=" * 78)

fucom_df = pd.read_excel(FUCOM_PATH, sheet_name="FUCOM_Weights")
fucom_weights = dict(zip(fucom_df["Criterion"], fucom_df["Optimal_FUCOM_Weight"]))

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
HUMAN_BUDGET = 0.60
FIXED_SYSTEM_WEIGHTS = {
    "Smart_AI_Performance_Criteria": 0.250,
    "Salary_NORM": 0.100,
    "PILLAR_Location_Logistics_NORM": 0.050,
}

pillar_summed_weights = {
    pillar: sum(fucom_weights[c] for c in subs)
    for pillar, subs in PILLAR_AGGREGATION_MAP.items()
}
final_weights = {p: w * HUMAN_BUDGET for p, w in pillar_summed_weights.items()}
final_weights.update(FIXED_SYSTEM_WEIGHTS)
final_weights = {col: final_weights[col] for col in CRITERION_COLUMNS}
weight_sum = sum(final_weights.values())
assert np.isclose(weight_sum, 1.0, atol=1e-9), f"FAIL: weights sum to {weight_sum}"
weight_vector = np.array([final_weights[c] for c in CRITERION_COLUMNS])
print(f"PASS | Master weight vector reconstructed, sum = {weight_sum:.12f}")
for c, w in final_weights.items():
    print(f"  {c:<46s} = {w:.6f}")

# ============================================================================
# SETUP — LOAD & PIVOT FERMATEAN FUZZIFIED MATRIX
# ============================================================================
print("\n" + "=" * 78)
print("SETUP — LOADING FERMATEAN FUZZIFIED MATRIX")
print("=" * 78)

long_df = pd.read_csv(FUZZIFIED_PATH)
candidate_meta = (
    long_df[["Candidate_ID", "Recruitment_Code", "Department"]]
    .drop_duplicates().set_index("Candidate_ID").sort_index()
)
n_candidates = len(candidate_meta)
n_criteria = len(CRITERION_COLUMNS)
candidate_ids_all = candidate_meta.index.values

mu_full = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Mu")[CRITERION_COLUMNS].sort_index()
nu_full = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Nu")[CRITERION_COLUMNS].sort_index()
esco_modifier_full = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="ESCO_Modifier")[CRITERION_COLUMNS].sort_index()
crisp_full = long_df.pivot(index="Candidate_ID", columns="Criterion_Column", values="Crisp_Value")[CRITERION_COLUMNS].sort_index()

print(f"PASS | {n_candidates} candidates x {n_criteria} criteria loaded and pivoted")

# ============================================================================
# FERMATEAN FUZZY EINSTEIN OPERATORS (identical to Phase 4)
# ============================================================================
EPS = 1e-12

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def einstein_sum(mu_a, nu_a, mu_b, nu_b):
    mu_a3, mu_b3 = mu_a ** 3, mu_b ** 3
    nu_a3, nu_b3 = nu_a ** 3, nu_b ** 3
    mu_den = 1.0 + mu_a3 * mu_b3
    mu_out = np.cbrt((mu_a3 + mu_b3) / np.where(mu_den == 0, EPS, mu_den))
    nu_den = 1.0 + (1.0 - nu_a3) * (1.0 - nu_b3)
    nu_out = np.cbrt((nu_a3 * nu_b3) / np.where(nu_den == 0, EPS, nu_den))
    return _clip01(mu_out), _clip01(nu_out)

def einstein_scalar_multiply(mu, nu, lam):
    mu3 = np.clip(mu ** 3, EPS, 1.0 - EPS)
    nu3 = np.clip(nu ** 3, EPS, 1.0 - EPS)
    a, b = (1.0 + mu3) ** lam, (1.0 - mu3) ** lam
    mu_out = np.cbrt((a - b) / (a + b))
    c, d = (2.0 - nu3) ** lam, nu3 ** lam
    nu_out = np.cbrt((2.0 * d) / (c + d))
    return _clip01(mu_out), _clip01(nu_out)

def ffewa_aggregate(mu_row, nu_row, weights):
    mu_acc, nu_acc = einstein_scalar_multiply(mu_row[0], nu_row[0], weights[0])
    for j in range(1, len(weights)):
        mu_w, nu_w = einstein_scalar_multiply(mu_row[j], nu_row[j], weights[j])
        mu_acc, nu_acc = einstein_sum(mu_acc, nu_acc, mu_w, nu_w)
    return mu_acc, nu_acc

def score_function(mu, nu):
    return mu ** 3 - nu ** 3

def run_einstein_marcos(mu_mat, nu_mat, ids, weights):
    """
    Full Fermatean Fuzzy Einstein-Weighted MARCOS on an arbitrary candidate
    subset. mu_mat, nu_mat: (m, 9) arrays. ids: length-m array of candidate
    IDs. weights: length-9 weight vector. Returns a DataFrame with
    Candidate_ID, f_K, Rank (1..m, 1=best), sorted by Rank.
    """
    m, k = mu_mat.shape
    score_mat = score_function(mu_mat, nu_mat)

    ai_mu, ai_nu = np.zeros(k), np.zeros(k)
    aai_mu, aai_nu = np.zeros(k), np.zeros(k)
    for j in range(k):
        best_row, worst_row = int(np.argmax(score_mat[:, j])), int(np.argmin(score_mat[:, j]))
        ai_mu[j], ai_nu[j] = mu_mat[best_row, j], nu_mat[best_row, j]
        aai_mu[j], aai_nu[j] = mu_mat[worst_row, j], nu_mat[worst_row, j]

    agg_mu, agg_nu = np.zeros(m), np.zeros(m)
    for i in range(m):
        agg_mu[i], agg_nu[i] = ffewa_aggregate(mu_mat[i, :], nu_mat[i, :], weights)

    ai_agg_mu, ai_agg_nu = ffewa_aggregate(ai_mu, ai_nu, weights)
    aai_agg_mu, aai_agg_nu = ffewa_aggregate(aai_mu, aai_nu, weights)

    S_i = score_function(agg_mu, agg_nu)
    S_AI = score_function(ai_agg_mu, ai_agg_nu) + 1.0
    S_AAI = score_function(aai_agg_mu, aai_agg_nu) + 1.0
    S_i_shift = S_i + 1.0

    K_plus = S_i_shift / S_AI
    K_minus = S_i_shift / S_AAI
    f_K = (K_plus + K_minus) / (1.0 + (K_plus / K_minus) + (K_minus / K_plus))

    out = pd.DataFrame({"Candidate_ID": ids, "Aggregated_Mu": agg_mu, "Aggregated_Nu": agg_nu,
                         "K_plus": K_plus, "K_minus": K_minus, "f_K": f_K})
    out = out.sort_values("f_K", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, m + 1))
    return out

# ============================================================================
# MASTER RANKING — REPRODUCE PHASE 4 BASELINE
# ============================================================================
print("\n" + "=" * 78)
print("MASTER BASELINE — RE-DERIVING PHASE 4 EINSTEIN-MARCOS RANKING")
print("=" * 78)

master_result = run_einstein_marcos(mu_full.values, nu_full.values, candidate_ids_all, weight_vector)
master_result = master_result.rename(columns={"Rank": "MARCOS_Rank", "f_K": "MARCOS_fK"})
master_lookup = master_result.set_index("Candidate_ID")
assert master_lookup.loc[50, "MARCOS_Rank"] == 1, "FAIL: Candidate 50 is not Master Rank 1 — methodology drift detected"
print(f"PASS | Candidate 50 confirmed Master Rank 1 (f(K)={master_lookup.loc[50,'MARCOS_fK']:.6f})")

master_top5_ids = master_result.sort_values("MARCOS_Rank").head(5)["Candidate_ID"].tolist()
print(f"PASS | Master Top 5 candidate IDs: {master_top5_ids}")

# Defuzzified crisp S matrix (shifted), used by all four crisp MCDM engines
S_matrix = score_function(mu_full.values, nu_full.values) + 1.0   # (293, 9), all benefit-oriented
S_df = pd.DataFrame(S_matrix, index=mu_full.index, columns=CRITERION_COLUMNS)

print(f"\nS matrix (defuzzified, shifted) range: [{S_matrix.min():.4f}, {S_matrix.max():.4f}]")

# ============================================================================
# TEST 1 — BIG FIVE MCDM ENGINE SUPER-ROSTER
# ============================================================================
print("\n" + "=" * 78)
print("TEST 1 — BIG FIVE MCDM ENGINE SUPER-ROSTER")
print("=" * 78)
# All four engines below operate on the SAME crisp, defuzzified, shifted
# score matrix S (293 x 9, all benefit-oriented) and the same weight vector
# as the Master MARCOS run, per explicit task specification.

def rank_from_scores(scores, ascending):
    """Rank 1 = best. ascending=True means lower score is better (VIKOR)."""
    order = np.argsort(scores) if ascending else np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

# --- Fermatean VIKOR (v = 0.5) ---
f_star = S_matrix.max(axis=0)   # best per criterion (all benefit)
f_minus = S_matrix.min(axis=0)  # worst per criterion
denom = np.where((f_star - f_minus) == 0, EPS, f_star - f_minus)
weighted_gap = weight_vector * (f_star - S_matrix) / denom

S_vikor = weighted_gap.sum(axis=1)             # utility measure
R_vikor = weighted_gap.max(axis=1)             # regret measure
S_best, S_worst = S_vikor.min(), S_vikor.max()
R_best, R_worst = R_vikor.min(), R_vikor.max()
v = 0.5
Q_vikor = (v * (S_vikor - S_best) / max(S_worst - S_best, EPS) +
           (1 - v) * (R_vikor - R_best) / max(R_worst - R_best, EPS))
VIKOR_rank = rank_from_scores(Q_vikor, ascending=True)  # lower Q = better
print("PASS | Fermatean VIKOR (v=0.5) computed")

# --- Fermatean COPRAS (all-benefit case) ---
col_sums = S_matrix.sum(axis=0)
r_normalized = S_matrix / col_sums
x_weighted = r_normalized * weight_vector
Q_copras = x_weighted.sum(axis=1)   # S-_i = 0 for all candidates (no cost criteria) -> Q_i = S+_i
COPRAS_rank = rank_from_scores(Q_copras, ascending=False)  # higher Q = better
print("PASS | Fermatean COPRAS (all-benefit degenerate case) computed")

# --- Fermatean WASPAS (lambda = 0.5) ---
col_max = S_matrix.max(axis=0)
r_waspas = S_matrix / col_max
Q1_wsm = (r_waspas * weight_vector).sum(axis=1)
Q2_wpm = np.prod(r_waspas ** weight_vector, axis=1)
lam = 0.5
Q_waspas = lam * Q1_wsm + (1 - lam) * Q2_wpm
WASPAS_rank = rank_from_scores(Q_waspas, ascending=False)
print("PASS | Fermatean WASPAS (lambda=0.5) computed")

# --- Fermatean EDAS ---
AV = S_matrix.mean(axis=0)
PDA = np.maximum(0, (S_matrix - AV)) / AV
NDA = np.maximum(0, (AV - S_matrix)) / AV
SP = (weight_vector * PDA).sum(axis=1)
SN = (weight_vector * NDA).sum(axis=1)
NSP = SP / max(SP.max(), EPS)
NSN = 1 - SN / max(SN.max(), EPS)
AS_edas = (NSP + NSN) / 2
EDAS_rank = rank_from_scores(AS_edas, ascending=False)
print("PASS | Fermatean EDAS computed")

bigfive_df = pd.DataFrame({
    "Candidate_ID": candidate_ids_all,
    "MARCOS_Rank": master_lookup.loc[candidate_ids_all, "MARCOS_Rank"].values,
    "VIKOR_Q": Q_vikor, "VIKOR_Rank": VIKOR_rank,
    "COPRAS_Q": Q_copras, "COPRAS_Rank": COPRAS_rank,
    "WASPAS_Q": Q_waspas, "WASPAS_Rank": WASPAS_rank,
    "EDAS_AS": AS_edas, "EDAS_Rank": EDAS_rank,
}).sort_values("MARCOS_Rank").reset_index(drop=True)

print(f"\nTop 3 across all five engines:")
print(bigfive_df.head(3)[["Candidate_ID", "MARCOS_Rank", "VIKOR_Rank", "COPRAS_Rank", "WASPAS_Rank", "EDAS_Rank"]].to_string(index=False))
print(f"\nBottom 3 across all five engines:")
print(bigfive_df.tail(3)[["Candidate_ID", "MARCOS_Rank", "VIKOR_Rank", "COPRAS_Rank", "WASPAS_Rank", "EDAS_Rank"]].to_string(index=False))

# ============================================================================
# TEST 2 — BELTON-GEAR RANK REVERSAL STRESS TEST
# ============================================================================
print("\n" + "=" * 78)
print("TEST 2 — BELTON-GEAR RANK REVERSAL STRESS TEST")
print("=" * 78)

master_sorted_ids = master_result.sort_values("MARCOS_Rank")["Candidate_ID"].values

def truncated_subset_marcos(drop_fraction):
    drop_n = round(drop_fraction * n_candidates)
    keep_ids = master_sorted_ids[: n_candidates - drop_n]   # drop bottom `drop_n` by master rank
    keep_mask = mu_full.index.isin(keep_ids)
    sub_mu = mu_full.values[keep_mask]
    sub_nu = nu_full.values[keep_mask]
    sub_ids = mu_full.index.values[keep_mask]
    result = run_einstein_marcos(sub_mu, sub_nu, sub_ids, weight_vector)
    return result, drop_n, len(sub_ids)

subset_A_result, dropA_n, keepA_n = truncated_subset_marcos(0.10)
subset_B_result, dropB_n, keepB_n = truncated_subset_marcos(0.25)

subA_lookup = subset_A_result.set_index("Candidate_ID")
subB_lookup = subset_B_result.set_index("Candidate_ID")

print(f"Subset A: dropped bottom {dropA_n} candidates ({dropA_n/n_candidates:.1%}), kept {keepA_n}")
print(f"Subset B: dropped bottom {dropB_n} candidates ({dropB_n/n_candidates:.1%}), kept {keepB_n}")

belton_gear_rows = []
for cid in master_top5_ids[:3]:  # Top 3 master candidates, per Table C4 spec
    master_rank = int(master_lookup.loc[cid, "MARCOS_Rank"])
    rankA = int(subA_lookup.loc[cid, "Rank"]) if cid in subA_lookup.index else None
    rankB = int(subB_lookup.loc[cid, "Rank"]) if cid in subB_lookup.index else None
    passA = (rankA == master_rank)
    passB = (rankB == master_rank)
    belton_gear_rows.append({
        "Candidate_ID": cid, "Master_Rank": master_rank,
        "Rank_Subset_A_10pct": rankA, "Status_A": "PASS" if passA else "FAIL (reordered)",
        "Rank_Subset_B_25pct": rankB, "Status_B": "PASS" if passB else "FAIL (reordered)",
    })
belton_gear_df = pd.DataFrame(belton_gear_rows)
print("\nBelton-Gear Rank Reversal Log (Top 3 Master Candidates):")
print(belton_gear_df.to_string(index=False))

candidate_50_rank_A = int(subA_lookup.loc[50, "Rank"])
candidate_50_rank_B = int(subB_lookup.loc[50, "Rank"])
print(f"\nCandidate 50 rank under Subset A (10% removed) : {candidate_50_rank_A}")
print(f"Candidate 50 rank under Subset B (25% removed) : {candidate_50_rank_B}")

# ============================================================================
# TEST 3 — AI-CRITERION WEIGHT PERTURBATION SWEEP
# ============================================================================
print("\n" + "=" * 78)
print("TEST 3 — AI-CRITERION WEIGHT PERTURBATION SWEEP")
print("=" * 78)

AI_COL_IDX = CRITERION_COLUMNS.index("Smart_AI_Performance_Criteria")
other_cols_idx = [j for j in range(n_criteria) if j != AI_COL_IDX]
base_other_weights = weight_vector[other_cols_idx]
base_other_sum = base_other_weights.sum()  # = 0.75 (the non-AI budget in the master vector)

ai_weight_scenarios = [0.10, 0.15, 0.25, 0.35, 0.45]
scenario_labels = [f"W{i+1} (AI={w:.2f})" for i, w in enumerate(ai_weight_scenarios)]

sweep_results = {}   # label -> full ranking DataFrame
sweep_weight_vectors = {}
for label, ai_w in zip(scenario_labels, ai_weight_scenarios):
    scaled_others = base_other_weights * (1.0 - ai_w) / base_other_sum
    new_vector = np.zeros(n_criteria)
    new_vector[AI_COL_IDX] = ai_w
    for idx, j in enumerate(other_cols_idx):
        new_vector[j] = scaled_others[idx]
    assert np.isclose(new_vector.sum(), 1.0, atol=1e-9), f"FAIL: scenario {label} weights sum to {new_vector.sum()}"
    sweep_weight_vectors[label] = new_vector

    result = run_einstein_marcos(mu_full.values, nu_full.values, candidate_ids_all, new_vector)
    sweep_results[label] = result.set_index("Candidate_ID")
    print(f"PASS | {label}: weight vector sums to 1.0, MARCOS re-run on all 293 candidates")

trajectory_rows = []
for cid in master_top5_ids:
    row = {"Candidate_ID": cid, "Master_Rank": int(master_lookup.loc[cid, "MARCOS_Rank"])}
    for label in scenario_labels:
        row[label] = int(sweep_results[label].loc[cid, "Rank"])
    trajectory_rows.append(row)
trajectory_df = pd.DataFrame(trajectory_rows)
print("\nWeight Perturbation Trajectory (Master Top 5 candidates):")
print(trajectory_df.to_string(index=False))

# ============================================================================
# TEST 4 — MACHINE LEARNING BACKBONE SENSITIVITY
# ============================================================================
print("\n" + "=" * 78)
print("TEST 4 — ML BACKBONE SENSITIVITY (XGBoost -> HistGradientBoostingClassifier)")
print("=" * 78)
# NOTE: see module docstring — using the classifier variant (predict_proba)
# to preserve architectural parity with the original LR+XGB averaging scheme.

ml_df = pd.read_excel(ML_PATH, sheet_name="ML_Scoring_Matrix")
feature_cols = [
    "PILLAR_Experience_NORM", "PILLAR_Education_NORM",
    "PILLAR_Technical_Skills_Certifications_NORM", "PILLAR_Psychological_Composite_NORM",
    "PILLAR_Adaptability_TimeManagement_NORM", "PILLAR_Cultural_Fit_Creativity_NORM",
]
X = ml_df[feature_cols].values
y = ml_df["Target_ML_Performance"].values
ml_ids = ml_df["ID"].values

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ml_ids, test_size=59, random_state=42, stratify=y
)
print(f"PASS | Replicated documented split: train={len(X_train)}, test={len(X_test)} (stratified, seed=42)")

hgb_model = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
hgb_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score
test_mask = np.isin(ml_ids, ids_test)
hgb_test_acc = accuracy_score(y[test_mask], hgb_model.predict(X[test_mask]))
hgb_test_auc = roc_auc_score(y[test_mask], hgb_model.predict_proba(X[test_mask])[:, 1])
print(f"PASS | HistGradientBoostingClassifier — Test Acc={hgb_test_acc:.4f}, Test AUC={hgb_test_auc:.4f}")

hgb_proba_all = hgb_model.predict_proba(X)[:, 1]  # probability for ALL 293, matching original architecture
alt_smart_ai = (ml_df["ML_Prob_LR"].values + hgb_proba_all) / 2.0  # LR backbone unchanged, only boosting model swapped
alt_smart_ai_df = pd.DataFrame({"Candidate_ID": ml_ids, "Smart_AI_Performance_Criteria_Alt": alt_smart_ai}).set_index("Candidate_ID").sort_index()

print(f"PASS | Alt Smart_AI_Performance_Criteria computed for all {len(alt_smart_ai)} candidates")
print(f"  Correlation (original vs alt AI score): {np.corrcoef(ml_df.set_index('ID').sort_index()['Smart_AI_Performance_Criteria'].values, alt_smart_ai_df['Smart_AI_Performance_Criteria_Alt'].values)[0,1]:.4f}")

# --- Re-fuzzify ONLY the Smart_AI_Performance_Criteria column with the new crisp values ---
# Re-use the ORIGINAL ESCO contextual modifier per candidate (it is a property of the
# candidate's Department, independent of the crisp ML score), per Phase-3 methodology.
ai_modifier = esco_modifier_full["Smart_AI_Performance_Criteria"].sort_index().values
ai_crisp_alt = alt_smart_ai_df.loc[mu_full.index, "Smart_AI_Performance_Criteria_Alt"].values

mu_ai_alt = ai_crisp_alt  # beneficial criterion
nu_ai_alt = 1.0 - ai_crisp_alt
cubic_base = np.minimum(mu_ai_alt ** 3 + nu_ai_alt ** 3, 1.0)
pi_base_alt = (1.0 - cubic_base) ** (1.0 / 3.0)
pi_inflated_alt = pi_base_alt * (1.0 + ai_modifier)
total_alt = mu_ai_alt ** 3 + nu_ai_alt ** 3 + pi_inflated_alt ** 3
scale_alt = (1.0 / total_alt) ** (1.0 / 3.0)
mu_ai_final = mu_ai_alt * scale_alt
nu_ai_final = nu_ai_alt * scale_alt

mu_alt_full = mu_full.copy()
nu_alt_full = nu_full.copy()
mu_alt_full["Smart_AI_Performance_Criteria"] = mu_ai_final
nu_alt_full["Smart_AI_Performance_Criteria"] = nu_ai_final

alt_ml_result = run_einstein_marcos(mu_alt_full.values, nu_alt_full.values, candidate_ids_all, weight_vector)
alt_ml_result = alt_ml_result.rename(columns={"Rank": "Rank_Alt_ML", "f_K": "f_K_Alt_ML"})
alt_ml_lookup = alt_ml_result.set_index("Candidate_ID")

print(f"\nCandidate 50 — Master Rank: {int(master_lookup.loc[50,'MARCOS_Rank'])}, Alt-ML Rank: {int(alt_ml_lookup.loc[50,'Rank_Alt_ML'])}")
print(f"Top 5 Master candidates' Alt-ML ranks: {[int(alt_ml_lookup.loc[c,'Rank_Alt_ML']) for c in master_top5_ids]}")

# ============================================================================
# TEST 5 — TOP-HEAVY NON-PARAMETRIC CORRELATION
# ============================================================================
print("\n" + "=" * 78)
print("TEST 5 — TOP-HEAVY NON-PARAMETRIC CORRELATION")
print("=" * 78)

def weighted_spearman(rank_x, rank_y, n):
    """
    da Costa & Soares (2005) weighted rank correlation coefficient.
    Penalizes disagreements at the top of the ranking more heavily than
    disagreements at the bottom — the 'top-heavy' correlation requested.
        r_w = 1 - [6 * sum((R_i-Q_i)^2 * ((n-R_i+1)+(n-Q_i+1)))] / (n^4+n^3-n^2-n)
    """
    diff2 = (rank_x - rank_y) ** 2
    weight = (n - rank_x + 1) + (n - rank_y + 1)
    numerator = 6.0 * np.sum(diff2 * weight)
    denominator = n ** 4 + n ** 3 - n ** 2 - n
    return 1.0 - numerator / denominator

rank_vectors = {
    "MARCOS (Master)": master_lookup.loc[candidate_ids_all, "MARCOS_Rank"].values,
    "VIKOR": bigfive_df.set_index("Candidate_ID").loc[candidate_ids_all, "VIKOR_Rank"].values,
    "COPRAS": bigfive_df.set_index("Candidate_ID").loc[candidate_ids_all, "COPRAS_Rank"].values,
    "WASPAS": bigfive_df.set_index("Candidate_ID").loc[candidate_ids_all, "WASPAS_Rank"].values,
    "EDAS": bigfive_df.set_index("Candidate_ID").loc[candidate_ids_all, "EDAS_Rank"].values,
    "Alt-ML (HistGB)": alt_ml_lookup.loc[candidate_ids_all, "Rank_Alt_ML"].values,
}
method_names = list(rank_vectors.keys())
n_methods = len(method_names)

spearman_matrix = np.eye(n_methods)
kendall_matrix = np.eye(n_methods)
rw_matrix = np.eye(n_methods)

for i in range(n_methods):
    for j in range(n_methods):
        if i == j:
            continue
        rx, ry = rank_vectors[method_names[i]], rank_vectors[method_names[j]]
        spearman_matrix[i, j] = spearmanr(rx, ry).statistic
        kendall_matrix[i, j] = kendalltau(rx, ry).statistic
        rw_matrix[i, j] = weighted_spearman(rx, ry, n_candidates)

spearman_df = pd.DataFrame(spearman_matrix, index=method_names, columns=method_names)
kendall_df = pd.DataFrame(kendall_matrix, index=method_names, columns=method_names)
rw_df = pd.DataFrame(rw_matrix, index=method_names, columns=method_names)

print("Spearman's rho matrix:")
print(spearman_df.round(4).to_string())
print("\nKendall's tau-b matrix:")
print(kendall_df.round(4).to_string())
print("\nWeighted Spearman (r_w) matrix:")
print(rw_df.round(4).to_string())

# ============================================================================
# FIGURE C2 — 2x2 PUBLICATION-GRADE ROBUSTNESS PANEL
# ============================================================================
print("\n" + "=" * 78)
print("GENERATING FIGURE C2 — MASTER ROBUSTNESS PANEL")
print("=" * 78)

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, 2, figsize=(15, 13))

# --- Top-Left: r_w Heatmap ---
ax = axes[0, 0]
sns.heatmap(rw_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
            square=True, linewidths=0.5, cbar_kws={"label": "Weighted Spearman ($r_w$)"}, ax=ax)
ax.set_title("(A) Weighted Spearman ($r_w$) Correlation Matrix", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.tick_params(axis="y", rotation=0, labelsize=8)

# --- Top-Right: Top 5 Weight Sweep Trajectory ---
ax = axes[0, 1]
x_positions = np.arange(len(ai_weight_scenarios))
palette = sns.color_palette("tab10", n_colors=len(master_top5_ids))
for color, cid in zip(palette, master_top5_ids):
    ranks = [int(sweep_results[label].loc[cid, "Rank"]) for label in scenario_labels]
    ax.plot(x_positions, ranks, marker="o", linewidth=2, color=color, label=f"Candidate #{cid}")
ax.set_xticks(x_positions)
ax.set_xticklabels([f"{w:.2f}" for w in ai_weight_scenarios])
ax.set_xlabel("Smart_AI_Performance_Criteria Weight")
ax.set_ylabel("Rank (1 = Best)")
ax.invert_yaxis()
ax.set_title("(B) Top-5 Weight Sweep Rank Trajectory", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="best")
ax.axvline(x=2, color="gray", linestyle="--", alpha=0.5, linewidth=1)

# --- Bottom-Left: Rank Bump Chart (Top 10, 5 engines) ---
ax = axes[1, 0]
top10_ids = master_result.sort_values("MARCOS_Rank").head(10)["Candidate_ID"].tolist()
engine_names = ["MARCOS_Rank", "VIKOR_Rank", "COPRAS_Rank", "WASPAS_Rank", "EDAS_Rank"]
engine_labels = ["MARCOS", "VIKOR", "COPRAS", "WASPAS", "EDAS"]
bigfive_lookup = bigfive_df.set_index("Candidate_ID")
palette10 = sns.color_palette("tab10", n_colors=10)
for color, cid in zip(palette10, top10_ids):
    ranks = [int(bigfive_lookup.loc[cid, e]) for e in engine_names]
    ax.plot(range(len(engine_labels)), ranks, marker="o", linewidth=1.8, color=color, label=f"#{cid}")
ax.set_xticks(range(len(engine_labels)))
ax.set_xticklabels(engine_labels)
ax.set_ylabel("Rank (1 = Best)")
ax.invert_yaxis()
ax.set_title("(C) Rank Bump Chart — Top 10 Across 5 MCDM Engines", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))

# --- Bottom-Right: Alt-ML vs Master Rank Scatter ---
ax = axes[1, 1]
master_ranks_plot = master_lookup.loc[candidate_ids_all, "MARCOS_Rank"].values
alt_ranks_plot = alt_ml_lookup.loc[candidate_ids_all, "Rank_Alt_ML"].values
ax.scatter(master_ranks_plot, alt_ranks_plot, alpha=0.5, s=22, color="#2E75B6", edgecolor="white", linewidth=0.3)
ax.plot([1, n_candidates], [1, n_candidates], color="red", linestyle="--", linewidth=1.5, label="y = x (perfect agreement)")
ax.set_xlabel("Master Rank (XGBoost Backbone)")
ax.set_ylabel("Alt-ML Rank (HistGradientBoosting Backbone)")
ax.set_title(f"(D) Alt-ML vs Master Rank ($r_w$={rw_df.loc['MARCOS (Master)','Alt-ML (HistGB)']:.4f})", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"PASS | Figure saved -> {OUTPUT_FIG}")

# ============================================================================
# EXPORT — MULTI-SHEET ROBUSTNESS AUDIT WORKBOOK
# ============================================================================
print("\n" + "=" * 78)
print("EXPORTING Q1_Master_Robustness_Audit.xlsx")
print("=" * 78)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    bigfive_df.to_excel(writer, sheet_name="Test1_BigFive_Engines", index=False)
    belton_gear_df.to_excel(writer, sheet_name="Test2_RankReversal_Log", index=False)
    subset_A_result.to_excel(writer, sheet_name="Test2_SubsetA_Full", index=False)
    subset_B_result.to_excel(writer, sheet_name="Test2_SubsetB_Full", index=False)
    trajectory_df.to_excel(writer, sheet_name="Test3_WeightSweep_Top5", index=False)
    pd.DataFrame(sweep_weight_vectors, index=CRITERION_COLUMNS).to_excel(writer, sheet_name="Test3_WeightVectors")
    alt_ml_result.to_excel(writer, sheet_name="Test4_AltML_FullRanking", index=False)
    pd.DataFrame({"Metric": ["Test Accuracy", "Test ROC-AUC"], "HistGradientBoostingClassifier": [hgb_test_acc, hgb_test_auc]}).to_excel(writer, sheet_name="Test4_AltML_Performance", index=False)
    spearman_df.to_excel(writer, sheet_name="Test5_Spearman_Matrix")
    kendall_df.to_excel(writer, sheet_name="Test5_KendallTaub_Matrix")
    rw_df.to_excel(writer, sheet_name="Test5_WeightedSpearman_Matrix")
    master_result.to_excel(writer, sheet_name="Master_Baseline_FullRanking", index=False)

print(f"EXPORT COMPLETE -> {OUTPUT_XLSX}")
print("=" * 78)
print("ALL 5 TESTS COMPLETE")
print("=" * 78)





