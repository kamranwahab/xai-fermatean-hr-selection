"""
================================================================================
CANONICAL SHAP REGENERATION — LAYER 2 (6-FEATURE MODEL ONLY)
================================================================================
Strictly reproduces the documented Pipeline_Audit_Log configuration:
  Feature cols      : 6 NORM pillars (Experience, Education, Technical Skills
                       & Certifications, Psychological Composite, Adaptability
                       & Time Management, Cultural Fit & Creativity)
  Train/test split  : Train=234, Test=59, stratified, seed=42
  XGBoost           : n_estimators=300, max_depth=4, learning_rate=0.05

Location_Logistics and Salary are explicitly EXCLUDED — they are Phase-4
MARCOS criteria, never Layer-2 predictive features. This script supersedes
and discards the three previously-uploaded 7-feature SHAP images.
================================================================================
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

ML_PATH = "/mnt/user-data/uploads/ML_Augmented_Mendeley_Matrix.xlsx"
OUT_DIR = "/mnt/user-data/outputs"

FEATURE_COLS = [
    "PILLAR_Experience_NORM",
    "PILLAR_Education_NORM",
    "PILLAR_Technical_Skills_Certifications_NORM",
    "PILLAR_Psychological_Composite_NORM",
    "PILLAR_Adaptability_TimeManagement_NORM",
    "PILLAR_Cultural_Fit_Creativity_NORM",
]
FEATURE_LABELS = {
    "PILLAR_Experience_NORM": "Work Experience",
    "PILLAR_Education_NORM": "Education Level",
    "PILLAR_Technical_Skills_Certifications_NORM": "Technical & Certifications",
    "PILLAR_Psychological_Composite_NORM": "Psychological Composite",
    "PILLAR_Adaptability_TimeManagement_NORM": "Adaptability & Time Mgmt",
    "PILLAR_Cultural_Fit_Creativity_NORM": "Cultural Fit & Teamwork",
}

print("=" * 78)
print("STEP 1 — REPRODUCE DOCUMENTED 6-FEATURE MODEL")
print("=" * 78)

ml_df = pd.read_excel(ML_PATH, sheet_name="ML_Scoring_Matrix")
X = ml_df[FEATURE_COLS].values
y = ml_df["Target_ML_Performance"].values
ids = ml_df["ID"].values
feature_display = [FEATURE_LABELS[c] for c in FEATURE_COLS]

assert X.shape == (293, 6), f"FAIL: expected (293,6) feature matrix, got {X.shape}"
print(f"PASS | Feature matrix shape: {X.shape} (6 canonical pillars, NO Location_Logistics, NO Salary)")

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=59, random_state=42, stratify=y
)
print(f"PASS | Train={len(X_train)}, Test={len(X_test)} (documented: 234/59)")

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", enable_categorical=False,
)
model.fit(X_train, y_train)

test_acc = accuracy_score(y_test, model.predict(X_test))
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"INFO | Reproduced Test Accuracy: {test_acc:.4f}  (audit log documented: 0.9492)")
print(f"INFO | Reproduced Test AUC     : {test_auc:.4f}  (audit log documented: 0.9942)")
corr_to_documented = np.corrcoef(model.predict_proba(X)[:, 1], ml_df["ML_Prob_XGB"].values)[0, 1]
print(f"INFO | Correlation to saved ML_Prob_XGB column: {corr_to_documented:.6f}")
print("NOTE | Exact metric reproduction is not guaranteed across XGBoost library")
print("       versions even with a fixed seed (tree tie-breaking can differ).")
print("       This is flagged as a reproducibility caveat in the forensic audit,")
print("       not silently smoothed over.")

print("\n" + "=" * 78)
print("STEP 2 — SHAP TreeExplainer (PROBABILITY SPACE, INTERVENTIONAL)")
print("=" * 78)

background_masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
explainer = shap.TreeExplainer(model, data=background_masker, model_output="probability")
shap_values = explainer.shap_values(X)
base_value = float(explainer.expected_value)
print(f"PASS | SHAP values computed: shape {shap_values.shape}")
print(f"PASS | Base value E[f(X)] = {base_value:.4f}")

# Sanity check: base_value + sum(shap_row) should reconstruct predict_proba closely
proba_all = model.predict_proba(X)[:, 1]
reconstructed = base_value + shap_values.sum(axis=1)
max_reconstruction_error = np.abs(reconstructed - proba_all).max()
print(f"PASS | Max |reconstructed - predict_proba| = {max_reconstruction_error:.6f} "
      f"({'OK, interventional approx' if max_reconstruction_error < 0.05 else 'WARNING: large gap'})")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"Feature": feature_display, "Mean_Abs_SHAP": mean_abs_shap}).sort_values(
    "Mean_Abs_SHAP", ascending=False
)
print("\nGlobal feature importance (mean |SHAP|, probability space):")
print(importance_df.to_string(index=False))

print("\n" + "=" * 78)
print("STEP 3 — LOCAL EXPLANATIONS: CANDIDATE 50 (RANK #1) AND 291 (RANK #293)")
print("=" * 78)

local_explanations = {}
for target_id, label in [(50, "Rank #1"), (291, "Rank #293")]:
    row_idx = np.where(ids == target_id)[0][0]
    fx = base_value + shap_values[row_idx].sum()
    print(f"\n--- Candidate ID {target_id} ({label}) — f(x) = {fx:.4f}, E[f(X)] = {base_value:.4f} ---")
    rows = []
    for j in range(len(FEATURE_COLS)):
        rows.append({
            "Feature": feature_display[j],
            "Value": X[row_idx, j],
            "SHAP": shap_values[row_idx, j],
        })
    rows_sorted = sorted(rows, key=lambda r: -abs(r["SHAP"]))
    for r in rows_sorted:
        print(f"  {r['Feature']:<32s} value={r['Value']:.3f}  SHAP={r['SHAP']:+.4f}")
    local_explanations[target_id] = {"fx": fx, "rows": rows_sorted, "row_idx": row_idx}

print("\n" + "=" * 78)
print("STEP 4 — GENERATE FIGURES")
print("=" * 78)

plt.rcParams.update({"font.size": 13})

# --- Figure X1: Global SHAP Beeswarm ---
explanation_obj = shap.Explanation(
    values=shap_values, base_values=np.full(len(X), base_value), data=X,
    feature_names=feature_display,
)
plt.figure(figsize=(11, 7))
shap.plots.beeswarm(explanation_obj, show=False, max_display=6)
plt.title("SHAP Global Feature Attribution (Predictive AI Layer — Canonical 6-Feature Model)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Figure_X1_Global_SHAP_Beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"PASS | Saved {OUT_DIR}/Figure_X1_Global_SHAP_Beeswarm.png")

# --- Figure X2: Local Waterfall, Candidate 50 (Rank #1) ---
row_idx_50 = local_explanations[50]["row_idx"]
exp_50 = shap.Explanation(
    values=shap_values[row_idx_50], base_values=base_value, data=X[row_idx_50],
    feature_names=feature_display,
)
plt.figure(figsize=(11, 7))
shap.plots.waterfall(exp_50, show=False, max_display=6)
plt.title("SHAP Local Waterfall Explanation — Rank #1 (Candidate ID: 50)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Figure_X2_Local_Waterfall_Rank1.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"PASS | Saved {OUT_DIR}/Figure_X2_Local_Waterfall_Rank1.png")

# --- Figure X3: Local Waterfall, Candidate 291 (Rank #293) ---
row_idx_291 = local_explanations[291]["row_idx"]
exp_291 = shap.Explanation(
    values=shap_values[row_idx_291], base_values=base_value, data=X[row_idx_291],
    feature_names=feature_display,
)
plt.figure(figsize=(11, 7))
shap.plots.waterfall(exp_291, show=False, max_display=6)
plt.title("SHAP Local Waterfall Explanation — Rank #293 (Candidate ID: 291)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Figure_X3_Local_Waterfall_RankLast.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"PASS | Saved {OUT_DIR}/Figure_X3_Local_Waterfall_RankLast.png")

print("\n" + "=" * 78)
print("CANONICAL SHAP REGENERATION COMPLETE")
print("=" * 78)
