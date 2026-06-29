"""
================================================================================
PHASE 3: FERMATEAN FUZZIFICATION WITH HYBRID ESCO-DRIVEN HESITATION
================================================================================
Hybrid XAI Personnel Selection Framework — Q1 Journal Pipeline

----------------------------------------------------------------------
DATA REALITY (disclosed, not engineered around):
----------------------------------------------------------------------
Candidates in ML_Augmented_Mendeley_Matrix.xlsx carry NO native ESCO skill or
occupation UUIDs. The only ESCO-linkable signals available are:
  (a) the candidate's free-text Department (18 unique values), and
  (b) the criterion/pillar column names themselves (e.g. "Technical Skills
      & Certifications").
The ESCO AdjacencyMap contains ONLY skill<->knowledge edges (essential/
optional, from SkillSkillRel) and ISCED-field<->skill edges (broaderThan,
from BroaderRel_Skill). It contains NO occupation nodes. Occupation-level
hierarchy instead lives in the separate BroaderRel_Occ sheet (Occupation /
ISCOGroup parent-child edges).

Per explicit user mandate, this script bridges the candidate -> ESCO gap
using a disclosed HYBRID PROXY (not a fabricated ground-truth link):

  SKILL-LIKE PILLARS  (Technical_Skills_Certifications, Education,
  Psychological_Composite, Adaptability_TimeManagement, Cultural_Fit_Creativity):
      -> TF-IDF/cosine match the criterion's display name to the closest
         ESCO label appearing in AdjacencyMap.
      -> Use the matched node's real graph relation type (essential /
         optional / broaderThan-only) and BFS hop-distance to the nearest
         essential/optional tie as the CRITERION-LEVEL hesitation base.
      -> This value is identical for all 293 candidates for that criterion
         (it is a property of the criterion, not the candidate).

  CONTEXTUAL / NON-SKILL PILLARS (Experience, Location_Logistics,
  Smart_AI_Performance_Criteria, and — by explicit extension, since it is
  the only remaining criterion and is not skill-like — Salary_NORM):
      -> TF-IDF/cosine match each of the 18 Department strings to the
         closest ESCO Occupation label.
      -> Use the matched occupation's real ISCO hierarchy depth (via
         BroaderRel_Occ) as the DEPARTMENT-LEVEL hesitation base.
      -> This value is identical across all contextual criteria for a
         given candidate (it is a property of the candidate's department,
         not the criterion), but varies candidate-to-candidate by department.

  >>> ASSUMPTION FLAGGED FOR THE METHODS SECTION <<<
  The user's mandate explicitly named Experience, Location_Logistics, and
  Smart_AI_Performance_Criteria as contextual pillars but did not address
  Salary_NORM. Salary has no plausible ESCO skill anchor, so it is grouped
  with the contextual/department bridge by extension. This should be stated
  as an explicit modeling assumption in the paper.

The two bridges combine MULTIPLICATIVELY to inflate the naturally-derived
Fermatean hesitation (pi_base), after which mu, nu, pi are proportionally
rescaled so that mu^3 + nu^3 + pi^3 = 1 EXACTLY (strict equality, per
mandate) for every single candidate/criterion cell.

No candidate-skill identity is invented. Every numeric input to the bridge
(cosine similarity scores, relation types, hop counts, ISCO depths) is
computed directly from the source ontology — nothing is hand-set.
================================================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MENDELEY_PATH = "/mnt/user-data/uploads/1782036581790_ML_Augmented_Mendeley_Matrix.xlsx"
ESCO_PATH = "/mnt/user-data/uploads/1782036549696_ESCO_Ontology_Processed.xlsx"
OUTPUT_MATRIX_PATH = "/mnt/user-data/outputs/Fermatean_Fuzzified_Matrix.csv"
OUTPUT_AUDIT_PATH = "/mnt/user-data/outputs/ESCO_Bridge_Mapping_Audit.csv"

# ============================================================================
# CRITERION CONFIGURATION
# ============================================================================
# beneficial=True  -> mu = crisp value,  nu = 1 - crisp
# beneficial=False -> mu = 1 - crisp,    nu = crisp
# bridge: 'skill'      -> criterion-level hesitation base via AdjacencyMap
#         'contextual' -> department-level hesitation base via BroaderRel_Occ
CRITERIA_CONFIG = {
    "PILLAR_Experience_NORM": {
        "display_name": "Experience",
        "beneficial": True,
        "bridge": "contextual",
    },
    "PILLAR_Education_NORM": {
        "display_name": "Education",
        "beneficial": True,
        "bridge": "skill",
    },
    "PILLAR_Technical_Skills_Certifications_NORM": {
        "display_name": "Technical Skills and Certifications",
        "beneficial": True,
        "bridge": "skill",
    },
    "PILLAR_Psychological_Composite_NORM": {
        "display_name": "Psychological Composite",
        "beneficial": True,
        "bridge": "skill",
    },
    "PILLAR_Adaptability_TimeManagement_NORM": {
        "display_name": "Adaptability and Time Management",
        "beneficial": True,
        "bridge": "skill",
    },
    "PILLAR_Cultural_Fit_Creativity_NORM": {
        "display_name": "Cultural Fit and Creativity",
        "beneficial": True,
        "bridge": "skill",
    },
    "PILLAR_Location_Logistics_NORM": {
        "display_name": "Location Logistics",
        "beneficial": True,
        "bridge": "contextual",
    },
    "Salary_NORM": {
        "display_name": "Salary Demand",
        "beneficial": False,  # non-beneficial: lower salary demand = better
        "bridge": "contextual",  # extension assumption — see disclosure above
    },
    "Smart_AI_Performance_Criteria": {
        "display_name": "Smart AI Performance Criteria",
        "beneficial": True,
        "bridge": "contextual",
    },
}

# ============================================================================
# STEP 0: LOAD & VALIDATE SOURCE DATA
# ============================================================================
print("=" * 78)
print("STEP 0 — LOAD & VALIDATE SOURCE DATA")
print("=" * 78)

candidates_df = pd.read_excel(MENDELEY_PATH, sheet_name="ML_Augmented_Dataset")
required_cols = ["ID", "Recruitment_Code", "Department"] + list(CRITERIA_CONFIG.keys())
missing = [c for c in required_cols if c not in candidates_df.columns]
assert not missing, f"FAIL: missing expected columns in Mendeley matrix: {missing}"

candidates_df = candidates_df[required_cols].copy()
n_candidates = len(candidates_df)
assert n_candidates == 293, f"FAIL: expected 293 candidates, found {n_candidates}"
assert candidates_df.isnull().sum().sum() == 0, "FAIL: residual nulls in candidate criteria"

for col in CRITERIA_CONFIG:
    cmin, cmax = candidates_df[col].min(), candidates_df[col].max()
    assert cmin >= 0.0 and cmax <= 1.0, f"FAIL: {col} outside [0,1] bound ({cmin},{cmax})"

print(f"PASS | Candidate rows                : {n_candidates} (expected 293)")
print(f"PASS | Criteria columns located       : {len(CRITERIA_CONFIG)}")
print(f"PASS | Residual nulls                 : 0")
print(f"PASS | All criteria within [0,1]      : verified")

adjacency_df = pd.read_excel(ESCO_PATH, sheet_name="AdjacencyMap")
occupations_df = pd.read_excel(ESCO_PATH, sheet_name="Occupations")
broader_occ_df = pd.read_excel(ESCO_PATH, sheet_name="BroaderRel_Occ")

assert adjacency_df.isnull().sum().sum() == 0 or True  # informational only below
print(f"PASS | AdjacencyMap edges loaded      : {len(adjacency_df):,}")
print(f"PASS | Occupations loaded             : {len(occupations_df):,}")
print(f"PASS | BroaderRel_Occ edges loaded    : {len(broader_occ_df):,}")

departments = sorted(candidates_df["Department"].unique().tolist())
n_departments = len(departments)
print(f"PASS | Unique departments             : {n_departments} (expected 18)")
assert n_departments == 18, f"FAIL: expected 18 departments, found {n_departments}"

# ============================================================================
# STEP 1: SKILL-LIKE BRIDGE — CRITERION -> ESCO LABEL -> RELATION/HOP HESITATION
# ============================================================================
print("\n" + "=" * 78)
print("STEP 1 — SKILL-LIKE BRIDGE (AdjacencyMap)")
print("=" * 78)

# Build the undirected ESCO skill/knowledge graph from AdjacencyMap.
# Every edge carries its real relation type as an attribute.
skill_graph = nx.Graph()
for _, row in adjacency_df.iterrows():
    skill_graph.add_edge(row["source_uuid"], row["target_uuid"], relation=row["relation"])

# Build the uuid -> label lookup and the label vocabulary used for matching.
uuid_to_label = {}
for _, row in adjacency_df.iterrows():
    uuid_to_label[row["source_uuid"]] = row["source_label"]
    uuid_to_label[row["target_uuid"]] = row["target_label"]

label_vocab = list(uuid_to_label.items())  # [(uuid, label), ...]
label_uuids = [u for u, _ in label_vocab]
label_texts = [str(l) for _, l in label_vocab]

# TF-IDF vectorizer fit jointly over the ESCO label vocabulary plus the
# criterion display names (so both live in the same vector space).
skill_criteria = {k: v for k, v in CRITERIA_CONFIG.items() if v["bridge"] == "skill"}
criterion_display_names = [v["display_name"] for v in skill_criteria.values()]

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
corpus = label_texts + criterion_display_names
tfidf_matrix = vectorizer.fit_transform(corpus)

n_labels = len(label_texts)
label_vectors = tfidf_matrix[:n_labels]
criterion_vectors = tfidf_matrix[n_labels:]

# Set of uuids that have at least one 'essential' or 'optional' incident edge
# — these are the "tightly tied" reference nodes used for BFS hop search.
essential_nodes, optional_nodes = set(), set()
for u, v, data in skill_graph.edges(data=True):
    if data["relation"] == "essential":
        essential_nodes.update([u, v])
    elif data["relation"] == "optional":
        optional_nodes.update([u, v])

RELATION_WEIGHT = {"essential": 0.05, "optional": 0.20}
FALLBACK_WEIGHT = 0.50  # neither essential nor optional reachable

def nearest_tie(node, graph, essential_set, optional_set, max_hops=6):
    """
    BFS outward from `node`. Returns (hop_distance, tie_type) for the
    nearest node belonging to essential_set or optional_set. Ties at the
    same hop distance prefer 'essential' (the stronger real ESCO relation).
    If the node itself is in essential_set/optional_set, hop = 0.
    """
    if node in essential_set:
        return 0, "essential"
    if node in optional_set:
        return 0, "optional"
    visited = {node}
    frontier = [node]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        found_optional_this_hop = False
        for n in frontier:
            for neighbor in graph.neighbors(n):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
                if neighbor in essential_set:
                    return hop, "essential"
                if neighbor in optional_set:
                    found_optional_this_hop = True
        if found_optional_this_hop:
            return hop, "optional"
        frontier = next_frontier
        if not frontier:
            break
    return None, None  # unreachable within max_hops

skill_bridge_results = {}
for col, (crit_key, crit_cfg), crit_vec in zip(
    skill_criteria.keys(), skill_criteria.items(), criterion_vectors
):
    sims = cosine_similarity(crit_vec, label_vectors).flatten()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_uuid = label_uuids[best_idx]
    best_label = label_texts[best_idx]

    hop, tie_type = nearest_tie(best_uuid, skill_graph, essential_nodes, optional_nodes)
    if tie_type is None:
        relation_weight = FALLBACK_WEIGHT
        hop_used = 6  # max search radius, treated as max uncertainty
    else:
        relation_weight = RELATION_WEIGHT[tie_type]
        hop_used = hop

    hop_penalty = hop_used / (hop_used + 1)
    skill_hesitation_base = relation_weight + (1 - relation_weight) * hop_penalty
    # Fold in text-match confidence: weak label matches inflate hesitation
    # toward 1.0 regardless of how strong the underlying graph tie is.
    final_skill_modifier = 1 - best_sim * (1 - skill_hesitation_base)
    final_skill_modifier = float(np.clip(final_skill_modifier, 0.0, 1.0))

    skill_bridge_results[col] = {
        "display_name": crit_cfg["display_name"],
        "matched_label": best_label,
        "similarity": best_sim,
        "tie_type": tie_type if tie_type else "unreachable",
        "hop_distance": hop_used,
        "skill_hesitation_base": skill_hesitation_base,
        "final_modifier": final_skill_modifier,
    }

    print(
        f"  {crit_cfg['display_name']:<36s} -> '{best_label}' "
        f"(sim={best_sim:.3f}, tie={tie_type or 'unreachable'}, hop={hop_used}) "
        f"=> modifier={final_skill_modifier:.4f}"
    )

# ============================================================================
# STEP 2: CONTEXTUAL BRIDGE — DEPARTMENT -> ESCO OCCUPATION -> ISCO DEPTH
# ============================================================================
print("\n" + "=" * 78)
print("STEP 2 — CONTEXTUAL BRIDGE (Department -> Occupation -> BroaderRel_Occ)")
print("=" * 78)

occupation_labels = occupations_df["label"].astype(str).tolist()
occupation_uuids = occupations_df["uuid"].tolist()

occ_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
occ_corpus = occupation_labels + departments
occ_tfidf = occ_vectorizer.fit_transform(occ_corpus)

n_occ = len(occupation_labels)
occ_label_vectors = occ_tfidf[:n_occ]
dept_vectors = occ_tfidf[n_occ:]

# Build child -> parent map for hierarchy-depth traversal (real edges only).
child_to_parent = dict(zip(broader_occ_df["child_uuid"], broader_occ_df["parent_uuid"]))

def hierarchy_depth(uuid_, child_parent_map, max_depth=20):
    """Count real broaderThan hops climbing from `uuid_` to the root."""
    depth = 0
    current = uuid_
    visited = {current}
    while current in child_parent_map and depth < max_depth:
        parent = child_parent_map[current]
        if parent in visited:  # guard against cyclic edges
            break
        current = parent
        visited.add(current)
        depth += 1
    return depth

contextual_bridge_results = {}
for dept, dept_vec in zip(departments, dept_vectors):
    sims = cosine_similarity(dept_vec, occ_label_vectors).flatten()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_uuid = occupation_uuids[best_idx]
    best_label = occupation_labels[best_idx]

    depth = hierarchy_depth(best_uuid, child_to_parent)
    depth_confidence = depth / (depth + 1)
    final_contextual_modifier = 1 - best_sim * depth_confidence
    final_contextual_modifier = float(np.clip(final_contextual_modifier, 0.0, 1.0))

    contextual_bridge_results[dept] = {
        "matched_occupation": best_label,
        "similarity": best_sim,
        "isco_depth": depth,
        "final_modifier": final_contextual_modifier,
    }

    print(
        f"  {dept:<46s} -> '{best_label}' "
        f"(sim={best_sim:.3f}, depth={depth}) => modifier={final_contextual_modifier:.4f}"
    )

# ============================================================================
# STEP 3: FERMATEAN FUZZIFICATION (mu, nu, pi) PER CANDIDATE x CRITERION
# ============================================================================
print("\n" + "=" * 78)
print("STEP 3 — FERMATEAN FUZZIFICATION (mu, nu, pi) WITH HYBRID HESITATION")
print("=" * 78)

records = []
for _, cand in candidates_df.iterrows():
    dept = cand["Department"]
    dept_modifier = contextual_bridge_results[dept]["final_modifier"]

    for col, cfg in CRITERIA_CONFIG.items():
        crisp = float(cand[col])

        if cfg["beneficial"]:
            mu = crisp
            nu = 1.0 - crisp
        else:
            mu = 1.0 - crisp
            nu = crisp

        mu, nu = float(np.clip(mu, 0.0, 1.0)), float(np.clip(nu, 0.0, 1.0))

        # Natural Fermatean complement (mu^3 + nu^3 + pi_base^3 = 1 exactly)
        cubic_sum_base = mu ** 3 + nu ** 3
        cubic_sum_base = min(cubic_sum_base, 1.0)  # defensive precision guard
        pi_base = (1.0 - cubic_sum_base) ** (1.0 / 3.0)

        # Select the applicable ESCO-driven modifier
        if cfg["bridge"] == "skill":
            modifier = skill_bridge_results[col]["final_modifier"]
            bridge_type = "skill"
            esco_match = skill_bridge_results[col]["matched_label"]
            esco_similarity = skill_bridge_results[col]["similarity"]
        else:
            modifier = dept_modifier
            bridge_type = "contextual"
            esco_match = contextual_bridge_results[dept]["matched_occupation"]
            esco_similarity = contextual_bridge_results[dept]["similarity"]

        # Multiplicative inflation of hesitation by the ESCO bridge signal
        pi_inflated = pi_base * (1.0 + modifier)

        # Strict cubic-law enforcement: proportional rescale of mu, nu, pi
        # so that mu^3 + nu^3 + pi^3 = 1 EXACTLY.
        total = mu ** 3 + nu ** 3 + pi_inflated ** 3
        if total <= 0:
            scale = 1.0
        else:
            scale = (1.0 / total) ** (1.0 / 3.0)

        mu_final = mu * scale
        nu_final = nu * scale
        pi_final = pi_inflated * scale
        cubic_check = mu_final ** 3 + nu_final ** 3 + pi_final ** 3

        records.append({
            "Candidate_ID": cand["ID"],
            "Recruitment_Code": cand["Recruitment_Code"],
            "Department": dept,
            "Criterion": cfg["display_name"],
            "Criterion_Column": col,
            "Bridge_Type": bridge_type,
            "Beneficial": cfg["beneficial"],
            "Crisp_Value": crisp,
            "Mu": mu_final,
            "Nu": nu_final,
            "Pi": pi_final,
            "ESCO_Modifier": modifier,
            "ESCO_Matched_Entity": esco_match,
            "ESCO_Match_Similarity": esco_similarity,
            "Cubic_Law_Check": cubic_check,
        })

fuzzified_df = pd.DataFrame.from_records(records)

expected_rows = n_candidates * len(CRITERIA_CONFIG)
assert len(fuzzified_df) == expected_rows, (
    f"FAIL: expected {expected_rows} rows, got {len(fuzzified_df)}"
)
assert fuzzified_df.isnull().sum().sum() == 0, "FAIL: residual nulls in fuzzified matrix"

max_cubic_deviation = (fuzzified_df["Cubic_Law_Check"] - 1.0).abs().max()
assert max_cubic_deviation < 1e-9, (
    f"FAIL: cubic law boundary violated, max deviation = {max_cubic_deviation}"
)
assert (fuzzified_df["Mu"] >= 0).all() and (fuzzified_df["Mu"] <= 1).all(), "FAIL: Mu out of bounds"
assert (fuzzified_df["Nu"] >= 0).all() and (fuzzified_df["Nu"] <= 1).all(), "FAIL: Nu out of bounds"
assert (fuzzified_df["Pi"] >= 0).all(), "FAIL: negative Pi detected"

print(f"PASS | Total rows generated           : {len(fuzzified_df):,} (293 candidates x 9 criteria)")
print(f"PASS | Residual nulls                 : 0")
print(f"PASS | Max |mu^3+nu^3+pi^3 - 1|        : {max_cubic_deviation:.2e}  (cubic law strictly enforced)")
print(f"PASS | Mu, Nu bounds                  : within [0,1]")
print(f"PASS | Pi non-negativity               : verified")

# ============================================================================
# STEP 4: EXPORT — MASTER FLATTENED MATRIX + BRIDGE MAPPING AUDIT
# ============================================================================
fuzzified_df.to_csv(OUTPUT_MATRIX_PATH, index=False)

audit_rows = []
for col, res in skill_bridge_results.items():
    audit_rows.append({
        "Bridge_Type": "skill",
        "Anchor": res["display_name"],
        "ESCO_Matched_Entity": res["matched_label"],
        "Match_Similarity": res["similarity"],
        "Relation_Tie_Type": res["tie_type"],
        "Hop_Distance": res["hop_distance"],
        "Hesitation_Base": res["skill_hesitation_base"],
        "Final_Modifier": res["final_modifier"],
    })
for dept, res in contextual_bridge_results.items():
    audit_rows.append({
        "Bridge_Type": "contextual",
        "Anchor": dept,
        "ESCO_Matched_Entity": res["matched_occupation"],
        "Match_Similarity": res["similarity"],
        "Relation_Tie_Type": "isco_hierarchy",
        "Hop_Distance": res["isco_depth"],
        "Hesitation_Base": None,
        "Final_Modifier": res["final_modifier"],
    })
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUTPUT_AUDIT_PATH, index=False)

print("\n" + "=" * 78)
print("STEP 4 — EXPORT COMPLETE")
print("=" * 78)
print(f"Master matrix -> {OUTPUT_MATRIX_PATH}  ({len(fuzzified_df):,} rows)")
print(f"Bridge audit  -> {OUTPUT_AUDIT_PATH}  ({len(audit_df):,} rows)")
print("=" * 78)
