import json
import os
import pandas as pd
from src.loader import load_candidates
from src.scoring import combine_scores

ARTIFACTS_DIR = "artifacts_sample"
CANDIDATES_PATH = "sampledata/candidates.jsonl"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

candidates = load_candidates(CANDIDATES_PATH)
feature_data = []
rerank_scores = {}
rerank_reasoning = {}
candidate_titles = []

for c in candidates:
    score, evidence = combine_scores(c)
    feature_data.append({
        "candidate_id": c["candidate_id"],
        "feature_score": score,
        "evidence": json.dumps(evidence),
    })
    rerank_scores[c["candidate_id"]] = 3
    rerank_reasoning[c["candidate_id"]] = "Neutral rerank placeholder"
    candidate_titles.append(c.get("profile", {}).get("current_title", ""))

features_path = os.path.join(ARTIFACTS_DIR, "features.parquet")
pd.DataFrame(feature_data).to_parquet(features_path, index=False)
print(f"Wrote {features_path}")

with open(os.path.join(ARTIFACTS_DIR, "rerank_scores.json"), "w", encoding="utf-8") as f:
    json.dump(rerank_scores, f)
print("Wrote rerank_scores.json")

with open(os.path.join(ARTIFACTS_DIR, "rerank_reasoning.json"), "w", encoding="utf-8") as f:
    json.dump(rerank_reasoning, f)
print("Wrote rerank_reasoning.json")

with open(os.path.join(ARTIFACTS_DIR, "candidate_titles.json"), "w", encoding="utf-8") as f:
    json.dump(candidate_titles, f)
print("Wrote candidate_titles.json")
