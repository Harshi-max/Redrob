"""Quick analysis of scoring distribution."""
from src.loader import load_candidates
from src.scoring import combine_scores

cands = load_candidates("sampledata/candidates.jsonl")

ml = []
hr = []
senior_ml = []
for c in cands:
    title = c["profile"]["current_title"].lower()
    if any(k in title for k in ["ml engineer", "machine learning", "ai engineer", "data scientist"]):
        ml.append(c)
    if "hr manager" in title:
        hr.append(c)
    if "senior" in title and ("ml" in title or "machine learning" in title or "ai" in title):
        senior_ml.append(c)

print(f"ML-related: {len(ml)}, HR managers: {len(hr)}, Senior ML: {len(senior_ml)}")

scored = []
for c in cands:
    s, ev = combine_scores(c)
    scored.append((s, c))

scored.sort(key=lambda x: -x[0])
print("\nTop 10:")
for s, c in scored[:10]:
    p = c["profile"]
    print(f"  {s:.4f} | {c['candidate_id']} | {p['current_title']} | {p['years_of_experience']:.1f}yr")

print("\nTop 5 Senior ML:")
senior_scored = [(combine_scores(c)[0], c) for c in senior_ml]
senior_scored.sort(key=lambda x: -x[0])
for s, c in senior_scored[:5]:
    p = c["profile"]
    print(f"  {s:.4f} | {c['candidate_id']} | {p['current_title']}")

print("\nTop 5 HR (should be low):")
hr_scored = [(combine_scores(c)[0], c) for c in hr[:200]]
hr_scored.sort(key=lambda x: -x[0])
for s, c in hr_scored[:5]:
    p = c["profile"]
    print(f"  {s:.4f} | {c['candidate_id']} | {p['current_title']}")
