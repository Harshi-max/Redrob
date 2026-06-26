"""Rule-based recruiter-style reranking without external LLM APIs."""

from src.config import (
    ALL_RELEVANT_SKILLS,
    POSITIVE_CAREER_PHRASES,
    RETRIEVAL_SKILLS,
    EMBEDDING_SKILLS,
    VECTOR_DB_SKILLS,
    skill_matches_taxonomy,
    count_taxonomy_skills,
)


STRONG_ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer", "senior ml engineer",
    "senior machine learning engineer", "senior ai engineer", "staff ml engineer",
    "staff machine learning engineer", "applied ml engineer", "applied scientist",
    "search engineer", "recommendation systems engineer", "retrieval engineer",
    "senior nlp engineer", "nlp engineer", "data scientist",
}

WEAK_TITLES = {
    "hr manager", "accountant", "content writer", "graphic designer",
    "sales executive", "customer support", "marketing manager", "operations manager",
    "civil engineer", "mechanical engineer", "project manager", "business analyst",
}


def _title_match(title: str, keywords: set[str]) -> bool:
    t = (title or "").lower()
    return any(kw in t for kw in keywords)


def local_rerank_candidate(
    candidate: dict,
    feature_score: float,
    evidence: dict | None = None,
) -> tuple[int, str]:
    """Return (1-5 score, short reasoning) mimicking recruiter judgment."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")

    career_text = " ".join(h.get("description", "") for h in career).lower()
    positive_hits = sum(1 for p in POSITIVE_CAREER_PHRASES if p in career_text)

    retrieval_skills = count_taxonomy_skills(skills, RETRIEVAL_SKILLS)
    embedding_skills = count_taxonomy_skills(skills, EMBEDDING_SKILLS)
    vector_skills = count_taxonomy_skills(skills, VECTOR_DB_SKILLS)
    core_skills = count_taxonomy_skills(skills, ALL_RELEVANT_SKILLS)

    # Base score from blended feature signal (0-1 -> 1-5)
    score = feature_score * 5.0

    if _title_match(title, STRONG_ML_TITLES):
        score = max(score, 3.5)
        if positive_hits >= 2 and yoe >= 5:
            score = max(score, 4.5)
        if retrieval_skills + embedding_skills >= 3 and yoe >= 5:
            score = max(score, 4.8)

    if _title_match(title, WEAK_TITLES):
        score = min(score, 2.0)
        if positive_hits == 0 and core_skills >= 5:
            score = min(score, 1.5)

    if yoe < 3:
        score = min(score, 3.0)

    if yoe >= 5 and positive_hits >= 1 and retrieval_skills + vector_skills >= 2:
        score = max(score, 4.0)

    github = signals.get("github_activity_score", -1)
    if github >= 60 and _title_match(title, STRONG_ML_TITLES):
        score = max(score, 4.2)

    score_int = max(1, min(5, round(score)))

    parts = [f"{title} at {company}, {yoe:.1f}yr"]
    if retrieval_skills + embedding_skills + vector_skills >= 2:
        parts.append(
            f"retrieval/embedding stack ({retrieval_skills + embedding_skills + vector_skills:.0f} skill points)"
        )
    if positive_hits:
        parts.append(f"{positive_hits} production ML signals in career history")
    resp = signals.get("recruiter_response_rate", 0)
    if resp >= 0.5:
        parts.append(f"response rate {resp:.0%}")

    reasoning = "; ".join(parts) + "."
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."

    return score_int, reasoning


def local_rerank_all(
    candidates: list[dict],
    feature_scores: dict[str, float],
    evidence_map: dict[str, dict] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    scores = {}
    reasoning = {}
    for c in candidates:
        cid = c["candidate_id"]
        fs = feature_scores.get(cid, 0.0)
        ev = (evidence_map or {}).get(cid)
        s, r = local_rerank_candidate(c, fs, ev)
        scores[cid] = s
        reasoning[cid] = r
    return scores, reasoning
