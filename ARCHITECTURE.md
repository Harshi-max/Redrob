# System Architecture

## Overall Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Intelligent Candidate Discovery System               │
│                           (Redrob Hackathon)                            │
└─────────────────────────────────────────────────────────────────────────┘

INPUT:
├─ candidates.jsonl (100k profiles)
├─ job_description.txt
└─ behavioral_signals

                              PREPROCESSING
                         (Can be offline/cached)
                                  ↓
                    ┌──────────────────────────┐
                    │   Stage 1: Parse JD      │
                    │ Extract requirements,    │
                    │ skills, constraints      │
                    └──────────────────────────┘
                                  ↓
                    ┌──────────────────────────┐
                    │ Stage 2: Parse Candidates│
                    │ Extract normalized       │
                    │ documents & features     │
                    └──────────────────────────┘
                                  ↓
                    ┌──────────────────────────┐
                    │ Stage 3: Embeddings      │
                    │ Generate + cache         │
                    │ semantic embeddings      │
                    └──────────────────────────┘

                           MAIN RANKING (CPU-only)
                        <5 min, <16GB RAM constraint
                                  ↓
                    ┌──────────────────────────┐
         ┌──────────▶│ Stage 4: Hybrid Retrieval│
         │           │ Semantic + BM25 + FAISS │
         │           │ Score with multi-signals │
         │           └──────────────────────────┘
         │                        ↓
         │           ┌──────────────────────────┐
    [Cached]────────▶│ Stage 5: Hard Filters    │
    [Embeddings]     │ Remove honeypots         │
                     │ & disqualified profiles  │
                     └──────────────────────────┘
                                  ↓
                     ┌──────────────────────────┐
                     │ Stage 6: Behavioral Sig. │
                     │ Score: response rate,    │
                     │ GitHub, notice, etc.     │
                     └──────────────────────────┘
                                  ↓
                     ┌──────────────────────────┐
                     │ Stage 7: Feature Eng.    │
                     │ Compute 10 features:     │
                     │ semantic, skills, exp,   │
                     │ startup, product, edu... │
                     └──────────────────────────┘
                                  ↓
                     ┌──────────────────────────┐
                     │ Stage 8: Final Ranking   │
                     │ Weighted score formula:  │
                     │ 0.40×sem + 0.20×skills..│
                     └──────────────────────────┘
                                  ↓
                     ┌──────────────────────────┐
                     │ Stage 9: Reasoning Gen.  │
                     │ 1-2 sentence facts       │
                     │ per candidate            │
                     └──────────────────────────┘
                                  ↓
                     ┌──────────────────────────┐
                     │ Stage 10: Output Fmt.    │
                     │ CSV: rank, id, score,    │
                     │ reasoning (100 rows)     │
                     └──────────────────────────┘

OUTPUT:
└─ submission.csv (100 top candidates with scores & reasoning)
```

---

## Component Architecture

### High-Level Module Organization

```
┌─────────────────────────────────────────────────────────┐
│                     Main Entry Point                    │
│                      main.py / rank.py                  │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┼────────┬─────────────────────┐
        │        │        │                     │
        ▼        ▼        ▼                     ▼
    ┌────┐  ┌────┐  ┌────┐                 ┌───────┐
    │app │  │src │  │mdl │                 │models │
    │ (new)  │    │  │    │                 │ (new) │
    └────┘  └────┘  └────┘                 └───────┘
        │        │        │
    ┌───┴────┐   │    ┌───┴─────┐          ┌────┬────┐
    │        │   │    │         │          │    │    │
 ranker.py  │   │ scr.py      │        JD  Cand
 feature    │   │           honey.py    Enc Enc
 embedding  │   └───────────────┘        (rule (norm
 retrieval      │                      extract)  profiles)
 scoring        config.py               │
 reasoning      loader.py               │
 utils.py       query.py                │
                reasoning.py     ┌──────┘
                retrieval.py     │
                scoring.py       │
                reranking.py     │
                pipeline.py      │
                                 │
                    ┌────────────┘
                    │
                    ▼
              (Shared Logic:
               embeddings,
               normalization,
               caching)
```

---

## Data Flow Diagram

### Stage-by-Stage Processing

```
INPUT: candidates.jsonl (100k profiles)
  │
  ├─ Profile Schema:
  │  ├─ profile (current_title, company, experience, location)
  │  ├─ skills (name, proficiency, endorsements)
  │  ├─ career_history (title, company, duration, description)
  │  ├─ education (degree, field, institution)
  │  └─ redrob_signals (behavioral data)
  │
  ▼

STAGE 1: Parse Candidates
  Input:  Raw candidate profiles
  Output: Normalized documents + extracted fields
  └─ Extract: skills_set, experience_summary, education_level, etc.

STAGE 2: Parse JD
  Input:  Job description text
  Output: Structured requirements
  └─ Extract: required_skills, location_constraints, experience_range, etc.

STAGE 3: Generate Embeddings
  Input:  JD summary + candidate documents
  Output: Embedding vectors (cached)
  └─ Using: sentence-transformers/all-MiniLM-L6-v2 (384-dim vectors)
  └─ Storage: artifacts/embeddings.npy, artifacts/query_embedding.npy
  └─ Index: FAISS for fast retrieval (artifacts/faiss.index)

STAGE 4: Hybrid Retrieval
  Input:  Embeddings + BM25 index
  Output: Retrieval scores (0-1) per candidate
  Formula:
    score = 0.45 × semantic_similarity +
            0.25 × skills_match +
            0.15 × experience_match +
            0.15 × behavioral_signals

STAGE 5: Apply Hard Filters
  Input:  Candidates + honeypot detection
  Output: Filtered candidate set (honeypots removed)
  Filters:
    - Honeypot detection (impossible profiles)
    - Pure research background
    - LangChain-only projects
    - Consulting company dominance
    - Domain mismatch (CV/Speech/Robotics-only)

STAGE 6: Behavioral Scoring
  Input:  Redrob signals (response_rate, github_score, notice_period, etc.)
  Output: Behavioral score (0-1) per candidate
  Components:
    - Recruiter response rate (30%)
    - Open to work flag (20%)
    - Notice period (20%)
    - GitHub activity score (20%)
    - Profile completeness (10%)

STAGE 7: Feature Engineering
  Input:  All previous scores + raw profiles
  Output: 10 normalized features per candidate
  Features:
    1. semantic_similarity        [0.40 weight]
    2. skills_overlap             [0.20 weight]
    3. behavior_score             [0.15 weight]
    4. production_ml_score        [0.10 weight]
    5. career_stability_score     [0.05 weight]
    6. startup_score              [0.05 weight]
    7. location_score             [0.05 weight]
    + years_experience_score
    + product_company_score
    + education_score

STAGE 8: Final Ranking
  Input:  7 weighted features
  Output: Final score (0-1) per candidate
  Formula:
    final_score = Σ(weight_i × feature_i)
    Normalized to [0, 1]

STAGE 9: Reasoning Generation
  Input:  Candidate profile + feature breakdown
  Output: 1-2 sentence justification (max 200 chars)
  Rules:
    - Mention actual facts only
    - Note concerns (if any)
    - Include specific metrics (years, skills, rates)
    - Avoid hallucinations

STAGE 10: Output Formatting
  Input:  Final scores + reasoning
  Output: CSV (100 rows)
  Columns:
    - rank (1-100)
    - candidate_id (unique ID)
    - score (0-1, monotonically decreasing)
    - reasoning (string, max 200 chars)

OUTPUT: submission.csv
```

---

## Performance Characteristics

### Time Complexity

| Stage | Time | Notes |
|-------|------|-------|
| 1-3: Preprocessing | ~2-3 min | Can be cached/offline |
| 4: Retrieval | ~30-40 sec | FAISS + BM25 |
| 5: Filters | ~5 sec | Quick checks |
| 6: Behavioral | ~10 sec | Signal aggregation |
| 7: Features | ~20 sec | Vectorized operations |
| 8: Ranking | ~5 sec | Weighted sum |
| 9: Reasoning | ~10 sec | Rule-based generation |
| 10: Output | ~2 sec | CSV formatting |
| **Total** | **<5 min** | Meets hackathon requirement |

### Memory Characteristics

| Component | Memory | Notes |
|-----------|--------|-------|
| Embeddings cache | ~300 MB | 100k × 384-dim × 4 bytes |
| FAISS index | ~400 MB | Approximate structure |
| Candidate profiles | ~500 MB | When fully loaded |
| Working memory | ~200 MB | Processing buffers |
| **Total** | **~1.5 GB** | Well under 16 GB limit |

---

## Caching Strategy

```
OFFLINE (Can use API/network):
  artifacts/
    ├─ embeddings.npy              [100k × 384]
    ├─ query_embedding.npy         [1 × 384]
    ├─ faiss.index                 [FAISS index]
    ├─ bm25_index.pkl              [BM25 index]
    ├─ ideal_profile.txt           [JD summary]
    └─ honeypot_flags.json         [Cached flags]

ONLINE (Main ranking, CPU-only, <5 min):
  1. Load cached artifacts
  2. Stream candidates from JSONL
  3. Score in batches
  4. Output results
```

---

## Optimization Techniques

### 1. **Embedding Batching**
- Process 32-64 texts per batch
- Vectorized operations in NumPy
- Reduce memory allocation overhead

### 2. **FAISS Indexing**
- Use IndexFlatL2 for exact retrieval
- Alternatively: IndexIVFFlat for approximate (larger datasets)
- Cosine similarity via normalized L2

### 3. **BM25 Caching**
- Pre-build BM25 index offline
- Serialize to disk (pickle)
- Load once at startup

### 4. **Memory Efficiency**
- Stream candidate processing (don't load all at once)
- Reuse numpy arrays
- Del intermediate results

### 5. **Vectorization**
- Use numpy operations (broadcast, einsum)
- Avoid Python loops for numerical work
- Leverage BLAS/LAPACK under the hood

---

## Scoring Formula Reference

### Hybrid Retrieval Score
$$\text{Retrieval} = 0.45 \times \text{Semantic} + 0.25 \times \text{Skills} + 0.15 \times \text{Experience} + 0.15 \times \text{Behavioral}$$

### Behavioral Score
$$\text{Behavioral} = 0.30 \times \text{ResponseRate} + 0.20 \times \text{OpenToWork} + 0.20 \times \text{NoticePeriod} + 0.20 \times \text{GitHub} + 0.10 \times \text{Completeness}$$

### Final Ranking Score
$$\text{Final} = 0.40 \times \text{Semantic} + 0.20 \times \text{Skills} + 0.15 \times \text{Behavioral} + 0.10 \times \text{ML} + 0.05 \times \text{Stability} + 0.05 \times \text{Startup} + 0.05 \times \text{Location}$$

All features normalized to $[0, 1]$.

---

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│          Docker Container               │
├─────────────────────────────────────────┤
│  Python 3.10 slim base image            │
│  All dependencies in requirements.txt   │
│  Mounts volumes for:                    │
│    - /app/sample (input data)           │
│    - /app/artifacts (cached)            │
│    - /app/outputs (results)             │
└─────────────────────────────────────────┘

Build:
  docker build -t candidate-ranker .

Run:
  docker run -v /path/to/data:/app/sample \
             -v /path/to/artifacts:/app/artifacts \
             -v /path/to/outputs:/app/outputs \
             candidate-ranker
```

---

## Error Handling & Validation

```
Input Validation:
  ├─ Check candidates.jsonl format
  ├─ Verify field presence (profile, skills, career_history)
  ├─ Check behavioral signals structure
  └─ Validate JD text non-empty

Processing Validation:
  ├─ Monitor memory usage
  ├─ Check for NaN/Inf in embeddings
  ├─ Verify score bounds [0, 1]
  └─ Ensure honeypot detection consistency

Output Validation:
  ├─ Check 100 rows exactly
  ├─ Verify monotonically decreasing scores
  ├─ Check all required columns present
  ├─ Ensure unique candidate IDs
  └─ Validate reasoning length (<200 chars)
```
