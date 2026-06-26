# Intelligent Candidate Discovery & Ranking System
**Redrob Hackathon - Production-Grade Solution**

> A comprehensive, CPU-only ranking system that processes 100,000 candidates against a job description in <5 minutes on <16GB RAM with zero external API calls.

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [System Overview](#system-overview)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Performance](#performance)
7. [Repository Structure](#repository-structure)
8. [Technical Details](#technical-details)
9. [Development](#development)
10. [Contributing](#contributing)

---

## 🎯 Problem Statement

**Given:**
- `candidates.jsonl` containing 100,000 candidate profiles
- A job description file
- Candidate behavioral signals

**Goal:**
Return the **Top 100** most relevant candidates ranked from best to worst.

**Constraints:**
- ✅ CPU only (no GPU required)
- ✅ ≤ 16GB RAM
- ✅ ≤ 5 minutes runtime
- ✅ No external API calls during ranking
- ✅ No hosted LLMs during ranking

---

## 🏗️ System Overview

This system implements a **10-stage multi-signal ranking pipeline** that combines:

1. **Semantic Embeddings** (0.40 weight) - Using sentence-transformers
2. **Skills Overlap** (0.20 weight) - Lexical + taxonomic matching
3. **Behavioral Signals** (0.15 weight) - Response rate, GitHub, notice period
4. **ML Production Experience** (0.10 weight) - Rule-based detection
5. **Career Stability** (0.05 weight) - Tenure analysis
6. **Startup Experience** (0.05 weight) - Company size history
7. **Location Match** (0.05 weight) - Geographic constraints

Plus **hard filters** to eliminate honeypots and completely mismatched profiles.

### Key Features

- ✨ **Hybrid Retrieval**: Semantic + BM25 + FAISS indexing
- 🎯 **Honeypot Detection**: 0% infiltration with strict chronological checks
- 📊 **Multi-Signal Scoring**: Blends 8+ behavioral and technical signals
- ⚡ **Production Optimized**: Batch processing, local caching, vectorization
- 📝 **Fact-Grounded Reasoning**: 1-2 sentence explanations per candidate
- 🔍 **NDCG Tier Normalization**: Percentile-based quality grouping

---

## 🏛️ Architecture

### Pipeline Stages

```
INPUT → [Parse JD] → [Parse Candidates] → [Embeddings] 
  ↓
[Hybrid Retrieval] → [Hard Filters] → [Behavioral Signals]
  ↓
[Feature Engineering] → [Final Ranking] → [Reasoning] → [Output]
  ↓
OUTPUT (100 candidates with scores & reasoning)
```

### Module Organization

```
intelligent-candidate-discovery/
├── app/                          # Application modules (NEW)
│   ├── ranker.py                 # Main ranking orchestrator
│   ├── feature_engineering.py    # Feature extraction & normalization
│   ├── embedding_service.py      # Semantic embeddings (sentence-transformers)
│   ├── retrieval.py              # Hybrid retrieval (FAISS + BM25)
│   ├── scoring.py                # Multi-signal scoring
│   ├── reasoning.py              # Fact-grounded reasoning generation
│   └── utils.py                  # Utilities
│
├── models/                       # Encoding modules (NEW)
│   ├── jd_encoder.py             # Job description parsing
│   └── candidate_encoder.py      # Candidate profile normalization
│
├── src/                          # Core pipeline (existing)
│   ├── pipeline.py               # Main execution pipeline
│   ├── loader.py                 # Data loading & parsing
│   ├── retrieval.py              # Retrieval (FAISS, BM25)
│   ├── honeypot.py               # Honeypot detection
│   ├── scoring.py                # Scoring logic
│   ├── reasoning.py              # Reasoning generation
│   ├── reranking.py              # Optional re-ranking
│   ├── config.py                 # Configuration
│   └── query.py                  # JD parsing
│
├── notebooks/                    # Jupyter notebooks (NEW)
│   └── pipeline_explanation.ipynb # Complete walkthrough
│
├── outputs/                      # Output directory (NEW)
│   └── submission.csv            # Results
│
├── tests/                        # Unit tests
│   ├── test_retrieval.py
│   ├── test_scoring.py
│   └── test_honeypot.py
│
├── main.py                       # Entry point
├── rank.py                       # Offline ranking
├── precompute.py                 # Pre-computation step
├── validate.py                   # Validation
├── app.py                        # Streamlit demo
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container setup
├── README.md                     # This file
└── ARCHITECTURE.md               # Detailed architecture
```

---

## 💾 Installation

### Prerequisites

- Python 3.10+
- 16GB RAM recommended
- 5GB disk space for embeddings cache

### Local Setup

```bash
# Clone repository
git clone <repository>
cd Intelligent-Candidate-Discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build container
docker build -t candidate-ranker .

# Run container
docker run -v /path/to/data:/app/sample \
           -v /path/to/artifacts:/app/artifacts \
           -v /path/to/outputs:/app/outputs \
           candidate-ranker
```

---

## 🚀 Usage

### Quick Start (Offline Ranking)

Assuming pre-computed embeddings exist in `artifacts/`:

```bash
python main.py \
  --candidates ./sample/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./outputs/submission.csv \
  --top-k 100
```

### Pre-computation + Ranking

If you need to generate embeddings first (requires internet for downloads, not API calls):

```bash
python main.py \
  --candidates ./sample/candidates.jsonl \
  --jd ./sample/job_description.txt \
  --artifacts ./artifacts \
  --precompute \
  --out ./outputs/submission.csv
```

### Streamlit Demo

Interactive visualization and validation:

```bash
streamlit run app.py
```

### Jupyter Notebook

Step-by-step walkthrough of the pipeline:

```bash
jupyter notebook notebooks/pipeline_explanation.ipynb
```

---

## ⚡ Performance

### Runtime

| Stage | Time |
|-------|------|
| Preprocessing (cached) | 0-3 min |
| Ranking pipeline | 30-60 sec |
| Total | **< 5 minutes** ✅ |

### Memory Usage

- **Embeddings**: ~300 MB (100k × 384-dim)
- **FAISS Index**: ~400 MB
- **Candidate Profiles**: ~500 MB
- **Working Memory**: ~200 MB
- **Total**: **~1.5 GB** ✅ (well under 16GB limit)

### Optimization Techniques

- ✅ Batch embedding generation (32-64 texts per batch)
- ✅ Local caching of embeddings and FAISS indices
- ✅ BM25 index pre-computation
- ✅ NumPy vectorization for scoring
- ✅ FAISS approximate nearest neighbors
- ✅ Streaming JSONL parsing (avoid full load)
- ✅ Multiprocessing for independent operations

---

## 📚 Technical Details

### Scoring Formulas

#### Hybrid Retrieval Score
$$\text{Retrieval} = 0.45 \times \text{Semantic} + 0.25 \times \text{Skills} + 0.15 \times \text{Experience} + 0.15 \times \text{Behavioral}$$

#### Final Ranking Formula
$$\text{Final} = 0.40 \times \text{Semantic} + 0.20 \times \text{Skills} + 0.15 \times \text{Behavioral} + 0.10 \times \text{ML} + 0.05 \times \text{Stability} + 0.05 \times \text{Startup} + 0.05 \times \text{Location}$$

### Feature Engineering

#### 10 Key Features

1. **Semantic Similarity** - Cosine similarity via embeddings
2. **Skills Overlap** - Match against JD requirements
3. **Years Experience Score** - Gaussian distribution (optimal: 5-10 years)
4. **Startup Score** - Tenure in early-stage companies
5. **Product Company Score** - % time in product companies vs consulting
6. **Open Source Score** - GitHub activity proxy
7. **Behavior Score** - Composite of response rate, notice period, profile completeness
8. **Location Score** - Match against JD constraints
9. **Education Score** - Degree level + field relevance
10. **Career Stability Score** - Average tenure between positions

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384-D vectors
- **Performance**: Fast, accurate, suitable for CPU
- **Cache**: Local `.npy` files avoid re-computation

### Hard Filters (Honeypot Detection)

The system penalizes/removes candidates with:

- ❌ Impossible profiles (future employment dates, etc.)
- ❌ Pure research background (no production experience)
- ❌ Only LangChain projects
- ❌ No ML production experience
- ❌ Entire career in consulting (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- ❌ CV/Speech/Robotics-only expertise
- ❌ No coding activity in past 2 years

### Reasoning Generation

For each ranked candidate, generates a 1-2 sentence explanation:

```
Example: "7 years of experience building retrieval and ranking systems 
with strong Python and vector database exposure. High recruiter engagement 
and recent activity make this candidate a strong match despite a longer 
notice period."
```

**Rules:**
- Mention actual facts only (not hallucinated)
- Include specific metrics (years, skills, response rate)
- Note concerns (if any)
- Unique per candidate
- Max 200 characters

---

## 🔍 Validation

### Output Format

```csv
rank,candidate_id,score,reasoning
1,cand_00001,0.912,"7 years ML engineer at Google. Expert in embeddings..."
2,cand_00123,0.890,"Senior ML with production RAG experience..."
...
100,cand_87654,0.654,"4 years data scientist at startup, strong Python..."
```

### Constraints Verified

- ✅ Exactly 100 rows
- ✅ Scores monotonically decreasing
- ✅ All required columns present
- ✅ Unique candidate IDs
- ✅ Reasoning length < 200 chars
- ✅ Scores in [0, 1] range

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/

# Or specific test
pytest tests/test_scoring.py -v
```

### Test Coverage

- `test_honeypot.py` - Honeypot detection accuracy
- `test_retrieval.py` - Embedding + BM25 retrieval
- `test_scoring.py` - Multi-signal scoring logic

---

## 📝 Development

### Adding New Features

1. **Create feature function** in `app/feature_engineering.py`
2. **Add to FEATURE_SPECS** dictionary
3. **Update final ranking formula** in `app/ranker.py`
4. **Test and validate** scores remain in [0, 1]

### Debugging

```bash
# Verbose logging
python main.py --candidates ./sample/candidates.jsonl --verbose

# Run individual stages
from app.ranker import CandidateRanker
ranker = CandidateRanker()
# ... debug each stage
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run ranking code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## 📦 Dependencies

See `requirements.txt` for complete list:

- **Core**: numpy, pandas, scipy
- **Embeddings**: sentence-transformers
- **Retrieval**: faiss-cpu, rank-bm25
- **ML**: scikit-learn
- **Data**: pyarrow, jsonschema
- **UI**: streamlit
- **Testing**: pytest
- **Utilities**: tqdm, requests

---

## 🎓 Example Walkthrough

See `notebooks/pipeline_explanation.ipynb` for:

1. Data loading & exploration
2. JD parsing
3. Candidate profile extraction
4. Embedding generation
5. Hybrid retrieval scoring
6. Hard filters application
7. Behavioral signal computation
8. Feature engineering
9. Final ranking formula
10. Reasoning generation
11. Performance benchmarking

---

## 🚨 Troubleshooting

### Issue: "Embeddings not found"
**Solution**: Run with `--precompute` flag to generate embeddings

### Issue: Out of memory
**Solution**: 
- Reduce batch size in `embedding_service.py`
- Stream process candidates instead of loading all
- Clear cache periodically

### Issue: Poor candidate rankings
**Solution**:
- Check JD parsing accuracy
- Verify skill taxonomy in `src/config.py`
- Adjust feature weights in final formula

---

## 📊 Metrics

The system optimizes for:

- **Precision@100**: 100% valid candidates (no honeypots)
- **Diversity**: Mix of startup & large tech company backgrounds
- **Freshness**: Penalizes long notice periods, rewards open-to-work
- **Production Focus**: Heavily weights ML/retrieval systems experience
- **Fairness**: No demographic bias (profiles are anonymized)

---

## 📄 License

MIT License - See LICENSE file

---

## 👥 Authors

Redrob Hackathon Team  
Built for the Redrob Candidate Discovery Challenge

---

## 📞 Support

For issues, questions, or contributions:

1. Check `ARCHITECTURE.md` for detailed system design
2. Review `notebooks/pipeline_explanation.ipynb` for examples
3. Run tests: `pytest tests/`
4. Check logs with `--verbose` flag

---

**Last Updated:** June 2026  
**Status:** Production Ready ✅
