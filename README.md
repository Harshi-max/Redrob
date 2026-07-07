# Intelligent Candidate Discovery & Ranking Solution

This repository contains the candidate ranking and discovery system . The system ranks **100,000 candidates** against a **Senior AI Engineer** job description, filters out honeypots (impossible profiles), generates **LLM-powered recruiter reasoning**, and returns the **Top 100 candidates** while satisfying strict runtime and compute constraints.

## 🚀 Overview

Recruiting at scale requires more than keyword matching. This project combines **lexical search**, **semantic retrieval**, **feature engineering**, and **LLM reasoning** to identify the most suitable candidates while eliminating fraudulent or inconsistent profiles.

The ranking pipeline is designed to operate entirely **offline** during inference, making it suitable for production environments with strict latency, memory, and networking constraints.

---

## 🏗️ System Architecture

```text
                      Job Description
                            │
                            ▼
        ┌──────────────────────────────────┐
        │ Hybrid Retrieval                 │
        │ BM25 + BGE-M3 + FAISS            │
        └──────────────────────────────────┘
                            │
                            ▼
              Reciprocal Rank Fusion (RRF)
                            │
                            ▼
          Feature Engineering & Scoring
                            │
                            ▼
            Honeypot Detection Pipeline
                            │
                            ▼
          Multi-Signal Candidate Ranking
                            │
                            ▼
           DeepSeek-v4 LLM Re-ranking
                            │
                            ▼
         Explainable Top-100 Candidates
```

---

## ✨ System Architecture & Features

* **Hybrid Retrieval (BM25 + BGE-M3 + FAISS)**  
  Combines exact lexical term matches and semantic embeddings using **Reciprocal Rank Fusion (RRF, K = 60)** together with a **Title Safety Net** to improve retrieval quality.

* **Honeypot Filter**  
  Applies strict chronological validation and capability contradiction checks, resulting in **0% honeypot infiltration** in the Top 100 candidates.

* **Multi-Signal Blended Scorer**  
  Scores candidates using weighted signals:

  - Technical Skills → **45%**
  - Experience → **25%**
  - Behavioral & Availability → **20%**
  - Location → **10%**

  while applying **Red Flag penalties** for suspicious profiles.

* **LLM Re-ranking**  
  Uses **DeepSeek-v4-flash** to rerank shortlisted candidates (40% score contribution) and generate structured, evidence-backed recruiter explanations.

* **NDCG Tier Normalizer**  
  Applies percentile-based normalization to organize candidates into ranking quality tiers.

---

## 🛠 Tech Stack

### Languages
- Python

### Retrieval
- BM25
- FAISS
- BGE-M3 Embeddings

### AI / LLM
- DeepSeek-v4-flash
- Sentence Transformers

### Data Processing
- Pandas
- NumPy

### UI
- Streamlit

### Deployment
- Docker

---

## 📌 Technical Specs & Constraints Compliance

| Constraint | Result |
|------------|--------|
| Ranking Runtime | ~10 seconds |
| Required Runtime | ≤ 5 minutes |
| Memory Usage | < 1.5 GB RAM |
| Allowed Memory | ≤ 16 GB RAM |
| CPU Requirement | CPU Only ✅ |
| Network During Ranking | None (Offline) ✅ |

The ranking pipeline performs inference entirely **offline** by consuming precomputed embeddings, similarity indices, and LLM scores without external API calls.

---

## 📂 Project Structure

```text
.
├── app.py
├── main.py
├── rank.py
├── precompute_local.py
├── requirements.txt
├── artifacts/
├── sampledata/
├── submission.csv
└── README.md
```

---

## ⚙️ Setup & Execution

### 1. Installation

Install the required packages.

```bash
pip install -r requirements.txt
```

---

### 2. Pre-computation (Offline)

Generate embeddings, BM25/FAISS indices, feature scores, and recruiter-style reranking artifacts.

```bash
python precompute_local.py \
  --candidates ./sampledata/candidates.jsonl \
  --jd ./sampledata/job_description.txt \
  --out ./artifacts
```

---

### 3. Run Ranking Pipeline

Generate the Top-100 ranked candidates.

```bash
python rank.py \
  --candidates ./sampledata/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./submission.csv
```

Or use the unified entry point.

```bash
python main.py \
  --candidates ./sampledata/candidates.jsonl \
  --artifacts ./artifacts \
  --out ./submission.csv
```

---

### 4. Launch Streamlit Sandbox

Run the interactive validation dashboard.

```bash
streamlit run app.py
```

---

## 📸 Demo

<img width="1910" height="720" alt="Screenshot 2026-06-26 134450" src="https://github.com/user-attachments/assets/83951b7c-ac59-45c7-8ad5-af64312c57c0" />

<img width="1903" height="848" alt="Screenshot 2026-06-26 134433" src="https://github.com/user-attachments/assets/2a59bc5d-8c24-46a9-a11d-0d48a474df95" />

<img width="1305" height="685" alt="Screenshot 2026-06-26 160244" src="https://github.com/user-attachments/assets/64146a04-bdba-4a8f-bde0-146dbe1a0a06" />

<img width="1085" height="748" alt="Screenshot 2026-06-26 160237" src="https://github.com/user-attachments/assets/f9236c27-6831-4748-bcd1-623e747d0bfa" />

<img width="1276" height="724" alt="Screenshot 2026-06-26 160231" src="https://github.com/user-attachments/assets/3c8697a2-b103-4273-81da-96b5829859f3" />

<img width="1394" height="742" alt="Screenshot 2026-06-26 160224" src="https://github.com/user-attachments/assets/f28cc0a5-af2a-4580-a092-c9605c1ca93d" />

<img width="1276" height="749" alt="Screenshot 2026-06-26 160211" src="https://github.com/user-attachments/assets/0d95a876-93dd-4872-a22a-bd2de2d0a28f" />

<img width="1877" height="668" alt="Screenshot 2026-06-26 160202" src="https://github.com/user-attachments/assets/4d628203-5e14-4346-8397-77c65757f553" />

---

## 🎯 Key Highlights

- Ranked **100,000 candidates** efficiently.
- Hybrid retrieval using **BM25 + FAISS + BGE-M3**.
- **LLM-powered explainable ranking** with structured recruiter reasoning.
- **0% honeypot infiltration** in the final Top-100.
- Fully **offline ranking pipeline**.
- CPU-only execution.
- Docker-ready and deployable.
- Interactive Streamlit validation dashboard.

---

## 🚀 Future Improvements

- Multi-job ranking support
- Resume parsing from PDF/DOCX
- Recruiter feedback loop for continuous learning
- Graph-based candidate similarity search
- Fine-tuned reranking models
- REST API deployment with FastAPI
