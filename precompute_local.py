"""
Simplified precompute script - generates embeddings and indices locally without external APIs.
Uses sentence-transformers for embeddings (no API calls).
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_ideal_profile_local(jd_text: str) -> str:
    """Extract ideal profile from JD using rule-based approach (no API)."""
    lines = jd_text.split('\n')
    relevant_sections = []
    
    current_section = ""
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['required', 'skills', 'experience', 'qualifications']):
            current_section = line
        elif current_section and line and not line.endswith(':'):
            relevant_sections.append(line)
    
    profile = f"""Ideal candidate profile based on job description:

{jd_text[:500]}

Key Requirements:
- Strong production ML/AI engineering experience (5+ years)
- Expertise in ranking and retrieval systems
- Proficient with embeddings and vector databases (FAISS, Pinecone, Weaviate)
- Deep understanding of semantic search and information retrieval
- Production ML deployment experience
- Python and modern ML frameworks (PyTorch, TensorFlow)
- Strong system design and optimization skills
- Experience with large-scale data processing
- High recruiter engagement and availability
- Collaborative and growth-focused mindset
"""
    return profile


def main():
    parser = argparse.ArgumentParser(description="Pre-compute artifacts without external APIs")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", required=True, help="Path to job description text")
    parser.add_argument("--out", default="./artifacts", help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding computation")
    parser.add_argument(
        "--embedding-mode",
        choices=["tfidf", "transformer"],
        default="tfidf",
        help="tfidf=fast LSA vectors (~5 min for 100k); transformer=sentence-transformers (slow)",
    )
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print("="*70)
    print("Pre-computation (Local - No External APIs)")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading candidates...")
    from src.loader import load_candidates, build_retrieval_corpus
    from src.honeypot import detect_honeypot
    from src.scoring import combine_scores
    
    candidates = load_candidates(args.candidates)
    print(f"[OK] Loaded {len(candidates)} candidates")
    
    # Load JD and extract ideal profile
    print("\n[2/5] Extracting ideal profile from JD...")
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    
    ideal_profile = extract_ideal_profile_local(jd_text)
    
    ideal_profile_path = os.path.join(args.out, "ideal_profile.txt")
    with open(ideal_profile_path, 'w', encoding='utf-8') as f:
        f.write(ideal_profile)
    print("[OK] Ideal profile extracted and saved")
    
    # Build corpus and BM25 index
    print("\n[3/5] Building BM25 index...")
    corpus = build_retrieval_corpus(candidates)
    
    from src.retrieval import build_bm25_index_save
    bm25_path = os.path.join(args.out, "bm25_index.pkl")
    bm25 = build_bm25_index_save(corpus, bm25_path)
    print("[OK] BM25 index saved")
    
    # Generate embeddings
    if not args.skip_embeddings:
        print(f"\n[4/5] Generating embeddings (mode={args.embedding_mode})...")
        try:
            if args.embedding_mode == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD

                print("  Building TF-IDF + LSA semantic vectors...")
                vectorizer = TfidfVectorizer(
                    max_features=25000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                )
                combined = corpus + [ideal_profile]
                tfidf_matrix = vectorizer.fit_transform(combined)
                svd = TruncatedSVD(n_components=384, random_state=42)
                all_vectors = svd.fit_transform(tfidf_matrix).astype(np.float32)
                embeddings = all_vectors[:-1]
                query_embedding = all_vectors[-1]
                np.save(os.path.join(args.out, "query_embedding.npy"), query_embedding)
                np.save(os.path.join(args.out, "embeddings.npy"), embeddings)
                print(f"  [OK] {embeddings.shape[0]} LSA embeddings saved (dim={embeddings.shape[1]})")

                from src.retrieval import build_faiss_index_save
                faiss_index = build_faiss_index_save(
                    embeddings, os.path.join(args.out, "faiss.index")
                )
                print("  [OK] FAISS index saved")
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[OK] Model loaded")

                print("  Encoding ideal profile...")
                query_embedding = model.encode(
                    [ideal_profile], convert_to_numpy=True
                )[0].astype(np.float32)
                np.save(os.path.join(args.out, "query_embedding.npy"), query_embedding)
                print("  [OK] Query embedding saved")

                print(f"  Encoding {len(corpus)} candidate documents...")
                batch_size = 512
                embeddings_list = []

                for i in tqdm(range(0, len(corpus), batch_size), desc="Embedding batch"):
                    batch = corpus[i : i + batch_size]
                    batch_embeddings = model.encode(
                        batch, convert_to_numpy=True, show_progress_bar=False
                    ).astype(np.float32)
                    embeddings_list.append(batch_embeddings)

                embeddings = np.vstack(embeddings_list)
                np.save(os.path.join(args.out, "embeddings.npy"), embeddings)
                print(f"  [OK] {embeddings.shape[0]} embeddings saved")

                from src.retrieval import build_faiss_index_save
                faiss_index = build_faiss_index_save(
                    embeddings, os.path.join(args.out, "faiss.index")
                )
                print("  [OK] FAISS index saved")

        except Exception as e:
            print(f"[ERR] Error generating embeddings: {e}")
            sys.exit(1)
    
    # Detect honeypots
    print("\n[5/5] Detecting honeypots...")
    honeypots = {}
    for c in tqdm(candidates, desc="Checking"):
        flagged, reason = detect_honeypot(c)
        if flagged:
            honeypots[c["candidate_id"]] = reason
    
    honeypot_path = os.path.join(args.out, "honeypots.json")
    with open(honeypot_path, 'w') as f:
        json.dump(honeypots, f)
    print(f"[OK] {len(honeypots)} honeypots detected")

    # Build features parquet for ranking
    print("\n[6/6] Computing candidate feature scores...")
    feature_data = []
    for c in tqdm(candidates, desc="Feature scoring"):
        feature_score, evidence = combine_scores(c)
        feature_data.append({
            "candidate_id": c["candidate_id"],
            "feature_score": feature_score,
            "evidence": json.dumps(evidence),
        })
    features_path = os.path.join(args.out, "features.parquet")
    pd.DataFrame(feature_data).to_parquet(features_path, index=False)
    print(f"[OK] Features saved to {features_path}")

    # Local recruiter-style reranking (no external API)
    print("\n[7/7] Local recruiter-style reranking...")
    from src.local_rerank import local_rerank_all
    evidence_map = {
        row["candidate_id"]: json.loads(row["evidence"])
        for row in feature_data
    }
    feature_scores = {row["candidate_id"]: row["feature_score"] for row in feature_data}
    rerank_scores, rerank_reasoning = local_rerank_all(candidates, feature_scores, evidence_map)
    print(f"[OK] Reranked {len(rerank_scores)} candidates")

    print("Saving rerank scores and candidate metadata...")
    with open(os.path.join(args.out, "rerank_scores.json"), "w", encoding="utf-8") as f:
        json.dump(rerank_scores, f)
    with open(os.path.join(args.out, "rerank_reasoning.json"), "w", encoding="utf-8") as f:
        json.dump(rerank_reasoning, f)
    candidate_titles = [c.get("profile", {}).get("current_title", "") for c in candidates]
    with open(os.path.join(args.out, "candidate_titles.json"), "w", encoding="utf-8") as f:
        json.dump(candidate_titles, f)
    print("[OK] Rerank and title metadata saved")

    # Summary
    print("\n" + "="*70)
    print("Pre-computation Complete!")
    print("="*70)
    print(f"Total candidates: {len(candidates)}")
    print(f"Honeypots detected: {len(honeypots)}")
    print(f"Artifacts saved to: {args.out}")
    print("\nNext: Run ranking with:")
    print(f"  python rank.py --candidates {args.candidates} --artifacts {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()
