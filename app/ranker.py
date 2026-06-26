"""
Main Ranker Module - Orchestrates the multi-stage ranking pipeline.

This module coordinates all stages of the ranking pipeline:
1. Candidate Parsing
2. Job Description Understanding
3. Semantic Embeddings
4. Hybrid Retrieval
5. Hard Filters
6. Behavioral Signal Scoring
7. Feature Engineering
8. Final Ranking
9. Reasoning Generation
10. Output Generation
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import time

logger = logging.getLogger(__name__)


class CandidateRanker:
    """
    Multi-stage ranking pipeline for intelligent candidate discovery.
    
    Attributes:
        jd_profile: Parsed job description profile
        candidates: List of candidate profiles
        embedding_service: Service for semantic embeddings
        retrieval: Hybrid retrieval component
        feature_engineer: Feature engineering component
        scores_cache: Cache for computed scores
    """
    
    def __init__(self, embedding_service=None, retrieval=None, config=None):
        """Initialize the ranker with required components."""
        self.embedding_service = embedding_service
        self.retrieval = retrieval
        self.config = config or {}
        self.jd_profile = None
        self.candidates = None
        self.scores_cache = {}
        self.honeypot_flags = {}
        self.rankings = []
        
        # Timing metrics
        self.timing = {}
        
    def rank(self, candidates: List[Dict], jd_text: str, top_k: int = 100) -> pd.DataFrame:
        """
        Execute the complete ranking pipeline.
        
        Args:
            candidates: List of candidate profiles
            jd_text: Job description text
            top_k: Number of top candidates to return
            
        Returns:
            DataFrame with columns: [rank, candidate_id, score, reasoning]
        """
        logger.info(f"Starting ranking pipeline for {len(candidates)} candidates")
        start_time = time.time()
        
        # Stage 1: Parse JD
        logger.info("Stage 1: Parsing job description...")
        stage_start = time.time()
        self.jd_profile = self._parse_job_description(jd_text)
        self.timing['parse_jd'] = time.time() - stage_start
        
        # Store candidates
        self.candidates = candidates
        
        # Stage 2: Parse candidates (already done in loader, but can enhance here)
        logger.info("Stage 2: Preparing candidate documents...")
        stage_start = time.time()
        candidate_documents = self._prepare_candidate_documents(candidates)
        self.timing['parse_candidates'] = time.time() - stage_start
        
        # Stage 3 & 4: Embeddings + Hybrid Retrieval
        logger.info("Stage 3-4: Computing embeddings and retrieval scores...")
        stage_start = time.time()
        retrieval_scores = self._hybrid_retrieval(candidate_documents)
        self.timing['retrieval'] = time.time() - stage_start
        
        # Stage 5: Apply hard filters
        logger.info("Stage 5: Applying hard filters...")
        stage_start = time.time()
        filtered_scores = self._apply_hard_filters(retrieval_scores)
        self.timing['hard_filters'] = time.time() - stage_start
        
        # Stage 6 & 7: Behavioral signals + Feature engineering
        logger.info("Stages 6-7: Computing behavioral signals and features...")
        stage_start = time.time()
        features = self._compute_features(filtered_scores)
        self.timing['features'] = time.time() - stage_start
        
        # Stage 8: Final ranking
        logger.info("Stage 8: Computing final scores...")
        stage_start = time.time()
        final_scores = self._final_ranking(features)
        self.timing['final_ranking'] = time.time() - stage_start
        
        # Stage 9: Reasoning generation
        logger.info("Stage 9: Generating reasoning...")
        stage_start = time.time()
        reasoning = self._generate_reasoning(final_scores)
        self.timing['reasoning'] = time.time() - stage_start
        
        # Stage 10: Format output
        logger.info("Stage 10: Formatting output...")
        stage_start = time.time()
        output_df = self._format_output(final_scores, reasoning, top_k)
        self.timing['output'] = time.time() - stage_start
        
        total_time = time.time() - start_time
        logger.info(f"Ranking completed in {total_time:.2f}s")
        logger.info(f"Timing breakdown: {self.timing}")
        
        return output_df
    
    def _parse_job_description(self, jd_text: str) -> Dict[str, Any]:
        """Stage 1: Parse JD into components."""
        from src.query import extract_ideal_profile
        
        profile = {
            'raw': jd_text,
            'ideal_profile': extract_ideal_profile(jd_text),
        }
        return profile
    
    def _prepare_candidate_documents(self, candidates: List[Dict]) -> List[str]:
        """Stage 2: Prepare normalized candidate documents."""
        from src.loader import build_retrieval_corpus
        return build_retrieval_corpus(candidates)
    
    def _hybrid_retrieval(self, documents: List[str]) -> Dict[int, float]:
        """Stage 3-4: Compute hybrid retrieval scores (semantic + BM25)."""
        from src.retrieval import hybrid_retrieve
        
        ideal_profile = self.jd_profile['ideal_profile']
        scores = hybrid_retrieve(
            query=ideal_profile,
            corpus=documents,
            bm25=None,  # Will be built if needed
            embeddings=None,  # Will be computed if needed
        )
        return scores
    
    def _apply_hard_filters(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Stage 5: Apply hard disqualifying filters."""
        from src.honeypot import detect_honeypot
        
        filtered = {}
        for candidate_idx, score in scores.items():
            candidate = self.candidates[candidate_idx]
            is_honeypot, reason = detect_honeypot(candidate)
            
            if not is_honeypot:
                filtered[candidate_idx] = score
                self.honeypot_flags[candidate_idx] = False
            else:
                self.honeypot_flags[candidate_idx] = True
                logger.debug(f"Filtered honeypot {candidate_idx}: {reason}")
        
        return filtered
    
    def _compute_features(self, scores: Dict[int, float]) -> Dict[int, Dict[str, float]]:
        """Stage 6-7: Compute behavioral signals and features."""
        from src.scoring import (
            score_technical_fit, score_experience_fit,
            score_behavioral_fit, score_location_fit
        )
        
        features = {}
        for candidate_idx in scores:
            candidate = self.candidates[candidate_idx]
            
            tech_score, tech_evidence = score_technical_fit(candidate)
            exp_score, exp_evidence = score_experience_fit(candidate)
            behav_score, behav_evidence = score_behavioral_fit(candidate)
            loc_score, loc_evidence = score_location_fit(candidate)
            
            features[candidate_idx] = {
                'semantic_similarity': scores.get(candidate_idx, 0) * 0.5 + 0.5,  # Normalize
                'technical_fit': tech_score,
                'experience_fit': exp_score,
                'behavioral_fit': behav_score,
                'location_fit': loc_score,
                'evidence': {
                    'tech': tech_evidence,
                    'exp': exp_evidence,
                    'behav': behav_evidence,
                    'loc': loc_evidence,
                }
            }
        
        return features
    
    def _final_ranking(self, features: Dict[int, Dict]) -> Dict[int, float]:
        """Stage 8: Compute final ranking scores."""
        final_scores = {}
        
        # Final ranking formula
        weights = {
            'semantic_similarity': 0.40,
            'technical_fit': 0.20,
            'behavioral_fit': 0.15,
            'experience_fit': 0.10,
            'location_fit': 0.15,
        }
        
        for candidate_idx, feature_dict in features.items():
            score = sum(
                weights[key] * feature_dict[key]
                for key in weights
                if key in feature_dict
            )
            final_scores[candidate_idx] = max(0, min(1, score))  # Normalize to [0, 1]
        
        return final_scores
    
    def _generate_reasoning(self, scores: Dict[int, float]) -> Dict[int, str]:
        """Stage 9: Generate reasoning for each candidate."""
        from src.reasoning import generate_reasoning
        
        reasoning = {}
        for candidate_idx in scores:
            candidate = self.candidates[candidate_idx]
            # Get evidence from features cache if available
            evidence = self.scores_cache.get(candidate_idx, {})
            reasoning[candidate_idx] = generate_reasoning(candidate, evidence)
        
        return reasoning
    
    def _format_output(self, scores: Dict[int, float], reasoning: Dict[int, str], 
                      top_k: int = 100) -> pd.DataFrame:
        """Stage 10: Format output as CSV."""
        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        output = []
        for rank, (candidate_idx, score) in enumerate(sorted_indices, 1):
            candidate = self.candidates[candidate_idx]
            output.append({
                'rank': rank,
                'candidate_id': candidate.get('candidate_id', f'UNK_{candidate_idx}'),
                'score': score,
                'reasoning': reasoning.get(candidate_idx, ''),
            })
        
        df = pd.DataFrame(output)
        
        # Validate monotonic decreasing
        assert df['score'].is_monotonic_decreasing, "Scores must be monotonically decreasing"
        
        return df
