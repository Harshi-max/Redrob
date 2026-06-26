"""
Scoring Module - Multi-signal candidate scoring system.

Combines technical fit, experience, behavioral signals, and location
into a comprehensive scoring framework.
"""

import numpy as np
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class CandidateScorer:
    """Multi-signal candidate scoring system."""
    
    # Scoring weights
    WEIGHTS = {
        'semantic_similarity': 0.40,
        'skills_overlap': 0.20,
        'behavior_score': 0.15,
        'production_ml_score': 0.10,
        'career_stability_score': 0.05,
        'startup_score': 0.05,
        'location_score': 0.05,
    }
    
    def __init__(self, jd_profile: Dict[str, Any] = None):
        """Initialize scorer."""
        self.jd_profile = jd_profile or {}
    
    def score_candidate(self, candidate: Dict[str, Any], 
                       features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute comprehensive score for a candidate.
        
        Args:
            candidate: Candidate profile
            features: Pre-computed features
            
        Returns:
            Tuple of (score, evidence)
        """
        score_components = {}
        evidence = {}
        
        # Weighted scoring
        for component, weight in self.WEIGHTS.items():
            if component in features:
                component_score = features[component]
                score_components[component] = weight * component_score
                evidence[component] = {
                    'score': component_score,
                    'weight': weight,
                    'contribution': weight * component_score
                }
        
        # Apply hard filter penalties
        penalties = self._compute_penalties(candidate)
        for penalty_name, penalty_value in penalties.items():
            score_components[f'penalty_{penalty_name}'] = -penalty_value
            evidence[f'penalty_{penalty_name}'] = {'value': -penalty_value}
        
        # Final score (normalized to [0, 1])
        final_score = sum(score_components.values())
        final_score = max(0, min(1, final_score))
        
        return final_score, evidence
    
    def _compute_penalties(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Compute hard filter penalties."""
        penalties = {}
        
        # Honeypot penalty (handled separately)
        # Check for pure research background
        profile = candidate.get('profile', {})
        current_role = profile.get('current_title', '').lower()
        
        if 'researcher' in current_role and 'engineer' not in current_role:
            penalties['research_only'] = 0.2
        
        # Check for LangChain-only projects
        skills = [s['name'].lower() for s in candidate.get('skills', [])]
        if 'langchain' in skills and len(skills) < 3:
            penalties['langchain_only'] = 0.15
        
        # Check for consulting company background
        career = candidate.get('career_history', [])
        consulting_companies = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
        
        consulting_months = 0
        total_months = sum(pos.get('duration_months', 0) for pos in career)
        
        for pos in career:
            company = pos.get('company', '').lower()
            if any(cc in company for cc in consulting_companies):
                consulting_months += pos.get('duration_months', 0)
        
        if total_months > 0 and consulting_months > total_months * 0.8:
            penalties['consulting_heavy'] = 0.25
        
        return penalties
    
    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {idx: 0.5 for idx in scores}
        
        normalized = {
            idx: (score - min_val) / (max_val - min_val)
            for idx, score in scores.items()
        }
        
        return normalized
    
    def tier_normalize(self, scores: Dict[int, float], 
                      tier_boundaries: Dict[str, float] = None) -> Dict[int, Dict[str, Any]]:
        """
        Apply NDCG tier normalization.
        
        Groups candidates into quality bands (Tier 1, 2, 3, etc.)
        based on percentile boundaries.
        """
        if tier_boundaries is None:
            tier_boundaries = {
                'tier1': 0.9,
                'tier2': 0.7,
                'tier3': 0.5,
                'tier4': 0.3,
            }
        
        sorted_scores = sorted(scores.values(), reverse=True)
        
        result = {}
        for idx, score in scores.items():
            tier = self._assign_tier(score, sorted_scores, tier_boundaries)
            result[idx] = {
                'score': score,
                'tier': tier,
            }
        
        return result
    
    def _assign_tier(self, score: float, all_scores: List[float],
                    boundaries: Dict[str, float]) -> str:
        """Assign NDCG tier based on percentile."""
        percentile = len([s for s in all_scores if s >= score]) / len(all_scores)
        
        for tier_name, threshold in sorted(boundaries.items(), key=lambda x: x[1], reverse=True):
            if percentile >= threshold:
                return tier_name
        
        return 'tier_low'
