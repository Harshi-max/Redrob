"""
Feature Engineering Module - Comprehensive feature extraction and normalization.

Creates semantic, skills, experience, behavioral, and stability features
for the ranking pipeline.
"""

import numpy as np
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extracts and normalizes features for ranking."""
    
    FEATURE_SPECS = {
        'semantic_similarity': {'type': 'float', 'min': 0, 'max': 1},
        'skills_overlap': {'type': 'float', 'min': 0, 'max': 1},
        'years_experience_score': {'type': 'float', 'min': 0, 'max': 1},
        'startup_score': {'type': 'float', 'min': 0, 'max': 1},
        'product_company_score': {'type': 'float', 'min': 0, 'max': 1},
        'open_source_score': {'type': 'float', 'min': 0, 'max': 1},
        'behavior_score': {'type': 'float', 'min': 0, 'max': 1},
        'location_score': {'type': 'float', 'min': 0, 'max': 1},
        'education_score': {'type': 'float', 'min': 0, 'max': 1},
        'career_stability_score': {'type': 'float', 'min': 0, 'max': 1},
    }
    
    def __init__(self, jd_profile: Dict[str, Any] = None):
        """Initialize feature engineer."""
        self.jd_profile = jd_profile or {}
        
    def extract_all_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract all features for a candidate."""
        features = {}
        
        features['semantic_similarity'] = self._semantic_similarity_feature(candidate)
        features['skills_overlap'] = self._skills_overlap_feature(candidate)
        features['years_experience_score'] = self._years_experience_feature(candidate)
        features['startup_score'] = self._startup_feature(candidate)
        features['product_company_score'] = self._product_company_feature(candidate)
        features['open_source_score'] = self._open_source_feature(candidate)
        features['behavior_score'] = self._behavioral_feature(candidate)
        features['location_score'] = self._location_feature(candidate)
        features['education_score'] = self._education_feature(candidate)
        features['career_stability_score'] = self._career_stability_feature(candidate)
        
        return features
    
    def _semantic_similarity_feature(self, candidate: Dict) -> float:
        """Score semantic similarity between candidate and JD."""
        # This would be populated from embedding similarity
        return candidate.get('semantic_score', 0.5)
    
    def _skills_overlap_feature(self, candidate: Dict) -> float:
        """Score overlap between candidate skills and JD requirements."""
        candidate_skills = set(s['name'].lower() for s in candidate.get('skills', []))
        jd_skills = set(self.jd_profile.get('required_skills', []))
        
        if not jd_skills:
            return 0.5
        
        overlap = len(candidate_skills & jd_skills)
        total = len(jd_skills)
        
        return min(1.0, overlap / total)
    
    def _years_experience_feature(self, candidate: Dict) -> float:
        """Score years of experience (target: 5-10 years)."""
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        
        if years < 2:
            return 0.2
        elif years < 5:
            return 0.6 + (years - 2) * 0.067
        elif years < 10:
            return 0.8 + (years - 5) * 0.04
        elif years < 15:
            return 1.0
        else:
            return max(0.8, 1.0 - (years - 15) * 0.02)
    
    def _startup_feature(self, candidate: Dict) -> float:
        """Score startup experience."""
        career = candidate.get('career_history', [])
        startup_months = 0
        
        for position in career:
            company_size = position.get('company_size', 'unknown').lower()
            if company_size in ['startup', 'small', '1-50']:
                startup_months += position.get('duration_months', 0)
        
        return min(1.0, startup_months / 24)  # 2 years = full score
    
    def _product_company_feature(self, candidate: Dict) -> float:
        """Score experience at product companies."""
        career = candidate.get('career_history', [])
        product_months = 0
        total_months = sum(p.get('duration_months', 0) for p in career)
        
        product_industries = {'saas', 'software', 'ai', 'tech', 'internet'}
        
        for position in career:
            industry = position.get('industry', '').lower()
            if any(prod in industry for prod in product_industries):
                product_months += position.get('duration_months', 0)
        
        if total_months == 0:
            return 0.5
        
        return min(1.0, product_months / total_months)
    
    def _open_source_feature(self, candidate: Dict) -> float:
        """Score open source contributions."""
        signals = candidate.get('redrob_signals', {})
        github_score = signals.get('github_activity_score', 0)
        
        return min(1.0, github_score / 100)
    
    def _behavioral_feature(self, candidate: Dict) -> float:
        """Score behavioral signals (availability, response rate, etc)."""
        signals = candidate.get('redrob_signals', {})
        
        components = []
        
        # Response rate (0-1)
        resp_rate = signals.get('recruiter_response_rate', 0.3)
        components.append(resp_rate)
        
        # Open to work (bool -> 0-1)
        open_to_work = signals.get('open_to_work_flag', False)
        components.append(1.0 if open_to_work else 0.3)
        
        # Notice period (fewer days is better)
        notice_days = signals.get('notice_period_days', 30)
        notice_score = max(0.3, 1.0 - (notice_days / 90))  # 90 days is 0.3
        components.append(notice_score)
        
        # Profile completeness
        completeness = signals.get('profile_completeness_score', 0.5)
        components.append(completeness)
        
        return np.mean(components) if components else 0.5
    
    def _location_feature(self, candidate: Dict) -> float:
        """Score location fit."""
        profile = candidate.get('profile', {})
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        
        # Adjust based on JD location requirements
        preferred_locations = self.jd_profile.get('preferred_locations', [])
        
        if not preferred_locations:
            return 0.8  # Default if no constraint
        
        for pref in preferred_locations:
            if pref.lower() in location.lower() or pref.lower() in country.lower():
                return 1.0
        
        return 0.5  # Different location but not disqualified
    
    def _education_feature(self, candidate: Dict) -> float:
        """Score educational background."""
        education = candidate.get('education', [])
        
        if not education:
            return 0.5
        
        score = 0.6  # Base score for any education
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            field = edu.get('field_of_study', '').lower()
            
            # Boost for advanced degrees
            if 'phd' in degree or 'doctorate' in degree:
                score = max(score, 0.95)
            elif "master's" in degree or 'masters' in degree:
                score = max(score, 0.85)
            
            # Boost for relevant fields
            relevant_fields = {'computer science', 'engineering', 'mathematics', 'statistics', 'physics'}
            if any(rf in field for rf in relevant_fields):
                score = max(score, 0.85)
        
        return score
    
    def _career_stability_feature(self, candidate: Dict) -> float:
        """Score career stability (longevity at companies)."""
        career = candidate.get('career_history', [])
        
        if not career:
            return 0.5
        
        avg_tenure = np.mean([p.get('duration_months', 12) for p in career])
        
        if avg_tenure < 12:
            return 0.4
        elif avg_tenure < 24:
            return 0.6
        elif avg_tenure < 48:
            return 0.8
        else:
            return 0.95
    
    def normalize_features(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize all features to [0, 1] range."""
        normalized = {}
        for feature_name, value in features_dict.items():
            spec = self.FEATURE_SPECS.get(feature_name)
            if spec:
                min_val = spec['min']
                max_val = spec['max']
                normalized[feature_name] = (value - min_val) / (max_val - min_val)
                normalized[feature_name] = max(min_val, min(max_val, normalized[feature_name]))
            else:
                normalized[feature_name] = value
        
        return normalized
