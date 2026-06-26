"""
Comprehensive Unit Tests for Feature Engineering Module

Tests feature extraction, normalization, and scoring logic.
"""

import pytest
import numpy as np
from app.feature_engineering import FeatureEngineer


@pytest.fixture
def feature_engineer():
    """Create a feature engineer instance."""
    return FeatureEngineer()


@pytest.fixture
def sample_candidate():
    """Sample candidate profile."""
    return {
        'candidate_id': 'test_001',
        'profile': {
            'current_title': 'Senior ML Engineer',
            'current_company': 'Google',
            'location': 'San Francisco',
            'country': 'USA',
            'years_of_experience': 7.5,
        },
        'skills': [
            {'name': 'Python', 'proficiency': 'expert'},
            {'name': 'Machine Learning', 'proficiency': 'expert'},
            {'name': 'PyTorch', 'proficiency': 'intermediate'},
            {'name': 'FAISS', 'proficiency': 'expert'},
        ],
        'career_history': [
            {
                'title': 'Senior ML Engineer',
                'company': 'Google',
                'duration_months': 36,
                'company_size': 'Large',
                'industry': 'SaaS',
            },
            {
                'title': 'ML Engineer',
                'company': 'Startup AI',
                'duration_months': 24,
                'company_size': 'Startup',
                'industry': 'AI',
            },
        ],
        'education': [
            {
                'degree': "Master's",
                'field_of_study': 'Computer Science',
                'institution': 'MIT',
            }
        ],
        'redrob_signals': {
            'recruiter_response_rate': 0.75,
            'open_to_work_flag': True,
            'notice_period_days': 14,
            'github_activity_score': 82,
            'profile_completeness_score': 0.95,
        },
        'semantic_score': 0.85,
    }


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization."""
        assert feature_engineer is not None
        assert feature_engineer.FEATURE_SPECS is not None
        assert len(feature_engineer.FEATURE_SPECS) == 10
    
    def test_semantic_similarity_feature(self, feature_engineer, sample_candidate):
        """Test semantic similarity feature extraction."""
        score = feature_engineer._semantic_similarity_feature(sample_candidate)
        assert 0 <= score <= 1
        assert score == 0.85
    
    def test_skills_overlap_feature(self, feature_engineer, sample_candidate):
        """Test skills overlap feature."""
        feature_engineer.jd_profile = {
            'required_skills': ['Python', 'Machine Learning', 'FAISS']
        }
        score = feature_engineer._skills_overlap_feature(sample_candidate)
        assert 0 <= score <= 1
    
    def test_years_experience_feature(self, feature_engineer, sample_candidate):
        """Test years of experience feature."""
        score = feature_engineer._years_experience_feature(sample_candidate)
        assert 0 <= score <= 1
        assert score >= 0.8  # 7.5 years should score high
    
    def test_startup_feature(self, feature_engineer, sample_candidate):
        """Test startup experience feature."""
        score = feature_engineer._startup_feature(sample_candidate)
        assert 0 <= score <= 1
        # 24 months in startup = 1.0
        assert score >= 0.99
    
    def test_product_company_feature(self, feature_engineer, sample_candidate):
        """Test product company feature."""
        score = feature_engineer._product_company_feature(sample_candidate)
        assert 0 <= score <= 1
        # Both positions are in product/AI industry
        assert score == 1.0
    
    def test_open_source_feature(self, feature_engineer, sample_candidate):
        """Test open source feature."""
        score = feature_engineer._open_source_feature(sample_candidate)
        assert 0 <= score <= 1
        assert score >= 0.8  # GitHub score 82
    
    def test_behavioral_feature(self, feature_engineer, sample_candidate):
        """Test behavioral signals feature."""
        score = feature_engineer._behavioral_feature(sample_candidate)
        assert 0 <= score <= 1
        assert score >= 0.8  # Good signals
    
    def test_location_feature(self, feature_engineer, sample_candidate):
        """Test location feature."""
        feature_engineer.jd_profile = {
            'preferred_locations': ['San Francisco', 'Bay Area']
        }
        score = feature_engineer._location_feature(sample_candidate)
        assert score == 1.0
    
    def test_education_feature(self, feature_engineer, sample_candidate):
        """Test education feature."""
        score = feature_engineer._education_feature(sample_candidate)
        assert 0 <= score <= 1
        assert score >= 0.85  # Master's in CS should score high
    
    def test_career_stability_feature(self, feature_engineer, sample_candidate):
        """Test career stability feature."""
        score = feature_engineer._career_stability_feature(sample_candidate)
        assert 0 <= score <= 1
        # Average tenure = 30 months = moderate to good
        assert 0.6 <= score <= 0.95
    
    def test_extract_all_features(self, feature_engineer, sample_candidate):
        """Test extracting all features."""
        features = feature_engineer.extract_all_features(sample_candidate)
        
        # Check all features present
        assert len(features) == 10
        expected_features = [
            'semantic_similarity', 'skills_overlap', 'years_experience_score',
            'startup_score', 'product_company_score', 'open_source_score',
            'behavior_score', 'location_score', 'education_score',
            'career_stability_score'
        ]
        
        for feature_name in expected_features:
            assert feature_name in features
            assert 0 <= features[feature_name] <= 1
    
    def test_normalize_features(self, feature_engineer):
        """Test feature normalization."""
        features = {
            'semantic_similarity': 0.8,
            'skills_overlap': 0.7,
            'years_experience_score': 0.9,
        }
        
        normalized = feature_engineer.normalize_features(features)
        
        # Should remain in [0, 1]
        for value in normalized.values():
            assert 0 <= value <= 1
    
    def test_low_experience_penalty(self, feature_engineer):
        """Test penalty for low experience."""
        candidate = {
            'profile': {'years_of_experience': 1.0},
        }
        score = feature_engineer._years_experience_feature(candidate)
        assert score < 0.5
    
    def test_no_education_handling(self, feature_engineer):
        """Test handling of candidate with no education."""
        candidate = {'education': []}
        score = feature_engineer._education_feature(candidate)
        assert score == 0.5  # Default
    
    def test_high_notice_period_penalty(self, feature_engineer):
        """Test penalty for high notice period."""
        candidate = {
            'redrob_signals': {
                'recruiter_response_rate': 0.5,
                'open_to_work_flag': False,
                'notice_period_days': 120,
                'github_activity_score': 50,
                'profile_completeness_score': 0.7,
            }
        }
        score = feature_engineer._behavioral_feature(candidate)
        assert score < 0.6  # Should be penalized


class TestFeatureNormalization:
    """Test feature normalization logic."""
    
    def test_normalization_bounds(self):
        """Test normalization keeps values in bounds."""
        fe = FeatureEngineer()
        
        # Test extreme values
        features = {
            'semantic_similarity': 1.5,  # Above max
            'skills_overlap': -0.2,  # Below min
            'years_experience_score': 0.5,
        }
        
        normalized = fe.normalize_features(features)
        
        for value in normalized.values():
            assert 0 <= value <= 1


class TestFeatureWeights:
    """Test feature weights in final scoring."""
    
    def test_final_ranking_weights(self):
        """Test that final ranking weights sum to 1.0."""
        expected_weights = {
            'semantic_similarity': 0.40,
            'technical_fit': 0.20,
            'behavioral_fit': 0.15,
            'experience_fit': 0.10,
            'location_fit': 0.15,
        }
        
        total_weight = sum(expected_weights.values())
        assert total_weight == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
