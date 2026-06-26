"""
JD Encoder - Job Description Encoder Module

Parses and encodes job descriptions into structured profiles
for semantic matching with candidates.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class JDEncoder:
    """Encodes job descriptions into structured profiles."""
    
    def __init__(self):
        """Initialize JD encoder."""
        pass
    
    def encode(self, jd_text: str) -> Dict[str, Any]:
        """
        Encode JD into structured components.
        
        Extracts:
        - Required skills
        - Preferred skills
        - Experience range
        - Disqualifiers
        - Location constraints
        - Product vs research preference
        - Startup/scale-up experience
        - Domain expertise
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            Structured JD profile
        """
        profile = {
            'raw': jd_text,
            'required_skills': self._extract_required_skills(jd_text),
            'preferred_skills': self._extract_preferred_skills(jd_text),
            'disqualifiers': self._extract_disqualifiers(jd_text),
            'experience_range': self._extract_experience_range(jd_text),
            'location_constraints': self._extract_locations(jd_text),
            'product_preference': self._assess_product_preference(jd_text),
            'startup_preference': self._assess_startup_preference(jd_text),
            'domain_expertise': self._extract_domain_expertise(jd_text),
        }
        
        return profile
    
    def _extract_required_skills(self, jd_text: str) -> List[str]:
        """Extract required technical skills."""
        required_keywords = [
            'python', 'javascript', 'typescript', 'java', 'c++',
            'machine learning', 'deep learning', 'neural networks',
            'pytorch', 'tensorflow', 'transformers',
            'retrieval', 'ranking', 'search', 'embeddings',
            'database', 'sql', 'nosql',
            'docker', 'kubernetes', 'cloud', 'aws', 'gcp', 'azure'
        ]
        
        text_lower = jd_text.lower()
        found_skills = [skill for skill in required_keywords if skill in text_lower]
        
        return list(set(found_skills))
    
    def _extract_preferred_skills(self, jd_text: str) -> List[str]:
        """Extract preferred (nice-to-have) skills."""
        # Search for "nice to have", "preferred", "plus" sections
        preferred_keywords = [
            'llm', 'langchain', 'rag', 'vector database', 'faiss',
            'nlp', 'recommendation systems', 'information retrieval',
            'scala', 'rust', 'go', 'rust', 'golang',
            'apache spark', 'hadoop', 'kafka'
        ]
        
        text_lower = jd_text.lower()
        found_skills = [skill for skill in preferred_keywords if skill in text_lower]
        
        return list(set(found_skills))
    
    def _extract_disqualifiers(self, jd_text: str) -> List[str]:
        """Extract explicit disqualifiers."""
        disqualifiers = []
        text_lower = jd_text.lower()
        
        if 'no visa sponsorship' in text_lower:
            disqualifiers.append('visa_required')
        
        if 'must have work authorization' in text_lower:
            disqualifiers.append('no_visa_sponsorship')
        
        return disqualifiers
    
    def _extract_experience_range(self, jd_text: str) -> Dict[str, int]:
        """Extract required years of experience."""
        import re
        
        # Look for patterns like "5+ years", "3-7 years"
        pattern = r'(\d+)(?:\+|-(\d+))?\s*years?\s*of\s*experience'
        matches = re.findall(pattern, jd_text.lower())
        
        if not matches:
            return {'min': 0, 'max': 10}
        
        min_exp = int(matches[0][0])
        max_exp = int(matches[0][1]) if matches[0][1] else min_exp + 5
        
        return {'min': min_exp, 'max': max_exp}
    
    def _extract_locations(self, jd_text: str) -> List[str]:
        """Extract location constraints."""
        locations = []
        
        location_keywords = [
            'sf', 'san francisco', 'bay area', 'silicon valley',
            'new york', 'nyc', 'london', 'bengaluru', 'bangalore',
            'delhi', 'hyderabad', 'mumbai', 'remote', 'distributed'
        ]
        
        text_lower = jd_text.lower()
        for loc in location_keywords:
            if loc in text_lower:
                locations.append(loc)
        
        return locations
    
    def _assess_product_preference(self, jd_text: str) -> float:
        """Assess preference for product (vs research) experience (0-1)."""
        text_lower = jd_text.lower()
        
        product_indicators = [
            'product', 'shipped', 'deployment', 'production',
            'users', 'scale', 'infrastructure', 'systems'
        ]
        
        product_count = sum(1 for ind in product_indicators if ind in text_lower)
        
        return min(1.0, product_count / len(product_indicators))
    
    def _assess_startup_preference(self, jd_text: str) -> float:
        """Assess preference for startup experience (0-1)."""
        text_lower = jd_text.lower()
        
        startup_indicators = [
            'startup', 'scale-up', 'early stage', 'fast paced',
            'wear many hats', 'scrappy', 'entrepreneurial'
        ]
        
        startup_count = sum(1 for ind in startup_indicators if ind in text_lower)
        
        return min(1.0, startup_count / len(startup_indicators))
    
    def _extract_domain_expertise(self, jd_text: str) -> List[str]:
        """Extract required domain expertise."""
        domains = []
        
        domain_keywords = {
            'ai': ['ai', 'artificial intelligence', 'ml', 'machine learning'],
            'nlp': ['nlp', 'language', 'text'],
            'cv': ['vision', 'image', 'video'],
            'ranking': ['ranking', 'retrieval', 'information retrieval'],
            'recommendation': ['recommendation', 'recommender'],
            'search': ['search', 'elasticsearch'],
        }
        
        text_lower = jd_text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)
        
        return domains
