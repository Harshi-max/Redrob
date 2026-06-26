"""
Candidate Encoder - Candidate Profile Encoder Module

Parses and normalizes candidate profiles for ranking.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CandidateEncoder:
    """Encodes candidate profiles into structured representations."""
    
    def __init__(self):
        """Initialize candidate encoder."""
        pass
    
    def encode(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode and normalize candidate profile.
        
        Structures:
        - Skills and expertise
        - Experience and career trajectory
        - Companies and industries
        - Education credentials
        - Certifications
        - Behavioral signals
        
        Args:
            candidate: Raw candidate profile
            
        Returns:
            Structured candidate profile
        """
        profile = candidate  # Already structured from loader
        
        # Add computed fields
        profile['_processed'] = {
            'skills_set': self._normalize_skills(candidate.get('skills', [])),
            'experience_summary': self._summarize_experience(candidate),
            'career_trajectory': self._analyze_trajectory(candidate),
            'education_level': self._assess_education(candidate),
            'behavioral_score': self._compute_behavioral_score(candidate),
        }
        
        return profile
    
    def _normalize_skills(self, skills: List[Dict]) -> set:
        """Normalize and deduplicate skills."""
        skill_set = set()
        for skill in skills:
            name = skill.get('name', '').lower().strip()
            if name:
                skill_set.add(name)
        return skill_set
    
    def _summarize_experience(self, candidate: Dict) -> str:
        """Create experience summary."""
        profile = candidate.get('profile', {})
        years = profile.get('years_of_experience', 0)
        current_title = profile.get('current_title', '')
        current_company = profile.get('current_company', '')
        
        summary = f"{years:.1f} years as {current_title}"
        if current_company:
            summary += f" at {current_company}"
        
        return summary
    
    def _analyze_trajectory(self, candidate: Dict) -> Dict[str, Any]:
        """Analyze career trajectory."""
        career = candidate.get('career_history', [])
        
        if not career:
            return {'type': 'unknown', 'positions': 0}
        
        positions = len(career)
        avg_tenure = sum(p.get('duration_months', 12) for p in career) / positions
        
        # Trajectory type
        trajectory_type = 'stable'
        if avg_tenure < 12:
            trajectory_type = 'job_hopper'
        elif avg_tenure > 48:
            trajectory_type = 'long_tenure'
        
        # Career growth
        titles = [p.get('title', '').lower() for p in career]
        has_growth = len(set(titles)) > 1  # Different titles indicate growth
        
        return {
            'type': trajectory_type,
            'positions': positions,
            'avg_tenure_months': avg_tenure,
            'career_growth': has_growth,
        }
    
    def _assess_education(self, candidate: Dict) -> str:
        """Assess education level."""
        education = candidate.get('education', [])
        
        if not education:
            return 'unknown'
        
        degrees = set()
        for edu in education:
            degree = edu.get('degree', '').lower()
            if 'phd' in degree or 'doctorate' in degree:
                return 'phd'
            elif "master's" in degree or 'masters' in degree:
                degrees.add('masters')
            elif "bachelor's" in degree or 'bachelors' in degree or 'bachelor' in degree:
                degrees.add('bachelors')
        
        if 'masters' in degrees:
            return 'masters'
        elif 'bachelors' in degrees:
            return 'bachelors'
        else:
            return 'other'
    
    def _compute_behavioral_score(self, candidate: Dict) -> float:
        """Compute behavioral signal score."""
        signals = candidate.get('redrob_signals', {})
        
        components = []
        
        # Response rate
        resp_rate = signals.get('recruiter_response_rate', 0)
        components.append(resp_rate)
        
        # Open to work
        open_to_work = signals.get('open_to_work_flag', False)
        components.append(1.0 if open_to_work else 0.5)
        
        # GitHub activity
        github = signals.get('github_activity_score', 0)
        components.append(min(1.0, github / 100))
        
        # Profile completeness
        completeness = signals.get('profile_completeness_score', 0.5)
        components.append(completeness)
        
        # Notice period (fewer days is better)
        notice = signals.get('notice_period_days', 30)
        notice_score = max(0.3, 1.0 - (notice / 90))
        components.append(notice_score)
        
        return sum(components) / len(components) if components else 0.5
