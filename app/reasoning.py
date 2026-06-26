"""
Reasoning Module - Generate evidence-grounded explanations for rankings.

Creates 1-2 sentence justifications mentioning actual candidate facts,
concerns, and compatibility indicators.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ReasoningGenerator:
    """Generates evidence-grounded reasoning for candidate rankings."""
    
    def __init__(self, jd_profile: Dict[str, Any] = None):
        """Initialize reasoning generator."""
        self.jd_profile = jd_profile or {}
    
    def generate_reasoning(self, candidate: Dict[str, Any], 
                          features: Dict[str, float],
                          evidence: Dict[str, Any] = None) -> str:
        """
        Generate reasoning for a ranked candidate.
        
        Returns a 1-2 sentence explanation mentioning:
        - Actual facts from profile
        - Key strengths
        - Any concerns
        - Specific metrics
        
        Args:
            candidate: Candidate profile
            features: Computed features
            evidence: Evidence from scoring
            
        Returns:
            Reasoning string (max 200 chars to avoid hallucination)
        """
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        signals = candidate.get('redrob_signals', {})
        
        parts = []
        
        # Fact 1: Current role + experience
        title = profile.get('current_title', 'Professional')
        company = profile.get('current_company', 'current company')
        years = profile.get('years_of_experience', 0)
        
        parts.append(f"{years:.0f} years in {title} roles")
        
        # Fact 2: Key skills
        skills = candidate.get('skills', [])
        skill_names = [s['name'] for s in skills[:3]]  # Top 3 skills
        if skill_names:
            parts.append(f"expertise in {', '.join(skill_names)}")
        
        # Fact 3: Behavioral signal
        resp_rate = signals.get('recruiter_response_rate', 0)
        if resp_rate > 0.6:
            parts.append(f"strong engagement ({resp_rate:.0%} response rate)")
        
        notice = signals.get('notice_period_days', 30)
        open_to_work = signals.get('open_to_work_flag', False)
        
        if notice <= 30 or open_to_work:
            availability = []
            if notice <= 30:
                availability.append(f"{notice}d notice")
            if open_to_work:
                availability.append("actively looking")
            parts.append("; ".join(availability))
        
        # Concern if any
        concerns = self._identify_concerns(candidate, features)
        if concerns:
            parts.append(f"note: {concerns}")
        
        # Join and truncate
        reasoning = "; ".join(parts) + "."
        
        # Truncate to 200 chars
        if len(reasoning) > 200:
            reasoning = reasoning[:197] + "..."
        
        return reasoning
    
    def _identify_concerns(self, candidate: Dict[str, Any], 
                          features: Dict[str, float]) -> str:
        """Identify and note any concerns about the candidate."""
        concerns = []
        
        # Low experience
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        if years < 2:
            concerns.append("early-career")
        
        # High notice period
        signals = candidate.get('redrob_signals', {})
        notice = signals.get('notice_period_days', 30)
        if notice > 90:
            concerns.append(f"{notice}d notice period")
        
        # Low engagement
        resp_rate = signals.get('recruiter_response_rate', 0)
        if resp_rate < 0.3:
            concerns.append("limited responsiveness")
        
        # Research-focused without production experience
        profile = candidate.get('profile', {})
        title = profile.get('current_title', '').lower()
        if 'researcher' in title:
            career = candidate.get('career_history', [])
            has_production = any(
                'production' in p.get('description', '').lower() 
                for p in career
            )
            if not has_production:
                concerns.append("research-focused background")
        
        return "; ".join(concerns) if concerns else ""
    
    def generate_batch_reasoning(self, candidates: List[Dict[str, Any]],
                                features_list: List[Dict[str, float]]) -> Dict[int, str]:
        """Generate reasoning for multiple candidates."""
        reasoning = {}
        for i, (candidate, features) in enumerate(zip(candidates, features_list)):
            reasoning[i] = self.generate_reasoning(candidate, features)
        return reasoning
