"""
Attack Success Normalizer for Top-Journal Experiment Suite.

Implements normalization per design.md §7.3.
Computes privacy_protection per §7.1.
Computes summary metrics per §7.4.

**Validates: Requirements §7.3, §7.4**
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .attack_framework import AttackType, ThreatLevel


# Normalization bounds per attack type (frozen per §7.3)
NORMALIZATION_BOUNDS = {
    AttackType.FACE_VERIFICATION: {
        'lower': 'random',  # Random guess TAR
        'upper': 'p2p',     # P2P mode TAR
        'random_value': 0.001,  # TAR@FAR=1e-3 for random
    },
    AttackType.ATTRIBUTE_INFERENCE: {
        'lower': 0.5,       # Random guess AUC
        'upper': 'p2p',     # P2P mode AUC
    },
    AttackType.RECONSTRUCTION: {
        'lower': 0.0,       # Completely dissimilar
        'upper': 'p2p',     # P2P mode similarity
    },
    AttackType.MEMBERSHIP_INFERENCE: {
        'lower': 0.5,       # Random guess AUC
        'upper': 'p2p',     # P2P mode AUC
    },
    AttackType.PROPERTY_INFERENCE: {
        'lower': 0.5,       # Random guess AUC
        'upper': 'p2p',     # P2P mode AUC
    },
}

# Threat level weights for weighted privacy protection (frozen per §7.4)
THREAT_LEVEL_WEIGHTS = {
    ThreatLevel.A0: 0.2,
    ThreatLevel.A1: 0.3,
    ThreatLevel.A2: 0.5,
}


@dataclass
class NormalizationResult:
    """Result of attack success normalization."""
    attack_type: AttackType
    raw_attack_success: float
    normalized_attack_success: float
    privacy_protection: float
    lower_bound: float
    upper_bound: float


@dataclass
class PrivacySummary:
    """Summary privacy metrics per §7.4."""
    avg_privacy_protection: float
    worst_case_privacy_protection: float
    weighted_privacy_protection: float
    per_attack_type: Dict[str, float] = field(default_factory=dict)
    per_threat_level: Dict[str, float] = field(default_factory=dict)


class AttackNormalizer:
    """
    Normalizes attack success values for cross-attack comparability.
    
    Normalization formulas per §7.3:
    - face_verification: (x - x_random) / (x_P2P - x_random)
    - attribute_inference: (x - 0.5) / (x_P2P - 0.5)
    - reconstruction: x / x_P2P
    - membership_inference: (x - 0.5) / (x_P2P - 0.5)
    - property_inference: (x - 0.5) / (x_P2P - 0.5)
    
    Normalized range: [0, 1]
    - 0 = complete protection
    - 1 = no protection (equivalent to P2P)
    """
    
    def __init__(self, p2p_baselines: Dict[AttackType, float] = None):
        """
        Initialize normalizer.
        
        Args:
            p2p_baselines: P2P baseline values for each attack type
        """
        self.p2p_baselines = p2p_baselines or {}
    
    def set_p2p_baseline(self, attack_type: AttackType, value: float) -> None:
        """Set P2P baseline for an attack type."""
        self.p2p_baselines[attack_type] = value
    
    def normalize(
        self,
        attack_type: AttackType,
        attack_success: float,
        p2p_value: Optional[float] = None,
    ) -> NormalizationResult:
        """
        Normalize attack success value.
        
        Args:
            attack_type: Type of attack
            attack_success: Raw attack success value
            p2p_value: P2P baseline (uses stored if not provided)
            
        Returns:
            NormalizationResult with normalized value and privacy_protection
        """
        # Get P2P baseline
        if p2p_value is None:
            p2p_value = self.p2p_baselines.get(attack_type)
        
        if p2p_value is None:
            # Default P2P values if not provided
            default_p2p = {
                AttackType.FACE_VERIFICATION: 0.9,
                AttackType.ATTRIBUTE_INFERENCE: 0.95,
                AttackType.RECONSTRUCTION: 0.95,
                AttackType.MEMBERSHIP_INFERENCE: 0.7,
                AttackType.PROPERTY_INFERENCE: 0.8,
            }
            p2p_value = default_p2p.get(attack_type, 1.0)
        
        # Get bounds
        bounds = NORMALIZATION_BOUNDS.get(attack_type, {})
        lower = bounds.get('lower', 0.0)
        
        if lower == 'random':
            lower = bounds.get('random_value', 0.0)
        elif isinstance(lower, str):
            lower = 0.0
        
        upper = p2p_value
        
        # Normalize based on attack type
        normalized = self._normalize_value(
            attack_type, attack_success, lower, upper
        )
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Compute privacy protection
        privacy_protection = 1.0 - normalized
        
        return NormalizationResult(
            attack_type=attack_type,
            raw_attack_success=attack_success,
            normalized_attack_success=normalized,
            privacy_protection=privacy_protection,
            lower_bound=lower,
            upper_bound=upper,
        )
    
    def _normalize_value(
        self,
        attack_type: AttackType,
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        """Apply normalization formula based on attack type."""
        if attack_type == AttackType.FACE_VERIFICATION:
            # (x - x_random) / (x_P2P - x_random)
            if upper - lower == 0:
                return 0.0
            return (value - lower) / (upper - lower)
        
        elif attack_type in [
            AttackType.ATTRIBUTE_INFERENCE,
            AttackType.MEMBERSHIP_INFERENCE,
            AttackType.PROPERTY_INFERENCE,
        ]:
            # (x - 0.5) / (x_P2P - 0.5)
            if upper - 0.5 == 0:
                return 0.0
            return (value - 0.5) / (upper - 0.5)
        
        elif attack_type == AttackType.RECONSTRUCTION:
            # x / x_P2P
            if upper == 0:
                return 0.0
            return value / upper
        
        else:
            # Default: linear normalization
            if upper - lower == 0:
                return 0.0
            return (value - lower) / (upper - lower)
    
    def compute_privacy_protection(
        self,
        attack_type: AttackType,
        attack_success: float,
        p2p_value: Optional[float] = None,
    ) -> float:
        """
        Compute privacy protection score.
        
        privacy_protection = 1 - normalized(attack_success)
        
        Args:
            attack_type: Type of attack
            attack_success: Raw attack success value
            p2p_value: P2P baseline
            
        Returns:
            Privacy protection score in [0, 1]
        """
        result = self.normalize(attack_type, attack_success, p2p_value)
        return result.privacy_protection
    
    def compute_summary(
        self,
        results: List[NormalizationResult],
        threat_levels: Optional[List[ThreatLevel]] = None,
    ) -> PrivacySummary:
        """
        Compute summary privacy metrics per §7.4.
        
        Args:
            results: List of normalization results
            threat_levels: Corresponding threat levels (for weighted)
            
        Returns:
            PrivacySummary with avg, worst-case, and weighted metrics
        """
        if not results:
            return PrivacySummary(
                avg_privacy_protection=0.0,
                worst_case_privacy_protection=0.0,
                weighted_privacy_protection=0.0,
            )
        
        privacy_values = [r.privacy_protection for r in results]
        
        # Average privacy protection (uniform weights)
        avg_pp = float(np.mean(privacy_values))
        
        # Worst-case privacy protection (minimum)
        worst_pp = float(np.min(privacy_values))
        
        # Weighted privacy protection by threat level
        if threat_levels and len(threat_levels) == len(results):
            weighted_sum = 0.0
            weight_sum = 0.0
            for result, threat in zip(results, threat_levels):
                weight = THREAT_LEVEL_WEIGHTS.get(threat, 0.33)
                weighted_sum += result.privacy_protection * weight
                weight_sum += weight
            weighted_pp = weighted_sum / weight_sum if weight_sum > 0 else avg_pp
        else:
            weighted_pp = avg_pp
        
        # Per attack type
        per_attack = {}
        for attack_type in AttackType:
            type_results = [r for r in results if r.attack_type == attack_type]
            if type_results:
                per_attack[attack_type.value] = float(np.mean(
                    [r.privacy_protection for r in type_results]
                ))
        
        # Per threat level
        per_threat = {}
        if threat_levels:
            for threat in ThreatLevel:
                threat_results = [
                    r for r, t in zip(results, threat_levels) 
                    if t == threat
                ]
                if threat_results:
                    per_threat[threat.value] = float(np.mean(
                        [r.privacy_protection for r in threat_results]
                    ))
        
        return PrivacySummary(
            avg_privacy_protection=avg_pp,
            worst_case_privacy_protection=worst_pp,
            weighted_privacy_protection=weighted_pp,
            per_attack_type=per_attack,
            per_threat_level=per_threat,
        )


def normalize_attack_results(
    attack_results: List[Dict],
    p2p_baselines: Dict[str, float],
) -> List[Dict]:
    """
    Normalize a list of attack result dictionaries.
    
    Args:
        attack_results: List of attack result dicts with 'attack_type' and 'attack_success'
        p2p_baselines: P2P baseline values keyed by attack_type string
        
    Returns:
        List of dicts with added 'normalized_attack_success' and 'privacy_protection'
    """
    normalizer = AttackNormalizer()
    
    # Set baselines
    for attack_type_str, value in p2p_baselines.items():
        try:
            attack_type = AttackType(attack_type_str)
            normalizer.set_p2p_baseline(attack_type, value)
        except ValueError:
            pass
    
    normalized_results = []
    for result in attack_results:
        result_copy = result.copy()
        
        try:
            attack_type = AttackType(result['attack_type'])
            attack_success = result['attack_success']
            
            norm_result = normalizer.normalize(attack_type, attack_success)
            
            result_copy['normalized_attack_success'] = norm_result.normalized_attack_success
            result_copy['privacy_protection'] = norm_result.privacy_protection
        except (KeyError, ValueError):
            result_copy['normalized_attack_success'] = None
            result_copy['privacy_protection'] = None
        
        normalized_results.append(result_copy)
    
    return normalized_results
