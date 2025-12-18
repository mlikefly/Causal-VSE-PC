"""
Attack implementations for Top-Journal Experiment Suite.

Implements 5 attack types per design.md §6.1:
1. Face Verification Attack
2. Attribute Inference Attack
3. Reconstruction Attack
4. Membership Inference Attack
5. Property Inference Attack

Plus A2 Adaptive Attacker per §6.3.
"""

from .face_verification import FaceVerificationAttack
from .attribute_inference import AttributeInferenceAttack
from .reconstruction import ReconstructionAttack
from .membership_inference import MembershipInferenceAttack
from .property_inference import PropertyInferenceAttack
from .adaptive_attacker import (
    AdaptiveAttacker,
    AdaptiveStrategy,
    A2StrengthContract,
    compute_worst_case_attack_success,
)

__all__ = [
    'FaceVerificationAttack',
    'AttributeInferenceAttack',
    'ReconstructionAttack',
    'MembershipInferenceAttack',
    'PropertyInferenceAttack',
    # A2 Adaptive
    'AdaptiveAttacker',
    'AdaptiveStrategy',
    'A2StrengthContract',
    'compute_worst_case_attack_success',
]
