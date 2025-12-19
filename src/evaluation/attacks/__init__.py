"""
顶级期刊实验套件的攻击实现。

按 design.md §6.1 实现 5 种攻击类型：
1. 人脸验证攻击
2. 属性推断攻击
3. 重建攻击
4. 成员推断攻击
5. 属性推断攻击

以及按 §6.3 的 A2 自适应攻击器。
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
