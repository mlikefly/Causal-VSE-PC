# -*- coding: utf-8 -*-
"""
对抗性语义人脸加密系统 - 评估模块

包含：
- SecurityMetrics: 基础安全指标
- DualViewSecurityMetrics: 双视图安全指标评估器
- AttackEvaluator: 攻击评估器（四类攻击）
- UtilityEvaluator: 任务效用评估器
- BaselineComparator: Baseline 对标比较器
"""

from .security_metrics import SecurityMetrics
from .dual_view_metrics import (
    DualViewSecurityMetrics,
    ZViewMetrics,
    CViewMetrics,
    get_zview_metric_keys,
    get_cview_metric_keys
)
from .attack_evaluator import (
    AttackEvaluator,
    AttackEvaluationMatrix,
    IdentityAttackResult,
    ReconstructionAttackResult,
    AttributeInferenceResult,
    LinkabilityAttackResult
)
from .utility_evaluator import (
    UtilityEvaluator,
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
    FairnessResult,
    PrivacyUtilityCurve
)
from .baseline_comparator import (
    BaselineComparator,
    BaselineResult,
    ParetoPoint,
    InstaHideBaseline,
    P3Baseline,
    ChaoticBaseline
)

__all__ = [
    # 基础安全指标
    'SecurityMetrics',
    # 双视图安全指标
    'DualViewSecurityMetrics',
    'ZViewMetrics',
    'CViewMetrics',
    'get_zview_metric_keys',
    'get_cview_metric_keys',
    # 攻击评估
    'AttackEvaluator',
    'AttackEvaluationMatrix',
    'IdentityAttackResult',
    'ReconstructionAttackResult',
    'AttributeInferenceResult',
    'LinkabilityAttackResult',
    # 效用评估
    'UtilityEvaluator',
    'ClassificationResult',
    'DetectionResult',
    'SegmentationResult',
    'FairnessResult',
    'PrivacyUtilityCurve',
    # Baseline 对标
    'BaselineComparator',
    'BaselineResult',
    'ParetoPoint',
    'InstaHideBaseline',
    'P3Baseline',
    'ChaoticBaseline'
]









