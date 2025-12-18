# -*- coding: utf-8 -*-
"""
对抗性语义人脸加密系统 - 评估模块

包含：
- SecurityMetrics: 基础安全指标
- DualViewSecurityMetrics: 双视图安全指标评估器
- AttackEvaluator: 攻击评估器（四类攻击）
- UtilityEvaluator: 任务效用评估器
- BaselineComparator: Baseline 对标比较器
- AttackFramework: 五类攻击框架（顶刊实验套件）
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
from .attack_framework import (
    AttackBase,
    AttackFitContext,
    AttackEvalContext,
    AttackResult,
    AttackType,
    ThreatLevel,
    AttackerStrength,
    AttackRegistry,
    ATTACK_SUCCESS_MAPPING,
    validate_a2_exists,
    validate_attack_coverage,
)
from .attack_normalizer import (
    AttackNormalizer,
    NormalizationResult,
    PrivacySummary,
    NORMALIZATION_BOUNDS,
    THREAT_LEVEL_WEIGHTS,
    normalize_attack_results,
)
from .attacks import (
    FaceVerificationAttack,
    AttributeInferenceAttack,
    ReconstructionAttack,
    MembershipInferenceAttack,
    PropertyInferenceAttack,
    AdaptiveAttacker,
    A2StrengthContract,
    compute_worst_case_attack_success,
)
from .cview_security import (
    TamperType,
    FlipType,
    SecurityTestStatus,
    NIST_MIN_BITS,
    TAMPER_FAIL_RATE_THRESHOLD,
    REPLAY_REJECT_RATE_THRESHOLD,
    NIST_P_VALUE_THRESHOLD,
    AVALANCHE_FLIP_RATE_RANGE,
    TamperTestResult,
    TamperTestSummary,
    ReplayTestResult,
    NISTTestResult,
    NISTTestSummary,
    AvalancheTestResult,
    SecurityReport,
    TamperTester,
    ReplayTester,
    NISTTestRunner,
    AvalancheEffectTester,
    CViewSecurityEvaluator,
)
from .causal_effects import (
    InterventionGrid,
    InterventionResult,
    CausalEffectRow,
    CausalEffectEstimator,
    CausalBudgetOptimizer,
    CausalEffectsEvaluator,
    create_intervention_grid,
    run_causal_analysis,
)
from .baseline_matrix import (
    BaselineName,
    TaskType,
    TrainingMode as BaselineTrainingMode,
    BASELINE_CAPABILITIES,
    BaselineComparisonRow,
    BaselineMetrics,
    PixelationBaseline,
    GaussianBlurBaseline,
    DPSGDBaseline,
    BaselineMatrixComparator,
    BaselineEvaluator,
    run_baseline_comparison,
)
from .statistics_engine import (
    MIN_N_BOOT,
    DEFAULT_ALPHA,
    DEFAULT_CI_LEVEL,
    BootstrapCIResult,
    MultipleComparisonResult,
    StatisticalSummary,
    StatisticsEngine,
    StatisticsValidator,
    compute_bootstrap_ci,
    bh_fdr_correction,
    generate_family_id,
)
from .figure_generator import (
    FigureSpecs,
    FigureManifestEntry,
    FigureManifest,
    FigureGenerator,
    generate_all_figures,
)
from .ci_integration import (
    CIMode,
    AttackerStrength as CIAttackerStrength,
    SMOKE_TEST_CONFIG,
    FULL_CONFIG,
    SplitLeakageResult,
    ShadowSplitResult,
    DataLeakageChecker,
    CIIntegration,
    ValidateRunOnePage,
    run_ci_integration,
)
from .ablation_runner import (
    AblationID,
    ABLATION_CONFIGS,
    AblationResult,
    AblationCSVRow,
    AblationRunner,
    AblationEvaluator,
    run_ablation_study,
    get_ablation_configs,
)
from .robustness_efficiency import (
    PerturbationType,
    RobustnessMetricsRow,
    EfficiencyMetricsRow,
    RobustnessEvaluator,
    EfficiencyEvaluator,
    run_robustness_evaluation,
    run_efficiency_evaluation,
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
    'ChaoticBaseline',
    # 攻击框架（顶刊实验套件）
    'AttackBase',
    'AttackFitContext',
    'AttackEvalContext',
    'AttackResult',
    'AttackType',
    'ThreatLevel',
    'AttackerStrength',
    'AttackRegistry',
    'ATTACK_SUCCESS_MAPPING',
    'validate_a2_exists',
    'validate_attack_coverage',
    # 归一化
    'AttackNormalizer',
    'NormalizationResult',
    'PrivacySummary',
    'NORMALIZATION_BOUNDS',
    'THREAT_LEVEL_WEIGHTS',
    'normalize_attack_results',
    # 五类攻击实现
    'FaceVerificationAttack',
    'AttributeInferenceAttack',
    'ReconstructionAttack',
    'MembershipInferenceAttack',
    'PropertyInferenceAttack',
    'AdaptiveAttacker',
    'A2StrengthContract',
    'compute_worst_case_attack_success',
    # C-view 安全评估套件
    'TamperType',
    'FlipType',
    'SecurityTestStatus',
    'NIST_MIN_BITS',
    'TAMPER_FAIL_RATE_THRESHOLD',
    'REPLAY_REJECT_RATE_THRESHOLD',
    'NIST_P_VALUE_THRESHOLD',
    'AVALANCHE_FLIP_RATE_RANGE',
    'TamperTestResult',
    'TamperTestSummary',
    'ReplayTestResult',
    'NISTTestResult',
    'NISTTestSummary',
    'AvalancheTestResult',
    'SecurityReport',
    'TamperTester',
    'ReplayTester',
    'NISTTestRunner',
    'AvalancheEffectTester',
    'CViewSecurityEvaluator',
    # 因果效应估计（T7）
    'InterventionGrid',
    'InterventionResult',
    'CausalEffectRow',
    'CausalEffectEstimator',
    'CausalBudgetOptimizer',
    'CausalEffectsEvaluator',
    'create_intervention_grid',
    'run_causal_analysis',
    # 基线矩阵与对比（T8）
    'BaselineName',
    'TaskType',
    'BaselineTrainingMode',
    'BASELINE_CAPABILITIES',
    'BaselineComparisonRow',
    'BaselineMetrics',
    'PixelationBaseline',
    'GaussianBlurBaseline',
    'DPSGDBaseline',
    'BaselineMatrixComparator',
    'BaselineEvaluator',
    'run_baseline_comparison',
    # 统计引擎（T9）
    'MIN_N_BOOT',
    'DEFAULT_ALPHA',
    'DEFAULT_CI_LEVEL',
    'BootstrapCIResult',
    'MultipleComparisonResult',
    'StatisticalSummary',
    'StatisticsEngine',
    'StatisticsValidator',
    'compute_bootstrap_ci',
    'bh_fdr_correction',
    'generate_family_id',
    # 图表生成（T10）
    'FigureSpecs',
    'FigureManifestEntry',
    'FigureManifest',
    'FigureGenerator',
    'generate_all_figures',
    # CI 集成（T11）
    'CIMode',
    'CIAttackerStrength',
    'SMOKE_TEST_CONFIG',
    'FULL_CONFIG',
    'SplitLeakageResult',
    'ShadowSplitResult',
    'DataLeakageChecker',
    'CIIntegration',
    'ValidateRunOnePage',
    'run_ci_integration',
    # 消融实验（T12）
    'AblationID',
    'ABLATION_CONFIGS',
    'AblationResult',
    'AblationCSVRow',
    'AblationRunner',
    'AblationEvaluator',
    'run_ablation_study',
    'get_ablation_configs',
    # 稳健性与效率（T13）
    'PerturbationType',
    'RobustnessMetricsRow',
    'EfficiencyMetricsRow',
    'RobustnessEvaluator',
    'EfficiencyEvaluator',
    'run_robustness_evaluation',
    'run_efficiency_evaluation',
]









