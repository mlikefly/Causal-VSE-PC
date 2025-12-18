# -*- coding: utf-8 -*-
"""
训练模块 (Training Module)

包含：
- TrainingModeManager: 训练模式管理器（P2P/P2Z/Z2Z/Mix2Z）
- UtilityThresholdChecker: 效用门槛检查器

**Validates: Property 6, Property 7**
"""

from .training_mode_manager import (
    TrainingMode,
    ViewAccessError,
    TrainingModeManager,
    UtilityThresholdChecker,
    UtilityMetricsRow,
    UTILITY_THRESHOLDS,
    generate_utility_metrics_csv,
    generate_utility_failure_analysis,
)

__all__ = [
    'TrainingMode',
    'ViewAccessError',
    'TrainingModeManager',
    'UtilityThresholdChecker',
    'UtilityMetricsRow',
    'UTILITY_THRESHOLDS',
    'generate_utility_metrics_csv',
    'generate_utility_failure_analysis',
]
