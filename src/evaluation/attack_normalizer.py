"""
攻击成功率归一化器 - 顶级期刊实验套件。

实现 design.md §7.3 中的归一化方法。
计算 §7.1 中的隐私保护度。
计算 §7.4 中的汇总指标。

**验证: 需求 §7.3, §7.4**
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .attack_framework import AttackType, ThreatLevel


# 每种攻击类型的归一化边界（按 §7.3 冻结）
NORMALIZATION_BOUNDS = {
    AttackType.FACE_VERIFICATION: {
        'lower': 'random',  # 随机猜测 TAR
        'upper': 'p2p',     # P2P 模式 TAR
        'random_value': 0.001,  # 随机情况下的 TAR@FAR=1e-3
    },
    AttackType.ATTRIBUTE_INFERENCE: {
        'lower': 0.5,       # 随机猜测 AUC
        'upper': 'p2p',     # P2P 模式 AUC
    },
    AttackType.RECONSTRUCTION: {
        'lower': 0.0,       # Completely dissimilar
        'upper': 'p2p',     # P2P mode similarity
    },
    AttackType.MEMBERSHIP_INFERENCE: {
        'lower': 0.5,       # 随机猜测 AUC
        'upper': 'p2p',     # P2P 模式 AUC
    },
    AttackType.PROPERTY_INFERENCE: {
        'lower': 0.5,       # 随机猜测 AUC
        'upper': 'p2p',     # P2P 模式 AUC
    },
}

# 加权隐私保护的威胁级别权重（按 §7.4 冻结）
THREAT_LEVEL_WEIGHTS = {
    ThreatLevel.A0: 0.2,
    ThreatLevel.A1: 0.3,
    ThreatLevel.A2: 0.5,
}


@dataclass
class NormalizationResult:
    """攻击成功率归一化结果。"""
    attack_type: AttackType
    raw_attack_success: float
    normalized_attack_success: float
    privacy_protection: float
    lower_bound: float
    upper_bound: float


@dataclass
class PrivacySummary:
    """按 §7.4 的汇总隐私指标。"""
    avg_privacy_protection: float
    worst_case_privacy_protection: float
    weighted_privacy_protection: float
    per_attack_type: Dict[str, float] = field(default_factory=dict)
    per_threat_level: Dict[str, float] = field(default_factory=dict)


class AttackNormalizer:
    """
    归一化攻击成功率以实现跨攻击可比性。
    
    按 §7.3 的归一化公式：
    - face_verification: (x - x_random) / (x_P2P - x_random)
    - attribute_inference: (x - 0.5) / (x_P2P - 0.5)
    - reconstruction: x / x_P2P
    - membership_inference: (x - 0.5) / (x_P2P - 0.5)
    - property_inference: (x - 0.5) / (x_P2P - 0.5)
    
    归一化范围: [0, 1]
    - 0 = 完全保护
    - 1 = 无保护（等同于 P2P）
    """
    
    def __init__(self, p2p_baselines: Dict[AttackType, float] = None):
        """
        初始化归一化器。
        
        参数:
            p2p_baselines: 每种攻击类型的 P2P 基线值
        """
        self.p2p_baselines = p2p_baselines or {}
    
    def set_p2p_baseline(self, attack_type: AttackType, value: float) -> None:
        """设置某攻击类型的 P2P 基线。"""
        self.p2p_baselines[attack_type] = value
    
    def normalize(
        self,
        attack_type: AttackType,
        attack_success: float,
        p2p_value: Optional[float] = None,
    ) -> NormalizationResult:
        """
        归一化攻击成功率值。
        
        参数:
            attack_type: 攻击类型
            attack_success: 原始攻击成功率值
            p2p_value: P2P 基线（如果未提供则使用存储的值）
            
        返回:
            包含归一化值和 privacy_protection 的 NormalizationResult
        """
        # 获取 P2P 基线
        if p2p_value is None:
            p2p_value = self.p2p_baselines.get(attack_type)
        
        if p2p_value is None:
            # 如果未提供则使用默认 P2P 值
            default_p2p = {
                AttackType.FACE_VERIFICATION: 0.9,
                AttackType.ATTRIBUTE_INFERENCE: 0.95,
                AttackType.RECONSTRUCTION: 0.95,
                AttackType.MEMBERSHIP_INFERENCE: 0.7,
                AttackType.PROPERTY_INFERENCE: 0.8,
            }
            p2p_value = default_p2p.get(attack_type, 1.0)
        
        # 获取边界
        bounds = NORMALIZATION_BOUNDS.get(attack_type, {})
        lower = bounds.get('lower', 0.0)
        
        if lower == 'random':
            lower = bounds.get('random_value', 0.0)
        elif isinstance(lower, str):
            lower = 0.0
        
        upper = p2p_value
        
        # 根据攻击类型归一化
        normalized = self._normalize_value(
            attack_type, attack_success, lower, upper
        )
        
        # 限制到 [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # 计算隐私保护
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
        """根据攻击类型应用归一化公式。"""
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
        计算隐私保护分数。
        
        privacy_protection = 1 - normalized(attack_success)
        
        参数:
            attack_type: 攻击类型
            attack_success: 原始攻击成功率值
            p2p_value: P2P 基线
            
        返回:
            [0, 1] 范围内的隐私保护分数
        """
        result = self.normalize(attack_type, attack_success, p2p_value)
        return result.privacy_protection
    
    def compute_summary(
        self,
        results: List[NormalizationResult],
        threat_levels: Optional[List[ThreatLevel]] = None,
    ) -> PrivacySummary:
        """
        按 §7.4 计算汇总隐私指标。
        
        参数:
            results: 归一化结果列表
            threat_levels: 对应的威胁级别（用于加权）
            
        返回:
            包含平均、最坏情况和加权指标的 PrivacySummary
        """
        if not results:
            return PrivacySummary(
                avg_privacy_protection=0.0,
                worst_case_privacy_protection=0.0,
                weighted_privacy_protection=0.0,
            )
        
        privacy_values = [r.privacy_protection for r in results]
        
        # 平均隐私保护（均匀权重）
        avg_pp = float(np.mean(privacy_values))
        
        # 最坏情况隐私保护（最小值）
        worst_pp = float(np.min(privacy_values))
        
        # 按威胁级别加权的隐私保护
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
        
        # 按攻击类型
        per_attack = {}
        for attack_type in AttackType:
            type_results = [r for r in results if r.attack_type == attack_type]
            if type_results:
                per_attack[attack_type.value] = float(np.mean(
                    [r.privacy_protection for r in type_results]
                ))
        
        # 按威胁级别
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
    归一化攻击结果字典列表。
    
    参数:
        attack_results: 包含 'attack_type' 和 'attack_success' 的攻击结果字典列表
        p2p_baselines: 按 attack_type 字符串键控的 P2P 基线值
        
    返回:
        添加了 'normalized_attack_success' 和 'privacy_protection' 的字典列表
    """
    normalizer = AttackNormalizer()
    
    # 设置基线
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
