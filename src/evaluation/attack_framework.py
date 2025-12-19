"""
顶级期刊实验套件攻击框架。

按 design.md §6.2 实现攻击 API 和基类。
支持 5 种攻击类型和 A0/A1/A2 威胁级别。

**验证: 属性 3 - 攻击成功率映射一致性**
**验证: 属性 14 - A2 攻击强制存在**
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ThreatLevel(Enum):
    """威胁级别枚举（按 §5.1）。"""
    A0 = "A0"  # 黑盒: 仅观察 Z-view 输出
    A1 = "A1"  # 灰盒: 知道算法和架构
    A2 = "A2"  # 白盒自适应: 知道掩码、预算，可设计自适应损失


class AttackType(Enum):
    """攻击类型枚举（按 §6.1）。"""
    FACE_VERIFICATION = "face_verification"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    RECONSTRUCTION = "reconstruction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    PROPERTY_INFERENCE = "property_inference"


class AttackerStrength(Enum):
    """攻击者强度（用于 CI 例外条款，按 §10.4）。"""
    LITE = "lite"   # 5 轮，1 次实例化，子集数据
    FULL = "full"   # 100 轮，完整实例化，完整数据


# attack_success mapping table per GC7 (frozen)
ATTACK_SUCCESS_MAPPING = {
    AttackType.FACE_VERIFICATION: {
        "metric": "TAR@FAR=1e-3",
        "direction": "lower_is_better",  # 较低的 attack_success = 更好的隐私
    },
    AttackType.ATTRIBUTE_INFERENCE: {
        "metric": "AUC",
        "direction": "lower_is_better",
    },
    AttackType.RECONSTRUCTION: {
        "metric": "identity_similarity",
        "direction": "lower_is_better",
    },
    AttackType.MEMBERSHIP_INFERENCE: {
        "metric": "AUC",
        "direction": "lower_is_better",
    },
    AttackType.PROPERTY_INFERENCE: {
        "metric": "AUC",
        "direction": "lower_is_better",
    },
}


@dataclass
class AttackFitContext:
    """
    攻击训练上下文（按 §6.2 冻结接口）。
    
    包含训练攻击模型所需的所有信息。
    """
    run_id: str
    dataset: str
    task: str
    method: str
    training_mode: str  # P2P/P2Z/Z2Z/Mix2Z
    privacy_level: float
    seed: int
    threat_level: ThreatLevel
    attacker_visible: Dict[str, Any] = field(default_factory=dict)
    
    # 可选元数据
    attacker_strength: AttackerStrength = AttackerStrength.FULL
    degrade_reason: Optional[str] = None  # 如果 strength=lite 则必需
    
    def __post_init__(self):
        """验证上下文。"""
        if isinstance(self.threat_level, str):
            self.threat_level = ThreatLevel(self.threat_level)
        if isinstance(self.attacker_strength, str):
            self.attacker_strength = AttackerStrength(self.attacker_strength)
        
        # 验证 lite 模式需要 degrade_reason
        if self.attacker_strength == AttackerStrength.LITE and not self.degrade_reason:
            raise ValueError(
                "attacker_strength=lite 需要设置 degrade_reason"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便记录日志。"""
        return {
            'run_id': self.run_id,
            'dataset': self.dataset,
            'task': self.task,
            'method': self.method,
            'training_mode': self.training_mode,
            'privacy_level': self.privacy_level,
            'seed': self.seed,
            'threat_level': self.threat_level.value,
            'attacker_visible': self.attacker_visible,
            'attacker_strength': self.attacker_strength.value,
            'degrade_reason': self.degrade_reason,
        }


@dataclass
class AttackEvalContext:
    """
    攻击评估上下文。
    
    包含评估攻击所需的信息。
    """
    run_id: str
    dataset: str
    task: str
    method: str
    training_mode: str
    privacy_level: float
    seed: int
    threat_level: ThreatLevel
    split: str = "test"  # train/val/test
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            'run_id': self.run_id,
            'dataset': self.dataset,
            'task': self.task,
            'method': self.method,
            'training_mode': self.training_mode,
            'privacy_level': self.privacy_level,
            'seed': self.seed,
            'threat_level': self.threat_level.value,
            'split': self.split,
        }


@dataclass
class AttackResult:
    """
    攻击评估结果。
    
    包含攻击指标和元数据。
    """
    attack_type: AttackType
    threat_level: ThreatLevel
    attack_success: float  # 按 GC7 的统一指标
    metric_name: str
    metric_value: float
    status: str = "success"  # success/failed
    
    # 统计字段
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    stat_method: str = "bootstrap"
    n_boot: int = 500
    
    # 元数据
    attacker_strength: AttackerStrength = AttackerStrength.FULL
    degrade_reason: Optional[str] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便 CSV 输出。"""
        return {
            'attack_type': self.attack_type.value,
            'threat_level': self.threat_level.value,
            'attack_success': self.attack_success,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'status': self.status,
            'ci_low': self.ci_low,
            'ci_high': self.ci_high,
            'stat_method': self.stat_method,
            'n_boot': self.n_boot,
            'attacker_strength': self.attacker_strength.value,
            'degrade_reason': self.degrade_reason,
            **self.additional_metrics,
        }


class AttackBase(ABC):
    """
    攻击基类（按 §6.2 冻结签名）。
    
    所有攻击实现必须继承此类。
    """
    
    attack_type: AttackType = None
    
    def __init__(self, device: str = None):
        """
        初始化攻击。
        
        Args:
            device: 计算设备 (cuda/cpu)
        """
        self.device = device or 'cpu'
        self.is_fitted = False
        self.fit_context: Optional[AttackFitContext] = None
    
    @abstractmethod
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练攻击模型。
        
        Args:
            ctx: 攻击训练上下文
            **kwargs: 额外训练数据
        """
        pass
    
    @abstractmethod
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估攻击成功率。
        
        Args:
            ctx: 攻击评估上下文
            **kwargs: 评估数据
            
        Returns:
            包含 attack_success 和指标的 AttackResult
        """
        pass
    
    def get_attack_success(self, metrics: Dict[str, float]) -> float:
        """
        按 GC7 映射表计算统一的 attack_success。
        
        Args:
            metrics: 原始攻击指标
            
        Returns:
            统一的 attack_success 值
        """
        if self.attack_type is None:
            raise NotImplementedError("子类必须设置 attack_type")
        
        mapping = ATTACK_SUCCESS_MAPPING[self.attack_type]
        metric_name = mapping["metric"]
        
        # 将原始指标映射到 attack_success
        if metric_name in metrics:
            return metrics[metric_name]
        
        # 回退映射
        fallback_mappings = {
            "TAR@FAR=1e-3": ["tar_at_far_1e3", "tar", "verification_rate"],
            "AUC": ["auc", "roc_auc", "auroc"],
            "identity_similarity": ["identity_sim", "face_similarity", "cosine_similarity"],
        }
        
        for fallback in fallback_mappings.get(metric_name, []):
            if fallback in metrics:
                return metrics[fallback]
        
        raise ValueError(
            f"无法计算 {self.attack_type} 的 attack_success: "
            f"指标 '{metric_name}' 未在 {list(metrics.keys())} 中找到"
        )
    
    def validate_threat_level(self, ctx: AttackFitContext) -> None:
        """
        验证威胁级别要求。
        
        Args:
            ctx: 攻击上下文
            
        Raises:
            ValueError: 如果威胁级别要求未满足
        """
        if ctx.threat_level == ThreatLevel.A2:
            # A2 要求完整强度（按 GC10 不允许降级）
            if ctx.attacker_strength != AttackerStrength.FULL:
                raise ValueError(
                    "A2 攻击必须使用 attacker_strength=full (GC10)"
                )


class AttackRegistry:
    """
    攻击实现注册表。
    
    管理攻击实例化和验证。
    """
    
    _attacks: Dict[AttackType, type] = {}
    
    @classmethod
    def register(cls, attack_type: AttackType):
        """注册攻击类的装饰器。"""
        def decorator(attack_cls: type):
            cls._attacks[attack_type] = attack_cls
            attack_cls.attack_type = attack_type
            return attack_cls
        return decorator
    
    @classmethod
    def get(cls, attack_type: AttackType) -> type:
        """按类型获取攻击类。"""
        if attack_type not in cls._attacks:
            raise ValueError(f"攻击类型 {attack_type} 未注册")
        return cls._attacks[attack_type]
    
    @classmethod
    def create(cls, attack_type: AttackType, **kwargs) -> AttackBase:
        """创建攻击实例。"""
        attack_cls = cls.get(attack_type)
        return attack_cls(**kwargs)
    
    @classmethod
    def list_attacks(cls) -> List[AttackType]:
        """列出已注册的攻击类型。"""
        return list(cls._attacks.keys())


def validate_a2_exists(attack_results: List[AttackResult]) -> bool:
    """
    验证 A2 攻击结果是否存在。
    
    按属性 14，attack_metrics.csv 必须包含 threat_level=A2。
    
    Args:
        attack_results: 攻击结果列表
        
    Returns:
        如果 A2 存在则返回 True
        
    Raises:
        ValueError: 如果 A2 缺失（硬失败）
    """
    has_a2 = any(
        r.threat_level == ThreatLevel.A2 
        for r in attack_results
    )
    
    if not has_a2:
        raise ValueError(
            "A2 攻击结果缺失（属性 14 违规）。"
            "attack_metrics.csv 必须包含至少一条 threat_level=A2 记录。"
        )
    
    return True


def validate_attack_coverage(
    attack_results: List[AttackResult],
    required_types: List[AttackType] = None,
) -> Dict[str, Any]:
    """
    验证攻击覆盖率。
    
    Args:
        attack_results: 攻击结果列表
        required_types: 必需的攻击类型（默认：全部 5 种）
        
    Returns:
        覆盖率报告
    """
    if required_types is None:
        required_types = list(AttackType)
    
    covered_types = set(r.attack_type for r in attack_results)
    missing_types = set(required_types) - covered_types
    
    coverage = len(covered_types) / len(required_types) if required_types else 1.0
    
    return {
        'coverage': coverage,
        'covered_types': [t.value for t in covered_types],
        'missing_types': [t.value for t in missing_types],
        'total_results': len(attack_results),
        'has_a2': any(r.threat_level == ThreatLevel.A2 for r in attack_results),
    }
