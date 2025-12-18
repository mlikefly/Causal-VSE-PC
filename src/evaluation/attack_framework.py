"""
Attack Framework for Top-Journal Experiment Suite.

Implements the attack API and base classes per design.md §6.2.
Supports 5 attack types with A0/A1/A2 threat levels.

**Validates: Property 3 - 攻击成功率映射一致性**
**Validates: Property 14 - A2 攻击强制存在**
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ThreatLevel(Enum):
    """Threat level enumeration per §5.1."""
    A0 = "A0"  # Black-box: only observes Z-view output
    A1 = "A1"  # Gray-box: knows algorithm and architecture
    A2 = "A2"  # White-box Adaptive: knows mask, budget, can design adaptive loss


class AttackType(Enum):
    """Attack type enumeration per §6.1."""
    FACE_VERIFICATION = "face_verification"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    RECONSTRUCTION = "reconstruction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    PROPERTY_INFERENCE = "property_inference"


class AttackerStrength(Enum):
    """Attacker strength for CI exception clause per §10.4."""
    LITE = "lite"   # 5 epochs, 1 instantiation, subset data
    FULL = "full"   # 100 epochs, full instantiation, full data


# attack_success mapping table per GC7 (frozen)
ATTACK_SUCCESS_MAPPING = {
    AttackType.FACE_VERIFICATION: {
        "metric": "TAR@FAR=1e-3",
        "direction": "lower_is_better",  # Lower attack_success = better privacy
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
    Attack training context (frozen interface per §6.2).
    
    Contains all information needed to train an attack model.
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
    
    # Optional metadata
    attacker_strength: AttackerStrength = AttackerStrength.FULL
    degrade_reason: Optional[str] = None  # Required if strength=lite
    
    def __post_init__(self):
        """Validate context."""
        if isinstance(self.threat_level, str):
            self.threat_level = ThreatLevel(self.threat_level)
        if isinstance(self.attacker_strength, str):
            self.attacker_strength = AttackerStrength(self.attacker_strength)
        
        # Validate lite mode requires degrade_reason
        if self.attacker_strength == AttackerStrength.LITE and not self.degrade_reason:
            raise ValueError(
                "attacker_strength=lite requires degrade_reason to be set"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
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
    Attack evaluation context.
    
    Contains information needed to evaluate an attack.
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
        """Convert to dictionary."""
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
    Attack evaluation result.
    
    Contains attack metrics and metadata.
    """
    attack_type: AttackType
    threat_level: ThreatLevel
    attack_success: float  # Unified metric per GC7
    metric_name: str
    metric_value: float
    status: str = "success"  # success/failed
    
    # Statistical fields
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    stat_method: str = "bootstrap"
    n_boot: int = 500
    
    # Metadata
    attacker_strength: AttackerStrength = AttackerStrength.FULL
    degrade_reason: Optional[str] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV output."""
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
    Attack base class (frozen signature per §6.2).
    
    All attack implementations must inherit from this class.
    """
    
    attack_type: AttackType = None
    
    def __init__(self, device: str = None):
        """
        Initialize attack.
        
        Args:
            device: Compute device (cuda/cpu)
        """
        self.device = device or 'cpu'
        self.is_fitted = False
        self.fit_context: Optional[AttackFitContext] = None
    
    @abstractmethod
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train attack model.
        
        Args:
            ctx: Attack training context
            **kwargs: Additional training data
        """
        pass
    
    @abstractmethod
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        Evaluate attack success rate.
        
        Args:
            ctx: Attack evaluation context
            **kwargs: Evaluation data
            
        Returns:
            AttackResult with attack_success and metrics
        """
        pass
    
    def get_attack_success(self, metrics: Dict[str, float]) -> float:
        """
        Compute unified attack_success per GC7 mapping table.
        
        Args:
            metrics: Raw attack metrics
            
        Returns:
            Unified attack_success value
        """
        if self.attack_type is None:
            raise NotImplementedError("attack_type must be set in subclass")
        
        mapping = ATTACK_SUCCESS_MAPPING[self.attack_type]
        metric_name = mapping["metric"]
        
        # Map raw metric to attack_success
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Fallback mappings
        fallback_mappings = {
            "TAR@FAR=1e-3": ["tar_at_far_1e3", "tar", "verification_rate"],
            "AUC": ["auc", "roc_auc", "auroc"],
            "identity_similarity": ["identity_sim", "face_similarity", "cosine_similarity"],
        }
        
        for fallback in fallback_mappings.get(metric_name, []):
            if fallback in metrics:
                return metrics[fallback]
        
        raise ValueError(
            f"Cannot compute attack_success for {self.attack_type}: "
            f"metric '{metric_name}' not found in {list(metrics.keys())}"
        )
    
    def validate_threat_level(self, ctx: AttackFitContext) -> None:
        """
        Validate threat level requirements.
        
        Args:
            ctx: Attack context
            
        Raises:
            ValueError: If threat level requirements not met
        """
        if ctx.threat_level == ThreatLevel.A2:
            # A2 requires full strength (no degrade allowed per GC10)
            if ctx.attacker_strength != AttackerStrength.FULL:
                raise ValueError(
                    "A2 attacks must use attacker_strength=full (GC10)"
                )


class AttackRegistry:
    """
    Registry for attack implementations.
    
    Manages attack instantiation and validation.
    """
    
    _attacks: Dict[AttackType, type] = {}
    
    @classmethod
    def register(cls, attack_type: AttackType):
        """Decorator to register an attack class."""
        def decorator(attack_cls: type):
            cls._attacks[attack_type] = attack_cls
            attack_cls.attack_type = attack_type
            return attack_cls
        return decorator
    
    @classmethod
    def get(cls, attack_type: AttackType) -> type:
        """Get attack class by type."""
        if attack_type not in cls._attacks:
            raise ValueError(f"Attack type {attack_type} not registered")
        return cls._attacks[attack_type]
    
    @classmethod
    def create(cls, attack_type: AttackType, **kwargs) -> AttackBase:
        """Create attack instance."""
        attack_cls = cls.get(attack_type)
        return attack_cls(**kwargs)
    
    @classmethod
    def list_attacks(cls) -> List[AttackType]:
        """List registered attack types."""
        return list(cls._attacks.keys())


def validate_a2_exists(attack_results: List[AttackResult]) -> bool:
    """
    Validate that A2 attack results exist.
    
    Per Property 14, attack_metrics.csv must contain threat_level=A2.
    
    Args:
        attack_results: List of attack results
        
    Returns:
        True if A2 exists
        
    Raises:
        ValueError: If A2 is missing (hard fail)
    """
    has_a2 = any(
        r.threat_level == ThreatLevel.A2 
        for r in attack_results
    )
    
    if not has_a2:
        raise ValueError(
            "A2 attack results missing (Property 14 violation). "
            "attack_metrics.csv must contain at least one threat_level=A2 record."
        )
    
    return True


def validate_attack_coverage(
    attack_results: List[AttackResult],
    required_types: List[AttackType] = None,
) -> Dict[str, Any]:
    """
    Validate attack coverage.
    
    Args:
        attack_results: List of attack results
        required_types: Required attack types (default: all 5)
        
    Returns:
        Coverage report
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
