"""
A2 Adaptive Attacker implementation.

White-box adaptive attacker per design.md §6.3.
Implements A2 strength contract per §5.4.

**Validates: Property 14 - A2 攻击强制存在**
**Validates: Requirements 16.1, 16.3, 16.5**
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..attack_framework import (
    AttackBase,
    AttackFitContext,
    AttackEvalContext,
    AttackResult,
    AttackType,
    ThreatLevel,
    AttackerStrength,
)


@dataclass
class A2StrengthContract:
    """
    A2 Strength Contract (frozen per §5.4).
    
    Ensures A2 attacks are auditable, reproducible, and not cherry-picked.
    """
    
    # Attack families (§5.4.1)
    attack_families: Dict[str, List[str]] = field(default_factory=lambda: {
        'reconstruction': ['unet_decoder', 'gan_inversion'],
        'inference': ['linear_probe', 'mlp_classifier', 'contrastive_learning'],
        'optimization': ['gradient_based', 'evolutionary_search'],
    })
    
    # Attack budget (§5.4.2)
    max_epochs: int = 100
    lr_search: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    max_gpu_hours_per_family: int = 24
    
    # Minimum instantiations per family
    min_instantiations: Dict[str, int] = field(default_factory=lambda: {
        'reconstruction': 2,
        'inference': 3,
        'optimization': 2,
    })
    
    def validate(self, attack_config: Dict) -> bool:
        """Validate attack configuration against contract."""
        family = attack_config.get('family')
        if family not in self.attack_families:
            return False
        
        epochs = attack_config.get('epochs', 0)
        if epochs > self.max_epochs:
            return False
        
        lr = attack_config.get('lr')
        if lr is not None and lr not in self.lr_search:
            return False
        
        return True


@dataclass
class AdaptiveStrategy:
    """Adaptive attack strategy."""
    mask_analysis: Dict[str, Any] = field(default_factory=dict)
    budget_analysis: Dict[str, Any] = field(default_factory=dict)
    attack_loss: Optional[str] = None
    strategy_description: str = ""


class AdaptiveAttacker(AttackBase):
    """
    A2 White-box Adaptive Attacker.
    
    Attacker capabilities (per §6.3):
    - Knows complete algorithm (encryption flow, mask generation, budget allocation)
    - Knows model structure (but not training weights)
    - Cannot access encryption key
    - Can design adaptive attack strategies
    
    This class wraps other attacks with A2-level knowledge.
    """
    
    attack_type = None  # Set dynamically based on wrapped attack
    
    def __init__(
        self,
        device: str = None,
        base_attack: Optional[AttackBase] = None,
        attack_type: AttackType = AttackType.ATTRIBUTE_INFERENCE,
    ):
        """
        Initialize adaptive attacker.
        
        Args:
            device: Compute device
            base_attack: Base attack to enhance with A2 knowledge
            attack_type: Type of attack to perform
        """
        super().__init__(device)
        self.base_attack = base_attack
        self._attack_type = attack_type
        self.contract = A2StrengthContract()
        self.adaptive_strategy: Optional[AdaptiveStrategy] = None
        self.algorithm_info: Dict[str, Any] = {}
    
    @property
    def attack_type(self):
        return self._attack_type
    
    @attack_type.setter
    def attack_type(self, value):
        self._attack_type = value
    
    def design_adaptive_strategy(
        self,
        algorithm_info: Dict,
        mask_generator=None,
        budget_allocator=None,
    ) -> AdaptiveStrategy:
        """
        Design adaptive attack strategy using algorithm knowledge.
        
        Per §6.3, the attacker can:
        1. Analyze mask generation logic to find semantic-preserving regions
        2. Analyze budget allocation rules to find low-protection regions
        3. Design targeted loss functions
        
        Args:
            algorithm_info: Information about the encryption algorithm
            mask_generator: Semantic mask generator (if available)
            budget_allocator: Budget allocator (if available)
            
        Returns:
            AdaptiveStrategy with attack plan
        """
        self.algorithm_info = algorithm_info
        
        strategy = AdaptiveStrategy()
        
        # Analyze mask patterns
        if mask_generator is not None:
            strategy.mask_analysis = self._analyze_mask_patterns(mask_generator)
        
        # Analyze budget patterns
        if budget_allocator is not None:
            strategy.budget_analysis = self._analyze_budget_patterns(budget_allocator)
        
        # Design adaptive loss
        strategy.attack_loss = self._design_adaptive_loss(algorithm_info)
        
        # Generate strategy description
        strategy.strategy_description = self._generate_strategy_description(strategy)
        
        self.adaptive_strategy = strategy
        return strategy
    
    def _analyze_mask_patterns(self, mask_generator) -> Dict[str, Any]:
        """Analyze mask generation patterns."""
        analysis = {
            'semantic_regions': [],
            'preservation_regions': [],
            'vulnerability_regions': [],
        }
        
        # Extract information about mask generation
        if hasattr(mask_generator, 'region_names'):
            analysis['semantic_regions'] = mask_generator.region_names
        
        if hasattr(mask_generator, 'preservation_weights'):
            # Find regions with high preservation (low protection)
            weights = mask_generator.preservation_weights
            for region, weight in weights.items():
                if weight > 0.5:
                    analysis['preservation_regions'].append(region)
                if weight < 0.3:
                    analysis['vulnerability_regions'].append(region)
        
        return analysis
    
    def _analyze_budget_patterns(self, budget_allocator) -> Dict[str, Any]:
        """Analyze budget allocation patterns."""
        analysis = {
            'low_budget_regions': [],
            'high_budget_regions': [],
            'budget_distribution': {},
        }
        
        if hasattr(budget_allocator, 'region_budgets'):
            budgets = budget_allocator.region_budgets
            analysis['budget_distribution'] = budgets
            
            for region, budget in budgets.items():
                if budget < 0.3:
                    analysis['low_budget_regions'].append(region)
                elif budget > 0.7:
                    analysis['high_budget_regions'].append(region)
        
        return analysis
    
    def _design_adaptive_loss(self, algorithm_info: Dict) -> str:
        """Design adaptive loss function based on algorithm knowledge."""
        loss_components = []
        
        # If we know about frequency domain processing
        if algorithm_info.get('uses_frequency_domain'):
            loss_components.append('frequency_reconstruction_loss')
        
        # If we know about chaotic scrambling
        if algorithm_info.get('uses_chaotic_scrambling'):
            loss_components.append('inverse_scrambling_loss')
        
        # If we know about semantic preservation
        if algorithm_info.get('preserves_semantics'):
            loss_components.append('semantic_extraction_loss')
        
        if not loss_components:
            loss_components.append('standard_reconstruction_loss')
        
        return ' + '.join(loss_components)
    
    def _generate_strategy_description(self, strategy: AdaptiveStrategy) -> str:
        """Generate human-readable strategy description."""
        desc = ["A2 Adaptive Attack Strategy:"]
        
        if strategy.mask_analysis.get('vulnerability_regions'):
            desc.append(f"- Target vulnerable regions: {strategy.mask_analysis['vulnerability_regions']}")
        
        if strategy.budget_analysis.get('low_budget_regions'):
            desc.append(f"- Exploit low-budget regions: {strategy.budget_analysis['low_budget_regions']}")
        
        if strategy.attack_loss:
            desc.append(f"- Custom loss: {strategy.attack_loss}")
        
        return '\n'.join(desc)
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        Train adaptive attack.
        
        Args:
            ctx: Attack training context (must be A2)
            **kwargs: Training data and algorithm info
        """
        # Validate A2 requirement
        if ctx.threat_level != ThreatLevel.A2:
            raise ValueError(
                f"AdaptiveAttacker requires threat_level=A2, got {ctx.threat_level}"
            )
        
        # A2 must use full strength (GC10)
        if ctx.attacker_strength != AttackerStrength.FULL:
            raise ValueError(
                "A2 attacks must use attacker_strength=full (GC10)"
            )
        
        self.fit_context = ctx
        
        # Design adaptive strategy if algorithm info provided
        algorithm_info = kwargs.get('algorithm_info', {})
        mask_generator = kwargs.get('mask_generator')
        budget_allocator = kwargs.get('budget_allocator')
        
        if algorithm_info or mask_generator or budget_allocator:
            self.design_adaptive_strategy(
                algorithm_info, mask_generator, budget_allocator
            )
        
        # Train base attack with adaptive enhancements
        if self.base_attack is not None:
            self.base_attack.fit(ctx, **kwargs)
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        Evaluate adaptive attack.
        
        Args:
            ctx: Evaluation context
            **kwargs: Evaluation data
            
        Returns:
            AttackResult with A2 threat level
        """
        if ctx.threat_level != ThreatLevel.A2:
            raise ValueError(
                f"AdaptiveAttacker requires threat_level=A2, got {ctx.threat_level}"
            )
        
        if self.base_attack is not None:
            result = self.base_attack.evaluate(ctx, **kwargs)
            # Override threat level to A2
            result.threat_level = ThreatLevel.A2
            return result
        
        # Default result if no base attack
        return AttackResult(
            attack_type=self._attack_type,
            threat_level=ThreatLevel.A2,
            attack_success=0.0,
            metric_name="attack_success",
            metric_value=0.0,
            status="failed",
            attacker_strength=AttackerStrength.FULL,
        )
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """Get report of adaptive strategy for audit."""
        return {
            'threat_level': 'A2',
            'attacker_strength': 'full',
            'algorithm_knowledge': self.algorithm_info,
            'adaptive_strategy': {
                'mask_analysis': self.adaptive_strategy.mask_analysis if self.adaptive_strategy else {},
                'budget_analysis': self.adaptive_strategy.budget_analysis if self.adaptive_strategy else {},
                'attack_loss': self.adaptive_strategy.attack_loss if self.adaptive_strategy else None,
                'description': self.adaptive_strategy.strategy_description if self.adaptive_strategy else '',
            },
            'contract': {
                'max_epochs': self.contract.max_epochs,
                'lr_search': self.contract.lr_search,
                'max_gpu_hours': self.contract.max_gpu_hours_per_family,
            }
        }


def compute_worst_case_attack_success(
    attack_results: List[AttackResult],
    group_by: List[str] = None,
) -> Dict[str, float]:
    """
    Compute worst-case attack success per §5.4.3.
    
    worst_case_attack_success = max(attack_success) over all attacks
    in same (dataset, task, privacy_level, threat_level)
    
    Args:
        attack_results: List of attack results
        group_by: Fields to group by (default: dataset, task, privacy_level, threat_level)
        
    Returns:
        Dictionary mapping group key to worst-case attack_success
    """
    if group_by is None:
        group_by = ['dataset', 'task', 'privacy_level', 'threat_level']
    
    # Group results
    groups: Dict[str, List[float]] = {}
    
    for result in attack_results:
        result_dict = result.to_dict()
        
        # Build group key
        key_parts = []
        for field in group_by:
            if field in result_dict:
                key_parts.append(f"{field}={result_dict[field]}")
        key = '|'.join(key_parts)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(result.attack_success)
    
    # Compute worst case for each group
    worst_case = {}
    for key, values in groups.items():
        worst_case[key] = max(values)
    
    return worst_case
