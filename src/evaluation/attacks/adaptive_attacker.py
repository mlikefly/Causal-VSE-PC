"""
A2 自适应攻击器实现。

按 design.md §6.3 实现白盒自适应攻击器。
按 §5.4 实现 A2 强度契约。

**验证: 属性 14 - A2 攻击强制存在**
**需求: 16.1, 16.3, 16.5**
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
    A2 强度契约（按 §5.4 冻结）。
    
    确保 A2 攻击可审计、可复现且非挑选性。
    """
    
    # 攻击族（§5.4.1）
    attack_families: Dict[str, List[str]] = field(default_factory=lambda: {
        'reconstruction': ['unet_decoder', 'gan_inversion'],
        'inference': ['linear_probe', 'mlp_classifier', 'contrastive_learning'],
        'optimization': ['gradient_based', 'evolutionary_search'],
    })
    
    # 攻击预算（§5.4.2）
    max_epochs: int = 100
    lr_search: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    max_gpu_hours_per_family: int = 24
    
    # 每个攻击族的最小实例化数量
    min_instantiations: Dict[str, int] = field(default_factory=lambda: {
        'reconstruction': 2,
        'inference': 3,
        'optimization': 2,
    })
    
    def validate(self, attack_config: Dict) -> bool:
        """验证攻击配置是否符合契约。"""
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
    """自适应攻击策略。"""
    mask_analysis: Dict[str, Any] = field(default_factory=dict)
    budget_analysis: Dict[str, Any] = field(default_factory=dict)
    attack_loss: Optional[str] = None
    strategy_description: str = ""


class AdaptiveAttacker(AttackBase):
    """
    A2 白盒自适应攻击器。
    
    攻击者能力（按 §6.3）：
    - 知道完整算法（加密流程、掩码生成、预算分配）
    - 知道模型结构（但不知道训练权重）
    - 无法访问加密密钥
    - 可以设计自适应攻击策略
    
    此类用 A2 级别知识包装其他攻击。
    """
    
    attack_type = None  # 根据包装的攻击动态设置
    
    def __init__(
        self,
        device: str = None,
        base_attack: Optional[AttackBase] = None,
        attack_type: AttackType = AttackType.ATTRIBUTE_INFERENCE,
    ):
        """
        初始化自适应攻击器。
        
        参数:
            device: 计算设备
            base_attack: 要用 A2 知识增强的基础攻击
            attack_type: 要执行的攻击类型
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
        使用算法知识设计自适应攻击策略。
        
        按 §6.3，攻击者可以：
        1. 分析掩码生成逻辑以找到语义保留区域
        2. 分析预算分配规则以找到低保护区域
        3. 设计针对性损失函数
        
        参数:
            algorithm_info: 关于加密算法的信息
            mask_generator: 语义掩码生成器（如果可用）
            budget_allocator: 预算分配器（如果可用）
            
        返回:
            包含攻击计划的 AdaptiveStrategy
        """
        self.algorithm_info = algorithm_info
        
        strategy = AdaptiveStrategy()
        
        # 分析掩码模式
        if mask_generator is not None:
            strategy.mask_analysis = self._analyze_mask_patterns(mask_generator)
        
        # 分析预算模式
        if budget_allocator is not None:
            strategy.budget_analysis = self._analyze_budget_patterns(budget_allocator)
        
        # 设计自适应损失
        strategy.attack_loss = self._design_adaptive_loss(algorithm_info)
        
        # 生成策略描述
        strategy.strategy_description = self._generate_strategy_description(strategy)
        
        self.adaptive_strategy = strategy
        return strategy
    
    def _analyze_mask_patterns(self, mask_generator) -> Dict[str, Any]:
        """分析掩码生成模式。"""
        analysis = {
            'semantic_regions': [],
            'preservation_regions': [],
            'vulnerability_regions': [],
        }
        
        # 提取掩码生成信息
        if hasattr(mask_generator, 'region_names'):
            analysis['semantic_regions'] = mask_generator.region_names
        
        if hasattr(mask_generator, 'preservation_weights'):
            # 找到高保留（低保护）的区域
            weights = mask_generator.preservation_weights
            for region, weight in weights.items():
                if weight > 0.5:
                    analysis['preservation_regions'].append(region)
                if weight < 0.3:
                    analysis['vulnerability_regions'].append(region)
        
        return analysis
    
    def _analyze_budget_patterns(self, budget_allocator) -> Dict[str, Any]:
        """分析预算分配模式。"""
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
        """基于算法知识设计自适应损失函数。"""
        loss_components = []
        
        # 如果知道使用频域处理
        if algorithm_info.get('uses_frequency_domain'):
            loss_components.append('frequency_reconstruction_loss')
        
        # 如果知道使用混沌置乱
        if algorithm_info.get('uses_chaotic_scrambling'):
            loss_components.append('inverse_scrambling_loss')
        
        # 如果知道保留语义
        if algorithm_info.get('preserves_semantics'):
            loss_components.append('semantic_extraction_loss')
        
        if not loss_components:
            loss_components.append('standard_reconstruction_loss')
        
        return ' + '.join(loss_components)
    
    def _generate_strategy_description(self, strategy: AdaptiveStrategy) -> str:
        """生成人类可读的策略描述。"""
        desc = ["A2 自适应攻击策略:"]
        
        if strategy.mask_analysis.get('vulnerability_regions'):
            desc.append(f"- 目标脆弱区域: {strategy.mask_analysis['vulnerability_regions']}")
        
        if strategy.budget_analysis.get('low_budget_regions'):
            desc.append(f"- 利用低预算区域: {strategy.budget_analysis['low_budget_regions']}")
        
        if strategy.attack_loss:
            desc.append(f"- 自定义损失: {strategy.attack_loss}")
        
        return '\n'.join(desc)
    
    def fit(self, ctx: AttackFitContext, **kwargs) -> None:
        """
        训练自适应攻击。
        
        参数:
            ctx: 攻击训练上下文（必须是 A2）
            **kwargs: 训练数据和算法信息
        """
        # 验证 A2 要求
        if ctx.threat_level != ThreatLevel.A2:
            raise ValueError(
                f"AdaptiveAttacker 需要 threat_level=A2，得到 {ctx.threat_level}"
            )
        
        # A2 必须使用完整强度（GC10）
        if ctx.attacker_strength != AttackerStrength.FULL:
            raise ValueError(
                "A2 攻击必须使用 attacker_strength=full（GC10）"
            )
        
        self.fit_context = ctx
        
        # 如果提供了算法信息，设计自适应策略
        algorithm_info = kwargs.get('algorithm_info', {})
        mask_generator = kwargs.get('mask_generator')
        budget_allocator = kwargs.get('budget_allocator')
        
        if algorithm_info or mask_generator or budget_allocator:
            self.design_adaptive_strategy(
                algorithm_info, mask_generator, budget_allocator
            )
        
        # 使用自适应增强训练基础攻击
        if self.base_attack is not None:
            self.base_attack.fit(ctx, **kwargs)
        
        self.is_fitted = True
    
    def evaluate(self, ctx: AttackEvalContext, **kwargs) -> AttackResult:
        """
        评估自适应攻击。
        
        参数:
            ctx: 评估上下文
            **kwargs: 评估数据
            
        返回:
            具有 A2 威胁级别的 AttackResult
        """
        if ctx.threat_level != ThreatLevel.A2:
            raise ValueError(
                f"AdaptiveAttacker 需要 threat_level=A2，得到 {ctx.threat_level}"
            )
        
        if self.base_attack is not None:
            result = self.base_attack.evaluate(ctx, **kwargs)
            # 覆盖威胁级别为 A2
            result.threat_level = ThreatLevel.A2
            return result
        
        # 如果没有基础攻击，返回默认结果
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
        """获取自适应策略报告用于审计。"""
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
    按 §5.4.3 计算最坏情况攻击成功率。
    
    worst_case_attack_success = max(attack_success)，在相同的
    (dataset, task, privacy_level, threat_level) 组合中取最大值
    
    参数:
        attack_results: 攻击结果列表
        group_by: 分组字段（默认: dataset, task, privacy_level, threat_level）
        
    返回:
        分组键到最坏情况 attack_success 的字典映射
    """
    if group_by is None:
        group_by = ['dataset', 'task', 'privacy_level', 'threat_level']
    
    # 分组结果
    groups: Dict[str, List[float]] = {}
    
    for result in attack_results:
        result_dict = result.to_dict()
        
        # 构建分组键
        key_parts = []
        for field in group_by:
            if field in result_dict:
                key_parts.append(f"{field}={result_dict[field]}")
        key = '|'.join(key_parts)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(result.attack_success)
    
    # 计算每个分组的最坏情况
    worst_case = {}
    for key, values in groups.items():
        worst_case[key] = max(values)
    
    return worst_case
