"""
因果分析模块 (Causal Analysis Module)
=====================================

本模块实现了基于因果推断的隐私-可用性分析器，用于量化隐私策略对下游ML任务的因果效应。

核心组件：
1. CausalPrivacyAnalyzer: 分析隐私预算分配的因果效应
2. CausalPerformanceAnalyzer: 分析加密对ML性能的因果影响

理论基础：
- 因果图 (Causal Graph): X (Semantic) -> Z (Privacy Budget) -> Y (ML Performance)
- ATE (Average Treatment Effect): E[Y(1) - Y(0)]
- CATE (Conditional ATE): E[Y(1) - Y(0) | X=x]

参考理论证明文档：docs/Causal-VSE-PC_理论证明.md
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings


class CausalPrivacyAnalyzer:
    """
    因果隐私分析器
    
    分析隐私预算分配策略的因果合理性，解释"为什么这样分配"。
    
    核心功能：
    1. 构建结构因果模型（SCM）
    2. 计算ATE/CATE
    3. 生成因果建议
    4. 生成可解释的因果报告
    """
    
    def __init__(self, use_history_baseline: bool = True):
        """
        初始化因果隐私分析器
        
        Args:
            use_history_baseline: 是否使用历史数据计算基线（默认True）
        """
        # 历史数据缓存，用于统计基线
        self.history = []
        self.use_history_baseline = use_history_baseline
        
        # 历史基线缓存 {task_type: {semantic_type: baseline_performance}}
        self.baseline_cache = defaultdict(lambda: defaultdict(list))
        
    def _build_causal_graph(
        self,
        semantic_mask: torch.Tensor,
        task_type: str
    ) -> Dict:
        """
        构建结构因果模型（SCM）
        
        节点：
        - X: 语义区域类型（敏感/任务相关/背景）
        - T: 任务类型（分类/分割/检测）
        - Z: 隐私预算/加密强度
        - Y: ML任务性能（准确率/mIoU/mAP）
        
        边：
        - X -> Z: 语义决定隐私预算
        - T -> Z: 任务类型影响隐私预算
        - Z -> Y: 隐私预算影响ML性能
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            task_type: 任务类型
            
        Returns:
            causal_graph: 因果图结构字典
        """
        B = semantic_mask.shape[0]
        
        # 识别语义区域类型
        sensitive_regions = (semantic_mask > 0.7).float()  # 敏感区域
        task_regions = ((semantic_mask > 0.3) & (semantic_mask <= 0.7)).float()  # 任务相关区域
        background_regions = (semantic_mask <= 0.3).float()  # 背景区域
        
        # 计算各区域占比
        sensitive_ratio = sensitive_regions.mean(dim=[-2, -1]).mean().item()  # [B] -> scalar
        task_ratio = task_regions.mean(dim=[-2, -1]).mean().item()
        background_ratio = background_regions.mean(dim=[-2, -1]).mean().item()
        
        return {
            'nodes': {
                'X': {
                    'sensitive_ratio': sensitive_ratio,
                    'task_ratio': task_ratio,
                    'background_ratio': background_ratio
                },
                'T': task_type,
                'Z': None,  # 将在后续填充
                'Y': None   # 将在后续填充
            },
            'edges': [
                ('X', 'Z'),  # X -> Z
                ('T', 'Z'),  # T -> Z
                ('Z', 'Y')   # Z -> Y
            ],
            'semantic_mask': semantic_mask,
            'task_type': task_type
        }
    
    def _compute_baseline(
        self,
        task_type: str,
        semantic_type: str = 'default'
    ) -> float:
        """
        计算历史基线性能
        
        如果没有历史数据，返回默认基线
        
        Args:
            task_type: 任务类型
            semantic_type: 语义类型（'sensitive', 'task', 'background'）
            
        Returns:
            baseline: 基线性能值
        """
        if self.use_history_baseline and semantic_type in self.baseline_cache[task_type]:
            history = self.baseline_cache[task_type][semantic_type]
            if len(history) > 0:
                return np.mean(history)
        
        # 默认基线
        default_baselines = {
            'classification': 0.95,
            'segmentation': 0.85,
            'detection': 0.80
        }
        return default_baselines.get(task_type, 0.90)
        
    def analyze_allocation(
        self,
        semantic_mask: torch.Tensor,
        task_type: str,
        privacy_map: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        分析隐私预算分配的因果合理性，生成因果建议
        
        这是核心方法，用于在加密前生成因果驱动的隐私预算分配建议。
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            task_type: 任务类型 {'classification', 'segmentation', 'detection'}
            privacy_map: [B, 1, H, W] 隐私预算图（可选，如果提供则分析现有分配）
            
        Returns:
            causal_suggestion: 因果建议字典
                - causal_graph: 因果图结构
                - suggestion: 隐私预算分配建议
                - explanation: 自然语言解释
        """
        B = semantic_mask.shape[0]
        device = semantic_mask.device
        
        # 1. 构建因果图
        causal_graph = self._build_causal_graph(semantic_mask, task_type)
        
        # 2. 计算历史基线
        baseline_sensitive = self._compute_baseline(task_type, 'sensitive')
        baseline_task = self._compute_baseline(task_type, 'task')
        baseline_background = self._compute_baseline(task_type, 'background')
        
        # 3. 生成因果建议
        suggestion = self._generate_suggestion(
            causal_graph, 
            baseline_sensitive, 
            baseline_task, 
            baseline_background
        )
        
        # 4. 如果有现有隐私预算图，分析其因果合理性
        allocation_analysis = None
        if privacy_map is not None:
            allocation_analysis = self._analyze_existing_allocation(
                semantic_mask, privacy_map, task_type
            )
        
        # 5. 生成自然语言解释
        explanation = self._generate_causal_explanation(
            causal_graph, suggestion, allocation_analysis
        )
        
        return {
            'causal_graph': causal_graph,
            'suggestion': suggestion,
            'explanation': explanation,
            'allocation_analysis': allocation_analysis
        }
    
    def _generate_suggestion(
        self,
        causal_graph: Dict,
        baseline_sensitive: float,
        baseline_task: float,
        baseline_background: float
    ) -> Dict:
        """
        基于因果图和历史基线生成隐私预算分配建议
        
        Args:
            causal_graph: 因果图结构
            baseline_*: 各区域的基线性能
            
        Returns:
            suggestion: 隐私预算分配建议
        """
        task_type = causal_graph['task_type']
        sensitive_ratio = causal_graph['nodes']['X']['sensitive_ratio']
        
        # 基于任务类型和语义占比生成建议
        # 这是一个简化的因果模型，实际中可以使用更复杂的模型
        
        if task_type == 'classification':
            # 分类任务：保护身份，保留物体特征
            suggested_privacy = {
                'sensitive': 0.9,  # 强加密
                'task': 0.3,       # 弱加密
                'background': 0.0  # 不加密
            }
        elif task_type == 'segmentation':
            # 分割任务：保护敏感区域，保留场景结构
            suggested_privacy = {
                'sensitive': 1.0,  # 最强加密
                'task': 0.1,       # 极弱加密
                'background': 0.0  # 不加密
            }
        elif task_type == 'detection':
            # 检测任务：保护身份，保留物体检测能力
            suggested_privacy = {
                'sensitive': 0.8,  # 强加密
                'task': 0.2,       # 弱加密
                'background': 0.0  # 不加密
            }
        else:
            # 默认策略
            suggested_privacy = {
                'sensitive': 0.7,
                'task': 0.3,
                'background': 0.0
            }
        
        # 根据敏感区域占比调整建议
        if sensitive_ratio > 0.5:
            # 如果敏感区域占比大，整体提高隐私强度
            suggested_privacy['sensitive'] = min(1.0, suggested_privacy['sensitive'] + 0.1)
        elif sensitive_ratio < 0.1:
            # 如果敏感区域占比小，可以适当降低
            suggested_privacy['sensitive'] = max(0.5, suggested_privacy['sensitive'] - 0.1)
        
        return {
            'privacy_budget': suggested_privacy,
            'rationale': f"基于{task_type}任务的因果分析，建议对敏感/任务/背景区域分别使用{suggested_privacy['sensitive']:.1f}/{suggested_privacy['task']:.1f}/{suggested_privacy['background']:.1f}的隐私预算。"
        }
    
    def _analyze_existing_allocation(
        self,
        semantic_mask: torch.Tensor,
        privacy_map: torch.Tensor,
        task_type: str
    ) -> Dict:
        """
        分析现有隐私预算分配的因果合理性
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            privacy_map: [B, 1, H, W] 隐私预算图
            task_type: 任务类型
            
        Returns:
            analysis: 分配分析结果
        """
        B = semantic_mask.shape[0]
        
        sensitive_regions = (semantic_mask > 0.7).float()
        task_regions = ((semantic_mask > 0.3) & (semantic_mask <= 0.7)).float()
        background_regions = (semantic_mask <= 0.3).float()
        
        # 计算各区域的平均隐私强度
        avg_privacy_sensitive = (privacy_map * sensitive_regions).sum(dim=[-2, -1]) / (sensitive_regions.sum(dim=[-2, -1]) + 1e-6)
        avg_privacy_task = (privacy_map * task_regions).sum(dim=[-2, -1]) / (task_regions.sum(dim=[-2, -1]) + 1e-6)
        avg_privacy_background = (privacy_map * background_regions).sum(dim=[-2, -1]) / (background_regions.sum(dim=[-2, -1]) + 1e-6)
        
        return {
            'avg_privacy_sensitive': avg_privacy_sensitive.mean().item(),
            'avg_privacy_task': avg_privacy_task.mean().item(),
            'avg_privacy_background': avg_privacy_background.mean().item(),
            'consistency': self._check_allocation_consistency(
                avg_privacy_sensitive.mean().item(),
                avg_privacy_task.mean().item(),
                avg_privacy_background.mean().item(),
                task_type
            )
        }
    
    def _check_allocation_consistency(
        self,
        privacy_sensitive: float,
        privacy_task: float,
        privacy_background: float,
        task_type: str
    ) -> bool:
        """
        检查隐私预算分配是否符合因果推理的逻辑一致性
        
        期望：privacy_sensitive >= privacy_task >= privacy_background
        """
        return (privacy_sensitive >= privacy_task >= privacy_background) or \
               (abs(privacy_sensitive - privacy_task) < 0.1 and privacy_task >= privacy_background)
    
    def _generate_causal_explanation(
        self,
        causal_graph: Dict,
        suggestion: Dict,
        allocation_analysis: Optional[Dict]
    ) -> str:
        """
        生成自然语言因果解释
        
        Args:
            causal_graph: 因果图结构
            suggestion: 因果建议
            allocation_analysis: 现有分配分析（可选）
            
        Returns:
            explanation: 自然语言解释
        """
        task_type = causal_graph['task_type']
        sensitive_ratio = causal_graph['nodes']['X']['sensitive_ratio']
        privacy_budget = suggestion['privacy_budget']
        
        explanation = f"基于结构因果模型(SCM)的因果分析：\n"
        explanation += f"1. 语义分析：检测到敏感区域占比{sensitive_ratio:.1%}。\n"
        explanation += f"2. 任务类型：{task_type}任务的因果路径显示，隐私预算对性能的因果效应取决于区域类型。\n"
        explanation += f"3. 因果建议：对敏感区域建议隐私预算{privacy_budget['sensitive']:.1f}，"
        explanation += f"对任务相关区域建议{privacy_budget['task']:.1f}，对背景区域建议{privacy_budget['background']:.1f}。\n"
        
        if allocation_analysis:
            consistency = allocation_analysis['consistency']
            if consistency:
                explanation += f"4. 现有分配检查：当前隐私预算分配符合因果推理的逻辑一致性。\n"
            else:
                explanation += f"4. 现有分配警告：当前隐私预算分配存在逻辑不一致，建议调整。\n"
        
        return explanation
    
    def compute_ate(
        self,
        performance_high: torch.Tensor,
        performance_low: torch.Tensor,
        conf_interval: bool = True
    ) -> Dict:
        """
        计算平均处理效应（ATE: Average Treatment Effect）
        
        ATE = E[Y(high_privacy) - Y(low_privacy)]
        
        其中：
        - Y(high_privacy): 高隐私加密下的ML性能
        - Y(low_privacy): 低隐私加密下的ML性能
        
        Args:
            performance_high: [N] 高隐私加密下的性能指标
            performance_low: [N] 低隐私加密下的性能指标
            conf_interval: 是否计算置信区间
            
        Returns:
            ate_result: ATE计算结果
                - ate: ATE值
                - std: 标准差
                - ci_lower: 置信区间下界（95%）
                - ci_upper: 置信区间上界（95%）
        """
        # 确保是numpy数组
        if isinstance(performance_high, torch.Tensor):
            perf_high = performance_high.cpu().numpy()
        else:
            perf_high = np.array(performance_high)
            
        if isinstance(performance_low, torch.Tensor):
            perf_low = performance_low.cpu().numpy()
        else:
            perf_low = np.array(performance_low)
        
        # 计算个体处理效应
        individual_effects = perf_high - perf_low
        
        # 计算ATE
        ate = np.mean(individual_effects)
        std = np.std(individual_effects, ddof=1)
        n = len(individual_effects)
        
        result = {
            'ate': float(ate),
            'std': float(std),
            'n': n
        }
        
        # 计算95%置信区间（使用t分布）
        if conf_interval and n > 1:
            from scipy import stats
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95%置信区间
            se = std / np.sqrt(n)
            margin = t_critical * se
            
            result['ci_lower'] = float(ate - margin)
            result['ci_upper'] = float(ate + margin)
            result['ci_level'] = 0.95
        
        return result
    
    def compute_cate(
        self,
        semantic_mask: torch.Tensor,
        performance_high: torch.Tensor,
        performance_low: torch.Tensor,
        region_type: str = 'sensitive',
        conf_interval: bool = True
    ) -> Dict:
        """
        计算条件平均处理效应（CATE: Conditional Average Treatment Effect）
        
        CATE(X=x) = E[Y(high_privacy) - Y(low_privacy) | X=x]
        
        其中X=x表示特定的语义区域类型（如敏感区域）
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            performance_high: [B] 高隐私加密下的性能指标
            performance_low: [B] 低隐私加密下的性能指标
            region_type: 区域类型 {'sensitive', 'task', 'background'}
            conf_interval: 是否计算置信区间
            
        Returns:
            cate_result: CATE计算结果
        """
        B = semantic_mask.shape[0]
        
        # 识别指定区域
        if region_type == 'sensitive':
            region_mask = (semantic_mask > 0.7).float()
        elif region_type == 'task':
            region_mask = ((semantic_mask > 0.3) & (semantic_mask <= 0.7)).float()
        elif region_type == 'background':
            region_mask = (semantic_mask <= 0.3).float()
        else:
            raise ValueError(f"Unknown region_type: {region_type}")
        
        # 计算每个样本中该区域的占比
        region_ratios = region_mask.mean(dim=[-2, -1])  # [B, 1] -> [B]
        
        # 只考虑该区域占比足够大的样本（阈值>0.1）
        valid_indices = (region_ratios > 0.1).squeeze(-1)  # [B]
        
        if valid_indices.sum() == 0:
            warnings.warn(f"No valid samples found for region_type={region_type}")
            return {
                'cate': np.nan,
                'std': np.nan,
                'n': 0,
                'region_type': region_type
            }
        
        # 提取有效样本的性能
        perf_high_valid = performance_high[valid_indices]
        perf_low_valid = performance_low[valid_indices]
        
        # 计算CATE
        individual_effects = perf_high_valid - perf_low_valid
        
        if isinstance(individual_effects, torch.Tensor):
            cate = individual_effects.mean().item()
            std = individual_effects.std(unbiased=True).item()
            individual_effects = individual_effects.cpu().numpy()
        else:
            cate = np.mean(individual_effects)
            std = np.std(individual_effects, ddof=1)
        
        n = len(individual_effects)
        
        result = {
            'cate': float(cate),
            'std': float(std),
            'n': n,
            'region_type': region_type
        }
        
        # 计算置信区间
        if conf_interval and n > 1:
            try:
                from scipy import stats
                t_critical = stats.t.ppf(0.975, df=n-1)
                se = std / np.sqrt(n)
                margin = t_critical * se
                
                result['ci_lower'] = float(cate - margin)
                result['ci_upper'] = float(cate + margin)
                result['ci_level'] = 0.95
            except ImportError:
                warnings.warn("scipy not available, skipping confidence interval")
        
        return result
    
    def compute_causal_effects(
        self,
        semantic_mask: torch.Tensor,
        privacy_map: torch.Tensor,
        performance_encrypted: torch.Tensor,
        performance_original: Optional[torch.Tensor],
        task_type: str,
        conf_interval: bool = True
    ) -> Dict:
        """
        计算因果效应（后处理分析）
        
        这是核心的后处理方法，用于在加密和ML推理后计算因果效应。
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            privacy_map: [B, 1, H, W] 隐私预算图
            performance_encrypted: [B] 加密图像上的ML性能
            performance_original: [B] 原始图像上的ML性能（可选，如果没有则使用历史基线）
            task_type: 任务类型
            conf_interval: 是否计算置信区间
            
        Returns:
            causal_report: 因果分析报告
        """
        B = semantic_mask.shape[0]
        device = semantic_mask.device
        
        # 1. 如果没有原始性能，使用历史基线
        if performance_original is None:
            baseline = self._compute_baseline(task_type)
            performance_original = torch.full((B,), baseline, device=device)
        
        # 2. 划分高隐私和低隐私组
        # 定义阈值：privacy > 0.7 为高隐私，privacy < 0.3 为低隐私
        avg_privacy = privacy_map.mean(dim=[-2, -1]).squeeze(-1)  # [B]
        
        high_privacy_mask = (avg_privacy > 0.7)  # [B]
        low_privacy_mask = (avg_privacy < 0.3)   # [B]
        
        # 3. 计算ATE
        # 使用分层方式：分别计算高隐私和低隐私组的效果
        if high_privacy_mask.sum() > 0 and low_privacy_mask.sum() > 0:
            perf_high_group = performance_encrypted[high_privacy_mask]
            perf_low_group = performance_encrypted[low_privacy_mask]
            perf_orig_high = performance_original[high_privacy_mask]
            perf_orig_low = performance_original[low_privacy_mask]
            
            # 计算相对于原始性能的效应
            effect_high = perf_high_group - perf_orig_high
            effect_low = perf_low_group - perf_orig_low
            
            # ATE = E[Y(high) - Y(low)]
            # 这里简化为: ATE = mean(effect_high) - mean(effect_low)
            ate_result = self.compute_ate(effect_high, effect_low, conf_interval)
        else:
            # 如果没有足够的高/低隐私样本，使用整体效应
            overall_effect = performance_encrypted - performance_original
            ate_result = {
                'ate': float(overall_effect.mean().item()),
                'std': float(overall_effect.std().item()),
                'n': B
            }
            if conf_interval:
                try:
                    from scipy import stats
                    t_critical = stats.t.ppf(0.975, df=B-1)
                    se = overall_effect.std().item() / np.sqrt(B)
                    margin = t_critical * se
                    ate_result['ci_lower'] = float(overall_effect.mean().item() - margin)
                    ate_result['ci_upper'] = float(overall_effect.mean().item() + margin)
                    ate_result['ci_level'] = 0.95
                except ImportError:
                    pass
        
        # 4. 计算CATE（不同区域的因果效应）
        cate_results = {}
        for region_type in ['sensitive', 'task', 'background']:
            cate_result = self.compute_cate(
                semantic_mask,
                performance_encrypted,
                performance_original,
                region_type=region_type,
                conf_interval=conf_interval
            )
            cate_results[region_type] = cate_result
        
        # 5. 生成报告
        report = {
            'ate': ate_result,
            'cate': cate_results,
            'summary': self._generate_causal_summary(ate_result, cate_results, task_type)
        }
        
        return report
    
    def _generate_causal_summary(
        self, 
        ate_result: Dict,
        cate_results: Dict,
        task_type: str
    ) -> str:
        """
        生成因果分析摘要（自然语言）
        """
        ate = ate_result.get('ate', np.nan)
        cate_sensitive = cate_results.get('sensitive', {}).get('cate', np.nan)
        cate_task = cate_results.get('task', {}).get('cate', np.nan)
        
        summary = f"因果效应分析报告 ({task_type}任务):\n"
        summary += f"1. 平均处理效应(ATE): {ate:.4f}"
        if 'ci_lower' in ate_result:
            summary += f" (95% CI: [{ate_result['ci_lower']:.4f}, {ate_result['ci_upper']:.4f}])"
        summary += "\n"
        
        summary += f"2. 条件处理效应(CATE):\n"
        summary += f"   - 敏感区域: {cate_sensitive:.4f}"
        if 'ci_lower' in cate_results.get('sensitive', {}):
            ci = cate_results['sensitive']
            summary += f" (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])"
        summary += "\n"
        
        summary += f"   - 任务区域: {cate_task:.4f}"
        if 'ci_lower' in cate_results.get('task', {}):
            ci = cate_results['task']
            summary += f" (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])"
        summary += "\n"
        
        # 解释
        if task_type == 'classification':
            if ate > -0.20:
                summary += "3. 解释：加密对分类准确率的因果效应较小，语义保留策略有效。\n"
            else:
                summary += "3. 解释：观测到显著的性能下降，建议优化隐私预算分配。\n"
        elif task_type == 'segmentation':
            if ate > -0.30:
                summary += "3. 解释：加密对分割mIoU的因果效应在可接受范围内。\n"
        else:
                summary += "3. 解释：观测到显著的性能下降，建议降低任务区域的加密强度。\n"
        
        return summary


class CausalPerformanceAnalyzer:
    """
    因果性能分析器
    
    量化加密操作对ML模型性能的因果效应 (ATE)。
    需要配合ML模型的推理结果使用。
    """
    
    def __init__(self):
        self.baseline_metrics = {
            'classification': {'acc': 0.95},
            'segmentation': {'miou': 0.85},
            'detection': {'map': 0.80}
        }
        
    def estimate_causal_effect(
        self,
        task_type: str,
        current_metric: float,
        privacy_map: torch.Tensor
    ) -> Dict:
        """
        估计加密对性能的因果效应 (ATE)
        
        ATE = E[Y_encrypted - Y_original]
        
        Args:
            task_type: 任务类型
            current_metric: 当前加密图像上的模型指标（如置信度、IoU估计）
            privacy_map: 使用的隐私预算图
            
        Returns:
            effect_analysis: 因果效应分析
        """
        baseline = self.baseline_metrics.get(task_type, {}).get('acc', 0.90)
        
        # 1. 计算处理效应 (Treatment Effect)
        # 这里Y(1)是current_metric, Y(0)是baseline (反事实估计)
        treatment_effect = current_metric - baseline
        
        # 2. 分析异质性 (Heterogeneity)
        # 越强的加密通常导致越大的性能下降
        avg_encryption = privacy_map.mean().item()
        
        # 3. 归因分析
        # 估算性能下降中有多少是因为加密强度造成的
        # 简单的线性因果模型假设: Performance_Drop = beta * Encryption_Strength + epsilon
        expected_drop = -0.15 * avg_encryption  # 假设系数 beta = -0.15
        
        unexpected_drop = treatment_effect - expected_drop
        
        return {
            'ate': treatment_effect,
            'expected_causal_effect': expected_drop,
            'residual_effect': unexpected_drop,
            'interpretation': self._interpret_effect(treatment_effect, avg_encryption)
        }
        
    def _interpret_effect(self, effect: float, strength: float) -> str:
        if effect > -0.05:
            return f"因果效应微弱 ({effect:.2%})。尽管平均加密强度为 {strength:.2f}，语义保留策略成功切断了对ML特征的负面因果路径。"
        else:
            return f"观测到显著的负面因果效应 ({effect:.2%})。高强度加密 ({strength:.2f}) 导致了特征空间的显著偏移，建议在非敏感区域降低干预强度。"
