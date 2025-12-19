# -*- coding: utf-8 -*-
"""
因果效应估计与预算优化模块

实现 T7: 因果两阶段方法
- 阶段 I: 干预网格 (InterventionGrid)
- 阶段 II: ATE/CATE 估计 → 预算优化求解

Requirements: §9.2, §2.5, C1
Validates: Property - ATE/CATE + CI 完整性

Inputs/Outputs Contract:
- 输入: regions, beta_values, n_samples_per_cell, utility_threshold
- 输出: tables/causal_effects.csv, tables/optimal_allocation.json, figures/fig_causal_ate_cate.png
- 约束: ATE/CATE 必须含 CI；优化目标 min max_a attack_success s.t. utility >= threshold
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import csv

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy import stats
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class InterventionGrid:
    """
    干预网格配置（冻结）
    
    对每个语义区域 r_i 系统干预 β_i，观测 attack_success 与 utility
    
    Requirements: §9.2
    """
    regions: List[str] = field(default_factory=lambda: ["face", "background", "sensitive_attr"])
    beta_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    n_samples_per_cell: int = 100
    
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """
        生成所有干预实验配置
        
        Returns:
            experiments: 干预实验配置列表
        """
        experiments = []
        exp_id = 0
        
        # 单区域干预实验
        for region in self.regions:
            for beta in self.beta_values:
                experiments.append({
                    "experiment_id": f"single_{exp_id:04d}",
                    "intervention_type": "single",
                    "intervention": {region: beta},
                    "n_samples": self.n_samples_per_cell,
                    "region": region,
                    "beta": beta
                })
                exp_id += 1
        
        # 全区域联合干预实验（用于交互效应）
        for beta in self.beta_values:
            intervention = {region: beta for region in self.regions}
            experiments.append({
                "experiment_id": f"joint_{exp_id:04d}",
                "intervention_type": "joint",
                "intervention": intervention,
                "n_samples": self.n_samples_per_cell,
                "region": "all",
                "beta": beta
            })
            exp_id += 1
        
        return experiments
    
    def get_grid_size(self) -> int:
        """获取网格总大小"""
        single_experiments = len(self.regions) * len(self.beta_values)
        joint_experiments = len(self.beta_values)
        return single_experiments + joint_experiments
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class InterventionResult:
    """单次干预实验结果"""
    experiment_id: str
    region: str
    beta: float
    attack_success: float
    utility: float
    attack_success_std: float = 0.0
    utility_std: float = 0.0
    n_samples: int = 0
    attack_type: str = "aggregate"
    threat_level: str = "A2"


@dataclass
class CausalEffectRow:
    """
    causal_effects.csv 行数据
    
    Requirements: §9.2
    """
    region: str
    effect_type: str  # ATE 或 CATE
    ate: float
    ate_std: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_samples: int
    stat_method: str
    n_boot: int
    family_id: str
    condition: Optional[str] = None  # 对于 CATE：条件描述
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalEffectEstimator:
    """
    因果效应估计器
    
    实现 ATE/CATE 估计，支持 bootstrap CI
    
    Requirements: §2.5, §9.2
    """
    
    def __init__(self, n_boot: int = 500, ci_level: float = 0.95, seed: int = 42):
        """
        初始化因果效应估计器
        
        Args:
            n_boot: Bootstrap 重采样次数（≥500）
            ci_level: 置信区间水平
            seed: 随机种子
        """
        self.n_boot = max(500, n_boot)  # 强制 ≥ 500
        self.ci_level = ci_level
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def estimate_ate(
        self,
        outcomes_treatment: np.ndarray,
        outcomes_control: np.ndarray
    ) -> Dict[str, Any]:
        """
        估计平均处理效应 (ATE)
        
        ATE = E[Y|do(β=1)] - E[Y|do(β=0)]
        
        Args:
            outcomes_treatment: 处理组结果 (β=1 或高 β)
            outcomes_control: 对照组结果 (β=0 或低 β)
        
        Returns:
            ate_result: 包含 ATE、CI、统计信息
        """
        outcomes_treatment = np.asarray(outcomes_treatment).flatten()
        outcomes_control = np.asarray(outcomes_control).flatten()
        
        # 点估计
        ate = np.mean(outcomes_treatment) - np.mean(outcomes_control)
        
        # Bootstrap CI
        boot_ates = []
        n_t, n_c = len(outcomes_treatment), len(outcomes_control)
        
        for _ in range(self.n_boot):
            idx_t = self.rng.choice(n_t, size=n_t, replace=True)
            idx_c = self.rng.choice(n_c, size=n_c, replace=True)
            boot_ate = np.mean(outcomes_treatment[idx_t]) - np.mean(outcomes_control[idx_c])
            boot_ates.append(boot_ate)
        
        boot_ates = np.array(boot_ates)
        ate_std = np.std(boot_ates, ddof=1)
        
        # 百分位法 CI
        alpha = 1 - self.ci_level
        ci_low = np.percentile(boot_ates, 100 * alpha / 2)
        ci_high = np.percentile(boot_ates, 100 * (1 - alpha / 2))
        
        return {
            "ate": float(ate),
            "ate_std": float(ate_std),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_level": self.ci_level,
            "n_treatment": n_t,
            "n_control": n_c,
            "n_boot": self.n_boot,
            "stat_method": "bootstrap_percentile"
        }
    
    def estimate_cate(
        self,
        outcomes_treatment: np.ndarray,
        outcomes_control: np.ndarray,
        condition_mask_treatment: np.ndarray,
        condition_mask_control: np.ndarray,
        condition_name: str = "condition"
    ) -> Dict[str, Any]:
        """
        估计条件平均处理效应 (CATE)
        
        CATE = E[Y|do(β=1), X=x] - E[Y|do(β=0), X=x]
        
        Args:
            outcomes_treatment: 处理组结果
            outcomes_control: 对照组结果
            condition_mask_treatment: 处理组条件掩码
            condition_mask_control: 对照组条件掩码
            condition_name: 条件名称
        
        Returns:
            cate_result: 包含 CATE、CI、统计信息
        """
        outcomes_treatment = np.asarray(outcomes_treatment).flatten()
        outcomes_control = np.asarray(outcomes_control).flatten()
        condition_mask_treatment = np.asarray(condition_mask_treatment).flatten().astype(bool)
        condition_mask_control = np.asarray(condition_mask_control).flatten().astype(bool)
        
        # 筛选满足条件的样本
        y_t_cond = outcomes_treatment[condition_mask_treatment]
        y_c_cond = outcomes_control[condition_mask_control]
        
        if len(y_t_cond) == 0 or len(y_c_cond) == 0:
            return {
                "cate": float('nan'),
                "cate_std": float('nan'),
                "ci_low": float('nan'),
                "ci_high": float('nan'),
                "ci_level": self.ci_level,
                "n_treatment": len(y_t_cond),
                "n_control": len(y_c_cond),
                "n_boot": self.n_boot,
                "stat_method": "bootstrap_percentile",
                "condition": condition_name
            }
        
        # 点估计
        cate = np.mean(y_t_cond) - np.mean(y_c_cond)
        
        # Bootstrap CI
        boot_cates = []
        n_t, n_c = len(y_t_cond), len(y_c_cond)
        
        for _ in range(self.n_boot):
            idx_t = self.rng.choice(n_t, size=n_t, replace=True)
            idx_c = self.rng.choice(n_c, size=n_c, replace=True)
            boot_cate = np.mean(y_t_cond[idx_t]) - np.mean(y_c_cond[idx_c])
            boot_cates.append(boot_cate)
        
        boot_cates = np.array(boot_cates)
        cate_std = np.std(boot_cates, ddof=1)
        
        alpha = 1 - self.ci_level
        ci_low = np.percentile(boot_cates, 100 * alpha / 2)
        ci_high = np.percentile(boot_cates, 100 * (1 - alpha / 2))
        
        return {
            "cate": float(cate),
            "cate_std": float(cate_std),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_level": self.ci_level,
            "n_treatment": n_t,
            "n_control": n_c,
            "n_boot": self.n_boot,
            "stat_method": "bootstrap_percentile",
            "condition": condition_name
        }


class CausalBudgetOptimizer:
    """
    基于因果效应的预算优化器
    
    实现两阶段方法：
    1. 干预网格实验 → 收集数据
    2. ATE/CATE 估计 → 预算优化求解
    
    Requirements: §2.5, §9.2
    """
    
    def __init__(
        self,
        regions: Optional[List[str]] = None,
        n_boot: int = 500,
        seed: int = 42
    ):
        """
        初始化预算优化器
        
        Args:
            regions: 语义区域列表
            n_boot: Bootstrap 重采样次数
            seed: 随机种子
        """
        self.regions = regions or ["face", "background", "sensitive_attr"]
        self.estimator = CausalEffectEstimator(n_boot=n_boot, seed=seed)
        self.intervention_results: List[InterventionResult] = []
        self.ate_results: Dict[str, Dict] = {}
        self.cate_results: Dict[str, Dict] = {}
        self.seed = seed
    
    def add_intervention_result(self, result: InterventionResult) -> None:
        """添加干预实验结果"""
        self.intervention_results.append(result)
    
    def estimate_ate(
        self,
        intervention_results: Optional[List[InterventionResult]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        估计每个区域的 ATE
        
        ATE = E[A|do(β=1)] - E[A|do(β=0)]
        
        Args:
            intervention_results: 干预实验结果列表
        
        Returns:
            ate_by_region: {region: ate_result}
        """
        results = intervention_results or self.intervention_results
        
        ate_by_region = {}
        
        for region in self.regions:
            # 筛选该区域的实验结果
            region_results = [r for r in results if r.region == region]
            
            if not region_results:
                continue
            
            # 分组：高 β (≥0.75) vs 低 β (≤0.25)
            high_beta = [r for r in region_results if r.beta >= 0.75]
            low_beta = [r for r in region_results if r.beta <= 0.25]
            
            if not high_beta or not low_beta:
                continue
            
            # 提取 attack_success
            outcomes_high = np.array([r.attack_success for r in high_beta])
            outcomes_low = np.array([r.attack_success for r in low_beta])
            
            ate_result = self.estimator.estimate_ate(outcomes_high, outcomes_low)
            ate_result["region"] = region
            ate_by_region[region] = ate_result
        
        self.ate_results = ate_by_region
        return ate_by_region
    
    def estimate_cate(
        self,
        intervention_results: Optional[List[InterventionResult]] = None,
        condition_fn: Optional[callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        估计每个区域的 CATE
        
        CATE = E[A|do(β=1), X=x] - E[A|do(β=0), X=x]
        
        Args:
            intervention_results: 干预实验结果列表
            condition_fn: 条件函数，接受 InterventionResult 返回 bool
        
        Returns:
            cate_by_region: {region: cate_result}
        """
        results = intervention_results or self.intervention_results
        
        # 默认条件：utility > 0.5
        if condition_fn is None:
            condition_fn = lambda r: r.utility > 0.5
        
        cate_by_region = {}
        
        for region in self.regions:
            region_results = [r for r in results if r.region == region]
            
            if not region_results:
                continue
            
            high_beta = [r for r in region_results if r.beta >= 0.75]
            low_beta = [r for r in region_results if r.beta <= 0.25]
            
            if not high_beta or not low_beta:
                continue
            
            outcomes_high = np.array([r.attack_success for r in high_beta])
            outcomes_low = np.array([r.attack_success for r in low_beta])
            
            mask_high = np.array([condition_fn(r) for r in high_beta])
            mask_low = np.array([condition_fn(r) for r in low_beta])
            
            cate_result = self.estimator.estimate_cate(
                outcomes_high, outcomes_low,
                mask_high, mask_low,
                condition_name="utility>0.5"
            )
            cate_result["region"] = region
            cate_by_region[region] = cate_result
        
        self.cate_results = cate_by_region
        return cate_by_region


    def solve_optimal_allocation(
        self,
        utility_threshold: float = 0.65,
        attack_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        求解最优预算分配
        
        目标: min max_a attack_success_a
        约束: utility >= utility_threshold
        
        Args:
            utility_threshold: 效用门槛
            attack_weights: 攻击类型权重（可选）
        
        Returns:
            optimal_allocation: 最优预算分配结果
        """
        if not SCIPY_AVAILABLE:
            return self._solve_heuristic(utility_threshold)
        
        results = self.intervention_results
        if not results:
            raise ValueError("No intervention results available for optimization")
        
        # 构建响应面模型（简化：线性插值）
        region_data = {}
        for region in self.regions:
            region_results = [r for r in results if r.region == region]
            if region_results:
                betas = np.array([r.beta for r in region_results])
                attacks = np.array([r.attack_success for r in region_results])
                utilities = np.array([r.utility for r in region_results])
                region_data[region] = {
                    "betas": betas,
                    "attacks": attacks,
                    "utilities": utilities
                }
        
        if not region_data:
            return self._solve_heuristic(utility_threshold)
        
        def interpolate(region: str, beta: float, metric: str) -> float:
            """线性插值"""
            data = region_data.get(region)
            if data is None:
                return 0.5
            betas = data["betas"]
            values = data[metric]
            return float(np.interp(beta, np.sort(betas), values[np.argsort(betas)]))
        
        def objective(beta_vec: np.ndarray) -> float:
            """目标函数：min max attack_success"""
            max_attack = 0.0
            for i, region in enumerate(self.regions):
                attack = interpolate(region, beta_vec[i], "attacks")
                max_attack = max(max_attack, attack)
            return max_attack
        
        def utility_constraint(beta_vec: np.ndarray) -> float:
            """约束：utility >= threshold"""
            total_utility = 0.0
            for i, region in enumerate(self.regions):
                utility = interpolate(region, beta_vec[i], "utilities")
                total_utility += utility
            avg_utility = total_utility / len(self.regions)
            return avg_utility - utility_threshold
        
        # 优化求解
        n_regions = len(self.regions)
        bounds = [(0.0, 1.0)] * n_regions
        
        try:
            # 使用差分进化全局优化
            result = differential_evolution(
                objective,
                bounds,
                constraints={'type': 'ineq', 'fun': utility_constraint},
                seed=self.seed,
                maxiter=100,
                tol=1e-4,
                polish=True
            )
            
            optimal_betas = result.x
            optimal_attack = result.fun
            success = result.success
        except Exception:
            # 回退到启发式方法
            return self._solve_heuristic(utility_threshold)
        
        # 计算最优分配下的效用
        optimal_utility = 0.0
        for i, region in enumerate(self.regions):
            optimal_utility += interpolate(region, optimal_betas[i], "utilities")
        optimal_utility /= len(self.regions)
        
        allocation = {
            region: float(optimal_betas[i])
            for i, region in enumerate(self.regions)
        }
        
        return {
            "optimal_allocation": allocation,
            "optimal_attack_success": float(optimal_attack),
            "optimal_utility": float(optimal_utility),
            "utility_threshold": utility_threshold,
            "optimization_success": success,
            "method": "differential_evolution",
            "timestamp": datetime.now().isoformat()
        }
    
    def _solve_heuristic(self, utility_threshold: float) -> Dict[str, Any]:
        """启发式求解（当 scipy 不可用时）"""
        # 基于 ATE 的启发式分配
        allocation = {}
        
        for region in self.regions:
            ate_result = self.ate_results.get(region, {})
            ate = ate_result.get("ate", 0.0)
            
            # ATE > 0 表示高 β 增加攻击成功率，应降低 β
            # ATE < 0 表示高 β 降低攻击成功率，可提高 β
            if ate > 0.1:
                allocation[region] = 0.3  # 低隐私预算
            elif ate < -0.1:
                allocation[region] = 0.8  # 高隐私预算
            else:
                allocation[region] = 0.5  # 中等隐私预算
        
        return {
            "optimal_allocation": allocation,
            "optimal_attack_success": None,
            "optimal_utility": None,
            "utility_threshold": utility_threshold,
            "optimization_success": True,
            "method": "heuristic_ate_based",
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_causal_effects_csv(
        self,
        output_path: Union[str, Path],
        dataset: str = "default",
        task: str = "default"
    ) -> Path:
        """
        生成 causal_effects.csv
        
        Args:
            output_path: 输出路径
            dataset: 数据集名称
            task: 任务名称
        
        Returns:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        
        # ATE 结果
        for region, ate_result in self.ate_results.items():
            family_id = self._generate_family_id(dataset, task, "ate", region)
            row = {
                "dataset": dataset,
                "task": task,
                "region": region,
                "effect_type": "ATE",
                "effect_value": ate_result.get("ate", float('nan')),
                "effect_std": ate_result.get("ate_std", float('nan')),
                "ci_low": ate_result.get("ci_low", float('nan')),
                "ci_high": ate_result.get("ci_high", float('nan')),
                "ci_level": ate_result.get("ci_level", 0.95),
                "n_treatment": ate_result.get("n_treatment", 0),
                "n_control": ate_result.get("n_control", 0),
                "stat_method": ate_result.get("stat_method", "bootstrap_percentile"),
                "n_boot": ate_result.get("n_boot", 500),
                "family_id": family_id,
                "condition": ""
            }
            rows.append(row)
        
        # CATE 结果
        for region, cate_result in self.cate_results.items():
            family_id = self._generate_family_id(dataset, task, "cate", region)
            row = {
                "dataset": dataset,
                "task": task,
                "region": region,
                "effect_type": "CATE",
                "effect_value": cate_result.get("cate", float('nan')),
                "effect_std": cate_result.get("cate_std", float('nan')),
                "ci_low": cate_result.get("ci_low", float('nan')),
                "ci_high": cate_result.get("ci_high", float('nan')),
                "ci_level": cate_result.get("ci_level", 0.95),
                "n_treatment": cate_result.get("n_treatment", 0),
                "n_control": cate_result.get("n_control", 0),
                "stat_method": cate_result.get("stat_method", "bootstrap_percentile"),
                "n_boot": cate_result.get("n_boot", 500),
                "family_id": family_id,
                "condition": cate_result.get("condition", "")
            }
            rows.append(row)
        
        # 写入 CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return output_path
    
    def generate_optimal_allocation_json(
        self,
        output_path: Union[str, Path],
        utility_threshold: float = 0.65
    ) -> Path:
        """
        生成 optimal_allocation.json
        
        Args:
            output_path: 输出路径
            utility_threshold: 效用门槛
        
        Returns:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        allocation_result = self.solve_optimal_allocation(utility_threshold)
        
        # 添加元数据
        allocation_result["metadata"] = {
            "regions": self.regions,
            "n_intervention_results": len(self.intervention_results),
            "ate_regions": list(self.ate_results.keys()),
            "cate_regions": list(self.cate_results.keys())
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(allocation_result, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    @staticmethod
    def _generate_family_id(dataset: str, task: str, effect_type: str, region: str) -> str:
        """生成 family_id (GC9)"""
        key = f"{dataset}|{task}|{effect_type}|{region}"
        return hashlib.sha1(key.encode()).hexdigest()[:10]


class CausalEffectsEvaluator:
    """
    因果效应评估器（主入口）
    
    整合干预网格、ATE/CATE 估计、预算优化
    
    Evidence: tables/causal_effects.csv, tables/optimal_allocation.json
    """
    
    def __init__(
        self,
        run_dir: Union[str, Path],
        regions: Optional[List[str]] = None,
        n_boot: int = 500,
        seed: int = 42
    ):
        """
        初始化因果效应评估器
        
        Args:
            run_dir: 运行目录
            regions: 语义区域列表
            n_boot: Bootstrap 重采样次数
            seed: 随机种子
        """
        self.run_dir = Path(run_dir)
        self.tables_dir = self.run_dir / "tables"
        self.figures_dir = self.run_dir / "figures"
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.grid = InterventionGrid(regions=regions or ["face", "background", "sensitive_attr"])
        self.optimizer = CausalBudgetOptimizer(
            regions=self.grid.regions,
            n_boot=n_boot,
            seed=seed
        )
        self.seed = seed
    
    def run_intervention_experiments(
        self,
        experiment_fn: callable,
        dataset: str = "default",
        task: str = "default"
    ) -> List[InterventionResult]:
        """
        运行干预实验
        
        Args:
            experiment_fn: 实验函数，接受 (region, beta, n_samples) 返回 (attack_success, utility)
            dataset: 数据集名称
            task: 任务名称
        
        Returns:
            results: 干预实验结果列表
        """
        experiments = self.grid.generate_experiments()
        results = []
        
        for exp in experiments:
            region = exp["region"]
            beta = exp["beta"]
            n_samples = exp["n_samples"]
            
            try:
                attack_success, utility = experiment_fn(region, beta, n_samples)
                
                result = InterventionResult(
                    experiment_id=exp["experiment_id"],
                    region=region,
                    beta=beta,
                    attack_success=attack_success,
                    utility=utility,
                    n_samples=n_samples
                )
                results.append(result)
                self.optimizer.add_intervention_result(result)
            except Exception as e:
                print(f"Warning: Experiment {exp['experiment_id']} failed: {e}")
        
        return results
    
    def estimate_effects(self) -> Tuple[Dict, Dict]:
        """
        估计 ATE 和 CATE
        
        Returns:
            (ate_results, cate_results)
        """
        ate_results = self.optimizer.estimate_ate()
        cate_results = self.optimizer.estimate_cate()
        return ate_results, cate_results
    
    def optimize_allocation(self, utility_threshold: float = 0.65) -> Dict[str, Any]:
        """
        优化预算分配
        
        Args:
            utility_threshold: 效用门槛
        
        Returns:
            optimal_allocation: 最优分配结果
        """
        return self.optimizer.solve_optimal_allocation(utility_threshold)
    
    def generate_outputs(
        self,
        dataset: str = "default",
        task: str = "default",
        utility_threshold: float = 0.65
    ) -> Dict[str, Path]:
        """
        生成所有输出文件
        
        Args:
            dataset: 数据集名称
            task: 任务名称
            utility_threshold: 效用门槛
        
        Returns:
            output_paths: 输出文件路径字典
        """
        # 生成 causal_effects.csv
        csv_path = self.optimizer.generate_causal_effects_csv(
            self.tables_dir / "causal_effects.csv",
            dataset=dataset,
            task=task
        )
        
        # 生成 optimal_allocation.json
        json_path = self.optimizer.generate_optimal_allocation_json(
            self.tables_dir / "optimal_allocation.json",
            utility_threshold=utility_threshold
        )
        
        return {
            "causal_effects_csv": csv_path,
            "optimal_allocation_json": json_path
        }
    
    def generate_report(self) -> str:
        """
        生成因果效应分析报告
        
        Returns:
            report: Markdown 格式报告
        """
        lines = [
            "# 因果效应分析报告",
            "",
            f"生成时间: {datetime.now().isoformat()}",
            "",
            "## 1. 干预网格配置",
            "",
            f"- 区域: {', '.join(self.grid.regions)}",
            f"- β 值: {', '.join(map(str, self.grid.beta_values))}",
            f"- 每单元样本数: {self.grid.n_samples_per_cell}",
            f"- 总实验数: {self.grid.get_grid_size()}",
            "",
            "## 2. ATE 估计结果",
            "",
            "| 区域 | ATE | 95% CI | n_boot |",
            "|------|-----|--------|--------|"
        ]
        
        for region, ate in self.optimizer.ate_results.items():
            ci_str = f"[{ate.get('ci_low', 'N/A'):.4f}, {ate.get('ci_high', 'N/A'):.4f}]"
            lines.append(f"| {region} | {ate.get('ate', 'N/A'):.4f} | {ci_str} | {ate.get('n_boot', 'N/A')} |")
        
        lines.extend([
            "",
            "## 3. CATE 估计结果",
            "",
            "| 区域 | CATE | 95% CI | 条件 |",
            "|------|------|--------|------|"
        ])
        
        for region, cate in self.optimizer.cate_results.items():
            ci_str = f"[{cate.get('ci_low', 'N/A'):.4f}, {cate.get('ci_high', 'N/A'):.4f}]"
            condition = cate.get('condition', 'N/A')
            lines.append(f"| {region} | {cate.get('cate', 'N/A'):.4f} | {ci_str} | {condition} |")
        
        lines.extend([
            "",
            "## 4. 解释",
            "",
            "- ATE > 0: 高隐私预算增加攻击成功率（不利）",
            "- ATE < 0: 高隐私预算降低攻击成功率（有利）",
            "- CATE 提供条件下的因果效应估计",
            ""
        ])
        
        return "\n".join(lines)


# 便捷函数
def create_intervention_grid(
    regions: Optional[List[str]] = None,
    beta_values: Optional[List[float]] = None,
    n_samples_per_cell: int = 100
) -> InterventionGrid:
    """创建干预网格"""
    return InterventionGrid(
        regions=regions or ["face", "background", "sensitive_attr"],
        beta_values=beta_values or [0.0, 0.25, 0.5, 0.75, 1.0],
        n_samples_per_cell=n_samples_per_cell
    )


def run_causal_analysis(
    run_dir: Union[str, Path],
    experiment_fn: callable,
    dataset: str = "default",
    task: str = "default",
    utility_threshold: float = 0.65,
    n_boot: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    运行完整的因果分析流程
    
    Args:
        run_dir: 运行目录
        experiment_fn: 实验函数
        dataset: 数据集名称
        task: 任务名称
        utility_threshold: 效用门槛
        n_boot: Bootstrap 重采样次数
        seed: 随机种子
    
    Returns:
        results: 分析结果
    """
    evaluator = CausalEffectsEvaluator(
        run_dir=run_dir,
        n_boot=n_boot,
        seed=seed
    )
    
    # 运行干预实验
    intervention_results = evaluator.run_intervention_experiments(
        experiment_fn, dataset, task
    )
    
    # 估计因果效应
    ate_results, cate_results = evaluator.estimate_effects()
    
    # 优化预算分配
    optimal_allocation = evaluator.optimize_allocation(utility_threshold)
    
    # 生成输出文件
    output_paths = evaluator.generate_outputs(dataset, task, utility_threshold)
    
    # 生成报告
    report = evaluator.generate_report()
    
    return {
        "intervention_results": intervention_results,
        "ate_results": ate_results,
        "cate_results": cate_results,
        "optimal_allocation": optimal_allocation,
        "output_paths": output_paths,
        "report": report
    }
