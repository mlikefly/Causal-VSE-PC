# -*- coding: utf-8 -*-
"""
统计引擎模块

实现 T9: 统计引擎
- Bootstrap CI 计算 (n_boot ≥ 500)
- BH-FDR 多重比较校正
- Family ID 生成 (GC9)

Requirements: Property 11, Property 16, GC3, GC9
Validates: Requirements 10.1, 10.4, 10.5, 10.6

Inputs/Outputs Contract:
- 输入: values (array), run_seed, dataset, task, metric_name, privacy_level
- 输出: 所有 CSV 补齐 ci_low, ci_high, n_boot, family_id, alpha
- 约束: n_boot ≥ 500；family_id 确定性（相同输入 → 相同输出）
"""

import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# 常量
MIN_N_BOOT = 500
DEFAULT_ALPHA = 0.05
DEFAULT_CI_LEVEL = 0.95


@dataclass
class BootstrapCIResult:
    """Bootstrap 置信区间结果"""
    mean: float
    ci_low: float
    ci_high: float
    ci_level: float
    std: float
    n_samples: int
    n_boot: int
    stat_method: str = "bootstrap_percentile"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultipleComparisonResult:
    """多重比较校正结果"""
    original_p_values: List[float]
    adjusted_p_values: List[float]
    rejected: List[bool]
    alpha: float
    method: str = "BH-FDR"
    n_tests: int = 0
    n_rejected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalSummary:
    """统计摘要"""
    dataset: str
    task: str
    metric_name: str
    privacy_level: float
    mean: float
    std: float
    ci_low: float
    ci_high: float
    ci_level: float
    stat_method: str
    n_boot: int
    family_id: str
    alpha: float
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticsEngine:
    """
    统计引擎
    
    支持：
    - Bootstrap CI 计算
    - BH-FDR 多重比较校正
    - Family ID 生成
    
    Requirements: Property 11, Property 16
    """
    
    def __init__(
        self,
        n_boot: int = 1000,
        alpha: float = 0.05,
        ci_level: float = 0.95,
        seed: int = 42
    ):
        """
        初始化统计引擎
        
        Args:
            n_boot: Bootstrap 重采样次数（强制 ≥ 500）
            alpha: 显著性水平
            ci_level: 置信区间水平
            seed: 随机种子
        """
        self.n_boot = max(MIN_N_BOOT, n_boot)
        self.alpha = alpha
        self.ci_level = ci_level
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def compute_ci(
        self,
        values: Union[np.ndarray, List[float]],
        ci_level: Optional[float] = None,
        method: str = "percentile"
    ) -> BootstrapCIResult:
        """
        计算 Bootstrap 置信区间
        
        Property 11: 统计严谨性完整性
        - n_boot ≥ 500
        - 输出包含 ci_low, ci_high, stat_method, n_boot
        
        Args:
            values: 样本值数组
            ci_level: 置信区间水平（可选，默认使用初始化值）
            method: CI 计算方法 ("percentile" 或 "bca")
        
        Returns:
            result: Bootstrap CI 结果
        """
        values = np.asarray(values).flatten()
        n = len(values)
        
        if n == 0:
            return BootstrapCIResult(
                mean=float('nan'),
                ci_low=float('nan'),
                ci_high=float('nan'),
                ci_level=ci_level or self.ci_level,
                std=float('nan'),
                n_samples=0,
                n_boot=self.n_boot,
                stat_method=f"bootstrap_{method}"
            )
        
        ci_level = ci_level or self.ci_level
        
        # 点估计
        sample_mean = np.mean(values)
        sample_std = np.std(values, ddof=1) if n > 1 else 0.0
        
        if n == 1:
            return BootstrapCIResult(
                mean=float(sample_mean),
                ci_low=float(sample_mean),
                ci_high=float(sample_mean),
                ci_level=ci_level,
                std=0.0,
                n_samples=1,
                n_boot=self.n_boot,
                stat_method=f"bootstrap_{method}"
            )
        
        # Bootstrap 重采样
        boot_means = []
        for _ in range(self.n_boot):
            boot_sample = self.rng.choice(values, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))
        
        boot_means = np.array(boot_means)
        
        # 计算 CI
        alpha = 1 - ci_level
        
        if method == "percentile":
            ci_low = np.percentile(boot_means, 100 * alpha / 2)
            ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
        elif method == "bca" and SCIPY_AVAILABLE:
            # BCa (Bias-Corrected and Accelerated) 方法
            ci_low, ci_high = self._compute_bca_ci(values, boot_means, alpha)
        else:
            # 回退到百分位法
            ci_low = np.percentile(boot_means, 100 * alpha / 2)
            ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
        
        return BootstrapCIResult(
            mean=float(sample_mean),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            ci_level=ci_level,
            std=float(sample_std),
            n_samples=n,
            n_boot=self.n_boot,
            stat_method=f"bootstrap_{method}"
        )
    
    def _compute_bca_ci(
        self,
        values: np.ndarray,
        boot_means: np.ndarray,
        alpha: float
    ) -> Tuple[float, float]:
        """计算 BCa 置信区间"""
        n = len(values)
        sample_mean = np.mean(values)
        
        # 偏差校正因子 z0
        prop_less = np.mean(boot_means < sample_mean)
        z0 = scipy_stats.norm.ppf(prop_less) if 0 < prop_less < 1 else 0.0
        
        # 加速因子 a (使用 jackknife)
        jackknife_means = []
        for i in range(n):
            jack_sample = np.delete(values, i)
            jackknife_means.append(np.mean(jack_sample))
        jackknife_means = np.array(jackknife_means)
        
        jack_mean = np.mean(jackknife_means)
        num = np.sum((jack_mean - jackknife_means) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0.0
        
        # 调整后的百分位数
        z_alpha_low = scipy_stats.norm.ppf(alpha / 2)
        z_alpha_high = scipy_stats.norm.ppf(1 - alpha / 2)
        
        def adjusted_percentile(z_alpha):
            num = z0 + z_alpha
            denom = 1 - a * num
            if denom == 0:
                return 0.5
            adjusted_z = z0 + num / denom
            return scipy_stats.norm.cdf(adjusted_z)
        
        p_low = adjusted_percentile(z_alpha_low)
        p_high = adjusted_percentile(z_alpha_high)
        
        ci_low = np.percentile(boot_means, 100 * p_low)
        ci_high = np.percentile(boot_means, 100 * p_high)
        
        return ci_low, ci_high
    
    def multiple_comparison_correction(
        self,
        p_values: Union[np.ndarray, List[float]],
        method: str = "BH-FDR",
        alpha: Optional[float] = None
    ) -> MultipleComparisonResult:
        """
        多重比较校正
        
        Requirements: R10.2
        
        Args:
            p_values: 原始 p 值列表
            method: 校正方法 ("BH-FDR", "bonferroni", "holm")
            alpha: 显著性水平
        
        Returns:
            result: 校正结果
        """
        p_values = np.asarray(p_values).flatten()
        n = len(p_values)
        alpha = alpha or self.alpha
        
        if n == 0:
            return MultipleComparisonResult(
                original_p_values=[],
                adjusted_p_values=[],
                rejected=[],
                alpha=alpha,
                method=method,
                n_tests=0,
                n_rejected=0
            )
        
        if method == "BH-FDR":
            adjusted, rejected = self._bh_fdr_correction(p_values, alpha)
        elif method == "bonferroni":
            adjusted = np.minimum(p_values * n, 1.0)
            rejected = adjusted < alpha
        elif method == "holm":
            adjusted, rejected = self._holm_correction(p_values, alpha)
        else:
            # 默认使用 BH-FDR
            adjusted, rejected = self._bh_fdr_correction(p_values, alpha)
        
        return MultipleComparisonResult(
            original_p_values=p_values.tolist(),
            adjusted_p_values=adjusted.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
            method=method,
            n_tests=n,
            n_rejected=int(np.sum(rejected))
        )
    
    def _bh_fdr_correction(
        self,
        p_values: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR 校正
        
        GC3: BH-FDR 多重比较校正
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # 计算调整后的 p 值
        adjusted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                adjusted[i] = sorted_p[i]
            else:
                adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
        
        # 恢复原始顺序
        adjusted_original = np.zeros(n)
        adjusted_original[sorted_indices] = adjusted
        
        # 确定拒绝
        rejected = adjusted_original < alpha
        
        return adjusted_original, rejected
    
    def _holm_correction(
        self,
        p_values: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Holm-Bonferroni 校正"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # 计算调整后的 p 值
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = sorted_p[i] * (n - i)
        
        # 单调性校正
        for i in range(1, n):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])
        
        adjusted = np.minimum(adjusted, 1.0)
        
        # 恢复原始顺序
        adjusted_original = np.zeros(n)
        adjusted_original[sorted_indices] = adjusted
        
        rejected = adjusted_original < alpha
        
        return adjusted_original, rejected
    
    @staticmethod
    def generate_family_id(
        dataset: str,
        task: str,
        metric_name: str,
        privacy_level: float
    ) -> str:
        """
        生成 Family ID
        
        Property 16: Family ID 确定性
        - 相同输入 → 相同输出
        - family_id = sha1(f"{dataset}|{task}|{metric_name}|{privacy_level}")[:10]
        
        GC9: Family ID 计算规则
        
        Args:
            dataset: 数据集名称
            task: 任务名称
            metric_name: 指标名称
            privacy_level: 隐私等级
        
        Returns:
            family_id: 10 字符的哈希 ID
        """
        key = f"{dataset}|{task}|{metric_name}|{privacy_level}"
        return hashlib.sha1(key.encode()).hexdigest()[:10]
    
    def compute_summary(
        self,
        values: Union[np.ndarray, List[float]],
        dataset: str,
        task: str,
        metric_name: str,
        privacy_level: float
    ) -> StatisticalSummary:
        """
        计算完整的统计摘要
        
        Args:
            values: 样本值数组
            dataset: 数据集名称
            task: 任务名称
            metric_name: 指标名称
            privacy_level: 隐私等级
        
        Returns:
            summary: 统计摘要
        """
        ci_result = self.compute_ci(values)
        family_id = self.generate_family_id(dataset, task, metric_name, privacy_level)
        
        return StatisticalSummary(
            dataset=dataset,
            task=task,
            metric_name=metric_name,
            privacy_level=privacy_level,
            mean=ci_result.mean,
            std=ci_result.std,
            ci_low=ci_result.ci_low,
            ci_high=ci_result.ci_high,
            ci_level=ci_result.ci_level,
            stat_method=ci_result.stat_method,
            n_boot=ci_result.n_boot,
            family_id=family_id,
            alpha=self.alpha,
            n_samples=ci_result.n_samples
        )


    def validate_statistical_fields(
        self,
        row: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证统计字段完整性
        
        Property 11: 统计严谨性完整性
        - stat_method, n_boot (≥500), ci_low, ci_high, alpha, family_id 必须存在且有效
        
        Args:
            row: CSV 行数据
            required_fields: 必需字段列表
        
        Returns:
            (is_valid, missing_fields)
        """
        if required_fields is None:
            required_fields = [
                "stat_method", "n_boot", "ci_low", "ci_high", "alpha", "family_id"
            ]
        
        missing = []
        
        for field in required_fields:
            if field not in row:
                missing.append(field)
            elif row[field] is None:
                missing.append(f"{field} (null)")
            elif field == "n_boot" and row[field] < MIN_N_BOOT:
                missing.append(f"{field} (< {MIN_N_BOOT})")
        
        return len(missing) == 0, missing
    
    def augment_csv_row(
        self,
        row: Dict[str, Any],
        values: Optional[Union[np.ndarray, List[float]]] = None,
        dataset: Optional[str] = None,
        task: Optional[str] = None,
        metric_name: Optional[str] = None,
        privacy_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        为 CSV 行补齐统计字段
        
        Args:
            row: 原始 CSV 行
            values: 样本值（用于计算 CI）
            dataset: 数据集名称
            task: 任务名称
            metric_name: 指标名称
            privacy_level: 隐私等级
        
        Returns:
            augmented_row: 补齐后的行
        """
        augmented = row.copy()
        
        # 从行中提取信息（如果未提供）
        dataset = dataset or row.get("dataset", "unknown")
        task = task or row.get("task", "unknown")
        metric_name = metric_name or row.get("metric_name", "unknown")
        privacy_level = privacy_level if privacy_level is not None else row.get("privacy_level", 0.5)
        
        # 生成 family_id
        if "family_id" not in augmented or not augmented["family_id"]:
            augmented["family_id"] = self.generate_family_id(
                dataset, task, metric_name, privacy_level
            )
        
        # 补齐统计字段
        if "stat_method" not in augmented or not augmented["stat_method"]:
            augmented["stat_method"] = "bootstrap_percentile"
        
        if "n_boot" not in augmented or not augmented["n_boot"]:
            augmented["n_boot"] = self.n_boot
        
        if "alpha" not in augmented or augmented["alpha"] is None:
            augmented["alpha"] = self.alpha
        
        # 如果提供了值，计算 CI
        if values is not None and len(values) > 0:
            ci_result = self.compute_ci(values)
            if "ci_low" not in augmented or augmented["ci_low"] is None:
                augmented["ci_low"] = ci_result.ci_low
            if "ci_high" not in augmented or augmented["ci_high"] is None:
                augmented["ci_high"] = ci_result.ci_high
        
        return augmented


class StatisticsValidator:
    """
    统计验证器
    
    验证 CSV 文件的统计字段完整性
    """
    
    def __init__(self, engine: Optional[StatisticsEngine] = None):
        """
        初始化验证器
        
        Args:
            engine: 统计引擎实例
        """
        self.engine = engine or StatisticsEngine()
    
    def validate_csv(
        self,
        csv_path: Union[str, Path],
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        验证 CSV 文件的统计字段
        
        Args:
            csv_path: CSV 文件路径
            required_fields: 必需字段列表
        
        Returns:
            validation_result: 验证结果
        """
        import csv
        
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            return {
                "valid": False,
                "error": f"File not found: {csv_path}",
                "total_rows": 0,
                "valid_rows": 0,
                "invalid_rows": []
            }
        
        invalid_rows = []
        total_rows = 0
        valid_rows = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                total_rows += 1
                is_valid, missing = self.engine.validate_statistical_fields(
                    row, required_fields
                )
                if is_valid:
                    valid_rows += 1
                else:
                    invalid_rows.append({
                        "row_index": i,
                        "missing_fields": missing
                    })
        
        return {
            "valid": len(invalid_rows) == 0,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "coverage": valid_rows / total_rows if total_rows > 0 else 0.0
        }
    
    def validate_family_id_determinism(
        self,
        csv_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        验证 Family ID 确定性
        
        Property 16: 相同 (dataset, task, metric_name, privacy_level) → 相同 family_id
        
        Args:
            csv_path: CSV 文件路径
        
        Returns:
            validation_result: 验证结果
        """
        import csv
        
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            return {
                "valid": False,
                "error": f"File not found: {csv_path}"
            }
        
        # 收集 (key, family_id) 对
        key_to_family_ids: Dict[str, set] = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("dataset", "")
                task = row.get("task", "")
                metric_name = row.get("metric_name", "")
                privacy_level = row.get("privacy_level", "")
                family_id = row.get("family_id", "")
                
                key = f"{dataset}|{task}|{metric_name}|{privacy_level}"
                
                if key not in key_to_family_ids:
                    key_to_family_ids[key] = set()
                key_to_family_ids[key].add(family_id)
        
        # 检查每个 key 是否只有一个 family_id
        inconsistent_keys = []
        for key, family_ids in key_to_family_ids.items():
            if len(family_ids) > 1:
                inconsistent_keys.append({
                    "key": key,
                    "family_ids": list(family_ids)
                })
        
        return {
            "valid": len(inconsistent_keys) == 0,
            "total_keys": len(key_to_family_ids),
            "inconsistent_keys": inconsistent_keys
        }


# 便捷函数
def compute_bootstrap_ci(
    values: Union[np.ndarray, List[float]],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> BootstrapCIResult:
    """
    计算 Bootstrap 置信区间（便捷函数）
    
    Args:
        values: 样本值
        n_boot: Bootstrap 重采样次数
        ci_level: 置信区间水平
        seed: 随机种子
    
    Returns:
        result: Bootstrap CI 结果
    """
    engine = StatisticsEngine(n_boot=n_boot, ci_level=ci_level, seed=seed)
    return engine.compute_ci(values)


def bh_fdr_correction(
    p_values: Union[np.ndarray, List[float]],
    alpha: float = 0.05
) -> MultipleComparisonResult:
    """
    BH-FDR 多重比较校正（便捷函数）
    
    Args:
        p_values: 原始 p 值
        alpha: 显著性水平
    
    Returns:
        result: 校正结果
    """
    engine = StatisticsEngine(alpha=alpha)
    return engine.multiple_comparison_correction(p_values, method="BH-FDR")


def generate_family_id(
    dataset: str,
    task: str,
    metric_name: str,
    privacy_level: float
) -> str:
    """
    生成 Family ID（便捷函数）
    
    Args:
        dataset: 数据集名称
        task: 任务名称
        metric_name: 指标名称
        privacy_level: 隐私等级
    
    Returns:
        family_id: 10 字符的哈希 ID
    """
    return StatisticsEngine.generate_family_id(dataset, task, metric_name, privacy_level)
