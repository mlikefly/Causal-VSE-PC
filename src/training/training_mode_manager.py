# -*- coding: utf-8 -*-
"""
Training Mode Manager for Top-Journal Experiment Suite.

Implements 4 training modes (P2P/P2Z/Z2Z/Mix2Z) and C-view Guard.
Corresponds to design.md §8.1, §8.2 and tasks.md T6.

**Validates:**
- Property 6: 训练模式数据隔离
- Property 7: 效用门槛计算
- Requirements: 4.1, 4.2, 4.3, 4.4, 4.6, 4.8, 11.3, 11.6
"""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# Try to import torch for DataLoader
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DataLoader = None
    Dataset = None


# =============================================================================
# Enums and Constants
# =============================================================================

class TrainingMode(Enum):
    """
    训练模式枚举。
    
    | 模式 | 训练数据 | 测试数据 | 说明 |
    |------|----------|----------|------|
    | P2P | Plaintext | Plaintext | 基线（无隐私保护） |
    | P2Z | Plaintext | Z-view | 域迁移测试 |
    | Z2Z | Z-view | Z-view | 完全加密训练 |
    | Mix2Z | 50% Plaintext + 50% Z-view | Z-view | 混合训练 |
    """
    P2P = "P2P"
    P2Z = "P2Z"
    Z2Z = "Z2Z"
    Mix2Z = "Mix2Z"


class ViewType(Enum):
    """视图类型枚举。"""
    PLAINTEXT = "plaintext"
    Z_VIEW = "z_view"
    C_VIEW = "c_view"


# 效用门槛定义（冻结）- 来自 design.md §9.3
UTILITY_THRESHOLDS = {
    0.3: 0.75,  # privacy_level=0.3 时需达到 75% P2P 性能
    0.5: 0.65,  # privacy_level=0.5 时需达到 65% P2P 性能
    0.7: 0.55,  # privacy_level=0.7 时需达到 55% P2P 性能
}


# =============================================================================
# Exceptions
# =============================================================================

class ViewAccessError(Exception):
    """
    C-view 访问错误。
    
    训练时请求 c_view 数据时抛出此异常。
    
    **Validates: Property 6**
    """
    pass


class UtilityThresholdError(Exception):
    """效用门槛未达标错误。"""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UtilityMetricsRow:
    """
    utility_metrics.csv 行数据。
    
    对应 design.md §10.1 UTILITY_FIELDS。
    """
    dataset: str
    task: str
    method: str
    training_mode: str
    privacy_level: float
    seed: int
    metric_name: str
    metric_value: float
    relative_to: str = "P2P_mean"
    relative_performance: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    stat_method: str = "bootstrap"
    n_boot: int = 500
    family_id: str = ""
    alpha: float = 0.05
    
    def __post_init__(self):
        """生成 family_id（如果未提供）。"""
        if not self.family_id:
            key = f"{self.dataset}|{self.task}|{self.metric_name}|{self.privacy_level}"
            self.family_id = hashlib.sha1(key.encode()).hexdigest()[:10]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            'dataset': self.dataset,
            'task': self.task,
            'method': self.method,
            'training_mode': self.training_mode,
            'privacy_level': self.privacy_level,
            'seed': self.seed,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'relative_to': self.relative_to,
            'relative_performance': self.relative_performance,
            'ci_low': self.ci_low,
            'ci_high': self.ci_high,
            'stat_method': self.stat_method,
            'n_boot': self.n_boot,
            'family_id': self.family_id,
            'alpha': self.alpha,
        }


@dataclass
class DataLoaderAuditEntry:
    """DataLoader 审计日志条目。"""
    timestamp: str
    split: str
    training_mode: str
    requested_views: List[str]
    returned_views: List[str]
    blocked_views: List[str]
    sample_count: int
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'split': self.split,
            'training_mode': self.training_mode,
            'requested_views': self.requested_views,
            'returned_views': self.returned_views,
            'blocked_views': self.blocked_views,
            'sample_count': self.sample_count,
        }



# =============================================================================
# TrainingModeManager
# =============================================================================

class TrainingModeManager:
    """
    训练模式管理器。
    
    支持 4 种训练模式：P2P/P2Z/Z2Z/Mix2Z
    实现 C-view Guard：训练时永不返回 c_view
    
    **Validates: Property 6 - 训练模式数据隔离**
    """
    
    TRAINING_MODES = [TrainingMode.P2P, TrainingMode.P2Z, TrainingMode.Z2Z, TrainingMode.Mix2Z]
    
    def __init__(
        self,
        mode: Union[str, TrainingMode],
        run_dir: Optional[Path] = None,
        mix_ratio: float = 0.5,
    ):
        """
        初始化训练模式管理器。
        
        Args:
            mode: 训练模式
            run_dir: 运行目录（用于审计日志）
            mix_ratio: Mix2Z 模式中 plaintext 的比例（默认 0.5）
        """
        if isinstance(mode, str):
            mode = TrainingMode(mode)
        self.mode = mode
        self.run_dir = Path(run_dir) if run_dir else None
        self.mix_ratio = mix_ratio
        self.audit_log: List[DataLoaderAuditEntry] = []
    
    def get_train_view_type(self) -> List[ViewType]:
        """
        获取训练时使用的视图类型。
        
        Returns:
            训练视图类型列表
        """
        if self.mode == TrainingMode.P2P:
            return [ViewType.PLAINTEXT]
        elif self.mode == TrainingMode.P2Z:
            return [ViewType.PLAINTEXT]
        elif self.mode == TrainingMode.Z2Z:
            return [ViewType.Z_VIEW]
        elif self.mode == TrainingMode.Mix2Z:
            return [ViewType.PLAINTEXT, ViewType.Z_VIEW]
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")
    
    def get_test_view_type(self) -> ViewType:
        """
        获取测试时使用的视图类型。
        
        Returns:
            测试视图类型
        """
        if self.mode == TrainingMode.P2P:
            return ViewType.PLAINTEXT
        else:
            return ViewType.Z_VIEW
    
    def validate_view_request(
        self,
        split: str,
        requested_views: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        验证视图请求，实现 C-view Guard。
        
        Args:
            split: 数据分割（train/val/test）
            requested_views: 请求的视图列表
            
        Returns:
            (allowed_views, blocked_views) 元组
            
        Raises:
            ViewAccessError: 训练时请求 c_view
        """
        allowed = []
        blocked = []
        
        for view in requested_views:
            view_lower = view.lower()
            
            # C-view Guard: 训练时永不返回 c_view
            if view_lower == "c_view" and split == "train":
                blocked.append(view)
                raise ViewAccessError(
                    f"C-view access denied during training. "
                    f"Split: {split}, Mode: {self.mode.value}. "
                    f"C-view is only allowed for security evaluation, not training."
                )
            
            # 根据模式和分割验证
            if split == "train":
                if self.mode == TrainingMode.P2P:
                    if view_lower == "plaintext":
                        allowed.append(view)
                    else:
                        blocked.append(view)
                elif self.mode == TrainingMode.P2Z:
                    if view_lower == "plaintext":
                        allowed.append(view)
                    else:
                        blocked.append(view)
                elif self.mode == TrainingMode.Z2Z:
                    if view_lower == "z_view":
                        allowed.append(view)
                    else:
                        blocked.append(view)
                elif self.mode == TrainingMode.Mix2Z:
                    if view_lower in ["plaintext", "z_view"]:
                        allowed.append(view)
                    else:
                        blocked.append(view)
            else:
                # val/test 分割
                if view_lower != "c_view":
                    allowed.append(view)
                else:
                    blocked.append(view)
        
        # 记录审计日志
        self._log_access(split, requested_views, allowed, blocked)
        
        return allowed, blocked
    
    def _log_access(
        self,
        split: str,
        requested: List[str],
        allowed: List[str],
        blocked: List[str],
        sample_count: int = 0,
    ) -> None:
        """记录访问审计日志。"""
        entry = DataLoaderAuditEntry(
            timestamp=datetime.now().isoformat(),
            split=split,
            training_mode=self.mode.value,
            requested_views=requested,
            returned_views=allowed,
            blocked_views=blocked,
            sample_count=sample_count,
        )
        self.audit_log.append(entry)
    
    def get_sample_view(
        self,
        sample_idx: int,
        split: str,
        plaintext: Any,
        z_view: Any,
        c_view: Optional[Any] = None,
    ) -> Any:
        """
        根据模式和分割获取样本的正确视图。
        
        Args:
            sample_idx: 样本索引
            split: 数据分割
            plaintext: 明文数据
            z_view: Z-view 数据
            c_view: C-view 数据（可选）
            
        Returns:
            正确的视图数据
            
        Raises:
            ViewAccessError: 训练时请求 c_view
        """
        if split == "train":
            if self.mode == TrainingMode.P2P:
                return plaintext
            elif self.mode == TrainingMode.P2Z:
                return plaintext
            elif self.mode == TrainingMode.Z2Z:
                return z_view
            elif self.mode == TrainingMode.Mix2Z:
                # 根据 mix_ratio 决定返回哪个视图
                if np.random.random() < self.mix_ratio:
                    return plaintext
                else:
                    return z_view
        else:
            # val/test
            if self.mode == TrainingMode.P2P:
                return plaintext
            else:
                return z_view
    
    def persist_audit_log(self, output_path: Optional[Path] = None) -> Path:
        """
        持久化审计日志。
        
        Args:
            output_path: 输出路径
            
        Returns:
            日志文件路径
        """
        if output_path is None:
            if self.run_dir:
                output_path = self.run_dir / "logs" / "dataloader_audit.json"
            else:
                output_path = Path("dataloader_audit.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'training_mode': self.mode.value,
            'mix_ratio': self.mix_ratio,
            'total_entries': len(self.audit_log),
            'entries': [e.to_dict() for e in self.audit_log],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def check_cview_leakage(self) -> Tuple[bool, List[DataLoaderAuditEntry]]:
        """
        检查是否存在 C-view 泄漏。
        
        Returns:
            (is_clean, violations) 元组
        """
        violations = []
        for entry in self.audit_log:
            if entry.split == "train" and "c_view" in [v.lower() for v in entry.returned_views]:
                violations.append(entry)
        
        return len(violations) == 0, violations



# =============================================================================
# UtilityThresholdChecker
# =============================================================================

class UtilityThresholdChecker:
    """
    效用门槛检查器。
    
    检查 relative_performance 是否达到门槛：
    - privacy_level=0.3 → 75% P2P
    - privacy_level=0.5 → 65% P2P
    - privacy_level=0.7 → 55% P2P
    
    **Validates: Property 7 - 效用门槛计算**
    """
    
    def __init__(
        self,
        thresholds: Optional[Dict[float, float]] = None,
        run_dir: Optional[Path] = None,
    ):
        """
        初始化效用门槛检查器。
        
        Args:
            thresholds: 门槛字典 {privacy_level: threshold}
            run_dir: 运行目录
        """
        self.thresholds = thresholds or UTILITY_THRESHOLDS.copy()
        self.run_dir = Path(run_dir) if run_dir else None
        self.p2p_baselines: Dict[str, float] = {}  # {metric_name: p2p_mean}
    
    def set_p2p_baseline(self, metric_name: str, p2p_mean: float) -> None:
        """
        设置 P2P 基线值。
        
        Args:
            metric_name: 指标名称
            p2p_mean: P2P 模式的平均值
        """
        self.p2p_baselines[metric_name] = p2p_mean
    
    def compute_relative_performance(
        self,
        metric_value: float,
        metric_name: str,
        p2p_mean: Optional[float] = None,
    ) -> float:
        """
        计算相对性能。
        
        relative_performance = metric_value / P2P_mean
        
        Args:
            metric_value: 当前指标值
            metric_name: 指标名称
            p2p_mean: P2P 基线值（可选，如果已设置）
            
        Returns:
            相对性能值
        """
        if p2p_mean is None:
            p2p_mean = self.p2p_baselines.get(metric_name)
        
        if p2p_mean is None or p2p_mean == 0:
            return 0.0
        
        return metric_value / p2p_mean
    
    def get_threshold(self, privacy_level: float) -> float:
        """
        获取指定隐私级别的门槛。
        
        Args:
            privacy_level: 隐私级别
            
        Returns:
            门槛值
        """
        # 精确匹配
        if privacy_level in self.thresholds:
            return self.thresholds[privacy_level]
        
        # 线性插值
        levels = sorted(self.thresholds.keys())
        if privacy_level <= levels[0]:
            return self.thresholds[levels[0]]
        if privacy_level >= levels[-1]:
            return self.thresholds[levels[-1]]
        
        # 找到相邻的两个级别
        for i in range(len(levels) - 1):
            if levels[i] <= privacy_level <= levels[i + 1]:
                t = (privacy_level - levels[i]) / (levels[i + 1] - levels[i])
                return self.thresholds[levels[i]] * (1 - t) + self.thresholds[levels[i + 1]] * t
        
        return 0.5  # 默认
    
    def check_threshold(
        self,
        metric_value: float,
        metric_name: str,
        privacy_level: float,
        p2p_mean: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        检查是否达到门槛。
        
        Args:
            metric_value: 当前指标值
            metric_name: 指标名称
            privacy_level: 隐私级别
            p2p_mean: P2P 基线值
            
        Returns:
            (passed, relative_performance, threshold) 元组
        """
        relative_perf = self.compute_relative_performance(metric_value, metric_name, p2p_mean)
        threshold = self.get_threshold(privacy_level)
        passed = relative_perf >= threshold
        
        return passed, relative_perf, threshold
    
    def check_all(
        self,
        metrics: List[UtilityMetricsRow],
    ) -> Tuple[bool, List[Dict]]:
        """
        检查所有指标是否达到门槛。
        
        Args:
            metrics: 指标行列表
            
        Returns:
            (all_passed, failures) 元组
        """
        failures = []
        
        for row in metrics:
            if row.training_mode == "P2P":
                continue  # P2P 是基线，不检查
            
            passed, rel_perf, threshold = self.check_threshold(
                row.metric_value,
                row.metric_name,
                row.privacy_level,
            )
            
            if not passed:
                failures.append({
                    'dataset': row.dataset,
                    'task': row.task,
                    'training_mode': row.training_mode,
                    'privacy_level': row.privacy_level,
                    'metric_name': row.metric_name,
                    'metric_value': row.metric_value,
                    'relative_performance': rel_perf,
                    'threshold': threshold,
                    'gap': threshold - rel_perf,
                })
        
        return len(failures) == 0, failures
    
    def update_metrics_with_relative_performance(
        self,
        metrics: List[UtilityMetricsRow],
    ) -> List[UtilityMetricsRow]:
        """
        更新指标行的 relative_performance 字段。
        
        Args:
            metrics: 指标行列表
            
        Returns:
            更新后的指标行列表
        """
        # 首先收集 P2P 基线
        p2p_values: Dict[str, List[float]] = {}
        for row in metrics:
            if row.training_mode == "P2P":
                key = f"{row.dataset}|{row.task}|{row.metric_name}"
                if key not in p2p_values:
                    p2p_values[key] = []
                p2p_values[key].append(row.metric_value)
        
        # 计算 P2P 均值
        p2p_means = {k: np.mean(v) for k, v in p2p_values.items()}
        
        # 更新 relative_performance
        for row in metrics:
            key = f"{row.dataset}|{row.task}|{row.metric_name}"
            p2p_mean = p2p_means.get(key, 0)
            
            if p2p_mean > 0:
                row.relative_performance = row.metric_value / p2p_mean
            else:
                row.relative_performance = 0.0
            
            row.relative_to = "P2P_mean"
        
        return metrics


# =============================================================================
# CSV 和报告生成
# =============================================================================

def generate_utility_metrics_csv(
    metrics: List[UtilityMetricsRow],
    output_path: Path,
) -> Path:
    """
    生成 utility_metrics.csv。
    
    Args:
        metrics: 指标行列表
        output_path: 输出路径
        
    Returns:
        CSV 文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'dataset', 'task', 'method', 'training_mode', 'privacy_level',
        'seed', 'metric_name', 'metric_value', 'relative_to',
        'relative_performance', 'ci_low', 'ci_high', 'stat_method',
        'n_boot', 'family_id', 'alpha'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.to_dict())
    
    return output_path


def generate_utility_failure_analysis(
    failures: List[Dict],
    output_path: Path,
    include_recommendations: bool = True,
) -> Path:
    """
    生成 utility_failure_analysis.md。
    
    当 relative_performance < 门槛时生成此报告。
    
    Args:
        failures: 失败列表
        output_path: 输出路径
        include_recommendations: 是否包含建议
        
    Returns:
        报告文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Utility Failure Analysis Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Total Failures**: {len(failures)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "The following configurations failed to meet the utility threshold:",
        "",
        "| Dataset | Task | Mode | Privacy | Metric | Value | Rel.Perf | Threshold | Gap |",
        "|---------|------|------|---------|--------|-------|----------|-----------|-----|",
    ]
    
    for f in failures:
        lines.append(
            f"| {f['dataset']} | {f['task']} | {f['training_mode']} | "
            f"{f['privacy_level']:.1f} | {f['metric_name']} | "
            f"{f['metric_value']:.4f} | {f['relative_performance']:.2%} | "
            f"{f['threshold']:.2%} | {f['gap']:.2%} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Analysis",
        "",
    ])
    
    # 按训练模式分组
    by_mode: Dict[str, List[Dict]] = {}
    for f in failures:
        mode = f['training_mode']
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(f)
    
    for mode, mode_failures in by_mode.items():
        lines.extend([
            f"### {mode} Mode",
            "",
            f"**Failures**: {len(mode_failures)}",
            "",
        ])
        
        for f in mode_failures:
            lines.extend([
                f"- **{f['dataset']}/{f['task']}** at privacy={f['privacy_level']}:",
                f"  - Metric: {f['metric_name']} = {f['metric_value']:.4f}",
                f"  - Relative Performance: {f['relative_performance']:.2%}",
                f"  - Required: {f['threshold']:.2%}",
                f"  - Gap: {f['gap']:.2%}",
                "",
            ])
    
    if include_recommendations:
        lines.extend([
            "---",
            "",
            "## Recommendations",
            "",
            "1. **Reduce privacy level**: Consider using a lower privacy level for better utility.",
            "2. **Adjust budget allocation**: Review the causal budget allocation strategy.",
            "3. **Model fine-tuning**: Fine-tune the downstream model on Z-view data.",
            "4. **Data augmentation**: Apply domain-specific augmentation to bridge the gap.",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        "*Report generated by UtilityThresholdChecker*",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # Enums
    'TrainingMode',
    'ViewType',
    # Exceptions
    'ViewAccessError',
    'UtilityThresholdError',
    # Constants
    'UTILITY_THRESHOLDS',
    # Data classes
    'UtilityMetricsRow',
    'DataLoaderAuditEntry',
    # Classes
    'TrainingModeManager',
    'UtilityThresholdChecker',
    # Functions
    'generate_utility_metrics_csv',
    'generate_utility_failure_analysis',
]
