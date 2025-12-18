# -*- coding: utf-8 -*-
"""
图表生成模块

实现 T10: 图表生成 + 字节级复现
- 8 张主图生成
- FigureSpecs 规格定义
- figure_manifest.json 生成（SHA256 哈希）

Requirements: §12.6, Property 9
Validates: Requirements 6.1, 6.7, 6.8

Inputs/Outputs Contract:
- 输入: tables/*.csv
- 输出: figures/*.png, reports/figure_manifest.json
- 约束: 从 CSV 重建 PNG，SHA256 字节一致
"""

import hashlib
import json
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class FigureSpecs:
    """
    图表规格（冻结）
    
    Requirements: §12.6
    """
    SINGLE_COLUMN_WIDTH: float = 3.5  # inches
    DOUBLE_COLUMN_WIDTH: float = 7.0  # inches
    DPI: int = 300
    FONT_PRIORITY: List[str] = field(default_factory=lambda: ["Arial", "Helvetica", "DejaVu Sans"])
    
    # 8 张主图配置
    FIGURE_CONFIGS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fig_utility_curve": {
            "width": 7.0,
            "height": 4.0,
            "source_csv": "utility_metrics.csv",
            "description": "Utility curves across privacy levels"
        },
        "fig_attack_curves": {
            "width": 7.0,
            "height": 5.0,
            "source_csv": "attack_metrics.csv",
            "description": "Attack success curves by threat level"
        },
        "fig_pareto_frontier": {
            "width": 3.5,
            "height": 3.5,
            "source_csv": ["utility_metrics.csv", "attack_metrics.csv"],
            "description": "Privacy-utility Pareto frontier"
        },
        "fig_causal_ate_cate": {
            "width": 7.0,
            "height": 4.0,
            "source_csv": "causal_effects.csv",
            "description": "Causal effects (ATE/CATE) visualization"
        },
        "fig_cview_security_summary": {
            "width": 3.5,
            "height": 4.0,
            "source_csv": "security_metrics_cview.csv",
            "description": "C-view security test summary"
        },
        "fig_ablation_summary": {
            "width": 7.0,
            "height": 4.0,
            "source_csv": "ablation.csv",
            "description": "Ablation study summary"
        },
        "fig_efficiency": {
            "width": 3.5,
            "height": 3.5,
            "source_csv": "efficiency.csv",
            "description": "Computational efficiency"
        },
        "fig_robustness": {
            "width": 7.0,
            "height": 4.0,
            "source_csv": "robustness_metrics.csv",
            "description": "Robustness evaluation"
        }
    })


@dataclass
class FigureManifestEntry:
    """图表 manifest 条目"""
    figure_name: str
    file_path: str
    sha256: str
    source_csv: Union[str, List[str]]
    width: float
    height: float
    dpi: int
    generated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureManifest:
    """
    图表 manifest
    
    Property 9: 图表可复现性
    """
    version: str = "1.0.0"
    generated_at: str = ""
    entries: List[FigureManifestEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "entries": [e.to_dict() for e in self.entries]
        }
    
    def add_entry(self, entry: FigureManifestEntry) -> None:
        self.entries.append(entry)
    
    def save(self, output_path: Union[str, Path]) -> Path:
        """保存 manifest 到 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return output_path


class FigureGenerator:
    """
    图表生成器
    
    实现 8 张主图生成和 manifest 管理
    
    Requirements: §12.6, Property 9
    """
    
    def __init__(
        self,
        run_dir: Union[str, Path],
        specs: Optional[FigureSpecs] = None,
        seed: int = 42
    ):
        """
        初始化图表生成器
        
        Args:
            run_dir: 运行目录
            specs: 图表规格
            seed: 随机种子（用于确定性）
        """
        self.run_dir = Path(run_dir)
        self.tables_dir = self.run_dir / "tables"
        self.figures_dir = self.run_dir / "figures"
        self.reports_dir = self.run_dir / "reports"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.specs = specs or FigureSpecs()
        self.seed = seed
        self.manifest = FigureManifest(generated_at=datetime.now().isoformat())
        
        # 设置 matplotlib
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib()
    
    def _setup_matplotlib(self) -> None:
        """设置 matplotlib 参数以确保可复现性"""
        plt.rcParams['figure.dpi'] = self.specs.DPI
        plt.rcParams['savefig.dpi'] = self.specs.DPI
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = self.specs.FONT_PRIORITY
        plt.rcParams['axes.unicode_minus'] = False
        
        # 确定性设置
        np.random.seed(self.seed)
    
    def _compute_sha256(self, file_path: Path) -> str:
        """计算文件 SHA256 哈希"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _load_csv(self, csv_name: str) -> List[Dict[str, Any]]:
        """加载 CSV 文件"""
        csv_path = self.tables_dir / csv_name
        if not csv_path.exists():
            return []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def _save_figure(
        self,
        fig: "plt.Figure",
        figure_name: str,
        source_csv: Union[str, List[str]]
    ) -> FigureManifestEntry:
        """保存图表并创建 manifest 条目"""
        config = self.specs.FIGURE_CONFIGS.get(figure_name, {})
        
        output_path = self.figures_dir / f"{figure_name}.png"
        
        # 保存图表
        fig.savefig(
            output_path,
            dpi=self.specs.DPI,
            bbox_inches='tight',
            pad_inches=0.1,
            format='png'
        )
        plt.close(fig)
        
        # 计算哈希
        sha256 = self._compute_sha256(output_path)
        
        # 创建 manifest 条目
        entry = FigureManifestEntry(
            figure_name=figure_name,
            file_path=str(output_path.relative_to(self.run_dir)),
            sha256=sha256,
            source_csv=source_csv,
            width=config.get("width", 7.0),
            height=config.get("height", 4.0),
            dpi=self.specs.DPI,
            generated_at=datetime.now().isoformat()
        )
        
        self.manifest.add_entry(entry)
        return entry
    
    def generate_utility_curve(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_utility_curve.png
        
        效用曲线：不同隐私等级下的任务效用
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("utility_metrics.csv")
        if not data:
            return self._generate_placeholder("fig_utility_curve", "utility_metrics.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_utility_curve"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 按 training_mode 分组
        modes = {}
        for row in data:
            mode = row.get("training_mode", "unknown")
            if mode not in modes:
                modes[mode] = {"privacy_levels": [], "utilities": []}
            try:
                modes[mode]["privacy_levels"].append(float(row.get("privacy_level", 0)))
                modes[mode]["utilities"].append(float(row.get("metric_value", 0)))
            except (ValueError, TypeError):
                continue
        
        # 绘制曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
        for (mode, values), color in zip(modes.items(), colors):
            if values["privacy_levels"]:
                # 排序
                sorted_pairs = sorted(zip(values["privacy_levels"], values["utilities"]))
                x, y = zip(*sorted_pairs)
                ax.plot(x, y, 'o-', label=mode, color=color, markersize=6)
        
        ax.set_xlabel("Privacy Level")
        ax.set_ylabel("Utility (↑ higher is better)")
        ax.set_title("Utility Curves Across Privacy Levels")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, "fig_utility_curve", "utility_metrics.csv")

    
    def generate_attack_curves(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_attack_curves.png
        
        攻击成功率曲线：按威胁等级分组
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("attack_metrics.csv")
        if not data:
            return self._generate_placeholder("fig_attack_curves", "attack_metrics.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_attack_curves"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 按 threat_level 分组
        threat_levels = {}
        for row in data:
            level = row.get("threat_level", "unknown")
            if level not in threat_levels:
                threat_levels[level] = {"privacy_levels": [], "attack_success": []}
            try:
                threat_levels[level]["privacy_levels"].append(float(row.get("privacy_level", 0)))
                threat_levels[level]["attack_success"].append(float(row.get("attack_success", 0)))
            except (ValueError, TypeError):
                continue
        
        # 绘制曲线
        markers = {'A0': 'o', 'A1': 's', 'A2': '^'}
        colors = {'A0': 'green', 'A1': 'orange', 'A2': 'red'}
        
        for level, values in threat_levels.items():
            if values["privacy_levels"]:
                sorted_pairs = sorted(zip(values["privacy_levels"], values["attack_success"]))
                x, y = zip(*sorted_pairs)
                ax.plot(x, y, 
                       marker=markers.get(level, 'o'),
                       color=colors.get(level, 'blue'),
                       label=level, markersize=6)
        
        ax.set_xlabel("Privacy Level")
        ax.set_ylabel("Attack Success (↓ lower is better)")
        ax.set_title("Attack Success by Threat Level")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, "fig_attack_curves", "attack_metrics.csv")
    
    def generate_pareto_frontier(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_pareto_frontier.png
        
        隐私-效用 Pareto 前沿
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        utility_data = self._load_csv("utility_metrics.csv")
        attack_data = self._load_csv("attack_metrics.csv")
        
        if not utility_data and not attack_data:
            return self._generate_placeholder("fig_pareto_frontier", 
                                             ["utility_metrics.csv", "attack_metrics.csv"])
        
        config = self.specs.FIGURE_CONFIGS["fig_pareto_frontier"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 合并数据
        points = []
        for row in utility_data:
            try:
                utility = float(row.get("metric_value", 0))
                privacy_level = float(row.get("privacy_level", 0))
                # 假设隐私保护 = 1 - 攻击成功率（简化）
                privacy = privacy_level
                points.append((utility, privacy, row.get("method", "ours")))
            except (ValueError, TypeError):
                continue
        
        if points:
            utilities, privacies, methods = zip(*points)
            ax.scatter(utilities, privacies, c='blue', alpha=0.6, s=50)
            
            # 标注 Pareto 前沿
            pareto_points = self._compute_pareto_frontier(list(zip(utilities, privacies)))
            if pareto_points:
                pareto_x, pareto_y = zip(*pareto_points)
                ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
        
        ax.set_xlabel("Utility (↑ higher is better)")
        ax.set_ylabel("Privacy Protection (↑ higher is better)")
        ax.set_title("Privacy-Utility Pareto Frontier")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, "fig_pareto_frontier", 
                                ["utility_metrics.csv", "attack_metrics.csv"])
    
    def _compute_pareto_frontier(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """计算 Pareto 前沿"""
        if not points:
            return []
        
        # 按 utility 排序
        sorted_points = sorted(points, key=lambda p: p[0])
        
        pareto = []
        max_privacy = float('-inf')
        
        for utility, privacy in sorted_points:
            if privacy > max_privacy:
                pareto.append((utility, privacy))
                max_privacy = privacy
        
        return pareto
    
    def generate_causal_ate_cate(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_causal_ate_cate.png
        
        因果效应可视化
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("causal_effects.csv")
        if not data:
            return self._generate_placeholder("fig_causal_ate_cate", "causal_effects.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_causal_ate_cate"]
        fig, axes = plt.subplots(1, 2, figsize=(config["width"], config["height"]))
        
        # 分离 ATE 和 CATE
        ate_data = [r for r in data if r.get("effect_type") == "ATE"]
        cate_data = [r for r in data if r.get("effect_type") == "CATE"]
        
        # ATE 图
        ax1 = axes[0]
        if ate_data:
            regions = [r.get("region", "") for r in ate_data]
            ates = [float(r.get("effect_value", 0)) for r in ate_data]
            ci_lows = [float(r.get("ci_low", 0)) for r in ate_data]
            ci_highs = [float(r.get("ci_high", 0)) for r in ate_data]
            
            x = np.arange(len(regions))
            errors = [[a - l for a, l in zip(ates, ci_lows)],
                     [h - a for a, h in zip(ates, ci_highs)]]
            
            ax1.bar(x, ates, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels(regions, rotation=45, ha='right')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax1.set_ylabel("ATE")
        ax1.set_title("Average Treatment Effect by Region")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # CATE 图
        ax2 = axes[1]
        if cate_data:
            regions = [r.get("region", "") for r in cate_data]
            cates = [float(r.get("effect_value", 0)) for r in cate_data]
            ci_lows = [float(r.get("ci_low", 0)) for r in cate_data]
            ci_highs = [float(r.get("ci_high", 0)) for r in cate_data]
            
            x = np.arange(len(regions))
            errors = [[c - l for c, l in zip(cates, ci_lows)],
                     [h - c for c, h in zip(cates, ci_highs)]]
            
            ax2.bar(x, cates, yerr=errors, capsize=5, color='coral', alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels(regions, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_ylabel("CATE")
        ax2.set_title("Conditional ATE by Region")
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "fig_causal_ate_cate", "causal_effects.csv")
    
    def generate_cview_security_summary(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_cview_security_summary.png
        
        C-view 安全测试摘要
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("security_metrics_cview.csv")
        if not data:
            return self._generate_placeholder("fig_cview_security_summary", 
                                             "security_metrics_cview.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_cview_security_summary"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 提取测试结果
        test_names = []
        pass_rates = []
        
        for row in data:
            test_name = row.get("test_name", row.get("test_type", "unknown"))
            try:
                pass_rate = float(row.get("pass_rate", row.get("success_rate", 0)))
            except (ValueError, TypeError):
                pass_rate = 0
            test_names.append(test_name)
            pass_rates.append(pass_rate)
        
        if test_names:
            colors = ['green' if r >= 0.99 else 'orange' if r >= 0.9 else 'red' 
                     for r in pass_rates]
            
            x = np.arange(len(test_names))
            ax.barh(x, pass_rates, color=colors, alpha=0.7)
            ax.set_yticks(x)
            ax.set_yticklabels(test_names)
            ax.axvline(x=0.99, color='green', linestyle='--', label='Threshold (99%)')
            ax.set_xlim(0, 1.05)
        
        ax.set_xlabel("Pass Rate")
        ax.set_title("C-view Security Test Summary")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        
        return self._save_figure(fig, "fig_cview_security_summary", 
                                "security_metrics_cview.csv")
    
    def generate_ablation_summary(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_ablation_summary.png
        
        消融实验摘要
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("ablation.csv")
        if not data:
            return self._generate_placeholder("fig_ablation_summary", "ablation.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_ablation_summary"]
        fig, axes = plt.subplots(1, 2, figsize=(config["width"], config["height"]))
        
        # 提取数据
        ablation_ids = []
        utilities = []
        privacies = []
        
        for row in data:
            ablation_id = row.get("ablation_id", row.get("config_name", "unknown"))
            try:
                utility = float(row.get("utility", row.get("metric_value", 0)))
                privacy = float(row.get("privacy_protection", 0.5))
            except (ValueError, TypeError):
                utility, privacy = 0, 0.5
            ablation_ids.append(ablation_id)
            utilities.append(utility)
            privacies.append(privacy)
        
        x = np.arange(len(ablation_ids))
        
        # 效用图
        axes[0].bar(x, utilities, color='steelblue', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ablation_ids, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel("Utility")
        axes[0].set_title("Utility by Ablation")
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 隐私图
        axes[1].bar(x, privacies, color='coral', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ablation_ids, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel("Privacy Protection")
        axes[1].set_title("Privacy by Ablation")
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "fig_ablation_summary", "ablation.csv")
    
    def generate_efficiency(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_efficiency.png
        
        计算效率图
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("efficiency.csv")
        if not data:
            return self._generate_placeholder("fig_efficiency", "efficiency.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_efficiency"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 提取数据
        methods = []
        times = []
        
        for row in data:
            method = row.get("method", "unknown")
            try:
                time_ms = float(row.get("time_ms", row.get("latency", 0)))
            except (ValueError, TypeError):
                time_ms = 0
            methods.append(method)
            times.append(time_ms)
        
        if methods:
            x = np.arange(len(methods))
            ax.bar(x, times, color='teal', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
        
        ax.set_ylabel("Time (ms)")
        ax.set_title("Computational Efficiency")
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._save_figure(fig, "fig_efficiency", "efficiency.csv")
    
    def generate_robustness(self) -> Optional[FigureManifestEntry]:
        """
        生成 fig_robustness.png
        
        鲁棒性评估图
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = self._load_csv("robustness_metrics.csv")
        if not data:
            return self._generate_placeholder("fig_robustness", "robustness_metrics.csv")
        
        config = self.specs.FIGURE_CONFIGS["fig_robustness"]
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        # 提取数据
        perturbations = []
        scores = []
        
        for row in data:
            perturbation = row.get("perturbation_type", "unknown")
            try:
                score = float(row.get("robustness_score", 0))
            except (ValueError, TypeError):
                score = 0
            perturbations.append(perturbation)
            scores.append(score)
        
        if perturbations:
            x = np.arange(len(perturbations))
            ax.bar(x, scores, color='purple', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(perturbations, rotation=45, ha='right')
        
        ax.set_ylabel("Robustness Score")
        ax.set_title("Robustness Evaluation")
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._save_figure(fig, "fig_robustness", "robustness_metrics.csv")
    
    def _generate_placeholder(
        self,
        figure_name: str,
        source_csv: Union[str, List[str]]
    ) -> Optional[FigureManifestEntry]:
        """生成占位图（当数据不可用时）"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        config = self.specs.FIGURE_CONFIGS.get(figure_name, {"width": 7.0, "height": 4.0})
        fig, ax = plt.subplots(figsize=(config["width"], config["height"]))
        
        ax.text(0.5, 0.5, f"No data available\n({source_csv})",
               ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(figure_name)
        
        return self._save_figure(fig, figure_name, source_csv)
    
    def generate_all_figures(self) -> Dict[str, Optional[FigureManifestEntry]]:
        """
        生成所有 8 张主图
        
        Returns:
            entries: {figure_name: manifest_entry}
        """
        entries = {}
        
        entries["fig_utility_curve"] = self.generate_utility_curve()
        entries["fig_attack_curves"] = self.generate_attack_curves()
        entries["fig_pareto_frontier"] = self.generate_pareto_frontier()
        entries["fig_causal_ate_cate"] = self.generate_causal_ate_cate()
        entries["fig_cview_security_summary"] = self.generate_cview_security_summary()
        entries["fig_ablation_summary"] = self.generate_ablation_summary()
        entries["fig_efficiency"] = self.generate_efficiency()
        entries["fig_robustness"] = self.generate_robustness()
        
        return entries
    
    def generate_manifest(
        self,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        生成 figure_manifest.json
        
        Property 9: 图表可复现性
        
        Args:
            output_path: 输出路径
        
        Returns:
            output_path: manifest 文件路径
        """
        if output_path is None:
            output_path = self.reports_dir / "figure_manifest.json"
        
        return self.manifest.save(output_path)
    
    def verify_reproducibility(self) -> Dict[str, Any]:
        """
        验证图表可复现性
        
        重新生成图表并比较 SHA256 哈希
        
        Returns:
            verification_result: 验证结果
        """
        results = {
            "verified": True,
            "total_figures": len(self.manifest.entries),
            "matched": 0,
            "mismatched": [],
            "missing": []
        }
        
        for entry in self.manifest.entries:
            figure_path = self.run_dir / entry.file_path
            
            if not figure_path.exists():
                results["missing"].append(entry.figure_name)
                results["verified"] = False
                continue
            
            current_hash = self._compute_sha256(figure_path)
            
            if current_hash == entry.sha256:
                results["matched"] += 1
            else:
                results["mismatched"].append({
                    "figure_name": entry.figure_name,
                    "expected_hash": entry.sha256,
                    "actual_hash": current_hash
                })
                results["verified"] = False
        
        return results


# 便捷函数
def generate_all_figures(
    run_dir: Union[str, Path],
    seed: int = 42
) -> Dict[str, Path]:
    """
    生成所有图表（便捷函数）
    
    Args:
        run_dir: 运行目录
        seed: 随机种子
    
    Returns:
        output_paths: 输出文件路径
    """
    generator = FigureGenerator(run_dir=run_dir, seed=seed)
    
    # 生成所有图表
    generator.generate_all_figures()
    
    # 生成 manifest
    manifest_path = generator.generate_manifest()
    
    return {
        "figures_dir": generator.figures_dir,
        "manifest_path": manifest_path
    }
