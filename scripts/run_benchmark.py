#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基准测试脚本 (Benchmark Runner)

统一运行 Utility/Attack/Causal 三大评估模块，输出结构化结果

Requirements: 8.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.utility_evaluator import (
    UtilityEvaluator,
    PrivacyUtilityCurve,
    ClassificationResult,
    FairnessResult
)
from src.evaluation.attack_evaluator import (
    AttackEvaluator,
    AttackEvaluationMatrix
)
from src.evaluation.dual_view_metrics import (
    DualViewSecurityMetrics,
    ZViewMetrics,
    CViewMetrics
)


class BenchmarkRunner:
    """基准测试运行器"""
    
    PRIVACY_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    def __init__(
        self,
        output_dir: str = "results/benchmark",
        device: str = None
    ):
        """
        初始化基准测试运行器
        
        Args:
            output_dir: 输出目录
            device: 计算设备
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化评估器
        self.utility_evaluator = UtilityEvaluator(device=self.device)
        self.attack_evaluator = AttackEvaluator(device=self.device)
        self.security_evaluator = DualViewSecurityMetrics()
        
        # 结果存储
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'privacy_levels': self.PRIVACY_LEVELS
            },
            'utility': {},
            'attack': {},
            'security': {},
            'causal': {}
        }
    
    def run_utility_benchmark(
        self,
        images_dict: Dict[float, torch.Tensor],
        labels: torch.Tensor,
        task_type: str = "classification",
        group_labels: torch.Tensor = None
    ) -> Dict:
        """
        运行效用基准测试
        
        Args:
            images_dict: {privacy_level: images}
            labels: 标签
            task_type: 任务类型
            group_labels: 分组标签（用于公平性评估）
        
        Returns:
            效用评估结果
        """
        print("\n" + "=" * 70)
        print("效用基准测试")
        print("=" * 70)
        
        results = {}
        
        # 1. 隐私-效用曲线
        curve = self.utility_evaluator.evaluate_at_privacy_levels(
            images_dict, labels, task_type
        )
        results['privacy_utility_curve'] = curve.to_dict()
        
        print(f"\n隐私-效用曲线:")
        for level, utility in zip(curve.privacy_levels, curve.utility_values):
            print(f"  {level:.1f}: {utility:.4f}")
        
        # 2. 各隐私级别详细评估
        results['per_level'] = {}
        for level, images in images_dict.items():
            if task_type == "classification":
                result = self.utility_evaluator.evaluate_classification(images, labels)
                results['per_level'][str(level)] = result.to_dict()
        
        # 3. 公平性评估（如果有分组标签）
        if group_labels is not None and 0.0 in images_dict:
            fairness = self.utility_evaluator.evaluate_fairness(
                images_dict[0.0], labels, group_labels
            )
            results['fairness'] = fairness.to_dict()
            print(f"\n公平性评估:")
            print(f"  准确率差距: {fairness.accuracy_gap:.4f}")
        
        self.results['utility'] = results
        return results
    
    def run_attack_benchmark(
        self,
        original_images: torch.Tensor,
        encrypted_images: torch.Tensor,
        identity_labels: torch.Tensor = None,
        attribute_labels: Dict[str, torch.Tensor] = None,
        sensitive_mask: torch.Tensor = None
    ) -> Dict:
        """
        运行攻击基准测试
        
        Args:
            original_images: 原始图像
            encrypted_images: 加密图像
            identity_labels: 身份标签
            attribute_labels: 属性标签
            sensitive_mask: 敏感区域掩码
        
        Returns:
            攻击评估结果
        """
        print("\n" + "=" * 70)
        print("攻击基准测试")
        print("=" * 70)
        
        matrix = self.attack_evaluator.evaluate_all(
            original_images=original_images,
            encrypted_images=encrypted_images,
            identity_labels=identity_labels,
            attribute_labels=attribute_labels,
            sensitive_mask=sensitive_mask
        )
        
        results = matrix.to_dict()
        
        if matrix.identity:
            print(f"\n身份识别攻击:")
            print(f"  Top-1: {matrix.identity.top1_accuracy:.4f}")
            print(f"  Top-5: {matrix.identity.top5_accuracy:.4f}")
        
        if matrix.reconstruction:
            print(f"\n重建攻击:")
            print(f"  PSNR: {matrix.reconstruction.psnr:.2f} dB")
            print(f"  SSIM: {matrix.reconstruction.ssim:.4f}")
        
        if matrix.linkability:
            print(f"\n可链接性攻击:")
            print(f"  AUC: {matrix.linkability.linkage_auc:.4f}")
        
        self.results['attack'] = results
        return results
    
    def run_security_benchmark(
        self,
        original_images: torch.Tensor,
        z_view: torch.Tensor,
        c_view: torch.Tensor
    ) -> Dict:
        """
        运行安全性基准测试
        
        Args:
            original_images: 原始图像
            z_view: Z-view 密文
            c_view: C-view 密文
        
        Returns:
            安全性评估结果
        """
        print("\n" + "=" * 70)
        print("安全性基准测试")
        print("=" * 70)
        
        results = {}
        
        # Z-view 评估
        z_metrics = self.security_evaluator.evaluate_zview(original_images, z_view)
        z_checks = self.security_evaluator.check_zview_standards(z_metrics)
        results['z_view'] = {
            'metrics': z_metrics.to_dict(),
            'checks': z_checks,
            'passed': sum(z_checks.values()),
            'total': len(z_checks)
        }
        
        print(f"\nZ-view 安全性:")
        print(f"  熵: {z_metrics.entropy_encrypted:.4f}")
        print(f"  NPCR: {z_metrics.npcr:.2f}%")
        print(f"  通过: {sum(z_checks.values())}/{len(z_checks)}")
        
        # C-view 评估
        c_metrics = self.security_evaluator.evaluate_cview(original_images, c_view)
        c_checks = self.security_evaluator.check_cview_standards(c_metrics)
        results['c_view'] = {
            'metrics': c_metrics.to_dict(),
            'checks': c_checks,
            'passed': sum(c_checks.values()),
            'total': len(c_checks)
        }
        
        print(f"\nC-view 安全性:")
        print(f"  熵: {c_metrics.entropy_encrypted:.4f}")
        print(f"  NPCR: {c_metrics.npcr:.2f}%")
        print(f"  NIST monobit: {'Pass' if c_metrics.nist_monobit_pass else 'Fail'}")
        print(f"  通过: {sum(c_checks.values())}/{len(c_checks)}")
        
        self.results['security'] = results
        return results
    
    def save_results(self, filename: str = None):
        """保存结果到 JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")
        return output_path


class LaTeXGenerator:
    """LaTeX 表格生成器"""
    
    @staticmethod
    def generate_utility_table(results: Dict) -> str:
        """
        生成效用指标 LaTeX 表格
        
        **Requirements 10.1**: 效用指标表格
        """
        curve = results.get('privacy_utility_curve', {})
        levels = curve.get('privacy_levels', [])
        values = curve.get('utility_values', [])
        metric = curve.get('metric_name', 'Accuracy')
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Privacy-Utility Trade-off}\n"
        latex += "\\begin{tabular}{c" + "c" * len(levels) + "}\n"
        latex += "\\toprule\n"
        
        # 表头
        latex += "Privacy Level"
        for level in levels:
            latex += f" & {level:.1f}"
        latex += " \\\\\n"
        latex += "\\midrule\n"
        
        # 数据行
        latex += metric
        for value in values:
            latex += f" & {value:.4f}"
        latex += " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\label{tab:privacy-utility}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def generate_attack_table(results: Dict) -> str:
        """
        生成攻击成功率 LaTeX 表格
        
        **Requirements 10.2**: 攻击成功率表格
        """
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Attack Evaluation Results}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Attack Type & Metric & Value \\\\\n"
        latex += "\\midrule\n"
        
        # 身份识别攻击
        if 'identity' in results:
            identity = results['identity']
            latex += f"Identity Recognition & Top-1 Acc & {identity.get('top1_accuracy', 0):.4f} \\\\\n"
            latex += f" & Top-5 Acc & {identity.get('top5_accuracy', 0):.4f} \\\\\n"
        
        # 重建攻击
        if 'reconstruction' in results:
            recon = results['reconstruction']
            latex += f"Reconstruction & PSNR (dB) & {recon.get('psnr', 0):.2f} \\\\\n"
            latex += f" & SSIM & {recon.get('ssim', 0):.4f} \\\\\n"
        
        # 可链接性攻击
        if 'linkability' in results:
            link = results['linkability']
            latex += f"Linkability & AUC & {link.get('linkage_auc', 0):.4f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\label{tab:attack-results}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def generate_security_table(results: Dict) -> str:
        """生成安全性指标 LaTeX 表格"""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Security Metrics Comparison}\n"
        latex += "\\begin{tabular}{lccc}\n"
        latex += "\\toprule\n"
        latex += "Metric & Z-view & C-view & Standard \\\\\n"
        latex += "\\midrule\n"
        
        z_metrics = results.get('z_view', {}).get('metrics', {})
        c_metrics = results.get('c_view', {}).get('metrics', {})
        
        latex += f"Entropy & {z_metrics.get('entropy_encrypted', 0):.4f} & {c_metrics.get('entropy_encrypted', 0):.4f} & $\\geq 7.9$ \\\\\n"
        latex += f"NPCR (\\%) & {z_metrics.get('npcr', 0):.2f} & {c_metrics.get('npcr', 0):.2f} & $\\geq 99.5$ \\\\\n"
        latex += f"UACI (\\%) & {z_metrics.get('uaci', 0):.2f} & {c_metrics.get('uaci', 0):.2f} & $30$-$36$ \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\label{tab:security-metrics}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def is_valid_latex(latex_str: str) -> bool:
        """
        检查 LaTeX 输出有效性
        
        **Property 13: LaTeX Output Validity**
        """
        # 检查基本结构
        checks = [
            '\\begin{table}' in latex_str,
            '\\end{table}' in latex_str,
            '\\begin{tabular}' in latex_str,
            '\\end{tabular}' in latex_str,
            '\\toprule' in latex_str,
            '\\bottomrule' in latex_str
        ]
        return all(checks)


class VisualizationGenerator:
    """可视化图表生成器"""
    
    @staticmethod
    def generate_privacy_utility_plot_data(curve: Dict) -> Dict:
        """
        生成隐私-效用曲线绘图数据
        
        **Requirements 10.3**: ATE/CATE 曲线图
        """
        return {
            'x': curve.get('privacy_levels', []),
            'y': curve.get('utility_values', []),
            'xlabel': 'Privacy Level',
            'ylabel': curve.get('metric_name', 'Utility'),
            'title': f"Privacy-Utility Curve ({curve.get('task_type', '')})"
        }
    
    @staticmethod
    def generate_pareto_plot_data(
        utility_values: List[float],
        privacy_values: List[float],
        method_names: List[str]
    ) -> Dict:
        """
        生成 Pareto 前沿图数据
        
        **Requirements 10.4**: Pareto 前沿图
        """
        return {
            'x': privacy_values,
            'y': utility_values,
            'labels': method_names,
            'xlabel': 'Privacy Protection',
            'ylabel': 'Task Utility',
            'title': 'Privacy-Utility Pareto Frontier'
        }
    
    @staticmethod
    def generate_fairness_bar_data(fairness: Dict) -> Dict:
        """
        生成公平性分组柱状图数据
        
        **Requirements 10.5**: 公平性分组柱状图
        """
        group_acc = fairness.get('group_accuracies', {})
        return {
            'groups': list(group_acc.keys()),
            'accuracies': list(group_acc.values()),
            'xlabel': 'Demographic Group',
            'ylabel': 'Accuracy',
            'title': 'Fairness Analysis by Group'
        }
    
    @staticmethod
    def generate_comparison_grid_data(
        original: torch.Tensor,
        z_view: torch.Tensor,
        c_view: torch.Tensor,
        reconstructed: torch.Tensor = None
    ) -> Dict:
        """
        生成并排对比图数据
        
        **Requirements 10.6**: 明文/Z-view/C-view/重建图对比
        """
        data = {
            'original': original.cpu().numpy() if isinstance(original, torch.Tensor) else original,
            'z_view': z_view.cpu().numpy() if isinstance(z_view, torch.Tensor) else z_view,
            'c_view': c_view.cpu().numpy() if isinstance(c_view, torch.Tensor) else c_view,
            'titles': ['Original', 'Z-view', 'C-view']
        }
        
        if reconstructed is not None:
            data['reconstructed'] = reconstructed.cpu().numpy() if isinstance(reconstructed, torch.Tensor) else reconstructed
            data['titles'].append('Reconstructed')
        
        return data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行基准测试')
    parser.add_argument('--output-dir', type=str, default='results/benchmark',
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备')
    parser.add_argument('--demo', action='store_true',
                        help='运行演示模式')
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        device=args.device
    )
    
    if args.demo:
        print("运行演示模式...")
        
        # 创建演示数据
        torch.manual_seed(42)
        n_samples = 20
        
        original = torch.rand(n_samples, 3, 64, 64)
        z_view = torch.rand(n_samples, 3, 64, 64)
        c_view = torch.rand(n_samples, 3, 64, 64)
        labels = torch.randint(0, 5, (n_samples,))
        identity_labels = torch.arange(n_samples) % 4
        
        images_dict = {
            0.0: original,
            0.3: original + torch.randn_like(original) * 0.1,
            0.5: original + torch.randn_like(original) * 0.2,
            0.7: original + torch.randn_like(original) * 0.3,
            1.0: z_view
        }
        
        # 运行基准测试
        runner.run_utility_benchmark(images_dict, labels)
        runner.run_attack_benchmark(
            original, z_view,
            identity_labels=identity_labels,
            attribute_labels={'gender': torch.randint(0, 2, (n_samples,))}
        )
        runner.run_security_benchmark(original, z_view, c_view)
        
        # 保存结果
        runner.save_results()
        
        # 生成 LaTeX 表格
        print("\n" + "=" * 70)
        print("LaTeX 表格")
        print("=" * 70)
        
        utility_latex = LaTeXGenerator.generate_utility_table(runner.results['utility'])
        print("\n效用表格:")
        print(utility_latex)
        
        attack_latex = LaTeXGenerator.generate_attack_table(runner.results['attack'])
        print("\n攻击表格:")
        print(attack_latex)
        
        # 验证 LaTeX 有效性
        print(f"\n效用表格有效性: {LaTeXGenerator.is_valid_latex(utility_latex)}")
        print(f"攻击表格有效性: {LaTeXGenerator.is_valid_latex(attack_latex)}")
    
    else:
        print("请使用 --demo 运行演示模式，或提供数据路径")


if __name__ == "__main__":
    main()
