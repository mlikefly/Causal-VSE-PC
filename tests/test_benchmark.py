# -*- coding: utf-8 -*-
"""
基准测试脚本测试

Property 13: LaTeX Output Validity
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '.')


def test_property_13_latex_validity():
    """
    Property 13: LaTeX Output Validity
    
    验证 LaTeX 输出有效性
    """
    print("=" * 70)
    print("Property 13: LaTeX Output Validity")
    print("=" * 70)
    
    from scripts.run_benchmark import LaTeXGenerator
    
    # 测试效用表格
    utility_results = {
        'privacy_utility_curve': {
            'privacy_levels': [0.0, 0.3, 0.5, 0.7, 1.0],
            'utility_values': [0.95, 0.88, 0.82, 0.75, 0.65],
            'metric_name': 'Accuracy',
            'task_type': 'classification'
        }
    }
    
    utility_latex = LaTeXGenerator.generate_utility_table(utility_results)
    assert LaTeXGenerator.is_valid_latex(utility_latex)
    print("OK 效用表格 LaTeX 有效")
    
    # 验证包含必要元素
    assert '\\begin{table}' in utility_latex
    assert '\\end{table}' in utility_latex
    assert '\\begin{tabular}' in utility_latex
    assert '\\caption' in utility_latex
    assert '\\label' in utility_latex
    print("OK 效用表格包含所有必要元素")
    
    # 测试攻击表格
    attack_results = {
        'identity': {
            'top1_accuracy': 0.15,
            'top5_accuracy': 0.45
        },
        'reconstruction': {
            'psnr': 12.5,
            'ssim': 0.35
        },
        'linkability': {
            'linkage_auc': 0.52
        }
    }
    
    attack_latex = LaTeXGenerator.generate_attack_table(attack_results)
    assert LaTeXGenerator.is_valid_latex(attack_latex)
    print("OK 攻击表格 LaTeX 有效")
    
    # 验证包含攻击类型
    assert 'Identity' in attack_latex
    assert 'Reconstruction' in attack_latex
    assert 'Linkability' in attack_latex
    print("OK 攻击表格包含所有攻击类型")
    
    # 测试安全性表格
    security_results = {
        'z_view': {
            'metrics': {
                'entropy_encrypted': 7.85,
                'npcr': 99.6,
                'uaci': 33.2
            }
        },
        'c_view': {
            'metrics': {
                'entropy_encrypted': 7.95,
                'npcr': 99.8,
                'uaci': 33.5
            }
        }
    }
    
    security_latex = LaTeXGenerator.generate_security_table(security_results)
    assert LaTeXGenerator.is_valid_latex(security_latex)
    print("OK 安全性表格 LaTeX 有效")
    
    print("\nOK Property 13 测试通过")
    return True


def test_benchmark_runner():
    """测试基准测试运行器"""
    print("\n" + "=" * 70)
    print("测试基准测试运行器")
    print("=" * 70)
    
    from scripts.run_benchmark import BenchmarkRunner
    
    runner = BenchmarkRunner(output_dir='results/test_benchmark', device='cpu')
    
    # 创建测试数据
    torch.manual_seed(42)
    n_samples = 10
    
    original = torch.rand(n_samples, 3, 32, 32)
    z_view = torch.rand(n_samples, 3, 32, 32)
    c_view = torch.rand(n_samples, 3, 32, 32)
    labels = torch.randint(0, 3, (n_samples,))
    identity_labels = torch.arange(n_samples) % 3
    
    images_dict = {
        0.0: original,
        0.3: original + torch.randn_like(original) * 0.1,
        0.5: original + torch.randn_like(original) * 0.2,
        0.7: original + torch.randn_like(original) * 0.3,
        1.0: z_view
    }
    
    # 运行效用基准测试
    utility_results = runner.run_utility_benchmark(images_dict, labels)
    assert 'privacy_utility_curve' in utility_results
    print("OK 效用基准测试完成")
    
    # 运行攻击基准测试
    attack_results = runner.run_attack_benchmark(
        original, z_view,
        identity_labels=identity_labels
    )
    assert 'identity' in attack_results or 'reconstruction' in attack_results
    print("OK 攻击基准测试完成")
    
    # 运行安全性基准测试
    security_results = runner.run_security_benchmark(original, z_view, c_view)
    assert 'z_view' in security_results
    assert 'c_view' in security_results
    print("OK 安全性基准测试完成")
    
    print("\nOK 基准测试运行器测试通过")
    return True


def test_visualization_generator():
    """测试可视化生成器"""
    print("\n" + "=" * 70)
    print("测试可视化生成器")
    print("=" * 70)
    
    from scripts.run_benchmark import VisualizationGenerator
    
    # 测试隐私-效用曲线数据
    curve = {
        'privacy_levels': [0.0, 0.3, 0.5, 0.7, 1.0],
        'utility_values': [0.95, 0.88, 0.82, 0.75, 0.65],
        'metric_name': 'Accuracy',
        'task_type': 'classification'
    }
    
    plot_data = VisualizationGenerator.generate_privacy_utility_plot_data(curve)
    assert 'x' in plot_data
    assert 'y' in plot_data
    assert 'xlabel' in plot_data
    assert 'ylabel' in plot_data
    print("OK 隐私-效用曲线数据生成")
    
    # 测试 Pareto 数据
    pareto_data = VisualizationGenerator.generate_pareto_plot_data(
        utility_values=[0.9, 0.85, 0.8],
        privacy_values=[0.3, 0.5, 0.7],
        method_names=['Ours', 'Baseline1', 'Baseline2']
    )
    assert 'labels' in pareto_data
    print("OK Pareto 前沿数据生成")
    
    # 测试公平性数据
    fairness = {
        'group_accuracies': {'Male': 0.85, 'Female': 0.82, 'Other': 0.80}
    }
    bar_data = VisualizationGenerator.generate_fairness_bar_data(fairness)
    assert 'groups' in bar_data
    assert 'accuracies' in bar_data
    print("OK 公平性柱状图数据生成")
    
    # 测试对比图数据
    torch.manual_seed(42)
    original = torch.rand(1, 3, 32, 32)
    z_view = torch.rand(1, 3, 32, 32)
    c_view = torch.rand(1, 3, 32, 32)
    
    grid_data = VisualizationGenerator.generate_comparison_grid_data(
        original, z_view, c_view
    )
    assert 'original' in grid_data
    assert 'z_view' in grid_data
    assert 'c_view' in grid_data
    assert 'titles' in grid_data
    print("OK 对比图数据生成")
    
    print("\nOK 可视化生成器测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("基准测试脚本测试")
    print("=" * 70)
    
    tests = [
        ("Property 13: LaTeX Output Validity", test_property_13_latex_validity),
        ("基准测试运行器", test_benchmark_runner),
        ("可视化生成器", test_visualization_generator),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"X {name} 失败")
        except Exception as e:
            failed += 1
            print(f"X {name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed}/{passed + failed} 通过")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
