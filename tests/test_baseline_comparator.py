# -*- coding: utf-8 -*-
"""
Baseline 对标模块测试
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '.')


def test_instahide_baseline():
    """测试 InstaHide baseline"""
    print("=" * 70)
    print("测试 InstaHide Baseline")
    print("=" * 70)
    
    from src.evaluation.baseline_comparator import InstaHideBaseline
    
    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    
    baseline = InstaHideBaseline(k=4, device='cpu')
    encrypted, info = baseline.encrypt(images)
    
    assert encrypted.shape == images.shape
    print(f"OK 加密形状正确: {encrypted.shape}")
    
    assert info['method'] == 'instahide'
    print(f"OK 方法标识正确: {info['method']}")
    
    result = baseline.evaluate(images, encrypted)
    assert 0 <= result.utility_score <= 1
    assert 0 <= result.privacy_score <= 1
    print(f"OK 效用分数: {result.utility_score:.4f}")
    print(f"OK 隐私分数: {result.privacy_score:.4f}")
    
    print("\nOK InstaHide 测试通过")
    return True


def test_p3_baseline():
    """测试 P3 baseline"""
    print("\n" + "=" * 70)
    print("测试 P3 Baseline")
    print("=" * 70)
    
    from src.evaluation.baseline_comparator import P3Baseline
    
    torch.manual_seed(42)
    images = torch.rand(4, 3, 64, 64)
    privacy_mask = torch.zeros(4, 1, 64, 64)
    privacy_mask[:, :, 16:48, 16:48] = 1  # 中心区域为隐私区域
    
    baseline = P3Baseline(device='cpu')
    encrypted, info = baseline.encrypt(images, privacy_mask)
    
    assert encrypted.shape == images.shape
    print(f"OK 加密形状正确: {encrypted.shape}")
    
    result = baseline.evaluate(images, encrypted, privacy_mask)
    assert 0 <= result.utility_score <= 1
    assert 0 <= result.privacy_score <= 1
    print(f"OK 效用分数: {result.utility_score:.4f}")
    print(f"OK 隐私分数: {result.privacy_score:.4f}")
    
    print("\nOK P3 测试通过")
    return True


def test_chaotic_baseline():
    """测试混沌加密 baseline"""
    print("\n" + "=" * 70)
    print("测试混沌加密 Baseline")
    print("=" * 70)
    
    from src.evaluation.baseline_comparator import ChaoticBaseline
    
    torch.manual_seed(42)
    images = torch.rand(2, 3, 32, 32)
    
    # Arnold
    arnold = ChaoticBaseline(method='arnold', device='cpu')
    enc_arnold, info = arnold.encrypt(images, iterations=3)
    assert enc_arnold.shape == images.shape
    print(f"OK Arnold 加密形状正确")
    
    result_arnold = arnold.evaluate(images, enc_arnold)
    print(f"OK Arnold 隐私分数: {result_arnold.privacy_score:.4f}")
    
    # Logistic
    logistic = ChaoticBaseline(method='logistic', device='cpu')
    enc_logistic, info = logistic.encrypt(images)
    assert enc_logistic.shape == images.shape
    print(f"OK Logistic 加密形状正确")
    
    result_logistic = logistic.evaluate(images, enc_logistic)
    print(f"OK Logistic 隐私分数: {result_logistic.privacy_score:.4f}")
    
    print("\nOK 混沌加密测试通过")
    return True


def test_baseline_comparator():
    """测试 Baseline 对标比较器"""
    print("\n" + "=" * 70)
    print("测试 Baseline 对标比较器")
    print("=" * 70)
    
    from src.evaluation.baseline_comparator import BaselineComparator, BaselineResult
    
    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    privacy_mask = torch.ones(4, 1, 32, 32) * 0.5
    
    # 模拟我们方法的结果
    our_result = BaselineResult(
        method_name='Ours (VSE-PC)',
        utility_score=0.85,
        privacy_score=0.90,
        efficiency_score=0.75,
        attack_resistance={
            'identity': 0.15,
            'reconstruction': 0.20,
            'attribute': 0.25
        }
    )
    
    comparator = BaselineComparator(device='cpu')
    results = comparator.run_comparison(images, privacy_mask, our_result)
    
    assert len(results) >= 4  # 至少 4 个方法
    print(f"OK 对比方法数: {len(results)}")
    
    # 计算 Pareto 前沿
    pareto_points = comparator.compute_pareto_frontier()
    assert len(pareto_points) == len(results)
    print(f"OK Pareto 点数: {len(pareto_points)}")
    
    # 检查是否有 Pareto 最优
    optimal_count = sum(1 for p in pareto_points if p.is_pareto_optimal)
    print(f"OK Pareto 最优数: {optimal_count}")
    
    # 生成 LaTeX 表格
    latex = comparator.generate_comparison_table()
    assert '\\begin{table}' in latex
    assert '\\end{table}' in latex
    print("OK LaTeX 表格生成成功")
    
    # 打印报告
    comparator.print_report()
    
    print("\nOK Baseline 对标比较器测试通过")
    return True


def test_experiment_tracker():
    """测试实验配置记录器"""
    print("\n" + "=" * 70)
    print("测试实验配置记录器")
    print("=" * 70)
    
    from src.utils.experiment_tracker import ExperimentTracker, set_global_seed
    
    # 测试全局种子设置
    set_global_seed(42)
    
    # 验证种子生效
    val1 = torch.rand(1).item()
    set_global_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2, "随机种子未正确固定"
    print("OK 随机种子固定成功")
    
    # 测试实验记录器
    tracker = ExperimentTracker(
        experiment_name="test_experiment",
        output_dir="results/test_experiments"
    )
    
    tracker.set_seed(42)
    print("OK 实验种子设置成功")
    
    # 记录数据版本
    data_versions = tracker.record_data_version(['configs/benchmark.yaml'])
    assert len(data_versions) > 0
    print(f"OK 数据版本记录: {len(data_versions)} 个文件")
    
    # 打印摘要
    tracker.print_summary()
    
    print("\nOK 实验配置记录器测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("Baseline 对标与实验配置测试")
    print("=" * 70)
    
    tests = [
        ("InstaHide Baseline", test_instahide_baseline),
        ("P3 Baseline", test_p3_baseline),
        ("混沌加密 Baseline", test_chaotic_baseline),
        ("Baseline 对标比较器", test_baseline_comparator),
        ("实验配置记录器", test_experiment_tracker),
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
