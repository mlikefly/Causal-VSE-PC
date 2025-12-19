# -*- coding: utf-8 -*-
"""
统计引擎测试

**Property 11: 统计严谨性完整性**
**Property 16: Family ID 确定性**
**Validates: Requirements 10.1, 10.4, 10.5, 10.6, GC3, GC9**
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_bootstrap_ci_basic():
    """测试基本 Bootstrap CI 计算"""
    print("=" * 70)
    print("测试基本 Bootstrap CI 计算")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import compute_bootstrap_ci, MIN_N_BOOT
    
    # 生成测试数据
    np.random.seed(42)
    values = np.random.normal(0.5, 0.1, 100)
    
    # 计算 CI
    result = compute_bootstrap_ci(values, n_boot=MIN_N_BOOT)
    
    assert result.mean is not None
    assert result.ci_low is not None
    assert result.ci_high is not None
    assert result.ci_low < result.mean < result.ci_high
    assert result.n_boot >= MIN_N_BOOT
    
    print(f"✓ Mean: {result.mean:.4f}")
    print(f"✓ CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"✓ n_boot: {result.n_boot}")
    print("✓ Bootstrap CI 基本测试通过\n")


def test_family_id_generation():
    """测试 Family ID 生成"""
    print("=" * 70)
    print("测试 Family ID 生成")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import generate_family_id
    
    # 测试确定性
    family_id_1 = generate_family_id(
        dataset="celeba",
        task="classification",
        metric_name="accuracy",
        privacy_level=0.5
    )
    
    family_id_2 = generate_family_id(
        dataset="celeba",
        task="classification",
        metric_name="accuracy",
        privacy_level=0.5
    )
    
    assert family_id_1 == family_id_2, "相同输入应产生相同 family_id"
    assert len(family_id_1) == 10, "family_id 应为 10 字符"
    
    print(f"✓ Family ID: {family_id_1}")
    print(f"✓ 长度: {len(family_id_1)}")
    print("✓ Family ID 生成测试通过\n")


def test_family_id_uniqueness():
    """测试 Family ID 唯一性"""
    print("=" * 70)
    print("测试 Family ID 唯一性")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import generate_family_id
    
    # 不同输入应产生不同 family_id
    family_ids = set()
    
    for dataset in ["celeba", "fairface"]:
        for task in ["classification", "detection"]:
            for metric in ["accuracy", "auc"]:
                for level in [0.3, 0.5, 0.7]:
                    fid = generate_family_id(dataset, task, metric, level)
                    family_ids.add(fid)
    
    expected_count = 2 * 2 * 2 * 3  # 24 个组合
    assert len(family_ids) == expected_count, \
        f"应有 {expected_count} 个唯一 family_id，实际 {len(family_ids)}"
    
    print(f"✓ 生成了 {len(family_ids)} 个唯一 family_id")
    print("✓ Family ID 唯一性测试通过\n")


def test_bh_fdr_correction():
    """测试 BH-FDR 多重比较校正"""
    print("=" * 70)
    print("测试 BH-FDR 多重比较校正")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import bh_fdr_correction
    
    # 测试 p 值
    p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5]
    
    # 校正
    result = bh_fdr_correction(p_values, alpha=0.05)
    
    assert len(result.p_adjusted) == len(p_values)
    assert len(result.rejected) == len(p_values)
    
    # 校正后的 p 值应该 >= 原始 p 值
    for orig, adj in zip(p_values, result.p_adjusted):
        assert adj >= orig, "校正后 p 值应 >= 原始 p 值"
    
    print(f"✓ 原始 p 值: {p_values}")
    print(f"✓ 校正后 p 值: {[f'{p:.4f}' for p in result.p_adjusted]}")
    print(f"✓ 拒绝原假设: {result.rejected}")
    print("✓ BH-FDR 校正测试通过\n")


def test_statistics_engine_compute_ci():
    """测试 StatisticsEngine.compute_ci()"""
    print("=" * 70)
    print("测试 StatisticsEngine.compute_ci()")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import StatisticsEngine
    
    engine = StatisticsEngine(n_boot=500, alpha=0.05, ci_level=0.95)
    
    # 生成测试数据
    np.random.seed(42)
    values = np.random.normal(0.7, 0.05, 50)
    
    # 计算 CI
    result = engine.compute_ci(values)
    
    assert result.mean is not None
    assert result.ci_low < result.ci_high
    assert result.stat_method == "bootstrap"
    assert result.n_boot >= 500
    
    print(f"✓ Mean: {result.mean:.4f}")
    print(f"✓ CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"✓ stat_method: {result.stat_method}")
    print("✓ StatisticsEngine.compute_ci() 测试通过\n")


def test_property_11_statistical_rigor():
    """
    **Property 11: 统计严谨性完整性**
    
    验证所有核心指标都有 CI
    """
    print("=" * 70)
    print("Property 11: 统计严谨性完整性")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import StatisticsEngine, MIN_N_BOOT
    
    engine = StatisticsEngine(n_boot=MIN_N_BOOT)
    
    # 模拟多个指标的数据
    np.random.seed(42)
    metrics_data = {
        'accuracy': np.random.normal(0.85, 0.03, 30),
        'auc': np.random.normal(0.92, 0.02, 30),
        'f1_score': np.random.normal(0.80, 0.04, 30),
    }
    
    # 验证每个指标都能计算 CI
    for metric_name, values in metrics_data.items():
        result = engine.compute_ci(values)
        
        assert result.ci_low is not None, f"{metric_name} 缺少 ci_low"
        assert result.ci_high is not None, f"{metric_name} 缺少 ci_high"
        assert result.n_boot >= MIN_N_BOOT, f"{metric_name} n_boot 不足"
        
        print(f"✓ {metric_name}: {result.mean:.4f} [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    
    print("✓ Property 11 测试通过\n")


def test_property_16_family_id_determinism():
    """
    **Property 16: Family ID 确定性**
    
    验证相同输入产生相同 family_id
    """
    print("=" * 70)
    print("Property 16: Family ID 确定性")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import generate_family_id
    
    # 多次生成相同输入的 family_id
    results = []
    for _ in range(10):
        fid = generate_family_id(
            dataset="celeba",
            task="classification",
            metric_name="accuracy",
            privacy_level=0.5
        )
        results.append(fid)
    
    # 验证所有结果相同
    assert len(set(results)) == 1, "相同输入应产生相同 family_id"
    
    print(f"✓ 10 次生成结果: {results[0]}")
    print(f"✓ 唯一值数量: {len(set(results))}")
    print("✓ Property 16 测试通过\n")


def test_statistics_validator():
    """测试统计验证器"""
    print("=" * 70)
    print("测试统计验证器")
    print("=" * 70)
    
    from src.evaluation.statistics_engine import StatisticsValidator
    
    validator = StatisticsValidator()
    
    # 有效的统计记录
    valid_record = {
        'stat_method': 'bootstrap',
        'n_boot': 500,
        'ci_low': 0.75,
        'ci_high': 0.85,
        'alpha': 0.05,
        'family_id': 'abc1234567'
    }
    
    is_valid, errors = validator.validate_record(valid_record)
    assert is_valid, f"有效记录应通过验证: {errors}"
    
    print(f"✓ 有效记录验证通过")
    
    # 无效的统计记录（缺少字段）
    invalid_record = {
        'stat_method': 'bootstrap',
        'n_boot': 100,  # 不足 500
    }
    
    is_valid, errors = validator.validate_record(invalid_record)
    assert not is_valid, "无效记录应验证失败"
    
    print(f"✓ 无效记录验证失败: {errors}")
    print("✓ 统计验证器测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("统计引擎测试")
    print("=" * 70 + "\n")
    
    test_bootstrap_ci_basic()
    test_family_id_generation()
    test_family_id_uniqueness()
    test_bh_fdr_correction()
    test_statistics_engine_compute_ci()
    test_property_11_statistical_rigor()
    test_property_16_family_id_determinism()
    test_statistics_validator()
    
    print("=" * 70)
    print("✓ 所有统计引擎测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
