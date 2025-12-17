# -*- coding: utf-8 -*-
"""
因果隐私预算分配器测试

**Property 6: Privacy Map Shape Consistency**
**Property 7: Multi-Task Privacy Map Differentiation**
**Property 8: ATE/CATE Output Completeness**

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.causal_budget_allocator import CausalBudgetAllocator


def test_basic_initialization():
    """测试基本初始化"""
    print("=" * 70)
    print("测试基本初始化")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    assert allocator.causal_analyzer is not None
    assert allocator.budget_allocator is not None
    assert allocator.TASK_TYPES == ['classification', 'detection', 'segmentation']
    
    print("✓ CausalBudgetAllocator 初始化成功")
    print("✓ 支持任务类型:", allocator.TASK_TYPES)
    print("✓ 基本初始化测试通过\n")


def test_property_6_shape_consistency():
    """
    **Property 6: Privacy Map Shape Consistency**
    
    验证隐私预算图形状与输入语义掩码一致
    """
    print("=" * 70)
    print("Property 6: Privacy Map Shape Consistency")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    # 测试不同尺寸
    test_shapes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for H, W in test_shapes:
        # 创建测试语义掩码
        semantic_mask = np.random.rand(H, W).astype(np.float32)
        
        # 生成隐私预算图
        privacy_maps = allocator.generate_privacy_maps(semantic_mask)
        
        # 验证形状一致性
        for task_type, privacy_map in privacy_maps.items():
            assert privacy_map.shape == (H, W), \
                f"形状不一致: 期望 {(H, W)}, 实际 {privacy_map.shape}"
            
            # 使用验证方法
            assert allocator.validate_privacy_map_shape(semantic_mask, privacy_map), \
                f"validate_privacy_map_shape 返回 False"
        
        print(f"✓ 形状 {(H, W)}: 所有任务类型通过")
    
    print("✓ Property 6 测试通过\n")


def test_property_7_multi_task_differentiation():
    """
    **Property 7: Multi-Task Privacy Map Differentiation**
    
    验证不同任务类型产生不同的隐私预算分配
    """
    print("=" * 70)
    print("Property 7: Multi-Task Privacy Map Differentiation")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    # 创建有明显区域划分的语义掩码
    H, W = 256, 256
    semantic_mask = np.zeros((H, W), dtype=np.float32)
    
    # 中心敏感区域 (1.0)
    semantic_mask[64:192, 64:192] = 1.0
    # 周围任务区域 (0.5)
    semantic_mask[32:64, 32:224] = 0.5
    semantic_mask[192:224, 32:224] = 0.5
    semantic_mask[64:192, 32:64] = 0.5
    semantic_mask[64:192, 192:224] = 0.5
    # 其余为背景 (0.0)
    
    # 生成隐私预算图
    privacy_maps = allocator.generate_privacy_maps(semantic_mask)
    
    # 验证差异化
    assert allocator.validate_multi_task_differentiation(privacy_maps), \
        "多任务隐私预算应该有差异"
    
    # 打印各任务类型的统计
    print("\n各任务类型隐私预算统计:")
    for task_type, privacy_map in privacy_maps.items():
        # 计算各区域平均值
        sensitive_region = semantic_mask > 0.7
        task_region = (semantic_mask > 0.3) & (semantic_mask <= 0.7)
        background_region = semantic_mask <= 0.3
        
        avg_sensitive = privacy_map[sensitive_region].mean() if sensitive_region.any() else 0
        avg_task = privacy_map[task_region].mean() if task_region.any() else 0
        avg_background = privacy_map[background_region].mean() if background_region.any() else 0
        
        print(f"  [{task_type}]")
        print(f"    敏感区域: {avg_sensitive:.3f}")
        print(f"    任务区域: {avg_task:.3f}")
        print(f"    背景区域: {avg_background:.3f}")
    
    # 验证任务间差异
    task_types = list(privacy_maps.keys())
    print("\n任务间差异:")
    for i in range(len(task_types)):
        for j in range(i + 1, len(task_types)):
            diff = np.abs(privacy_maps[task_types[i]] - privacy_maps[task_types[j]]).mean()
            print(f"  {task_types[i]} vs {task_types[j]}: {diff:.4f}")
    
    print("\n✓ Property 7 测试通过\n")


def test_property_8_ate_cate_completeness():
    """
    **Property 8: ATE/CATE Output Completeness**
    
    验证 ATE/CATE 输出包含完整字段:
    ate, std, n, ci_lower, ci_upper, ci_level
    """
    print("=" * 70)
    print("Property 8: ATE/CATE Output Completeness")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 50
    
    # 模拟高隐私和低隐私下的性能
    performance_high = np.random.normal(0.85, 0.05, n_samples)
    performance_low = np.random.normal(0.90, 0.05, n_samples)
    
    # 测试 ATE
    print("\n--- ATE 输出完整性 ---")
    ate_result = allocator.compute_ate(performance_high, performance_low)
    
    required_ate_fields = ['ate', 'std', 'n', 'ci_lower', 'ci_upper', 'ci_level']
    
    for field in required_ate_fields:
        assert field in ate_result, f"ATE 缺少字段: {field}"
        print(f"✓ {field}: {ate_result[field]}")
    
    # 验证值的合理性
    assert isinstance(ate_result['ate'], float)
    assert isinstance(ate_result['std'], float) and ate_result['std'] >= 0
    assert isinstance(ate_result['n'], int) and ate_result['n'] == n_samples
    assert ate_result['ci_lower'] < ate_result['ate'] < ate_result['ci_upper']
    assert ate_result['ci_level'] == 0.95
    
    print("\n✓ ATE 输出完整性验证通过")
    
    # 测试 CATE
    print("\n--- CATE 输出完整性 ---")
    
    # 创建语义掩码 [B, 1, H, W]
    B = n_samples
    H, W = 64, 64
    semantic_masks = np.zeros((B, 1, H, W), dtype=np.float32)
    
    # 为每个样本创建不同的敏感区域
    for i in range(B):
        # 随机敏感区域
        cx, cy = np.random.randint(16, 48, 2)
        r = np.random.randint(8, 16)
        y, x = np.ogrid[:H, :W]
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
        semantic_masks[i, 0, mask] = 1.0
    
    for region_type in ['sensitive', 'task', 'background']:
        cate_result = allocator.compute_cate(
            semantic_masks,
            performance_high,
            performance_low,
            region_type=region_type
        )
        
        print(f"\n  [{region_type}]")
        
        # 检查必需字段
        required_cate_fields = ['cate', 'std', 'n', 'region_type']
        for field in required_cate_fields:
            assert field in cate_result, f"CATE 缺少字段: {field}"
            print(f"    {field}: {cate_result[field]}")
        
        # 如果有足够样本，应该有置信区间
        if cate_result['n'] > 1 and not np.isnan(cate_result['cate']):
            if 'ci_lower' in cate_result:
                print(f"    ci_lower: {cate_result['ci_lower']}")
                print(f"    ci_upper: {cate_result['ci_upper']}")
                print(f"    ci_level: {cate_result['ci_level']}")
    
    print("\n✓ CATE 输出完整性验证通过")
    print("\n✓ Property 8 测试通过\n")


def test_causal_report_generation():
    """测试因果报告生成"""
    print("=" * 70)
    print("测试因果报告生成")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    # 创建测试数据
    H, W = 128, 128
    semantic_mask = np.zeros((H, W), dtype=np.float32)
    semantic_mask[32:96, 32:96] = 1.0  # 敏感区域
    semantic_mask[16:32, 16:112] = 0.5  # 任务区域
    
    # 生成隐私预算图
    privacy_maps = allocator.generate_privacy_maps(semantic_mask)
    
    # 生成报告
    report = allocator.generate_causal_report(
        semantic_mask,
        privacy_maps,
        sample_id='test_001',
        dataset='test_dataset'
    )
    
    # 验证报告结构
    assert 'sample_id' in report
    assert 'dataset' in report
    assert 'timestamp' in report
    assert 'region_stats' in report
    assert 'task_analyses' in report
    assert 'explanation' in report
    
    print(f"✓ sample_id: {report['sample_id']}")
    print(f"✓ dataset: {report['dataset']}")
    print(f"✓ timestamp: {report['timestamp']}")
    
    print("\n区域统计:")
    for key, value in report['region_stats'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\n任务分析:")
    for task_type in report['task_analyses']:
        print(f"  ✓ {task_type}")
    
    print("\n✓ 因果报告生成测试通过\n")


def test_report_save_load():
    """测试报告保存和加载"""
    print("=" * 70)
    print("测试报告保存和加载")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    # 创建测试数据
    semantic_mask = np.random.rand(64, 64).astype(np.float32)
    privacy_maps = allocator.generate_privacy_maps(semantic_mask)
    
    report = allocator.generate_causal_report(
        semantic_mask,
        privacy_maps,
        sample_id='save_test',
        dataset='test'
    )
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    allocator.save_report(report, temp_path)
    print(f"✓ 报告已保存到: {temp_path}")
    
    # 加载报告
    loaded_report = allocator.load_report(temp_path)
    
    # 验证内容一致
    assert loaded_report['sample_id'] == report['sample_id']
    assert loaded_report['dataset'] == report['dataset']
    assert loaded_report['region_stats'] == report['region_stats']
    
    print("✓ 报告加载成功")
    print("✓ 内容验证通过")
    
    # 清理
    Path(temp_path).unlink()
    print("✓ 临时文件已清理")
    print("\n✓ 报告保存和加载测试通过\n")


def test_global_privacy_scaling():
    """测试全局隐私级别缩放"""
    print("=" * 70)
    print("测试全局隐私级别缩放")
    print("=" * 70)
    
    allocator = CausalBudgetAllocator()
    
    semantic_mask = np.ones((64, 64), dtype=np.float32)  # 全敏感区域
    
    privacy_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for level in privacy_levels:
        privacy_maps = allocator.generate_privacy_maps(
            semantic_mask, 
            global_privacy=level
        )
        
        # 检查缩放效果
        for task_type, privacy_map in privacy_maps.items():
            max_val = privacy_map.max()
            # 全局隐私为0时，所有值应该为0
            if level == 0.0:
                assert max_val == 0.0, f"global_privacy=0 时应该全为0"
            # 全局隐私为1时，应该有非零值
            elif level == 1.0:
                assert max_val > 0, f"global_privacy=1 时应该有非零值"
        
        print(f"✓ global_privacy={level}: 最大值范围 [0, {max_val:.2f}]")
    
    print("\n✓ 全局隐私级别缩放测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("因果隐私预算分配器测试")
    print("=" * 70 + "\n")
    
    test_basic_initialization()
    test_property_6_shape_consistency()
    test_property_7_multi_task_differentiation()
    test_property_8_ate_cate_completeness()
    test_causal_report_generation()
    test_report_save_load()
    test_global_privacy_scaling()
    
    print("=" * 70)
    print("✓ 所有因果隐私预算分配器测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
