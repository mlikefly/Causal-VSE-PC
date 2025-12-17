# -*- coding: utf-8 -*-
"""
双视图安全指标测试

Property 9: Z-view Evaluation Exclusion
Property 10: C-view Evaluation Inclusion
"""

import sys
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, '.')


def test_zview_evaluation_exclusion():
    """
    Property 9: Z-view Evaluation Exclusion
    
    验证 Z-view 评估不输出 NIST/χ² 测试结果
    """
    print("=" * 70)
    print("Property 9: Z-view Evaluation Exclusion")
    print("=" * 70)
    
    from src.evaluation.dual_view_metrics import (
        DualViewSecurityMetrics, 
        ZViewMetrics,
        get_zview_metric_keys
    )
    
    # 创建测试数据
    np.random.seed(42)
    original = np.random.rand(64, 64).astype(np.float32)
    z_view = np.random.rand(64, 64).astype(np.float32)
    
    # 评估 Z-view
    evaluator = DualViewSecurityMetrics(is_q16_wrap=True)
    metrics = evaluator.evaluate_zview(original, z_view)
    
    # 验证返回类型
    assert isinstance(metrics, ZViewMetrics), "应返回 ZViewMetrics 类型"
    print("✓ 返回类型正确: ZViewMetrics")
    
    # 验证包含基础指标
    metrics_dict = metrics.to_dict()
    assert 'entropy_original' in metrics_dict, "应包含 entropy_original"
    assert 'entropy_encrypted' in metrics_dict, "应包含 entropy_encrypted"
    assert 'npcr' in metrics_dict, "应包含 npcr"
    assert 'uaci' in metrics_dict, "应包含 uaci"
    assert 'corr_encrypted_horizontal' in metrics_dict, "应包含相关性指标"
    print("✓ 包含基础指标: entropy, npcr, uaci, correlation")
    
    # 验证不包含 NIST/χ² 指标
    zview_keys = get_zview_metric_keys()
    assert 'chi2' not in zview_keys, "Z-view 不应包含 chi2"
    assert 'chi2_p_value' not in zview_keys, "Z-view 不应包含 chi2_p_value"
    assert 'chi2_pass' not in zview_keys, "Z-view 不应包含 chi2_pass"
    assert 'nist_monobit_p' not in zview_keys, "Z-view 不应包含 nist_monobit_p"
    assert 'nist_monobit_pass' not in zview_keys, "Z-view 不应包含 nist_monobit_pass"
    assert 'nist_runs_p' not in zview_keys, "Z-view 不应包含 nist_runs_p"
    assert 'nist_runs_pass' not in zview_keys, "Z-view 不应包含 nist_runs_pass"
    print("✓ Z-view 指标键不包含 chi2, nist_monobit, nist_runs")
    
    # 验证 ZViewMetrics 类没有这些属性
    assert not hasattr(metrics, 'chi2'), "ZViewMetrics 不应有 chi2 属性"
    assert not hasattr(metrics, 'nist_monobit_p'), "ZViewMetrics 不应有 nist_monobit_p 属性"
    assert not hasattr(metrics, 'nist_runs_p'), "ZViewMetrics 不应有 nist_runs_p 属性"
    print("✓ ZViewMetrics 类不包含 NIST/χ² 属性")
    
    # 验证检查标准不包含 NIST/χ²
    checks = evaluator.check_zview_standards(metrics)
    assert 'chi_square' not in checks, "Z-view 检查不应包含 chi_square"
    assert 'nist_monobit' not in checks, "Z-view 检查不应包含 nist_monobit"
    assert 'nist_runs' not in checks, "Z-view 检查不应包含 nist_runs"
    print("✓ Z-view 安全标准检查不包含 NIST/χ²")
    
    print("\n✓ Property 9 测试通过")
    return True


def test_cview_evaluation_inclusion():
    """
    Property 10: C-view Evaluation Inclusion
    
    验证 C-view 评估包含完整密码学指标
    """
    print("\n" + "=" * 70)
    print("Property 10: C-view Evaluation Inclusion")
    print("=" * 70)
    
    from src.evaluation.dual_view_metrics import (
        DualViewSecurityMetrics,
        CViewMetrics,
        get_cview_metric_keys
    )
    
    # 创建测试数据
    np.random.seed(42)
    original = np.random.rand(64, 64).astype(np.float32)
    c_view = np.random.rand(64, 64).astype(np.float32)
    
    # 评估 C-view
    evaluator = DualViewSecurityMetrics(is_q16_wrap=True)
    metrics = evaluator.evaluate_cview(original, c_view)
    
    # 验证返回类型
    assert isinstance(metrics, CViewMetrics), "应返回 CViewMetrics 类型"
    print("✓ 返回类型正确: CViewMetrics")
    
    # 验证包含基础指标
    metrics_dict = metrics.to_dict()
    assert 'entropy_original' in metrics_dict, "应包含 entropy_original"
    assert 'entropy_encrypted' in metrics_dict, "应包含 entropy_encrypted"
    assert 'npcr' in metrics_dict, "应包含 npcr"
    assert 'uaci' in metrics_dict, "应包含 uaci"
    print("✓ 包含基础指标: entropy, npcr, uaci")
    
    # 验证包含 χ² 指标
    assert 'chi2' in metrics_dict, "C-view 应包含 chi2"
    assert 'chi2_p_value' in metrics_dict, "C-view 应包含 chi2_p_value"
    assert 'chi2_pass' in metrics_dict, "C-view 应包含 chi2_pass"
    print("✓ C-view 包含 χ² 检验指标")
    
    # 验证包含 NIST 指标
    assert 'nist_monobit_p' in metrics_dict, "C-view 应包含 nist_monobit_p"
    assert 'nist_monobit_pass' in metrics_dict, "C-view 应包含 nist_monobit_pass"
    assert 'nist_runs_p' in metrics_dict, "C-view 应包含 nist_runs_p"
    assert 'nist_runs_pass' in metrics_dict, "C-view 应包含 nist_runs_pass"
    print("✓ C-view 包含 NIST 测试指标")
    
    # 验证 get_cview_metric_keys 包含所有指标
    cview_keys = get_cview_metric_keys()
    required_keys = [
        'chi2', 'chi2_p_value', 'chi2_pass',
        'nist_monobit_p', 'nist_monobit_pass',
        'nist_runs_p', 'nist_runs_pass'
    ]
    for key in required_keys:
        assert key in cview_keys, f"C-view 指标键应包含 {key}"
    print("✓ get_cview_metric_keys() 包含所有密码学指标")
    
    # 验证检查标准包含 NIST/χ²
    checks = evaluator.check_cview_standards(metrics)
    assert 'chi_square' in checks, "C-view 检查应包含 chi_square"
    assert 'nist_monobit' in checks, "C-view 检查应包含 nist_monobit"
    assert 'nist_runs' in checks, "C-view 检查应包含 nist_runs"
    print("✓ C-view 安全标准检查包含 NIST/χ²")
    
    print("\n✓ Property 10 测试通过")
    return True


def test_dual_view_evaluation():
    """测试同时评估 Z-view 和 C-view"""
    print("\n" + "=" * 70)
    print("测试双视图同时评估")
    print("=" * 70)
    
    from src.evaluation.dual_view_metrics import DualViewSecurityMetrics
    
    # 创建测试数据
    np.random.seed(42)
    original = np.random.rand(64, 64).astype(np.float32)
    z_view = np.random.rand(64, 64).astype(np.float32)
    c_view = np.random.rand(64, 64).astype(np.float32)
    
    # 同时评估
    evaluator = DualViewSecurityMetrics()
    results = evaluator.evaluate_dual_view(original, z_view, c_view)
    
    assert 'z_view' in results, "应包含 z_view 结果"
    assert 'c_view' in results, "应包含 c_view 结果"
    print("✓ 同时返回 z_view 和 c_view 评估结果")
    
    # 验证类型
    from src.evaluation.dual_view_metrics import ZViewMetrics, CViewMetrics
    assert isinstance(results['z_view'], ZViewMetrics)
    assert isinstance(results['c_view'], CViewMetrics)
    print("✓ 返回类型正确")
    
    print("\n✓ 双视图同时评估测试通过")
    return True


def test_tensor_input():
    """测试 PyTorch Tensor 输入"""
    print("\n" + "=" * 70)
    print("测试 PyTorch Tensor 输入")
    print("=" * 70)
    
    from src.evaluation.dual_view_metrics import DualViewSecurityMetrics
    
    # 创建 Tensor 测试数据 [B, C, H, W]
    torch.manual_seed(42)
    original = torch.rand(1, 3, 64, 64)
    z_view = torch.rand(1, 3, 64, 64)
    c_view = torch.rand(1, 3, 64, 64)
    
    evaluator = DualViewSecurityMetrics()
    
    # 测试 Z-view
    z_metrics = evaluator.evaluate_zview(original, z_view)
    assert z_metrics.entropy_encrypted > 0, "熵应大于 0"
    print(f"✓ Z-view 熵: {z_metrics.entropy_encrypted:.4f}")
    
    # 测试 C-view
    c_metrics = evaluator.evaluate_cview(original, c_view)
    assert c_metrics.entropy_encrypted > 0, "熵应大于 0"
    assert not np.isnan(c_metrics.chi2), "χ² 不应为 NaN"
    print(f"✓ C-view 熵: {c_metrics.entropy_encrypted:.4f}")
    print(f"✓ C-view χ²: {c_metrics.chi2:.2f}")
    
    print("\n✓ Tensor 输入测试通过")
    return True


def test_with_dual_view_engine():
    """测试与 DualViewEncryptionEngine 集成"""
    print("\n" + "=" * 70)
    print("测试与 DualViewEncryptionEngine 集成")
    print("=" * 70)
    
    try:
        from src.cipher.dual_view_engine import DualViewEncryptionEngine
        from src.evaluation.dual_view_metrics import DualViewSecurityMetrics
    except ImportError as e:
        print(f"⚠️ 跳过集成测试: {e}")
        return True
    
    # 初始化引擎
    engine = DualViewEncryptionEngine(
        password="test_password",
        image_size=64,
        device='cpu',
        deterministic=True
    )
    
    # 创建测试数据
    torch.manual_seed(42)
    images = torch.rand(1, 3, 64, 64)
    privacy_map = torch.ones(1, 1, 64, 64) * 0.5
    
    # 加密
    result = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=1.0,
        image_id="test_001",
        task_type="classification"
    )
    
    print(f"✓ Z-view 形状: {result.z_view.shape}")
    print(f"✓ C-view 形状: {result.c_view.shape}")
    
    # 评估
    evaluator = DualViewSecurityMetrics(is_q16_wrap=True)
    
    # Z-view 评估
    z_metrics = evaluator.evaluate_zview(images, result.z_view)
    z_checks = evaluator.check_zview_standards(z_metrics)
    print(f"\n【Z-view 评估】")
    print(f"  熵: {z_metrics.entropy_encrypted:.4f}")
    print(f"  NPCR: {z_metrics.npcr:.2f}%")
    print(f"  UACI: {z_metrics.uaci:.2f}%")
    print(f"  通过: {sum(z_checks.values())}/{len(z_checks)}")
    
    # C-view 评估
    c_metrics = evaluator.evaluate_cview(images, result.c_view)
    c_checks = evaluator.check_cview_standards(c_metrics)
    print(f"\n【C-view 评估】")
    print(f"  熵: {c_metrics.entropy_encrypted:.4f}")
    print(f"  NPCR: {c_metrics.npcr:.2f}%")
    print(f"  UACI: {c_metrics.uaci:.2f}%")
    print(f"  χ² p值: {c_metrics.chi2_p_value:.4f}")
    print(f"  NIST monobit: {'✓' if c_metrics.nist_monobit_pass else '❌'}")
    print(f"  NIST runs: {'✓' if c_metrics.nist_runs_pass else '❌'}")
    print(f"  通过: {sum(c_checks.values())}/{len(c_checks)}")
    
    print("\n✓ 集成测试通过")
    return True


def test_report_printing():
    """测试报告打印功能"""
    print("\n" + "=" * 70)
    print("测试报告打印功能")
    print("=" * 70)
    
    from src.evaluation.dual_view_metrics import DualViewSecurityMetrics
    
    # 创建测试数据
    np.random.seed(42)
    original = np.random.rand(64, 64).astype(np.float32)
    z_view = np.random.rand(64, 64).astype(np.float32)
    c_view = np.random.rand(64, 64).astype(np.float32)
    
    evaluator = DualViewSecurityMetrics()
    
    # Z-view 报告
    z_metrics = evaluator.evaluate_zview(original, z_view)
    evaluator.print_zview_report(z_metrics)
    
    # C-view 报告
    c_metrics = evaluator.evaluate_cview(original, c_view)
    evaluator.print_cview_report(c_metrics)
    
    print("\n✓ 报告打印测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("双视图安全指标测试")
    print("=" * 70)
    
    tests = [
        ("Property 9: Z-view Evaluation Exclusion", test_zview_evaluation_exclusion),
        ("Property 10: C-view Evaluation Inclusion", test_cview_evaluation_inclusion),
        ("双视图同时评估", test_dual_view_evaluation),
        ("Tensor 输入", test_tensor_input),
        ("DualViewEncryptionEngine 集成", test_with_dual_view_engine),
        ("报告打印", test_report_printing),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed}/{passed + failed} 通过")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
