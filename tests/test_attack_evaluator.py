# -*- coding: utf-8 -*-
"""
攻击评估器测试

Property 11: Attack Evaluation Matrix Completeness
"""

import sys
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, '.')


def test_attack_evaluation_matrix_completeness():
    """
    Property 11: Attack Evaluation Matrix Completeness
    
    验证攻击评估矩阵包含四类攻击的完整评估结果
    """
    print("=" * 70)
    print("Property 11: Attack Evaluation Matrix Completeness")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import (
        AttackEvaluator,
        AttackEvaluationMatrix,
        IdentityAttackResult,
        ReconstructionAttackResult,
        AttributeInferenceResult,
        LinkabilityAttackResult
    )
    
    # 创建测试数据
    torch.manual_seed(42)
    n_samples = 20
    n_identities = 5
    
    original_images = torch.rand(n_samples, 3, 64, 64)
    encrypted_images = torch.rand(n_samples, 3, 64, 64)
    identity_labels = torch.randint(0, n_identities, (n_samples,))
    attribute_labels = {
        'gender': torch.randint(0, 2, (n_samples,)),
        'race': torch.randint(0, 4, (n_samples,))
    }
    sensitive_mask = torch.ones(n_samples, 1, 64, 64)
    
    # 初始化评估器
    evaluator = AttackEvaluator(device='cpu')
    
    # 执行完整评估
    matrix = evaluator.evaluate_all(
        original_images=original_images,
        encrypted_images=encrypted_images,
        identity_labels=identity_labels,
        attribute_labels=attribute_labels,
        sensitive_mask=sensitive_mask
    )
    
    # 验证返回类型
    assert isinstance(matrix, AttackEvaluationMatrix), "应返回 AttackEvaluationMatrix"
    print("✓ 返回类型正确: AttackEvaluationMatrix")
    
    # 验证身份识别攻击结果
    assert matrix.identity is not None, "应包含身份识别攻击结果"
    assert isinstance(matrix.identity, IdentityAttackResult)
    assert 0 <= matrix.identity.top1_accuracy <= 1
    assert 0 <= matrix.identity.top5_accuracy <= 1
    print(f"✓ 身份识别攻击: Top-1={matrix.identity.top1_accuracy:.4f}, Top-5={matrix.identity.top5_accuracy:.4f}")
    
    # 验证重建攻击结果
    assert matrix.reconstruction is not None, "应包含重建攻击结果"
    assert isinstance(matrix.reconstruction, ReconstructionAttackResult)
    assert matrix.reconstruction.psnr > 0 or matrix.reconstruction.psnr == float('inf')
    assert -1 <= matrix.reconstruction.ssim <= 1
    print(f"✓ 重建攻击: PSNR={matrix.reconstruction.psnr:.2f}dB, SSIM={matrix.reconstruction.ssim:.4f}")
    
    # 验证属性推断攻击结果
    assert len(matrix.attribute_inference) > 0, "应包含属性推断攻击结果"
    assert 'gender' in matrix.attribute_inference
    assert 'race' in matrix.attribute_inference
    for attr_name, result in matrix.attribute_inference.items():
        assert isinstance(result, AttributeInferenceResult)
        assert 0 <= result.accuracy_original <= 1
        assert 0 <= result.accuracy_encrypted <= 1
        print(f"✓ 属性推断攻击 ({attr_name}): 原始={result.accuracy_original:.4f}, 加密={result.accuracy_encrypted:.4f}")
    
    # 验证可链接性攻击结果
    assert matrix.linkability is not None, "应包含可链接性攻击结果"
    assert isinstance(matrix.linkability, LinkabilityAttackResult)
    assert 0 <= matrix.linkability.linkage_auc <= 1
    print(f"✓ 可链接性攻击: AUC={matrix.linkability.linkage_auc:.4f}")
    
    # 验证完整性
    assert matrix.is_complete(), "评估矩阵应完整"
    print("✓ 评估矩阵完整性检查通过")
    
    # 验证 to_dict
    matrix_dict = matrix.to_dict()
    assert 'identity' in matrix_dict
    assert 'reconstruction' in matrix_dict
    assert 'attribute_inference' in matrix_dict
    assert 'linkability' in matrix_dict
    print("✓ to_dict() 包含所有攻击类型")
    
    print("\n✓ Property 11 测试通过")
    return True


def test_identity_attack():
    """测试身份识别攻击评估"""
    print("\n" + "=" * 70)
    print("测试身份识别攻击评估")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import AttackEvaluator
    
    torch.manual_seed(42)
    n_samples = 30
    n_identities = 6
    
    # 创建有一定结构的测试数据
    encrypted_images = torch.rand(n_samples, 3, 32, 32)
    identity_labels = torch.arange(n_samples) % n_identities
    
    evaluator = AttackEvaluator(device='cpu')
    result = evaluator.evaluate_identity_attack(encrypted_images, identity_labels)
    
    print(f"  Top-1 识别率: {result.top1_accuracy:.4f}")
    print(f"  Top-5 识别率: {result.top5_accuracy:.4f}")
    print(f"  身份数量: {result.num_identities}")
    print(f"  样本数量: {result.num_samples}")
    print(f"  嵌入维度: {result.embedding_dim}")
    
    assert result.num_identities == n_identities
    assert result.num_samples == n_samples
    
    print("\n✓ 身份识别攻击测试通过")
    return True


def test_reconstruction_attack():
    """测试重建攻击评估"""
    print("\n" + "=" * 70)
    print("测试重建攻击评估")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import AttackEvaluator
    
    torch.manual_seed(42)
    
    # 创建测试数据
    original = torch.rand(4, 3, 64, 64)
    encrypted = original + torch.randn_like(original) * 0.1  # 添加噪声
    encrypted = encrypted.clamp(0, 1)
    
    sensitive_mask = torch.zeros(4, 1, 64, 64)
    sensitive_mask[:, :, 16:48, 16:48] = 1  # 中心区域为敏感区域
    
    evaluator = AttackEvaluator(device='cpu')
    result = evaluator.evaluate_reconstruction_attack(
        encrypted, original, sensitive_mask
    )
    
    print(f"  PSNR: {result.psnr:.2f} dB")
    print(f"  SSIM: {result.ssim:.4f}")
    print(f"  LPIPS: {result.lpips:.4f}")
    print(f"  MSE: {result.mse:.6f}")
    print(f"  敏感区域 PSNR: {result.sensitive_psnr:.2f} dB")
    print(f"  敏感区域 SSIM: {result.sensitive_ssim:.4f}")
    
    # 验证指标合理性
    assert result.psnr > 0, "PSNR 应大于 0"
    assert -1 <= result.ssim <= 1, "SSIM 应在 [-1, 1] 范围内"
    
    print("\n✓ 重建攻击测试通过")
    return True


def test_attribute_inference_attack():
    """测试属性推断攻击评估"""
    print("\n" + "=" * 70)
    print("测试属性推断攻击评估")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import AttackEvaluator
    
    torch.manual_seed(42)
    n_samples = 20
    
    original = torch.rand(n_samples, 3, 32, 32)
    encrypted = torch.rand(n_samples, 3, 32, 32)
    gender_labels = torch.randint(0, 2, (n_samples,))
    
    evaluator = AttackEvaluator(device='cpu')
    result = evaluator.evaluate_attribute_inference(
        original, encrypted, gender_labels, "gender"
    )
    
    print(f"  属性: {result.attribute_name}")
    print(f"  原始准确率: {result.accuracy_original:.4f}")
    print(f"  加密准确率: {result.accuracy_encrypted:.4f}")
    print(f"  准确率下降: {result.accuracy_drop:.4f}")
    print(f"  类别数: {result.num_classes}")
    
    assert result.attribute_name == "gender"
    assert 0 <= result.accuracy_original <= 1
    assert 0 <= result.accuracy_encrypted <= 1
    
    print("\n✓ 属性推断攻击测试通过")
    return True


def test_linkability_attack():
    """测试可链接性攻击评估"""
    print("\n" + "=" * 70)
    print("测试可链接性攻击评估")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import AttackEvaluator
    
    torch.manual_seed(42)
    n_samples = 24
    n_identities = 4
    
    encrypted = torch.rand(n_samples, 3, 32, 32)
    identity_labels = torch.arange(n_samples) % n_identities
    
    evaluator = AttackEvaluator(device='cpu')
    result = evaluator.evaluate_linkability_attack(encrypted, identity_labels)
    
    print(f"  Linkage AUC: {result.linkage_auc:.4f}")
    print(f"  类内距离: {result.intra_class_distance:.4f}")
    print(f"  类间距离: {result.inter_class_distance:.4f}")
    print(f"  距离比: {result.distance_ratio:.4f}")
    print(f"  身份数量: {result.num_identities}")
    
    assert 0 <= result.linkage_auc <= 1
    assert result.num_identities == n_identities
    
    print("\n✓ 可链接性攻击测试通过")
    return True


def test_report_printing():
    """测试报告打印功能"""
    print("\n" + "=" * 70)
    print("测试报告打印功能")
    print("=" * 70)
    
    from src.evaluation.attack_evaluator import AttackEvaluator
    
    torch.manual_seed(42)
    n_samples = 16
    n_identities = 4
    
    original = torch.rand(n_samples, 3, 32, 32)
    encrypted = torch.rand(n_samples, 3, 32, 32)
    identity_labels = torch.arange(n_samples) % n_identities
    attribute_labels = {'gender': torch.randint(0, 2, (n_samples,))}
    
    evaluator = AttackEvaluator(device='cpu')
    matrix = evaluator.evaluate_all(
        original, encrypted, identity_labels, attribute_labels
    )
    
    evaluator.print_report(matrix)
    
    print("\n✓ 报告打印测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("攻击评估器测试")
    print("=" * 70)
    
    tests = [
        ("Property 11: Attack Evaluation Matrix Completeness", test_attack_evaluation_matrix_completeness),
        ("身份识别攻击", test_identity_attack),
        ("重建攻击", test_reconstruction_attack),
        ("属性推断攻击", test_attribute_inference_attack),
        ("可链接性攻击", test_linkability_attack),
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
