# -*- coding: utf-8 -*-
"""
任务效用评估器测试

Property 12: Privacy-Utility Curve Completeness
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '.')


def test_property_12_privacy_utility_curve():
    """
    Property 12: Privacy-Utility Curve Completeness
    
    验证隐私-效用曲线包含所有五档隐私级别
    """
    print("=" * 70)
    print("Property 12: Privacy-Utility Curve Completeness")
    print("=" * 70)
    
    from src.evaluation.utility_evaluator import (
        UtilityEvaluator,
        PrivacyUtilityCurve
    )
    
    evaluator = UtilityEvaluator(device='cpu')
    
    # 模拟评估函数
    def mock_evaluate(privacy_level):
        # 隐私越高，效用越低
        return 0.95 - 0.3 * privacy_level
    
    # 生成曲线
    curve = evaluator.generate_privacy_utility_curve(
        evaluate_fn=mock_evaluate,
        task_type="classification",
        metric_name="accuracy"
    )
    
    # 验证返回类型
    assert isinstance(curve, PrivacyUtilityCurve)
    print("✓ 返回类型正确: PrivacyUtilityCurve")
    
    # 验证包含五档隐私级别
    expected_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
    assert curve.privacy_levels == expected_levels
    print(f"✓ 包含五档隐私级别: {curve.privacy_levels}")
    
    # 验证效用值数量
    assert len(curve.utility_values) == 5
    print(f"✓ 效用值数量: {len(curve.utility_values)}")
    
    # 验证完整性检查
    assert curve.is_complete()
    print("✓ 曲线完整性检查通过")
    
    # 验证 to_dict
    curve_dict = curve.to_dict()
    assert 'privacy_levels' in curve_dict
    assert 'utility_values' in curve_dict
    assert 'task_type' in curve_dict
    assert 'metric_name' in curve_dict
    print("✓ to_dict() 包含所有必需字段")
    
    # 打印曲线
    evaluator.print_curve(curve)
    
    print("\n✓ Property 12 测试通过")
    return True


def test_classification_evaluation():
    """测试分类评估"""
    print("\n" + "=" * 70)
    print("测试分类评估")
    print("=" * 70)
    
    from src.evaluation.utility_evaluator import UtilityEvaluator, ClassificationResult
    
    torch.manual_seed(42)
    n_samples = 50
    n_classes = 5
    
    images = torch.rand(n_samples, 3, 32, 32)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    evaluator = UtilityEvaluator(device='cpu')
    result = evaluator.evaluate_classification(images, labels)
    
    assert isinstance(result, ClassificationResult)
    print(f"✓ 返回类型正确: ClassificationResult")
    
    assert 0 <= result.accuracy <= 1
    print(f"✓ 准确率: {result.accuracy:.4f}")
    
    assert result.num_samples == n_samples
    print(f"✓ 样本数: {result.num_samples}")
    
    assert result.num_classes == n_classes
    print(f"✓ 类别数: {result.num_classes}")
    
    evaluator.print_report(result, "分类评估报告")
    
    print("\n✓ 分类评估测试通过")
    return True


def test_segmentation_evaluation():
    """测试分割评估"""
    print("\n" + "=" * 70)
    print("测试分割评估")
    print("=" * 70)
    
    from src.evaluation.utility_evaluator import UtilityEvaluator, SegmentationResult
    
    torch.manual_seed(42)
    n_samples = 10
    
    images = torch.rand(n_samples, 3, 64, 64)
    gt_masks = torch.randint(0, 2, (n_samples, 64, 64))
    
    evaluator = UtilityEvaluator(device='cpu')
    result = evaluator.evaluate_segmentation(images, gt_masks, num_classes=2)
    
    assert isinstance(result, SegmentationResult)
    print(f"✓ 返回类型正确: SegmentationResult")
    
    assert 0 <= result.miou <= 1
    print(f"✓ mIoU: {result.miou:.4f}")
    
    assert 0 <= result.pixel_accuracy <= 1
    print(f"✓ 像素准确率: {result.pixel_accuracy:.4f}")
    
    assert 0 <= result.dice_score <= 1
    print(f"✓ Dice: {result.dice_score:.4f}")
    
    evaluator.print_report(result, "分割评估报告")
    
    print("\n✓ 分割评估测试通过")
    return True


def test_fairness_evaluation():
    """测试公平性评估"""
    print("\n" + "=" * 70)
    print("测试公平性评估")
    print("=" * 70)
    
    from src.evaluation.utility_evaluator import UtilityEvaluator, FairnessResult
    
    torch.manual_seed(42)
    n_samples = 60
    n_groups = 3
    
    images = torch.rand(n_samples, 3, 32, 32)
    labels = torch.randint(0, 2, (n_samples,))
    group_labels = torch.arange(n_samples) % n_groups
    group_names = ['Male', 'Female', 'Other']
    
    evaluator = UtilityEvaluator(device='cpu')
    result = evaluator.evaluate_fairness(images, labels, group_labels, group_names)
    
    assert isinstance(result, FairnessResult)
    print(f"✓ 返回类型正确: FairnessResult")
    
    assert len(result.group_accuracies) == n_groups
    print(f"✓ 分组数: {len(result.group_accuracies)}")
    
    assert result.accuracy_gap >= 0
    print(f"✓ 准确率差距: {result.accuracy_gap:.4f}")
    
    evaluator.print_report(result, "公平性评估报告")
    
    print("\n✓ 公平性评估测试通过")
    return True


def test_privacy_utility_at_levels():
    """测试多隐私级别评估"""
    print("\n" + "=" * 70)
    print("测试多隐私级别评估")
    print("=" * 70)
    
    from src.evaluation.utility_evaluator import UtilityEvaluator
    
    torch.manual_seed(42)
    n_samples = 20
    
    # 创建不同隐私级别的图像
    images_dict = {}
    for level in [0.0, 0.3, 0.5, 0.7, 1.0]:
        # 模拟：隐私越高，噪声越大
        images = torch.rand(n_samples, 3, 32, 32)
        noise = torch.randn_like(images) * level * 0.5
        images_dict[level] = (images + noise).clamp(0, 1)
    
    labels = torch.randint(0, 3, (n_samples,))
    
    evaluator = UtilityEvaluator(device='cpu')
    curve = evaluator.evaluate_at_privacy_levels(
        images_dict, labels, task_type="classification"
    )
    
    assert curve.is_complete()
    print("✓ 曲线完整")
    
    assert len(curve.privacy_levels) == 5
    print(f"✓ 隐私级别数: {len(curve.privacy_levels)}")
    
    evaluator.print_curve(curve)
    
    print("\n✓ 多隐私级别评估测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("任务效用评估器测试")
    print("=" * 70)
    
    tests = [
        ("Property 12: Privacy-Utility Curve Completeness", test_property_12_privacy_utility_curve),
        ("分类评估", test_classification_evaluation),
        ("分割评估", test_segmentation_evaluation),
        ("公平性评估", test_fairness_evaluation),
        ("多隐私级别评估", test_privacy_utility_at_levels),
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
