# -*- coding: utf-8 -*-
"""
语义掩码生成器测试

**Feature: top-tier-journal-upgrade, Property 5: Semantic Mask Three-Value Constraint**
**Validates: Requirements 3.1**
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.semantic_mask_generator import SemanticMaskGenerator


def test_mask_values():
    """测试掩码值常量"""
    print("=" * 70)
    print("测试掩码值常量")
    print("=" * 70)
    
    gen = SemanticMaskGenerator()
    
    assert gen.MASK_VALUES['sensitive'] == 1.0
    assert gen.MASK_VALUES['task'] == 0.5
    assert gen.MASK_VALUES['background'] == 0.0
    
    print(f"✓ sensitive = {gen.MASK_VALUES['sensitive']}")
    print(f"✓ task = {gen.MASK_VALUES['task']}")
    print(f"✓ background = {gen.MASK_VALUES['background']}")
    print("✓ 掩码值常量测试通过\n")


def test_three_value_constraint():
    """
    **Property 5: Semantic Mask Three-Value Constraint**
    
    测试所有生成的语义掩码只包含 {0.0, 0.5, 1.0} 三个值
    """
    print("=" * 70)
    print("Property 5: Semantic Mask Three-Value Constraint")
    print("=" * 70)
    
    gen = SemanticMaskGenerator(detector_type='opencv')
    
    # 创建测试图像（带有明显的人脸特征）
    test_images = []
    
    # 1. 随机图像
    np.random.seed(42)
    test_images.append(('random', np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)))
    
    # 2. 带有亮度变化的图像（模拟人脸）
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # 中心亮区（模拟人脸）
    cv_y, cv_x = np.ogrid[:256, :256]
    center_mask = ((cv_x - 128) ** 2 + (cv_y - 128) ** 2) < 60 ** 2
    img[center_mask] = [200, 180, 160]  # 肤色
    test_images.append(('face_like', img))
    
    # 3. 全黑图像
    test_images.append(('black', np.zeros((256, 256, 3), dtype=np.uint8)))
    
    # 4. 全白图像
    test_images.append(('white', np.full((256, 256, 3), 255, dtype=np.uint8)))
    
    valid_values = {0.0, 0.5, 1.0}
    
    for name, image in test_images:
        mask = gen.generate(image, dataset='fairface')
        
        # 验证三值约束
        unique_values = set(np.unique(mask))
        is_valid = unique_values.issubset(valid_values)
        
        # 使用 validate_mask 方法
        is_valid_method = gen.validate_mask(mask)
        
        stats = gen.get_region_stats(mask)
        
        print(f"\n{name}:")
        print(f"  唯一值: {sorted(unique_values)}")
        print(f"  三值约束: {'✓' if is_valid else '✗'}")
        print(f"  validate_mask: {'✓' if is_valid_method else '✗'}")
        print(f"  敏感区域: {stats['sensitive_ratio']*100:.1f}%")
        print(f"  任务区域: {stats['task_ratio']*100:.1f}%")
        print(f"  背景区域: {stats['background_ratio']*100:.1f}%")
        
        assert is_valid, f"掩码包含无效值: {unique_values - valid_values}"
        assert is_valid_method, "validate_mask 返回 False"
    
    print("\n✓ Property 5 测试通过\n")


def test_bbox_mask_generation():
    """测试从边界框生成掩码"""
    print("=" * 70)
    print("测试边界框掩码生成")
    print("=" * 70)
    
    gen = SemanticMaskGenerator()
    
    # 创建测试图像
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 测试边界框标注
    annotations = {
        'bbox': [
            {'x1': 0.2, 'y1': 0.2, 'x2': 0.8, 'y2': 0.8, 'category': 'face'},
        ]
    }
    
    mask = gen.generate(image, dataset='openimages', annotations=annotations)
    
    # 验证三值约束
    assert gen.validate_mask(mask), "掩码不符合三值约束"
    
    # 验证敏感区域存在
    stats = gen.get_region_stats(mask)
    assert stats['sensitive_ratio'] > 0, "应该有敏感区域"
    
    print(f"✓ 敏感区域: {stats['sensitive_ratio']*100:.1f}%")
    print(f"✓ 任务区域: {stats['task_ratio']*100:.1f}%")
    print(f"✓ 背景区域: {stats['background_ratio']*100:.1f}%")
    print("✓ 边界框掩码生成测试通过\n")


def test_celebamask_generation():
    """测试 CelebAMask-HQ 掩码生成"""
    print("=" * 70)
    print("测试 CelebAMask-HQ 掩码生成")
    print("=" * 70)
    
    celebamask_root = Path("data/CelebAMask-HQ")
    
    if not celebamask_root.exists():
        print("⚠️ CelebAMask-HQ 目录不存在，跳过测试")
        return
    
    gen = SemanticMaskGenerator(
        celebamask_root=str(celebamask_root),
        image_size=256
    )
    
    # 测试几个已知存在的图像
    test_ids = ['00000', '00001', '00100']
    
    for image_id in test_ids:
        # 创建空图像（实际不需要，因为使用标注）
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        mask = gen.generate(image, dataset='celebahq', image_id=image_id)
        
        if mask is not None and np.any(mask > 0):
            assert gen.validate_mask(mask), f"掩码 {image_id} 不符合三值约束"
            stats = gen.get_region_stats(mask)
            print(f"✓ {image_id}: sensitive={stats['sensitive_ratio']*100:.1f}%, "
                  f"task={stats['task_ratio']*100:.1f}%, "
                  f"background={stats['background_ratio']*100:.1f}%")
        else:
            print(f"⚠️ {image_id}: 未找到掩码标注")
    
    print("✓ CelebAMask-HQ 掩码生成测试通过\n")


def test_detection_fallback():
    """测试人脸检测回退方案"""
    print("=" * 70)
    print("测试人脸检测回退方案")
    print("=" * 70)
    
    gen = SemanticMaskGenerator(detector_type='opencv')
    
    # 创建一个简单的测试图像
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 使用检测回退
    mask = gen.generate(image, dataset='unknown')
    
    # 验证三值约束
    assert gen.validate_mask(mask), "掩码不符合三值约束"
    
    stats = gen.get_region_stats(mask)
    print(f"✓ 敏感区域: {stats['sensitive_ratio']*100:.1f}%")
    print(f"✓ 任务区域: {stats['task_ratio']*100:.1f}%")
    print(f"✓ 背景区域: {stats['background_ratio']*100:.1f}%")
    print("✓ 人脸检测回退方案测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("语义掩码生成器测试")
    print("=" * 70 + "\n")
    
    test_mask_values()
    test_three_value_constraint()
    test_bbox_mask_generation()
    test_celebamask_generation()
    test_detection_fallback()
    
    print("=" * 70)
    print("✓ 所有语义掩码生成器测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
