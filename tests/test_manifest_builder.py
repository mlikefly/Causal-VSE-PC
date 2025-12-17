# -*- coding: utf-8 -*-
"""
ManifestBuilder 单元测试

测试 Manifest 构建器的核心功能
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manifest_builder import ManifestBuilder, ManifestRecord, build_manifest


def test_manifest_record_creation():
    """测试 ManifestRecord 创建和验证"""
    print("=" * 70)
    print("测试 ManifestRecord 创建和验证")
    print("=" * 70)
    
    # 创建有效记录
    record = ManifestRecord(
        sample_id="abc123def456",
        dataset="celebahq",
        split="train",
        image_path="CelebA-HQ/train/00001.png",
        labels={"attributes": [1, 0, 1, 0]},
        sensitive_mask_path="masks/celebahq/train/00001.png",
        task_mask_path="masks/celebahq/train/00001.png",
        semantic_mask_path="masks/celebahq/train/00001.png",
        meta={"source_id": "00001"}
    )
    
    # 验证
    assert record.validate(), "有效记录应该通过验证"
    print("✓ 有效记录验证通过")
    
    # 测试 to_dict
    data = record.to_dict()
    assert data['sample_id'] == "abc123def456"
    assert data['dataset'] == "celebahq"
    print("✓ to_dict 转换正确")
    
    # 测试 from_dict
    record2 = ManifestRecord.from_dict(data)
    assert record2.sample_id == record.sample_id
    assert record2.dataset == record.dataset
    print("✓ from_dict 转换正确")
    
    print("✓ ManifestRecord 测试通过\n")


def test_sample_id_generation():
    """测试样本ID生成的稳定性"""
    print("=" * 70)
    print("测试样本ID生成")
    print("=" * 70)
    
    builder = ManifestBuilder(
        data_root="data",
        output_path="test_manifest.jsonl"
    )
    
    # 相同路径应生成相同ID
    path1 = "CelebA-HQ/train/00001.png"
    id1 = builder._generate_sample_id(path1)
    id2 = builder._generate_sample_id(path1)
    assert id1 == id2, "相同路径应生成相同ID"
    print(f"✓ 路径 '{path1}' -> ID '{id1}'")
    
    # 不同路径应生成不同ID
    path2 = "CelebA-HQ/train/00002.png"
    id3 = builder._generate_sample_id(path2)
    assert id1 != id3, "不同路径应生成不同ID"
    print(f"✓ 路径 '{path2}' -> ID '{id3}'")
    
    # ID长度应为16
    assert len(id1) == 16, "ID长度应为16"
    print(f"✓ ID长度正确: {len(id1)}")
    
    print("✓ 样本ID生成测试通过\n")


def test_manifest_record_completeness():
    """
    **Feature: top-tier-journal-upgrade, Property 4: Manifest Record Completeness**
    **Validates: Requirements 2.2, 2.3, 2.4**
    
    测试所有生成的 manifest 记录包含必需字段
    """
    print("=" * 70)
    print("Property 4: Manifest Record Completeness")
    print("=" * 70)
    
    required_fields = [
        'sample_id', 'dataset', 'split', 'image_path',
        'labels', 'sensitive_mask_path', 'task_mask_path',
        'semantic_mask_path', 'meta'
    ]
    
    # 创建测试记录
    test_records = [
        ManifestRecord(
            sample_id=f"test{i:012d}",
            dataset="celebahq",
            split="train",
            image_path=f"CelebA-HQ/train/{i:05d}.png",
            labels={"attributes": []},
            sensitive_mask_path=f"masks/celebahq/train/{i:05d}.png",
            task_mask_path=f"masks/celebahq/train/{i:05d}.png",
            semantic_mask_path=f"masks/celebahq/train/{i:05d}.png",
            meta={"source_id": f"{i:05d}"}
        )
        for i in range(10)
    ]
    
    # 验证每条记录
    for record in test_records:
        data = record.to_dict()
        for field in required_fields:
            assert field in data, f"缺少必需字段: {field}"
            assert data[field] is not None, f"字段 {field} 为 None"
        assert record.validate(), "记录验证失败"
    
    print(f"✓ 验证了 {len(test_records)} 条记录")
    print(f"✓ 所有记录包含必需字段: {required_fields}")
    print("✓ Property 4 测试通过\n")


def test_manifest_write_and_load():
    """测试 manifest 写入和加载"""
    print("=" * 70)
    print("测试 Manifest 写入和加载")
    print("=" * 70)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        # 创建测试记录
        records = [
            ManifestRecord(
                sample_id=f"test{i:012d}",
                dataset="celebahq",
                split="train",
                image_path=f"CelebA-HQ/train/{i:05d}.png",
                labels={"attributes": [1, 0, 1]},
                sensitive_mask_path=f"masks/{i:05d}.png",
                task_mask_path=f"masks/{i:05d}.png",
                semantic_mask_path=f"masks/{i:05d}.png",
                meta={"source_id": f"{i:05d}"}
            )
            for i in range(5)
        ]
        
        # 写入
        with open(temp_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
        print(f"✓ 写入 {len(records)} 条记录到 {temp_path}")
        
        # 加载
        loaded = ManifestBuilder.load_manifest(temp_path)
        assert len(loaded) == len(records), "加载的记录数不匹配"
        print(f"✓ 加载 {len(loaded)} 条记录")
        
        # 验证内容
        for orig, loaded_rec in zip(records, loaded):
            assert orig.sample_id == loaded_rec.sample_id
            assert orig.dataset == loaded_rec.dataset
            assert orig.image_path == loaded_rec.image_path
        print("✓ 记录内容验证通过")
        
        # 测试迭代器
        count = 0
        for rec in ManifestBuilder.iter_manifest(temp_path):
            count += 1
            assert rec.validate()
        assert count == len(records)
        print(f"✓ 迭代器测试通过 ({count} 条)")
        
    finally:
        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("✓ Manifest 写入和加载测试通过\n")


def test_celeba_scanner():
    """测试 CelebA-HQ 扫描器"""
    print("=" * 70)
    print("测试 CelebA-HQ 扫描器")
    print("=" * 70)
    
    data_root = Path("data")
    celeba_root = data_root / "CelebA-HQ"
    
    if not celeba_root.exists():
        print("⚠️ CelebA-HQ 目录不存在，跳过测试")
        return
    
    # 创建临时输出
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        builder = ManifestBuilder(
            data_root=str(data_root),
            output_path=temp_path
        )
        
        # 只扫描 test 集（较小）
        count = builder.build(datasets=['celebahq'], splits=['test'])
        
        print(f"✓ 扫描到 {count} 条 CelebA-HQ 记录")
        
        if count > 0:
            # 验证记录
            records = ManifestBuilder.load_manifest(temp_path)
            for rec in records[:5]:  # 检查前5条
                assert rec.dataset == 'celebahq'
                assert rec.split == 'test'
                assert rec.validate()
                print(f"  - {rec.sample_id}: {rec.image_path}")
            
            print("✓ CelebA-HQ 记录验证通过")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("✓ CelebA-HQ 扫描器测试通过\n")


def test_fairface_scanner():
    """测试 FairFace 扫描器"""
    print("=" * 70)
    print("测试 FairFace 扫描器")
    print("=" * 70)
    
    data_root = Path("data")
    fairface_root = data_root / "fairface-img-margin025-trainval"
    
    if not fairface_root.exists():
        print("⚠️ FairFace 目录不存在，跳过测试")
        return
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        builder = ManifestBuilder(
            data_root=str(data_root),
            output_path=temp_path
        )
        
        # 只扫描 val 集
        count = builder.build(datasets=['fairface'], splits=['val'])
        
        print(f"✓ 扫描到 {count} 条 FairFace 记录")
        
        if count > 0:
            records = ManifestBuilder.load_manifest(temp_path)
            for rec in records[:5]:
                assert rec.dataset == 'fairface'
                assert 'group' in rec.meta
                print(f"  - {rec.sample_id}: group={rec.meta.get('group', {})}")
            
            print("✓ FairFace 记录验证通过")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("✓ FairFace 扫描器测试通过\n")


def test_openimages_scanner():
    """测试 OpenImages 扫描器"""
    print("=" * 70)
    print("测试 OpenImages 扫描器")
    print("=" * 70)
    
    data_root = Path("data")
    openimages_root = data_root / "OpenImages"
    
    if not openimages_root.exists():
        print("⚠️ OpenImages 目录不存在，跳过测试")
        return
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        builder = ManifestBuilder(
            data_root=str(data_root),
            output_path=temp_path
        )
        
        count = builder.build(datasets=['openimages'], splits=['val'])
        
        print(f"✓ 扫描到 {count} 条 OpenImages 记录")
        
        if count > 0:
            records = ManifestBuilder.load_manifest(temp_path)
            for rec in records[:5]:
                assert rec.dataset == 'openimages'
                print(f"  - {rec.sample_id}: {rec.image_path}")
            
            print("✓ OpenImages 记录验证通过")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("✓ OpenImages 扫描器测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("ManifestBuilder 单元测试")
    print("=" * 70 + "\n")
    
    # 基础测试
    test_manifest_record_creation()
    test_sample_id_generation()
    test_manifest_record_completeness()
    test_manifest_write_and_load()
    
    # 扫描器测试（需要实际数据）
    test_celeba_scanner()
    test_fairface_scanner()
    test_openimages_scanner()
    
    print("=" * 70)
    print("✓ 所有 ManifestBuilder 测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
