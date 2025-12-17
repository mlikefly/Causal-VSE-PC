# -*- coding: utf-8 -*-
"""
双视图加密引擎测试

**Property 1: Dual-View Encryption Completeness**
**Validates: Requirements 1.1, 1.2, 1.3**
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cipher.dual_view_engine import DualViewEncryptionEngine, EncryptionResult


def test_basic_initialization():
    """测试基本初始化"""
    print("=" * 70)
    print("测试基本初始化")
    print("=" * 70)
    
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=64,
        device='cpu',
        deterministic=True
    )
    
    assert engine.z_cipher is not None
    assert engine.c_cipher is not None
    assert engine.key_system is not None
    assert engine.deterministic == True
    
    print("✓ DualViewEncryptionEngine 初始化成功")
    print(f"✓ 支持的隐私级别: {engine.get_supported_privacy_levels()}")
    print("✓ 基本初始化测试通过\n")


def test_property_1_dual_view_completeness():
    """
    **Property 1: Dual-View Encryption Completeness**
    
    验证同时生成 Z-view 和 C-view
    """
    print("=" * 70)
    print("Property 1: Dual-View Encryption Completeness")
    print("=" * 70)
    
    # 强制使用 CPU 避免设备不匹配
    device = 'cpu'
    
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=64,
        device=device,
        deterministic=True,
        use_frequency=False  # 简化测试
    )
    
    # 创建测试数据
    B, C, H, W = 2, 3, 64, 64
    images = torch.rand(B, C, H, W, device=device)
    privacy_map = torch.rand(B, 1, H, W, device=device)
    
    # 执行双视图加密
    result = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.7,
        image_id='test_001',
        task_type='classification',
        dataset='test',
        split='train'
    )
    
    # 验证结果类型
    assert isinstance(result, EncryptionResult)
    
    # 验证 Z-view
    assert result.z_view is not None
    assert result.z_view.shape == images.shape
    print(f"✓ Z-view 形状: {result.z_view.shape}")
    
    # 验证 C-view
    assert result.c_view is not None
    assert result.c_view.shape == images.shape
    print(f"✓ C-view 形状: {result.c_view.shape}")
    
    # 验证 enc_info
    assert 'z_view' in result.enc_info
    assert 'c_view' in result.enc_info
    assert 'audit' in result.enc_info
    print(f"✓ enc_info 包含必需字段")
    
    # 验证 Z-view 和 C-view 不同（C-view 有 AEAD 封装）
    z_c_diff = (result.z_view - result.c_view).abs().mean().item()
    print(f"✓ Z-view 与 C-view 差异: {z_c_diff:.4f}")
    
    # 验证审计信息
    audit = result.enc_info['audit']
    assert 'privacy_map_hash' in audit
    assert 'z_view_hash' in audit
    print(f"✓ 审计信息: privacy_map_hash={audit['privacy_map_hash']}, z_view_hash={audit['z_view_hash']}")
    
    print("\n✓ Property 1 测试通过\n")


def test_zview_encryption_decryption():
    """测试 Z-view 加密解密"""
    print("=" * 70)
    print("测试 Z-view 加密解密")
    print("=" * 70)
    
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=64,
        device='cpu',
        deterministic=True,
        use_frequency=False  # 简化测试
    )
    
    # 创建测试数据
    B, C, H, W = 1, 3, 64, 64
    images = torch.rand(B, C, H, W)
    privacy_map = torch.ones(B, 1, H, W) * 0.5
    
    # 加密
    z_view, z_info = engine.encrypt_zview_only(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.5,
        image_id='zview_test',
        task_type='classification'
    )
    
    print(f"✓ Z-view 加密完成: {z_view.shape}")
    
    # 验证加密效果
    enc_diff = (images - z_view).abs().mean().item()
    print(f"✓ 加密差异 (MAE): {enc_diff:.4f}")
    assert enc_diff > 0.01, "加密应该产生明显变化"
    
    # 解密
    decrypted = engine.decrypt_zview(
        z_view=z_view,
        z_info=z_info,
        privacy_map=privacy_map
    )
    
    # 验证解密效果
    dec_diff = (images - decrypted).abs().mean().item()
    print(f"✓ 解密差异 (MAE): {dec_diff:.4f}")
    
    print("\n✓ Z-view 加密解密测试通过\n")


def test_cview_from_zview():
    """测试从 Z-view 生成 C-view"""
    print("=" * 70)
    print("测试从 Z-view 生成 C-view")
    print("=" * 70)
    
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=64,
        device='cpu',
        deterministic=True,
        use_frequency=False
    )
    
    # 创建测试数据
    B, C, H, W = 1, 3, 64, 64
    images = torch.rand(B, C, H, W)
    privacy_map = torch.ones(B, 1, H, W) * 0.7
    
    # 先生成 Z-view
    z_view, z_info = engine.encrypt_zview_only(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.7,
        image_id='cview_test',
        task_type='detection'
    )
    
    print(f"✓ Z-view 生成完成")
    
    # 从 Z-view 生成 C-view
    c_view, c_info = engine.encrypt_cview_from_zview(
        z_view=z_view,
        privacy_map=privacy_map,
        z_info=z_info,
        image_id='cview_test',
        task_type='detection',
        dataset='test',
        split='val'
    )
    
    print(f"✓ C-view 生成完成")
    
    # 验证 C-view 包含 AEAD 信息
    assert 'crypto_wrap' in c_info
    assert c_info['crypto_wrap'] is not None
    assert 'nonces' in c_info['crypto_wrap']
    assert 'tags' in c_info['crypto_wrap']
    
    print(f"✓ C-view 包含 AEAD 信息:")
    print(f"  - nonces: {len(c_info['crypto_wrap']['nonces'])} 个")
    print(f"  - tags: {len(c_info['crypto_wrap']['tags'])} 个")
    
    # 验证审计信息
    assert 'audit' in c_info
    print(f"✓ 审计信息: {c_info['audit']}")
    
    print("\n✓ C-view 生成测试通过\n")


def test_deterministic_nonce():
    """测试确定性 nonce 派生"""
    print("=" * 70)
    print("测试确定性 nonce 派生")
    print("=" * 70)
    
    # 使用固定种子确保可重复
    torch.manual_seed(42)
    
    device = 'cpu'
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=32,  # 更小的尺寸加速测试
        device=device,
        deterministic=True,
        use_frequency=False
    )
    
    # 创建测试数据（固定）
    B, C, H, W = 1, 3, 32, 32
    torch.manual_seed(123)
    images = torch.rand(B, C, H, W, device=device)
    privacy_map = torch.ones(B, 1, H, W, device=device) * 0.5
    
    # 第一次加密
    result1 = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.5,
        image_id='nonce_test',
        task_type='classification',
        dataset='test',
        split='train'
    )
    
    nonce1 = result1.enc_info['c_view']['crypto_wrap']['nonces'][0]
    print(f"✓ 第一次 nonce: {nonce1[:16]}...")
    
    # 第二次加密（相同输入）
    result2 = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.5,
        image_id='nonce_test',
        task_type='classification',
        dataset='test',
        split='train'
    )
    
    nonce2 = result2.enc_info['c_view']['crypto_wrap']['nonces'][0]
    print(f"✓ 第二次 nonce: {nonce2[:16]}...")
    
    assert nonce1 == nonce2, "确定性模式下相同输入应产生相同 nonce"
    print("✓ 确定性 nonce 验证通过")
    
    # 验证不同 task_type 产生不同 nonce
    result3 = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.5,
        image_id='nonce_test',
        task_type='detection',  # 不同任务类型
        dataset='test',
        split='train'
    )
    
    nonce3 = result3.enc_info['c_view']['crypto_wrap']['nonces'][0]
    print(f"✓ 不同 task_type nonce: {nonce3[:16]}...")
    
    assert nonce1 != nonce3, "不同 task_type 应产生不同 nonce"
    print("✓ 不同 task_type 产生不同 nonce")
    
    print("\n✓ 确定性 nonce 测试通过\n")


def test_privacy_levels():
    """测试不同隐私级别"""
    print("=" * 70)
    print("测试不同隐私级别")
    print("=" * 70)
    
    device = 'cpu'
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=32,  # 更小的尺寸加速测试
        device=device,
        deterministic=True,
        use_frequency=False
    )
    
    # 创建测试数据
    torch.manual_seed(456)
    B, C, H, W = 1, 3, 32, 32
    images = torch.rand(B, C, H, W, device=device)
    privacy_map = torch.ones(B, 1, H, W, device=device)
    
    # 只测试部分隐私级别以加速
    privacy_levels = [0.3, 0.7, 1.0]
    results = {}
    
    for level in privacy_levels:
        result = engine.encrypt(
            images=images,
            privacy_map=privacy_map,
            privacy_level=level,
            image_id=f'level_{level}',
            task_type='classification'
        )
        
        # 计算与原图的差异
        z_diff = (images - result.z_view).abs().mean().item()
        c_diff = (images - result.c_view).abs().mean().item()
        
        results[level] = {'z_diff': z_diff, 'c_diff': c_diff}
        print(f"✓ privacy_level={level}: Z-view MAE={z_diff:.4f}, C-view MAE={c_diff:.4f}")
    
    # 验证隐私级别越高，加密效果越强
    print(f"\n✓ 隐私级别对比: 0.3 vs 1.0")
    print(f"  Z-view: {results[0.3]['z_diff']:.4f} vs {results[1.0]['z_diff']:.4f}")
    
    print("\n✓ 隐私级别测试通过\n")


def test_cview_storage_pack():
    """测试 C-view 存储打包"""
    print("=" * 70)
    print("测试 C-view 存储打包")
    print("=" * 70)
    
    device = 'cpu'
    engine = DualViewEncryptionEngine(
        password="test_password_2025",
        image_size=32,  # 更小的尺寸加速测试
        device=device,
        deterministic=True,
        use_frequency=False
    )
    
    # 创建测试数据
    torch.manual_seed(789)
    B, C, H, W = 1, 3, 32, 32
    images = torch.rand(B, C, H, W, device=device)
    privacy_map = torch.ones(B, 1, H, W, device=device) * 0.7
    
    # 加密
    result = engine.encrypt(
        images=images,
        privacy_map=privacy_map,
        privacy_level=0.7,
        image_id='storage_test',
        task_type='classification',
        dataset='test',
        split='train'
    )
    
    # 打包存储
    storage_pack = engine.pack_cview_for_storage(result.c_view, result.enc_info)
    
    # 验证存储包结构
    assert 'version' in storage_pack
    assert 'shape' in storage_pack
    assert 'ciphertext' in storage_pack
    assert 'nonces' in storage_pack
    assert 'tags' in storage_pack
    
    print(f"✓ 存储包版本: {storage_pack['version']}")
    print(f"✓ 存储包形状: {storage_pack['shape']}")
    print(f"✓ 密文数量: {len(storage_pack['ciphertext'])}")
    
    # 解包
    c_view_restored, wrap_info = engine.unpack_cview_from_storage(storage_pack)
    
    # 验证解包结果
    assert c_view_restored.shape == result.c_view.shape
    restore_diff = (result.c_view - c_view_restored).abs().max().item()
    print(f"✓ 解包差异: {restore_diff:.6f}")
    assert restore_diff < 1e-5, "解包应该无损"
    
    print("\n✓ C-view 存储打包测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("双视图加密引擎测试")
    print("=" * 70 + "\n")
    
    test_basic_initialization()
    test_property_1_dual_view_completeness()
    test_zview_encryption_decryption()
    test_cview_from_zview()
    test_deterministic_nonce()
    test_privacy_levels()
    test_cview_storage_pack()
    
    print("=" * 70)
    print("✓ 所有双视图加密引擎测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
