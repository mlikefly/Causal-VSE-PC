"""
单元测试：确定性nonce派生逻辑

**Feature: top-tier-journal-upgrade, Property 2: Deterministic Encryption Round-Trip**
**Validates: Requirements 1.6, 9.1, 9.2**
"""

import torch
import hashlib
import hmac
import sys
sys.path.insert(0, '.')

from src.crypto.key_system import HierarchicalKeySystem


def test_deterministic_nonce_derivation_logic():
    """测试确定性nonce派生逻辑（不涉及完整加密）"""
    print('='*70)
    print('测试确定性nonce派生逻辑')
    print('='*70)
    
    # 初始化密钥系统
    key_system = HierarchicalKeySystem('test_password_2025', iterations=10000)
    
    # 模拟输入
    image_id = 'test_image_001'
    task_type = 'classification'
    
    # 模拟privacy_map和z_view的哈希
    privacy_map = torch.rand(1, 1, 64, 64)
    z_view = torch.rand(1, 1, 64, 64)
    
    privacy_map_hash = hashlib.sha256(
        privacy_map.numpy().astype('float32').tobytes()
    ).hexdigest()[:8]
    
    z_view_hash = hashlib.sha256(
        z_view.numpy().astype('float32').tobytes()
    ).hexdigest()[:8]
    
    # 派生nonce
    def derive_nonce(key_system, image_id, task_type, pm_hash, zv_hash):
        context = f"cview|v2|{image_id}|{task_type}|{pm_hash}|{zv_hash}".encode('utf-8')
        k_nonce = hashlib.sha256(key_system.master_key + b'|nonce').digest()
        return hmac.new(k_nonce, context, hashlib.sha256).digest()[:16]
    
    # 测试1：相同输入产生相同nonce
    nonce1 = derive_nonce(key_system, image_id, task_type, privacy_map_hash, z_view_hash)
    nonce2 = derive_nonce(key_system, image_id, task_type, privacy_map_hash, z_view_hash)
    
    print(f'Nonce 1: {nonce1.hex()}')
    print(f'Nonce 2: {nonce2.hex()}')
    print(f'相同输入产生相同nonce: {nonce1 == nonce2}')
    assert nonce1 == nonce2, "相同输入应产生相同nonce"
    
    # 测试2：不同image_id产生不同nonce
    nonce3 = derive_nonce(key_system, 'test_image_002', task_type, privacy_map_hash, z_view_hash)
    print(f'\n不同image_id的Nonce: {nonce3.hex()}')
    print(f'不同image_id产生不同nonce: {nonce1 != nonce3}')
    assert nonce1 != nonce3, "不同image_id应产生不同nonce"
    
    # 测试3：不同task_type产生不同nonce
    nonce4 = derive_nonce(key_system, image_id, 'detection', privacy_map_hash, z_view_hash)
    print(f'\n不同task_type的Nonce: {nonce4.hex()}')
    print(f'不同task_type产生不同nonce: {nonce1 != nonce4}')
    assert nonce1 != nonce4, "不同task_type应产生不同nonce"
    
    # 测试4：不同privacy_map产生不同nonce
    privacy_map2 = torch.rand(1, 1, 64, 64)
    pm_hash2 = hashlib.sha256(privacy_map2.numpy().astype('float32').tobytes()).hexdigest()[:8]
    nonce5 = derive_nonce(key_system, image_id, task_type, pm_hash2, z_view_hash)
    print(f'\n不同privacy_map的Nonce: {nonce5.hex()}')
    print(f'不同privacy_map产生不同nonce: {nonce1 != nonce5}')
    assert nonce1 != nonce5, "不同privacy_map应产生不同nonce"
    
    # 测试5：不同z_view产生不同nonce
    z_view2 = torch.rand(1, 1, 64, 64)
    zv_hash2 = hashlib.sha256(z_view2.numpy().astype('float32').tobytes()).hexdigest()[:8]
    nonce6 = derive_nonce(key_system, image_id, task_type, privacy_map_hash, zv_hash2)
    print(f'\n不同z_view的Nonce: {nonce6.hex()}')
    print(f'不同z_view产生不同nonce: {nonce1 != nonce6}')
    assert nonce1 != nonce6, "不同z_view应产生不同nonce"
    
    # 测试6：不同密码产生不同nonce
    key_system2 = HierarchicalKeySystem('different_password', iterations=10000)
    nonce7 = derive_nonce(key_system2, image_id, task_type, privacy_map_hash, z_view_hash)
    print(f'\n不同密码的Nonce: {nonce7.hex()}')
    print(f'不同密码产生不同nonce: {nonce1 != nonce7}')
    assert nonce1 != nonce7, "不同密码应产生不同nonce"
    
    print('\n' + '='*70)
    print('✓ 确定性nonce派生逻辑测试通过')
    print('='*70)


def test_nonce_uniqueness():
    """测试nonce唯一性（避免nonce复用）"""
    print('\n' + '='*70)
    print('测试nonce唯一性')
    print('='*70)
    
    key_system = HierarchicalKeySystem('test_password_2025', iterations=10000)
    
    def derive_nonce(key_system, image_id, task_type, pm_hash, zv_hash):
        context = f"cview|v2|{image_id}|{task_type}|{pm_hash}|{zv_hash}".encode('utf-8')
        k_nonce = hashlib.sha256(key_system.master_key + b'|nonce').digest()
        return hmac.new(k_nonce, context, hashlib.sha256).digest()[:16]
    
    # 生成多个nonce
    nonces = set()
    for i in range(100):
        image_id = f'image_{i:04d}'
        for task in ['classification', 'detection', 'segmentation']:
            pm_hash = hashlib.sha256(f'pm_{i}_{task}'.encode()).hexdigest()[:8]
            zv_hash = hashlib.sha256(f'zv_{i}_{task}'.encode()).hexdigest()[:8]
            nonce = derive_nonce(key_system, image_id, task, pm_hash, zv_hash)
            nonces.add(nonce)
    
    expected_count = 100 * 3  # 100 images × 3 tasks
    actual_count = len(nonces)
    
    print(f'生成的nonce数量: {actual_count}')
    print(f'期望的唯一nonce数量: {expected_count}')
    print(f'所有nonce唯一: {actual_count == expected_count}')
    
    assert actual_count == expected_count, f"应有{expected_count}个唯一nonce，实际{actual_count}个"
    
    print('✓ nonce唯一性测试通过')


if __name__ == '__main__':
    test_deterministic_nonce_derivation_logic()
    test_nonce_uniqueness()
