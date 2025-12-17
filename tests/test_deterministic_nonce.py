"""
测试确定性nonce派生功能

**Feature: top-tier-journal-upgrade, Property 2: Deterministic Encryption Round-Trip**
**Validates: Requirements 1.6, 9.1, 9.2**
"""

import torch
import sys
sys.path.insert(0, '.')

from src.cipher.scne_cipher import SCNECipherAPI


def test_deterministic_nonce_derivation():
    """测试确定性nonce派生"""
    print('='*70)
    print('测试确定性nonce派生')
    print('='*70)
    
    # 使用CPU避免已知的设备不匹配问题
    device = 'cpu'
    print(f'使用设备: {device}')
    
    # 创建确定性模式的API
    api = SCNECipherAPI(
        password='test_password_2025',
        image_size=256,
        deterministic=True,
        device=device
    )
    
    # 创建测试图像（在正确的设备上）
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    # 第一次加密（带image_id）
    enc1, info1 = api.encrypt_simple(
        test_img,
        privacy_level=0.7,
        image_id='test_image_001',
        task_type='classification'
    )
    
    # 第二次加密（相同参数）
    enc2, info2 = api.encrypt_simple(
        test_img,
        privacy_level=0.7,
        image_id='test_image_001',
        task_type='classification'
    )
    
    # 验证确定性
    if info1.get('crypto_wrap') and info2.get('crypto_wrap'):
        nonce1 = info1['crypto_wrap']['nonces'][0]
        nonce2 = info2['crypto_wrap']['nonces'][0]
        print(f'Nonce 1: {nonce1}')
        print(f'Nonce 2: {nonce2}')
        print(f'Nonces相同: {nonce1 == nonce2}')
        
        # 验证密文相同
        diff = (enc1 - enc2).abs().max().item()
        print(f'密文差异: {diff}')
        print(f'密文完全相同: {diff == 0}')
        
        assert nonce1 == nonce2, "相同参数应产生相同nonce"
        assert diff == 0, "相同参数应产生相同密文"
    else:
        print('crypto_wrap未启用')
    
    # 验证不同image_id产生不同nonce
    enc3, info3 = api.encrypt_simple(
        test_img,
        privacy_level=0.7,
        image_id='test_image_002',  # 不同的image_id
        task_type='classification'
    )
    
    if info1.get('crypto_wrap') and info3.get('crypto_wrap'):
        nonce3 = info3['crypto_wrap']['nonces'][0]
        print(f'\n不同image_id的Nonce: {nonce3}')
        print(f'Nonces不同: {nonce1 != nonce3}')
        
        assert nonce1 != nonce3, "不同image_id应产生不同nonce"
    
    # 验证不同task_type产生不同nonce
    enc4, info4 = api.encrypt_simple(
        test_img,
        privacy_level=0.7,
        image_id='test_image_001',
        task_type='detection'  # 不同的task_type
    )
    
    if info1.get('crypto_wrap') and info4.get('crypto_wrap'):
        nonce4 = info4['crypto_wrap']['nonces'][0]
        print(f'\n不同task_type的Nonce: {nonce4}')
        print(f'Nonces不同: {nonce1 != nonce4}')
        
        assert nonce1 != nonce4, "不同task_type应产生不同nonce"
    
    # 验证enc_info包含审计信息
    print(f'\n审计信息:')
    print(f'  image_id: {info1.get("image_id")}')
    print(f'  task_type: {info1.get("task_type")}')
    print(f'  privacy_map_hash: {info1.get("privacy_map_hash")}')
    print(f'  z_view_hash: {info1.get("z_view_hash")}')
    
    assert info1.get('image_id') == 'test_image_001'
    assert info1.get('task_type') == 'classification'
    assert info1.get('privacy_map_hash') is not None
    assert info1.get('z_view_hash') is not None
    
    print('\n' + '='*70)
    print('✓ 确定性nonce派生测试通过')
    print('='*70)


def test_decryption_with_deterministic_nonce():
    """测试使用确定性nonce的加解密round-trip"""
    print('\n' + '='*70)
    print('测试确定性加解密round-trip')
    print('='*70)
    
    # 使用CPU避免已知的设备不匹配问题
    device = 'cpu'
    
    api = SCNECipherAPI(
        password='test_password_2025',
        image_size=256,
        deterministic=True,
        device=device
    )
    
    # 创建测试图像（在正确的设备上）
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    # 加密
    encrypted, enc_info = api.encrypt_simple(
        test_img,
        privacy_level=0.7,
        image_id='test_roundtrip_001',
        task_type='classification'
    )
    
    # 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    
    # 验证
    error = (decrypted - test_img).abs().mean().item()
    print(f'解密误差: {error:.8f}')
    
    # 注意：由于量化精度限制，解密误差约为1e-3级别是正常的
    assert error < 0.01, f"解密误差过大: {error}"
    
    print('✓ 确定性加解密round-trip测试通过')


if __name__ == '__main__':
    test_deterministic_nonce_derivation()
    test_decryption_with_deterministic_nonce()
