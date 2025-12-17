"""
属性测试：加密系统的正确性属性（独立运行版本）

**Feature: top-tier-journal-upgrade**
**Property 2: Deterministic Encryption Round-Trip**
**Property 3: C-view AEAD Integrity**
**Validates: Requirements 1.3, 1.6, 9.1, 9.2**
"""

import torch
import numpy as np
import hashlib
import hmac
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cipher.scne_cipher import SCNECipherAPI, SCNECipher
from src.crypto.key_system import HierarchicalKeySystem


# ============ Property 2: Deterministic Encryption Round-Trip ============

def test_deterministic_encryption_round_trip():
    """
    **Feature: top-tier-journal-upgrade, Property 2: Deterministic Encryption Round-Trip**
    **Validates: Requirements 1.6, 9.1, 9.2**
    
    *For any* master_key, image_id, task_type, and privacy_map combination, 
    encrypting the same image multiple times SHALL produce identical outputs.
    """
    print('='*70)
    print('Property 2: Deterministic Encryption Round-Trip')
    print('='*70)
    
    password = "test_property_password_2025"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    
    # 测试参数组合
    privacy_levels = [0.3, 0.5, 0.7, 1.0]
    task_types = ['classification', 'detection', 'segmentation']
    
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        deterministic=True,
        device=device
    )
    
    passed = 0
    failed = 0
    
    for privacy_level in privacy_levels:
        for task_type in task_types:
            # 固定随机种子
            torch.manual_seed(42)
            test_img = torch.rand(1, 1, 256, 256, device=device)
            image_id = f"test_{privacy_level}_{task_type}"
            
            # 第一次加密
            enc1, info1 = api.encrypt_simple(
                test_img.clone(),
                privacy_level=privacy_level,
                image_id=image_id,
                task_type=task_type
            )
            
            # 第二次加密
            enc2, info2 = api.encrypt_simple(
                test_img.clone(),
                privacy_level=privacy_level,
                image_id=image_id,
                task_type=task_type
            )
            
            # 验证
            diff = (enc1 - enc2).abs().max().item()
            nonce_match = True
            if info1.get('crypto_wrap') and info2.get('crypto_wrap'):
                nonce1 = info1['crypto_wrap'].get('nonces', [])
                nonce2 = info2['crypto_wrap'].get('nonces', [])
                nonce_match = (nonce1 == nonce2)
            
            if diff == 0 and nonce_match:
                passed += 1
                print(f'  ✓ privacy={privacy_level}, task={task_type}')
            else:
                failed += 1
                print(f'  ✗ privacy={privacy_level}, task={task_type} - diff={diff}, nonce_match={nonce_match}')
    
    print(f'\n结果: {passed} 通过, {failed} 失败')
    assert failed == 0, f"Property 2 测试失败: {failed} 个用例未通过"
    print('✓ Property 2 测试通过')
    return True


def test_different_inputs_produce_different_nonces():
    """
    测试不同输入产生不同nonce
    """
    print('\n' + '='*70)
    print('测试不同输入产生不同nonce')
    print('='*70)
    
    password = "test_property_password_2025"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        deterministic=True,
        device=device
    )
    
    torch.manual_seed(42)
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    # 收集不同参数组合的nonce
    nonces = {}
    
    for task_type in ['classification', 'detection', 'segmentation']:
        _, info = api.encrypt_simple(
            test_img.clone(),
            privacy_level=0.7,
            image_id='test_image_001',
            task_type=task_type
        )
        if info.get('crypto_wrap'):
            nonces[task_type] = info['crypto_wrap'].get('nonces', [None])[0]
    
    # 验证不同task_type产生不同nonce
    unique_nonces = set(nonces.values())
    assert len(unique_nonces) == 3, f"不同task_type应产生不同nonce，但只有{len(unique_nonces)}个唯一nonce"
    
    print('✓ 不同task_type产生不同nonce')
    return True


# ============ Property 3: C-view AEAD Integrity ============

def test_aead_integrity():
    """
    **Feature: top-tier-journal-upgrade, Property 3: C-view AEAD Integrity**
    **Validates: Requirements 1.3**
    
    *For any* C-view ciphertext, decryption with the correct key SHALL succeed;
    decryption with tampered AAD SHALL fail.
    """
    print('\n' + '='*70)
    print('Property 3: C-view AEAD Integrity')
    print('='*70)
    
    password = "test_aead_password_2025"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        deterministic=True,
        device=device
    )
    
    # 测试1：正确密钥解密成功
    print('\n--- 测试1：正确密钥解密成功 ---')
    
    test_cases = [
        {'privacy_level': 0.5, 'task_type': 'classification', 'dataset': 'celeba', 'split': 'train'},
        {'privacy_level': 0.7, 'task_type': 'detection', 'dataset': 'fairface', 'split': 'val'},
        {'privacy_level': 1.0, 'task_type': 'segmentation', 'dataset': 'openimages', 'split': 'test'},
    ]
    
    for tc in test_cases:
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=device)
        
        encrypted, enc_info = api.cipher.encrypt(
            test_img,
            mask=torch.ones(1, 1, 256, 256, device=device),
            chaos_params=torch.rand(1, 19, 3, device=device) * torch.tensor([10.0, 5.0, 1.0], device=device),
            password=password,
            privacy_level=tc['privacy_level'],
            image_id='test_aead_001',
            task_type=tc['task_type'],
            dataset=tc['dataset'],
            split=tc['split']
        )
        
        decrypted = api.cipher.decrypt(
            encrypted,
            enc_info,
            mask=torch.ones(1, 1, 256, 256, device=device),
            password=password
        )
        
        error = (decrypted - test_img).abs().mean().item()
        assert error < 0.01, f"解密误差过大: {error}"
        print(f'  ✓ {tc}')
    
    # 测试2：篡改AAD解密失败
    print('\n--- 测试2：篡改AAD解密失败 ---')
    
    torch.manual_seed(42)
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    encrypted, enc_info = api.cipher.encrypt(
        test_img,
        mask=torch.ones(1, 1, 256, 256, device=device),
        chaos_params=torch.rand(1, 19, 3, device=device) * torch.tensor([10.0, 5.0, 1.0], device=device),
        password=password,
        privacy_level=0.7,
        image_id='test_tamper_001',
        task_type='classification',
        dataset='celeba',
        split='train'
    )
    
    # 篡改AAD
    enc_info_tampered = dict(enc_info)
    if enc_info_tampered.get('crypto_wrap'):
        enc_info_tampered['crypto_wrap'] = dict(enc_info_tampered['crypto_wrap'])
        enc_info_tampered['crypto_wrap']['aad'] = 'v2|tampered|wrong|data|here'
    
    try:
        api.cipher.decrypt(
            encrypted,
            enc_info_tampered,
            mask=torch.ones(1, 1, 256, 256, device=device),
            password=password
        )
        raise AssertionError("篡改AAD后解密应该失败")
    except ValueError as e:
        assert "MAC verification failed" in str(e)
        print(f'  ✓ 篡改AAD正确触发MAC验证失败')
    
    print('\n✓ Property 3 测试通过')
    return True


def test_aad_format():
    """测试AAD格式"""
    print('\n' + '='*70)
    print('测试AAD格式')
    print('='*70)
    
    cipher = SCNECipher(password='test')
    
    # 测试完整AAD
    aad = cipher._build_aad(
        sample_id='sample_001',
        dataset='celeba',
        split='train',
        task_type='classification',
        version=2
    )
    print(f'完整AAD: {aad}')
    assert b'v2' in aad
    assert b'sample_001' in aad
    assert b'celeba' in aad
    assert b'train' in aad
    assert b'classification' in aad
    
    # 测试部分AAD
    aad_partial = cipher._build_aad(
        sample_id='sample_002',
        task_type='detection'
    )
    print(f'部分AAD: {aad_partial}')
    assert b'sample_002' in aad_partial
    assert b'detection' in aad_partial
    
    print('✓ AAD格式测试通过')
    return True


def test_nonce_uniqueness():
    """测试nonce唯一性"""
    print('\n' + '='*70)
    print('测试nonce唯一性')
    print('='*70)
    
    key_system = HierarchicalKeySystem('test_password', iterations=10000)
    
    def derive_nonce(image_id, task_type, pm_hash, zv_hash):
        context = f"cview|v2|{image_id}|{task_type}|{pm_hash}|{zv_hash}".encode('utf-8')
        k_nonce = hashlib.sha256(key_system.master_key + b'|nonce').digest()
        return hmac.new(k_nonce, context, hashlib.sha256).digest()[:16]
    
    nonces = set()
    for i in range(100):
        for task in ['classification', 'detection', 'segmentation']:
            pm_hash = hashlib.sha256(f'pm_{i}_{task}'.encode()).hexdigest()[:8]
            zv_hash = hashlib.sha256(f'zv_{i}_{task}'.encode()).hexdigest()[:8]
            nonce = derive_nonce(f'image_{i:04d}', task, pm_hash, zv_hash)
            nonces.add(nonce)
    
    print(f'生成的nonce数量: {len(nonces)}')
    print(f'期望的唯一nonce数量: 300')
    assert len(nonces) == 300, f"应有300个唯一nonce，实际{len(nonces)}个"
    
    print('✓ nonce唯一性测试通过')
    return True


if __name__ == '__main__':
    print('='*70)
    print('属性测试：加密系统正确性')
    print('='*70)
    
    all_passed = True
    
    try:
        test_aad_format()
    except Exception as e:
        print(f'✗ AAD格式测试失败: {e}')
        all_passed = False
    
    try:
        test_nonce_uniqueness()
    except Exception as e:
        print(f'✗ nonce唯一性测试失败: {e}')
        all_passed = False
    
    try:
        test_deterministic_encryption_round_trip()
    except Exception as e:
        print(f'✗ Property 2 测试失败: {e}')
        all_passed = False
    
    try:
        test_different_inputs_produce_different_nonces()
    except Exception as e:
        print(f'✗ 不同输入测试失败: {e}')
        all_passed = False
    
    try:
        test_aead_integrity()
    except Exception as e:
        print(f'✗ Property 3 测试失败: {e}')
        all_passed = False
    
    print('\n' + '='*70)
    if all_passed:
        print('✓ 所有属性测试通过！')
    else:
        print('✗ 部分测试失败')
    print('='*70)
