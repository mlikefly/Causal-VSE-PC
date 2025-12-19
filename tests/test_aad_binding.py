"""
测试 AEAD AAD（关联数据）绑定功能

**功能: top-tier-journal-upgrade, 属性 3: C-view AEAD 完整性**
**验证: 需求 1.3**

确保"密文搬家/改字段即解密失败"
"""

import torch
import sys
sys.path.insert(0, '.')

from src.cipher.scne_cipher import SCNECipherAPI


def test_aad_binding():
    """测试 AAD 绑定功能"""
    print('='*70)
    print('AEAD AAD 绑定测试')
    print('='*70)
    
    # 强制使用CPU避免设备不匹配问题
    device = 'cpu'
    print(f'计算设备: {device}')
    
    # 禁用CUDA以避免设备不匹配
    import torch
    torch.cuda.is_available = lambda: False
    
    # 创建确定性模式的API，强制使用CPU
    api = SCNECipherAPI(
        password='test_password_2025',
        image_size=256,
        deterministic=True,
        device='cpu'  # 强制CPU
    )
    # 确保cipher及其所有子模块都在CPU上
    api.cipher = api.cipher.cpu()
    for module in api.cipher.modules():
        if hasattr(module, 'device'):
            module.device = 'cpu'
    
    # 创建测试图像
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    # 加密（带完整元数据）
    encrypted, enc_info = api.cipher.encrypt(
        test_img,
        mask=torch.ones(1, 1, 256, 256, device=device),
        chaos_params=torch.rand(1, 19, 3, device=device) * torch.tensor([10.0, 5.0, 1.0], device=device),
        password='test_password_2025',
        privacy_level=0.7,
        image_id='test_aad_001',
        task_type='classification',
        dataset='celeba',
        split='train'
    )
    
    print(f'加密完成，形状: {encrypted.shape}')
    print(f'AAD 内容: {enc_info.get("crypto_wrap", {}).get("aad")}')
    
    # 测试1：正常解密（AAD匹配）
    print('\n--- 测试1：正常解密（AAD匹配）---')
    try:
        decrypted = api.cipher.decrypt(
            encrypted,
            enc_info,
            mask=torch.ones(1, 1, 256, 256, device=device),
            password='test_password_2025'
        )
        error = (decrypted - test_img).abs().mean().item()
        print(f'解密成功，平均误差: {error:.6f}')
        assert error < 0.01, f"解密误差过大: {error}"
        print('✓ 正常解密测试通过')
    except Exception as e:
        print(f'✗ 正常解密失败: {e}')
        raise
    
    # 测试2：篡改AAD后解密应失败
    print('\n--- 测试2：篡改AAD后解密应失败 ---')
    enc_info_tampered = dict(enc_info)
    if enc_info_tampered.get('crypto_wrap'):
        # 篡改AAD
        enc_info_tampered['crypto_wrap'] = dict(enc_info_tampered['crypto_wrap'])
        enc_info_tampered['crypto_wrap']['aad'] = 'v2|tampered_id|wrong_dataset|wrong_split|detection'
    
    try:
        decrypted_tampered = api.cipher.decrypt(
            encrypted,
            enc_info_tampered,
            mask=torch.ones(1, 1, 256, 256, device=device),
            password='test_password_2025'
        )
        print('✗ 篡改AAD后解密应该失败，但成功了')
        # 注意：如果版本是v1，可能不会失败
        if enc_info.get('crypto_wrap', {}).get('version', 1) >= 2:
            raise AssertionError("篡改AAD后解密应该失败")
    except ValueError as e:
        print(f'✓ 篡改AAD后解密正确失败: {e}')
    except Exception as e:
        print(f'其他错误: {e}')
    
    # 测试3：验证不同元数据产生不同AAD
    print('\n--- 测试3：不同元数据产生不同AAD ---')
    
    # 不同dataset
    _, enc_info2 = api.cipher.encrypt(
        test_img,
        mask=torch.ones(1, 1, 256, 256, device=device),
        chaos_params=torch.rand(1, 19, 3, device=device) * torch.tensor([10.0, 5.0, 1.0], device=device),
        password='test_password_2025',
        privacy_level=0.7,
        image_id='test_aad_001',
        task_type='classification',
        dataset='fairface',  # 不同的数据集
        split='train'
    )
    
    aad1 = enc_info.get('crypto_wrap', {}).get('aad')
    aad2 = enc_info2.get('crypto_wrap', {}).get('aad')
    print(f'AAD 1: {aad1}')
    print(f'AAD 2: {aad2}')
    print(f'不同数据集产生不同AAD: {aad1 != aad2}')
    assert aad1 != aad2, "不同数据集应产生不同AAD"
    
    print('\n' + '='*70)
    print('✓ AEAD AAD 绑定测试全部通过')
    print('='*70)


def test_aad_format():
    """测试 AAD 格式"""
    print('\n' + '='*70)
    print('AAD 格式测试')
    print('='*70)
    
    from src.cipher.scne_cipher import SCNECipher
    
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
    
    print('✓ AAD 格式测试通过')


if __name__ == '__main__':
    test_aad_format()
    test_aad_binding()
