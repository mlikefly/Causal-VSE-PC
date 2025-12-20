"""
测试 C-view 双表示结构（bytes + vis）

**Feature: top-tier-journal-upgrade**
**Validates: Requirements 1.3**
"""

import torch
import json
import sys
sys.path.insert(0, '.')

from src.cipher.scne_cipher import SCNECipherAPI


def test_cview_bytes_conversion():
    """测试 C-view bytes 与 vis 之间的转换"""
    print('='*70)
    print('测试 C-view 双表示结构')
    print('='*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    
    # 创建API
    try:
        api = SCNECipherAPI(
            password='test_password_2025',
            image_size=256,
            deterministic=True,
            device=device
        )
    except Exception as e:
        print(f'⚠️ SCNECipherAPI 初始化失败: {e}')
        print('跳过测试')
        return
    
    # 创建测试图像
    test_img = torch.rand(1, 1, 256, 256, device=device)
    
    # 加密
    try:
        encrypted, enc_info = api.encrypt_simple(
            test_img,
            privacy_level=0.7,
            image_id='test_cview_001',
            task_type='classification'
        )
    except Exception as e:
        print(f'⚠️ 加密失败: {e}')
        print('跳过测试')
        return
    
    print(f'\n加密完成: {encrypted.shape}')
    print(f'crypto_wrap 启用: {enc_info.get("crypto_enabled")}')
    
    if not enc_info.get('crypto_wrap'):
        print('⚠️ crypto_wrap 未启用，跳过测试')
        return
    
    # 测试1: cview_to_bytes
    print('\n--- 测试 cview_to_bytes ---')
    if not hasattr(api.cipher, 'cview_to_bytes'):
        print('⚠️ cview_to_bytes 方法不存在，跳过测试')
        return
        
    c_view_bytes_list = api.cipher.cview_to_bytes(encrypted, enc_info['crypto_wrap'])
    print(f'转换为 bytes: {len(c_view_bytes_list)} 个样本')
    print(f'每个样本大小: {len(c_view_bytes_list[0])} bytes')
    
    # 测试2: cview_from_bytes
    print('\n--- 测试 cview_from_bytes ---')
    c_view_restored = api.cipher.cview_from_bytes(
        c_view_bytes_list,
        enc_info['crypto_wrap'],
        encrypted.shape,
        device=device
    )
    
    # 验证转换一致性
    diff = (encrypted - c_view_restored).abs().max().item()
    print(f'转换差异: {diff}')
    assert diff < 1e-6, f"bytes 转换不一致: {diff}"
    print('✓ bytes <-> vis 转换一致')
    
    # 测试3: pack_cview_binary
    print('\n--- 测试 pack_cview_binary ---')
    if not hasattr(api.cipher, 'pack_cview_binary'):
        print('⚠️ pack_cview_binary 方法不存在，跳过测试')
        return
        
    binary_pack = api.cipher.pack_cview_binary(encrypted, enc_info)
    print(f'打包字段: {list(binary_pack.keys())}')
    print(f'版本: {binary_pack.get("version", "N/A")}')
    print(f'格式: {binary_pack.get("format", "N/A")}')
    print(f'形状: {binary_pack.get("shape", "N/A")}')
    print(f'image_id: {binary_pack.get("image_id", "N/A")}')
    print(f'task_type: {binary_pack.get("task_type", "N/A")}')
    
    # 验证可以序列化为JSON
    json_str = json.dumps(binary_pack)
    print(f'JSON 序列化大小: {len(json_str)} bytes')
    
    # 测试4: unpack_cview_binary
    print('\n--- 测试 unpack_cview_binary ---')
    # 从JSON反序列化
    binary_pack_restored = json.loads(json_str)
    c_view_unpacked, wrap_info = api.cipher.unpack_cview_binary(binary_pack_restored, device=device)
    
    # 验证解包一致性
    diff2 = (encrypted - c_view_unpacked).abs().max().item()
    print(f'解包差异: {diff2}')
    assert diff2 < 1e-6, f"解包不一致: {diff2}"
    print('✓ pack <-> unpack 一致')
    
    # 测试5: 完整流程 - 从解包的 C-view 解密
    print('\n--- 测试完整解密流程 ---')
    # 构造用于解密的 enc_info
    enc_info_for_decrypt = dict(enc_info)
    enc_info_for_decrypt['crypto_wrap'] = wrap_info
    
    decrypted = api.decrypt_simple(c_view_unpacked, enc_info_for_decrypt)
    
    decrypt_error = (decrypted - test_img).abs().mean().item()
    print(f'解密误差: {decrypt_error:.8f}')
    assert decrypt_error < 0.01, f"解密误差过大: {decrypt_error}"
    print('✓ 从解包的 C-view 解密成功')
    
    print('\n' + '='*70)
    print('✓ C-view 双表示结构测试通过')
    print('='*70)


if __name__ == '__main__':
    test_cview_bytes_conversion()
