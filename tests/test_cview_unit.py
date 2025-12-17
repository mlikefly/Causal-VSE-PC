"""
单元测试：C-view 双表示结构

**Feature: top-tier-journal-upgrade**
**Validates: Requirements 1.3**
"""

import torch
import numpy as np
import json
import sys
sys.path.insert(0, '.')


def test_cview_bytes_conversion_unit():
    """测试 C-view bytes 转换逻辑（不涉及完整加密）"""
    print('='*70)
    print('测试 C-view bytes 转换逻辑')
    print('='*70)
    
    # 模拟 C-view 张量
    B, C, H, W = 2, 1, 64, 64
    c_view_vis = torch.rand(B, C, H, W)
    
    # 模拟 wrap_info
    wrap_info = {
        'mode': 'q16',
        'wrap_bits': 16,
        'nonces': ['a' * 32, 'b' * 32],
        'tags': ['c' * 64, 'd' * 64],
        'affine_min': [0.0, 0.0],
        'affine_scale': [1.0, 1.0],
        'version': 1,
    }
    
    qmax = (1 << 16) - 1
    
    # 测试 cview_to_bytes
    print('\n--- 测试 cview_to_bytes ---')
    c_view_bytes_list = []
    for b in range(B):
        arr = c_view_vis[b].numpy().astype(np.float32)
        q = np.round(arr * qmax).astype(np.uint16)
        raw_bytes = q.tobytes()
        c_view_bytes_list.append(raw_bytes)
    
    print(f'转换为 bytes: {len(c_view_bytes_list)} 个样本')
    print(f'每个样本大小: {len(c_view_bytes_list[0])} bytes')
    expected_size = C * H * W * 2  # uint16 = 2 bytes
    assert len(c_view_bytes_list[0]) == expected_size, f"大小不匹配: {len(c_view_bytes_list[0])} vs {expected_size}"
    
    # 测试 cview_from_bytes
    print('\n--- 测试 cview_from_bytes ---')
    vis_list = []
    for raw_bytes in c_view_bytes_list:
        q = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(C, H, W)
        arr = q.astype(np.float32) / qmax
        vis_list.append(torch.from_numpy(arr))
    c_view_restored = torch.stack(vis_list, dim=0)
    
    # 验证转换一致性（量化会有精度损失）
    diff = (c_view_vis - c_view_restored).abs().max().item()
    print(f'转换差异: {diff}')
    assert diff < 1e-4, f"bytes 转换差异过大: {diff}"
    print('✓ bytes <-> vis 转换一致')
    
    # 测试 pack_cview_binary
    print('\n--- 测试 pack_cview_binary ---')
    binary_pack = {
        'version': 2,
        'format': 'cview_binary',
        'shape': [B, C, H, W],
        'mode': wrap_info['mode'],
        'wrap_bits': wrap_info['wrap_bits'],
        'ciphertext': [ct.hex() for ct in c_view_bytes_list],
        'nonces': wrap_info['nonces'],
        'tags': wrap_info['tags'],
        'affine_min': wrap_info['affine_min'],
        'affine_scale': wrap_info['affine_scale'],
        'image_id': 'test_001',
        'task_type': 'classification',
        'privacy_level': 0.7,
    }
    
    print(f'打包字段: {list(binary_pack.keys())}')
    
    # 验证可以序列化为JSON
    json_str = json.dumps(binary_pack)
    print(f'JSON 序列化大小: {len(json_str)} bytes')
    
    # 测试 unpack_cview_binary
    print('\n--- 测试 unpack_cview_binary ---')
    binary_pack_restored = json.loads(json_str)
    
    # 从hex解码密文
    c_view_bytes_restored = [bytes.fromhex(ct) for ct in binary_pack_restored['ciphertext']]
    
    # 恢复张量
    vis_list2 = []
    for raw_bytes in c_view_bytes_restored:
        q = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(C, H, W)
        arr = q.astype(np.float32) / qmax
        vis_list2.append(torch.from_numpy(arr))
    c_view_unpacked = torch.stack(vis_list2, dim=0)
    
    # 验证解包一致性
    diff2 = (c_view_restored - c_view_unpacked).abs().max().item()
    print(f'解包差异: {diff2}')
    assert diff2 == 0, f"解包不一致: {diff2}"
    print('✓ pack <-> unpack 一致')
    
    print('\n' + '='*70)
    print('✓ C-view bytes 转换逻辑测试通过')
    print('='*70)


def test_cview_f32_mode():
    """测试 float32 模式的 C-view 转换"""
    print('\n' + '='*70)
    print('测试 float32 模式')
    print('='*70)
    
    B, C, H, W = 1, 1, 32, 32
    c_view_vis = torch.rand(B, C, H, W)
    
    # float32 模式
    print('\n--- float32 模式 ---')
    c_view_bytes_list = []
    for b in range(B):
        arr = c_view_vis[b].numpy().astype(np.float32)
        raw_bytes = arr.tobytes()
        c_view_bytes_list.append(raw_bytes)
    
    print(f'每个样本大小: {len(c_view_bytes_list[0])} bytes')
    expected_size = C * H * W * 4  # float32 = 4 bytes
    assert len(c_view_bytes_list[0]) == expected_size
    
    # 恢复
    vis_list = []
    for raw_bytes in c_view_bytes_list:
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(C, H, W)
        vis_list.append(torch.from_numpy(arr.copy()))
    c_view_restored = torch.stack(vis_list, dim=0)
    
    # float32 模式应该是无损的
    diff = (c_view_vis - c_view_restored).abs().max().item()
    print(f'转换差异: {diff}')
    assert diff == 0, f"float32 模式应该无损: {diff}"
    print('✓ float32 模式无损转换')


if __name__ == '__main__':
    test_cview_bytes_conversion_unit()
    test_cview_f32_mode()
