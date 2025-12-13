#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCNE高级使用示例

演示高级功能：自定义参数、性能优化、安全分析等
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import time

from src.cipher.scne_cipher import SCNECipher, SCNECipherAPI
from src.crypto.key_system import HierarchicalKeySystem


def example_1_custom_parameters():
    """示例1：自定义加密参数"""
    print("="*70)
    print("示例1：自定义加密参数")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用核心SCNECipher（更灵活）
    cipher = SCNECipher(
        image_size=256,
        use_frequency=True,
        use_fft=True,
        password="custom_params_test",
        device=device
    )
    
    image = torch.rand(1, 1, 256, 256, device=device)
    mask = torch.ones(1, 1, 256, 256, device=device)
    
    # 自定义chaos_params
    # [iterations, complexity, strength]
    chaos_params = torch.tensor([
        [[20.0, 8.0, 0.9]],  # 高强度：20次迭代
        [[15.0, 5.0, 0.7]],  # 中等强度
        [[10.0, 3.0, 0.5]],  # 较低强度
        # ... 19个区域
    ] + [[[15.0, 5.0, 0.7]]] * 16, device=device)
    
    print(f"\n自定义参数:")
    print(f"  Arnold迭代次数: {chaos_params[0, 0, 0]:.0f}")
    print(f"  混沌复杂度: {chaos_params[0, 0, 1]:.0f}")
    print(f"  扩散强度: {chaos_params[0, 0, 2]:.2f}")
    
    # 加密
    start = time.time()
    encrypted, enc_info = cipher.encrypt(
        image, mask, chaos_params,
        privacy_level=1.0,
        semantic_preserving=False
    )
    encrypt_time = time.time() - start
    
    # 解密
    start = time.time()
    decrypted = cipher.decrypt(encrypted, enc_info, mask)
    decrypt_time = time.time() - start
    
    error = (decrypted - image).abs().mean().item()
    mae = (encrypted - image).abs().mean().item()
    
    print(f"\n结果:")
    print(f"  加密时间: {encrypt_time*1000:.2f}ms")
    print(f"  解密时间: {decrypt_time*1000:.2f}ms")
    print(f"  加密强度: {mae:.4f}")
    print(f"  解密误差: {error:.6f}")
    print()


def example_2_key_system():
    """示例2：密钥系统详解"""
    print("="*70)
    print("示例2：密钥系统层级派生")
    print("="*70)
    
    # 初始化密钥系统
    key_sys = HierarchicalKeySystem(
        password="demo_password_2025",
        iterations=100000  # PBKDF2迭代次数
    )
    
    print(f"\n[层级1] 主密钥:")
    print(f"  长度: {len(key_sys.master_key)*8} bits")
    print(f"  Salt: {key_sys.salt.hex()[:32]}...")
    
    # 图像密钥派生
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    image_key = key_sys.derive_image_key(test_image)
    
    print(f"\n[层级2] 图像密钥:")
    print(f"  长度: {len(image_key)*8} bits")
    print(f"  前16字节: {image_key[:16].hex()}")
    
    # 区域密钥派生
    region_keys = key_sys.derive_region_keys(image_key, num_regions=19)
    
    print(f"\n[层级3] 区域密钥:")
    print(f"  区域数量: {len(region_keys)}")
    print(f"  区域0密钥: {region_keys[0][:16].hex()}")
    print(f"  区域1密钥: {region_keys[1][:16].hex()}")
    
    # 混沌初值派生
    chaos_init = key_sys.derive_chaos_initial_state(region_keys[0], dimension=5)
    
    print(f"\n[层级4] 混沌初值:")
    print(f"  维度: {len(chaos_init)}")
    print(f"  初值: {chaos_init}")
    
    # S-box生成
    sbox = key_sys.generate_dynamic_sbox(region_keys[0])
    inv_sbox = key_sys.generate_inverse_sbox(sbox)
    
    print(f"\n[层级5] 动态S-box:")
    print(f"  S-box前10个: {sbox[:10].tolist()}")
    print(f"  逆S-box前10个: {inv_sbox[:10].tolist()}")
    print(f"  可逆性验证: {np.all(inv_sbox[sbox] == np.arange(256))}")
    
    # 密钥空间计算
    key_space = key_sys.compute_key_space()
    print(f"\n密钥空间: 2^{key_space:.2f}")
    print()


def example_3_performance_optimization():
    """示例3：性能优化"""
    print("="*70)
    print("示例3：性能优化对比")
    print("="*70)
    
    sizes = [128, 256, 512]
    api = SCNECipherAPI(password="perf_test")
    
    print(f"\n{'图像尺寸':<12} {'加密时间(ms)':<15} {'解密时间(ms)':<15} {'FPS':<10}")
    print("-" * 60)
    
    for size in sizes:
        image = torch.rand(1, 1, size, size)
        
        # 加密性能
        times_enc = []
        for _ in range(5):
            start = time.time()
            encrypted, enc_info = api.encrypt_simple(image)
            times_enc.append(time.time() - start)
        avg_enc = np.mean(times_enc) * 1000
        
        # 解密性能
        times_dec = []
        for _ in range(5):
            start = time.time()
            decrypted = api.decrypt_simple(encrypted, enc_info)
            times_dec.append(time.time() - start)
        avg_dec = np.mean(times_dec) * 1000
        
        fps = 1000 / (avg_enc + avg_dec)
        
        print(f"{size}x{size:<8} {avg_enc:<15.2f} {avg_dec:<15.2f} {fps:<10.2f}")
    
    print(f"\n设备: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print()


def example_4_security_analysis():
    """示例4：安全性分析"""
    print("="*70)
    print("示例4：密钥敏感性分析")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.rand(1, 1, 256, 256, device=device)
    
    # 使用密钥1加密
    api1 = SCNECipherAPI(password="password_A")
    encrypted1, _ = api1.encrypt_simple(image)
    
    # 使用密钥2加密（仅1字符差异）
    api2 = SCNECipherAPI(password="password_B")
    encrypted2, _ = api2.encrypt_simple(image)
    
    # 计算差异
    pixel_diff = (encrypted1 != encrypted2).float().mean().item()
    value_diff = (encrypted1 - encrypted2).abs().mean().item()
    
    print(f"\n密钥差异: 1个字符 ('A' vs 'B')")
    print(f"密文像素变化率: {pixel_diff*100:.2f}%")
    print(f"密文数值差异: {value_diff:.4f}")
    print(f"结论: {'✓ 强密钥敏感性' if pixel_diff > 0.90 else '⚠️ 密钥敏感性不足'}")
    
    # 单像素变化测试
    image_mod = image.clone()
    image_mod[0, 0, 128, 128] += 0.01  # 单像素微小变化
    
    encrypted_orig, _ = api1.encrypt_simple(image)
    encrypted_mod, _ = api1.encrypt_simple(image_mod)
    
    pixel_change = (encrypted_orig != encrypted_mod).float().mean().item()
    
    print(f"\n输入变化: 单像素+0.01")
    print(f"密文像素变化率(NPCR): {pixel_change*100:.2f}%")
    print(f"目标: >99%")
    print(f"结论: {'✓ 满足混淆要求' if pixel_change > 0.99 else '⚠️ 混淆不足'}")
    print()


def example_5_error_recovery():
    """示例5：错误处理和恢复"""
    print("="*70)
    print("示例5：错误密钥处理")
    print("="*70)
    
    api_correct = SCNECipherAPI(password="correct_password")
    api_wrong = SCNECipherAPI(password="wrong_password")
    
    image = torch.rand(1, 1, 256, 256)
    
    # 正确加密
    encrypted, enc_info = api_correct.encrypt_simple(image)
    
    # 正确解密
    print("\n[测试1] 使用正确密钥解密:")
    try:
        decrypted_correct = api_correct.decrypt_simple(encrypted, enc_info)
        error = (decrypted_correct - image).abs().mean().item()
        print(f"  解密误差: {error:.6f}")
        print(f"  状态: ✓ 成功")
    except Exception as e:
        print(f"  状态: ✗ 失败 - {e}")
    
    # 错误解密
    print("\n[测试2] 使用错误密钥解密:")
    try:
        decrypted_wrong = api_wrong.decrypt_simple(encrypted, enc_info)
        error = (decrypted_wrong - image).abs().mean().item()
        print(f"  解密误差: {error:.6f}")
        print(f"  状态: ⚠️ 产生乱码（这是预期行为）")
        print(f"  说明: 错误密钥不会报错，但解密结果是完全不同的乱码")
    except Exception as e:
        print(f"  状态: ✗ 异常 - {e}")
    print()


def main():
    """运行所有高级示例"""
    print("\n" + "="*70)
    print("SCNE高级使用示例")
    print("="*70 + "\n")
    
    try:
        example_1_custom_parameters()
        example_2_key_system()
        example_3_performance_optimization()
        example_4_security_analysis()
        example_5_error_recovery()
        
        print("="*70)
        print("✓ 所有高级示例运行完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()















