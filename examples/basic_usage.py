#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCNE基础使用示例

演示如何使用SCNE进行图像加密和解密
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from PIL import Image

from src.cipher.scne_cipher import SCNECipherAPI


def example_1_basic_encryption():
    """示例1：基础加密和解密"""
    print("="*70)
    print("示例1：基础加密和解密")
    print("="*70)
    
    # 1. 初始化API
    api = SCNECipherAPI(
        password="my_secure_password_2025",
        image_size=256
    )
    
    # 2. 创建测试图像（或从文件加载）
    image = torch.rand(1, 1, 256, 256)  # 随机图像
    print(f"原始图像大小: {image.shape}")
    
    # 3. 加密
    encrypted, enc_info = api.encrypt_simple(
        image,
        privacy_level=1.0,          # 最高隐私
        semantic_preserving=False   # 完全混淆
    )
    print(f"加密完成，密文大小: {encrypted.shape}")
    
    # 4. 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    print(f"解密完成，恢复图像大小: {decrypted.shape}")
    
    # 5. 验证
    error = (decrypted - image).abs().mean().item()
    mae = (encrypted - image).abs().mean().item()
    
    print(f"\n结果:")
    print(f"  加密强度 (MAE): {mae:.4f} (目标: >0.3)")
    print(f"  解密误差: {error:.6f} (目标: <0.02)")
    print(f"  状态: {'✓ 成功' if error < 0.02 else '✗ 失败'}")
    print()


def example_2_privacy_levels():
    """示例2：不同隐私级别"""
    print("="*70)
    print("示例2：不同隐私级别")
    print("="*70)
    
    api = SCNECipherAPI(password="test_password")
    image = torch.rand(1, 1, 256, 256)
    
    privacy_levels = [0.3, 0.5, 0.7, 1.0]
    
    print(f"\n{'隐私级别':<10} {'加密强度(MAE)':<15} {'解密误差':<12} {'适用场景':<20}")
    print("-" * 70)
    
    for privacy in privacy_levels:
        encrypted, enc_info = api.encrypt_simple(
            image,
            privacy_level=privacy,
            semantic_preserving=True if privacy < 1.0 else False
        )
        decrypted = api.decrypt_simple(encrypted, enc_info)
        
        mae = (encrypted - image).abs().mean().item()
        error = (decrypted - image).abs().mean().item()
        
        scenarios = {
            0.3: "云端隐私计算",
            0.5: "可搜索加密",
            0.7: "半信任环境",
            1.0: "完全存储加密"
        }
        
        print(f"{privacy:<10.1f} {mae:<15.4f} {error:<12.6f} {scenarios[privacy]:<20}")
    print()


def example_3_batch_processing():
    """示例3：批量处理"""
    print("="*70)
    print("示例3：批量处理多张图像")
    print("="*70)
    
    api = SCNECipherAPI(password="batch_test_2025")
    
    # 创建批量图像
    batch_size = 5
    images = [torch.rand(1, 1, 256, 256) for _ in range(batch_size)]
    
    print(f"\n批量加密 {batch_size} 张图像...")
    
    encrypted_list = []
    enc_info_list = []
    
    import time
    start = time.time()
    
    for i, img in enumerate(images):
        enc, info = api.encrypt_simple(img)
        encrypted_list.append(enc)
        enc_info_list.append(info)
        print(f"  [{i+1}/{batch_size}] 加密完成")
    
    encrypt_time = time.time() - start
    
    print(f"\n批量解密 {batch_size} 张图像...")
    start = time.time()
    
    decrypted_list = []
    for i, (enc, info) in enumerate(zip(encrypted_list, enc_info_list)):
        dec = api.decrypt_simple(enc, info)
        decrypted_list.append(dec)
        print(f"  [{i+1}/{batch_size}] 解密完成")
    
    decrypt_time = time.time() - start
    
    # 验证
    errors = [(decrypted_list[i] - images[i]).abs().mean().item() 
              for i in range(batch_size)]
    avg_error = np.mean(errors)
    
    print(f"\n结果:")
    print(f"  总加密时间: {encrypt_time:.3f}s ({encrypt_time/batch_size:.3f}s/张)")
    print(f"  总解密时间: {decrypt_time:.3f}s ({decrypt_time/batch_size:.3f}s/张)")
    print(f"  平均解密误差: {avg_error:.6f}")
    print()


def example_4_load_from_file():
    """示例4：从文件加载图像"""
    print("="*70)
    print("示例4：从文件加载图像（如果存在）")
    print("="*70)
    
    # 查找测试图像
    test_paths = [
        "../image/gray_sample0.png",
        "../data/CelebA-HQ/test/00000.png",
        "../data/DIV2K_processed/valid/0801.png"
    ]
    
    image_path = None
    for path in test_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            image_path = full_path
            break
    
    if image_path is None:
        print("⚠️  未找到测试图像，跳过此示例")
        print()
        return
    
    print(f"✓ 找到测试图像: {image_path}")
    
    # 加载图像
    img_pil = Image.open(image_path).convert('L')
    img_pil = img_pil.resize((256, 256))
    img_array = np.array(img_pil) / 255.0
    image = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    print(f"图像加载完成: {image.shape}")
    
    # 加密
    api = SCNECipherAPI(password="file_test_2025")
    encrypted, enc_info = api.encrypt_simple(image, privacy_level=1.0)
    
    # 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    
    # 验证
    error = (decrypted - image).abs().mean().item()
    mae = (encrypted - image).abs().mean().item()
    
    print(f"\n结果:")
    print(f"  加密强度: {mae:.4f}")
    print(f"  解密误差: {error:.6f}")
    print(f"  状态: {'✓ 成功' if error < 0.02 else '✗ 失败'}")
    print()


def example_5_semantic_preserving():
    """示例5：语义保持模式（用于云端计算）"""
    print("="*70)
    print("示例5：语义保持模式")
    print("="*70)
    
    api = SCNECipherAPI(password="semantic_test")
    image = torch.rand(1, 1, 256, 256)
    
    # 完全加密模式
    print("\n[模式1] 完全加密 (privacy=1.0, semantic_preserving=False)")
    enc1, info1 = api.encrypt_simple(
        image,
        privacy_level=1.0,
        semantic_preserving=False
    )
    dec1 = api.decrypt_simple(enc1, info1)
    mae1 = (enc1 - image).abs().mean().item()
    err1 = (dec1 - image).abs().mean().item()
    print(f"  加密强度: {mae1:.4f}")
    print(f"  解密误差: {err1:.6f}")
    print(f"  适用场景: 敏感数据存储、传输")
    
    # 语义保持模式
    print("\n[模式2] 语义保持 (privacy=0.3, semantic_preserving=True)")
    enc2, info2 = api.encrypt_simple(
        image,
        privacy_level=0.3,
        semantic_preserving=True
    )
    dec2 = api.decrypt_simple(enc2, info2)
    mae2 = (enc2 - image).abs().mean().item()
    err2 = (dec2 - image).abs().mean().item()
    print(f"  加密强度: {mae2:.4f}")
    print(f"  解密误差: {err2:.6f}")
    print(f"  适用场景: 云端AI处理、隐私计算")
    print(f"  特点: 云端能执行某些任务，但无法识别具体身份")
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("SCNE使用示例集")
    print("="*70 + "\n")
    
    try:
        example_1_basic_encryption()
        example_2_privacy_levels()
        example_3_batch_processing()
        example_4_load_from_file()
        example_5_semantic_preserving()
        
        print("="*70)
        print("✓ 所有示例运行完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()















