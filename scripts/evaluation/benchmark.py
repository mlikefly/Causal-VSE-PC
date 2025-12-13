#!/usr/bin/env python3
"""
性能基准测试脚本
测试优化前后的加密/解密速度
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cipher.scne_cipher import SCNECipherAPI
from src.evaluation.security_metrics import SecurityMetrics


def benchmark_performance(
    batch_size: int = 4,
    image_size: int = 256,
    num_iterations: int = 10,
    device: str = 'cuda',
    use_optimized: bool = True
):
    """测试加密解密性能"""
    
    print("\n" + "="*60)
    print("SCNE 性能基准测试")
    print("="*60)
    print(f"配置:")
    print(f"  批量大小: {batch_size}")
    print(f"  图像尺寸: {image_size}x{image_size}")
    print(f"  迭代次数: {num_iterations}")
    print(f"  设备: {device}")
    print(f"  优化版本: {'是' if use_optimized else '否'}")
    print("-"*60)
    
    # 初始化加密系统
    api = SCNECipherAPI(
        password='benchmark',
        use_frequency=True,
        deterministic=True
    )
    
    # 准备测试数据
    test_images = torch.rand(
        batch_size, 1, image_size, image_size, 
        device=device, dtype=torch.float32
    )
    
    # 预热GPU
    if device == 'cuda':
        for _ in range(3):
            _ = api.encrypt_simple(test_images)
        torch.cuda.synchronize()
    
    # 测试加密速度
    print("\n测试加密性能...")
    encrypt_times = []
    
    for i in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        encrypted, enc_info = api.encrypt_simple(test_images, privacy_level=1.0)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        encrypt_time = end_time - start_time
        encrypt_times.append(encrypt_time)
        
        # 显示进度
        if i == 0 or (i + 1) % 5 == 0:
            fps = batch_size / encrypt_time
            print(f"  迭代 {i+1}/{num_iterations}: {encrypt_time:.3f}s ({fps:.2f} FPS)")
    
    # 测试解密速度
    print("\n测试解密性能...")
    decrypt_times = []
    
    for i in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        decrypted = api.decrypt_simple(encrypted, enc_info)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        decrypt_time = end_time - start_time
        decrypt_times.append(decrypt_time)
        
        # 显示进度
        if i == 0 or (i + 1) % 5 == 0:
            fps = batch_size / decrypt_time
            print(f"  迭代 {i+1}/{num_iterations}: {decrypt_time:.3f}s ({fps:.2f} FPS)")
    
    # 测试密码学指标
    print("\n计算密码学指标...")
    metrics = SecurityMetrics()
    
    # 转换为numpy进行指标计算
    orig_np = (test_images[0, 0].cpu().numpy() * 255).astype(np.uint8)
    enc_np = (encrypted[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    npcr = metrics.calculate_npcr(orig_np, enc_np)
    uaci = metrics.calculate_uaci(orig_np, enc_np)
    entropy = metrics.calculate_entropy(enc_np)
    
    # 测试可逆性
    decrypt_error = torch.nn.functional.mse_loss(decrypted, test_images).item()
    psnr = -10 * np.log10(decrypt_error + 1e-10)
    
    # 计算统计
    avg_encrypt = np.mean(encrypt_times[1:])  # 排除第一次（可能有初始化开销）
    avg_decrypt = np.mean(decrypt_times[1:])
    std_encrypt = np.std(encrypt_times[1:])
    std_decrypt = np.std(decrypt_times[1:])
    
    encrypt_fps = batch_size / avg_encrypt
    decrypt_fps = batch_size / avg_decrypt
    
    # 内存使用
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0
    
    # 输出结果
    print("\n" + "="*60)
    print("性能测试结果:")
    print("="*60)
    
    print("\n⏱️ 速度性能:")
    print(f"  加密速度: {encrypt_fps:.2f} FPS ({avg_encrypt*1000:.1f}±{std_encrypt*1000:.1f} ms)")
    print(f"  解密速度: {decrypt_fps:.2f} FPS ({avg_decrypt*1000:.1f}±{std_decrypt*1000:.1f} ms)")
    print(f"  总处理速度: {min(encrypt_fps, decrypt_fps):.2f} FPS")
    
    print("\n🔒 密码学指标:")
    print(f"  NPCR: {npcr:.2f}% {'✅' if npcr > 99.5 else '⚠️'}")
    print(f"  UACI: {uaci:.2f}% {'✅' if 30 < uaci < 35 else '⚠️'}")
    print(f"  熵: {entropy:.4f} {'✅' if entropy > 7.99 else '⚠️'}")
    
    print("\n🔄 可逆性:")
    print(f"  解密误差: {decrypt_error:.6f}")
    print(f"  PSNR: {psnr:.2f} dB {'✅' if psnr > 60 else '⚠️'}")
    
    if device == 'cuda':
        print(f"\n💾 GPU内存:")
        print(f"  峰值使用: {memory_mb:.1f} MB")
    
    # 性能评级
    print("\n📊 性能评级:")
    if encrypt_fps >= 10:
        rating = "优秀 ⭐⭐⭐"
        status = "✅ 达到验收标准"
    elif encrypt_fps >= 5:
        rating = "良好 ⭐⭐"
        status = "⚠️ 接近验收标准"
    else:
        rating = "需改进 ⭐"
        status = "❌ 未达验收标准"
    
    print(f"  评级: {rating}")
    print(f"  状态: {status}")
    print("="*60)
    
    # 返回结果字典
    return {
        'encrypt_fps': encrypt_fps,
        'decrypt_fps': decrypt_fps,
        'npcr': npcr,
        'uaci': uaci,
        'entropy': entropy,
        'psnr': psnr,
        'memory_mb': memory_mb
    }


def compare_versions():
    """对比优化前后的性能"""
    print("\n" + "🔬 对比优化前后性能 " + "="*40)
    
    # 测试不同配置
    configs = [
        {'batch_size': 1, 'image_size': 256},
        {'batch_size': 4, 'image_size': 256},
        {'batch_size': 8, 'image_size': 256},
        {'batch_size': 4, 'image_size': 128},
        {'batch_size': 4, 'image_size': 512},
    ]
    
    results = []
    for config in configs:
        print(f"\n配置: 批量={config['batch_size']}, 尺寸={config['image_size']}x{config['image_size']}")
        result = benchmark_performance(
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_iterations=5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        result.update(config)
        results.append(result)
    
    # 汇总表格
    print("\n" + "="*70)
    print("性能对比汇总:")
    print("="*70)
    print(f"{'批量':<6} {'尺寸':<8} {'加密FPS':<10} {'解密FPS':<10} {'内存(MB)':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['batch_size']:<6} {r['image_size']:<8} "
              f"{r['encrypt_fps']:<10.2f} {r['decrypt_fps']:<10.2f} "
              f"{r.get('memory_mb', 0):<10.1f}")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compare', action='store_true', help='对比不同配置')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_versions()
    else:
        benchmark_performance(
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_iterations=args.iterations,
            device=args.device
        )
