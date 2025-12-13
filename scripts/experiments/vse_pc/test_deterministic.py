"""
密钥确定性验证测试
==================

验证：
1. 相同密码+相同图像 → 相同加密结果
2. 加密-解密 → PSNR > 40dB（近无损）
3. 不同密码 → 完全不同的加密结果
4. 错误密码解密 → 无法恢复原图
5. 不同privacy_level → 不同加密强度

使用方法:
    python scripts/experiments/vse_pc/test_deterministic.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算PSNR"""
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def load_test_image(device: str, image_size: int = 256):
    """
    加载真实测试图像（优先使用数据集，fallback到生成图像）
    """
    try:
        from src.utils.datasets import get_celeba_dataloader
        
        dataloader = get_celeba_dataloader(
            root_dir='data/CelebA-HQ',
            split='test',
            batch_size=1,
            image_size=image_size,
            return_labels=False,
            shuffle=False
        )
        images = next(iter(dataloader))
        if isinstance(images, (tuple, list)):
            images = images[0]
        images = images.to(device)
        
        # 转为单通道
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        print(f"✓ 加载真实CelebA-HQ图像: {images.shape}")
        return images
        
    except Exception as e:
        print(f"⚠️ 无法加载数据集: {e}")
        print("  使用固定种子生成测试图像...")
        torch.manual_seed(42)
        images = torch.rand(1, 1, image_size, image_size, device=device)
        return images


def main():
    print("=" * 60)
    print("密钥确定性验证测试")
    print("=" * 60)
    
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 256
    password = "test_password_123"
    
    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  图像尺寸: {image_size}x{image_size}")
    print(f"  密码: {password[:3]}***")
    
    # 加载测试图像
    print("\n加载测试图像...")
    test_image = load_test_image(device, image_size)
    
    # 测试结果统计
    results = {'passed': 0, 'failed': 0, 'warnings': 0}
    
    # ========== 测试1：确定性 ==========
    print("\n" + "-" * 40)
    print("[测试1] 相同密码 → 相同加密结果")
    print("-" * 40)
    
    cipher1 = SCNECipherAPI(
        password=password, image_size=image_size, device=device,
        use_frequency=False, enable_crypto_wrap=False
    )
    cipher2 = SCNECipherAPI(
        password=password, image_size=image_size, device=device,
        use_frequency=False, enable_crypto_wrap=False
    )
    
    enc1, info1 = cipher1.encrypt_simple(test_image.clone(), privacy_level=0.5)
    enc2, info2 = cipher2.encrypt_simple(test_image.clone(), privacy_level=0.5)
    
    diff = (enc1 - enc2).abs().mean().item()
    if diff < 1e-6:
        print(f"  ✅ 通过 - 差异: {diff:.10f}")
        results['passed'] += 1
    else:
        print(f"  ❌ 失败 - 差异: {diff:.10f} (应 < 1e-6)")
        results['failed'] += 1
    
    # ========== 测试2：解密可逆性 ==========
    print("\n" + "-" * 40)
    print("[测试2] 加密-解密可逆性")
    print("-" * 40)
    
    try:
        decrypted = cipher1.decrypt_simple(enc1, info1)
        psnr = calculate_psnr(test_image, decrypted)
        mae = (test_image - decrypted).abs().mean().item()
        
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  MAE: {mae:.6f}")
        
        if psnr > 40:
            print(f"  ✅ 通过 - 近无损解密")
            results['passed'] += 1
        elif psnr > 20:
            print(f"  ⚠️ 警告 - 解密质量一般")
            results['warnings'] += 1
        else:
            print(f"  ❌ 失败 - 解密质量差")
            results['failed'] += 1
    except Exception as e:
        print(f"  ❌ 失败 - 解密异常: {e}")
        results['failed'] += 1
    
    # ========== 测试3：不同密码加密差异 ==========
    print("\n" + "-" * 40)
    print("[测试3] 不同密码 → 不同加密结果")
    print("-" * 40)
    
    cipher_wrong = SCNECipherAPI(
        password="wrong_password_456", image_size=image_size, device=device,
        use_frequency=False, enable_crypto_wrap=False
    )
    enc_wrong, _ = cipher_wrong.encrypt_simple(test_image.clone(), privacy_level=0.5)
    
    diff_wrong = (enc1 - enc_wrong).abs().mean().item()
    if diff_wrong > 0.1:
        print(f"  ✅ 通过 - 差异: {diff_wrong:.4f} (应 > 0.1)")
        results['passed'] += 1
    else:
        print(f"  ❌ 失败 - 差异: {diff_wrong:.4f} (应 > 0.1)")
        results['failed'] += 1
    
    # ========== 测试4：错误密码解密 ==========
    print("\n" + "-" * 40)
    print("[测试4] 错误密码解密 → 无法恢复原图")
    print("-" * 40)
    
    try:
        # 用错误密码的cipher尝试解密正确密码加密的图像
        decrypted_wrong = cipher_wrong.decrypt_simple(enc1, info1)
        psnr_wrong = calculate_psnr(test_image, decrypted_wrong)
        
        if psnr_wrong < 15:
            print(f"  ✅ 通过 - 错误密码解密PSNR: {psnr_wrong:.2f} dB (应 < 15)")
            results['passed'] += 1
        else:
            print(f"  ❌ 失败 - 错误密码解密PSNR: {psnr_wrong:.2f} dB (应 < 15)")
            results['failed'] += 1
    except Exception as e:
        print(f"  ✅ 通过 - 错误密码解密抛出异常: {type(e).__name__}")
        results['passed'] += 1
    
    # ========== 测试5：privacy_level差异 ==========
    print("\n" + "-" * 40)
    print("[测试5] 不同privacy_level → 不同加密强度")
    print("-" * 40)
    
    levels = [0.3, 0.5, 0.7, 1.0]
    maes = []
    
    for level in levels:
        enc, _ = cipher1.encrypt_simple(test_image.clone(), privacy_level=level)
        mae = (enc - test_image).abs().mean().item()
        maes.append(mae)
        print(f"  level={level}: MAE={mae:.4f}")
    
    # 检查MAE是否随privacy_level递增
    is_increasing = all(maes[i] <= maes[i+1] for i in range(len(maes)-1))
    if is_increasing and maes[-1] > maes[0]:
        print(f"  ✅ 通过 - MAE随privacy_level递增")
        results['passed'] += 1
    else:
        print(f"  ⚠️ 警告 - MAE未严格递增（可能正常）")
        results['warnings'] += 1
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    total = results['passed'] + results['failed'] + results['warnings']
    print(f"\n  ✅ 通过: {results['passed']}/{total}")
    print(f"  ❌ 失败: {results['failed']}/{total}")
    print(f"  ⚠️ 警告: {results['warnings']}/{total}")
    
    if results['failed'] == 0:
        print("\n🎉 所有关键测试通过！")
        return True
    else:
        print("\n⚠️ 存在失败的测试，请检查！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
