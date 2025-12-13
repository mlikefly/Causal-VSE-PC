"""
分层解密测试 - 定位解密失败的具体层
====================================

测试策略：
1. 仅Layer 1（混沌）→ 验证PSNR
2. Layer 1+2（+频域）→ 验证PSNR  
3. 全流程（+crypto_wrap）→ 验证PSNR

使用方法:
    python scripts/experiments/vse_pc/test_decrypt_layers.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def load_test_image(device: str, image_size: int = 256):
    """加载真实测试图像"""
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
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        print(f"✓ 加载真实图像: {images.shape}")
        return images
    except Exception as e:
        print(f"⚠️ 无法加载数据集: {e}")
        torch.manual_seed(42)
        return torch.rand(1, 1, image_size, image_size, device=device)


def test_layer1_only():
    """测试仅Layer 1（混沌加密）的可逆性"""
    print("\n" + "=" * 60)
    print("[测试1] 仅Layer 1（混沌加密）")
    print("=" * 60)
    
    from src.core.chaotic_encryptor import StandardChaoticCipher
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    B, C, H, W = image.shape
    
    cipher = StandardChaoticCipher(device=device)
    
    # 固定密钥和参数
    torch.manual_seed(42)
    key = torch.rand(B, 2, device=device)
    params = {
        'iterations': 5,
        'strength': torch.tensor(0.5, device=device).view(1, 1, 1, 1)
    }
    
    # 加密
    encrypted = cipher.encrypt(image, key, params)
    print(f"  加密后范围: [{encrypted.min():.4f}, {encrypted.max():.4f}]")
    
    # 解密
    decrypted = cipher.decrypt(encrypted, key, params)
    
    # 验证
    psnr = calculate_psnr(image, decrypted)
    mae = (image - decrypted).abs().mean().item()
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ Layer 1 解密正常")
        return True
    else:
        print("  ❌ Layer 1 解密失败")
        return False


def test_layer2_only():
    """测试仅Layer 2（频域加密）的可逆性"""
    print("\n" + "=" * 60)
    print("[测试2] 仅Layer 2（频域FFT加密）")
    print("=" * 60)
    
    from src.core.frequency_cipher import FrequencySemanticCipherOptimized
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    
    cipher = FrequencySemanticCipherOptimized(use_learnable_band=True, num_radial_bins=8)
    cipher = cipher.to(device)
    
    # 生成region_key
    region_key = b'test_region_key_1234567890123456'
    
    # 加密
    encrypted, fft_info = cipher.encrypt_fft(
        image,
        region_key,
        privacy_level=0.5,
        semantic_preserving=False
    )
    print(f"  加密后范围: [{encrypted.min():.4f}, {encrypted.max():.4f}]")
    
    # 解密
    decrypted = cipher.decrypt_fft(encrypted, fft_info)
    
    # 验证
    psnr = calculate_psnr(image, decrypted)
    mae = (image - decrypted).abs().mean().item()
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ Layer 2 解密正常")
        return True
    else:
        print("  ❌ Layer 2 解密失败")
        return False


def test_layer1_plus_layer2():
    """测试Layer 1 + Layer 2的组合可逆性"""
    print("\n" + "=" * 60)
    print("[测试3] Layer 1 + Layer 2 组合")
    print("=" * 60)
    
    from src.core.chaotic_encryptor import StandardChaoticCipher
    from src.core.frequency_cipher import FrequencySemanticCipherOptimized
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    B, C, H, W = image.shape
    
    chaos_cipher = StandardChaoticCipher(device=device)
    freq_cipher = FrequencySemanticCipherOptimized(use_learnable_band=True, num_radial_bins=8).to(device)
    
    # Layer 1 参数
    torch.manual_seed(42)
    key = torch.rand(B, 2, device=device)
    params = {
        'iterations': 5,
        'strength': torch.tensor(0.5, device=device).view(1, 1, 1, 1)
    }
    region_key = b'test_region_key_1234567890123456'
    
    # 加密: Layer 1 -> Layer 2
    enc_l1 = chaos_cipher.encrypt(image, key, params)
    enc_l2, fft_info = freq_cipher.encrypt_fft(enc_l1, region_key, privacy_level=0.5)
    
    print(f"  Layer1后范围: [{enc_l1.min():.4f}, {enc_l1.max():.4f}]")
    print(f"  Layer2后范围: [{enc_l2.min():.4f}, {enc_l2.max():.4f}]")
    
    # 解密: Layer 2 -> Layer 1
    dec_l2 = freq_cipher.decrypt_fft(enc_l2, fft_info)
    dec_l1 = chaos_cipher.decrypt(dec_l2, key, params)
    
    # 验证
    psnr = calculate_psnr(image, dec_l1)
    mae = (image - dec_l1).abs().mean().item()
    
    # 中间层验证
    psnr_l2_only = calculate_psnr(enc_l1, dec_l2)
    
    print(f"  Layer2解密PSNR: {psnr_l2_only:.2f} dB")
    print(f"  最终PSNR: {psnr:.2f} dB")
    print(f"  最终MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ Layer 1+2 组合解密正常")
        return True
    else:
        print("  ❌ Layer 1+2 组合解密失败")
        return False


def test_full_scne_no_crypto():
    """测试完整SCNE流程（不含crypto_wrap）"""
    print("\n" + "=" * 60)
    print("[测试4] 完整SCNE（无crypto_wrap）")
    print("=" * 60)
    
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    password = "test_password_123"
    
    # 禁用crypto_wrap
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        device=device,
        enable_crypto_wrap=False,  # 关键：禁用
        use_frequency=True,
        use_fft=True
    )
    
    # 加密
    encrypted, enc_info = api.encrypt_simple(image, privacy_level=0.5)
    print(f"  加密后范围: [{encrypted.min():.4f}, {encrypted.max():.4f}]")
    
    # 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    
    # 验证
    psnr = calculate_psnr(image, decrypted)
    mae = (image - decrypted).abs().mean().item()
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ SCNE（无crypto_wrap）解密正常")
        return True
    else:
        print("  ❌ SCNE（无crypto_wrap）解密失败")
        return False


def test_full_scne_with_crypto():
    """测试完整SCNE流程（含crypto_wrap）"""
    print("\n" + "=" * 60)
    print("[测试5] 完整SCNE（含crypto_wrap）")
    print("=" * 60)
    
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    password = "test_password_123"
    
    # 启用crypto_wrap
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        device=device,
        enable_crypto_wrap=True,  # 启用
        use_frequency=True,
        use_fft=True
    )
    
    # 加密
    encrypted, enc_info = api.encrypt_simple(image, privacy_level=0.5)
    print(f"  加密后范围: [{encrypted.min():.4f}, {encrypted.max():.4f}]")
    
    # 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    
    # 验证
    psnr = calculate_psnr(image, decrypted)
    mae = (image - decrypted).abs().mean().item()
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ SCNE（含crypto_wrap）解密正常")
        return True
    else:
        print("  ❌ SCNE（含crypto_wrap）解密失败")
        return False


def test_chaos_only_no_freq():
    """测试仅混沌加密（禁用频域）"""
    print("\n" + "=" * 60)
    print("[测试6] SCNE仅混沌（禁用频域）")
    print("=" * 60)
    
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image(device)
    password = "test_password_123"
    
    api = SCNECipherAPI(
        password=password,
        image_size=256,
        device=device,
        enable_crypto_wrap=False,
        use_frequency=False,  # 关键：禁用频域
        use_fft=False
    )
    
    # 加密
    encrypted, enc_info = api.encrypt_simple(image, privacy_level=0.5)
    print(f"  加密后范围: [{encrypted.min():.4f}, {encrypted.max():.4f}]")
    
    # 解密
    decrypted = api.decrypt_simple(encrypted, enc_info)
    
    # 验证
    psnr = calculate_psnr(image, decrypted)
    mae = (image - decrypted).abs().mean().item()
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    
    if psnr > 40:
        print("  ✅ SCNE仅混沌解密正常")
        return True
    else:
        print("  ❌ SCNE仅混沌解密失败")
        return False


def main():
    print("=" * 60)
    print("分层解密测试 - 定位解密失败的具体层")
    print("=" * 60)
    
    results = {}
    
    # 测试各层
    results['Layer1_Only'] = test_layer1_only()
    results['Layer2_Only'] = test_layer2_only()
    results['Layer1+2'] = test_layer1_plus_layer2()
    results['SCNE_NoFreq'] = test_chaos_only_no_freq()
    results['SCNE_NoCrypto'] = test_full_scne_no_crypto()
    results['SCNE_Full'] = test_full_scne_with_crypto()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    # 诊断
    print("\n诊断结论:")
    if not results['Layer1_Only']:
        print("  → 问题在 Layer 1（混沌加密）")
    elif not results['Layer2_Only']:
        print("  → 问题在 Layer 2（频域加密）")
    elif not results['Layer1+2']:
        print("  → 问题在 Layer 1+2 组合")
    elif not results['SCNE_NoFreq']:
        print("  → 问题在 SCNE 框架（非频域部分）")
    elif not results['SCNE_NoCrypto']:
        print("  → 问题在 SCNE 频域集成")
    elif not results['SCNE_Full']:
        print("  → 问题在 crypto_wrap 层")
    else:
        print("  → 所有层均正常！")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
