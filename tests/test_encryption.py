"""
加密/解密核心功能测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_layer1_encryption():
    """测试Layer 1混沌加密的可逆性"""
    from src.core.chaotic_encryptor import StandardChaoticCipher
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cipher = StandardChaoticCipher(device=device)
    
    # 测试图像
    image = torch.rand(1, 1, 64, 64, device=device)
    key = torch.rand(1, 2, device=device)
    params = {'iterations': 3, 'strength': torch.tensor(0.5, device=device)}
    
    # 加密
    encrypted = cipher.encrypt(image, key, params)
    
    # 解密
    decrypted = cipher.decrypt(encrypted, key, params)
    
    # 验证
    diff = (image - decrypted).abs().max().item()
    assert diff < 1e-5, f"Layer 1解密失败，最大差异: {diff}"
    print(f"✓ Layer 1加密/解密测试通过，最大差异: {diff:.2e}")


def test_layer2_encryption():
    """测试Layer 2频域加密的可逆性"""
    from src.core.frequency_cipher import FrequencySemanticCipherOptimized
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cipher = FrequencySemanticCipherOptimized()
    
    # 测试图像
    image = torch.rand(1, 1, 64, 64, device=device)
    region_key = b'test_key_12345678'
    
    # 加密
    encrypted, info = cipher.encrypt_fft(image, region_key, privacy_level=0.7)
    
    # 解密
    decrypted = cipher.decrypt_fft(encrypted, info)
    
    # 验证
    psnr = 10 * torch.log10(1.0 / ((image - decrypted)**2).mean()).item()
    assert psnr > 30, f"Layer 2解密质量不足，PSNR: {psnr:.2f}dB"
    print(f"✓ Layer 2加密/解密测试通过，PSNR: {psnr:.2f}dB")


def test_full_scne_encryption():
    """测试完整SCNE加密流程"""
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    password = "test_password_123"
    
    cipher = SCNECipherAPI(
        password=password,
        image_size=64,
        device=device,
        use_frequency=True
    )
    
    # 测试图像
    image = torch.rand(1, 1, 64, 64, device=device)
    
    # 加密
    encrypted, enc_info = cipher.encrypt_simple(image, privacy_level=0.7)
    
    # 解密
    mask = torch.ones_like(image)
    decrypted = cipher.cipher.decrypt(encrypted, enc_info, mask, password=password)
    
    # 验证
    psnr = 10 * torch.log10(1.0 / ((image - decrypted)**2).mean()).item()
    assert psnr > 30, f"SCNE解密质量不足，PSNR: {psnr:.2f}dB"
    print(f"✓ SCNE完整加密/解密测试通过，PSNR: {psnr:.2f}dB")


def test_security_metrics():
    """测试安全指标计算"""
    from src.evaluation.security_metrics import SecurityMetrics
    
    # 生成测试数据
    original = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    encrypted = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # 计算指标
    metrics = SecurityMetrics.evaluate_image(original, encrypted)
    
    # 验证指标存在
    assert 'entropy_encrypted' in metrics
    assert 'npcr' in metrics
    assert 'uaci' in metrics
    print(f"✓ 安全指标计算测试通过")
    print(f"  熵: {metrics['entropy_encrypted']:.4f} bits")
    print(f"  NPCR: {metrics['npcr']:.2f}%")
    print(f"  UACI: {metrics['uaci']:.2f}%")


if __name__ == '__main__':
    print("=" * 60)
    print("Causal-VSE-PC 核心功能测试")
    print("=" * 60)
    
    test_layer1_encryption()
    test_layer2_encryption()
    test_full_scne_encryption()
    test_security_metrics()
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
