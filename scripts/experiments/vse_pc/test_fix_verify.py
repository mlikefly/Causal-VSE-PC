"""
验证混沌加密修复
"""
import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

print("验证混沌加密修复")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

from src.core.chaotic_encryptor import StandardChaoticCipher

cipher = StandardChaoticCipher(device=device)

# 测试1: 小图像
print("\n[测试1] 小图像 (64x64)")
torch.manual_seed(42)
image = torch.rand(1, 1, 64, 64, device=device)
key = torch.rand(1, 2, device=device)
params = {'iterations': 3, 'strength': torch.tensor(0.5, device=device).view(1,1,1,1)}

enc = cipher.encrypt(image, key, params)
dec = cipher.decrypt(enc, key, params)
psnr = calculate_psnr(image, dec)
print(f"  PSNR: {psnr:.2f} dB {'✅' if psnr > 40 else '❌'}")

# 测试2: 大图像
print("\n[测试2] 大图像 (256x256)")
image2 = torch.rand(1, 1, 256, 256, device=device)
key2 = torch.rand(1, 2, device=device)

enc2 = cipher.encrypt(image2, key2, params)
dec2 = cipher.decrypt(enc2, key2, params)
psnr2 = calculate_psnr(image2, dec2)
print(f"  PSNR: {psnr2:.2f} dB {'✅' if psnr2 > 40 else '❌'}")

# 测试3: 不同iterations
print("\n[测试3] 不同iterations")
for iters in [1, 3, 5, 10]:
    params3 = {'iterations': iters, 'strength': torch.tensor(0.5, device=device).view(1,1,1,1)}
    enc3 = cipher.encrypt(image, key, params3)
    dec3 = cipher.decrypt(enc3, key, params3)
    psnr3 = calculate_psnr(image, dec3)
    print(f"  iterations={iters}: PSNR={psnr3:.2f} dB {'✅' if psnr3 > 40 else '❌'}")

# 测试4: 不同strength
print("\n[测试4] 不同strength")
for strength in [0.1, 0.3, 0.5, 0.7, 1.0]:
    params4 = {'iterations': 3, 'strength': torch.tensor(strength, device=device).view(1,1,1,1)}
    enc4 = cipher.encrypt(image, key, params4)
    dec4 = cipher.decrypt(enc4, key, params4)
    psnr4 = calculate_psnr(image, dec4)
    print(f"  strength={strength}: PSNR={psnr4:.2f} dB {'✅' if psnr4 > 40 else '❌'}")

print("\n" + "=" * 50)
print("验证完成")
