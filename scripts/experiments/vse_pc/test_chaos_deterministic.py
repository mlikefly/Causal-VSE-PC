"""
测试混沌序列的确定性
"""
import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.chaotic_encryptor import StandardChaoticCipher

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

cipher = StandardChaoticCipher(device=device)

# 固定key
key = torch.tensor([[0.5, 0.3]], device=device)

# 生成两次混沌序列
seq1 = cipher._hyper_chaotic_map(100, key)
seq2 = cipher._hyper_chaotic_map(100, key)

diff = (seq1 - seq2).abs().mean().item()
print(f"两次生成的混沌序列差异: {diff}")

if diff < 1e-10:
    print("✅ 混沌序列是确定性的")
else:
    print("❌ 混沌序列不是确定性的！这是问题根源！")

# 测试加密-解密
print("\n测试加密-解密:")
image = torch.rand(1, 1, 64, 64, device=device)
params = {'iterations': 3, 'strength': torch.tensor(0.5, device=device).view(1,1,1,1)}

enc = cipher.encrypt(image, key, params)
dec = cipher.decrypt(enc, key, params)

mse = ((image - dec) ** 2).mean().item()
print(f"MSE: {mse}")

if mse < 1e-10:
    print("✅ 加密-解密可逆")
else:
    print("❌ 加密-解密不可逆")
    
    # 进一步诊断
    print("\n诊断:")
    
    # 检查混沌掩码
    chaos_seq_enc = cipher._hyper_chaotic_map(64*64, key)
    chaos_seq_dec = cipher._hyper_chaotic_map(64*64, key)
    mask_diff = (chaos_seq_enc - chaos_seq_dec).abs().mean().item()
    print(f"  加密/解密时混沌掩码差异: {mask_diff}")
