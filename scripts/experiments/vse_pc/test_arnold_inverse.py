"""
测试Arnold映射的可逆性
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

N = 64  # 图像尺寸
iterations = 3

print(f"\n测试 Arnold 映射 (N={N}, iterations={iterations})")

# 获取正向和逆向映射索引
fwd_idx = cipher._arnold_map_indices(N, iterations)
inv_idx = cipher._inverse_arnold_map_indices(N, iterations)

print(f"正向索引范围: [{fwd_idx.min()}, {fwd_idx.max()}]")
print(f"逆向索引范围: [{inv_idx.min()}, {inv_idx.max()}]")

# 测试：正向映射后逆向映射应该恢复原始索引
original_idx = torch.arange(N*N, device=device)

# 方法1：scatter然后gather
x_flat = original_idx.float().view(1, 1, -1)
x_scrambled = torch.zeros_like(x_flat)
fwd_exp = fwd_idx.view(1, 1, -1)
x_scrambled.scatter_(2, fwd_exp, x_flat)

inv_exp = inv_idx.view(1, 1, -1)
x_restored = torch.gather(x_scrambled, 2, inv_exp)

diff = (x_flat - x_restored).abs().mean().item()
print(f"\n方法1 (scatter+gather) 差异: {diff}")

# 方法2：直接验证索引组合
# 如果 fwd[i] = j，那么 inv[j] 应该 = i
# 即 inv[fwd[i]] = i
composed = inv_idx[fwd_idx]
expected = original_idx
diff2 = (composed - expected).abs().float().mean().item()
print(f"方法2 (inv[fwd[i]]==i) 差异: {diff2}")

# 方法3：fwd[inv[i]] = i
composed3 = fwd_idx[inv_idx]
diff3 = (composed3 - expected).abs().float().mean().item()
print(f"方法3 (fwd[inv[i]]==i) 差异: {diff3}")

if diff < 1e-6 and diff2 < 1e-6:
    print("\n✅ Arnold 映射可逆")
else:
    print("\n❌ Arnold 映射不可逆！")
    print("  这可能是问题根源")
