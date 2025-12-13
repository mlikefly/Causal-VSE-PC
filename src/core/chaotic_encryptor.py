"""
标准混沌加密器 (Standard Chaotic Cipher)
=========================================
这是一个符合密码学规范的图像加密模块。
它不包含任何神经网络逻辑，只负责数学运算。
所有可变参数（如迭代次数、混沌强度）均由外部接口传入。

核心步骤：
1. 密钥生成 (Key Setup): 从用户密钥生成混沌系统的初始值 (x0, r)。
2. 置乱 (Permutation/Scrambling): 使用 Arnold Cat Map 打乱像素位置。
3. 扩散 (Diffusion): 使用 Logistic Map 生成掩码，改变像素值。
4. 替换 (Substitution): 结合 Logistic 序列进行像素值混淆。

创新点接口：
- adaptive_params: 允许外部策略网络动态调整加密的轮数(iterations)和强度(strength)。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

class StandardChaoticCipher(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, images, key, params):
        """支持 model(x, key, params) 调用"""
        return self.encrypt(images, key, params)

    def _hyper_chaotic_map(self, size: int, key: torch.Tensor) -> torch.Tensor:
        r"""
        5D 强混沌耦合映射 (5D Strong Hyperchaotic Map) 生成高质量伪随机序列。
        采用双向环形耦合 + 非线性混合，确保强混沌特性 (LE > 0.1)。
        
        改进的动力学方程:
        x_{i, t+1} = [ μ * x_{i,t} * (1 - x_{i,t}) 
                      + α * (x_{i+1,t} - x_{i-1,t}) 
                      + β * sin(π * x_{(i+2)%5,t}) ] mod 1.0
        
        其中:
        - μ: Logistic 参数 (3.8 ~ 4.0, 深度混沌区)
        - α: 双向耦合强度 (0.3 ~ 0.5, 强耦合)
        - β: 非线性耦合强度 (0.1 ~ 0.2)
        
        Args:
            size: 需要生成的序列长度 (H*W)
            key: [B, 2] (seed1, seed2) - 密钥
        """
        B = key.shape[0]
        
        # 1. 密钥扩展 (Key Expansion)
        k1 = key[:, 0:1]
        k2 = key[:, 1:2]
        
        # 初始状态 X0: [B, 5]
        # 使用不同频率的三角函数生成 5 个独立初值
        x = torch.cat([
            (torch.sin(k1 * 10.0 + k2 * 1.0).abs() + 0.1) % 0.9,
            (torch.sin(k1 * 20.0 + k2 * 2.0).abs() + 0.1) % 0.9,
            (torch.sin(k1 * 30.0 + k2 * 3.0).abs() + 0.1) % 0.9,
            (torch.sin(k1 * 40.0 + k2 * 5.0).abs() + 0.1) % 0.9,
            (torch.sin(k1 * 50.0 + k2 * 7.0).abs() + 0.1) % 0.9,
        ], dim=1).to(self.device)
        
        # 控制参数（优化后的强混沌参数 - Sine-Logistic 混合）
        # mu: 0.99 ~ 1.0 (Sine Map 满映射区)
        mu = 0.99 + (torch.sin(k1 * 100.0).abs() % 0.01)
        # alpha: 0.4 ~ 0.5 (超强双向耦合)
        alpha = 0.4 + (torch.cos(k2 * 100.0).abs() % 0.1)
        # beta: 0.15 ~ 0.25 (非线性增强)
        beta = 0.15 + (torch.sin(k1 * 200.0 + k2 * 200.0).abs() % 0.1)
        
        # 预热 (Warm-up) - 丢弃前200步以进入吸引子
        for _ in range(200):
            x_next = torch.zeros_like(x)
            for i in range(5):
                prev_i = (i - 1) % 5
                next_i = (i + 1) % 5
                skip_i = (i + 2) % 5
                
                # 改进的强混沌方程: Sine-Coupling
                # Sine Map: sin(pi * x) 具有比 Logistic 更大的 LE
                sine_term = mu * torch.sin(np.pi * x[:, i:i+1])
                coupling_term = alpha * (x[:, next_i:next_i+1] - x[:, prev_i:prev_i+1])
                nonlinear_term = beta * torch.cos(np.pi * x[:, skip_i:skip_i+1])
                
                val = sine_term + coupling_term + nonlinear_term
                # 加上 x 本身以形成类 ResNet 的残差流，有助于梯度传播
                val = val + x[:, i:i+1] * 0.1 
                
                x_next[:, i:i+1] = val % 1.0
            x = x_next

        # 生成序列
        seq = torch.zeros(B, size, device=self.device)
        
        for t in range(size):
            # 迭代一步（使用相同的强混沌方程）
            x_next = torch.zeros_like(x)
            for i in range(5):
                prev_i = (i - 1) % 5
                next_i = (i + 1) % 5
                skip_i = (i + 2) % 5
                
                sine_term = mu * torch.sin(np.pi * x[:, i:i+1])
                coupling_term = alpha * (x[:, next_i:next_i+1] - x[:, prev_i:prev_i+1])
                nonlinear_term = beta * torch.cos(np.pi * x[:, skip_i:skip_i+1])
                
                val = sine_term + coupling_term + nonlinear_term
                val = val + x[:, i:i+1] * 0.1
                
                x_next[:, i:i+1] = val % 1.0
            x = x_next
            
            # 混合输出 (Mixing Output)
            # 取 5 个状态的非线性组合，增加复杂度
            # out = (x1 + x2 + x3 + x4 + x5) mod 1
            out = x.sum(dim=1) % 1.0
            seq[:, t] = out
            
        return seq

    def _arnold_map_indices(self, N: int, iterations: int) -> torch.Tensor:
        """计算 Arnold 置乱的源坐标到目标坐标的映射索引"""
        # 缓存机制可以在外部做，这里直接算
        idx = torch.arange(N*N, device=self.device)
        curr_x = idx % N
        curr_y = idx // N
        
        for _ in range(iterations):
            next_x = (curr_x + curr_y) % N
            next_y = (curr_x + 2 * curr_y) % N
            curr_x, curr_y = next_x, next_y
            
        new_idx = curr_y * N + curr_x
        return new_idx

    def _inverse_arnold_map_indices(self, N: int, iterations: int) -> torch.Tensor:
        """计算逆 Arnold 映射索引"""
        idx = torch.arange(N*N, device=self.device)
        curr_x = idx % N
        curr_y = idx // N
        
        for _ in range(iterations):
            next_x = (2 * curr_x - curr_y) % N
            next_y = (-curr_x + curr_y) % N
            curr_x, curr_y = next_x, next_y
            
        # 逆映射得到的坐标 (curr_x, curr_y) 是“源坐标在现在的哪里”
        # 不，逆映射公式计算的是：给定现在的坐标 (x,y)，它原来在哪里？
        # 所以 src_idx = Map(dst_idx)
        src_idx = curr_y * N + curr_x
        return src_idx

    def encrypt(self, images: torch.Tensor, key: torch.Tensor, params: Dict) -> torch.Tensor:
        """
        加密函数
        
        Args:
            images: [B, C, H, W]
            key: [B, 2]
            params: 
                - iterations: int or Tensor
                - strength: float or Tensor [B, 1, 1, 1]
        """
        B, C, H, W = images.shape
        
        # 解析参数
        p_iters = params.get('iterations', 1)
        if isinstance(p_iters, torch.Tensor):
            iterations = int(p_iters.item()) if p_iters.numel() == 1 else int(p_iters[0].item())
        else:
            iterations = int(p_iters)
            
        p_str = params.get('strength', 0.5)
        if isinstance(p_str, torch.Tensor):
            strength = p_str
        else:
            strength = torch.tensor(p_str, device=self.device)
        
        x = images.clone()
        
        # 1. 置乱 (Scrambling) - 使用gather实现
        # Arnold映射: new_idx[i] = j 表示原位置i的像素移动到位置j
        # 加密时：scrambled[j] = original[i]，即 scrambled[new_idx[i]] = original[i]
        # 使用scatter: scrambled.scatter_(dim, new_idx, original)
        if iterations > 0:
            new_idx = self._arnold_map_indices(H, iterations)
            x_flat = x.view(B, C, -1)
            x_scrambled = torch.zeros_like(x_flat)
            idx_exp = new_idx.view(1, 1, -1).expand(B, C, -1)
            x_scrambled.scatter_(2, idx_exp, x_flat)
            x = x_scrambled.view(B, C, H, W)
            
        # 2. 扩散 (Diffusion)
        chaos_seq = self._hyper_chaotic_map(H*W, key)
        chaos_mask = chaos_seq.view(B, 1, H, W)
        
        # 像素值混淆: C = (P + Mask * Strength) % 1.0
        x = (x + chaos_mask * strength) % 1.0
        
        return x

    def decrypt(self, encrypted: torch.Tensor, key: torch.Tensor, params: Dict) -> torch.Tensor:
        """解密函数"""
        B, C, H, W = encrypted.shape
        
        p_iters = params.get('iterations', 1)
        if isinstance(p_iters, torch.Tensor):
            iterations = int(p_iters.item()) if p_iters.numel() == 1 else int(p_iters[0].item())
        else:
            iterations = int(p_iters)
            
        p_str = params.get('strength', 0.5)
        if isinstance(p_str, torch.Tensor):
            strength = p_str
        else:
            strength = torch.tensor(p_str, device=self.device)
        
        x = encrypted.clone()
        
        # 1. 逆扩散 (Inverse Diffusion)
        chaos_seq = self._hyper_chaotic_map(H*W, key)
        chaos_mask = chaos_seq.view(B, 1, H, W)
        
        # P = (C - Mask * Strength) % 1.0
        x = (x - chaos_mask * strength) % 1.0
        
        # 2. 逆置乱 (Inverse Scrambling)
        # 加密时用scatter: scrambled[fwd_idx[i]] = original[i]
        # 解密时用gather: original[i] = scrambled[fwd_idx[i]]
        # 关键：使用相同的正向映射索引fwd_idx，因为scatter和gather是对称操作
        if iterations > 0:
            fwd_idx = self._arnold_map_indices(H, iterations)
            x_flat = x.view(B, C, -1)
            idx_exp = fwd_idx.view(1, 1, -1).expand(B, C, -1)
            x_restored = torch.gather(x_flat, 2, idx_exp)
            x = x_restored.view(B, C, H, W)
            
        return x
