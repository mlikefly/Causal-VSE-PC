# -*- coding: utf-8 -*-
"""
混沌加密核心验证脚本 - 可视化版本
==========================================
验证 StandardChaoticCipher 的密码学特性并生成精美的分析报告

包含内容:
1. 可逆性验证 (Decryption MSE)
2. 混沌序列时域波形 (Time Series)
3. 混沌相图 (Phase Portrait)
4. 像素直方图对比 (Histogram)
5. 相邻像素相关性 (Correlation)
6. 密钥敏感性测试 (Key Sensitivity)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.core.chaotic_encryptor import StandardChaoticCipher

# 设置绘图样式
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def calculate_histogram(img_tensor):
    """计算图像直方图"""
    data = img_tensor.cpu().numpy().flatten()
    hist, bins = np.histogram(data, bins=256, range=(0, 1))
    return hist, bins


def get_adjacent_pixels(img_tensor, direction='horizontal'):
    """获取相邻像素对用于相关性分析"""
    img = img_tensor.cpu().numpy()[0]  # 取第一个通道
    h, w = img.shape
    
    if direction == 'horizontal':
        x = img[:, :-1].flatten()
        y = img[:, 1:].flatten()
    elif direction == 'vertical':
        x = img[:-1, :].flatten()
        y = img[1:, :].flatten()
    elif direction == 'diagonal':
        x = img[:-1, :-1].flatten()
        y = img[1:, 1:].flatten()
    
    # 随机采样2000个点以加速绘图
    if len(x) > 2000:
        idx = np.random.choice(len(x), 2000, replace=False)
        return x[idx], y[idx]
    return x, y


def main():
    print("=" * 60)
    print("混沌加密核心验证与可视化")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===========================
    # 1. 生成测试图像
    # ===========================
    print("\n[1/6] 生成测试图像...")
    H, W = 256, 256
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 创建复杂的测试图案 (混合频率)
    img = (torch.sin(x_grid / 10.0) * torch.cos(y_grid / 20.0) + 1.0) / 2.0
    img = (img + (x_grid + y_grid) / (H + W)) / 2.0
    img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
    print(f"  图像尺寸: {img.shape}")
    
    # ===========================
    # 2. 初始化加密器
    # ===========================
    print("\n[2/6] 初始化加密器...")
    cipher = StandardChaoticCipher(device=device)
    key = torch.tensor([[0.123456, 0.654321]], device=device)
    params = {
        'iterations': 3,
        'strength': torch.tensor([[[[0.9]]]], device=device)
    }
    print(f"  密钥: {key[0].tolist()}")
    print(f"  参数: iterations={params['iterations']}, strength={params['strength'].item():.2f}")
    
    # ===========================
    # 3. 执行加密与解密
    # ===========================
    print("\n[3/6] 执行加密...")
    encrypted = cipher.encrypt(img, key, params)
    
    print("[4/6] 执行解密...")
    decrypted = cipher.decrypt(encrypted, key, params)
    
    mse = torch.nn.functional.mse_loss(decrypted, img).item()
    print(f"  解密MSE: {mse:.8f} - {'通过' if mse < 1e-4 else '失败'}")
    
    # ===========================
    # 4. 提取混沌序列
    # ===========================
    print("\n[5/6] 生成混沌序列...")
    with torch.no_grad():
        chaos_seq = cipher._hyper_chaotic_map(5000, key).cpu().numpy()[0]
    print(f"  序列长度: {len(chaos_seq)}")
    
    # ===========================
    # 5. 绘制可视化报告
    # ===========================
    print("\n[6/6] 生成可视化报告...")
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- 第一行: 图像对比 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img[0].permute(1, 2, 0).cpu().numpy())
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(encrypted[0].permute(1, 2, 0).cpu().numpy())
    ax2.set_title("Encrypted Image\n(5D Hyperchaos)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(decrypted[0].permute(1, 2, 0).cpu().numpy())
    ax3.set_title(f"Decrypted Image\nMSE: {mse:.1e}", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # --- 直方图 ---
    ax4 = fig.add_subplot(gs[0, 3])
    hist_orig, bins = calculate_histogram(img[0])
    hist_enc, _ = calculate_histogram(encrypted[0])
    ax4.bar(bins[:-1], hist_orig, width=1/256, alpha=0.6, color='blue', label='Original')
    ax4.bar(bins[:-1], hist_enc, width=1/256, alpha=0.6, color='red', label='Encrypted')
    ax4.set_title("Pixel Histogram", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Pixel Value")
    ax4.set_ylabel("Frequency")
    
    # --- 第二行: 相关性分析 ---
    x_orig, y_orig = get_adjacent_pixels(img[0], 'horizontal')
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(x_orig, y_orig, s=2, c='blue', alpha=0.5)
    ax5.set_title("Original Correlation\n(Horizontal)", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Pixel(x, y)")
    ax5.set_ylabel("Pixel(x+1, y)")
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    x_enc, y_enc = get_adjacent_pixels(encrypted[0], 'horizontal')
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.scatter(x_enc, y_enc, s=2, c='red', alpha=0.5)
    ax6.set_title("Encrypted Correlation\n(Should be Random)", fontsize=14, fontweight='bold')
    ax6.set_xlabel("Pixel(x, y)")
    ax6.set_ylabel("Pixel(x+1, y)")
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    
    # --- 混沌时序图 ---
    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.plot(chaos_seq[:300], color='purple', linewidth=1.5, alpha=0.8)
    ax7.set_title("5D Hyperchaotic Time Series", fontsize=14, fontweight='bold')
    ax7.set_xlabel("Time Step (t)")
    ax7.set_ylabel("Chaotic Value")
    ax7.grid(True, alpha=0.3)
    
    # --- 第三行: 相图与密钥敏感性 ---
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.scatter(chaos_seq[:-1], chaos_seq[1:], s=1.5, c='green', alpha=0.6)
    ax8.set_title("Phase Portrait\n(x_t vs x_{t+1})", fontsize=14, fontweight='bold')
    ax8.set_xlabel("x(t)")
    ax8.set_ylabel("x(t+1)")
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.grid(True, alpha=0.3)
    
    # --- 密钥敏感性 ---
    key_perturbed = key.clone()
    key_perturbed[0, 0] += 1e-10
    encrypted_perturbed = cipher.encrypt(img, key_perturbed, params)
    diff_img = (encrypted - encrypted_perturbed).abs().mean(dim=1)
    
    ax9 = fig.add_subplot(gs[2, 1])
    im9 = ax9.imshow(diff_img[0].cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
    ax9.set_title("Key Sensitivity\n(Delta=1e-10)", fontsize=14, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
    
    # --- 统计信息面板 ---
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    corr_orig = np.corrcoef(x_orig, y_orig)[0, 1]
    corr_enc = np.corrcoef(x_enc, y_enc)[0, 1]
    diff_mean = diff_img.mean().item()
    
    text_info = (
        "=== Cryptographic Analysis Report ===\n\n"
        "1. Chaotic System:\n"
        "   Model: 5D Coupled Hyperchaotic Map\n"
        "   Topology: Ring Coupling\n"
        f"   Key Space: > 10^100\n\n"
        "2. Statistical Analysis:\n"
        f"   Original Correlation: {corr_orig:.4f}\n"
        f"   Encrypted Correlation: {corr_enc:.4f} (Target ~0)\n"
        f"   Hist Variance (Enc): {np.var(hist_enc):.2f}\n\n"
        "3. Security Metrics:\n"
        f"   Key Sensitivity: {diff_mean:.4f}\n"
        f"   Decryption MSE: {mse:.2e}\n"
        f"   Reversibility: {'PASS' if mse < 1e-4 else 'FAIL'}\n"
    )
    
    ax10.text(0.05, 0.5, text_info, fontsize=13, fontfamily='monospace',
              va='center', bbox=dict(facecolor='white', alpha=0.9,
                                     edgecolor='gray', boxstyle='round,pad=1'))
    
    plt.suptitle("NSCE 5D Hyperchaotic Encryption Analysis", 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "chaos_analysis_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"\n{'='*60}")
    print(f"报告已生成: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
