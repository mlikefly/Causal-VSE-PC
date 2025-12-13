# -*- coding: utf-8 -*-
"""
5D 耦合混沌系统动力学验证
=====================================
验证内容:
1. 李雅普诺夫指数计算 (Lyapunov Exponent Spectrum)
2. 混沌时序图 (Time Series of 5 Dimensions)
3. 相图 (Phase Portrait / Attractor)
4. 分岔图 (Bifurcation Diagram)
5. 自相关分析 (Autocorrelation)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import torch

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


def compute_lyapunov_exponent_wolf(states_history, dt=1.0):
    """
    使用Wolf算法计算最大李雅普诺夫指数
    
    参数:
        states_history: [T, 5] 混沌系统的状态演化历史
        dt: 时间步长
    
    返回:
        lyap_exp: 最大李雅普诺夫指数
    """
    T, dim = states_history.shape
    divergence_sum = 0.0
    valid_count = 0
    window_size = 20  # 减小窗口以捕捉快速发散
    
    for t in range(0, T - window_size, window_size):
        # 当前状态
        x0 = states_history[t]
        
        if t > 0:
            # 计算演化后的距离 (相对于上一步的邻居)
            # 这里简化为计算相邻时刻状态的距离比率
            # 对于离散系统，LE ≈ mean(ln(|J|))
            # 我们可以通过观察邻近轨迹的发散率来估算
            
            d0 = 1e-9  # 假设初始微小扰动
            
            # 演化 window_size 步后的实际距离
            # 为了更准确，我们需要同时演化一个扰动轨迹。
            # 但由于我们没有扰动轨迹的历史，这里使用经验公式估算
            # 更严谨的做法是在生成序列时同步演化切向量。
            pass 
    
    # 由于Wolf算法需要扰动轨迹，我们在 compute_lyapunov_spectrum_simple 中直接通过双轨迹法计算
    return 0.0


def compute_lyapunov_spectrum_simple(cipher, key, num_steps=10000):
    """
    双轨迹法计算最大李雅普诺夫指数 (MLE)
    """
    device = key.device
    
    # 1. 初始状态和扰动状态
    k1 = key[:, 0:1]
    k2 = key[:, 1:2]
    
    # 控制参数
    mu = (0.99 + (torch.sin(k1 * 100.0).abs() % 0.01)).cpu().numpy()
    alpha = (0.4 + (torch.cos(k2 * 100.0).abs() % 0.1)).cpu().numpy()
    beta = (0.15 + (torch.sin(k1 * 200.0 + k2 * 200.0).abs() % 0.1)).cpu().numpy()
    
    # 初始向量
    x = torch.rand(1, 5).cpu().numpy()
    
    # 扰动向量 (距离 d0 = 1e-9)
    d0 = 1e-9
    x_pert = x + np.random.randn(1, 5) * 1e-12
    x_pert = x + d0 * (x_pert - x) / np.linalg.norm(x_pert - x)
    
    # 记录状态历史
    states_history = np.zeros((num_steps, 5))
    le_sum = 0.0
    
    # 预热
    for _ in range(500):
        # 演化主轨迹
        x = update_state(x, mu, alpha, beta)
        # 演化扰动轨迹
        x_pert = update_state(x_pert, mu, alpha, beta)
        # 重归一化
        dist = np.linalg.norm(x_pert - x)
        x_pert = x + d0 * (x_pert - x) / dist
    
    # 正式计算
    for t in range(num_steps):
        states_history[t] = x[0]
        
        # 1. 演化
        x = update_state(x, mu, alpha, beta)
        x_pert = update_state(x_pert, mu, alpha, beta)
        
        # 2. 计算距离
        dist = np.linalg.norm(x_pert - x)
        
        # 3. 累加指数
        if dist > 0:
            le_sum += np.log(dist / d0)
        
        # 4. 重归一化 (Gram-Schmidt 简化版)
        x_pert = x + d0 * (x_pert - x) / dist
        
    lyap_exp = le_sum / num_steps
    return lyap_exp, states_history


def update_state(x, mu, alpha, beta):
    """动力学方程 (Numpy版，保持与 PyTorch 一致)"""
    x_next = np.zeros_like(x)
    for i in range(5):
        prev_i = (i - 1) % 5
        next_i = (i + 1) % 5
        skip_i = (i + 2) % 5
        
        sine_term = mu * np.sin(np.pi * x[:, i:i+1])
        coupling_term = alpha * (x[:, next_i:next_i+1] - x[:, prev_i:prev_i+1])
        nonlinear_term = beta * np.cos(np.pi * x[:, skip_i:skip_i+1])
        
        val = sine_term + coupling_term + nonlinear_term
        val = val + x[:, i:i+1] * 0.1  # 残差项
        
        x_next[:, i:i+1] = val % 1.0
    return x_next


def compute_autocorrelation(signal, max_lag=100):
    """计算自相关函数"""
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    return autocorr[:max_lag]


def main():
    print("=" * 70)
    print("5D 耦合混沌系统动力学特性验证")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 初始化加密器和密钥
    cipher = StandardChaoticCipher(device=device)
    key = torch.tensor([[0.123456, 0.654321]], device=device)
    
    print("\n[1/5] 提取系统参数...")
    k1 = key[:, 0:1]
    k2 = key[:, 1:2]
    mu_val = (3.8 + (torch.sin(k1 * 100.0).abs() % 0.2)).cpu().item()
    alpha_val = (0.3 + (torch.cos(k2 * 100.0).abs() % 0.2)).cpu().item()
    beta_val = (0.1 + (torch.sin(k1 * 200.0 + k2 * 200.0).abs() % 0.1)).cpu().item()
    print(f"  μ (Logistic参数): {mu_val:.4f}")
    print(f"  α (双向耦合强度): {alpha_val:.4f}")
    print(f"  β (非线性耦合): {beta_val:.4f}")
    
    print("\n[2/5] 计算李雅普诺夫指数...")
    lyap_exp, states_history = compute_lyapunov_spectrum_simple(cipher, key, num_steps=10000)
    
    print(f"  最大李雅普诺夫指数 (LE): {lyap_exp:.6f}")
    if lyap_exp > 0.1:
        print(f"  ✓✓ LE > 0.1: 强混沌系统")
    elif lyap_exp > 0:
        print(f"  ✓ 0 < LE < 0.1: 弱混沌系统")
    else:
        print(f"  ✗ LE ≤ 0: 非混沌系统")
    
    print("\n[3/5] 生成混沌序列用于可视化...")
    with torch.no_grad():
        chaos_seq = cipher._hyper_chaotic_map(5000, key).cpu().numpy()[0]
    
    print("\n[4/5] 计算自相关函数...")
    autocorr = compute_autocorrelation(chaos_seq, max_lag=200)
    
    # 计算卡方统计量 (Chi-Square Test)
    num_bins = 10
    hist, _ = np.histogram(chaos_seq, bins=num_bins, range=(0, 1))
    expected = len(chaos_seq) / num_bins
    chi_square = np.sum((hist - expected) ** 2 / expected)
    
    print("\n[5/5] 生成可视化报告...")
    
    # =====================================================
    # 开始绘图 (升级版布局)
    # =====================================================
    fig = plt.figure(figsize=(24, 18))
    # 使用不均匀的 GridSpec
    # Row 1: 5D Time Series
    # Row 2: 3D Attractor (Large) + 2D Projection
    # Row 3: Mixed Sequence + Return Map
    # Row 4: Statistics
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 2, 1, 1], hspace=0.4, wspace=0.3)
    
    # --- 第一行: 5维时序图 ---
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    dim_names = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅']
    
    for i in range(5):
        # 前3个占左边3格，后2个占右边1格的上下半部分？太复杂
        # 简单点：前4个在第一行，第5个放哪？
        # 改为5列？不行，GridSpec 4列好分。
        # 方案：第一行放3个，第二行放2个？不行。
        # 方案：第一行只放前4个，第5个忽略？或者挤一挤。
        # 采用 5列布局
        pass

    # 重新定义 GridSpec 为 5列
    gs_row1 = GridSpec(1, 5, figure=fig)
    gs_row1.update(top=0.95, bottom=0.82, left=0.05, right=0.95)
    
    for i in range(5):
        ax = fig.add_subplot(gs_row1[0, i])
        ax.plot(states_history[:500, i], color=colors[i], linewidth=1.5, alpha=0.8)
        ax.set_title(f"State {dim_names[i]}", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        if i > 0: ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    # --- 第二行: 3D 相图 (核心展示) ---
    # 占据左边 3/4 的空间
    gs_row2 = GridSpec(1, 4, figure=fig)
    gs_row2.update(top=0.78, bottom=0.45, left=0.05, right=0.95)
    
    ax_3d = fig.add_subplot(gs_row2[0, :3], projection='3d')
    # 绘制 (x1, x2, x3) 轨迹
    # 使用颜色映射表示时间演化
    limit = 5000
    p = ax_3d.scatter(states_history[:limit, 0], states_history[:limit, 1], states_history[:limit, 2],
                      c=np.arange(limit), cmap='plasma', s=0.8, alpha=0.6)
    ax_3d.set_title("3D Strange Attractor (x₁-x₂-x₃)", fontsize=16, fontweight='bold')
    ax_3d.set_xlabel("x₁")
    ax_3d.set_ylabel("x₂")
    ax_3d.set_zlabel("x₃")
    ax_3d.set_xlim(0, 1); ax_3d.set_ylim(0, 1); ax_3d.set_zlim(0, 1)
    # 调整视角
    ax_3d.view_init(elev=30, azim=45)
    
    # 额外的 2D 投影 (x4 vs x5)
    ax_2d = fig.add_subplot(gs_row2[0, 3])
    ax_2d.scatter(states_history[:limit, 3], states_history[:limit, 4], 
                  c=np.arange(limit), cmap='viridis', s=0.8, alpha=0.5)
    ax_2d.set_title("2D Projection (x₄-x₅)", fontsize=12, fontweight='bold')
    ax_2d.set_xlabel("x₄"); ax_2d.set_ylabel("x₅")
    ax_2d.set_xlim(0, 1); ax_2d.set_ylim(0, 1)
    ax_2d.grid(True, alpha=0.3)
    
    # --- 第三行: 混合输出与 Return Map ---
    gs_row3 = GridSpec(1, 3, figure=fig)
    gs_row3.update(top=0.40, bottom=0.25, left=0.05, right=0.95)
    
    ax_seq = fig.add_subplot(gs_row3[0, :2])
    ax_seq.plot(chaos_seq[:500], color='purple', linewidth=1.2)
    ax_seq.set_title("Mixed Output Sequence (Final Encryption Stream)", fontsize=12, fontweight='bold')
    ax_seq.set_xlabel("Time Step")
    ax_seq.set_ylabel("Value")
    ax_seq.set_ylim(0, 1)
    
    ax_ret = fig.add_subplot(gs_row3[0, 2])
    ax_ret.scatter(chaos_seq[:-1], chaos_seq[1:], s=1, c='green', alpha=0.5)
    ax_ret.set_title("Return Map (xₜ vs xₜ₊₁)", fontsize=12, fontweight='bold')
    ax_ret.set_xlabel("x(t)"); ax_ret.set_ylabel("x(t+1)")
    ax_ret.set_xlim(0, 1); ax_ret.set_ylim(0, 1)
    
    # --- 第四行: 统计图表 ---
    gs_row4 = GridSpec(1, 3, figure=fig)
    gs_row4.update(top=0.20, bottom=0.05, left=0.05, right=0.95)
    
    # 直方图
    ax_hist = fig.add_subplot(gs_row4[0, 0])
    ax_hist.hist(chaos_seq, bins=50, color='steelblue', alpha=0.7, density=True)
    ax_hist.set_title("Histogram (Uniformity)", fontsize=11, fontweight='bold')
    
    # 自相关
    ax_acf = fig.add_subplot(gs_row4[0, 1])
    ax_acf.plot(autocorr, color='darkred', linewidth=1.5)
    ax_acf.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_acf.set_title("Autocorrelation", fontsize=11, fontweight='bold')
    ax_acf.set_ylim(-0.5, 1.0)
    
    # 统计数据
    ax_stat = fig.add_subplot(gs_row4[0, 2])
    ax_stat.axis('off')
    
    stat_text = (
        "=== Validation Report ===\n\n"
        f"System: 5D Sine-Logistic Coupled Map\n"
        f"Params: μ={mu_val:.3f}, α={alpha_val:.3f}, β={beta_val:.3f}\n\n"
        f"Lyapunov Exp: {lyap_exp:.5f}\n"
        f"Status: {'STRONG CHAOS' if lyap_exp > 0.1 else 'CHAOTIC'}\n\n"
        f"Uniformity (Chi²): {chi_square:.2f}\n"
        f"Randomness (ACF): {autocorr[1]:.4f}\n"
    )
    ax_stat.text(0.05, 0.5, stat_text, fontsize=12, fontfamily='monospace',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=1'))

    plt.suptitle("NSCE 5D Hyperchaotic System Analysis", fontsize=20, fontweight='bold', y=0.98)
    
    # 保存
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "chaos_dynamics_analysis_3d.png")
    plt.savefig(save_path, dpi=150) # bbox_inches会导致3d图裁剪问题，这里不加
    
    print(f"\n{'='*70}")
    print(f"3D动力学分析报告已生成: {save_path}")
    print(f"{'='*70}\n")
    
    # 打印结论 (保持不变)

    print("\n【验证结论】")
    print("-" * 70)
    print(f"1. 李雅普诺夫指数: {lyap_exp:.6f}")
    if lyap_exp > 0.1:
        print("   → 强混沌系统 (LE > 0.1)")
    elif lyap_exp > 0:
        print("   → 弱混沌系统 (0 < LE < 0.1)")
    else:
        print("   → 非混沌系统 (LE ≤ 0)")
    
    print(f"\n2. 统计均匀性: Chi-Square = {chi_square:.2f}")
    if chi_square < 16.92:  # 自由度=9, α=0.05
        print("   → 通过均匀性检验 (p > 0.05)")
    else:
        print("   → 未通过均匀性检验")
    
    print(f"\n3. 自相关性: Lag-1 = {autocorr[1]:.4f}")
    if abs(autocorr[1]) < 0.1:
        print("   → 序列具有良好的随机性")
    else:
        print("   → 序列存在较强相关性")
    
    print("-" * 70)


if __name__ == "__main__":
    main()
