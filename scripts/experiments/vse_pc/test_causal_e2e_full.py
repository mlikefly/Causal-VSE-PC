"""
Causal-VSE-PC 完整端到端测试脚本（含安全评估）
==============================================

功能：
- 测试完整的因果推断驱动加密流程
- 验证ATE/CATE计算
- 安全性评估（NPCR、UACI、熵、相关性等）
- 加密/解密可视化
- 解密验证（PSNR）
- 记录完整实验结果

使用方法:
    python scripts/experiments/vse_pc/test_causal_e2e_full.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vse_pc.causal_analysis import CausalPrivacyAnalyzer
from src.vse_pc.privacy_budget import AdaptivePrivacyBudget
from src.neural.unet import UNetSaliencyDetector
from src.cipher.scne_cipher import SCNECipherAPI
from src.crypto.key_system import HierarchicalKeySystem
from src.utils.datasets import get_celeba_dataloader
from src.evaluation.security_metrics import SecurityMetrics


# ============== 辅助函数 ==============

def create_rule_based_classifier(num_classes: int = 20):
    """
    创建基于规则的分类器（无需训练）
    规则：使用亮度+对比度直接分类
    """
    import torch.nn as nn
    
    class RuleBasedClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            
        def forward(self, x):
            B = x.shape[0]
            device = x.device
            
            brightness = x.mean(dim=[2, 3]).squeeze()
            contrast = x.std(dim=[2, 3]).squeeze()
            
            if brightness.dim() == 0:
                brightness = brightness.unsqueeze(0)
            if contrast.dim() == 0:
                contrast = contrast.unsqueeze(0)
            
            pred = ((brightness * 10 + contrast * 5) % self.num_classes).long()
            
            logits = torch.zeros(B, self.num_classes, device=device)
            logits.scatter_(1, pred.unsqueeze(1), 10.0)
            
            return logits
    
    return RuleBasedClassifier(num_classes)


def generate_labels_from_images(images: torch.Tensor, num_classes: int = 20) -> torch.Tensor:
    """使用与分类器相同的规则生成标签"""
    with torch.no_grad():
        brightness = images.mean(dim=[2, 3]).squeeze()
        contrast = images.std(dim=[2, 3]).squeeze()
        if brightness.dim() == 0:
            brightness = brightness.unsqueeze(0)
        if contrast.dim() == 0:
            contrast = contrast.unsqueeze(0)
        labels = ((brightness * 10 + contrast * 5) % num_classes).long()
    return labels


def generate_semantic_mask_from_brightness(images: torch.Tensor) -> torch.Tensor:
    """基于图像亮度生成语义掩码"""
    semantic_mask = torch.zeros_like(images)
    semantic_mask[images > 0.65] = 0.9  # 敏感区域
    semantic_mask[(images >= 0.35) & (images <= 0.65)] = 0.5  # 任务区域
    semantic_mask[images < 0.35] = 0.1  # 背景区域
    return semantic_mask


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算PSNR（峰值信噪比）"""
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算SSIM（结构相似性）- 简化版本"""
    # 简化实现，实际应使用skimage.metrics.structural_similarity
    orig = original.cpu().numpy().flatten()
    recon = reconstructed.cpu().numpy().flatten()
    
    mu_x = np.mean(orig)
    mu_y = np.mean(recon)
    sigma_x = np.std(orig)
    sigma_y = np.std(recon)
    sigma_xy = np.mean((orig - mu_x) * (recon - mu_y))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    
    return float(ssim)



# ============== 安全评估函数 ==============

def evaluate_security(
    original: torch.Tensor,
    encrypted: torch.Tensor,
    sample_idx: int = 0,
    is_q16_wrap: bool = True
) -> Dict:
    """
    评估加密的安全性指标
    
    重要说明（两视图架构）：
    - 如果 is_q16_wrap=True（C视图）：使用 q16→uint8 高8位转换，评估密码学随机性
    - 如果 is_q16_wrap=False（Z视图）：使用标准转换，评估隐私变换效果
    
    Args:
        original: 原始图像 [B, 1, H, W]
        encrypted: 加密图像 [B, 1, H, W]
        sample_idx: 要评估的样本索引
        is_q16_wrap: 是否来自 q16 wrap（影响 uint8 转换方式）
    
    Returns:
        metrics: 安全指标字典
    """
    # 转换为numpy float格式
    orig_f = original[sample_idx, 0].cpu().numpy()
    enc_f = encrypted[sample_idx, 0].cpu().numpy()
    
    # 根据是否 q16 wrap 选择正确的转换方式
    if is_q16_wrap:
        # C视图：q16 wrap 后的密文，使用高8位转换保持均匀性
        orig_np = SecurityMetrics.float_to_uint8(orig_f)
        enc_np = SecurityMetrics.q16_to_uint8(enc_f, method='high8')
    else:
        # Z视图：隐私变换表示，使用标准转换
        orig_np = SecurityMetrics.float_to_uint8(orig_f)
        enc_np = SecurityMetrics.float_to_uint8(enc_f)
    
    # 使用SecurityMetrics计算所有指标
    metrics = SecurityMetrics.evaluate_image(orig_np, enc_np, is_q16_wrap=is_q16_wrap)
    
    # 检查是否满足安全标准
    checks = SecurityMetrics.check_security_standards(metrics)
    
    return {
        'metrics': metrics,
        'checks': checks,
        'original_np': orig_np,
        'encrypted_np': enc_np,
        'view_type': 'C (crypto)' if is_q16_wrap else 'Z (transform)'
    }


def evaluate_decryption(
    original: torch.Tensor,
    decrypted: torch.Tensor
) -> Dict:
    """
    评估解密质量
    
    Args:
        original: 原始图像 [B, 1, H, W]
        decrypted: 解密图像 [B, 1, H, W]
    
    Returns:
        quality: 解密质量指标
    """
    psnr_values = []
    ssim_values = []
    mae_values = []
    
    B = original.shape[0]
    for i in range(B):
        psnr = calculate_psnr(original[i], decrypted[i])
        ssim = calculate_ssim(original[i], decrypted[i])
        mae = torch.mean(torch.abs(original[i] - decrypted[i])).item()
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mae_values.append(mae)
    
    return {
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'is_lossless': np.mean(psnr_values) > 40  # PSNR > 40dB 认为近无损
    }


def evaluate_key_sensitivity(
    cipher: SCNECipherAPI,
    images: torch.Tensor,
    correct_password: str,
    privacy_level: float = 1.0
) -> Dict:
    """
    评估密钥敏感性
    
    测试：使用错误密钥解密时的效果
    """
    # 使用正确密钥加密
    encrypted, enc_info = cipher.encrypt_simple(images, privacy_level=privacy_level)
    
    # 尝试使用错误密钥解密
    wrong_passwords = [
        correct_password + "1",  # 添加字符
        correct_password[:-1],   # 删除字符
        correct_password.replace(correct_password[0], 'X'),  # 修改字符
    ]
    
    results = {
        'correct_password': correct_password,
        'wrong_password_tests': []
    }
    
    for wrong_pwd in wrong_passwords:
        try:
            # 创建使用错误密钥的cipher
            wrong_cipher = SCNECipherAPI(
                password=wrong_pwd,
                image_size=images.shape[-1],
                device=str(images.device)
            )
            
            # 尝试解密（应该失败或产生错误结果）
            # 注意：这里可能会抛出异常或返回错误结果
            mask = torch.ones_like(images)
            decrypted = wrong_cipher.cipher.decrypt(
                encrypted, enc_info, mask, password=wrong_pwd
            )
            
            # 计算与原图的差异
            mae = torch.mean(torch.abs(images - decrypted)).item()
            psnr = calculate_psnr(images, decrypted)
            
            results['wrong_password_tests'].append({
                'password': wrong_pwd[:3] + '***',  # 隐藏密码
                'mae': mae,
                'psnr': psnr,
                'decryption_failed': psnr < 10  # PSNR < 10dB 认为解密失败
            })
        except Exception as e:
            results['wrong_password_tests'].append({
                'password': wrong_pwd[:3] + '***',
                'error': str(e),
                'decryption_failed': True
            })
    
    return results



# ============== 可视化函数 ==============

def visualize_encryption_results(
    original: torch.Tensor,
    encrypted_dict: Dict[float, torch.Tensor],
    decrypted: Optional[torch.Tensor],
    semantic_mask: torch.Tensor,
    privacy_map: torch.Tensor,
    security_results: Dict,
    causal_report: Dict,
    save_dir: Path,
    sample_idx: int = 0
):
    """
    生成完整的可视化报告
    
    Args:
        original: 原始图像 [B, 1, H, W]
        encrypted_dict: {privacy_level: encrypted_tensor}
        decrypted: 解密图像 [B, 1, H, W]
        semantic_mask: 语义掩码 [B, 1, H, W]
        privacy_map: 隐私预算图 [B, 1, H, W]
        security_results: 安全评估结果
        causal_report: 因果分析报告
        save_dir: 保存目录
        sample_idx: 要可视化的样本索引
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
        return
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 图1: 加密效果对比 =====
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle('Encryption Effect Comparison', fontsize=14, fontweight='bold')
    
    # 第一行：原始图像和不同隐私级别的加密图像
    orig_img = original[sample_idx, 0].cpu().numpy()
    axes1[0, 0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes1[0, 0].set_title('Original')
    axes1[0, 0].axis('off')
    
    privacy_levels = sorted([k for k in encrypted_dict.keys() if k > 0])[:3]
    for i, level in enumerate(privacy_levels):
        enc_img = encrypted_dict[level][sample_idx, 0].cpu().numpy()
        axes1[0, i+1].imshow(enc_img, cmap='gray', vmin=0, vmax=1)
        axes1[0, i+1].set_title(f'Privacy={level:.1f}')
        axes1[0, i+1].axis('off')
    
    # 第二行：语义掩码、隐私预算图、解密图像、差异图
    axes1[1, 0].imshow(semantic_mask[sample_idx, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes1[1, 0].set_title('Semantic Mask')
    axes1[1, 0].axis('off')
    
    axes1[1, 1].imshow(privacy_map[sample_idx, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes1[1, 1].set_title('Privacy Budget')
    axes1[1, 1].axis('off')
    
    if decrypted is not None:
        dec_img = decrypted[sample_idx, 0].cpu().numpy()
        axes1[1, 2].imshow(dec_img, cmap='gray', vmin=0, vmax=1)
        axes1[1, 2].set_title('Decrypted')
        axes1[1, 2].axis('off')
        
        diff_img = np.abs(orig_img - dec_img)
        axes1[1, 3].imshow(diff_img, cmap='hot', vmin=0, vmax=0.1)
        axes1[1, 3].set_title('Difference (x10)')
        axes1[1, 3].axis('off')
    else:
        axes1[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        axes1[1, 2].axis('off')
        axes1[1, 3].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        axes1[1, 3].axis('off')
    
    plt.tight_layout()
    fig1.savefig(save_dir / 'encryption_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ 保存: encryption_comparison.png")
    
    # ===== 图2: 直方图对比 =====
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Histogram Analysis', fontsize=14, fontweight='bold')
    
    orig_np = security_results['original_np']
    enc_np = security_results['encrypted_np']
    
    # 原始图像直方图
    axes2[0, 0].hist(orig_np.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes2[0, 0].set_title('Original Histogram')
    axes2[0, 0].set_xlabel('Pixel Value')
    axes2[0, 0].set_ylabel('Frequency')
    
    # 加密图像直方图
    axes2[0, 1].hist(enc_np.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
    axes2[0, 1].set_title('Encrypted Histogram')
    axes2[0, 1].set_xlabel('Pixel Value')
    axes2[0, 1].set_ylabel('Frequency')
    
    # 原始图像相关性散点图
    h_orig = orig_np[:, :-1].flatten()
    h_orig_neighbor = orig_np[:, 1:].flatten()
    sample_indices = np.random.choice(len(h_orig), min(5000, len(h_orig)), replace=False)
    axes2[1, 0].scatter(h_orig[sample_indices], h_orig_neighbor[sample_indices], 
                        s=1, alpha=0.3, c='blue')
    axes2[1, 0].set_title(f'Original Correlation (r={security_results["metrics"]["corr_original_horizontal"]:.4f})')
    axes2[1, 0].set_xlabel('Pixel Value')
    axes2[1, 0].set_ylabel('Adjacent Pixel Value')
    
    # 加密图像相关性散点图
    h_enc = enc_np[:, :-1].flatten()
    h_enc_neighbor = enc_np[:, 1:].flatten()
    axes2[1, 1].scatter(h_enc[sample_indices], h_enc_neighbor[sample_indices], 
                        s=1, alpha=0.3, c='red')
    axes2[1, 1].set_title(f'Encrypted Correlation (r={security_results["metrics"]["corr_encrypted_horizontal"]:.4f})')
    axes2[1, 1].set_xlabel('Pixel Value')
    axes2[1, 1].set_ylabel('Adjacent Pixel Value')
    
    plt.tight_layout()
    fig2.savefig(save_dir / 'histogram_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ 保存: histogram_analysis.png")
    
    # ===== 图3: ATE/CATE柱状图 =====
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Causal Effect Analysis', fontsize=14, fontweight='bold')
    
    # ATE
    ate_result = causal_report['ate']
    ate = ate_result.get('ate', 0)
    if 'ci_lower' in ate_result and 'ci_upper' in ate_result:
        yerr = [[ate - ate_result['ci_lower']], [ate_result['ci_upper'] - ate]]
    else:
        yerr = None
    
    axes3[0].bar(['ATE'], [ate], yerr=yerr, capsize=10, 
                 alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    axes3[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes3[0].set_ylabel('Average Treatment Effect')
    axes3[0].set_title('ATE with 95% CI')
    axes3[0].text(0, ate, f'{ate:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # CATE
    cate_results = causal_report['cate']
    regions = ['Sensitive', 'Task', 'Background']
    cates = [cate_results.get(r.lower(), {}).get('cate', 0) for r in regions]
    colors = ['red', 'orange', 'green']
    
    axes3[1].bar(regions, cates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes3[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes3[1].set_ylabel('Conditional Average Treatment Effect')
    axes3[1].set_title('CATE by Region Type')
    for i, (r, c) in enumerate(zip(regions, cates)):
        axes3[1].text(i, c, f'{c:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig3.savefig(save_dir / 'causal_effects.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ 保存: causal_effects.png")



def generate_security_report(
    security_results: Dict,
    decryption_quality: Optional[Dict],
    causal_report: Dict,
    performance_results: Dict,
    save_path: Path
):
    """
    生成完整的安全评估报告（Markdown格式）
    """
    metrics = security_results['metrics']
    checks = security_results['checks']
    
    report = []
    report.append("# Causal-VSE-PC 安全评估报告")
    report.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # 1. 安全性指标
    report.append("## 1. 安全性指标\n")
    report.append("| 指标 | 值 | 标准 | 状态 |")
    report.append("|------|-----|------|------|")
    report.append(f"| 信息熵 | {metrics['entropy_encrypted']:.4f} bits | ≥ 7.9 | {'✅' if checks['entropy'] else '❌'} |")
    report.append(f"| NPCR | {metrics['npcr']:.2f}% | ≥ 99.5% | {'✅' if checks['npcr'] else '❌'} |")
    report.append(f"| UACI | {metrics['uaci']:.2f}% | 30-36% | {'✅' if checks['uaci'] else '❌'} |")
    report.append(f"| 卡方检验 | χ²={metrics['chi2']:.2f} | p>0.05 | {'✅' if checks['chi_square'] else '❌'} |")
    report.append(f"| NIST Monobit | p={metrics.get('nist_monobit_p', 0):.4f} | p>0.01 | {'✅' if checks.get('nist_monobit', False) else '❌'} |")
    report.append(f"| NIST Runs | p={metrics.get('nist_runs_p', 0):.4f} | p>0.01 | {'✅' if checks.get('nist_runs', False) else '❌'} |")
    
    # 2. 相关性分析
    report.append("\n## 2. 相邻像素相关性\n")
    report.append("| 方向 | 原始图像 | 加密图像 | 状态 |")
    report.append("|------|----------|----------|------|")
    for direction in ['horizontal', 'vertical', 'diagonal']:
        orig_corr = metrics[f'corr_original_{direction}']
        enc_corr = metrics[f'corr_encrypted_{direction}']
        status = '✅' if abs(enc_corr) < 0.1 else '❌'
        report.append(f"| {direction} | {orig_corr:.4f} | {enc_corr:.4f} | {status} |")
    
    # 3. 解密质量
    if decryption_quality:
        report.append("\n## 3. 解密质量\n")
        report.append("| 指标 | 值 | 状态 |")
        report.append("|------|-----|------|")
        report.append(f"| PSNR | {decryption_quality['psnr_mean']:.2f} ± {decryption_quality['psnr_std']:.2f} dB | {'✅' if decryption_quality['psnr_mean'] > 30 else '❌'} |")
        report.append(f"| SSIM | {decryption_quality['ssim_mean']:.4f} ± {decryption_quality['ssim_std']:.4f} | {'✅' if decryption_quality['ssim_mean'] > 0.9 else '❌'} |")
        report.append(f"| MAE | {decryption_quality['mae_mean']:.6f} ± {decryption_quality['mae_std']:.6f} | {'✅' if decryption_quality['mae_mean'] < 0.01 else '❌'} |")
        report.append(f"| 近无损 | {'是' if decryption_quality['is_lossless'] else '否'} | - |")
    
    # 4. 因果效应分析
    report.append("\n## 4. 因果效应分析\n")
    ate = causal_report['ate']
    report.append(f"**ATE (平均处理效应)**: {ate['ate']:.4f}")
    if 'ci_lower' in ate:
        report.append(f" (95% CI: [{ate['ci_lower']:.4f}, {ate['ci_upper']:.4f}])")
    report.append("\n")
    
    report.append("\n| 区域类型 | CATE | 95% CI |")
    report.append("|----------|------|--------|")
    for region in ['sensitive', 'task', 'background']:
        cate = causal_report['cate'].get(region, {})
        cate_val = cate.get('cate', float('nan'))
        if 'ci_lower' in cate:
            ci = f"[{cate['ci_lower']:.4f}, {cate['ci_upper']:.4f}]"
        else:
            ci = "N/A"
        report.append(f"| {region} | {cate_val:.4f} | {ci} |")
    
    # 5. ML性能影响
    report.append("\n## 5. ML性能影响\n")
    report.append("| 隐私级别 | 准确率 | 性能下降 |")
    report.append("|----------|--------|----------|")
    for level, result in sorted(performance_results.items()):
        acc = result['accuracy']
        drop = performance_results[0.0]['accuracy'] - acc if 0.0 in performance_results else 0
        report.append(f"| {level:.1f} | {acc:.1%} | {drop:.1%} |")
    
    # 6. 总结
    report.append("\n## 6. 总结\n")
    passed = sum(checks.values())
    total = len(checks)
    report.append(f"- 安全指标通过: {passed}/{total}")
    report.append(f"- ATE显著性: {'✅ 显著' if ate['ate'] < -0.1 else '❌ 不显著'}")
    if decryption_quality:
        report.append(f"- 解密质量: {'✅ 良好' if decryption_quality['psnr_mean'] > 30 else '❌ 较差'}")
    
    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  ✓ 报告已保存: {save_path}")



# ============== 主测试函数 ==============

def main():
    """主测试函数"""
    print("=" * 70)
    print("Causal-VSE-PC 完整端到端测试（含安全评估）")
    print("=" * 70)
    
    # 设置
    batch_size = 32
    image_size = 256
    task_type = 'classification'
    password = "test_password_123"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 20
    
    # 结果保存目录
    result_dir = project_root / 'scripts' / 'results' / 'causal_analysis_full'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n设置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Task type: {task_type}")
    print(f"  Device: {device}")
    print(f"  结果目录: {result_dir}")
    
    # ===== 1. 初始化组件 =====
    print("\n[1/8] 初始化组件...")
    
    unet = UNetSaliencyDetector(in_channels=1, base_channels=64).to(device)
    unet.eval()
    
    causal_analyzer = CausalPrivacyAnalyzer(use_history_baseline=True)
    allocator = AdaptivePrivacyBudget()
    
    cipher = SCNECipherAPI(
        password=password,
        image_size=image_size,
        device=device,
        use_frequency=True,
        use_fft=True
    )
    
    classifier = create_rule_based_classifier(num_classes).to(device)
    classifier.eval()
    
    print("✓ 组件初始化完成")
    
    # ===== 2. 加载数据 =====
    print("\n[2/8] 加载数据...")
    
    try:
        dataloader = get_celeba_dataloader(
            root_dir='data/CelebA-HQ',
            split='test',
            batch_size=batch_size,
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
        
        print(f"✓ 成功加载 {len(images)} 张CelebA-HQ图像")
    except Exception as e:
        print(f"⚠️ 无法加载CelebA-HQ: {e}")
        print("  生成测试数据...")
        images = torch.rand(batch_size, 1, image_size, image_size, device=device)
        print(f"✓ 生成 {batch_size} 张测试图像")
    
    labels = generate_labels_from_images(images, num_classes)
    print(f"  图像形状: {images.shape}")
    print(f"  标签范围: [{labels.min().item()}, {labels.max().item()}]")
    
    # 验证基线准确率
    with torch.no_grad():
        logits = classifier(images)
        preds = logits.argmax(dim=1)
        baseline_acc = (preds == labels).float().mean().item()
    print(f"✓ 基线准确率: {baseline_acc:.1%}")
    
    # ===== 3. 语义分析 =====
    print("\n[3/8] 语义分析...")
    
    with torch.no_grad():
        try:
            semantic_mask = unet(images)
            semantic_mask = torch.sigmoid(semantic_mask)
            if (semantic_mask.max() - semantic_mask.min()) < 0.1:
                semantic_mask = generate_semantic_mask_from_brightness(images)
        except:
            semantic_mask = generate_semantic_mask_from_brightness(images)
    
    print(f"  语义掩码范围: [{semantic_mask.min():.3f}, {semantic_mask.max():.3f}]")
    
    # ===== 4. 因果分析与隐私分配 =====
    print("\n[4/8] 因果分析与隐私分配...")
    
    causal_result = causal_analyzer.analyze_allocation(
        semantic_mask, task_type=task_type, privacy_map=None
    )
    
    privacy_map = allocator.allocate(
        semantic_mask,
        task_type=task_type,
        global_privacy=1.0,
        causal_suggestion=causal_result['suggestion']
    )
    
    print(f"  因果建议: {causal_result['suggestion']['privacy_budget']}")
    print(f"  隐私预算范围: [{privacy_map.min():.3f}, {privacy_map.max():.3f}]")
    
    # ===== 5. 多强度加密测试（两视图架构） =====
    print("\n[5/8] 多强度加密测试（两视图架构）...")
    print("  说明：")
    print("    - Z视图（enable_crypto_wrap=False）：隐私变换表示，用于ML推理")
    print("    - C视图（enable_crypto_wrap=True）：强加密密文，用于存储/传输")
    
    # ===== Z视图：隐私变换（用于ML推理和攻击评估） =====
    print("\n  [Z视图] 隐私变换（enable_crypto_wrap=False）:")
    cipher_z = SCNECipherAPI(
        password=password,
        image_size=image_size,
        device=device,
        use_frequency=True,
        use_fft=True,
        enable_crypto_wrap=False  # 关闭 wrap，保持可用性
    )
    
    privacy_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
    encrypted_dict_z = {}  # Z视图加密结果
    enc_info_dict_z = {}
    
    for level in privacy_levels:
        if level == 0.0:
            encrypted_dict_z[level] = images.clone()
            enc_info_dict_z[level] = {'method': 'no_encryption'}
        else:
            try:
                encrypted, enc_info = cipher_z.encrypt_simple(
                    images,
                    privacy_level=level,
                    semantic_preserving=False,
                    mask=semantic_mask
                )
                encrypted_dict_z[level] = encrypted
                enc_info_dict_z[level] = enc_info
                mae = (encrypted - images).abs().mean().item()
                print(f"    privacy_level={level:.1f}: MAE={mae:.4f}")
            except Exception as e:
                print(f"    privacy_level={level:.1f}: 加密失败 - {e}")
                encrypted_dict_z[level] = images.clone()
                enc_info_dict_z[level] = {'method': 'failed'}
    
    # ===== C视图：强加密（用于密码学评估） =====
    print("\n  [C视图] 强加密（enable_crypto_wrap=True）:")
    encrypted_dict_c = {}  # C视图加密结果
    enc_info_dict_c = {}
    
    for level in [1.0]:  # C视图只测试最高隐私级别
        try:
            encrypted, enc_info = cipher.encrypt_simple(
                images,
                privacy_level=level,
                semantic_preserving=False,
                mask=semantic_mask
            )
            encrypted_dict_c[level] = encrypted
            enc_info_dict_c[level] = enc_info
            mae = (encrypted - images).abs().mean().item()
            print(f"    privacy_level={level:.1f}: MAE={mae:.4f} (含ChaCha20 wrap)")
        except Exception as e:
            print(f"    privacy_level={level:.1f}: 加密失败 - {e}")
            encrypted_dict_c[level] = images.clone()
            enc_info_dict_c[level] = {'method': 'failed'}
    
    # 兼容后续代码：使用Z视图作为主要加密结果
    encrypted_dict = encrypted_dict_z
    enc_info_dict = enc_info_dict_z
    
    # ===== 6. ML推理与因果效应（使用Z视图） =====
    print("\n[6/8] ML推理与因果效应计算（Z视图）...")
    print("  说明：使用Z视图（无crypto_wrap）进行ML推理，验证隐私-可用性权衡")
    
    performance_results = {}
    with torch.no_grad():
        for level in privacy_levels:
            encrypted = encrypted_dict_z[level]  # 使用Z视图
            logits = classifier(encrypted)
            preds = logits.argmax(dim=1)
            correct = (preds == labels).float()
            acc = correct.mean().item()
            performance_results[level] = {
                'accuracy': acc,
                'performance_per_sample': correct
            }
            print(f"  隐私级别 {level:.1f} - 准确率: {acc:.3f}")
    
    # 计算因果效应
    causal_report = causal_analyzer.compute_causal_effects(
        semantic_mask=semantic_mask,
        privacy_map=privacy_map,
        performance_encrypted=performance_results[1.0]['performance_per_sample'],
        performance_original=performance_results[0.0]['performance_per_sample'],
        task_type=task_type,
        conf_interval=True
    )
    
    print(f"\n  ATE: {causal_report['ate']['ate']:.4f}")
    if 'ci_lower' in causal_report['ate']:
        print(f"  95% CI: [{causal_report['ate']['ci_lower']:.4f}, {causal_report['ate']['ci_upper']:.4f}]")
    
    # ===== 7. 安全性评估（两视图分离） =====
    print("\n[7/8] 安全性评估（两视图分离）...")
    
    # ===== Z视图安全评估（隐私变换效果） =====
    print("\n  [Z视图] 隐私变换效果评估:")
    security_results_z = evaluate_security(
        images, encrypted_dict_z[1.0], sample_idx=0, is_q16_wrap=False
    )
    print(f"    熵: {security_results_z['metrics']['entropy_encrypted']:.4f} bits")
    print(f"    NPCR: {security_results_z['metrics']['npcr']:.2f}%")
    print(f"    UACI: {security_results_z['metrics']['uaci']:.2f}%")
    print(f"    水平相关性: {security_results_z['metrics']['corr_encrypted_horizontal']:.4f}")
    
    # ===== C视图安全评估（密码学随机性） =====
    print("\n  [C视图] 密码学随机性评估（含ChaCha20 wrap）:")
    security_results_c = evaluate_security(
        images, encrypted_dict_c[1.0], sample_idx=0, is_q16_wrap=True
    )
    
    # 打印C视图完整安全报告
    SecurityMetrics.print_report(security_results_c['metrics'], security_results_c['checks'])
    
    # 使用C视图作为主要安全结果（用于后续报告）
    security_results = security_results_c
    
    # ===== 解密验证（Z视图） =====
    print("\n解密验证（Z视图）...")
    decryption_quality = None
    try:
        mask = torch.ones_like(images)
        decrypted = cipher_z.cipher.decrypt(
            encrypted_dict_z[1.0],
            enc_info_dict_z[1.0],
            mask,
            password=password
        )
        decryption_quality = evaluate_decryption(images, decrypted)
        print(f"  PSNR: {decryption_quality['psnr_mean']:.2f} ± {decryption_quality['psnr_std']:.2f} dB")
        print(f"  SSIM: {decryption_quality['ssim_mean']:.4f} ± {decryption_quality['ssim_std']:.4f}")
        print(f"  近无损: {'是' if decryption_quality['is_lossless'] else '否'}")
    except Exception as e:
        print(f"  ⚠️ 解密失败: {e}")
        decrypted = None
    
    # ===== 8. 生成报告与可视化 =====
    print("\n[8/8] 生成报告与可视化...")
    
    # 可视化
    visualize_encryption_results(
        original=images,
        encrypted_dict=encrypted_dict,
        decrypted=decrypted,
        semantic_mask=semantic_mask,
        privacy_map=privacy_map,
        security_results=security_results,
        causal_report=causal_report,
        save_dir=result_dir,
        sample_idx=0
    )
    
    # 生成Markdown报告
    generate_security_report(
        security_results=security_results,
        decryption_quality=decryption_quality,
        causal_report=causal_report,
        performance_results=performance_results,
        save_path=result_dir / 'security_report.md'
    )
    
    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    print(f"\n关键结果（两视图架构）:")
    print(f"\n  [Z视图 - 隐私变换/ML推理]")
    print(f"    - 原始准确率: {performance_results[0.0]['accuracy']:.3f}")
    print(f"    - 加密准确率(0.3): {performance_results[0.3]['accuracy']:.3f}")
    print(f"    - 加密准确率(0.7): {performance_results[0.7]['accuracy']:.3f}")
    print(f"    - 加密准确率(1.0): {performance_results[1.0]['accuracy']:.3f}")
    print(f"    - ATE: {causal_report['ate']['ate']:.4f}")
    print(f"    - 熵(Z): {security_results_z['metrics']['entropy_encrypted']:.4f} bits")
    print(f"    - NPCR(Z): {security_results_z['metrics']['npcr']:.2f}%")
    
    print(f"\n  [C视图 - 密码学安全/存储传输]")
    print(f"    - 熵(C): {security_results_c['metrics']['entropy_encrypted']:.4f} bits")
    print(f"    - NPCR(C): {security_results_c['metrics']['npcr']:.2f}%")
    print(f"    - UACI(C): {security_results_c['metrics']['uaci']:.2f}%")
    print(f"    - 卡方p值: {security_results_c['metrics']['chi2_p_value']:.4f}")
    print(f"    - NIST Monobit p: {security_results_c['metrics'].get('nist_monobit_p', 0):.4f}")
    print(f"    - NIST Runs p: {security_results_c['metrics'].get('nist_runs_p', 0):.4f}")
    
    passed = sum(security_results_c['checks'].values())
    total = len(security_results_c['checks'])
    print(f"    - 安全指标: {passed}/{total} 通过")
    
    print(f"\n结果已保存到: {result_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
