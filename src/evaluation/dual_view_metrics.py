# -*- coding: utf-8 -*-
"""
双视图安全指标评估器 (Dual-View Security Metrics)

实现 Z-view 和 C-view 分离评估：
- Z-view: 输出 NPCR、UACI、熵、相关性（不输出 NIST/χ²）
- C-view: 输出完整密码学指标（包含 NIST monobit、runs、χ²）

Requirements: 1.4, 1.5
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from src.evaluation.security_metrics import SecurityMetrics


@dataclass
class ZViewMetrics:
    """Z-view 评估指标（结构可用的隐私变换）"""
    # 信息熵
    entropy_original: float = 0.0
    entropy_encrypted: float = 0.0
    
    # 差分分析
    npcr: float = 0.0  # 像素变化率 (%)
    uaci: float = 0.0  # 平均强度变化 (%)
    
    # 相关性分析
    corr_original_horizontal: float = 0.0
    corr_original_vertical: float = 0.0
    corr_original_diagonal: float = 0.0
    corr_encrypted_horizontal: float = 0.0
    corr_encrypted_vertical: float = 0.0
    corr_encrypted_diagonal: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'entropy_original': self.entropy_original,
            'entropy_encrypted': self.entropy_encrypted,
            'npcr': self.npcr,
            'uaci': self.uaci,
            'corr_original_horizontal': self.corr_original_horizontal,
            'corr_original_vertical': self.corr_original_vertical,
            'corr_original_diagonal': self.corr_original_diagonal,
            'corr_encrypted_horizontal': self.corr_encrypted_horizontal,
            'corr_encrypted_vertical': self.corr_encrypted_vertical,
            'corr_encrypted_diagonal': self.corr_encrypted_diagonal
        }


@dataclass
class CViewMetrics:
    """C-view 评估指标（完整密码学指标）"""
    # 信息熵
    entropy_original: float = 0.0
    entropy_encrypted: float = 0.0
    
    # 差分分析
    npcr: float = 0.0
    uaci: float = 0.0
    
    # 相关性分析
    corr_original_horizontal: float = 0.0
    corr_original_vertical: float = 0.0
    corr_original_diagonal: float = 0.0
    corr_encrypted_horizontal: float = 0.0
    corr_encrypted_vertical: float = 0.0
    corr_encrypted_diagonal: float = 0.0
    
    # 卡方检验（仅 C-view）
    chi2: float = 0.0
    chi2_p_value: float = 0.0
    chi2_pass: bool = False
    
    # NIST 测试（仅 C-view）
    nist_monobit_p: float = 0.0
    nist_monobit_pass: bool = False
    nist_runs_p: float = 0.0
    nist_runs_pass: bool = False
    
    def to_dict(self) -> Dict[str, Union[float, bool]]:
        """转换为字典"""
        return {
            'entropy_original': self.entropy_original,
            'entropy_encrypted': self.entropy_encrypted,
            'npcr': self.npcr,
            'uaci': self.uaci,
            'corr_original_horizontal': self.corr_original_horizontal,
            'corr_original_vertical': self.corr_original_vertical,
            'corr_original_diagonal': self.corr_original_diagonal,
            'corr_encrypted_horizontal': self.corr_encrypted_horizontal,
            'corr_encrypted_vertical': self.corr_encrypted_vertical,
            'corr_encrypted_diagonal': self.corr_encrypted_diagonal,
            'chi2': self.chi2,
            'chi2_p_value': self.chi2_p_value,
            'chi2_pass': self.chi2_pass,
            'nist_monobit_p': self.nist_monobit_p,
            'nist_monobit_pass': self.nist_monobit_pass,
            'nist_runs_p': self.nist_runs_p,
            'nist_runs_pass': self.nist_runs_pass
        }


class DualViewSecurityMetrics:
    """
    双视图安全指标评估器
    
    **Property 9: Z-view Evaluation Exclusion**
    Z-view 评估不输出 NIST/χ² 测试结果
    
    **Property 10: C-view Evaluation Inclusion**
    C-view 评估包含完整密码学指标
    
    评估策略：
    - Z-view: 评估混沌变换的视觉混淆效果（NPCR、UACI、熵、相关性）
    - C-view: 评估 AEAD 封装后的密码学随机性（完整指标）
    """
    
    def __init__(self, is_q16_wrap: bool = True):
        """
        初始化评估器
        
        Args:
            is_q16_wrap: 是否使用 q16 wrap 模式（影响 uint8 转换）
        """
        self.is_q16_wrap = is_q16_wrap
        self._base_metrics = SecurityMetrics()
    
    def evaluate_zview(
        self,
        original: Union[np.ndarray, torch.Tensor],
        z_view: Union[np.ndarray, torch.Tensor],
        channel_wise: bool = False
    ) -> ZViewMetrics:
        """
        评估 Z-view 安全指标
        
        **Property 9: Z-view Evaluation Exclusion**
        仅输出 NPCR、UACI、熵、相关性，不输出 NIST/χ²
        
        Args:
            original: 原始图像 [H, W] 或 [C, H, W] 或 [B, C, H, W]
            z_view: Z-view 密文
            channel_wise: 是否按通道分别评估
        
        Returns:
            ZViewMetrics: Z-view 评估指标
        """
        # 转换为 numpy
        original_np = self._to_numpy(original)
        z_view_np = self._to_numpy(z_view)
        
        # 如果是多通道，取灰度或第一通道
        original_2d = self._to_2d(original_np)
        z_view_2d = self._to_2d(z_view_np)
        
        # 转换为 uint8
        original_u8 = self._to_uint8(original_2d, is_encrypted=False)
        z_view_u8 = self._to_uint8(z_view_2d, is_encrypted=True)
        
        # 计算基础指标
        metrics = ZViewMetrics()
        
        # 信息熵
        metrics.entropy_original = SecurityMetrics.calculate_entropy(original_u8)
        metrics.entropy_encrypted = SecurityMetrics.calculate_entropy(z_view_u8)
        
        # NPCR 和 UACI
        metrics.npcr = SecurityMetrics.calculate_npcr(original_u8, z_view_u8)
        metrics.uaci = SecurityMetrics.calculate_uaci(original_u8, z_view_u8)
        
        # 相关性
        for direction in ['horizontal', 'vertical', 'diagonal']:
            orig_corr = SecurityMetrics.calculate_correlation(original_u8, direction)
            enc_corr = SecurityMetrics.calculate_correlation(z_view_u8, direction)
            setattr(metrics, f'corr_original_{direction}', orig_corr)
            setattr(metrics, f'corr_encrypted_{direction}', enc_corr)
        
        return metrics
    
    def evaluate_cview(
        self,
        original: Union[np.ndarray, torch.Tensor],
        c_view: Union[np.ndarray, torch.Tensor],
        channel_wise: bool = False
    ) -> CViewMetrics:
        """
        评估 C-view 安全指标
        
        **Property 10: C-view Evaluation Inclusion**
        输出完整密码学指标，包含 NIST monobit、runs、χ²
        
        Args:
            original: 原始图像
            c_view: C-view 密文
            channel_wise: 是否按通道分别评估
        
        Returns:
            CViewMetrics: C-view 评估指标
        """
        # 转换为 numpy
        original_np = self._to_numpy(original)
        c_view_np = self._to_numpy(c_view)
        
        # 转换为 2D
        original_2d = self._to_2d(original_np)
        c_view_2d = self._to_2d(c_view_np)
        
        # 转换为 uint8
        original_u8 = self._to_uint8(original_2d, is_encrypted=False)
        c_view_u8 = self._to_uint8(c_view_2d, is_encrypted=True)
        
        # 计算完整指标
        metrics = CViewMetrics()
        
        # 信息熵
        metrics.entropy_original = SecurityMetrics.calculate_entropy(original_u8)
        metrics.entropy_encrypted = SecurityMetrics.calculate_entropy(c_view_u8)
        
        # NPCR 和 UACI
        metrics.npcr = SecurityMetrics.calculate_npcr(original_u8, c_view_u8)
        metrics.uaci = SecurityMetrics.calculate_uaci(original_u8, c_view_u8)
        
        # 相关性
        for direction in ['horizontal', 'vertical', 'diagonal']:
            orig_corr = SecurityMetrics.calculate_correlation(original_u8, direction)
            enc_corr = SecurityMetrics.calculate_correlation(c_view_u8, direction)
            setattr(metrics, f'corr_original_{direction}', orig_corr)
            setattr(metrics, f'corr_encrypted_{direction}', enc_corr)
        
        # 卡方检验（仅 C-view）
        chi2, p_value, is_uniform = SecurityMetrics.chi_square_test(c_view_u8)
        metrics.chi2 = chi2
        metrics.chi2_p_value = p_value
        metrics.chi2_pass = is_uniform
        
        # NIST 测试（仅 C-view）
        try:
            p_mono, pass_mono = SecurityMetrics.nist_monobit_test(c_view_u8)
            p_runs, pass_runs = SecurityMetrics.nist_runs_test(c_view_u8)
        except Exception:
            p_mono, pass_mono = float('nan'), False
            p_runs, pass_runs = float('nan'), False
        
        metrics.nist_monobit_p = p_mono
        metrics.nist_monobit_pass = pass_mono
        metrics.nist_runs_p = p_runs
        metrics.nist_runs_pass = pass_runs
        
        return metrics
    
    def evaluate_dual_view(
        self,
        original: Union[np.ndarray, torch.Tensor],
        z_view: Union[np.ndarray, torch.Tensor],
        c_view: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Union[ZViewMetrics, CViewMetrics]]:
        """
        同时评估 Z-view 和 C-view
        
        Args:
            original: 原始图像
            z_view: Z-view 密文
            c_view: C-view 密文
        
        Returns:
            包含 'z_view' 和 'c_view' 指标的字典
        """
        return {
            'z_view': self.evaluate_zview(original, z_view),
            'c_view': self.evaluate_cview(original, c_view)
        }
    
    def check_zview_standards(self, metrics: ZViewMetrics) -> Dict[str, bool]:
        """
        检查 Z-view 是否满足安全标准
        
        标准（针对混沌变换）：
        - 信息熵 ≥ 7.5（比 C-view 宽松）
        - NPCR ≥ 99.0%
        - UACI ∈ [28%, 38%]
        - 相关性 |r| < 0.2
        """
        checks = {
            'entropy': metrics.entropy_encrypted >= 7.5,
            'npcr': metrics.npcr >= 99.0,
            'uaci': 28.0 <= metrics.uaci <= 38.0,
            'correlation': all([
                abs(metrics.corr_encrypted_horizontal) < 0.2,
                abs(metrics.corr_encrypted_vertical) < 0.2,
                abs(metrics.corr_encrypted_diagonal) < 0.2
            ])
        }
        return checks
    
    def check_cview_standards(self, metrics: CViewMetrics) -> Dict[str, bool]:
        """
        检查 C-view 是否满足安全标准
        
        标准（针对密码学加密）：
        - 信息熵 ≥ 7.9
        - NPCR ≥ 99.5%
        - UACI ∈ [30%, 36%]
        - 相关性 |r| < 0.1
        - 卡方检验通过
        - NIST monobit 通过
        - NIST runs 通过
        """
        checks = {
            'entropy': metrics.entropy_encrypted >= 7.9,
            'npcr': metrics.npcr >= 99.5,
            'uaci': 30.0 <= metrics.uaci <= 36.0,
            'correlation': all([
                abs(metrics.corr_encrypted_horizontal) < 0.1,
                abs(metrics.corr_encrypted_vertical) < 0.1,
                abs(metrics.corr_encrypted_diagonal) < 0.1
            ]),
            'chi_square': metrics.chi2_pass,
            'nist_monobit': metrics.nist_monobit_pass,
            'nist_runs': metrics.nist_runs_pass
        }
        return checks
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """转换为 numpy 数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    
    def _to_2d(self, data: np.ndarray) -> np.ndarray:
        """转换为 2D 数组（取第一个样本的灰度）"""
        if data.ndim == 4:  # [B, C, H, W]
            data = data[0]  # 取第一个样本
        if data.ndim == 3:  # [C, H, W]
            if data.shape[0] == 3:
                # RGB 转灰度
                data = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
            else:
                data = data[0]  # 取第一通道
        return data
    
    def _to_uint8(self, data: np.ndarray, is_encrypted: bool = False) -> np.ndarray:
        """转换为 uint8"""
        if data.dtype == np.uint8:
            return data
        
        if is_encrypted and self.is_q16_wrap:
            return SecurityMetrics.q16_to_uint8(data, method='high8')
        else:
            return SecurityMetrics.float_to_uint8(data)


    def print_zview_report(self, metrics: ZViewMetrics, checks: Dict[str, bool] = None):
        """打印 Z-view 评估报告"""
        if checks is None:
            checks = self.check_zview_standards(metrics)
        
        print("\n" + "=" * 70)
        print("Z-view 安全性评估报告（混沌变换层）")
        print("=" * 70)
        
        print("\n【信息熵】")
        print(f"  原始图像：{metrics.entropy_original:.4f} bits/pixel")
        print(f"  Z-view：{metrics.entropy_encrypted:.4f} bits/pixel")
        print(f"  标准：≥ 7.5 bits/pixel")
        print(f"  结果：{'✓ 通过' if checks['entropy'] else '❌ 未通过'}")
        
        print("\n【NPCR（像素变化率）】")
        print(f"  值：{metrics.npcr:.2f}%")
        print(f"  标准：≥ 99.0%")
        print(f"  结果：{'✓ 通过' if checks['npcr'] else '❌ 未通过'}")
        
        print("\n【UACI（平均强度变化）】")
        print(f"  值：{metrics.uaci:.2f}%")
        print(f"  标准：28% - 38%")
        print(f"  结果：{'✓ 通过' if checks['uaci'] else '❌ 未通过'}")
        
        print("\n【相关性】")
        for direction in ['horizontal', 'vertical', 'diagonal']:
            orig = getattr(metrics, f'corr_original_{direction}')
            enc = getattr(metrics, f'corr_encrypted_{direction}')
            print(f"  {direction:10s}：原始 {orig:6.4f} → Z-view {enc:6.4f}")
        print(f"  标准：|r| < 0.2")
        print(f"  结果：{'✓ 通过' if checks['correlation'] else '❌ 未通过'}")
        
        print("\n【注意】Z-view 不评估 NIST/χ² 测试（仅适用于 C-view）")
        
        self._print_summary(checks, "Z-view")
    
    def print_cview_report(self, metrics: CViewMetrics, checks: Dict[str, bool] = None):
        """打印 C-view 评估报告"""
        if checks is None:
            checks = self.check_cview_standards(metrics)
        
        print("\n" + "=" * 70)
        print("C-view 安全性评估报告（AEAD 密码学层）")
        print("=" * 70)
        
        print("\n【信息熵】")
        print(f"  原始图像：{metrics.entropy_original:.4f} bits/pixel")
        print(f"  C-view：{metrics.entropy_encrypted:.4f} bits/pixel")
        print(f"  标准：≥ 7.9 bits/pixel")
        print(f"  结果：{'✓ 通过' if checks['entropy'] else '❌ 未通过'}")
        
        print("\n【NPCR（像素变化率）】")
        print(f"  值：{metrics.npcr:.2f}%")
        print(f"  标准：≥ 99.5%")
        print(f"  结果：{'✓ 通过' if checks['npcr'] else '❌ 未通过'}")
        
        print("\n【UACI（平均强度变化）】")
        print(f"  值：{metrics.uaci:.2f}%")
        print(f"  标准：30% - 36%")
        print(f"  结果：{'✓ 通过' if checks['uaci'] else '❌ 未通过'}")
        
        print("\n【相关性】")
        for direction in ['horizontal', 'vertical', 'diagonal']:
            orig = getattr(metrics, f'corr_original_{direction}')
            enc = getattr(metrics, f'corr_encrypted_{direction}')
            print(f"  {direction:10s}：原始 {orig:6.4f} → C-view {enc:6.4f}")
        print(f"  标准：|r| < 0.1")
        print(f"  结果：{'✓ 通过' if checks['correlation'] else '❌ 未通过'}")
        
        print("\n【卡方均匀性检验】")
        print(f"  χ² 统计量：{metrics.chi2:.2f}")
        print(f"  p 值：{metrics.chi2_p_value:.4f}")
        print(f"  标准：p > 0.05（接受均匀分布）")
        print(f"  结果：{'✓ 通过' if checks['chi_square'] else '❌ 未通过'}")
        
        print("\n【NIST 子集测试】")
        print(f"  Monobit p：{metrics.nist_monobit_p:.4f}  标准：p > 0.01  结果：{'✓' if checks['nist_monobit'] else '❌'}")
        print(f"  Runs    p：{metrics.nist_runs_p:.4f}  标准：p > 0.01  结果：{'✓' if checks['nist_runs'] else '❌'}")
        
        self._print_summary(checks, "C-view")
    
    def _print_summary(self, checks: Dict[str, bool], view_name: str):
        """打印总结"""
        print("\n" + "=" * 70)
        print(f"{view_name} 总体评估")
        print("=" * 70)
        
        total = len(checks)
        passed = sum(checks.values())
        
        print(f"\n通过测试：{passed}/{total}")
        
        if passed == total:
            print(f"\n✓ {view_name} 所有安全指标均达标！")
        else:
            print(f"\n⚠️ {passed}/{total} 项指标达标")
            failed = [k for k, v in checks.items() if not v]
            print(f"未通过：{', '.join(failed)}")
        
        print("=" * 70)


def get_zview_metric_keys() -> List[str]:
    """
    获取 Z-view 指标键列表
    
    **Property 9: Z-view Evaluation Exclusion**
    不包含 chi2、nist_monobit、nist_runs
    """
    return [
        'entropy_original', 'entropy_encrypted',
        'npcr', 'uaci',
        'corr_original_horizontal', 'corr_original_vertical', 'corr_original_diagonal',
        'corr_encrypted_horizontal', 'corr_encrypted_vertical', 'corr_encrypted_diagonal'
    ]


def get_cview_metric_keys() -> List[str]:
    """
    获取 C-view 指标键列表
    
    **Property 10: C-view Evaluation Inclusion**
    包含完整密码学指标
    """
    return [
        'entropy_original', 'entropy_encrypted',
        'npcr', 'uaci',
        'corr_original_horizontal', 'corr_original_vertical', 'corr_original_diagonal',
        'corr_encrypted_horizontal', 'corr_encrypted_vertical', 'corr_encrypted_diagonal',
        'chi2', 'chi2_p_value', 'chi2_pass',
        'nist_monobit_p', 'nist_monobit_pass',
        'nist_runs_p', 'nist_runs_pass'
    ]
