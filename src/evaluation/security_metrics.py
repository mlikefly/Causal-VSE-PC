"""
统计安全指标评估

指标：
1. 信息熵（Information Entropy）
2. NPCR（Number of Pixel Change Rate）
3. UACI（Unified Average Changing Intensity）
4. 相关性（Correlation）
5. 卡方检验（Chi-Square Test）
6. NIST 子集测试：单比特频率（Monobit）、游程（Runs）
"""

import numpy as np
import math
try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None
from typing import Dict, Tuple


class SecurityMetrics:
    """安全性指标计算"""
    
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        计算信息熵
        
        Args:
            image: 图像数组
        
        Returns:
            entropy: 信息熵（bits/pixel）
        """
        # 计算直方图
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # 归一化为概率
        prob = hist / hist.sum()
        
        # 过滤零概率
        prob = prob[prob > 0]
        
        # 计算熵
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    @staticmethod
    def calculate_npcr(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        计算NPCR（像素变化率）
        
        Args:
            image1: 原始图像
            image2: 加密图像
        
        Returns:
            npcr: NPCR百分比
        """
        # 计算不同像素的数量
        diff = (image1 != image2).astype(np.float32)
        npcr = (diff.sum() / diff.size) * 100.0
        
        return npcr
    
    @staticmethod
    def calculate_uaci(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        计算UACI（平均强度变化）
        
        Args:
            image1: 原始图像
            image2: 加密图像
        
        Returns:
            uaci: UACI百分比
        """
        # 计算强度差异
        diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        uaci = (diff.sum() / (diff.size * 255.0)) * 100.0
        
        return uaci
    
    @staticmethod
    def calculate_correlation(image: np.ndarray, direction='horizontal') -> float:
        """
        计算相邻像素相关性
        
        Args:
            image: 图像数组
            direction: 方向（'horizontal', 'vertical', 'diagonal'）
        
        Returns:
            correlation: 相关系数
        """
        H, W = image.shape
        
        if direction == 'horizontal':
            x = image[:, :-1].flatten()
            y = image[:, 1:].flatten()
        elif direction == 'vertical':
            x = image[:-1, :].flatten()
            y = image[1:, :].flatten()
        elif direction == 'diagonal':
            x = image[:-1, :-1].flatten()
            y = image[1:, 1:].flatten()
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # 计算相关系数
        correlation = np.corrcoef(x, y)[0, 1]
        
        return correlation
    
    @staticmethod
    def chi_square_test(image: np.ndarray) -> Tuple[float, float, bool]:
        """
        卡方均匀性检验
        
        Args:
            image: 图像数组
        
        Returns:
            chi2: 卡方统计量
            p_value: p值
            is_uniform: 是否通过均匀性检验（p > 0.05）
        """
        # 计算直方图
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # 期望频率（均匀分布）
        expected = np.full(256, image.size / 256.0)
        
        # 卡方检验
        if stats is not None:
            chi2, p_value = stats.chisquare(hist, expected)
            # p > 0.05 表示接受均匀分布假设
            is_uniform = (p_value > 0.05)
        else:
            # 纯 numpy 回退：仅计算统计量，并用临界值判断是否均匀
            # 自由度 df = 256 - 1 = 255；alpha=0.05 的临界值近似 293.25
            chi2 = float(((hist - expected) ** 2 / (expected + 1e-12)).sum())
            p_value = float('nan')
            CRIT_95_DF255 = 293.25
            is_uniform = (chi2 < CRIT_95_DF255)
        
        return chi2, p_value, is_uniform
    
    @staticmethod
    def evaluate_image(
        original: np.ndarray, 
        encrypted: np.ndarray
    ) -> Dict[str, float]:
        """
        评估单张图像的所有指标
        
        Args:
            original: 原始图像
            encrypted: 加密图像
        
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 信息熵
        metrics['entropy_original'] = SecurityMetrics.calculate_entropy(original)
        metrics['entropy_encrypted'] = SecurityMetrics.calculate_entropy(encrypted)
        
        # NPCR和UACI
        metrics['npcr'] = SecurityMetrics.calculate_npcr(original, encrypted)
        metrics['uaci'] = SecurityMetrics.calculate_uaci(original, encrypted)
        
        # 相关性（三个方向）
        for direction in ['horizontal', 'vertical', 'diagonal']:
            metrics[f'corr_original_{direction}'] = SecurityMetrics.calculate_correlation(
                original, direction
            )
            metrics[f'corr_encrypted_{direction}'] = SecurityMetrics.calculate_correlation(
                encrypted, direction
            )
        
        chi2, p_value, is_uniform = SecurityMetrics.chi_square_test(encrypted)
        metrics['chi2'] = chi2
        metrics['chi2_p_value'] = p_value
        metrics['chi2_pass'] = is_uniform

        try:
            p_mono, pass_mono = SecurityMetrics.nist_monobit_test(encrypted)
            p_runs, pass_runs = SecurityMetrics.nist_runs_test(encrypted)
        except Exception:
            p_mono, pass_mono = float('nan'), False
            p_runs, pass_runs = float('nan'), False
        metrics['nist_monobit_p'] = p_mono
        metrics['nist_monobit_pass'] = pass_mono
        metrics['nist_runs_p'] = p_runs
        metrics['nist_runs_pass'] = pass_runs

        return metrics

    @staticmethod
    def _image_bits(image: np.ndarray) -> np.ndarray:
        """将 uint8 图像转为 bit 序列 [n_bits] in {0,1}."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        bits = np.unpackbits(image.flatten())
        return bits

    @staticmethod
    def nist_monobit_test(image: np.ndarray, alpha: float = 0.01) -> Tuple[float, bool]:
        """NIST SP800-22 单比特频率测试（Monobit）。返回 (p, pass)。"""
        bits = SecurityMetrics._image_bits(image)
        n = bits.size
        if n == 0:
            return float('nan'), False
        s = np.sum(2 * bits - 1)
        sobs = abs(s) / math.sqrt(n)
        p = math.erfc(sobs / math.sqrt(2.0))
        return float(p), bool(p > alpha)

    @staticmethod
    def nist_runs_test(image: np.ndarray, alpha: float = 0.01) -> Tuple[float, bool]:
        """NIST SP800-22 游程测试（Runs）。返回 (p, pass)。"""
        bits = SecurityMetrics._image_bits(image)
        n = bits.size
        if n < 2:
            return float('nan'), False
        pi_hat = bits.mean()
        tau = 2.0 / math.sqrt(n)
        if abs(pi_hat - 0.5) >= tau:
            return 0.0, False
        # 计算游程数
        v_obs = 1 + int(np.sum(bits[1:] != bits[:-1]))
        num = abs(v_obs - 2.0 * n * pi_hat * (1.0 - pi_hat))
        den = 2.0 * math.sqrt(2.0 * n) * pi_hat * (1.0 - pi_hat) + 1e-12
        p = math.erfc(num / den)
        return float(p), bool(p > alpha)
    
    @staticmethod
    def check_security_standards(metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        检查是否满足安全标准
        
        标准：
        - 信息熵 ≥ 7.9
        - NPCR ≥ 99.5%
        - UACI ∈ [30%, 36%]
        - 相关性 ≈ 0（|r| < 0.1）
        - 卡方检验通过
        
        Args:
            metrics: 指标字典
        
        Returns:
            checks: 检查结果
        """
        checks = {}
        
        # 信息熵
        checks['entropy'] = metrics['entropy_encrypted'] >= 7.9
        
        # NPCR
        checks['npcr'] = metrics['npcr'] >= 99.5
        
        # UACI
        checks['uaci'] = 30.0 <= metrics['uaci'] <= 36.0
        
        # 相关性（所有方向）
        corr_checks = []
        for direction in ['horizontal', 'vertical', 'diagonal']:
            corr = abs(metrics[f'corr_encrypted_{direction}'])
            corr_checks.append(corr < 0.1)
        checks['correlation'] = all(corr_checks)
        
        # 卡方检验
        checks['chi_square'] = metrics['chi2_pass']
        
        # NIST 子集测试
        checks['nist_monobit'] = bool(metrics.get('nist_monobit_pass', False))
        checks['nist_runs'] = bool(metrics.get('nist_runs_pass', False))
        
        return checks
    
    @staticmethod
    def print_report(metrics: Dict[str, float], checks: Dict[str, bool]):
        """打印评估报告"""
        
        print("\n" + "="*70)
        print("安全性评估报告")
        print("="*70)
        
        print("\n【信息熵】")
        print(f"  原始图像：{metrics['entropy_original']:.4f} bits/pixel")
        print(f"  加密图像：{metrics['entropy_encrypted']:.4f} bits/pixel")
        print(f"  标准：≥ 7.9 bits/pixel")
        print(f"  结果：{'✓ 通过' if checks['entropy'] else '❌ 未通过'}")
        
        print("\n【NPCR（像素变化率）】")
        print(f"  值：{metrics['npcr']:.2f}%")
        print(f"  标准：≥ 99.5%")
        print(f"  结果：{'✓ 通过' if checks['npcr'] else '❌ 未通过'}")
        
        print("\n【UACI（平均强度变化）】")
        print(f"  值：{metrics['uaci']:.2f}%")
        print(f"  标准：30% - 36%")
        print(f"  结果：{'✓ 通过' if checks['uaci'] else '❌ 未通过'}")
        
        print("\n【相关性】")
        for direction in ['horizontal', 'vertical', 'diagonal']:
            orig_corr = metrics[f'corr_original_{direction}']
            enc_corr = metrics[f'corr_encrypted_{direction}']
            print(f"  {direction:10s}：原始 {orig_corr:6.4f} → 加密 {enc_corr:6.4f}")
        print(f"  标准：|r| < 0.1")
        print(f"  结果：{'✓ 通过' if checks['correlation'] else '❌ 未通过'}")
        
        print("\n【卡方均匀性检验】")
        print(f"  χ² 统计量：{metrics['chi2']:.2f}")
        print(f"  p 值：{metrics['chi2_p_value']:.4f}")
        print(f"  标准：p > 0.05（接受均匀分布）")
        print(f"  结果：{'✓ 通过' if checks['chi_square'] else '❌ 未通过'}")
        
        print("\n【NIST 子集测试】")
        print(f"  Monobit p：{metrics.get('nist_monobit_p', float('nan')):.4f}  标准：p > 0.01  结果：{'✓' if checks.get('nist_monobit', False) else '❌'}")
        print(f"  Runs    p：{metrics.get('nist_runs_p', float('nan')):.4f}  标准：p > 0.01  结果：{'✓' if checks.get('nist_runs', False) else '❌'}")
        
        print("\n" + "="*70)
        print("总体评估")
        print("="*70)
        
        total_checks = len(checks)
        passed_checks = sum(checks.values())
        
        print(f"\n通过测试：{passed_checks}/{total_checks}")
        
        if passed_checks == total_checks:
            print("\n✓ 所有安全指标均达标！")
        else:
            print(f"\n⚠️ {total_checks - passed_checks} 项指标未达标")
        
        print("="*70)


def test_security_metrics():
    """测试安全指标计算"""
    
    print("="*70)
    print("测试安全指标计算")
    print("="*70)
    
    # 创建测试图像
    np.random.seed(42)
    
    original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    encrypted = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # 计算指标
    metrics = SecurityMetrics.evaluate_image(original, encrypted)
    
    # 检查标准
    checks = SecurityMetrics.check_security_standards(metrics)
    
    # 打印报告
    SecurityMetrics.print_report(metrics, checks)


if __name__ == "__main__":
    test_security_metrics()









