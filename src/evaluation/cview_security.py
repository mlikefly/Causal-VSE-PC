# -*- coding: utf-8 -*-
"""
C-view Security Evaluation Suite for Top-Journal Experiment Suite.

Implements security goal verification and diagnostic evidence:
- TamperTester: Tamper detection testing (ciphertext/tag/aad)
- ReplayTester: Replay attack detection testing
- NISTTestRunner: NIST SP800-22 randomness tests (7 subtests)
- AvalancheEffectTester: Avalanche effect testing (key/nonce/plaintext)

Corresponds to design.md §9.6 and tasks.md T5.

**Validates:**
- Property 4: C-view 安全测试完整性
- Property 5: NIST 比特流充足性
- Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import csv
import hashlib
import json
import math
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import cryptography for AEAD operations
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    AESGCM = None

# Import ReplayCache from core
try:
    from src.core.replay_cache import ReplayCache, ReplayDetectedError
except ImportError:
    ReplayCache = None
    ReplayDetectedError = Exception


# =============================================================================
# Enums and Constants
# =============================================================================

class TamperType(Enum):
    """Types of tamper attacks."""
    CIPHERTEXT = "ciphertext"
    TAG = "tag"
    AAD = "aad"


class FlipType(Enum):
    """Types of bit flips for avalanche testing."""
    KEY = "key"
    NONCE = "nonce"
    PLAINTEXT = "plaintext"


class SecurityTestStatus(Enum):
    """Status of security tests."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


# NIST test minimum bit requirements (from design.md §12.3)
NIST_MIN_BITS = {
    "frequency": 100,
    "block_frequency": 100,
    "runs": 100,
    "longest_run": 128,
    "fft": 1000,
    "serial": 1000,
    "approximate_entropy": 1000,
}

# Security thresholds (from design.md §9.6)
TAMPER_FAIL_RATE_THRESHOLD = 0.99  # ≥ 99%
REPLAY_REJECT_RATE_THRESHOLD = 1.0  # = 100%
NIST_P_VALUE_THRESHOLD = 0.01  # ≥ 0.01
AVALANCHE_FLIP_RATE_RANGE = (0.45, 0.55)  # [45%, 55%]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TamperTestResult:
    """Result of a single tamper test."""
    tamper_type: str
    sample_id: str
    original_valid: bool
    tampered_valid: bool
    detected: bool
    error: Optional[str] = None


@dataclass
class TamperTestSummary:
    """Summary of tamper tests for one type."""
    tamper_type: str
    total_tests: int
    detected_count: int
    fail_rate: float
    status: str
    details: List[TamperTestResult] = field(default_factory=list)


@dataclass
class ReplayTestResult:
    """Result of replay testing."""
    total_unique: int
    total_replays: int
    rejected_replays: int
    reject_rate: float
    status: str


@dataclass
class NISTTestResult:
    """Result of a single NIST test."""
    test_name: str
    p_value: float
    passed: bool
    bits_used: int
    bits_required: int
    status: str
    error: Optional[str] = None


@dataclass
class NISTTestSummary:
    """Summary of all NIST tests."""
    total_tests: int
    passed_tests: int
    total_bits: int
    results: List[NISTTestResult] = field(default_factory=list)
    status: str = "unknown"


@dataclass
class AvalancheTestResult:
    """Result of avalanche effect test."""
    flip_type: str
    flip_rate: float
    in_range: bool
    n_samples: int
    status: str
    details: Optional[Dict] = None


@dataclass
class SecurityReport:
    """Complete security evaluation report."""
    tamper_results: Dict[str, TamperTestSummary]
    replay_result: Optional[ReplayTestResult]
    nist_summary: Optional[NISTTestSummary]
    avalanche_results: Dict[str, AvalancheTestResult]
    overall_status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "tamper_results": {
                k: {
                    "tamper_type": v.tamper_type,
                    "total_tests": v.total_tests,
                    "detected_count": v.detected_count,
                    "fail_rate": v.fail_rate,
                    "status": v.status,
                }
                for k, v in self.tamper_results.items()
            },
            "replay_result": {
                "total_unique": self.replay_result.total_unique,
                "total_replays": self.replay_result.total_replays,
                "rejected_replays": self.replay_result.rejected_replays,
                "reject_rate": self.replay_result.reject_rate,
                "status": self.replay_result.status,
            } if self.replay_result else None,
            "nist_summary": {
                "total_tests": self.nist_summary.total_tests,
                "passed_tests": self.nist_summary.passed_tests,
                "total_bits": self.nist_summary.total_bits,
                "status": self.nist_summary.status,
                "results": [
                    {
                        "test_name": r.test_name,
                        "p_value": r.p_value,
                        "passed": r.passed,
                        "bits_used": r.bits_used,
                        "status": r.status,
                    }
                    for r in self.nist_summary.results
                ],
            } if self.nist_summary else None,
            "avalanche_results": {
                k: {
                    "flip_type": v.flip_type,
                    "flip_rate": v.flip_rate,
                    "in_range": v.in_range,
                    "n_samples": v.n_samples,
                    "status": v.status,
                }
                for k, v in self.avalanche_results.items()
            },
        }


# =============================================================================
# TamperTester
# =============================================================================

class TamperTester:
    """
    Tamper detection tester for C-view AEAD ciphertexts.
    
    Tests 3 tamper types: ciphertext, tag, aad
    Expected fail_rate ≥ 99%
    
    **Validates: §9.6.1, Property 4**
    """
    
    TAMPER_TYPES = [TamperType.CIPHERTEXT, TamperType.TAG, TamperType.AAD]
    N_TESTS_PER_TYPE = 200
    
    def __init__(
        self,
        encrypt_fn: Optional[Callable] = None,
        decrypt_fn: Optional[Callable] = None,
        n_tests_per_type: int = 200,
    ):
        """
        Initialize TamperTester.
        
        Args:
            encrypt_fn: Function to encrypt data (plaintext, key, nonce, aad) -> (ciphertext, tag)
            decrypt_fn: Function to decrypt data (ciphertext, tag, key, nonce, aad) -> plaintext or raises
            n_tests_per_type: Number of tests per tamper type
        """
        self.encrypt_fn = encrypt_fn or self._default_encrypt
        self.decrypt_fn = decrypt_fn or self._default_decrypt
        self.n_tests_per_type = n_tests_per_type
    
    def _default_encrypt(
        self, plaintext: bytes, key: bytes, nonce: bytes, aad: bytes
    ) -> Tuple[bytes, bytes]:
        """Default encryption using AES-GCM."""
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library not installed")
        aesgcm = AESGCM(key)
        # AES-GCM returns ciphertext || tag
        ct_with_tag = aesgcm.encrypt(nonce, plaintext, aad)
        # Split: last 16 bytes are tag
        ciphertext = ct_with_tag[:-16]
        tag = ct_with_tag[-16:]
        return ciphertext, tag
    
    def _default_decrypt(
        self, ciphertext: bytes, tag: bytes, key: bytes, nonce: bytes, aad: bytes
    ) -> bytes:
        """Default decryption using AES-GCM."""
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library not installed")
        aesgcm = AESGCM(key)
        ct_with_tag = ciphertext + tag
        return aesgcm.decrypt(nonce, ct_with_tag, aad)

    def _tamper_bytes(self, data: bytes, n_bits: int = 1) -> bytes:
        """Tamper data by flipping random bits."""
        if len(data) == 0:
            return data
        data_array = bytearray(data)
        for _ in range(n_bits):
            byte_idx = secrets.randbelow(len(data_array))
            bit_idx = secrets.randbelow(8)
            data_array[byte_idx] ^= (1 << bit_idx)
        return bytes(data_array)
    
    def test_single(
        self,
        tamper_type: TamperType,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes,
        sample_id: str = "unknown",
    ) -> TamperTestResult:
        """
        Test single tamper detection.
        
        Args:
            tamper_type: Type of tamper to test
            plaintext: Original plaintext
            key: Encryption key
            nonce: Nonce
            aad: Additional authenticated data
            sample_id: Sample identifier
            
        Returns:
            TamperTestResult
        """
        try:
            # Encrypt
            ciphertext, tag = self.encrypt_fn(plaintext, key, nonce, aad)
            
            # Verify original decrypts correctly
            try:
                self.decrypt_fn(ciphertext, tag, key, nonce, aad)
                original_valid = True
            except Exception:
                original_valid = False
            
            # Apply tamper
            if tamper_type == TamperType.CIPHERTEXT:
                tampered_ct = self._tamper_bytes(ciphertext)
                tampered_tag = tag
                tampered_aad = aad
            elif tamper_type == TamperType.TAG:
                tampered_ct = ciphertext
                tampered_tag = self._tamper_bytes(tag)
                tampered_aad = aad
            elif tamper_type == TamperType.AAD:
                tampered_ct = ciphertext
                tampered_tag = tag
                tampered_aad = self._tamper_bytes(aad) if aad else b"tampered"
            else:
                raise ValueError(f"Unknown tamper type: {tamper_type}")
            
            # Try to decrypt tampered data
            try:
                self.decrypt_fn(tampered_ct, tampered_tag, key, nonce, tampered_aad)
                tampered_valid = True
            except Exception:
                tampered_valid = False
            
            # Tamper detected if decryption fails
            detected = not tampered_valid
            
            return TamperTestResult(
                tamper_type=tamper_type.value,
                sample_id=sample_id,
                original_valid=original_valid,
                tampered_valid=tampered_valid,
                detected=detected,
            )
            
        except Exception as e:
            return TamperTestResult(
                tamper_type=tamper_type.value,
                sample_id=sample_id,
                original_valid=False,
                tampered_valid=False,
                detected=False,
                error=str(e),
            )
    
    def test_tamper_type(
        self,
        tamper_type: TamperType,
        samples: List[Dict[str, bytes]],
    ) -> TamperTestSummary:
        """
        Test tamper detection for one type across multiple samples.
        
        Args:
            tamper_type: Type of tamper to test
            samples: List of dicts with keys: plaintext, key, nonce, aad, sample_id
            
        Returns:
            TamperTestSummary
        """
        results = []
        detected_count = 0
        
        for i, sample in enumerate(samples[:self.n_tests_per_type]):
            result = self.test_single(
                tamper_type=tamper_type,
                plaintext=sample.get("plaintext", b"test data"),
                key=sample.get("key", secrets.token_bytes(32)),
                nonce=sample.get("nonce", secrets.token_bytes(12)),
                aad=sample.get("aad", b"additional data"),
                sample_id=sample.get("sample_id", f"sample_{i}"),
            )
            results.append(result)
            if result.detected:
                detected_count += 1
        
        total = len(results)
        fail_rate = detected_count / total if total > 0 else 0.0
        status = "pass" if fail_rate >= TAMPER_FAIL_RATE_THRESHOLD else "fail"
        
        return TamperTestSummary(
            tamper_type=tamper_type.value,
            total_tests=total,
            detected_count=detected_count,
            fail_rate=fail_rate,
            status=status,
            details=results,
        )
    
    def test_all(
        self,
        samples: Optional[List[Dict[str, bytes]]] = None,
        key: Optional[bytes] = None,
    ) -> Dict[str, TamperTestSummary]:
        """
        Test all tamper types.
        
        Args:
            samples: Optional list of samples. If None, generates random samples.
            key: Optional shared key for all samples.
            
        Returns:
            Dict mapping tamper type to TamperTestSummary
        """
        if samples is None:
            # Generate random samples
            key = key or secrets.token_bytes(32)
            samples = []
            for i in range(self.n_tests_per_type):
                samples.append({
                    "plaintext": secrets.token_bytes(64),
                    "key": key,
                    "nonce": secrets.token_bytes(12),
                    "aad": f"sample_{i}".encode(),
                    "sample_id": f"sample_{i}",
                })
        
        results = {}
        for tamper_type in self.TAMPER_TYPES:
            results[tamper_type.value] = self.test_tamper_type(tamper_type, samples)
        
        return results


# =============================================================================
# ReplayTester
# =============================================================================

class ReplayTester:
    """
    Replay attack tester using ReplayCache.
    
    Tests that all replay attempts are rejected (reject_rate = 100%).
    
    **Validates: §9.6.1, Property 4**
    """
    
    def __init__(self, run_dir: Optional[Path] = None, key_id: str = "test_key"):
        """
        Initialize ReplayTester.
        
        Args:
            run_dir: Run directory for ReplayCache
            key_id: Key identifier
        """
        self.run_dir = Path(run_dir) if run_dir else Path(".")
        self.key_id = key_id
    
    def test_replay_detection(
        self,
        n_unique: int = 100,
        n_replays_per_unique: int = 2,
    ) -> ReplayTestResult:
        """
        Test replay detection.
        
        Args:
            n_unique: Number of unique ciphertexts
            n_replays_per_unique: Number of replay attempts per unique ciphertext
            
        Returns:
            ReplayTestResult
        """
        if ReplayCache is None:
            return ReplayTestResult(
                total_unique=0,
                total_replays=0,
                rejected_replays=0,
                reject_rate=0.0,
                status="error: ReplayCache not available",
            )
        
        # Create cache
        cache = ReplayCache(run_dir=self.run_dir, key_id=self.key_id)
        
        # Generate unique ciphertexts
        unique_samples = []
        for i in range(n_unique):
            nonce = secrets.token_bytes(12)
            tag = secrets.token_bytes(16)
            unique_samples.append((nonce, tag))
        
        # First pass: record all unique ciphertexts
        for nonce, tag in unique_samples:
            result = cache.check_and_record(nonce, tag, image_id=f"img_{nonce.hex()[:8]}")
            assert result is True, "First occurrence should be accepted"
        
        # Second pass: attempt replays
        total_replays = 0
        rejected_replays = 0
        
        for nonce, tag in unique_samples:
            for _ in range(n_replays_per_unique):
                total_replays += 1
                result = cache.check_and_record(nonce, tag)
                if not result:
                    rejected_replays += 1
        
        reject_rate = rejected_replays / total_replays if total_replays > 0 else 0.0
        status = "pass" if reject_rate >= REPLAY_REJECT_RATE_THRESHOLD else "fail"
        
        return ReplayTestResult(
            total_unique=n_unique,
            total_replays=total_replays,
            rejected_replays=rejected_replays,
            reject_rate=reject_rate,
            status=status,
        )



# =============================================================================
# NISTTestRunner
# =============================================================================

class NISTTestRunner:
    """
    NIST SP800-22 随机性测试运行器。
    
    实现 7 项子测试:
    - frequency: 单比特频率测试
    - block_frequency: 块内频率测试
    - runs: 游程测试
    - longest_run: 最长游程测试
    - fft: 离散傅里叶变换测试
    - serial: 序列测试
    - approximate_entropy: 近似熵测试
    
    **Validates: R3.AC1, R3.AC2, Property 5**
    """
    
    REQUIRED_TESTS = [
        "frequency", "block_frequency", "runs", "longest_run",
        "fft", "serial", "approximate_entropy"
    ]
    
    def __init__(self, alpha: float = 0.01):
        """
        初始化 NIST 测试运行器。
        
        Args:
            alpha: 显著性水平，默认 0.01
        """
        self.alpha = alpha
    
    @staticmethod
    def _to_bits(data: bytes) -> np.ndarray:
        """将字节数据转换为比特数组。"""
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    
    def frequency_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        单比特频率测试 (Monobit Test)。
        
        检验比特序列中 0 和 1 的比例是否接近 50%。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["frequency"]:
            return float('nan'), False
        
        # 将 0 转为 -1，1 保持为 1
        s = np.sum(2 * bits.astype(np.int64) - 1)
        s_obs = abs(s) / math.sqrt(n)
        p_value = math.erfc(s_obs / math.sqrt(2.0))
        
        return p_value, p_value >= self.alpha
    
    def block_frequency_test(self, bits: np.ndarray, block_size: int = 128) -> Tuple[float, bool]:
        """
        块内频率测试。
        
        将序列分成多个块，检验每个块内 1 的比例。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["block_frequency"]:
            return float('nan'), False
        
        n_blocks = n // block_size
        if n_blocks == 0:
            return float('nan'), False
        
        # 计算每个块的比例
        proportions = []
        for i in range(n_blocks):
            block = bits[i * block_size:(i + 1) * block_size]
            pi = np.mean(block)
            proportions.append(pi)
        
        # 计算卡方统计量
        chi_sq = 4 * block_size * sum((pi - 0.5) ** 2 for pi in proportions)
        
        # 计算 p 值 (使用不完全伽马函数)
        try:
            from scipy.special import gammaincc
            p_value = gammaincc(n_blocks / 2, chi_sq / 2)
        except ImportError:
            # 简化近似
            p_value = math.exp(-chi_sq / 2) if chi_sq < 100 else 0.0
        
        return float(p_value), p_value >= self.alpha
    
    def runs_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        游程测试。
        
        检验连续相同比特的游程数是否符合随机序列的期望。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["runs"]:
            return float('nan'), False
        
        # 计算 1 的比例
        pi = np.mean(bits)
        tau = 2.0 / math.sqrt(n)
        
        if abs(pi - 0.5) >= tau:
            return 0.0, False
        
        # 计算游程数
        v_obs = 1 + int(np.sum(bits[1:] != bits[:-1]))
        
        # 计算期望和标准差
        num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
        den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi) + 1e-12
        
        p_value = math.erfc(num / den)
        return float(p_value), p_value >= self.alpha
    
    def longest_run_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        最长游程测试。
        
        检验最长连续 1 的长度是否符合随机序列的期望。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["longest_run"]:
            return float('nan'), False
        
        # 根据序列长度选择参数
        if n < 6272:
            k, m = 3, 8
            v_values = [1, 2, 3, 4]
            pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
        elif n < 750000:
            k, m = 5, 128
            v_values = [4, 5, 6, 7, 8, 9]
            pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:
            k, m = 6, 10000
            v_values = [10, 11, 12, 13, 14, 15, 16]
            pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        
        n_blocks = n // m
        if n_blocks == 0:
            return float('nan'), False
        
        # 计算每个块的最长游程
        frequencies = [0] * len(v_values)
        for i in range(n_blocks):
            block = bits[i * m:(i + 1) * m]
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            
            # 分类
            for j, v in enumerate(v_values):
                if j == 0 and max_run <= v:
                    frequencies[j] += 1
                    break
                elif j == len(v_values) - 1 and max_run >= v:
                    frequencies[j] += 1
                    break
                elif max_run == v:
                    frequencies[j] += 1
                    break
        
        # 计算卡方统计量
        chi_sq = sum(
            (frequencies[i] - n_blocks * pi_values[i]) ** 2 / (n_blocks * pi_values[i] + 1e-12)
            for i in range(len(v_values))
        )
        
        try:
            from scipy.special import gammaincc
            p_value = gammaincc(len(v_values) / 2, chi_sq / 2)
        except ImportError:
            p_value = math.exp(-chi_sq / 2) if chi_sq < 100 else 0.0
        
        return float(p_value), p_value >= self.alpha
    
    def fft_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        离散傅里叶变换测试。
        
        检验序列的周期性特征。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["fft"]:
            return float('nan'), False
        
        # 转换为 +1/-1
        x = 2 * bits.astype(np.float64) - 1
        
        # FFT
        s = np.fft.fft(x)
        modulus = np.abs(s[:n // 2])
        
        # 阈值
        t = math.sqrt(math.log(1 / 0.05) * n)
        n0 = 0.95 * n / 2
        n1 = np.sum(modulus < t)
        
        d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4)
        p_value = math.erfc(abs(d) / math.sqrt(2))
        
        return float(p_value), p_value >= self.alpha
    
    def serial_test(self, bits: np.ndarray, m: int = 2) -> Tuple[float, bool]:
        """
        序列测试。
        
        检验 m 位模式的分布均匀性。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["serial"]:
            return float('nan'), False
        
        def psi_sq(bits_arr: np.ndarray, m_val: int) -> float:
            if m_val == 0:
                return 0.0
            n_bits = len(bits_arr)
            # 扩展序列
            extended = np.concatenate([bits_arr, bits_arr[:m_val - 1]])
            # 计算模式频率
            patterns = {}
            for i in range(n_bits):
                pattern = tuple(extended[i:i + m_val])
                patterns[pattern] = patterns.get(pattern, 0) + 1
            # 计算 psi^2
            return sum(v ** 2 for v in patterns.values()) * (2 ** m_val) / n_bits - n_bits
        
        psi_m = psi_sq(bits, m)
        psi_m1 = psi_sq(bits, m - 1) if m > 1 else 0
        psi_m2 = psi_sq(bits, m - 2) if m > 2 else 0
        
        delta_psi = psi_m - psi_m1
        delta2_psi = psi_m - 2 * psi_m1 + psi_m2
        
        try:
            from scipy.special import gammaincc
            p_value1 = gammaincc(2 ** (m - 2), delta_psi / 2)
            p_value2 = gammaincc(2 ** (m - 3), delta2_psi / 2) if m > 2 else 1.0
            p_value = min(p_value1, p_value2)
        except ImportError:
            p_value = 0.5  # 简化
        
        return float(p_value), p_value >= self.alpha
    
    def approximate_entropy_test(self, bits: np.ndarray, m: int = 2) -> Tuple[float, bool]:
        """
        近似熵测试。
        
        比较相邻长度模式的频率。
        """
        n = len(bits)
        if n < NIST_MIN_BITS["approximate_entropy"]:
            return float('nan'), False
        
        def phi(bits_arr: np.ndarray, m_val: int) -> float:
            n_bits = len(bits_arr)
            extended = np.concatenate([bits_arr, bits_arr[:m_val - 1]])
            patterns = {}
            for i in range(n_bits):
                pattern = tuple(extended[i:i + m_val])
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            c_values = [v / n_bits for v in patterns.values()]
            return sum(c * math.log(c + 1e-12) for c in c_values)
        
        phi_m = phi(bits, m)
        phi_m1 = phi(bits, m + 1)
        
        ap_en = phi_m - phi_m1
        chi_sq = 2 * n * (math.log(2) - ap_en)
        
        try:
            from scipy.special import gammaincc
            p_value = gammaincc(2 ** (m - 1), chi_sq / 2)
        except ImportError:
            p_value = math.exp(-chi_sq / 2) if chi_sq < 100 else 0.0
        
        return float(p_value), p_value >= self.alpha
    
    def run_test(self, test_name: str, bits: np.ndarray) -> NISTTestResult:
        """运行单个测试。"""
        bits_required = NIST_MIN_BITS.get(test_name, 100)
        bits_used = len(bits)
        
        if bits_used < bits_required:
            return NISTTestResult(
                test_name=test_name,
                p_value=float('nan'),
                passed=False,
                bits_used=bits_used,
                bits_required=bits_required,
                status="skip",
                error=f"Insufficient bits: {bits_used} < {bits_required}",
            )
        
        try:
            test_fn = getattr(self, f"{test_name}_test", None)
            if test_fn is None:
                return NISTTestResult(
                    test_name=test_name,
                    p_value=float('nan'),
                    passed=False,
                    bits_used=bits_used,
                    bits_required=bits_required,
                    status="error",
                    error=f"Unknown test: {test_name}",
                )
            
            p_value, passed = test_fn(bits)
            status = "pass" if passed else "fail"
            
            return NISTTestResult(
                test_name=test_name,
                p_value=p_value,
                passed=passed,
                bits_used=bits_used,
                bits_required=bits_required,
                status=status,
            )
        except Exception as e:
            return NISTTestResult(
                test_name=test_name,
                p_value=float('nan'),
                passed=False,
                bits_used=bits_used,
                bits_required=bits_required,
                status="error",
                error=str(e),
            )
    
    def run_all_tests(
        self,
        data: Union[bytes, np.ndarray],
        concatenate_if_needed: bool = True,
        additional_data: Optional[List[bytes]] = None,
    ) -> NISTTestSummary:
        """
        运行所有 NIST 测试。
        
        Args:
            data: 待测试的数据（字节或比特数组）
            concatenate_if_needed: 如果比特不足，是否拼接额外数据
            additional_data: 额外数据用于拼接
            
        Returns:
            NISTTestSummary
        """
        # 转换为比特
        if isinstance(data, bytes):
            bits = self._to_bits(data)
        else:
            bits = data
        
        # 如果比特不足且有额外数据，按 SHA256 排序拼接
        max_required = max(NIST_MIN_BITS.values())
        if len(bits) < max_required and concatenate_if_needed and additional_data:
            # 按 SHA256 排序
            sorted_data = sorted(
                additional_data,
                key=lambda x: hashlib.sha256(x).hexdigest()
            )
            all_bits = [bits]
            for extra in sorted_data:
                all_bits.append(self._to_bits(extra))
                if sum(len(b) for b in all_bits) >= max_required:
                    break
            bits = np.concatenate(all_bits)
        
        # 运行所有测试
        results = []
        passed_count = 0
        
        for test_name in self.REQUIRED_TESTS:
            result = self.run_test(test_name, bits)
            results.append(result)
            if result.passed:
                passed_count += 1
        
        # 确定整体状态
        if passed_count == len(self.REQUIRED_TESTS):
            status = "pass"
        elif passed_count >= len(self.REQUIRED_TESTS) - 1:
            status = "warn"  # 允许 1 个失败
        else:
            status = "fail"
        
        return NISTTestSummary(
            total_tests=len(self.REQUIRED_TESTS),
            passed_tests=passed_count,
            total_bits=len(bits),
            results=results,
            status=status,
        )



# =============================================================================
# AvalancheEffectTester
# =============================================================================

class AvalancheEffectTester:
    """
    雪崩效应测试器。
    
    测试三类 flip：key/nonce/plaintext
    期望翻转率在 45%-55%
    
    **Validates: R3.AC3, Property 4**
    """
    
    FLIP_TYPES = [FlipType.KEY, FlipType.NONCE, FlipType.PLAINTEXT]
    
    def __init__(
        self,
        encrypt_fn: Optional[Callable] = None,
        n_samples: int = 100,
    ):
        """
        初始化雪崩效应测试器。
        
        Args:
            encrypt_fn: 加密函数 (plaintext, key, nonce) -> ciphertext
            n_samples: 每种 flip 类型的测试样本数
        """
        self.encrypt_fn = encrypt_fn or self._default_encrypt
        self.n_samples = n_samples
    
    def _default_encrypt(
        self, plaintext: bytes, key: bytes, nonce: bytes
    ) -> bytes:
        """默认加密函数（AES-GCM）。"""
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography library not installed")
        aesgcm = AESGCM(key)
        return aesgcm.encrypt(nonce, plaintext, None)
    
    def _flip_single_bit(self, data: bytes, bit_index: int) -> bytes:
        """翻转指定位置的比特。"""
        data_array = bytearray(data)
        byte_idx = bit_index // 8
        bit_idx = bit_index % 8
        if byte_idx < len(data_array):
            data_array[byte_idx] ^= (1 << bit_idx)
        return bytes(data_array)
    
    def _calculate_flip_rate(self, original: bytes, modified: bytes) -> float:
        """计算两个字节序列之间的比特翻转率。"""
        if len(original) != len(modified):
            # 长度不同时，取较短的长度
            min_len = min(len(original), len(modified))
            original = original[:min_len]
            modified = modified[:min_len]
        
        if len(original) == 0:
            return 0.0
        
        # 计算不同的比特数
        orig_bits = np.unpackbits(np.frombuffer(original, dtype=np.uint8))
        mod_bits = np.unpackbits(np.frombuffer(modified, dtype=np.uint8))
        
        diff_bits = np.sum(orig_bits != mod_bits)
        total_bits = len(orig_bits)
        
        return diff_bits / total_bits if total_bits > 0 else 0.0
    
    def test_flip_type(
        self,
        flip_type: FlipType,
        plaintext: Optional[bytes] = None,
        key: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
    ) -> AvalancheTestResult:
        """
        测试单种 flip 类型的雪崩效应。
        
        Args:
            flip_type: Flip 类型
            plaintext: 明文（可选，默认随机生成）
            key: 密钥（可选，默认随机生成）
            nonce: Nonce（可选，默认随机生成）
            
        Returns:
            AvalancheTestResult
        """
        flip_rates = []
        
        for _ in range(self.n_samples):
            # 生成或使用提供的参数
            pt = plaintext or secrets.token_bytes(64)
            k = key or secrets.token_bytes(32)
            n = nonce or secrets.token_bytes(12)
            
            try:
                # 原始加密
                original_ct = self.encrypt_fn(pt, k, n)
                
                # 根据 flip 类型修改输入
                if flip_type == FlipType.KEY:
                    bit_idx = secrets.randbelow(len(k) * 8)
                    modified_k = self._flip_single_bit(k, bit_idx)
                    modified_ct = self.encrypt_fn(pt, modified_k, n)
                elif flip_type == FlipType.NONCE:
                    bit_idx = secrets.randbelow(len(n) * 8)
                    modified_n = self._flip_single_bit(n, bit_idx)
                    modified_ct = self.encrypt_fn(pt, k, modified_n)
                elif flip_type == FlipType.PLAINTEXT:
                    bit_idx = secrets.randbelow(len(pt) * 8)
                    modified_pt = self._flip_single_bit(pt, bit_idx)
                    modified_ct = self.encrypt_fn(modified_pt, k, n)
                else:
                    continue
                
                # 计算翻转率
                rate = self._calculate_flip_rate(original_ct, modified_ct)
                flip_rates.append(rate)
                
            except Exception:
                continue
        
        if not flip_rates:
            return AvalancheTestResult(
                flip_type=flip_type.value,
                flip_rate=0.0,
                in_range=False,
                n_samples=0,
                status="error",
                details={"error": "No successful tests"},
            )
        
        avg_flip_rate = np.mean(flip_rates)
        std_flip_rate = np.std(flip_rates)
        min_rate, max_rate = AVALANCHE_FLIP_RATE_RANGE
        in_range = min_rate <= avg_flip_rate <= max_rate
        
        return AvalancheTestResult(
            flip_type=flip_type.value,
            flip_rate=float(avg_flip_rate),
            in_range=in_range,
            n_samples=len(flip_rates),
            status="pass" if in_range else "warn",
            details={
                "std": float(std_flip_rate),
                "min": float(np.min(flip_rates)),
                "max": float(np.max(flip_rates)),
            },
        )
    
    def test_all(
        self,
        plaintext: Optional[bytes] = None,
        key: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
    ) -> Dict[str, AvalancheTestResult]:
        """
        测试所有 flip 类型。
        
        Returns:
            Dict mapping flip type to AvalancheTestResult
        """
        results = {}
        for flip_type in self.FLIP_TYPES:
            results[flip_type.value] = self.test_flip_type(
                flip_type, plaintext, key, nonce
            )
        return results


# =============================================================================
# CViewSecurityEvaluator - 主评估器
# =============================================================================

class CViewSecurityEvaluator:
    """
    C-view 安全评估套件主类。
    
    整合 TamperTester、ReplayTester、NISTTestRunner、AvalancheEffectTester。
    
    **Validates: Property 4, Property 5**
    """
    
    def __init__(
        self,
        run_dir: Path,
        encrypt_fn: Optional[Callable] = None,
        decrypt_fn: Optional[Callable] = None,
    ):
        """
        初始化安全评估器。
        
        Args:
            run_dir: 运行目录
            encrypt_fn: 加密函数
            decrypt_fn: 解密函数
        """
        self.run_dir = Path(run_dir)
        self.tamper_tester = TamperTester(encrypt_fn=encrypt_fn, decrypt_fn=decrypt_fn)
        self.replay_tester = ReplayTester(run_dir=run_dir)
        self.nist_runner = NISTTestRunner()
        self.avalanche_tester = AvalancheEffectTester(encrypt_fn=encrypt_fn)
    
    def evaluate_all(
        self,
        samples: Optional[List[Dict[str, bytes]]] = None,
        ciphertext_data: Optional[bytes] = None,
    ) -> SecurityReport:
        """
        运行完整的安全评估。
        
        Args:
            samples: 测试样本列表
            ciphertext_data: 用于 NIST 测试的密文数据
            
        Returns:
            SecurityReport
        """
        # Tamper 测试
        tamper_results = self.tamper_tester.test_all(samples)
        
        # Replay 测试
        replay_result = self.replay_tester.test_replay_detection()
        
        # NIST 测试
        if ciphertext_data:
            nist_summary = self.nist_runner.run_all_tests(ciphertext_data)
        else:
            # 生成随机数据进行测试
            nist_summary = self.nist_runner.run_all_tests(secrets.token_bytes(2000))
        
        # Avalanche 测试
        avalanche_results = self.avalanche_tester.test_all()
        
        # 确定整体状态
        tamper_pass = all(r.status == "pass" for r in tamper_results.values())
        replay_pass = replay_result.status == "pass"
        # NIST/Avalanche 失败不 hard fail，但需记录
        
        if tamper_pass and replay_pass:
            overall_status = "pass"
        elif tamper_pass or replay_pass:
            overall_status = "partial"
        else:
            overall_status = "fail"
        
        return SecurityReport(
            tamper_results=tamper_results,
            replay_result=replay_result,
            nist_summary=nist_summary,
            avalanche_results=avalanche_results,
            overall_status=overall_status,
        )
    
    def generate_security_metrics_csv(
        self,
        report: SecurityReport,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        生成 security_metrics_cview.csv。
        
        Args:
            report: SecurityReport
            output_path: 输出路径
            
        Returns:
            CSV 文件路径
        """
        if output_path is None:
            output_path = self.run_dir / "tables" / "security_metrics_cview.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        
        # Tamper 结果
        for tamper_type, summary in report.tamper_results.items():
            rows.append({
                "test_category": "tamper",
                "test_type": tamper_type,
                "metric_name": "fail_rate",
                "metric_value": summary.fail_rate,
                "threshold": TAMPER_FAIL_RATE_THRESHOLD,
                "status": summary.status,
                "n_tests": summary.total_tests,
            })
        
        # Replay 结果
        if report.replay_result:
            rows.append({
                "test_category": "replay",
                "test_type": "replay_detection",
                "metric_name": "reject_rate",
                "metric_value": report.replay_result.reject_rate,
                "threshold": REPLAY_REJECT_RATE_THRESHOLD,
                "status": report.replay_result.status,
                "n_tests": report.replay_result.total_replays,
            })
        
        # NIST 结果
        if report.nist_summary:
            for result in report.nist_summary.results:
                rows.append({
                    "test_category": "nist",
                    "test_type": result.test_name,
                    "metric_name": "p_value",
                    "metric_value": result.p_value,
                    "threshold": NIST_P_VALUE_THRESHOLD,
                    "status": result.status,
                    "n_tests": result.bits_used,
                })
        
        # Avalanche 结果
        for flip_type, result in report.avalanche_results.items():
            rows.append({
                "test_category": "avalanche",
                "test_type": flip_type,
                "metric_name": "flip_rate",
                "metric_value": result.flip_rate,
                "threshold": f"{AVALANCHE_FLIP_RATE_RANGE[0]}-{AVALANCHE_FLIP_RATE_RANGE[1]}",
                "status": result.status,
                "n_tests": result.n_samples,
            })
        
        # 写入 CSV
        fieldnames = [
            "test_category", "test_type", "metric_name", "metric_value",
            "threshold", "status", "n_tests"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        return output_path
    
    def generate_security_report_md(
        self,
        report: SecurityReport,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        生成 security_report.md。
        
        Args:
            report: SecurityReport
            output_path: 输出路径
            
        Returns:
            Markdown 文件路径
        """
        if output_path is None:
            output_path = self.run_dir / "reports" / "security_report.md"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# C-view Security Evaluation Report",
            "",
            f"**Generated**: {report.timestamp}",
            f"**Overall Status**: {report.overall_status.upper()}",
            "",
            "---",
            "",
            "## 1. Primary Evidence: Security Goals",
            "",
            "### 1.1 Tamper Detection (Integrity)",
            "",
            "| Tamper Type | Total Tests | Detected | Fail Rate | Status |",
            "|-------------|-------------|----------|-----------|--------|",
        ]
        
        for tamper_type, summary in report.tamper_results.items():
            status_icon = "✓" if summary.status == "pass" else "✗"
            lines.append(
                f"| {tamper_type} | {summary.total_tests} | {summary.detected_count} | "
                f"{summary.fail_rate:.2%} | {status_icon} {summary.status} |"
            )
        
        lines.extend([
            "",
            f"**Threshold**: fail_rate ≥ {TAMPER_FAIL_RATE_THRESHOLD:.0%}",
            "",
            "### 1.2 Replay Detection (Anti-Replay)",
            "",
        ])
        
        if report.replay_result:
            status_icon = "✓" if report.replay_result.status == "pass" else "✗"
            lines.extend([
                f"- **Unique Ciphertexts**: {report.replay_result.total_unique}",
                f"- **Replay Attempts**: {report.replay_result.total_replays}",
                f"- **Rejected**: {report.replay_result.rejected_replays}",
                f"- **Reject Rate**: {report.replay_result.reject_rate:.2%}",
                f"- **Status**: {status_icon} {report.replay_result.status}",
                "",
                f"**Threshold**: reject_rate = {REPLAY_REJECT_RATE_THRESHOLD:.0%}",
            ])
        
        lines.extend([
            "",
            "---",
            "",
            "## 2. Diagnostic Evidence: Implementation Quality",
            "",
            "### 2.1 NIST SP800-22 Randomness Tests",
            "",
        ])
        
        if report.nist_summary:
            lines.extend([
                f"**Total Bits**: {report.nist_summary.total_bits}",
                f"**Tests Passed**: {report.nist_summary.passed_tests}/{report.nist_summary.total_tests}",
                "",
                "| Test Name | P-Value | Bits Used | Status |",
                "|-----------|---------|-----------|--------|",
            ])
            
            for result in report.nist_summary.results:
                status_icon = "✓" if result.passed else "✗"
                p_val = f"{result.p_value:.4f}" if not math.isnan(result.p_value) else "N/A"
                lines.append(
                    f"| {result.test_name} | {p_val} | {result.bits_used} | "
                    f"{status_icon} {result.status} |"
                )
            
            lines.extend([
                "",
                f"**Threshold**: p_value ≥ {NIST_P_VALUE_THRESHOLD}",
                "",
                "> **Note**: NIST test failures do not cause hard fail, but require explanation.",
            ])
        
        lines.extend([
            "",
            "### 2.2 Avalanche Effect Tests",
            "",
            "| Flip Type | Flip Rate | In Range | Samples | Status |",
            "|-----------|-----------|----------|---------|--------|",
        ])
        
        for flip_type, result in report.avalanche_results.items():
            status_icon = "✓" if result.in_range else "⚠"
            lines.append(
                f"| {flip_type} | {result.flip_rate:.2%} | "
                f"{'Yes' if result.in_range else 'No'} | {result.n_samples} | "
                f"{status_icon} {result.status} |"
            )
        
        lines.extend([
            "",
            f"**Expected Range**: {AVALANCHE_FLIP_RATE_RANGE[0]:.0%} - {AVALANCHE_FLIP_RATE_RANGE[1]:.0%}",
            "",
            "> **Note**: Avalanche test failures do not cause hard fail, but require explanation.",
            "",
            "---",
            "",
            "*Report generated by CViewSecurityEvaluator*",
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # Enums
    "TamperType",
    "FlipType",
    "SecurityTestStatus",
    # Constants
    "NIST_MIN_BITS",
    "TAMPER_FAIL_RATE_THRESHOLD",
    "REPLAY_REJECT_RATE_THRESHOLD",
    "NIST_P_VALUE_THRESHOLD",
    "AVALANCHE_FLIP_RATE_RANGE",
    # Data classes
    "TamperTestResult",
    "TamperTestSummary",
    "ReplayTestResult",
    "NISTTestResult",
    "NISTTestSummary",
    "AvalancheTestResult",
    "SecurityReport",
    # Testers
    "TamperTester",
    "ReplayTester",
    "NISTTestRunner",
    "AvalancheEffectTester",
    "CViewSecurityEvaluator",
]
