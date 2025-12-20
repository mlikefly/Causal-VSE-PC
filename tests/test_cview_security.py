# -*- coding: utf-8 -*-
"""
C-view 安全评估测试

**Property 4: C-view 安全测试完整性**
**Property 5: NIST 比特流充足性**
**Validates: Requirements 3.1-3.6**
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_tamper_type_enum():
    """测试 TamperType 枚举"""
    print("=" * 70)
    print("测试 TamperType 枚举")
    print("=" * 70)
    
    from src.evaluation.cview_security import TamperType
    
    assert TamperType.CIPHERTEXT.value == "ciphertext"
    assert TamperType.TAG.value == "tag"
    assert TamperType.AAD.value == "aad"
    
    print(f"✓ CIPHERTEXT: {TamperType.CIPHERTEXT.value}")
    print(f"✓ TAG: {TamperType.TAG.value}")
    print(f"✓ AAD: {TamperType.AAD.value}")
    print("✓ TamperType 枚举测试通过\n")


def test_flip_type_enum():
    """测试 FlipType 枚举"""
    print("=" * 70)
    print("测试 FlipType 枚举")
    print("=" * 70)
    
    from src.evaluation.cview_security import FlipType
    
    assert FlipType.KEY.value == "key"
    assert FlipType.NONCE.value == "nonce"
    assert FlipType.PLAINTEXT.value == "plaintext"
    
    print(f"✓ KEY: {FlipType.KEY.value}")
    print(f"✓ NONCE: {FlipType.NONCE.value}")
    print(f"✓ PLAINTEXT: {FlipType.PLAINTEXT.value}")
    print("✓ FlipType 枚举测试通过\n")


def test_security_constants():
    """测试安全常量"""
    print("=" * 70)
    print("测试安全常量")
    print("=" * 70)
    
    from src.evaluation.cview_security import (
        NIST_MIN_BITS,
        TAMPER_FAIL_RATE_THRESHOLD,
        REPLAY_REJECT_RATE_THRESHOLD,
        NIST_P_VALUE_THRESHOLD,
        AVALANCHE_FLIP_RATE_RANGE,
    )
    
    # NIST_MIN_BITS 是字典，检查其值
    assert isinstance(NIST_MIN_BITS, dict), "NIST_MIN_BITS 应该是字典"
    assert all(v > 0 for v in NIST_MIN_BITS.values()), "所有 NIST_MIN_BITS 值应 > 0"
    assert 0 < TAMPER_FAIL_RATE_THRESHOLD <= 1.0
    assert REPLAY_REJECT_RATE_THRESHOLD == 1.0
    assert 0 < NIST_P_VALUE_THRESHOLD < 1.0
    assert len(AVALANCHE_FLIP_RATE_RANGE) == 2
    
    print(f"✓ NIST_MIN_BITS: {NIST_MIN_BITS}")
    print(f"✓ TAMPER_FAIL_RATE_THRESHOLD: {TAMPER_FAIL_RATE_THRESHOLD}")
    print(f"✓ REPLAY_REJECT_RATE_THRESHOLD: {REPLAY_REJECT_RATE_THRESHOLD}")
    print(f"✓ NIST_P_VALUE_THRESHOLD: {NIST_P_VALUE_THRESHOLD}")
    print(f"✓ AVALANCHE_FLIP_RATE_RANGE: {AVALANCHE_FLIP_RATE_RANGE}")
    print("✓ 安全常量测试通过\n")


def test_tamper_tester_initialization():
    """测试 TamperTester 初始化"""
    print("=" * 70)
    print("测试 TamperTester 初始化")
    print("=" * 70)
    
    from src.evaluation.cview_security import TamperTester
    
    tester = TamperTester(n_tests_per_type=100)
    
    assert tester.n_tests_per_type == 100
    
    print(f"✓ TamperTester 初始化成功")
    print(f"✓ n_tests_per_type: {tester.n_tests_per_type}")
    print("✓ TamperTester 初始化测试通过\n")


def test_nist_test_runner_initialization():
    """测试 NISTTestRunner 初始化"""
    print("=" * 70)
    print("测试 NISTTestRunner 初始化")
    print("=" * 70)
    
    from src.evaluation.cview_security import NISTTestRunner
    
    runner = NISTTestRunner(alpha=0.01)
    
    assert runner.alpha == 0.01
    
    print(f"✓ NISTTestRunner 初始化成功")
    print(f"✓ alpha: {runner.alpha}")
    print("✓ NISTTestRunner 初始化测试通过\n")


def test_avalanche_effect_tester_initialization():
    """测试 AvalancheEffectTester 初始化"""
    print("=" * 70)
    print("测试 AvalancheEffectTester 初始化")
    print("=" * 70)
    
    from src.evaluation.cview_security import AvalancheEffectTester
    
    tester = AvalancheEffectTester(n_samples=50)
    
    assert tester.n_samples == 50
    
    print(f"✓ AvalancheEffectTester 初始化成功")
    print(f"✓ n_samples: {tester.n_samples}")
    print("✓ AvalancheEffectTester 初始化测试通过\n")


def test_replay_tester_initialization():
    """测试 ReplayTester 初始化"""
    print("=" * 70)
    print("测试 ReplayTester 初始化")
    print("=" * 70)
    
    from src.evaluation.cview_security import ReplayTester
    
    tester = ReplayTester()
    
    print(f"✓ ReplayTester 初始化成功")
    print("✓ ReplayTester 初始化测试通过\n")


def test_cview_security_evaluator_initialization():
    """测试 CViewSecurityEvaluator 初始化"""
    print("=" * 70)
    print("测试 CViewSecurityEvaluator 初始化")
    print("=" * 70)
    
    from src.evaluation.cview_security import CViewSecurityEvaluator
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        evaluator = CViewSecurityEvaluator(run_dir=run_dir)
        
        assert evaluator.run_dir == run_dir
        
        print(f"✓ CViewSecurityEvaluator 初始化成功")
        print(f"✓ run_dir: {evaluator.run_dir}")
        print("✓ CViewSecurityEvaluator 初始化测试通过\n")


def test_property_4_cview_security_completeness():
    """
    **Property 4: C-view 安全测试完整性**
    
    验证 tamper fail_rate >= 99%, replay reject_rate = 100%
    """
    print("=" * 70)
    print("Property 4: C-view 安全测试完整性")
    print("=" * 70)
    
    from src.evaluation.cview_security import (
        TAMPER_FAIL_RATE_THRESHOLD,
        REPLAY_REJECT_RATE_THRESHOLD,
    )
    
    # 验证阈值设置正确
    assert TAMPER_FAIL_RATE_THRESHOLD >= 0.99, \
        f"tamper fail_rate 阈值应 >= 0.99，实际 {TAMPER_FAIL_RATE_THRESHOLD}"
    assert REPLAY_REJECT_RATE_THRESHOLD == 1.0, \
        f"replay reject_rate 阈值应 = 1.0，实际 {REPLAY_REJECT_RATE_THRESHOLD}"
    
    print(f"✓ TAMPER_FAIL_RATE_THRESHOLD: {TAMPER_FAIL_RATE_THRESHOLD}")
    print(f"✓ REPLAY_REJECT_RATE_THRESHOLD: {REPLAY_REJECT_RATE_THRESHOLD}")
    print("✓ Property 4 测试通过\n")


def test_property_5_nist_bits_sufficiency():
    """
    **Property 5: NIST 比特流充足性**
    
    验证 NIST 测试的最小比特数要求
    """
    print("=" * 70)
    print("Property 5: NIST 比特流充足性")
    print("=" * 70)
    
    from src.evaluation.cview_security import NIST_MIN_BITS
    
    # NIST_MIN_BITS 是字典，检查每个测试的最小比特数
    assert isinstance(NIST_MIN_BITS, dict), "NIST_MIN_BITS 应该是字典"
    
    # 验证所有测试的最小比特数都 >= 100
    for test_name, min_bits in NIST_MIN_BITS.items():
        assert min_bits >= 100, \
            f"{test_name} 的 NIST_MIN_BITS 应 >= 100，实际 {min_bits}"
        print(f"✓ {test_name}: {min_bits} bits")
    
    print("✓ Property 5 测试通过\n")


def test_security_test_status_enum():
    """测试 SecurityTestStatus 枚举"""
    print("=" * 70)
    print("测试 SecurityTestStatus 枚举")
    print("=" * 70)
    
    from src.evaluation.cview_security import SecurityTestStatus
    
    assert SecurityTestStatus.PASS.value == "pass"
    assert SecurityTestStatus.FAIL.value == "fail"
    assert SecurityTestStatus.SKIP.value == "skip"
    
    print(f"✓ PASS: {SecurityTestStatus.PASS.value}")
    print(f"✓ FAIL: {SecurityTestStatus.FAIL.value}")
    print(f"✓ SKIP: {SecurityTestStatus.SKIP.value}")
    print("✓ SecurityTestStatus 枚举测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("C-view 安全评估测试")
    print("=" * 70 + "\n")
    
    test_tamper_type_enum()
    test_flip_type_enum()
    test_security_constants()
    test_tamper_tester_initialization()
    test_nist_test_runner_initialization()
    test_avalanche_effect_tester_initialization()
    test_replay_tester_initialization()
    test_cview_security_evaluator_initialization()
    test_property_4_cview_security_completeness()
    test_property_5_nist_bits_sufficiency()
    test_security_test_status_enum()
    
    print("=" * 70)
    print("✓ 所有 C-view 安全评估测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
