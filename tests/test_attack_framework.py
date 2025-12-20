# -*- coding: utf-8 -*-
"""
攻击框架测试

**Property 3: 攻击成功率映射一致性**
**Property 14: A2 攻击强制存在**
**Validates: Requirements 2.9, GC7, 16.1, 16.3, 16.5**
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_attack_type_enum():
    """测试 AttackType 枚举"""
    print("=" * 70)
    print("测试 AttackType 枚举")
    print("=" * 70)
    
    from src.evaluation.attack_framework import AttackType
    
    expected_types = [
        "face_verification",
        "attribute_inference",
        "reconstruction",
        "membership_inference",
        "property_inference",
    ]
    
    for attack_type in expected_types:
        assert hasattr(AttackType, attack_type.upper()), \
            f"AttackType 应包含 {attack_type}"
    
    print(f"✓ 5 类攻击类型已定义")
    for t in AttackType:
        print(f"  - {t.value}")
    print("✓ AttackType 枚举测试通过\n")


def test_threat_level_enum():
    """测试 ThreatLevel 枚举"""
    print("=" * 70)
    print("测试 ThreatLevel 枚举")
    print("=" * 70)
    
    from src.evaluation.attack_framework import ThreatLevel
    
    assert ThreatLevel.A0.value == "A0"
    assert ThreatLevel.A1.value == "A1"
    assert ThreatLevel.A2.value == "A2"
    
    print(f"✓ A0 (Black-box): {ThreatLevel.A0.value}")
    print(f"✓ A1 (Gray-box): {ThreatLevel.A1.value}")
    print(f"✓ A2 (White-box Adaptive): {ThreatLevel.A2.value}")
    print("✓ ThreatLevel 枚举测试通过\n")


def test_attacker_strength_enum():
    """测试 AttackerStrength 枚举"""
    print("=" * 70)
    print("测试 AttackerStrength 枚举")
    print("=" * 70)
    
    from src.evaluation.attack_framework import AttackerStrength
    
    assert AttackerStrength.LITE.value == "lite"
    assert AttackerStrength.FULL.value == "full"
    
    print(f"✓ LITE: {AttackerStrength.LITE.value}")
    print(f"✓ FULL: {AttackerStrength.FULL.value}")
    print("✓ AttackerStrength 枚举测试通过\n")


def test_attack_success_mapping():
    """测试攻击成功率映射表"""
    print("=" * 70)
    print("测试攻击成功率映射表")
    print("=" * 70)
    
    from src.evaluation.attack_framework import ATTACK_SUCCESS_MAPPING, AttackType
    
    # 验证所有攻击类型都有映射
    for attack_type in AttackType:
        assert attack_type in ATTACK_SUCCESS_MAPPING, \
            f"{attack_type} 缺少攻击成功率映射"
    
    # 验证映射内容（映射是字典，包含 metric 和 direction）
    expected_mappings = {
        AttackType.FACE_VERIFICATION: "TAR@FAR=1e-3",
        AttackType.ATTRIBUTE_INFERENCE: "AUC",
        AttackType.RECONSTRUCTION: "identity_similarity",
        AttackType.MEMBERSHIP_INFERENCE: "AUC",
        AttackType.PROPERTY_INFERENCE: "AUC",
    }
    
    for attack_type, expected_metric in expected_mappings.items():
        mapping = ATTACK_SUCCESS_MAPPING[attack_type]
        actual_metric = mapping["metric"] if isinstance(mapping, dict) else mapping
        assert actual_metric == expected_metric, \
            f"{attack_type} 映射错误: 期望 {expected_metric}, 实际 {actual_metric}"
        print(f"✓ {attack_type.value}: {actual_metric}")
    
    print("✓ 攻击成功率映射表测试通过\n")


def test_attack_fit_context():
    """测试 AttackFitContext 数据类"""
    print("=" * 70)
    print("测试 AttackFitContext 数据类")
    print("=" * 70)
    
    from src.evaluation.attack_framework import AttackFitContext, ThreatLevel
    
    context = AttackFitContext(
        run_id="test_run_001",
        dataset="celeba",
        task="classification",
        method="causal_vse_pc",
        training_mode="Z2Z",
        privacy_level=0.5,
        seed=42,
        threat_level=ThreatLevel.A1,
        attacker_visible=["z_view", "algorithm"]
    )
    
    assert context.run_id == "test_run_001"
    assert context.dataset == "celeba"
    assert context.threat_level == ThreatLevel.A1
    
    print(f"✓ run_id: {context.run_id}")
    print(f"✓ dataset: {context.dataset}")
    print(f"✓ threat_level: {context.threat_level}")
    print("✓ AttackFitContext 测试通过\n")


def test_attack_result():
    """测试 AttackResult 数据类"""
    print("=" * 70)
    print("测试 AttackResult 数据类")
    print("=" * 70)
    
    from src.evaluation.attack_framework import (
        AttackResult, AttackType, ThreatLevel, AttackerStrength
    )
    
    result = AttackResult(
        attack_type=AttackType.ATTRIBUTE_INFERENCE,
        threat_level=ThreatLevel.A1,
        attack_success=0.65,
        metric_name="AUC",
        metric_value=0.65,
        attacker_strength=AttackerStrength.FULL,
        additional_metrics={"accuracy": 0.60},
        degrade_reason=None
    )
    
    assert result.attack_type == AttackType.ATTRIBUTE_INFERENCE
    assert result.attack_success == 0.65
    assert result.attacker_strength == AttackerStrength.FULL
    
    print(f"✓ attack_type: {result.attack_type}")
    print(f"✓ attack_success: {result.attack_success}")
    print(f"✓ attacker_strength: {result.attacker_strength}")
    print("✓ AttackResult 测试通过\n")


def test_validate_a2_exists():
    """测试 A2 攻击存在性验证"""
    print("=" * 70)
    print("测试 A2 攻击存在性验证")
    print("=" * 70)
    
    from src.evaluation.attack_framework import (
        validate_a2_exists, AttackResult, AttackType, 
        ThreatLevel, AttackerStrength
    )
    
    # 包含 A2 的结果
    results_with_a2 = [
        AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            threat_level=ThreatLevel.A0,
            attack_success=0.7,
            metric_name="AUC",
            metric_value=0.7,
            attacker_strength=AttackerStrength.FULL,
        ),
        AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            threat_level=ThreatLevel.A2,
            attack_success=0.8,
            metric_name="AUC",
            metric_value=0.8,
            attacker_strength=AttackerStrength.FULL,
        ),
    ]
    
    # validate_a2_exists 返回 True 或抛出异常
    is_valid = validate_a2_exists(results_with_a2)
    assert is_valid, "包含 A2 的结果应通过验证"
    print(f"✓ 包含 A2 的结果验证通过")
    
    # 不包含 A2 的结果
    results_without_a2 = [
        AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            threat_level=ThreatLevel.A0,
            attack_success=0.7,
            metric_name="AUC",
            metric_value=0.7,
            attacker_strength=AttackerStrength.FULL,
        ),
        AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            threat_level=ThreatLevel.A1,
            attack_success=0.75,
            metric_name="AUC",
            metric_value=0.75,
            attacker_strength=AttackerStrength.FULL,
        ),
    ]
    
    try:
        validate_a2_exists(results_without_a2)
        assert False, "不包含 A2 的结果应抛出异常"
    except ValueError as e:
        print(f"✓ 不包含 A2 的结果验证失败: {e}")
    
    print("✓ A2 攻击存在性验证测试通过\n")


def test_property_3_attack_success_mapping_consistency():
    """
    **Property 3: 攻击成功率映射一致性**
    
    验证每个 attack_type 的 attack_success 字段按 GC7 映射计算
    """
    print("=" * 70)
    print("Property 3: 攻击成功率映射一致性")
    print("=" * 70)
    
    from src.evaluation.attack_framework import ATTACK_SUCCESS_MAPPING, AttackType
    
    # GC7 定义的映射
    gc7_mapping = {
        "face_verification": "TAR@FAR=1e-3",
        "attribute_inference": "AUC",
        "membership_inference": "AUC",
        "property_inference": "AUC",
        "reconstruction": "identity_similarity",
    }
    
    for attack_type in AttackType:
        expected = gc7_mapping[attack_type.value]
        mapping = ATTACK_SUCCESS_MAPPING[attack_type]
        actual = mapping["metric"] if isinstance(mapping, dict) else mapping
        assert actual == expected, \
            f"{attack_type.value} 映射不一致: 期望 {expected}, 实际 {actual}"
        print(f"✓ {attack_type.value}: {actual} (符合 GC7)")
    
    print("✓ Property 3 测试通过\n")


def test_property_14_a2_attack_mandatory():
    """
    **Property 14: A2 攻击强制存在**
    
    验证 attack_metrics.csv 必须包含 threat_level=A2
    """
    print("=" * 70)
    print("Property 14: A2 攻击强制存在")
    print("=" * 70)
    
    from src.evaluation.attack_framework import (
        validate_a2_exists, AttackResult, AttackType,
        ThreatLevel, AttackerStrength
    )
    
    # 模拟完整的攻击结果（包含 A2）
    complete_results = []
    for threat_level in [ThreatLevel.A0, ThreatLevel.A1, ThreatLevel.A2]:
        complete_results.append(AttackResult(
            attack_type=AttackType.RECONSTRUCTION,
            threat_level=threat_level,
            attack_success=0.5 + 0.1 * list(ThreatLevel).index(threat_level),
            metric_name="identity_similarity",
            metric_value=0.5 + 0.1 * list(ThreatLevel).index(threat_level),
            attacker_strength=AttackerStrength.FULL,
        ))
    
    is_valid = validate_a2_exists(complete_results)
    assert is_valid, "完整结果应包含 A2"
    
    print(f"✓ 完整结果包含 A2 验证通过")
    print("✓ Property 14 测试通过\n")


def test_attack_registry():
    """测试攻击注册表"""
    print("=" * 70)
    print("测试攻击注册表")
    print("=" * 70)
    
    from src.evaluation.attack_framework import AttackRegistry, AttackType
    
    # 导入攻击实现以触发注册
    try:
        from src.evaluation.attacks import (
            FaceVerificationAttack,
            AttributeInferenceAttack,
            ReconstructionAttack,
            MembershipInferenceAttack,
            PropertyInferenceAttack,
        )
    except ImportError:
        pass
    
    # 验证可以获取已注册的攻击类型
    attack_types = AttackRegistry.list_attacks()
    print(f"✓ 注册的攻击类型: {len(attack_types)}")
    for t in attack_types:
        print(f"  - {t.value}")
    
    # 验证 AttackType 枚举有 5 种类型
    all_types = list(AttackType)
    assert len(all_types) == 5, f"应有 5 类攻击类型，实际 {len(all_types)}"
    
    print("✓ 攻击注册表测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("攻击框架测试")
    print("=" * 70 + "\n")
    
    test_attack_type_enum()
    test_threat_level_enum()
    test_attacker_strength_enum()
    test_attack_success_mapping()
    test_attack_fit_context()
    test_attack_result()
    test_validate_a2_exists()
    test_property_3_attack_success_mapping_consistency()
    test_property_14_a2_attack_mandatory()
    test_attack_registry()
    
    print("=" * 70)
    print("✓ 所有攻击框架测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
