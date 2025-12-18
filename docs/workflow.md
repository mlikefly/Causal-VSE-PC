# Causal-VSE-PC 工作流程

**文档版本**: 2.1.1  
**更新日期**: 2024-12-18  
**核心创新**: 双视图架构 + 因果隐私预算分配

---

## 一、完整工作流程

### 1.1 系统流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Causal-VSE-PC Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 1. 输入图像  │───▶│ 2. 语义分析  │───▶│ 3. 因果预算  │                   │
│  │    Image     │    │ SemanticMask │    │ CausalBudget │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 6. 攻击评估  │◀───│ 5. 训练/推理 │◀───│ 4. 双视图加密│                   │
│  │ 5类+A2      │    │ P2P/Z2Z/Mix2Z│    │ Z-view+C-view│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                                       │                            │
│         ▼                                       ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 7. 因果效应  │    │ 8. 安全评估  │    │ 9. 结果验证  │                   │
│  │ ATE/CATE    │    │ NIST/Tamper  │    │ ValidateRun  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│                              ┌──────────────────────────────┐               │
│                              │ 10. 输出: 8张主图 + CSV表格  │               │
│                              └──────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、各阶段详细说明

### 2.1 阶段1-3: 数据准备

```python
# 1. 输入图像
image = load_image("path/to/image.jpg")  # [B, C, H, W]

# 2. 语义分析
from src.data.semantic_mask_generator import SemanticMaskGenerator
mask_gen = SemanticMaskGenerator()
semantic_mask = mask_gen.generate(image)  # {face, background, sensitive_attr}

# 3. 因果预算分配
from src.data.causal_budget_allocator import CausalBudgetAllocator
allocator = CausalBudgetAllocator()
privacy_map = allocator.allocate(
    semantic_mask, 
    privacy_level=0.5,
    task_type="classification"
)
```

### 2.2 阶段4: 双视图加密

```python
from src.cipher.dual_view_engine import DualViewEngine

# 初始化加密引擎
engine = DualViewEngine(master_key=key)

# 生成双视图
z_view, c_view, enc_info = engine.encrypt(
    image,
    privacy_map=privacy_map,
    privacy_level=0.5
)

# Z-view: 用于ML推理（保留语义）
# C-view: 用于安全存储（AEAD封装）
```

### 2.3 阶段5: 训练/推理

```python
from src.training.training_mode_manager import TrainingModeManager

# 训练模式管理
mode_manager = TrainingModeManager()

# 4种训练模式
# P2P: Plaintext → Plaintext (基线)
# P2Z: Plaintext → Z-view (域迁移)
# Z2Z: Z-view → Z-view (完全加密)
# Mix2Z: 50% Plaintext + 50% Z-view → Z-view (混合)

dataloader = mode_manager.get_dataloader(
    mode="Z2Z",
    split="train"
)
```

### 2.4 阶段6: 攻击评估

```python
from src.evaluation.attack_framework import AttackType, ThreatLevel
from src.evaluation.attacks import (
    FaceVerificationAttack,
    AttributeInferenceAttack,
    ReconstructionAttack,
    MembershipInferenceAttack,
    PropertyInferenceAttack,
    AdaptiveAttacker
)

# 5类攻击 + 3级威胁
attacks = [
    FaceVerificationAttack(),      # TAR@FAR=1e-3
    AttributeInferenceAttack(),    # AUC
    ReconstructionAttack(),        # identity_similarity
    MembershipInferenceAttack(),   # AUC (Shadow Models)
    PropertyInferenceAttack(),     # AUC
]

# A2自适应攻击（最强威胁）
adaptive = AdaptiveAttacker()
adaptive.design_adaptive_strategy(algorithm_info, mask_gen, allocator)
```

### 2.5 阶段7: 因果效应估计

```python
from src.evaluation.causal_effects import CausalEffectEstimator

estimator = CausalEffectEstimator()

# ATE: 平均处理效应
ate = estimator.estimate_ate(intervention_results)
# ATE = E[A|do(β=1)] - E[A|do(β=0)]

# CATE: 条件平均处理效应
cate = estimator.estimate_cate(intervention_results, condition="face")
# CATE = E[A|do(β=1), X=face] - E[A|do(β=0), X=face]

# 最优预算分配
optimal = estimator.solve_optimal_allocation(
    ate, 
    utility_threshold=0.65
)
```

### 2.6 阶段8: 安全评估

```python
from src.evaluation.cview_security import CViewSecurityEvaluator

evaluator = CViewSecurityEvaluator()

# 主证据: 安全目标验证
tamper_results = evaluator.test_tamper(c_view)      # fail_rate ≥ 99%
replay_results = evaluator.test_replay(c_view)      # reject_rate = 100%

# 诊断证据: 实现质量
nist_results = evaluator.run_nist_tests(c_view)     # p_value ≥ 0.01
avalanche_results = evaluator.test_avalanche(c_view) # flip_rate ∈ [0.45, 0.55]
```

### 2.7 阶段9: 结果验证

```python
from src.protocol.validate_run import ValidateRun

validator = ValidateRun(run_dir)

# R1-R10 红线检查
report = validator.validate_all()

# 红线检查项:
# R1: protocol_version == schema_version
# R2: coverage ≥ 98%
# R3: A2 存在且 attacker_strength=full
# R4: replay reject_rate = 100%
# R5: tamper fail_rate ≥ 99%
# R6: c_view guard 无泄漏
# R7: figure_manifest SHA256 可复现
# R8: nonce 无重用
# R9: train/val/test 零 ID 重叠
# R10: 所有 CSV 字段完整
```

### 2.8 阶段10: 输出生成

```python
from src.evaluation.figure_generator import FigureGenerator

generator = FigureGenerator(run_dir)

# 生成8张主图
generator.generate_all_figures()

# 主图列表:
# fig_utility_curve.png      - 效用随privacy_level变化
# fig_attack_curves.png      - 五类攻击曲线+CI
# fig_pareto_frontier.png    - 隐私-效用Pareto前沿
# fig_causal_ate_cate.png    - ATE/CATE + CI
# fig_cview_security_summary.png - C-view安全指标汇总
# fig_ablation_summary.png   - 消融实验对比
# fig_efficiency.png         - 效率对比
# fig_robustness.png         - 鲁棒性测试结果
```

---

## 三、训练模式对比

| 模式 | 训练数据 | 测试数据 | 说明 |
|------|----------|----------|------|
| P2P | Plaintext | Plaintext | 基线（无隐私保护） |
| P2Z | Plaintext | Z-view | 域迁移测试 |
| Z2Z | Z-view | Z-view | 完全加密训练 |
| Mix2Z | 50% P + 50% Z | Z-view | 混合训练 |

**效用门槛**:
- privacy_level=0.3 → 75% P2P
- privacy_level=0.5 → 65% P2P
- privacy_level=0.7 → 55% P2P

---

## 四、攻击评估矩阵

| 攻击类型 | attack_success映射 | 方向 |
|----------|-------------------|------|
| face_verification | TAR@FAR=1e-3 | ↓ 越低越好 |
| attribute_inference | AUC | ↓ 越低越好 |
| reconstruction | identity_similarity | ↓ 越低越好 |
| membership_inference | AUC | ↓ 越低越好 |
| property_inference | AUC | ↓ 越低越好 |

**归一化公式**:
- `privacy_protection = 1 - normalized(attack_success)`
- 范围: [0, 1]，0=完全保护，1=无保护

---

## 五、CI集成

### 5.1 smoke_test模式

```bash
# 时间预算 < 20 min
python scripts/run_benchmark.py --smoke_test

# 配置:
# - attacker_strength=lite
# - 5 epochs, 1 实例化, 子集数据
# - 用于管线健康检查
```

### 5.2 full模式

```bash
# 完整实验
python scripts/run_benchmark.py --full

# 配置:
# - attacker_strength=full
# - 100 epochs, 全实例化, 全数据
# - 用于主证据生成
```

---

## 六、输出目录结构

```
results/{exp_name}/{run_id}/
├── meta/
│   ├── config.yaml
│   ├── protocol_version.txt
│   ├── nonce_log.json
│   ├── replay_cache.json
│   └── hardware.json
├── tables/
│   ├── utility_metrics.csv
│   ├── attack_metrics.csv
│   ├── causal_effects.csv
│   ├── security_metrics_cview.csv
│   ├── ablation.csv
│   ├── efficiency.csv
│   ├── robustness_metrics.csv
│   └── baseline_comparison.csv
├── figures/
│   ├── fig_utility_curve.png
│   ├── fig_attack_curves.png
│   ├── fig_pareto_frontier.png
│   ├── fig_causal_ate_cate.png
│   ├── fig_cview_security_summary.png
│   ├── fig_ablation_summary.png
│   ├── fig_efficiency.png
│   └── fig_robustness.png
├── logs/
│   ├── stdout.log
│   └── errors.log
└── reports/
    ├── validate_run_onepage.md
    ├── figure_manifest.json
    └── protocol_snapshot.md
```

---

## 七、实现状态

### 已完成模块 ✅

| 阶段 | 模块 | 状态 |
|------|------|------|
| 数据准备 | SemanticMaskGenerator | ✅ 完成 |
| 数据准备 | CausalBudgetAllocator | ✅ 完成 |
| 双视图加密 | DualViewEngine | ✅ 完成 |
| 训练/推理 | TrainingModeManager | ✅ 完成 |
| 攻击评估 | 5类攻击 + A2自适应 | ✅ 完成 |
| 因果效应 | CausalEffectEstimator | ✅ 完成 |
| 安全评估 | CViewSecurityEvaluator | ✅ 完成 |
| 结果验证 | ValidateRun | ✅ 完成 |
| 输出生成 | FigureGenerator | ✅ 完成 |

---

## 八、相关文档

- [项目总览](project_overview.md)
- [开发日志](development_log.md)
- [目标与指标](goals_and_metrics.md)
- [设计文档](../.kiro/specs/top-journal-experiment-suite/design.md)
