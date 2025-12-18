# Causal-VSE-PC 源代码结构

> **版本**: 2.1.1 | **最后更新**: 2024-12

## 目录说明

```
src/
├── cipher/                   # 加密器模块
│   ├── scne_cipher.py        # SCNE主加密器（统一接口，支持双视图）
│   └── dual_view_engine.py   # 双视图加密引擎（Z-view + C-view）
│
├── core/                     # 核心算法
│   ├── chaotic_encryptor.py  # Layer 1: 混沌加密（Arnold + 5D超混沌）
│   ├── frequency_cipher.py   # Layer 2: 频域加密（FFT/DWT）
│   ├── chaos_systems.py      # 混沌系统实现
│   ├── nonce_manager.py      # Nonce管理器（确定性派生 + 唯一性检查）
│   └── replay_cache.py       # 重放检测缓存（C-view安全）
│
├── crypto/                   # 密码学组件
│   └── key_system.py         # 分层密钥系统（KDF + PRF + AEAD）
│
├── data/                     # 数据流水线模块
│   ├── manifest_builder.py   # Manifest构建器（四子集扫描 + 泄漏检查）
│   ├── semantic_mask_generator.py  # 语义掩码生成器
│   └── causal_budget_allocator.py  # 因果隐私预算分配器
│
├── protocol/                 # 协议与验证模块
│   ├── protocol_manager.py   # 协议版本管理（v2.1.1）
│   ├── results_schema.py     # 结果Schema定义（冻结字段）
│   └── validate_run.py       # 运行验证器（R1-R10红线检查）
│
├── training/                 # 训练模块
│   └── training_mode_manager.py  # 训练模式管理（P2P/P2Z/Z2Z/Mix2Z）
│
├── evaluation/               # 评估模块（✅ 全部完成）
│   ├── attack_framework.py   # 攻击框架（5类攻击 + A0/A1/A2威胁等级）
│   ├── attack_normalizer.py  # 攻击成功率归一化
│   ├── attack_evaluator.py   # 攻击评估器
│   ├── attacks/              # 五类攻击实现
│   │   ├── __init__.py       # 攻击模块初始化
│   │   ├── face_verification.py    # 人脸验证攻击
│   │   ├── attribute_inference.py  # 属性推断攻击
│   │   ├── reconstruction.py       # 重建攻击
│   │   ├── membership_inference.py # 成员推断攻击
│   │   ├── property_inference.py   # 属性推断攻击
│   │   └── adaptive_attacker.py    # A2自适应攻击者
│   ├── utility_evaluator.py  # 效用评估器（门槛检查）
│   ├── dual_view_metrics.py  # 双视图指标
│   ├── cview_security.py     # C-view安全评估（NIST/Avalanche/Tamper）
│   ├── security_metrics.py   # 安全指标（熵、NPCR、UACI等）
│   ├── causal_effects.py     # 因果效应估计（ATE/CATE）
│   ├── baseline_comparator.py    # 基线对比器（5个基线）
│   ├── baseline_matrix.py    # 基线矩阵
│   ├── statistics_engine.py  # 统计引擎（Bootstrap CI + BH-FDR）
│   ├── figure_generator.py   # 图表生成器（8张主图）
│   ├── ablation_runner.py    # 消融实验运行器（12项）
│   ├── robustness_efficiency.py  # 鲁棒性与效率评估
│   └── ci_integration.py     # CI集成（smoke_test + reproduce）
│
├── vse_pc/                   # 因果推断模块
│   ├── causal_analysis.py    # ATE/CATE因果效应分析
│   ├── privacy_budget.py     # 自适应隐私预算分配
│   ├── pipeline.py           # 完整处理流水线
│   ├── ciphertext_ml.py      # 密文域机器学习
│   └── verifiable.py         # 可验证性组件
│
├── neural/                   # 神经网络
│   └── unet.py               # U-Net显著性检测器
│
├── baselines/                # 基线方法
│   └── crypto_baselines.py   # 传统加密基线
│
├── utils/                    # 工具函数
│   ├── datasets.py           # 数据集加载
│   └── experiment_tracker.py # 实验追踪器
│
├── plotting/                 # 可视化
│   └── plot_style.py         # 绘图样式
│
└── weights/                  # 模型权重
    ├── checkpoints/          # 训练检查点
    └── pretrained/           # 预训练权重
```

## 核心模块说明

### 1. 双视图加密架构

本项目实现了 **Dual-View Privacy Encryption**：

- **Z-view (Utility View)**: 可用密文，用于密文域ML推理（Layer 1 + Layer 2）
- **C-view (Cryptographic View)**: 强加密密文，用于存储/传输（Z-view + AEAD封装）

```python
from src.cipher.dual_view_engine import DualViewEngine

engine = DualViewEngine(master_key=key)
z_view, c_view = engine.encrypt(image, privacy_level=0.5)
```

### 2. 协议与验证

协议版本管理和结果验证：

```python
from src.protocol.protocol_manager import ProtocolManager
from src.protocol.validate_run import ValidateRun

# 协议版本
pm = ProtocolManager()
pm.write_protocol_version(run_dir)

# 验证运行
validator = ValidateRun(run_dir)
report = validator.validate_all()  # R1-R10红线检查
```

### 3. 攻击评估框架

支持5类攻击 + 3级威胁模型：

```python
from src.evaluation.attack_framework import AttackBase, ThreatLevel, AttackType

# 威胁等级: A0(黑盒), A1(灰盒), A2(白盒自适应)
# 攻击类型: face_verification, attribute_inference, reconstruction,
#          membership_inference, property_inference
```

### 4. 因果效应估计

基于因果推断的隐私预算分配：

```python
from src.evaluation.causal_effects import CausalEffectEstimator

estimator = CausalEffectEstimator()
ate = estimator.estimate_ate(intervention_results)
optimal_budget = estimator.solve_optimal_allocation(ate, utility_threshold=0.65)
```

### 5. 统计引擎

Bootstrap CI + BH-FDR多重比较校正：

```python
from src.evaluation.statistics_engine import StatisticsEngine

engine = StatisticsEngine(n_boot=1000, alpha=0.05)
ci_low, ci_high = engine.compute_ci(values)
p_adj = engine.multiple_comparison_correction(p_values, method='bh')
```

## 安全边界声明

```
Security Boundary Declaration:
1. C-view security inherits from standard AEAD (AES-GCM/ChaCha20-Poly1305), 
   providing IND-CPA and IND-CCA guarantees as per the underlying algorithm.
2. The chaotic/frequency domain transformations serve as confusion/diffusion 
   layers and do NOT independently claim semantic security.
3. Z-view privacy is empirically demonstrated through attack success rate 
   reduction, not through cryptographic proofs.
4. This system does not defend against: side-channel attacks, physical attacks,
   or attacks with access to the encryption key.
```

## 测试

运行所有测试：

```bash
pytest tests/ -v
```

运行特定测试：

```bash
pytest tests/test_encryption.py -v
pytest tests/test_attack_evaluator.py -v
pytest tests/test_replay_cache.py -v
```

## 相关文档

- [实验协议](../docs/EXPERIMENT_PROTOCOL.md)
- [结果Schema](../docs/RESULTS_SCHEMA.md)
- [数据集说明](../docs/DATASETS.md)
- [设计文档](../.kiro/specs/top-journal-experiment-suite/design.md)
