# Causal-VSE-PC 源代码结构

## 目录说明

```
src/
├── cipher/                   # 加密器模块
│   └── scne_cipher.py        # SCNE主加密器（统一接口，支持双视图）
│
├── core/                     # 核心算法
│   ├── chaotic_encryptor.py  # Layer 1: 混沌加密（Arnold + 5D超混沌）
│   ├── frequency_cipher.py   # Layer 2: 频域加密（FFT/DWT）
│   └── chaos_systems.py      # 混沌系统实现
│
├── crypto/                   # 密码学组件
│   └── key_system.py         # 分层密钥系统（KDF + PRF + AEAD）
│
├── data/                     # 数据流水线模块 [NEW]
│   ├── manifest_builder.py   # Manifest构建器（四子集扫描）
│   ├── semantic_mask_generator.py  # 语义掩码生成器 [TODO]
│   └── causal_budget_allocator.py  # 因果隐私预算分配器 [TODO]
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
├── evaluation/               # 评估模块
│   ├── security_metrics.py   # 安全指标（熵、NPCR、UACI等）
│   ├── attack_models.py      # 攻击模型
│   └── strong_recognizers.py # 强识别器
│
├── baselines/                # 基线方法
│   └── crypto_baselines.py   # 传统加密基线
│
├── utils/                    # 工具函数
│   └── datasets.py           # 数据集加载
│
├── plotting/                 # 可视化
│   └── plot_style.py         # 绘图样式
│
└── weights/                  # 模型权重
    ├── checkpoints/          # 训练检查点
    └── pretrained/           # 预训练权重
```

## 核心模块

### SCNECipher (src/cipher/scne_cipher.py)

主加密器，整合三层加密：
- Layer 1: 空域混沌置乱
- Layer 2: 频域语义控制
- Layer 3: 字节级流加密

### CausalPrivacyAnalyzer (src/vse_pc/causal_analysis.py)

因果推断分析器：
- 计算ATE（平均处理效应）
- 计算CATE（条件平均处理效应）
- 提供隐私预算建议

### SecurityMetrics (src/evaluation/security_metrics.py)

安全指标评估：
- 信息熵
- NPCR/UACI
- 相邻像素相关性
- Chi-square检验
- NIST随机性测试

### ManifestBuilder (src/data/manifest_builder.py) [NEW]

统一数据清单生成器：
- 支持四子集扫描：CelebA-HQ、FairFace、OpenImages-PII
- 生成 manifest.jsonl 格式
- 包含 sample_id、labels、mask_path 等必需字段
- 支持公平性分组信息（race/gender/age）

## 双视图加密架构

本项目实现了 **Dual-View Privacy Encryption**：

- **Z-view**: 可用密文，用于密文域ML推理（Layer 1 + Layer 2）
- **C-view**: 强加密密文，用于存储/传输（Z-view + AEAD封装）

### 确定性加密

支持确定性 nonce 派生，绑定：
- image_id + task_type + privacy_map_hash + z_view_hash

确保相同输入产生相同输出，同时避免 nonce 复用风险。

### AEAD AAD 绑定

C-view 使用 ChaCha20-Poly1305 AEAD，AAD 格式：
```
v2|sample_id|dataset|split|task_type
```

确保"密文搬家/改字段即解密失败"。
