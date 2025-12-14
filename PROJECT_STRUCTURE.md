# 项目结构说明 (Project Structure)

## 📁 根目录文件

```
├── main.py              # 主入口文件
├── README.md            # 项目说明文档
├── requirements.txt     # Python 依赖列表
├── .env.example         # 环境变量示例
├── .gitignore           # Git 忽略配置
└── PROJECT_STRUCTURE.md # 本文件 - 项目结构说明
```

---

## 📁 configs/ - 配置文件

```
configs/
├── default.yaml    # 默认配置
├── basic.yaml      # 基础配置
├── advanced.yaml   # 高级配置
└── debug.yaml      # 调试配置
```

---

## 📁 src/ - 源代码

```
src/
├── __init__.py
├── README.md                    # 源码说明
│
├── core/                        # 🔐 核心加密模块
│   ├── __init__.py
│   ├── chaos_systems.py         # 混沌系统实现
│   ├── chaotic_encryptor.py     # 混沌加密器
│   └── frequency_cipher.py      # 频域加密
│
├── cipher/                      # 🔒 密码模块
│   ├── __init__.py
│   └── scne_cipher.py           # SCNE 密码实现
│
├── crypto/                      # 🔑 密钥系统
│   ├── __init__.py
│   └── key_system.py            # 密钥管理系统
│
├── neural/                      # 🧠 神经网络模块
│   ├── __init__.py
│   └── unet.py                  # U-Net 网络架构
│
├── evaluation/                  # 📊 评估模块
│   ├── __init__.py
│   ├── attack_models.py         # 攻击模型
│   ├── security_metrics.py      # 安全性指标
│   └── strong_recognizers.py    # 强识别器
│
├── baselines/                   # 📈 基线对比
│   ├── __init__.py
│   └── crypto_baselines.py      # 加密基线方法
│
├── vse_pc/                      # 🎯 VSE-PC 核心实现
│   ├── __init__.py
│   ├── causal_analysis.py       # 因果分析
│   ├── ciphertext_ml.py         # 密文机器学习
│   ├── interface.py             # 接口定义
│   ├── pipeline.py              # 处理流水线
│   ├── privacy_budget.py        # 隐私预算
│   └── verifiable.py            # 可验证性
│
├── models/                      # 📦 模型定义
│   ├── __init__.py
│   └── nsce_scheduler.pth       # NSCE 调度器权重
│
├── plotting/                    # 📉 可视化
│   ├── __init__.py
│   └── plot_style.py            # 绘图样式
│
├── utils/                       # 🛠️ 工具函数
│   ├── __init__.py
│   └── datasets.py              # 数据集工具
│
└── weights/                     # ⚖️ 模型权重 (Git 忽略)
    ├── checkpoints/             # 训练检查点
    │   ├── best_gnn_model.pth
    │   ├── trained_gnn_*.pth
    │   └── training_history_*.json
    └── pretrained/              # 预训练权重
        ├── unet_improved.pth
        ├── unet_mixed_optimized.pth
        ├── unet_v3_optimized.pth
        └── vggface2.pt
```

---

## 📁 scripts/ - 脚本文件

```
scripts/
├── __init__.py
├── README.md                    # 脚本说明
│
├── evaluation/                  # 📊 评估脚本
│   ├── attacks.py               # 攻击测试
│   ├── benchmark.py             # 性能基准测试
│   ├── ciphertext.py            # 密文分析
│   └── security.py              # 安全性评估
│
├── experiments/                 # 🧪 实验脚本
│   ├── train_strategy_real_q2.py    # 训练策略
│   ├── verify_chaos_core.py         # 混沌核心验证
│   ├── verify_chaos_dynamics.py     # 混沌动力学验证
│   └── vse_pc/                      # VSE-PC 实验
│       ├── __init__.py
│       ├── exp_baseline.py          # 基线实验
│       ├── FIXES.md                 # 修复记录
│       ├── test_arnold_inverse.py   # Arnold 逆变换测试
│       ├── test_causal_e2e_full.py  # 端到端因果测试
│       ├── test_chaos_deterministic.py  # 混沌确定性测试
│       ├── test_decrypt_layers.py   # 解密层测试
│       ├── test_deterministic.py    # 确定性测试
│       └── test_fix_verify.py       # 修复验证测试
│
├── training/                    # 🏋️ 训练脚本
│   ├── __init__.py
│   └── results/                 # 训练结果 (Git 忽略)
│
└── results/                     # 📈 脚本运行结果 (Git 忽略)
    └── causal_analysis_full/
```

---

## 📁 docs/ - 文档

```
docs/
├── project_overview.md              # 项目总览 (Project Overview)
├── workflow.md                      # 工作流程 (Workflow)
├── data_flow.md                     # 数据流向 (Data Flow)
├── dataset_analysis.md              # 数据集分析 (Dataset Analysis)
├── implementation_plan.md           # 实现计划 (Implementation Plan)
├── goals_and_metrics.md             # 目标与指标 (Goals & Metrics)
├── theoretical_proof.md             # 理论证明 (Theoretical Proof)
├── literature_review_2015_2025.md   # 文献综述 (Literature Review)
├── development_log.md               # 开发日志 (Development Log)
└── papers/                          # 参考论文
    └── .gitkeep
```

---

## 📁 examples/ - 示例代码

```
examples/
├── basic_usage.py      # 基础使用示例
└── advanced_usage.py   # 高级使用示例
```

---

## 📁 tests/ - 测试代码

```
tests/
├── __init__.py
└── test_encryption.py  # 加密测试
```

---

## 📁 data/ - 数据目录 (Git 忽略)

```
data/
├── CelebA-HQ/                   # CelebA-HQ 数据集
│   ├── train/                   # 训练集图像
│   ├── val/                     # 验证集图像
│   ├── test/                    # 测试集图像
│   ├── list_attr_celeba.txt     # 属性列表
│   └── split_manifest.json      # 数据集划分
│
├── CelebA-HQ-labels/            # 标签数据
│   ├── train/
│   └── val/
│
├── CelebAMask-HQ/               # CelebAMask-HQ 数据集
│   ├── CelebA-HQ-img/           # 原始图像
│   ├── CelebAMask-HQ/           # 掩码数据
│   ├── CelebAMask-HQ-mask-anno/ # 掩码标注
│   └── *.txt                    # 各类标注文件
│
├── weight/                      # 数据相关权重
│   └── checkpoints/
│
└── CelebAMask-HQ.zip            # 数据集压缩包
```

---

## 📁 results/ - 运行结果 (Git 忽略)

```
results/
└── .gitkeep            # 保持目录结构
```

---

## 🚫 已忽略的目录

以下目录已在 `.gitignore` 中配置，不会提交到 Git：

- `.venv-scne/` - Python 虚拟环境
- `.idea/` - PyCharm 配置
- `.vscode/` - VS Code 配置
- `.cursor/` - Cursor 配置
- `data/` - 数据文件
- `results/` - 运行结果
- `src/weights/` - 模型权重
- `__pycache__/` - Python 缓存

---

## 📝 备注

1. **模型权重**: 大型模型文件建议使用 Git LFS 或单独存储
2. **数据集**: CelebA-HQ 数据集需要单独下载
3. **配置文件**: 敏感配置请使用 `.env` 文件（参考 `.env.example`）
