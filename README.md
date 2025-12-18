# Causal-VSE-PC

**因果视觉语义加密与隐私控制系统**

[![协议版本](https://img.shields.io/badge/协议-v2.1.1-blue.svg)](docs/project_overview.md)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![许可证](https://img.shields.io/badge/许可证-MIT-yellow.svg)](LICENSE)

---

## 🎯 项目概述

Causal-VSE-PC 是一个面向顶刊发表（T-IFS/TIP/TNNLS）的隐私保护图像加密系统。系统实现了**双视图架构**与**因果隐私预算分配**。

### 核心贡献

| # | 贡献 | 证据 |
|---|------|------|
| C1 | **因果隐私预算分配** - 基于ATE/CATE的语义区域预算优化 | Pareto曲线 + 因果效应 |
| C2 | **双视图架构** - Z-view（效用）+ C-view（密码学）分离，支持A0/A1/A2威胁等级 | 攻击曲线 + worst-case聚合 |
| C3 | **全面攻击评估** - 5类攻击 + A2自适应攻击 | 攻击指标 + 统计显著性 |
| C4 | **可审计的AEAD安全** - 机密性/完整性/抗重放 | 安全验证 + 诊断证据 |
| C5 | **可复现协议** - 协议冻结、覆盖度验证、字节级图表复现 | Artifact清单 + CI结果 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Causal-VSE-PC 系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入图像 → 语义掩码 → 因果预算 → 双视图加密                      │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Z-view    │    │   C-view    │    │   评估层    │          │
│  │  (效用视图) │    │ (密码学视图)│    │  (5类攻击)  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                  │
│  训练模式: P2P / P2Z / Z2Z / Mix2Z                              │
│  威胁等级: A0 (黑盒) / A1 (灰盒) / A2 (自适应)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/mlikefly/Causal-VSE-PC.git
cd Causal-VSE-PC

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
from src.cipher.dual_view_engine import DualViewEngine

# 初始化加密引擎
engine = DualViewEngine(master_key=your_key)

# 双视图加密
z_view, c_view, enc_info = engine.encrypt(
    image,
    privacy_level=0.5
)

# Z-view: 用于ML推理（保留语义）
# C-view: 用于安全存储（AEAD封装）
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行smoke测试（< 20分钟）
python scripts/run_benchmark.py --smoke_test

# 运行完整实验
python scripts/run_benchmark.py --full
```

---

## 📊 指标

### 安全指标

| 指标 | 目标 | 当前值 |
|------|------|--------|
| NPCR | > 99.6% | ✅ 99.57% |
| UACI | 30-36% | ✅ 33.49% |
| 信息熵 | > 7.9 bits | ✅ 7.99 |
| Tamper失败率 | ≥ 99% | ✅ 已实现 |
| Replay拒绝率 | = 100% | ✅ 已实现 |

### 效用门槛

| 隐私级别 | 门槛 |
|----------|------|
| λ = 0.3 | ≥ 75% P2P |
| λ = 0.5 | ≥ 65% P2P |
| λ = 0.7 | ≥ 55% P2P |

---

## 🔒 安全边界声明

```
安全边界声明:
1. C-view安全性继承自标准AEAD（AES-GCM/ChaCha20-Poly1305），
   提供IND-CPA和IND-CCA保证。
2. 混沌/频域变换作为混淆/扩散层，
   不单独宣称语义安全。
3. Z-view隐私通过攻击成功率降低来实证证明，
   而非密码学证明。
4. 本系统不防御：侧信道攻击、物理攻击、
   或拥有加密密钥的攻击。
```

---

## 📁 项目结构

```
Causal-VSE-PC/
├── src/
│   ├── cipher/           # 加密引擎
│   ├── core/             # 核心算法（混沌、频域）
│   ├── crypto/           # 密码学组件
│   ├── data/             # 数据流水线
│   ├── protocol/         # 协议与验证
│   ├── training/         # 训练模式
│   └── evaluation/       # 评估框架
├── tests/                # 单元测试
├── scripts/              # 工具脚本
├── configs/              # 配置文件
├── docs/                 # 文档
└── .kiro/specs/          # 设计规范
```

---

## 📚 文档

- [项目总览](docs/project_overview.md)
- [工作流程](docs/workflow.md)
- [目标与指标](docs/goals_and_metrics.md)
- [开发日志](docs/development_log.md)
- [源代码指南](src/README.md)
- [设计文档](.kiro/specs/top-journal-experiment-suite/design.md)

---

## 🧪 攻击评估

### 5类攻击

| 攻击类型 | 指标 | 方向 |
|----------|------|------|
| 人脸验证 | TAR@FAR=1e-3 | ↓ 越低越好 |
| 属性推断 | AUC | ↓ 越低越好 |
| 重建攻击 | identity_similarity | ↓ 越低越好 |
| 成员推断 | AUC | ↓ 越低越好 |
| 属性推断 | AUC | ↓ 越低越好 |

### 威胁等级

| 等级 | 知识范围 | 攻击能力 |
|------|----------|----------|
| A0 | 仅Z-view输出 | 基于输出的推断 |
| A1 | 算法 + 架构 | 针对性攻击模型 |
| A2 | Mask + 预算分配 | 自适应攻击策略 |

---

## 📈 输出

### 8张主图

1. `fig_utility_curve.png` - 效用随privacy_level变化
2. `fig_attack_curves.png` - 5类攻击曲线+CI
3. `fig_pareto_frontier.png` - 隐私-效用Pareto前沿
4. `fig_causal_ate_cate.png` - ATE/CATE + CI
5. `fig_cview_security_summary.png` - C-view安全汇总
6. `fig_ablation_summary.png` - 消融实验对比
7. `fig_efficiency.png` - 效率对比
8. `fig_robustness.png` - 鲁棒性结果

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 👤 作者

- **mlikefly** - [GitHub](https://github.com/mlikefly)
- 邮箱: 1392792307@qq.com

---

## 🙏 致谢

- CelebA-HQ 数据集
- FairFace 数据集
- OpenImages 数据集
