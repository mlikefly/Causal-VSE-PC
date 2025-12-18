# Causal-VSE-PC 项目总览

**版本**: 2.1.1  
**更新日期**: 2024-12-18  
**目标期刊**: T-IFS / TIP / TNNLS / IEEE Access (SCI二区)

---

## 一、项目简介

Causal-VSE-PC (Causal Visual Semantic Encryption with Privacy Control) 是一个面向顶刊的隐私保护图像加密系统，核心创新在于：

1. **双视图架构** - Z-view（效用视图）+ C-view（密码学视图）分离
2. **因果隐私预算分配** - 基于ATE/CATE的语义区域预算优化
3. **5类攻击评估 + A2自适应攻击** - 完整的隐私保护证据链
4. **可审计的AEAD安全封装** - 机密性/完整性/抗重放安全目标

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Causal-VSE-PC System Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Protocol & Schema Layer                           │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ ProtocolManager │  │ ResultsSchema   │  │ ValidateRun     │       │   │
│  │  │ v2.1.1          │  │ (冻结字段)      │  │ (R1-R10红线)    │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Dual-View Encryption Layer                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ Z-view          │  │ C-view          │  │ NonceManager    │       │   │
│  │  │ (Utility View)  │  │ (Crypto View)   │  │ + ReplayCache   │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Evaluation Layer                                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐      │   │
│  │  │ 5类攻击    │  │ 效用评估   │  │ 因果效应   │  │ 安全评估   │      │   │
│  │  │ + A2自适应 │  │ P2P/Z2Z    │  │ ATE/CATE   │  │ NIST/Tamper│      │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Output Layer                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ 8张主图         │  │ 统计引擎        │  │ CI集成          │       │   │
│  │  │ FigureGenerator │  │ Bootstrap+BH-FDR│  │ smoke_test      │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心贡献

| 贡献编号 | 贡献内容 | 证据类型 |
|----------|----------|----------|
| C1 | **因果隐私预算分配** - 基于ATE/CATE的语义区域预算优化 | 主证据 |
| C2 | **双视图架构与威胁模型分离** - Z-view + C-view，支持A0/A1/A2 | 主证据 |
| C3 | **全面攻击评估 (5类 + A2)** - 完整的隐私保护证据链 | 主证据 |
| C4 | **可审计的AEAD安全封装** - 机密性/完整性/抗重放 | 主证据 |
| C5 | **可复现实验协议** - 协议冻结、覆盖度验证、图表复现 | 支撑证据 |

---

## 四、威胁模型

| 威胁等级 | 知识范围 | 攻击能力 |
|----------|----------|----------|
| **A0 (Black-box)** | 仅观察 Z-view 输出 | 基于输出的推断攻击 |
| **A1 (Gray-box)** | 知道算法流程、模型架构 | 可训练针对性攻击模型 |
| **A2 (White-box Adaptive)** | 知道 mask 生成逻辑、预算分配规则 | 可设计自适应攻击策略 |

---

## 五、安全边界声明

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

---

## 六、项目状态

### 已完成 ✅ 100%

**核心模块**:
- [x] 双视图加密引擎 (Z-view + C-view)
- [x] 混沌加密层 + 频域加密层
- [x] Nonce管理器 + 重放检测缓存
- [x] 分层密钥系统

**协议与验证**:
- [x] 协议与Schema基座 (v2.1.1)
- [x] 运行验证器 (R1-R10红线检查)

**攻击评估**:
- [x] 5类攻击框架 + A2自适应攻击
- [x] 攻击成功率归一化

**安全评估**:
- [x] C-view安全评估套件 (NIST/Avalanche/Tamper)
- [x] 安全指标计算

**训练与效用**:
- [x] 训练模式管理 (P2P/P2Z/Z2Z/Mix2Z)
- [x] 效用评估器

**因果分析**:
- [x] 因果效应估计 (ATE/CATE)
- [x] 预算优化求解

**基线与统计**:
- [x] 基线矩阵与对比
- [x] 统计引擎 (Bootstrap CI + BH-FDR)

**输出与CI**:
- [x] 图表生成器 (8张主图)
- [x] CI集成 (smoke_test + reproduce)
- [x] 12项消融实验
- [x] 鲁棒性与效率评估

### 下一步：测试与实验

- [ ] 运行完整测试套件 (`pytest tests/ -v`)
- [ ] 运行smoke_test验证管线
- [ ] 准备数据集 (CelebA-HQ, FairFace, OpenImages)
- [ ] 执行完整实验
- [ ] 生成论文图表
- [ ] 提交GitHub

---

## 七、目录结构

```
Causal-VSE-PC/
├── src/                      # 源代码
│   ├── cipher/               # 加密器模块
│   ├── core/                 # 核心算法
│   ├── crypto/               # 密码学组件
│   ├── data/                 # 数据流水线
│   ├── protocol/             # 协议与验证
│   ├── training/             # 训练模块
│   └── evaluation/           # 评估模块
├── tests/                    # 测试文件
├── scripts/                  # 脚本
├── configs/                  # 配置文件
├── docs/                     # 文档
└── .kiro/specs/              # 设计规范
```

---

## 八、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
pytest tests/ -v

# 3. 运行smoke_test
python scripts/run_benchmark.py --smoke_test

# 4. 运行完整实验
python scripts/run_benchmark.py --full
```

---

## 九、相关文档

- [设计文档](../.kiro/specs/top-journal-experiment-suite/design.md)
- [需求文档](../.kiro/specs/top-journal-experiment-suite/requirements.md)
- [任务清单](../.kiro/specs/top-journal-experiment-suite/tasks.md)
- [源代码说明](../src/README.md)
- [开发日志](development_log.md)
- [工作流程](workflow.md)
- [数据流向](data_flow.md)
- [数据集分析](dataset_analysis.md)
- [目标与指标](goals_and_metrics.md)
- [理论证明](theoretical_proof.md)
