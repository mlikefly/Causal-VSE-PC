# 设计文档：顶刊实验套件 (Top-Journal Experiment Suite)

> **文档结构说明**：本文档按三层叙事组织：
> - **Scientific Spec（科学规范）**：§1-§5 研究问题、威胁模型、指标定义、因果识别
> - **Protocol Spec（协议规范）**：§6-§10 训练协议、攻击协议、统计推断、覆盖度
> - **Artifact Spec（产物规范）**：§11-§17 目录结构、Schema、复现材料、CI

---

# Part I: Scientific Spec（科学规范）

---

## 1. 研究问题与贡献 (Research Questions & Contributions)

### 1.1 研究问题

**RQ1: 语义区域级隐私预算分配与因果效应估计**
我们是否能在 semantic-region 层级实现可控的隐私预算分配，并通过因果效应估计（ATE/CATE）指导预算优化，使 privacy-utility Pareto frontier 优于现有方法？

**RQ2: 自适应攻击下的鲁棒性 (A2)**
在 attacker adaptivity（A2 白盒自适应）下，dual-view 架构是否仍然保持隐私优势且不牺牲 utility？具体地，在冻结的攻击强度契约下，worst-case attack_success 是否仍优于 baseline？

**RQ3: C-view 安全目标与可验证性（安全定义驱动）**
C-view 是否满足明确的密码学安全目标：
- **机密性 (Confidentiality)**：在正确的 nonce 使用条件下，AEAD 封装提供 IND-CPA 级别保护
- **完整性 (Integrity)**：任何对 ciphertext/tag/aad 的篡改必须被检测（fail_rate ≥ 99%）
- **抗重放 (Replay Resistance)**：相同密文重放必须被拒绝

> **注意**：NIST SP800-22 和 Avalanche 测试作为**实现质量诊断**，而非安全证明。

**RQ4: 端到端可复现性**
实验流程是否可以实现端到端的确定性复现，包括统计推断、图表生成、覆盖度验证？

### 1.2 贡献

| 贡献编号 | 贡献内容 | 对应证据任务 | 证据类型 |
|----------|----------|--------------|----------|
| C1 | **因果隐私预算分配** - 基于因果效应估计（ATE/CATE）的语义区域预算分配，形式化为约束优化问题 | Pareto 曲线 + 因果效应表 + 统计检验 | 主证据 |
| C2 | **双视图架构与威胁模型分离** - Z-view（实证隐私）与 C-view（密码学安全）分离，支持 A0/A1/A2 三级威胁评估 | A2 攻击曲线 + worst-case 聚合 | 主证据 |
| C3 | **全面攻击评估 (5类 + A2)** - 覆盖 Face Verification、Attribute Inference、Reconstruction、Membership Inference、Property Inference | 攻击指标表 + 统计显著性 | 主证据 |
| C4 | **可审计的 AEAD 安全封装** - 满足机密性/完整性/抗重放安全目标，Tamper/Replay 作为完整性实验，NIST/Avalanche 作为实现质量诊断 | 安全目标验证表 + 诊断报告 | 主证据 + 诊断 |
| C5 | **可复现实验协议** - 协议冻结、确定性统计、覆盖度机器证明、图表字节级复现 | artifact checklist + CI 结果 | 支撑证据 |

---

## 2. 问题设置与符号表 (Problem Setup & Notation)

### 2.1 输入空间

| 符号 | 定义 | 说明 |
|------|------|------|
| $x \in \mathbb{R}^{H \times W \times 3}$ | 输入图像/视频帧 | RGB 格式 |
| $\{r_1, r_2, \ldots, r_K\}$ | 语义区域集合 | 如人脸、背景、敏感属性区域 |
| $m \in \{0,1\}^{H \times W}$ | 区域掩码 | 标识每个像素属于哪个区域 |
| $y$ | 任务标签 | 分类、分割、属性等 |

### 2.2 输出空间

| 符号 | 定义 | 说明 |
|------|------|------|
| $z = \mathcal{E}_Z(x, \beta, m)$ | Z-view (Utility View) | 可用于任务但隐私保护的视图 |
| $c = \mathcal{E}_C(z, k, n, \text{aad})$ | C-view (Cryptographic View) | AEAD 封装视图（含 nonce 和 AAD） |

### 2.3 隐私预算

| 符号 | 定义 | 说明 |
|------|------|------|
| $\beta_i \in [0, 1]$ | 每区域预算 | 区域 $r_i$ 的隐私预算 |
| $B = \sum_{i=1}^K \beta_i \cdot |r_i|$ | 全局预算 | 全局预算约束 |
| $\lambda \in \{0.0, 0.3, 0.5, 0.7, 1.0\}$ | 隐私等级 | 冻结网格 |

### 2.4 攻击者模型

| 威胁等级 | 观测集合 $\mathcal{O}_a$ | 说明 |
|----------|--------------------------|------|
| A0 (Black-box) | $\{z\}$ | 仅观察 Z-view |
| A1 (Gray-box) | $\{z, \text{algorithm}, \text{architecture}\}$ | 知道算法和架构 |
| A2 (White-box Adaptive) | $\{z, \text{algorithm}, \text{architecture}, m, \beta\}$ | 知道 mask 和预算分配，可设计自适应 loss |

### 2.5 因果效应符号

| 符号 | 定义 | 说明 |
|------|------|------|
| $\tau_i = \mathbb{E}[A | do(\beta_i = 1)] - \mathbb{E}[A | do(\beta_i = 0)]$ | 区域 $r_i$ 的 ATE | 平均处理效应 |
| $\tau_i(x) = \mathbb{E}[A | do(\beta_i = 1), X=x] - \mathbb{E}[A | do(\beta_i = 0), X=x]$ | 区域 $r_i$ 的 CATE | 条件平均处理效应 |
| $\beta^* = \arg\min_\beta \max_a A_a(z)$ s.t. $U(z) \geq U_{threshold}$ | 最优预算分配 | 约束优化目标 |

### 2.6 指标定义

| 指标 | 定义 | 方向 |
|------|------|------|
| **Utility** $U(z, y)$ | 任务效用（accuracy、F1、mIoU 等） | 越高越好 ↑ |
| **attack_success** $A(z, x)$ | 攻击成功率 | 越高=攻击越成功=隐私越弱 ↑ |
| **privacy_protection** | $1 - \text{normalized}(A(z, x))$ | 越高=保护越强 ↑ |

> **重要约定**：全文统一使用 `attack_success`（越高越差）作为攻击指标，所有图表必须标注 "higher is better / lower is better"。

---

## 3. 系统概览 (System Overview)

本设计文档描述 Causal-VSE-PC 项目冲击顶刊（T-IFS/TIP/TNNLS）的完整实验套件架构。基于已完成的双视图加密引擎和数据流水线，本阶段聚焦于：

1. **协议冻结与版本化** - 实验协议、结果 Schema、覆盖度验证
2. **5类攻击评估完善** - 新增 Membership Inference 和 Property Inference
3. **C-view 安全目标验证** - 机密性/完整性/抗重放 + NIST/Avalanche 诊断
4. **Z-view 训练策略** - Z2Z/Mix2Z 训练模式实现
5. **消融实验矩阵** - 12项冻结消融清单
6. **论文级输出工程化** - 一键生成图表、validate_run
7. **A2 自适应攻击** - 最强威胁等级覆盖 + 强度契约冻结
8. **因果效应估计** - ATE/CATE 估计 + 预算优化求解

### 3.1 核心设计原则

- **可审计性**：所有实验结果可追溯到协议版本、数据 manifest hash、代码 commit
- **可复现性**：确定性 nonce（受控选择）、固定 seed、冻结协议
- **覆盖度保证**：98% 组合覆盖 + hard fail 机制
- **统计严谨性**：bootstrap CI、BH-FDR 多重比较校正、family_id 一致性
- **安全定义驱动**：安全目标（机密性/完整性/抗重放）优先于统计测试

### 3.2 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Top-Journal Experiment Suite Architecture                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Protocol & Schema Layer                           │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ EXPERIMENT_     │  │ RESULTS_        │  │ protocol_       │       │   │
│  │  │ PROTOCOL.md     │  │ SCHEMA.md       │  │ version.txt     │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Data & Manifest Layer                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ ManifestBuilder │  │ ManifestDataset │  │ validate_splits │       │   │
│  │  │ + SHA256 hash   │  │ + C-view guard  │  │ + leakage check │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Experiment Execution Layer                         │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐      │   │
│  │  │run_utility │  │run_attacks │  │run_security│  │run_ablation│      │   │
│  │  │P2P/P2Z/Z2Z │  │5 types +A2 │  │NIST/Tamper │  │12 items    │      │   │
│  │  │/Mix2Z      │  │            │  │/Avalanche  │  │            │      │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Results & Validation Layer                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ validate_run    │  │ StatisticsEngine│  │ CoverageChecker │       │   │
│  │  │ structure/schema│  │ bootstrap/BH-FDR│  │ ≥98% gate       │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Output & Reporting Layer                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ make_figures    │  │ figure_specs    │  │ figure_manifest │       │   │
│  │  │ 8 main figures  │  │ dpi/size/font   │  │ .json           │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心机制：双视图 + 因果预算 (Core Mechanism: Dual-View + Causal Budget)

### 4.1 机制直觉 (Mechanism Intuition)

**Dual-View 的意义**：
- **Z-view (Utility View)**：用于下游任务（分类、检测等），保留语义信息但隐藏敏感属性
- **C-view (Cryptographic View)**：用于强加密审计/对照，提供密码学级别的安全保证

**为什么需要分离**：
- Z-view 面向实证隐私评估（攻击者能否推断敏感信息）
- C-view 面向密码学安全验证（NIST、Avalanche、Tamper）
- 两者的威胁模型和评估方法完全不同，必须解耦

### 4.2 形式化机制 (Formal Mechanism)

**语义掩码生成**：
$$m = \text{SemanticMaskGenerator}(x) \rightarrow \{r_1, r_2, \ldots, r_K\}$$

**因果预算分配**：
$$\beta_i = \text{CausalBudgetAllocator}(r_i, \lambda, \text{task}) \in [0, 1]$$

其中：
- $\lambda$ 是全局隐私等级
- task 是下游任务类型
- $\beta_i$ 决定区域 $r_i$ 的扰动强度

**Z-view 生成**：
$$z = \mathcal{E}_Z(x, \beta, m) = \text{FrequencyTransform} \circ \text{ChaoticScramble} \circ \text{RegionMask}(x, \beta, m)$$

**C-view 生成**：
$$c = \mathcal{E}_C(z, k) = \text{AEAD}(z, k, \text{nonce}, \text{aad})$$

### 4.3 必要性论证 (Why Necessary)

| 去掉组件 | 后果 | 影响 |
|----------|------|------|
| 去掉 Causal Budget | 只能全局噪声/全局扰动 | Pareto frontier 被压扁，无法细粒度权衡 |
| 去掉 Dual-View | 无法同时满足 utility 与可验证安全性 | 要么牺牲任务性能，要么无法通过安全审计 |
| 去掉 Semantic Mask | 无法区分敏感/非敏感区域 | 隐私保护过度或不足 |
| 去掉 AEAD 封装 | C-view 无法抵抗 tamper/replay | 安全性无法验证 |

---

## 5. 威胁模型 (Threat Model)

### 5.1 攻击者能力定义

| 威胁等级 | 知识范围 | 攻击能力 |
|----------|----------|----------|
| **A0 (Black-box)** | 仅观察 Z-view 输出 | 基于输出的推断攻击 |
| **A1 (Gray-box)** | 知道算法流程、模型架构 | 可训练针对性攻击模型 |
| **A2 (White-box Adaptive)** | 知道 mask 生成逻辑、预算分配规则 | 可设计自适应攻击策略 |

### 5.2 攻击者不可访问

- 加密密钥 $k$
- 训练权重（除非明确声明）
- 原始明文 $x$（攻击目标）

### 5.3 威胁模型边界与非目标 (Threat Model Boundary & Non-goals)

| 范围 | 内容 | 原因 | 影响的属性 |
|------|------|------|------------|
| **In-scope** | A0/A1/A2 攻击、黑盒/白盒推断 | 核心研究问题 | Property 3, 14 |
| **In-scope** | Tamper/Replay 攻击 | C-view 完整性验证 | Property 4 |
| **Out-of-scope** | Key compromise | 假设密钥安全存储 | Property 4 失效 |
| **Out-of-scope** | Gradient access (训练时) | 问题定义不同，需另一研究方向 | Property 6 失效 |
| **Out-of-scope** | Poisoning / Backdoor | 数据完整性假设 | Property 12 失效 |
| **Out-of-scope** | Physical side-channel | 超出软件层面 | N/A |
| **Out-of-scope** | Nonce misuse (同一 key+nonce 加密不同明文) | 协议级保证 | Property 4 失效 |

**Expected Failure 详细说明**：

| 超出边界场景 | 破坏的属性 | 后果 |
|--------------|------------|------|
| Key compromise | Property 4 (C-view 安全性) | 攻击者可解密所有 C-view |
| Gradient access | Property 6 (训练隔离) | 攻击者可通过梯度推断训练数据 |
| Nonce reuse | Property 4 (C-view 安全性) | AES-GCM 安全性完全崩溃 |
| Data poisoning | Property 12 (数据泄漏防护) | 攻击者可植入后门 |

### 5.4 A2 强度契约 (A2 Strongness Contract) - 冻结

为确保 A2 攻击的强度可审计、可复现、不可被挑软柿子，冻结以下三件事：

#### 5.4.1 攻击族谱 (Attack Families)

| Family | 描述 | 实例化策略 | 最少实例数 |
|--------|------|------------|------------|
| **Reconstruction** | 重建原始图像 | (1) U-Net decoder (2) GAN-based inversion | 2 |
| **Inference** | 推断敏感属性/身份 | (1) Linear probe (2) MLP classifier (3) Contrastive learning | 3 |
| **Optimization** | 自适应优化攻击 | (1) Gradient-based (2) Evolutionary search | 2 |

#### 5.4.2 攻击预算 (Hyperparam + Compute Budget)

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练轮数 | 100 epochs | 每个攻击模型 |
| 学习率搜索 | {1e-4, 1e-3, 1e-2} | 3 候选 |
| 架构搜索 | 固定（见 protocol_snapshot） | 不允许 NAS |
| 总 GPU 时间 | ≤ 24h per attack family | 防止无限搜索 |

#### 5.4.3 Worst-case 聚合规则

```
worst_case_attack_success = max(attack_success) over all attacks in same (dataset, task, privacy_level, threat_level)
```

> **冻结声明**：以上契约写入 `protocol_snapshot.md`，任何修改必须更新 protocol_version。

### 5.5 Nonce 策略：受控选择 (Nonce Strategy: Controlled Choice)

#### 5.5.1 问题背景

当前默认使用确定性 nonce（`SHA256(image_id + master_key)[:12]`）来服务确定性复现。但这存在风险：
- 同一 `image_id` 多次加密会导致 nonce 重用
- Nonce 重用在 AES-GCM 中是高风险点（安全性完全崩溃）

#### 5.5.2 协议级选择（冻结）

| 选项 | 描述 | 适用场景 | 默认 |
|------|------|----------|------|
| **Option A (推荐)** | 确定性 nonce + 唯一性约束 | 复现优先，每个 image_id 只加密一次 | ✓ |
| **Option B** | 随机 nonce + 记录到 meta/ | 安全优先，复现靠记录 | |

#### 5.5.3 唯一性约束实现（协议唯一元组）

> **问题**：如果只用 `SHA256(image_id + master_key)`，同一 image 因不同 privacy_level / method / 复跑被重复加密会触发 nonce 重用。

**解决方案**：Nonce derivation 输入改为**协议唯一元组**（冻结）：

```python
@dataclass
class NonceDerivationInput:
    """Nonce 派生输入元组（冻结）"""
    image_id: str
    method: str           # e.g., "causal_vse_pc"
    privacy_level: float  # e.g., 0.5
    training_mode: str    # e.g., "Z2Z"
    purpose: str          # e.g., "c_view_encrypt", "tamper_test", "avalanche_test"

class NonceManager:
    """Nonce 管理器 - 确保唯一性"""
    
    def __init__(self, master_key: bytes, run_dir: Path):
        self.master_key = master_key
        self.used_nonces: Set[bytes] = set()
        self.nonce_log = run_dir / "meta" / "nonce_log.json"
        self.log_entries: List[Dict] = []
        
    def derive_nonce(self, input: NonceDerivationInput) -> bytes:
        """
        确定性 nonce 派生 + 唯一性检查
        
        nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
        """
        derivation_string = f"{input.image_id}|{input.method}|{input.privacy_level}|{input.training_mode}|{input.purpose}"
        nonce = hashlib.sha256(
            self.master_key + derivation_string.encode()
        ).digest()[:12]
        
        if nonce in self.used_nonces:
            raise NonceReuseError(
                f"Nonce reuse detected for input={input}"
            )
        self.used_nonces.add(nonce)
        self._log_nonce(input, nonce)
        return nonce
    
    def _log_nonce(self, input: NonceDerivationInput, nonce: bytes) -> None:
        """记录 nonce 用于审计"""
        self.log_entries.append({
            "image_id": input.image_id,
            "method": input.method,
            "privacy_level": input.privacy_level,
            "training_mode": input.training_mode,
            "purpose": input.purpose,
            "nonce_hex": nonce.hex(),
            "timestamp": datetime.now().isoformat()
        })
    
    def persist(self) -> None:
        """运行结束时落盘"""
        with open(self.nonce_log, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
```

**purpose 枚举（冻结）**：

| purpose | 说明 |
|---------|------|
| `c_view_encrypt` | 正常 C-view 加密 |
| `tamper_test` | Tamper 测试用加密 |
| `avalanche_test` | Avalanche 测试用加密 |
| `replay_test` | Replay 测试用加密 |

> **核心原则**：复现性目标不应靠破坏 AEAD 的使用前提来换。

---

# Part II: Protocol Spec（协议规范）

---

## 6. 攻击面与攻击 API (Attack Surfaces & Attack APIs)

### 6.1 五类攻击定义

| 攻击类型 | 目标 | attack_success 映射 |
|----------|------|---------------------|
| Face Verification | 身份识别 | TAR@FAR=1e-3 |
| Attribute Inference | 敏感属性推断 | AUC |
| Reconstruction | 图像重建 | identity_similarity |
| Membership Inference | 成员推断 | AUC |
| Property Inference | 群体属性分布推断 | AUC |

### 6.2 攻击 API 冻结接口

```python
@dataclass
class AttackFitContext:
    """攻击训练上下文（冻结接口）"""
    run_id: str
    dataset: str
    task: str
    method: str
    training_mode: str
    privacy_level: float
    seed: int
    threat_level: str  # A0/A1/A2
    attacker_visible: Dict[str, Any]  # 攻击者可见信息（必须落盘审计）

class AttackBase(ABC):
    """攻击基类（冻结签名）"""
    
    @abstractmethod
    def fit(self, ctx: AttackFitContext) -> None:
        """训练攻击模型"""
        pass
    
    @abstractmethod
    def evaluate(self, ctx: AttackEvalContext) -> Dict[str, Any]:
        """评估攻击成功率"""
        pass
    
    def get_attack_success(self, metrics: Dict) -> float:
        """根据 GC7 映射表计算统一 attack_success"""
        raise NotImplementedError
```

### 6.3 A2 自适应攻击者

```python
class AdaptiveAttacker(AttackBase):
    """
    A2 白盒自适应攻击者
    
    攻击者能力：
    - 知道完整算法（加密流程、mask 生成、预算分配）
    - 知道模型结构（但无训练权重）
    - 无法访问加密密钥
    - 可自适应设计攻击策略
    """
    
    def design_adaptive_strategy(self, 
                                  algorithm_info: Dict,
                                  mask_generator: SemanticMaskGenerator,
                                  budget_allocator: CausalPrivacyBudgetAllocator) -> None:
        """
        设计自适应攻击策略
        
        利用算法知识：
        1. 分析 mask 生成逻辑，找到语义保留区域
        2. 分析预算分配规则，找到低保护区域
        3. 设计针对性的 loss 函数
        """
        self.adaptive_strategy = {
            "mask_analysis": self._analyze_mask_patterns(mask_generator),
            "budget_analysis": self._analyze_budget_patterns(budget_allocator),
            "attack_loss": self._design_adaptive_loss(algorithm_info)
        }
```

---

## 7. 指标与约定 (Metrics & Conventions)

### 7.1 指标方向统一表

| 指标名称 | 方向 | 说明 |
|----------|------|------|
| `attack_success` | 越高越差 ↑ | 攻击越成功 = 隐私越弱 |
| `privacy_protection` | 越高越好 ↑ | = 1 - normalized(attack_success) |
| `utility` (accuracy/F1/mIoU) | 越高越好 ↑ | 任务性能 |
| `relative_performance` | 越高越好 ↑ | = metric_value / P2P_mean |
| `NIST p_value` | ≥ 0.01 为 pass | 随机性检验 |
| `avalanche flip_rate` | 0.45-0.55 为 pass | 雪崩效应 |
| `tamper_fail_rate` | ≥ 0.99 为 pass | 篡改检测率 |

### 7.2 attack_success 映射表 (GC7)

| 攻击类型 | 原始指标 | 映射到 attack_success |
|----------|----------|----------------------|
| face_verification | TAR@FAR=1e-3 | 直接使用 |
| attribute_inference | AUC | 直接使用 |
| reconstruction | identity_similarity | 直接使用 |
| membership_inference | AUC | 直接使用 |
| property_inference | AUC | 直接使用 |

### 7.3 attack_success 归一化定义（冻结）

为确保跨攻击可比性，冻结以下归一化方式：

| 攻击类型 | 归一化方式 | 下界来源 | 上界来源 |
|----------|------------|----------|----------|
| face_verification | $(x - x_{random}) / (x_{P2P} - x_{random})$ | 随机猜测 TAR | P2P 模式 TAR |
| attribute_inference | $(x - 0.5) / (x_{P2P} - 0.5)$ | 随机猜测 AUC=0.5 | P2P 模式 AUC |
| reconstruction | $x / x_{P2P}$ | 0 (完全不相似) | P2P 模式 similarity |
| membership_inference | $(x - 0.5) / (x_{P2P} - 0.5)$ | 随机猜测 AUC=0.5 | P2P 模式 AUC |
| property_inference | $(x - 0.5) / (x_{P2P} - 0.5)$ | 随机猜测 AUC=0.5 | P2P 模式 AUC |

**归一化后范围**：[0, 1]，其中 0 = 完全保护，1 = 无保护（等同 P2P）

### 7.4 汇总指标定义（冻结）

| 汇总指标 | 计算方式 | 权重 |
|----------|----------|------|
| `avg_privacy_protection` | 所有攻击类型的 privacy_protection 均值 | 均匀权重 |
| `worst_case_privacy_protection` | 所有攻击类型的 privacy_protection 最小值 | N/A |
| `weighted_privacy_protection` | 按威胁等级加权 | A0:0.2, A1:0.3, A2:0.5 |

> **推荐**：论文主表使用 `worst_case_privacy_protection`，附录使用 `avg_privacy_protection`。

### 7.5 图表标注规范

- 所有图表必须标注 "↑ higher is better" 或 "↓ lower is better"
- 同名指标在全文只允许一个方向
- Pareto 图：X 轴 = utility (↑)，Y 轴 = privacy_protection (↑)

---

## 8. 协议：训练/推理/密钥/种子 (Protocol: Training / Inference / Keying / Seeds)

### 8.1 训练模式

| 模式 | 训练数据 | 测试数据 | 说明 |
|------|----------|----------|------|
| P2P | Plaintext | Plaintext | 基线（无隐私保护） |
| P2Z | Plaintext | Z-view | 域迁移测试 |
| Z2Z | Z-view | Z-view | 完全加密训练 |
| Mix2Z | 50% Plaintext + 50% Z-view | Z-view | 混合训练 |

### 8.2 C-view Guard

**规则**：训练时永不返回 C-view，只允许 plaintext 和 z_view。

```python
class TrainingModeManager:
    def get_dataloader(self, mode: str, split: str) -> DataLoader:
        # C-view guard: 训练时永不返回 c_view
        if split == "train":
            assert "c_view" not in requested_fields
```

### 8.3 种子与确定性

| 组件 | 种子来源 | 确定性保证 |
|------|----------|------------|
| 数据分割 | manifest seed | 相同 seed → 相同分割 |
| 模型初始化 | config seed | 相同 seed → 相同权重 |
| Nonce 生成 | SHA256(image_id + master_key) | 确定性 nonce |
| 统计采样 | bootstrap seed | 相同 seed → 相同 CI |

### 8.4 协议版本管理

```python
class ProtocolManager:
    PROTOCOL_VERSION = "2.1.1"  # 与文档版本一致
    SCHEMA_VERSION = "2.1.1"    # 与协议版本一致
    
    def validate_consistency(self, config: Dict) -> bool:
        """验证配置与协议快照一致性"""
        # protocol_version 必须与 schema_version 匹配
        # protocol_snapshot.md 必须与 config.yaml 一致
```

**版本号 bump 规则（冻结）**：

| 变更类型 | 版本号变化 | 示例 |
|----------|------------|------|
| Schema 字段新增/删除 | Major (X.0.0) | 新增 CSV 必需字段 |
| 协议逻辑变更 | Minor (0.X.0) | A2 强度契约修改 |
| Bug 修复/文档澄清 | Patch (0.0.X) | Nonce 派生输入元组澄清 |

---

## 9. 评估计划 (Evaluation Plan)

### 9.1 核心声明与所需证据 (Core Claims & Required Evidence)

| Claim | 声明内容 | 所需证据 | 输出物 | 证据类型 |
|-------|----------|----------|--------|----------|
| **C1** | Pareto frontier 优于 baseline + 因果效应指导预算分配 | Pareto 曲线 + ATE/CATE 表 + 统计检验 | fig_pareto_frontier.png, causal_effects.csv | 主证据 |
| **C2** | A2 自适应攻击下仍保持隐私优势（在冻结强度契约下） | A2 攻击曲线 + worst-case 聚合 | fig_attack_curves.png, table_a2.csv | 主证据 |
| **C3** | C-view 满足安全目标（机密性/完整性/抗重放） | Tamper/Replay 实验 + NIST/Avalanche 诊断 | security_report.md, fig_cview_security.png | 主证据 + 诊断 |
| **C4** | 全流程可复现、可审计 | seed lock + schema freeze + CI 结果 | ARTIFACT_CHECKLIST.md, reproduce.sh | 支撑证据 |

### 9.2 因果效应估计：两阶段方法 (Causal Effect Estimation)

#### 阶段 I：干预网格 (Intervention Grid)

对每个语义区域 $r_i$ 系统干预 $\beta_i$，观测 attack_success 与 utility：

```python
@dataclass
class InterventionGrid:
    """干预网格配置"""
    regions: List[str] = ["face", "background", "sensitive_attr"]
    beta_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5 级网格
    n_samples_per_cell: int = 100  # 每个网格单元的样本数
    
    def generate_experiments(self) -> List[Dict]:
        """生成所有干预实验"""
        experiments = []
        for region in self.regions:
            for beta in self.beta_values:
                experiments.append({
                    "intervention": {region: beta},
                    "n_samples": self.n_samples_per_cell
                })
        return experiments
```

#### 阶段 II：效应估计 → 预算优化 (Effect → Allocation Solver)

```python
class CausalBudgetOptimizer:
    """基于因果效应的预算优化器"""
    
    def estimate_ate(self, intervention_results: pd.DataFrame) -> Dict[str, float]:
        """估计每个区域的 ATE"""
        ate = {}
        for region in self.regions:
            high_beta = intervention_results[intervention_results[f"beta_{region}"] >= 0.75]
            low_beta = intervention_results[intervention_results[f"beta_{region}"] <= 0.25]
            ate[region] = high_beta["attack_success"].mean() - low_beta["attack_success"].mean()
        return ate
    
    def solve_optimal_allocation(self, ate: Dict[str, float], 
                                  utility_threshold: float) -> Dict[str, float]:
        """
        求解最优预算分配
        
        目标：min max_a attack_success_a
        约束：utility >= utility_threshold
        """
        # 使用 scipy.optimize 或 cvxpy 求解
        pass
```

**输出物**：
- `causal_effects.csv`：每个区域的 ATE/CATE
- `fig_causal_ate_cate.png`：因果效应可视化
- `optimal_allocation.json`：最优预算分配

### 9.3 效用评估

**门槛定义**：
| privacy_level | 门槛 | 说明 |
|---------------|------|------|
| 0.3 | 75% P2P | 低隐私保护，高效用要求 |
| 0.5 | 65% P2P | 中等隐私保护 |
| 0.7 | 55% P2P | 高隐私保护，允许更多效用损失 |

**失败分析**：如果 relative_performance < 门槛，必须生成失败分析报告。

### 9.4 基线矩阵（冻结）(Baseline Matrix)

为避免"挑弱 baseline"的质疑，冻结以下基线矩阵：

| Baseline | 适用任务 | 支持 Z2Z | 支持 Mix2Z | 支持 Region-level | 支持 A2 | 来源 |
|----------|----------|----------|------------|-------------------|---------|------|
| **InstaHide** | Classification | ✓ | ✓ | ✗ | ✓ | ICML 2020 |
| **P3** | Classification | ✓ | ✗ | ✗ | ✓ | CVPR 2021 |
| **DP-SGD** | Classification | ✓ | ✗ | ✗ | ✓ | CCS 2016 |
| **Pixelation** | All | ✓ | ✓ | ✓ | ✓ | Traditional |
| **Gaussian Blur** | All | ✓ | ✓ | ✓ | ✓ | Traditional |

**覆盖规则**：
- 每个 baseline 必须在其适用任务上评估
- 如果 baseline 不支持某项（如 Region-level），记录为 "N/A" 并给出理由
- 覆盖度计算时，"N/A" 不计入缺失

### 9.5 攻击评估

**覆盖要求**：
- 5 类攻击 × 3 威胁等级 × 5 隐私等级 × 3 seeds = 225 组合
- 覆盖度 ≥ 98%

**A2 强制存在**：
- attack_metrics.csv 必须包含至少一条 threat_level=A2 的记录
- 缺少 A2 记录 → hard fail

### 9.6 安全评估（安全目标驱动）

#### 9.6.1 主证据：安全目标验证

| 安全目标 | 验证方法 | 通过标准 | 输出 |
|----------|----------|----------|------|
| **机密性** | AEAD 正确使用 + nonce 唯一性检查 | 无 nonce 重用 | nonce_log.json |
| **完整性** | Tamper 实验（ciphertext/tag/aad） | fail_rate ≥ 99% | tamper_results.csv |
| **抗重放** | Replay 实验 + ReplayCache | reject_rate = 100% | replay_results.csv |

#### 9.6.1.1 抗重放实现细节（ReplayCache）

> **重要**：AEAD 本身不防重放，必须在解密侧引入 ReplayCache。

```python
class ReplayCache:
    """
    重放检测缓存
    
    Key: (key_id, nonce, tag_prefix)
    生命周期: per-run（每次实验运行独立）
    落盘策略: 运行结束时写入 meta/replay_cache.json
    """
    
    def __init__(self, run_dir: Path, key_id: str):
        self.key_id = key_id
        self.seen: Set[Tuple[str, bytes, bytes]] = set()
        self.cache_path = run_dir / "meta" / "replay_cache.json"
        
    def check_and_record(self, nonce: bytes, tag: bytes, ciphertext: bytes) -> bool:
        """
        检查是否重放，如果是新的则记录
        
        Returns:
            True: 新密文，允许解密
            False: 重放检测，拒绝解密
        """
        # 使用完整 tag 避免碰撞风险
        # 备选方案：SHA256(ciphertext||tag)[:16] 可进一步减少存储
        key = (self.key_id, nonce, tag)  # full_tag 确保零碰撞
        if key in self.seen:
            return False  # 重放！
        self.seen.add(key)
        return True
    
    # 注：如需减少存储，可改用 tag[:8]，碰撞概率上界为 2^-64
    # 对于研究 per-run 缓存（典型 <100k 样本），碰撞概率 < 10^-9 可接受
    # 但建议在生产环境使用 full_tag
    
    def persist(self) -> None:
        """运行结束时落盘"""
        # 写入 JSON 便于审计
```

**生命周期选项（冻结）**：

| 选项 | 描述 | 适用场景 |
|------|------|----------|
| **per-run** (默认) | 每次实验运行独立 | 研究实验 |
| per-session | 跨多次运行持久化 | 生产部署（非本项目范围） |

#### 9.6.2 诊断证据：实现质量

| 诊断项 | 验证方法 | 期望范围 | 说明 |
|--------|----------|----------|------|
| **NIST SP800-22** | 7 项子测试 | p_value ≥ 0.01 | 随机性诊断，非安全证明 |
| **Avalanche** | 3 种 flip 类型 | flip_rate ∈ [0.45, 0.55] | 扩散性诊断 |

> **重要**：NIST/Avalanche 失败不直接导致安全目标失败，但需要在报告中说明原因。

---

# Part III: Artifact Spec（产物规范）

---

## 10. 可复现性与验证 (Reproducibility & Verification)

### 10.0 复现材料与证明分离

将复现体系分为两层：

| 层次 | 内容 | 目的 |
|------|------|------|
| **复现材料 (Artifacts)** | 环境、数据、代码、配置 | 让他人能重跑实验 |
| **复现证明 (Verification)** | Schema 检查、覆盖度、字节级比对 | 证明结果一致 |

#### 10.0.1 复现材料清单（冻结）

| 材料类型 | 内容 | 存储位置 | 必需 |
|----------|------|----------|------|
| **环境** | Python 版本、库版本、CUDA 版本 | meta/env.txt | ✓ |
| **硬件** | GPU 型号、CPU 型号、内存 | meta/hardware.json | ✓ |
| **数据** | manifest_hash + split ids | meta/dataset_manifest_hash.txt | ✓ |
| **数据获取** | 下载脚本 + 校验脚本 | scripts/download_data.sh | ✓ |
| **许可说明** | 数据集许可证 | DATA_LICENSE.md | ✓ |
| **代码** | git commit hash | meta/git_commit.txt | ✓ |
| **配置** | 完整配置文件 | meta/config.yaml | ✓ |
| **Nonce 日志** | 所有使用的 nonce | meta/nonce_log.json | ✓ |

#### 10.0.2 复现证明输出

| 证明类型 | 验证内容 | 输出文件 |
|----------|----------|----------|
| **Schema 合规** | CSV 字段完整性 | reports/schema_validation.json |
| **覆盖度** | ≥ 98% 组合覆盖 | reports/coverage_report.json |
| **字节级复现** | 图表 SHA256 一致 | reports/figure_manifest.json |
| **统计复现** | 相同 seed → 相同 CI | reports/statistical_reproducibility.json |

#### 10.0.3 审稿人友好报告

`validate_run` 输出一页报告，包含：
- ✓/✗ 每项检查的通过状态
- 失败项的具体原因
- 缺失组合的列表（如有）
- 复现命令

### 10.1 Schema 冻结

```python
@dataclass
class ResultsSchema:
    """结果 Schema 定义（冻结）"""
    
    # utility_metrics.csv 必须字段
    UTILITY_FIELDS = [
        ("dataset", str, True),
        ("task", str, True),
        ("method", str, True),
        ("training_mode", str, True),  # P2P/P2Z/Z2Z/Mix2Z
        ("privacy_level", float, True),
        ("seed", int, True),
        ("metric_name", str, True),
        ("metric_value", float, True),
        ("relative_to", str, True),  # P2P_mean
        ("relative_performance", float, True),
        ("ci_low", float, True),
        ("ci_high", float, True),
        ("stat_method", str, True),
        ("n_boot", int, True),
        ("family_id", str, True),
        ("alpha", float, True),
    ]
    
    # attack_metrics.csv 必须字段
    ATTACK_FIELDS = [
        ("dataset", str, True),
        ("task", str, True),
        ("method", str, True),
        ("training_mode", str, True),
        ("attack_type", str, True),  # 5 types
        ("threat_level", str, True),  # A0/A1/A2
        ("privacy_level", float, True),
        ("seed", int, True),
        ("attack_success", float, True),  # 统一指标
        ("metric_name", str, True),
        ("metric_value", float, True),
        ("ci_low", float, True),
        ("ci_high", float, True),
        ("status", str, True),  # success/failed
        ("stat_method", str, True),
        ("n_boot", int, True),
        ("family_id", str, True),
        ("alpha", float, True),
    ]
```

### 10.2 Manifest 与数据完整性

```python
class ManifestBuilder:
    """数据 Manifest 构建器"""
    
    def build(self, data_dir: Path) -> Dict:
        return {
            "manifest_hash": self._compute_hash(),
            "split_info": {
                "train": {"count": N, "ids": [...]},
                "val": {"count": M, "ids": [...]},
                "test": {"count": K, "ids": [...]}
            },
            "leakage_check": self._check_split_leakage()
        }
```

### 10.3 确定性 Nonce

> **实现引用**：使用 §5.5.3 定义的 `NonceManager` 和协议唯一元组派生。

```python
# 参见 §5.5.3 NonceManager 完整实现
# nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
from src.core.nonce_manager import NonceManager, NonceDerivationInput

nonce_mgr = NonceManager(master_key, run_dir)
nonce = nonce_mgr.derive_nonce(NonceDerivationInput(
    image_id="img_001",
    method="causal_vse_pc",
    privacy_level=0.5,
    training_mode="Z2Z",
    purpose="c_view_encrypt"
))
```

### 10.4 CI 集成

**smoke_test 模式**：
- 时间预算 < 20 min
- 覆盖核心路径
- 验证 Schema 合规性

**CI 例外条款（冻结）**：

由于 A2 强度契约要求 100 epochs + 24h/family，与 smoke_test < 20min 存在冲突，特定义以下例外：

| 模式 | attacker_strength | 配置 | 用途 |
|------|-------------------|------|------|
| **smoke_test (CI)** | `lite` | 5 epochs, 1 实例化, 子集数据 | 管线健康检查 |
| **full (主证据)** | `full` | 100 epochs, 全实例化, 全数据 | C1-C4 主证据 |

**lite 模式约束**：
- 必须产出完整 schema（字段齐全）
- 必须在 CSV 中标记 `attacker_strength=lite`
- **不得用于任何主证据（C1-C4）**，仅用于 CI 健康检查
- 对 lite 模式的降级**不触发 GC10 hard fail**

**reproduce.sh**：
- 一键复现全部实验（使用 `full` 模式）
- 输出 ARTIFACT_CHECKLIST.md

### 10.5 覆盖度验证

```python
class ValidateRun:
    """运行结果验证器"""
    
    COVERAGE_THRESHOLD = 0.98  # 98% 覆盖度门槛
    
    def validate(self) -> Tuple[bool, Dict]:
        # 1. 目录结构检查
        # 2. Schema 检查
        # 3. 覆盖度检查 (≥ 98%)
        # 4. 协议一致性检查
        
        if coverage < self.COVERAGE_THRESHOLD:
            self._write_missing_matrix()  # 输出缺失组合
            return False, {"error": "Coverage gate failed"}
```

### 10.6 图表字节级复现

```python
class FigureManifest:
    """图表 manifest 管理器"""
    
    def verify(self) -> bool:
        """验证所有图表可从 csv 重建"""
        for entry in self.entries:
            figure_path = self.run_dir / "figures" / f"{entry.figure_name}.png"
            if self._compute_sha256(figure_path) != entry.sha256:
                return False
        return True
```

---

## 11. 消融实验与非目标 (Ablations & Non-goals)

### 11.1 消融实验目录（冻结）

| ID | 名称 | 描述 | 配置覆盖 | 验证目标 |
|----|------|------|----------|----------|
| A1 | remove_layer1 | 去混沌层 | `chaos_enabled: False` | 混沌层对隐私的贡献 |
| A2 | remove_layer2 | 去频域层 | `freq_enabled: False` | 频域层对隐私的贡献 |
| A3 | remove_crypto_wrap | 去AEAD封装 | `aead_enabled: False` | AEAD 对安全性的贡献 |
| A4 | causal_to_uniform | 因果预算→均匀预算 | `budget_mode: uniform` | 因果分配 vs 均匀分配 |
| A5 | causal_to_sensitive_only | 因果预算→仅敏感区域 | `budget_mode: sensitive_only` | 区域选择策略 |
| A6 | causal_to_task_only | 因果预算→仅任务区域 | `budget_mode: task_only` | 区域选择策略 |
| A7 | mask_strong_to_weak | 强监督mask→弱监督mask | `mask_supervision: weak` | mask 质量影响 |
| A8 | fft_to_dwt | FFT→DWT | `freq_transform: dwt` | 频域变换选择 |
| A9 | semantic_preserving_off | 语义保留关闭 | `semantic_preserving: False` | 语义保留的必要性 |
| A10 | deterministic_nonce_to_random | 确定性nonce→随机nonce | `deterministic_nonce: False` | nonce 策略对复现性的影响 |
| A11 | budget_normalization_variantA | 预算归一策略A | `budget_norm: variant_a` | 归一化策略选择 |
| A12 | budget_normalization_variantB | 预算归一策略B | `budget_norm: variant_b` | 归一化策略选择 |

> **A10 说明**：此消融验证 nonce 策略选择（§5.5）的影响。随机 nonce 模式下，nonce 记录到 meta/nonce_log.json 以支持复现。

### 11.2 非目标 (Non-goals)

本实验套件**不包含**以下内容：

| 非目标 | 原因 |
|--------|------|
| 用户验收测试 | 超出代码实现范围 |
| 生产环境部署 | 研究原型，非生产系统 |
| 性能指标收集（延迟、吞吐） | 可选扩展，非核心 claim |
| 端到端手动测试 | 使用自动化测试替代 |
| 文档创建（除 artifact checklist） | 论文撰写阶段处理 |

---

## 12. 组件与接口 (Components and Interfaces)

### 12.1 协议管理器

```python
class ProtocolManager:
    """协议版本管理器"""
    
    # 版本号与文档版本一致（参见 §8.4 版本号 bump 规则）
    PROTOCOL_VERSION = "2.1.1"
    SCHEMA_VERSION = "2.1.1"
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.meta_dir = run_dir / "meta"
        
    def write_protocol_version(self) -> None:
        """写入协议版本到 meta/protocol_version.txt"""
        (self.meta_dir / "protocol_version.txt").write_text(self.PROTOCOL_VERSION)
    
    def write_protocol_snapshot(self, config: Dict) -> None:
        """写入协议快照到 reports/protocol_snapshot.md"""
        snapshot = self._generate_snapshot(config)
        (self.run_dir / "reports" / "protocol_snapshot.md").write_text(snapshot)
    
    def validate_consistency(self, config: Dict) -> bool:
        """验证配置与协议快照一致性"""
        snapshot_path = self.run_dir / "reports" / "protocol_snapshot.md"
        if not snapshot_path.exists():
            raise ProtocolError("Missing protocol_snapshot.md")
        return self._compare_config(config, snapshot_path)
```

### 12.2 攻击评估系统

#### MembershipInferenceAttack

```python
class MembershipInferenceAttack(AttackBase):
    """
    成员推断攻击 (Membership Inference Attack)
    
    使用 Shadow Models 方法：
    1. 训练多个 shadow 模型模拟目标模型行为
    2. 收集 shadow 模型的 in/out 样本预测置信度
    3. 训练攻击分类器区分 member/non-member
    
    **Validates: R2.AC1**
    """
    
    attack_type = "membership_inference"
    
    def fit(self, train_data: DataLoader, shadow_data: DataLoader) -> None:
        """训练 shadow 模型和攻击分类器"""
        # 注意：shadow_data 必须与 train_data 严格分离
        pass
    
    def evaluate(self, test_data: DataLoader, target_model: nn.Module) -> Dict:
        """评估成员推断攻击"""
        return {"auc": auc, "advantage": advantage}
    
    def get_attack_success(self, metrics: Dict) -> float:
        """GC7: membership_inference -> AUC"""
        return metrics["auc"]
```

#### PropertyInferenceAttack

```python
class PropertyInferenceAttack(AttackBase):
    """
    属性推断攻击 (Property Inference Attack)
    
    推断训练数据集的群体属性分布（如 race/gender 比例）
    
    **Validates: R2.AC2**
    """
    
    attack_type = "property_inference"
    
    def evaluate(self, target_embeddings: torch.Tensor, 
                 true_distribution: Dict[str, Dict]) -> Dict:
        """评估属性推断攻击"""
        return {"auc": auc, "kl_divergence": kl_div}
    
    def get_attack_success(self, metrics: Dict) -> float:
        """GC7: property_inference -> AUC"""
        return metrics["auc"]
```

### 12.3 C-view 安全评估

#### NISTTestRunner

```python
class NISTTestRunner:
    """
    NIST SP800-22 随机性测试运行器
    
    **Validates: R3.AC1, R3.AC2**
    """
    
    REQUIRED_TESTS = [
        "frequency", "block_frequency", "runs", "longest_run",
        "fft", "serial", "approximate_entropy"
    ]
    
    MIN_BITS = {
        "frequency": 100, "block_frequency": 100, "runs": 100,
        "longest_run": 128, "fft": 1000, "serial": 1000,
        "approximate_entropy": 1000
    }
    
    def run_tests(self, chunks: List[CipherChunk]) -> Dict[str, Dict]:
        """
        运行 NIST 测试
        
        如果 bitstream 不足，按 SHA256(image_id) 哈希排序拼接
        """
        # 1. 按哈希排序
        sorted_chunks = sorted(chunks, key=lambda c: hashlib.sha256(c.image_id.encode()).hexdigest())
        # 2. 拼接密文
        concatenated = b"".join(c.ciphertext for c in sorted_chunks)
        # 3. 运行测试
        return results
```

#### AvalancheEffectTester

```python
class AvalancheEffectTester:
    """
    雪崩效应测试器
    
    测试三类 flip：key/nonce/plaintext
    期望翻转率在 45%-55%
    
    **Validates: R3.AC3**
    """
    
    FLIP_TYPES = ["key", "nonce", "plaintext"]
    EXPECTED_RANGE = (0.45, 0.55)
    
    def test_all(self, test_images: List[torch.Tensor]) -> Dict[str, Dict]:
        """测试所有 flip 类型"""
        pass
```

#### TamperReplayTester

```python
class TamperReplayTester:
    """
    Tamper/Replay 抗性测试器
    
    **Validates: R3.AC4, R3.AC5, R3.AC6**
    """
    
    TAMPER_TYPES = ["ciphertext", "tag", "aad"]
    N_TESTS_PER_TYPE = 200
    
    def test_tamper(self, test_samples: List[Dict]) -> Dict[str, Dict]:
        """测试 tamper 抗性，期望 fail_rate ≥ 99%"""
        pass
```

### 12.4 训练策略系统

```python
class TrainingModeManager:
    """
    训练模式管理器
    
    支持 4 种训练模式：P2P/P2Z/Z2Z/Mix2Z
    
    **Validates: R4**
    """
    
    TRAINING_MODES = ["P2P", "P2Z", "Z2Z", "Mix2Z"]
    
    def get_dataloader(self, mode: str, split: str, batch_size: int) -> DataLoader:
        """
        获取指定模式的 DataLoader
        
        C-view guard: 训练时永不返回 c_view
        """
        pass

class UtilityThresholdChecker:
    """
    效用门槛检查器
    
    **Validates: R4.AC3, R4.AC4, R4.AC8**
    """
    
    THRESHOLDS = {
        0.3: 0.75,  # privacy_level=0.3 时需达到 75% P2P 性能
        0.5: 0.65,  # privacy_level=0.5 时需达到 65% P2P 性能
    }
    
    def check(self, metrics: Dict, privacy_level: float) -> Tuple[bool, Dict]:
        """检查是否达到门槛"""
        pass
```

### 12.5 统计引擎

```python
class StatisticsEngine:
    """
    统计引擎
    
    支持：bootstrap CI、BH-FDR 多重比较校正、family_id 生成
    
    **Validates: R10**
    """
    
    def __init__(self, n_boot: int = 1000, alpha: float = 0.05):
        self.n_boot = n_boot
        self.alpha = alpha
        
    def compute_ci(self, values: np.ndarray) -> Tuple[float, float, float]:
        """计算 bootstrap 置信区间"""
        pass
    
    def multiple_comparison_correction(self, p_values: List[float]) -> List[float]:
        """BH-FDR 多重比较校正"""
        pass
    
    @staticmethod
    def generate_family_id(dataset: str, task: str, 
                           metric_name: str, privacy_level: float) -> str:
        """生成 family_id (GC9)"""
        key = f"{dataset}|{task}|{metric_name}|{privacy_level}"
        return hashlib.sha1(key.encode()).hexdigest()[:10]
```

### 12.6 图表系统

```python
class FigureSpecs:
    """图表规格（冻结）"""
    
    SINGLE_COLUMN_WIDTH = 3.5  # inches
    DOUBLE_COLUMN_WIDTH = 7.0  # inches
    DPI = 300
    FONT_PRIORITY = ["Arial", "Helvetica", "DejaVu Sans"]
    
    FIGURE_CONFIGS = {
        "fig_utility_curve": {"width": 7.0, "height": 4.0, "source_csv": "utility_metrics.csv"},
        "fig_attack_curves": {"width": 7.0, "height": 5.0, "source_csv": "attack_metrics.csv"},
        "fig_pareto_frontier": {"width": 3.5, "height": 3.5, "source_csv": ["utility_metrics.csv", "attack_metrics.csv"]},
        "fig_causal_ate_cate": {"width": 7.0, "height": 4.0, "source_csv": "causal_effects.csv"},
        "fig_cview_security_summary": {"width": 3.5, "height": 4.0, "source_csv": "security_metrics_cview.csv"},
        "fig_ablation_summary": {"width": 7.0, "height": 4.0, "source_csv": "ablation.csv"},
        "fig_efficiency": {"width": 3.5, "height": 3.5, "source_csv": "efficiency.csv"},
        "fig_robustness": {"width": 7.0, "height": 4.0, "source_csv": "robustness_metrics.csv"},
    }
```

---

## 13. 数据模型 (Data Models)

### 13.1 运行元数据

```python
@dataclass
class RunMetadata:
    """运行元数据"""
    config: Dict[str, Any]
    git_commit: str
    seed: int
    env: Dict[str, str]
    dataset_manifest_hash: str
    protocol_version: str
    hardware: HardwareInfo
    
@dataclass
class HardwareInfo:
    """硬件信息"""
    gpu_model: str
    gpu_count: int
    cpu_model: str
    cpu_cores: int
    memory_gb: float
    batch_size: int
    n_runs: int  # 效率测量重复次数
```

### 13.2 攻击指标行

```python
@dataclass
class AttackMetricsRow:
    """attack_metrics.csv 行数据"""
    dataset: str
    task: str
    method: str
    training_mode: str
    attack_type: str  # 5 types
    threat_level: str  # A0/A1/A2
    privacy_level: float
    seed: int
    attack_success: float  # 统一指标（GC7）
    metric_name: str
    metric_value: float
    ci_low: float
    ci_high: float
    attacker_strength: Optional[str]  # lite/full
    status: str  # success/failed
    stat_method: str
    n_boot: int
    family_id: str
    alpha: float
```

---

## 14. 正确性属性 (Correctness Properties)

*正确性属性是系统在所有有效执行中都应保持为真的特征或行为——本质上是关于系统应该做什么的形式化声明。属性是人类可读规范与机器可验证正确性保证之间的桥梁。*

### 核心属性（方法相关）

#### 属性 1：运行目录结构完整性
*对于任意*完成的实验运行，运行目录必须包含所有必需的子目录（meta/、tables/、figures/、logs/、reports/）以及每个子目录内的所有必需文件。
**验证需求：Requirements 1.7, 7.1, 7.2, 7.3, 7.4, 7.5**

#### 属性 2：CSV Schema 合规性
*对于任意*生成的 CSV 文件，RESULTS_SCHEMA.md 中定义的所有必需字段必须存在，且类型正确、值有效（例如 privacy_level ∈ {0.0, 0.3, 0.5, 0.7, 1.0}，threat_level ∈ {A0, A1, A2}）。
**验证需求：Requirements 1.3, 2.6, 2.9, GC6, GC7**

#### 属性 3：攻击成功率映射一致性
*对于任意*攻击评估结果，attack_success 字段必须按照 GC7 映射表计算（face_verification→TAR@FAR=1e-3，attribute_inference→AUC，membership_inference→AUC，property_inference→AUC，reconstruction→identity_similarity）。
**验证需求：Requirements 2.9, GC7**

#### 属性 4：C-view 安全测试完整性
*对于任意* C-view 安全评估，输出必须包含：(a) 至少 7 项 NIST SP800-22 测试结果并记录 nist_bits，(b) 所有 3 种 flip 类型的雪崩测试结果，(c) 所有 3 种类型的 tamper 测试结果且 fail_rate ≥ 99%。
**验证需求：Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

#### 属性 5：NIST 比特流充足性
*对于任意* NIST 子测试，如果初始比特流不足，系统必须按 SHA256(image_id) 哈希顺序自动拼接密文直到满足最小长度要求；如果仍然不足，测试必须失败并记录原因。
**验证需求：Requirements 3.2**

#### 属性 6：训练模式数据隔离
*对于任意* Z2Z 或 Mix2Z 模式的训练 DataLoader，请求 c_view 数据必须抛出 ViewAccessError；训练时只允许 plaintext 和 z_view。
**验证需求：Requirements 4.1, 4.2, 11.3, 11.6**

### 工程属性（验证相关）

#### 属性 7：效用门槛计算
*对于任意*效用评估，relative_performance 字段必须等于 metric_value / P2P_mean，且 relative_to 必须为 "P2P_mean"；如果 relative_performance < 门槛，必须生成失败分析。
**验证需求：Requirements 4.3, 4.4, 4.6, 4.8**

#### 属性 8：消融实验目录完整性
*对于任意*消融实验运行，冻结的消融目录（A1-A12）中的所有 12 项必须执行，且每项必须同时产出效用和攻击指标。
**验证需求：Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

#### 属性 9：图表可复现性
*对于任意*生成的图表，使用 make_figures.py 从源 CSV 重新生成必须产出字节相同的 PNG 文件（通过 figure_manifest.json 中的 SHA256 哈希验证）。
**验证需求：Requirements 6.1, 6.7, 6.8**

#### 属性 10：覆盖度门槛强制执行
*对于任意* CSV 文件，覆盖度（实际组合数 / 期望组合数）必须 ≥ 98%；否则 validate_run 必须失败并输出 missing_matrix.csv 列出所有缺失组合。
**验证需求：Requirements 7.6, 7.8, GC4**

#### 属性 11：统计严谨性完整性
*对于任意* CSV 中的指标行，以下统计字段必须存在且有效：stat_method、n_boot（≥500）、ci_low、ci_high、alpha、family_id（按 GC9 计算）。
**验证需求：Requirements 10.1, 10.4, 10.5, 10.6, GC3, GC9**

#### 属性 12：数据泄漏防护
*对于任意* manifest，train/val/test 分割之间必须零 ID 重叠；对于任意攻击评估，shadow 分割必须与评估样本严格分离。
**验证需求：Requirements 11.1, 11.2, 11.5**

#### 属性 13：协议版本一致性
*对于任意*运行，meta/protocol_version.txt 中的 protocol_version 必须与 schema_version 匹配，且 protocol_snapshot.md 必须与 meta/config.yaml 一致。
**验证需求：Requirements 1.7, 1.8, GC2**

#### 属性 14：A2 攻击强制存在
*对于任意*完整的攻击评估，attack_metrics.csv 必须包含至少一条 threat_level=A2 的记录；缺少 A2 记录必须导致 hard fail。
**验证需求：Requirements 16.1, 16.3, 16.5**

#### 属性 15：基线对比完整性
*对于任意*基线对比，baseline_comparison.csv 必须包含至少 2 个基线（InstaHide + P3）的记录。
**验证需求：Requirements 8.1, 8.2, 8.6, 8.7**

#### 属性 16：Family ID 确定性
*对于任意*两个具有相同（dataset, task, metric_name, privacy_level）元组的 CSV 行，family_id 必须相同且等于 sha1(f"{dataset}|{task}|{metric_name}|{privacy_level}")[:10]。
**验证需求：Requirements 10.5, GC9**

---

## 15. 错误处理 (Error Handling)

### 15.1 协议与 Schema 错误

| 错误类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| `ProtocolVersionMismatchError` | protocol_version != schema_version | hard fail，记录到 errors.log |
| `SchemaValidationError` | CSV 缺少必需字段 | hard fail，列出缺失字段 |
| `PrivacyLevelError` | privacy_level 不在 {0.0, 0.3, 0.5, 0.7, 1.0} 中 | hard fail |
| `ThreatLevelError` | threat_level 不在 {A0, A1, A2} 中 | hard fail |

### 15.2 攻击评估错误

| 错误类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| `ShadowSplitLeakageError` | Shadow 分割与评估集重叠 | hard fail |
| `A2AttackMissingError` | attack_metrics.csv 中无 A2 记录 | hard fail (GC10) |
| `AttackTrainingFailure` | 攻击模型训练崩溃 | 写入 failure_modes.md，status=failed |
| `AttackerDegradeNotAllowed` | 对 A2/tamper/NIST 尝试降级 | hard fail (GC10) |

### 15.3 安全评估错误

| 错误类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| `NISTBitstreamInsufficientError` | 拼接后比特流仍 < 最小值 | hard fail，记录原因 |
| `TamperFailRateLowError` | tamper_fail_rate < 99% | hard fail |
| `AvalancheOutOfRangeWarning` | flip_rate 不在 [0.45, 0.55] | 警告，记录到报告 |

### 15.4 数据流水线错误

| 错误类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| `SplitLeakageError` | train/val/test 之间 ID 重叠 | hard fail，输出 split_leakage_report.md |
| `CViewAccessError` | 训练 DataLoader 请求 C-view | hard fail |
| `CoverageGateError` | 覆盖度 < 98% | hard fail，输出 missing_matrix.csv |

---

## 16. 测试策略 (Testing Strategy)

### 16.1 属性测试框架

使用 **Hypothesis** 库进行属性测试，配置每个属性测试运行至少 100 次迭代。

```python
from hypothesis import given, settings, strategies as st
import pytest

# Property 1: Run Directory Structure Completeness
@settings(max_examples=100)
@given(
    exp_name=st.text(min_size=1, max_size=20),
    run_id=st.text(min_size=1, max_size=10)
)
def test_run_directory_structure_completeness(exp_name, run_id):
    """
    **Feature: top-journal-experiment-suite, Property 1: Run Directory Structure Completeness**
    """
    run_dir = create_run_directory(exp_name, run_id)
    for subdir in ["meta", "tables", "figures", "logs", "reports"]:
        assert (run_dir / subdir).exists()

# Property 3: Attack Success Mapping Consistency
@settings(max_examples=100)
@given(
    attack_type=st.sampled_from(["face_verification", "attribute_inference", 
                                  "reconstruction", "membership_inference", "property_inference"]),
    raw_metrics=st.dictionaries(
        keys=st.sampled_from(["auc", "tar_at_far_1e3", "identity_similarity"]),
        values=st.floats(min_value=0.0, max_value=1.0),
        min_size=1
    )
)
def test_attack_success_mapping_consistency(attack_type, raw_metrics):
    """
    **Feature: top-journal-experiment-suite, Property 3: Attack Success Mapping Consistency**
    """
    attack = create_attack(attack_type)
    attack_success = attack.get_attack_success(raw_metrics)
    expected_metric = GC7_MAPPING[attack_type]
    if expected_metric in raw_metrics:
        assert attack_success == raw_metrics[expected_metric]

# Property 16: Family ID Determinism
@settings(max_examples=100)
@given(
    dataset=st.text(min_size=1, max_size=20),
    task=st.text(min_size=1, max_size=20),
    metric_name=st.text(min_size=1, max_size=20),
    privacy_level=st.sampled_from([0.0, 0.3, 0.5, 0.7, 1.0])
)
def test_family_id_determinism(dataset, task, metric_name, privacy_level):
    """
    **Feature: top-journal-experiment-suite, Property 16: Family ID Determinism**
    """
    family_id_1 = StatisticsEngine.generate_family_id(dataset, task, metric_name, privacy_level)
    family_id_2 = StatisticsEngine.generate_family_id(dataset, task, metric_name, privacy_level)
    assert family_id_1 == family_id_2
```

### 16.2 单元测试

每个组件的核心方法需要单元测试：
- `NISTTestRunner`: 各子测试的正确性
- `StatisticsEngine`: bootstrap CI 计算、BH-FDR 校正
- `ValidateRun`: 目录结构检查、Schema 验证
- `AttackBase` 子类: fit/evaluate 接口一致性

### 16.3 集成测试

- 端到端实验流水线测试（smoke_test 模式）
- 跨组件数据流测试
- CI 时间预算验证（< 20 min）

### 16.4 测试文件组织

```
tests/
├── test_protocol_manager.py        # Property 13
├── test_results_schema.py          # Property 2
├── test_attack_evaluator.py        # Property 3, 14
├── test_membership_inference.py    # R2.AC1
├── test_property_inference.py      # R2.AC2
├── test_adaptive_attacker.py       # R16
├── test_nist_runner.py             # Property 4, 5
├── test_avalanche_tester.py        # R3.AC3
├── test_tamper_replay.py           # R3.AC4, R3.AC5
├── test_training_mode.py           # Property 6, 7
├── test_ablation_runner.py         # Property 8
├── test_figure_generator.py        # Property 9
├── test_validate_run.py            # Property 1, 10
├── test_statistics_engine.py       # Property 11, 16
├── test_split_leakage.py           # Property 12
├── test_baseline_comparator.py     # Property 15
├── test_smoke_test.py              # R14.AC4
└── conftest.py                     # Shared fixtures
```

---

## 17. 任务分解预览 (Task Decomposition Preview)

本节预告下一阶段 task 的拆分方式，确保从 design 到 task 的无缝过渡。

### 17.1 Claim → Task → Output 映射

| Claim | 任务（下一阶段实现） | 输出物 | 优先级 |
|-------|---------------------|--------|--------|
| **C1** Pareto 优势 + 因果效应 | Baseline 对齐 + 因果效应估计 + Pareto 曲线生成 | fig_pareto_frontier.png, causal_effects.csv | P0 |
| **C2** A2 鲁棒 | 自适应攻击 pipeline + 强度契约实现 | fig_attack_curves.png, table_a2.csv | P0 |
| **C3** C-view 安全 | 安全目标验证 + NIST/Avalanche 诊断 | security_report.md, fig_cview_security.png | P0 |
| **C4** 可复现 | seed/schema/CI 集成 + 复现材料清单 | ARTIFACT_CHECKLIST.md, reproduce.sh | P1 |

### 17.2 核心任务分组

| 任务组 | 包含任务 | 依赖 | 估计工时 |
|--------|----------|------|----------|
| **T1: 协议与 Schema** | 协议管理器、Schema 验证器、版本控制、Nonce 管理器 | 无 | 2d |
| **T2: 攻击评估** | 5类攻击实现、A2 自适应攻击、attack_success 映射、强度契约 | T1 | 5d |
| **T3: 安全评估** | 安全目标验证、NIST 测试、Avalanche 测试、Tamper/Replay 测试 | T1 | 3d |
| **T4: 训练策略** | Z2Z/Mix2Z DataLoader、效用门槛检查 | T1 | 2d |
| **T5: 因果效应** | 干预网格、ATE/CATE 估计、预算优化求解 | T2, T4 | 3d |
| **T6: 消融实验** | 12项消融配置、消融运行器 | T2, T4 | 2d |
| **T7: 统计与验证** | bootstrap CI、BH-FDR、覆盖度检查、归一化 | T2, T3, T4 | 2d |
| **T8: 图表生成** | 8张主图、figure_manifest | T7 | 2d |
| **T9: CI 集成** | smoke_test、reproduce.sh、复现材料清单 | T8 | 1d |
| **T10: 基线对比** | 5个基线实现、基线矩阵验证 | T2, T4 | 3d |

### 17.3 属性 → 测试任务映射

| 属性 | 测试任务 | 测试类型 | 所属任务组 |
|------|----------|----------|------------|
| Property 1 | test_run_directory_structure | 属性测试 | T1 |
| Property 2 | test_csv_schema_compliance | 属性测试 | T1 |
| Property 3 | test_attack_success_mapping | 属性测试 | T2 |
| Property 4 | test_cview_security_completeness | 属性测试 | T3 |
| Property 6 | test_training_mode_isolation | 属性测试 | T4 |
| Property 10 | test_coverage_gate | 属性测试 | T7 |
| Property 14 | test_a2_mandatory | 属性测试 | T2 |
| Property 16 | test_family_id_determinism | 属性测试 | T7 |

### 17.4 新增任务（基于 8 个升级建议）

| 升级建议 | 对应任务 | 输出物 |
|----------|----------|--------|
| RQ3 安全定义驱动 | T3: 安全目标验证 | security_goals_report.md |
| Nonce 策略 | T1: Nonce 管理器 | nonce_log.json |
| A2 强度契约 | T2: 强度契约实现 | protocol_snapshot.md |
| 因果效应估计 | T5: 干预网格 + ATE/CATE | causal_effects.csv |
| attack_success 归一化 | T7: 归一化实现 | normalized_metrics.csv |
| 基线矩阵 | T10: 基线对比 | baseline_comparison.csv |
| 复现材料分离 | T9: 复现材料清单 | ARTIFACT_CHECKLIST.md |
| 三层结构 | 文档重组 | design.md (已完成) |

---

## 附录 A：术语表 (Glossary)

| 术语 | 定义 |
|------|------|
| **Z-view** | Utility View，可用于下游任务但隐私保护的视图 |
| **C-view** | Cryptographic View，强加密视图（AEAD 封装） |
| **attack_success** | 攻击成功率，越高表示攻击越成功、隐私越弱 |
| **privacy_protection** | 隐私保护强度，= 1 - normalized(attack_success) |
| **A0/A1/A2** | 三级威胁等级：黑盒/灰盒/白盒自适应 |
| **GC7** | 全局约束7：attack_success 映射表 |
| **GC9** | 全局约束9：family_id 计算规则 |
| **NIST SP800-22** | 美国国家标准与技术研究院随机性测试标准 |
| **Avalanche Effect** | 雪崩效应，输入微小变化导致输出大幅变化 |
| **BH-FDR** | Benjamini-Hochberg False Discovery Rate 校正 |
| **bootstrap CI** | 自助法置信区间 |

---

## 附录 B：文件结构 (File Structure)

```
results/{exp_name}/{run_id}/
├── meta/
│   ├── config.yaml
│   ├── git_commit.txt
│   ├── seed.txt
│   ├── env.txt
│   ├── dataset_manifest_hash.txt
│   ├── protocol_version.txt
│   └── hardware.json
├── tables/
│   ├── utility_metrics.csv
│   ├── attack_metrics.csv
│   ├── security_metrics_cview.csv
│   ├── causal_effects.csv
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
│   ├── experiment.log
│   └── errors.log
└── reports/
    ├── protocol_snapshot.md
    ├── figure_manifest.json
    ├── coverage_report.json
    └── missing_matrix.csv (if coverage < 98%)
```

---

## 附录 C：升级日志 (Upgrade Log)

### v2.1.1 (2024-12) - 最后一公里修补

| 修补项 | 修改内容 | 章节 |
|--------|----------|------|
| 1. ReplayCache 实现 | 明确抗重放的实现细节（key 改用 full_tag、生命周期、落盘策略） | §9.6.1.1 |
| 2. Nonce 派生元组 | 输入改为协议唯一元组（image_id, method, privacy_level, training_mode, purpose） | §5.5.3 |
| 3. 版本号一致性 | ProtocolManager 版本号更新为 2.1.1（§8.4 和 §12.1 统一） | §8.4, §12.1 |
| 4. §10.3 nonce 引用 | 删除旧版派生代码，改为引用 §5.5.3 NonceManager | §10.3 |
| 5. CI 例外条款 | 新增 smoke_test lite 模式，解决与 A2 强度契约的冲突 | §10.4 |

### v2.1.0 (2024-12) - 顶刊级升级

基于 8 个升级建议进行的修改：

| 升级项 | 修改内容 | 章节 |
|--------|----------|------|
| 1. RQ3/C4 安全定义驱动 | NIST/Avalanche 降级为诊断，安全目标（机密性/完整性/抗重放）作为主证据 | §1.1, §9.6 |
| 2. Nonce 策略 | 新增 §5.5 Nonce 策略：受控选择，明确唯一性约束 | §5.5 |
| 3. A2 强度契约 | 新增 §5.4 A2 强度契约，冻结攻击族谱、预算、聚合规则 | §5.4 |
| 4. 因果效应估计 | 新增 §9.2 两阶段方法（干预网格 + 效应估计 → 预算优化） | §9.2 |
| 5. attack_success 归一化 | 新增 §7.3 归一化定义，§7.4 汇总指标定义 | §7.3, §7.4 |
| 6. 基线矩阵 | 新增 §9.4 基线矩阵（冻结），明确每个 baseline 的适用范围 | §9.4 |
| 7. 复现材料分离 | 新增 §10.0 复现材料与证明分离，审稿人友好报告 | §10.0 |
| 8. 三层结构 | 文档重组为 Scientific/Protocol/Artifact 三层 | 全文 |

### v2.0.0 (2024-12) - 初始版本

- 完整的顶刊实验套件设计
- 16 个正确性属性
- 12 项消融实验

---

*文档版本：2.1.1*
*最后更新：2024-12*
*结构：Scientific Spec → Protocol Spec → Artifact Spec*
