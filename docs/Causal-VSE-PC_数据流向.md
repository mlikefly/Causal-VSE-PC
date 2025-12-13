# Causal-VSE-PC 数据流向详解

**文档版本**: 1.0  
**创建日期**: 2025年12月  
**核心创新**: 因果推断驱动的加密策略

---

## 一、总体数据流架构

### 1.1 完整流程图

```
输入阶段
├─ 原始图像 I: [B, C, H, W]
├─ 任务类型 T: {'classification', 'segmentation', 'detection'}
└─ 主密钥 K_master: bytes (用户密码派生)

    ↓

语义分析阶段 (U-Net)
├─ 输入: I
├─ 输出: 语义掩码 M_semantic: [B, 1, H, W] (值域[0,1])
└─ 组件: UNetSaliencyDetector (已有)

    ↓

因果分析阶段 (核心创新)
├─ 输入: M_semantic, T
├─ 因果图构建: X(Semantic) → Z(Privacy Budget) → Y(ML Performance)
├─ ATE计算: E[Y(1) - Y(0)] = 加密对ML性能的因果效应
├─ 输出: 因果报告 + 隐私预算建议
└─ 组件: CausalPrivacyAnalyzer (新设计)

    ↓

隐私预算分配阶段
├─ 输入: M_semantic, T, 因果分析结果
├─ 规则: 
│   ├─ 敏感区域(人脸): privacy = 1.0 (完全加密)
│   ├─ 任务相关区域: privacy = 0.3 (弱加密，保留语义)
│   └─ 背景区域: privacy = 0.0 (不加密)
├─ 输出: 隐私预算图 P(x,y): [B, 1, H, W]
└─ 组件: AdaptivePrivacyBudget (已有)

    ↓

密钥派生阶段
├─ 输入: K_master, I
├─ 派生过程:
│   ├─ Step 1: Image Hash = SHA256(I)
│   ├─ Step 2: Image Key = HMAC-SHA512(K_master, Image Hash)
│   ├─ Step 3: Chaos Key = Image Key[0:16] → (seed1, seed2) ∈ [0,1]
│   ├─ Step 4: Region Key = Image Key[16:48] (32 bytes)
│   └─ Step 5: Chaos Initial State = derive_from_key(seed1, seed2)
└─ 输出: chaos_key, region_key, chaos_state

    ↓

分层加密阶段

Layer 1: 空域混沌置乱与扩散
├─ 输入: I, chaos_key, P(x,y)
├─ 操作1: Arnold Cat Map (置乱/Scrambling)
│   ├─ 迭代次数: iterations = 5 (固定，避免STE问题)
│   └─ 变换公式: 
│       ├─ x' = (x + y) mod N
│       └─ y' = (x + 2*y) mod N
│
├─ 操作2: 5D超混沌扩散 (Diffusion)
│   ├─ 混沌系统: Sine-Coupling 5D Hyperchaotic Map
│   ├─ 动力学方程:
│   │   x_{i,t+1} = [μ·sin(π·x_{i,t}) 
│   │              + α·(x_{i+1,t} - x_{i-1,t})
│   │              + β·cos(π·x_{(i+2)%5,t})] mod 1.0
│   ├─ 参数: μ=0.99, α=0.4~0.5, β=0.15~0.25
│   ├─ 初始状态: 从chaos_key派生，5维状态 [x1, x2, x3, x4, x5]
│   ├─ 预热: 200步 (进入吸引子)
│   └─ 输出: 伪随机序列 mask: [B, H*W]
│
├─ 操作3: 像素值混淆 (Substitution)
│   ├─ 公式: C = (P + mask * strength) mod 1.0
│   ├─ strength: 从P(x,y)映射，strength = 0.5 + 0.5 * P(x,y)
│   └─ 输出: I_scrambled: [B, C, H, W]
│
└─ 组件: StandardChaoticCipher (已有，使用5D混沌系统)

    ↓

Layer 2: 频域语义控制
├─ 输入: I_scrambled, region_key, P(x,y)
├─ 操作1: FFT变换 (比DWT快)
│   ├─ 变换: F = FFT(I_scrambled)
│   ├─ 分解: F = F_magnitude * exp(i·F_phase)
│   └─ 频域表示: (magnitude, phase)
│
├─ 操作2: 语义保留扰动
│   ├─ 低频(语义区): 弱扰动
│   │   ├─ LL子带 (或FFT中心): α_low = 0.05 + 0.15 * P(x,y)
│   │   └─ 扰动: LL_enc = LL + perturbation * α_low
│   │
│   ├─ 高频(细节区): 强扰动
│   │   ├─ LH/HL/HH子带 (或FFT边缘): α_high = 0.30 + 0.70 * P(x,y)
│   │   └─ 扰动: HH_enc = HH + perturbation * α_high
│   │
│   └─ 扰动生成: 使用region_key生成伪随机扰动
│
├─ 操作3: IFFT逆变换
│   └─ 输出: I_encrypted: [B, C, H, W]
│
└─ 组件: FrequencySemanticCipherOptimized (已有，支持privacy_map)

    ↓

可验证性证明阶段
├─ 输入: I, I_encrypted, enc_params
├─ 证明生成:
│   ├─ Hash Commitment: commit = SHA256(I || I_encrypted || enc_params)
│   ├─ HMAC标签: tag = HMAC-SHA256(K_master, commit)
│   └─ 证明: π = (commit, tag)
│
└─ 组件: VerifiableEncryption (已有，轻量级Hash Commitment)

    ↓

输出阶段
├─ 加密图像: I_encrypted: [B, C, H, W]
├─ 加密信息: enc_info (包含所有参数)
├─ 可验证性证明: proof = (commit, tag)
├─ 因果分析报告: causal_report (解释为什么这样加密)
└─ 隐私预算图: P(x,y) (可视化用)
```

---

## 二、详细数据流：密钥派生

### 2.1 密钥派生树

```
Master Key K_master (用户密码)
│
├─ PBKDF2-SHA512(password, salt, iterations=100000)
│  └─ 输出: 64 bytes
│
└─ Image Key K_image (每张图像唯一)
   │
   ├─ SHA256(Image) → Image Hash (32 bytes)
   │
   ├─ HMAC-SHA512(K_master, Image Hash) → K_image (64 bytes)
   │
   └─ 分解为:
      │
      ├─ Chaos Key (用于Layer 1)
      │  ├─ K_image[0:8] → seed1 (float32, 范围[0,1])
      │  ├─ K_image[8:16] → seed2 (float32, 范围[0,1])
      │  └─ chaos_key = [seed1, seed2] (torch.Tensor, [B, 2])
      │
      ├─ Region Key (用于Layer 2)
      │  └─ K_image[16:48] → region_key (32 bytes)
      │
      └─ Chaos Initial State (5D混沌系统)
         ├─ 从seed1, seed2派生5个初始状态
         ├─ x0[i] = sin(seed1 * freq[i] + seed2 * phase[i])
         └─ chaos_state = [x1, x2, x3, x4, x5] (5维)
```

### 2.2 密钥安全性

- **主密钥**: 用户密码，PBKDF2派生，100000次迭代，抗暴力破解
- **图像密钥**: 每张图像唯一，HMAC派生，抗已知明文攻击
- **密钥分离**: Layer 1和Layer 2使用不同密钥，安全性分层

---

## 三、详细数据流：加密操作

### 3.1 Layer 1: 空域混沌加密

```
输入: I [B, C, H, W], chaos_key [B, 2], P(x,y) [B, 1, H, W]

Step 1: Arnold Cat Map 置乱
├─ 计算索引映射: idx_map = arnold_map(H, iterations=5)
├─ 置乱操作: I_scrambled = I[idx_map]
└─ 输出: I_scrambled [B, C, H, W]

Step 2: 5D超混沌序列生成
├─ 从chaos_key派生初始状态: state0 = derive_5d_state(chaos_key)
├─ 预热200步: state = warmup(state0, steps=200)
├─ 生成序列: 
│   ├─ for t in range(H*W):
│   │   state = 5d_hyperchaotic_iterate(state, μ, α, β)
│   │   seq[t] = sum(state) mod 1.0
│   └─ 输出: seq [B, H*W]
└─ 重塑: mask = seq.view([B, 1, H, W])

Step 3: 像素值混淆
├─ 计算strength: strength = 0.5 + 0.5 * P(x,y) [B, 1, H, W]
├─ 混淆公式: C = (I_scrambled + mask * strength) mod 1.0
└─ 输出: I_layer1 [B, C, H, W]
```

### 3.2 Layer 2: 频域语义控制

```
输入: I_layer1 [B, C, H, W], region_key (32 bytes), P(x,y) [B, 1, H, W]

Step 1: FFT变换
├─ 变换: F = FFT(I_layer1)  # 复数域
├─ 分解: 
│   ├─ magnitude = |F|
│   └─ phase = angle(F)
└─ 输出: F [B, C, H, W] (复数)

Step 2: 区域级隐私预算下采样
├─ 频域空间: H_freq = H // 2, W_freq = W // 2
├─ 下采样: P_freq = downsample(P(x,y), size=(H_freq, W_freq))
└─ 输出: P_freq [B, 1, H_freq, W_freq]

Step 3: 语义保留扰动
├─ 低频区 (中心区域):
│   ├─ 扰动强度: α_low = 0.05 + 0.15 * P_freq
│   ├─ 生成扰动: pert_low = random_perturbation(region_key, size=中心)
│   └─ 扰动: F_low_enc = F_low + pert_low * α_low
│
├─ 高频区 (边缘区域):
│   ├─ 扰动强度: α_high = 0.30 + 0.70 * P_freq
│   ├─ 生成扰动: pert_high = random_perturbation(region_key, size=边缘)
│   └─ 扰动: F_high_enc = F_high + pert_high * α_high
│
└─ 输出: F_enc [B, C, H, W] (复数)

Step 4: IFFT逆变换
├─ 变换: I_encrypted = IFFT(F_enc)
└─ 输出: I_encrypted [B, C, H, W]
```

---

## 四、详细数据流：因果分析

### 4.1 因果图结构

```
因果图 (Structural Causal Model, SCM):

X (Semantic Region) ──┐
                      ├─→ Z (Privacy Budget Allocation)
T (Task Type) ────────┤
                      │
                      └─→ Y (ML Performance)

其中:
- X: 语义区域类型 (敏感/任务相关/背景)
- T: 任务类型 (分类/分割/检测)
- Z: 隐私预算分配策略 (加密强度)
- Y: ML任务性能 (准确率/mIoU/mAP)
```

### 4.2 因果效应计算

```
输入: M_semantic [B, 1, H, W], P(x,y) [B, 1, H, W], T (str)

Step 1: 构建因果图
├─ 节点: X (语义), T (任务), Z (隐私预算), Y (性能)
├─ 边: X → Z, T → Z, Z → Y
└─ 输出: Causal Graph G

Step 2: 计算ATE (Average Treatment Effect)
├─ 干预: do(Z = high_privacy) vs do(Z = low_privacy)
├─ 反事实推理:
│   ├─ Y(high) = E[Y | do(Z=high), X, T]
│   └─ Y(low) = E[Y | do(Z=low), X, T]
├─ ATE计算: ATE = E[Y(high) - Y(low)]
└─ 输出: ATE值 (隐私对性能的因果影响)

Step 3: 条件ATE (CATE)
├─ CATE(X=sensitive) = E[Y(high) - Y(low) | X=sensitive]
├─ CATE(X=task_relevant) = E[Y(high) - Y(low) | X=task_relevant]
└─ 输出: 不同区域的因果效应

Step 4: 生成因果解释
├─ 自然语言解释: "为什么对人脸区域使用强加密？"
├─ 因果链条: "人脸区域(敏感) → 高隐私预算(1.0) → 识别率下降90% → 任务准确率仅下降5%"
└─ 输出: causal_report (可解释的加密策略)
```

---

## 五、密文域ML推理数据流

### 5.1 ML推理流程

```
输入: I_encrypted [B, C, H, W], ML_model, task_type

Step 1: 直接推理 (无需解密)
├─ 输入: I_encrypted (加密图像)
├─ 模型: ML_model (标准CNN，无需修改)
├─ 前向传播: predictions = ML_model(I_encrypted)
└─ 输出: predictions (分类/分割/检测结果)

原理:
- ML模型主要依赖低频特征（语义信息）
- Layer 2保留了低频语义（α_low = 0.05~0.20）
- 因此ML模型仍能提取有效特征
```

### 5.2 性能对比

```
原始图像 → ML推理: accuracy = 95%
加密图像 → ML推理: accuracy = 80% (目标)

下降原因:
- 高频细节丢失 (不影响分类)
- 低频语义保留 (关键特征保留)
```

---

## 六、解密流程

### 6.1 解密步骤

```
输入: I_encrypted [B, C, H, W], enc_info (包含所有参数), K_master

Step 1: 密钥恢复
├─ 从enc_info提取Image Hash
├─ 重新派生: K_image = HMAC-SHA512(K_master, Image Hash)
├─ 恢复: chaos_key, region_key
└─ 输出: 密钥对

Step 2: Layer 2逆变换 (频域)
├─ FFT: F_enc = FFT(I_encrypted)
├─ 逆扰动: F = F_enc - perturbation (使用region_key和enc_info中的参数)
├─ IFFT: I_layer1 = IFFT(F)
└─ 输出: I_layer1

Step 3: Layer 1逆变换 (空域)
├─ 逆扩散: 
│   ├─ 重新生成chaos序列: mask = generate_chaos(chaos_key, H*W)
│   ├─ 恢复: I_scrambled = (I_layer1 - mask * strength) mod 1.0
│   └─ 输出: I_scrambled
│
├─ 逆置乱:
│   ├─ 计算逆Arnold映射: inv_idx_map = inverse_arnold_map(H, iterations=5)
│   ├─ 恢复: I = I_scrambled[inv_idx_map]
│   └─ 输出: I (原始图像)
│
└─ 输出: I [B, C, H, W]
```

---

## 七、数据流关键点总结

### 7.1 加密方法总结

| 操作类型 | 具体方法 | 对称/非对称 | 组件 |
|---------|---------|-----------|------|
| **置乱** | Arnold Cat Map | 对称 | StandardChaoticCipher |
| **扩散** | 5D超混沌系统 | 对称 | StandardChaoticCipher |
| **替换** | 像素值混淆 (mod 1.0) | 对称 | StandardChaoticCipher |
| **语义控制** | FFT频域扰动 | 对称 | FrequencySemanticCipherOptimized |
| **可验证性** | Hash Commitment + HMAC | 对称 (MAC) | VerifiableEncryption |

**结论**: 全部使用**对称加密**，满足实时性需求（>10 FPS）

### 7.2 密钥空间

```
主密钥空间: 2^512 (HMAC-SHA512输出)
图像密钥空间: 2^512 (HMAC派生)
混沌密钥空间: 2^128 (两个float32种子)
区域密钥空间: 2^256 (32 bytes)

总密钥空间: 2^512 (足够大，抗暴力破解)
```

### 7.3 安全性与效率权衡

- **安全性**: 5D超混沌系统 (高复杂性)，HMAC认证 (完整性)
- **效率**: 对称加密 (快速)，FFT优化 (GPU加速)
- **语义保留**: 频域分层控制 (任务可用性)

---

## 八、数据流可视化

### 8.1 维度变化追踪

```
I: [B, C, H, W]
    ↓
M_semantic: [B, 1, H, W]  (U-Net输出)
    ↓
P(x,y): [B, 1, H, W]  (隐私预算)
    ↓
chaos_key: [B, 2]  (密钥派生)
    ↓
mask: [B, 1, H, W]  (混沌序列)
    ↓
I_scrambled: [B, C, H, W]  (Layer 1输出)
    ↓
F_enc: [B, C, H, W] (复数)  (FFT)
    ↓
I_encrypted: [B, C, H, W]  (最终输出)
```

### 8.2 内存占用估算

```
输入图像: B * C * H * W * 4 bytes (float32)
语义掩码: B * 1 * H * W * 4 bytes
隐私预算: B * 1 * H * W * 4 bytes
混沌序列: B * H * W * 4 bytes
频域表示: B * C * H * W * 8 bytes (复数)

总内存 (B=1, C=3, H=W=256): 
≈ 1.5 MB (原始) + 2.5 MB (加密中间结果) + 1 MB (语义分析)
≈ 5 MB (单张图像处理)
```

---

## 九、5D混沌系统使用说明

### 9.1 系统实现位置

```
核心实现：
├─ src/core/chaos_systems.py
│  └─ ChaosSystem (5维超混沌系统)
│     ├─ complexity='high' → 5维
│     ├─ 动力学方程: Chen-Logistic混合
│     └─ generate_trajectory() → 生成序列
│
└─ src/core/chaotic_encryptor.py
   └─ StandardChaoticCipher._hyper_chaotic_map()
      ├─ 5D Sine-Coupling超混沌映射
      ├─ 双向环形耦合
      └─ 非线性混合输出
```

### 9.2 5D混沌系统特性

```
系统类型: 5维超混沌系统 (5D Hyper-chaotic System)
参数设置:
├─ 维度: 5维状态 [x1, x2, x3, x4, x5]
├─ 混沌特性: 多正Lyapunov指数 (LE > 0.1)
├─ 耦合方式: 双向环形耦合 + Sine/Logistic混合
└─ 输出: 伪随机序列 [B, H*W]

优势:
✅ 高复杂性: 5维状态空间，抗攻击能力强
✅ 伪随机性好: 通过统计测试
✅ GPU加速: 向量化实现，高效
✅ 已集成: 可直接使用
```

### 9.3 在加密流程中的使用

```
Layer 1: 空域混沌扩散
├─ 输入: chaos_key [B, 2] (seed1, seed2)
├─ 派生: 5维初始状态
│   ├─ x0[i] = sin(seed1 * freq[i] + seed2 * phase[i])
│   └─ 5个独立初值
├─ 预热: 200步 (进入吸引子)
├─ 迭代: 
│   ├─ Sine-Coupling方程
│   ├─ 双向耦合: x[i+1] - x[i-1]
│   └─ 非线性项: cos(π·x[(i+2)%5])
└─ 输出: 伪随机mask [B, H*W]

用途:
✅ 像素值混淆 (Diffusion)
✅ 密钥派生的一部分
✅ 保证加密的随机性
```

### 9.4 可用性评估

```
✅ 完全可用
├─ 代码完整: src/core/chaos_systems.py
├─ 集成良好: 已在StandardChaoticCipher中使用
├─ 性能良好: GPU加速，满足实时需求
└─ 安全性高: 5维超混沌，复杂度足够

建议:
✅ 直接使用现有实现
✅ 无需修改（已优化）
✅ 注意参数固定（避免梯度问题）
```

---

## 十、数据流总结

### 10.1 完整数据流路径

```
输入图像 I
    ↓
U-Net语义分析 → M_semantic
    ↓
因果分析 → causal_report + P_map建议
    ↓
隐私预算分配 → P(x,y)
    ↓
密钥派生 → chaos_key, region_key
    ↓
Layer 1 (5D混沌) → I_scrambled
    ↓
Layer 2 (频域) → I_encrypted
    ↓
可验证性证明 → proof
    ↓
密文域ML推理 → predictions
```

### 10.2 关键数据转换

| 阶段 | 输入维度 | 输出维度 | 数据类型 | 说明 |
|------|---------|---------|---------|------|
| 原始图像 | [B, C, H, W] | [B, C, H, W] | float32 [0,1] | 输入 |
| 语义掩码 | [B, C, H, W] | [B, 1, H, W] | float32 [0,1] | U-Net输出 |
| 隐私预算 | [B, 1, H, W] | [B, 1, H, W] | float32 [0,1] | 规则分配 |
| 混沌密钥 | - | [B, 2] | float32 [0,1] | 密钥派生 |
| 混沌序列 | [B, 2] | [B, H*W] | float32 [0,1] | 5D混沌输出 |
| 加密图像 | [B, C, H, W] | [B, C, H, W] | float32 [0,1] | 最终输出 |

### 10.3 性能瓶颈分析

```
1. 语义分析: ~5ms (GPU, 可接受)
2. 5D混沌序列生成: ~20ms (GPU, 可优化)
3. FFT变换: ~10ms (GPU, 已优化)
4. ML推理: ~50ms (取决于模型)
5. 可验证性: ~10ms (CPU, 可接受)

总计: ~95ms (约10 FPS)
目标: < 100ms (10+ FPS) ✅ 接近目标
```

---

## 十一、安全性与隐私保障

### 11.1 密钥管理

```
✅ 分层密钥系统
├─ 主密钥: PBKDF2派生 (100000次迭代)
├─ 图像密钥: HMAC派生 (每张图像唯一)
├─ 区域密钥: 32 bytes (频域加密)
└─ 混沌密钥: 2个float32种子 (5D混沌)

✅ 密钥分离
├─ Layer 1和Layer 2使用不同密钥
├─ 降低单点泄露风险
└─ 支持密钥轮换
```

### 11.2 加密强度

```
✅ 5D超混沌系统
├─ 高复杂度: 5维状态空间
├─ 强随机性: 通过统计测试
└─ 抗攻击: 暴力破解不可行

✅ 频域语义控制
├─ 低频保留: 任务可用性
├─ 高频破坏: 隐私保护
└─ 自适应: 根据隐私预算调整
```

---

**相关文档**:
- [返回总览](Causal-VSE-PC_项目总览.md)
- [数据集分析](Causal-VSE-PC_数据集分析与使用.md)
- [工作流程](Causal-VSE-PC_工作流程.md)
- [理论证明](Causal-VSE-PC_理论证明.md)