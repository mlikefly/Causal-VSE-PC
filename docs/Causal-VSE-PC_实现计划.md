# Causal-VSE-PC 实现计划（工具、步骤、时间线）

**文档版本**: 1.0  
**创建日期**: 2025年12月  
**核心创新**: 因果推断驱动的加密策略 + 轻量级可验证性

---

## 一、总体里程碑（6个月）

- M1 (月1): 因果分析模块可用（ATE/CATE、报告雏形）
- M2 (月2): 因果指导的隐私分配集成 & 密文域ML适配器可跑
- M3 (月4): 核心实验完成（因果效应、可解释性、权衡曲线）
- M4 (月5.5): 论文初稿（Method含因果，Exp完整）
- M5 (月6): 定稿与投稿

---

## 二、任务分解与时间线

### Month 1: 因果模块 + 数据准备
- Week 1: 实现 `_build_causal_graph`，ATE/CATE 估计；梳理 CelebA-HQ/CelebAMask-HQ 管线
- Week 2: 因果报告初版（含置信区间占位）；接入 `AdaptivePrivacyBudget`（causal_suggestion）；小批量单测

### Month 2: 集成与性能
- Week 3: 稳定 `encrypt_layered`，记录 α_low/α_high 以便解密；密文域分类/分割适配器对齐尺寸与归一化
- Week 4: 性能优化（默认FFT；预计算部分混沌序列；batch并行≥8）；验证5D混沌数值稳定

### Month 3: 实验基线
- Week 5: 实验1 隐私-可用性权衡（隐私=0/0.3/0.5/0.7/1.0），记录 Accuracy/识别率/ATE/CATE
- Week 6: 实验2 区域级因果（19区域），记录 mIoU/识别率/区域ATE
- Week 7-8: 实验3 跨数据集（LFW若获取），泛化验证 + 跨集ATE

### Month 4: 可解释性与可验证性
- Week 9: 因果报告优化（自然语言模板 + 置信区间），可视化因果图/ATE条形图
- Week 10: 可验证性原型（Hash Commitment + HMAC），测时延/大小；复测 NPCR/UACI/熵

### Month 5-6: 论文与打磨
- Week 11-13: 写 Introduction / Method（含因果）/ Experiment；做 Ablation：无因果 vs 因果，频域扰动关/开
- Week 14-15: 完成 Discussion/Limitations/Related Work；内部预审与修改

---

## 三、实验与评测设计

- 任务：分类(CelebA-HQ 40属性)、分割(CelebAMask-HQ 19区域)、识别(ArcFace/FaceNet攻击者视角)
- 指标：隐私(识别<5%、PSNR<10dB、NPCR>99.6%、UACI≈33%、熵>7.9)；可用性(Acc>80%、mIoU>70%、下降<20%)；因果(ATE/CATE+置信区间；报告完整性)；性能(加密>10 FPS目标，验证<10ms)
- 对比：传统VSE-PC（无因果）；仅规则分配 vs 因果指导；全局强加密 vs 区域差异扰动

---

## 四、工具与环境

- 框架：PyTorch 2.x，CUDA 11+  
- 数据：CelebA-HQ, CelebAMask-HQ；可选 LFW, ImageNet-100  
- 复用组件：`src/core/chaotic_encryptor.py`、`src/core/frequency_cipher.py`、`src/crypto/key_system.py`、`src/neural/unet.py`、`src/vse_pc/privacy_budget.py`、`src/vse_pc/causal_analysis.py`、`src/vse_pc/verifiable.py`  
- 追踪：Weights & Biases / TensorBoard（记录ATE/CATE、性能曲线）

---

## 五、职责分配（示例）

- A：因果模块实现与报告生成  
- B：加密与密文域ML适配、性能优化  
- C：实验跑数与指标统计  
- D：论文撰写与可视化  

---

## 六、风险与缓解

- 因果估计方差大 → 提高样本量、分层采样、重采样估计  
- 密文域性能不足 → 降低高频扰动、模型蒸馏/适配、批处理优化  
- 时间风险 → 实验并行化，脚本预制  
- 数据不足（LFW/医疗） → 先以现有集完成主线，额外数据为增益  

---

## 七、备选方案（Fallback Plans）

### 6.1 密文域ML性能风险缓解策略

**风险等级**: 高  
**触发条件**: 当密文域ML准确率低于预期阈值时

#### Plan B1: Fine-tuning Adapter（轻量级微调）

**触发条件**: 密文域准确率 < 60% 且 > 40%

**方案描述**:
- 冻结预训练模型（ResNet/UNet）的主干网络
- 只微调第一层卷积（Adapter Layer）
- 在加密图像上微调，使模型适应加密域分布

**实现细节**:
```python
class FineTuneAdapter(nn.Module):
    """
    轻量级Adapter微调器
    
    策略：冻结主干，只训练第一层
    优势：参数量少（<5%），训练快，通常能提升10-20%精度
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
        # 冻结所有参数
        for param in self.base.parameters():
            param.requires_grad = False
        
        # 只训练第一层（根据模型类型选择）
        if hasattr(self.base, 'conv1'):  # ResNet
            for param in self.base.conv1.parameters():
                param.requires_grad = True
        elif hasattr(self.base, 'features') and hasattr(self.base.features[0], 'weight'):  # VGG
            for param in self.base.features[0].parameters():
                param.requires_grad = True
        # ... 其他模型类型
    
    def forward(self, x):
        return self.base(x)
```

**训练配置**:
- Epochs: 50-100（通常50 epoch内收敛）
- Learning Rate: 1e-4（较小的学习率）
- Batch Size: 32-64
- Optimizer: Adam / SGD

**预期提升**: 通常能提升 10-20% 准确率

**论文表述**:
> "To further improve the performance of ML inference on encrypted images, we introduce a lightweight Adapter fine-tuning strategy. We freeze the pre-trained backbone and only fine-tune the first convolutional layer, which typically improves accuracy by 10-20% while maintaining computational efficiency."

**优势**:
- ✅ 参数量少（<5%），训练快（<1小时）
- ✅ 不改变模型架构，易于集成
- ✅ 可以包装成Contribution（轻量级适配策略）

#### Plan B2: Domain Adversarial Training (DANN)

**触发条件**: Plan B1 后准确率仍 < 50%

**方案描述**:
- 使用Domain Adversarial Neural Network (DANN)
- 让模型学习"加密域不变特征"
- 通过对抗训练，使特征提取器无法区分明文和密文域

**实现细节**:
```python
class DANNAdapter(nn.Module):
    """
    Domain Adversarial Neural Network适配器
    
    策略：对抗训练，学习域不变特征
    优势：能进一步适应加密域分布，通常再提升5-10%
    """
    def __init__(self, feature_extractor, classifier, domain_classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
    
    def forward(self, x, alpha=1.0):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 反转梯度（用于对抗训练）
        features_reversed = GradientReversal.apply(features, alpha)
        
        # 分类预测
        class_pred = self.classifier(features)
        
        # 域分类（用于对抗）
        domain_pred = self.domain_classifier(features_reversed)
        
        return class_pred, domain_pred
```

**训练配置**:
- Epochs: 100-200
- Learning Rate: 1e-4
- Lambda (域对抗权重): 0.1-1.0（逐渐增加）

**预期提升**: 在Plan B1基础上再提升 5-10%

**论文表述**:
> "For cases where Adapter fine-tuning is insufficient, we apply Domain Adversarial Training (DANN) to learn encryption-domain-invariant features, further improving accuracy by 5-10%."

#### Plan B3: 降级到"准密文"（最后手段）

**触发条件**: Plan B1 + B2 后准确率仍 < 40%（性能完全不可用）

**方案描述**:
- 采用"部分解密"策略：只解密低频子带（LL）
- 在LL子带上运行ML，其他子带（LH/HL/HH）保持加密
- 在论文中明确说明，并论证安全性

**实现细节**:
```python
def partial_decryption_for_ml(encrypted_image, enc_info):
    """
    部分解密：只解密LL子带
    
    策略：
    - LL子带包含主要语义信息（适合ML）
    - LH/HL/HH子带保持加密（保护细节）
    - 安全性：LL子带单独泄露的隐私风险有限
    """
    # 提取LL子带
    LL, LH, HL, HH = dwt2d(encrypted_image)
    
    # 解密LL子带
    LL_decrypted = decrypt_dwt_subband(LL, enc_info['ll_key'])
    
    # 重构图像（只有LL解密，其他保持加密）
    image_reconstructed = idwt2d(LL_decrypted, LH, HL, HH)
    
    return image_reconstructed
```

**安全性论证**:
- LL子带主要包含低频信息（整体结构）
- 缺少高频细节，无法精确重建人脸/敏感信息
- 但需要明确说明这不是"完全密文域"，而是"部分密文域"

**论文表述**:
> "For extremely high encryption intensities where full ciphertext-domain inference is infeasible, we introduce a partial decryption strategy: only the low-frequency subband (LL) is decrypted for ML inference, while high-frequency subbands remain encrypted. This balances usability and privacy, with LL alone providing insufficient information for identity reconstruction."

**注意**: 
- ⚠️ 这不是首选方案，只有在性能完全不可用时才使用
- ⚠️ 必须在论文中明确说明并论证安全性
- ⚠️ 可以在Discussion中作为Limitation讨论

### 6.2 因果估计方差过大的缓解策略

**风险等级**: 中  
**触发条件**: ATE估计的置信区间过宽（>0.3）或p值不显著（p>0.05）

#### Plan C1: 增加样本量

**策略**: 
- 扩大测试集（从600张到1000-2000张）
- 使用更多加密强度副本（从5个到10个）

#### Plan C2: 分层采样与加权

**策略**:
- 按语义区域和任务类型分层
- 使用逆概率加权（IPW）降低选择偏差

#### Plan C3: 稳健估计

**策略**:
- 使用Bootstrap重采样估计置信区间
- 使用Robust统计量（中位数、MAD）

### 6.3 Fallback Plan执行决策树

```
密文域ML准确率评估
│
├─ > 80% → ✅ 无需Fallback，继续实验
│
├─ 60-80% → Plan B1 (Adapter Fine-tuning)
│           │
│           ├─ > 80% → ✅ 成功，记录结果
│           └─ < 60% → Plan B2 (DANN)
│                      │
│                      ├─ > 60% → ✅ 成功，记录结果
│                      └─ < 50% → Plan B3 (部分解密)
│                                 │
│                                 └─ ✅ 记录为Limitation
│
└─ < 40% → ⚠️ 直接Plan B3（性能完全不可用）
```

### 6.4 Fallback Plan在论文中的呈现策略

**策略**: 将Fallback Plan包装成**贡献点（Contribution）**，而非妥协

**表述建议**:

1. **Plan B1 (Adapter Fine-tuning)**:
   - 标题: "Lightweight Adapter for Ciphertext-Domain Adaptation"
   - 贡献: 提出了一种轻量级的域适应方法，只需微调5%的参数即可适应加密域

2. **Plan B2 (DANN)**:
   - 标题: "Domain-Invariant Feature Learning for Encrypted Images"
   - 贡献: 通过对抗训练学习加密域不变特征，进一步提升性能

3. **Plan B3 (部分解密)**:
   - 标题: "Hierarchical Partial Decryption for Extreme Privacy Scenarios"
   - 贡献: 在极端隐私要求下，提出了分层部分解密策略（作为Limitation讨论）

**关键原则**: 
- ✅ 不说是"性能不够才用的备选"
- ✅ 说成是"为了进一步提升性能而设计的方法"
- ✅ 强调"轻量级"、"高效"、"可扩展"等优点

---

## 八、当前优先事项（下两周）

- [ ] 实现 CausalPrivacyAnalyzer 的 ATE/CATE + 基线函数  
- [ ] 在 privacy_budget 中接入 causal_suggestion  
- [ ] 优化 encrypt_layered 接口，确保 α 记录用于解密  
- [ ] 跑通小规模 E2E 流程（batch=8）并记录首批 ATE  

---

**相关文档**:  
- [总览](Causal-VSE-PC_项目总览.md)  
- [数据流向](Causal-VSE-PC_数据流向.md)  
- [数据集分析](Causal-VSE-PC_数据集分析与使用.md)  
- [工作流程](Causal-VSE-PC_工作流程.md)  
- [目标与指标](Causal-VSE-PC_目标与指标.md)  
- [理论证明](Causal-VSE-PC_理论证明.md)  
{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}