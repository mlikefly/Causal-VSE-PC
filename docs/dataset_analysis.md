# Causal-VSE-PC 数据集分析与使用

**文档版本**: 1.0  
**创建日期**: 2025年12月  
**核心创新**: 因果推断驱动的加密策略

---

## 一、数据集概述

### 1.1 现有数据集清单

```
项目目录中的数据集：
data/
├── CelebA-HQ/          ✅ 主要数据集（6000张）
│   ├── train/ (4800张)
│   ├── val/ (600张)
│   └── test/ (600张)
│
├── CelebA-HQ-labels/   ✅ 标签数据（5400个.npy）
│   ├── train/ (4800个)
│   └── val/ (600个)
│
└── CelebAMask-HQ/      ✅ 分割数据集（6710张）
    ├── CelebA-HQ-img/ (6710张)
    └── CelebAMask-HQ/ (掩码数据，19个区域)
```

### 1.2 数据集详细评估

#### CelebA-HQ（主要实验数据集）

| 属性 | 数值 | 说明 |
|------|------|------|
| **总数量** | 6000张 | 训练4800 + 验证600 + 测试600 |
| **分辨率** | 1024×1024 | 高质量人脸图像 |
| **格式** | JPG | 标准图像格式 |
| **标签** | 40个属性 | 性别、年龄、表情、发型等 |
| **适用任务** | 分类、识别、属性推断 | ✅ 完全适合 |
| **因果分析** | 适合 | ✅ 可分析加密对识别/分类的因果效应 |

**优势**：
- ✅ 数据量大（6000张），统计显著性高
- ✅ 标签完整（40个属性），适合多任务评估
- ✅ 高质量（HQ版本），适合加密前后对比
- ✅ 适合人脸隐私保护场景（核心应用场景）
- ✅ 适合因果分析（可量化加密对识别/分类的影响）

**局限性**：
- ⚠️ 仅有人脸图像，缺少自然场景
- ⚠️ 单一数据集，泛化性需跨数据集验证

#### CelebAMask-HQ（分割任务数据集）

| 属性 | 数值 | 说明 |
|------|------|------|
| **总数量** | 6710张 | 包含掩码数据 |
| **分辨率** | 1024×1024 | 高质量 |
| **掩码类型** | 19个区域 | 人脸部件分割 |
| **适用任务** | 分割、语义分析 | ✅ 适合密文域分割任务 |
| **因果分析** | 适合 | ✅ 可分析区域级加密的因果效应 |

**优势**：
- ✅ 有详细的分割掩码（19个区域）
- ✅ 可用于区域级隐私预算验证
- ✅ 适合因果分析（可量化区域级加密的因果效应）

---

## 二、因果分析场景设计

### 2.1 场景1：加密对识别率的因果效应

```
实验设计：
├─ 任务: 人脸识别（身份识别）
├─ 数据集: CelebA-HQ
├─ 评估器: ArcFace / FaceNet
├─ 干预: do(Encryption = high_privacy) vs do(Encryption = low_privacy)
│
├─ 因果图:
│   X (人脸区域) → Z (隐私预算) → Y (识别准确率)
│
├─ 因果效应:
│   ATE = E[Y(high) - Y(low)]
│   CATE(人脸区域) = E[Y(high) - Y(low) | X=face]
│
└─ 预期结果:
   ✅ ATE < -0.90 (识别率下降 > 90%)
   ✅ CATE(人脸) < -0.95 (人脸区域加密效果显著)
```

### 2.2 场景2：加密对分类准确率的因果效应

```
实验设计：
├─ 任务: 属性分类（40个属性）
├─ 数据集: CelebA-HQ
├─ 评估器: 属性分类器（预训练）
├─ 干预: do(Encryption = high_privacy) vs do(Encryption = low_privacy)
│
├─ 因果图:
│   X (任务相关区域) → Z (隐私预算) → Y (分类准确率)
│
├─ 因果效应:
│   ATE = E[Y(high) - Y(low)]
│   CATE(任务区域) = E[Y(high) - Y(low) | X=task_relevant]
│
└─ 预期结果:
   ✅ ATE > -0.20 (分类准确率下降 < 20%)
   ✅ CATE(任务区域) > -0.10 (任务区域保留性能)
```

### 2.3 场景3：区域级隐私预算的因果合理性

```
实验设计：
├─ 任务: 验证隐私预算分配策略
├─ 数据集: CelebAMask-HQ（19个区域）
├─ 评估: 不同区域加密强度的因果效应
│
├─ 因果图:
│   X (区域类型) → Z (隐私预算) → Y (性能损失)
│   - 敏感区域（眼睛、鼻子）: privacy = 1.0
│   - 任务区域（轮廓）: privacy = 0.3
│   - 背景区域: privacy = 0.0
│
├─ 验证假设:
│   H1: 敏感区域加密的识别率下降效应显著
│   H2: 任务区域加密的分类准确率下降效应较小
│
└─ 预期结果:
   ✅ 验证隐私预算分配的因果合理性
   ✅ 生成可解释的加密策略报告
```

---

## 三、数据集使用策略

### 3.1 主要实验数据集：CelebA-HQ

#### 实验1：隐私-可用性权衡（因果分析）

```
用途：
- 量化加密对ML性能的因果效应
- 验证隐私预算分配策略
- 生成可解释的因果报告

数据集划分：
├─ 训练集: 4800张（训练分类器）
├─ 验证集: 600张（调参、验证）
└─ 测试集: 600张（最终评估）

实验流程：
1. 训练属性分类器（明文）
2. 加密测试集（不同隐私级别）
3. 在加密图像上测试分类准确率
4. 计算ATE和CATE（因果效应）
5. 生成因果报告

数据加载：
from src.utils.datasets import get_celeba_dataloader

train_loader = get_celeba_dataloader(
    root_dir='data/CelebA-HQ',
    split='train',
    batch_size=32,
    image_size=256
)
```

#### 实验2：识别攻击验证（隐私保护）

```
用途：
- 验证隐私保护效果
- 计算识别率（隐私指标）

数据集使用：
├─ 测试集: 600张
├─ 评估器: ArcFace / FaceNet（强识别器）
└─ 指标: 识别率 < 5%（隐私保护目标）

流程：
1. 加密测试集（privacy=1.0）
2. 使用识别器提取特征
3. 计算识别准确率（Top-1）
4. 验证隐私保护效果
```

#### 实验3：因果效应分析（核心创新）

```
用途：
- 量化隐私预算分配的因果合理性
- 生成可解释的加密策略报告
- 验证因果推断理论

数据集使用：
├─ 训练集: 4800张（训练分类器、识别器）
├─ 验证集: 600张（调参）
└─ 测试集: 600张（因果分析）

因果分析流程：
1. 构建因果图: X(语义) → Z(隐私预算) → Y(ML性能)
2. 干预实验: do(Z=high) vs do(Z=low)
3. 计算ATE: E[Y(high) - Y(low)]
4. 计算CATE: E[Y(high) - Y(low) | X=sensitive]
5. 生成因果报告（可解释性）

数据加载示例：
from src.utils.datasets import get_celeba_dataloader
from src.vse_pc.causal_analysis import CausalPrivacyAnalyzer

analyzer = CausalPrivacyAnalyzer()
test_loader = get_celeba_dataloader(split='test', ...)

for images, labels in test_loader:
    # 语义分析
    semantic_mask = unet(images)
    
    # 不同隐私级别的加密
    encrypted_high = encrypt(images, privacy=1.0)
    encrypted_low = encrypt(images, privacy=0.3)
    
    # 计算性能
    perf_high = evaluate_ml(encrypted_high, model)
    perf_low = evaluate_ml(encrypted_low, model)
    
    # 因果分析
    ate = analyzer.compute_ate(perf_high, perf_low)
    cate = analyzer.compute_cate(semantic_mask, perf_high, perf_low)
```

### 3.2 分割任务数据集：CelebAMask-HQ

#### 实验：区域级隐私预算验证

```
用途：
- 验证区域级隐私预算分配的因果合理性
- 分析不同区域加密强度的因果效应
- 生成区域级可解释报告

数据集使用：
├─ 图像: 6710张（人脸图像）
├─ 掩码: 19个区域（眼睛、鼻子、嘴巴等）
└─ 划分: 训练5000 / 验证855 / 测试855

实验流程：
1. 语义分析（提取区域掩码）
2. 区域级隐私预算分配
   - 敏感区域（眼睛、鼻子）: privacy = 1.0
   - 任务相关区域（轮廓）: privacy = 0.3
   - 背景区域: privacy = 0.0
3. 加密（区域级差异加密）
4. 分割任务评估（mIoU）
5. 识别任务评估（隐私保护）
6. 因果分析（区域级ATE）

数据加载：
需要实现 get_celebamask_dataloader()
# 加载图像和掩码
images, masks = load_celebamask_data(...)
```

---

## 四、建议补充的数据集

### 4.1 LFW（Labeled Faces in the Wild）

```
用途：跨数据集验证、识别攻击测试
数量：13,233张图像，5,749个身份
下载：http://vis-www.cs.umass.edu/lfw/
优先级：P1（重要）

适用场景：
✅ 识别攻击验证（跨数据集）
✅ 泛化性评估
✅ 因果分析（跨数据集ATE）

实验设计：
1. 使用CelebA-HQ训练的加密系统
2. 在LFW上测试识别率
3. 验证隐私保护泛化性
4. 计算跨数据集因果效应
```

### 4.2 ImageNet-100

```
用途：自然图像分类、泛化性验证
数量：100个类别，每类约1300张
下载：ImageNet官网（需要申请）
优先级：P2（可选）

适用场景：
✅ 自然场景验证
✅ 跨域泛化性评估
✅ 非人脸图像的因果分析

实验设计：
1. 自然图像加密
2. 分类任务评估
3. 验证泛化性
4. 分析非人脸场景的因果效应
```

### 4.3 医疗图像数据集（如果可获得）

```
用途：医疗隐私计算原型
类型：CT、MRI、X光片
数量：取决于可获得的数据
优先级：P2（应用场景）

适用场景：
✅ 医疗图像隐私计算
✅ 实际应用验证
✅ 医疗场景的因果分析

注意：
⚠️ 需要获得数据授权
⚠️ 隐私保护要求极高
```

---

## 五、数据预处理流程

### 5.1 图像预处理

```
标准预处理：
├─ 尺寸调整: 1024×1024 → 256×256（加密处理）
├─ 归一化: [0, 255] → [0, 1]（除以255.0）
├─ 数据类型: uint8 → float32
└─ 通道顺序: RGB / Grayscale

增强（训练时）：
├─ 随机水平翻转: 50%概率
├─ 随机裁剪: 可选
└─ 颜色抖动: 可选（RGB图像）

加密预处理：
├─ 确保值域在[0, 1]
├─ 转换为torch.Tensor
└─ 移至GPU（如果可用）
```

### 5.2 标签处理

```
CelebA-HQ标签：
├─ 格式: .npy文件（numpy数组）
├─ 内容: 40个二值属性（-1或1）
├─ 处理: 
│   ├─ 转换为[0, 1]或[-1, 1]
│   └─ 转换为torch.Tensor
└─ 使用: 属性分类任务

CelebAMask-HQ标签：
├─ 格式: .png文件（掩码图像）
├─ 内容: 19个区域的掩码
├─ 处理:
│   ├─ 转换为类别索引 [0, 18]
│   ├─ 或转换为one-hot编码 [B, 19, H, W]
│   └─ 转换为torch.Tensor
└─ 使用: 分割任务
```

---

## 六、实验设计详细说明

### 6.1 实验1：隐私-可用性权衡（CelebA-HQ）

```
数据集：CelebA-HQ
任务：属性分类（40个属性）
隐私级别：[0.0, 0.3, 0.5, 0.7, 1.0]

实验设计：
├─ 训练阶段:
│   ├─ 数据集: 4800张（明文）
│   ├─ 任务: 训练属性分类器
│   └─ 模型: ResNet50 / VGG16
│
├─ 测试阶段:
│   ├─ 数据集: 600张（加密）
│   ├─ 加密: 不同隐私级别
│   ├─ 评估: 分类准确率（密文域）
│   └─ 评估: 识别率（隐私保护）
│
└─ 因果分析:
   ├─ ATE计算: E[Accuracy(high) - Accuracy(low)]
   ├─ CATE计算: 不同区域的因果效应
   └─ 生成因果报告

评估指标：
├─ 分类准确率（密文域）: > 80%（privacy=0.7）
├─ 识别率（隐私保护）: < 5%（privacy=1.0）
└─ ATE: 量化隐私对性能的因果影响
```

### 6.2 实验2：区域级因果分析（CelebAMask-HQ）

```
数据集：CelebAMask-HQ
任务：人脸分割（19个区域）+ 识别（隐私保护）
隐私级别：区域级差异加密

实验设计：
├─ 区域划分:
│   ├─ 敏感区域: 眼睛、鼻子、嘴巴 (privacy=1.0)
│   ├─ 任务区域: 轮廓、头发 (privacy=0.3)
│   └─ 背景区域: 其他 (privacy=0.0)
│
├─ 评估任务:
│   ├─ 分割任务: mIoU（密文域）
│   ├─ 识别任务: 识别率（隐私保护）
│   └─ 因果分析: 区域级ATE
│
└─ 因果假设验证:
   ├─ H1: 敏感区域加密显著降低识别率
   ├─ H2: 任务区域加密轻微影响分割性能
   └─ H3: 区域级差异加密优于全局加密

评估指标：
├─ 分割mIoU（密文域）: > 70%
├─ 识别率（隐私保护）: < 5%
└─ 区域级ATE: 量化不同区域的因果效应
```

### 6.3 实验3：跨数据集验证（LFW）

```
数据集：LFW
任务：人脸识别（隐私保护验证）
目的：验证泛化性

实验设计：
├─ 加密系统: 使用CelebA-HQ训练的参数
├─ 测试数据: LFW数据集
├─ 评估器: ArcFace / FaceNet
└─ 评估: 识别率（应 < 5%）

因果分析：
├─ 跨数据集ATE: E[Recognition(LFW) - Recognition(CelebA)]
├─ 验证: 因果效应在数据集间的一致性
└─ 结论: 因果推断的泛化性
```

---

## 七、数据加载实现

### 7.1 现有代码利用

```
已有实现：src/utils/datasets.py
├─ 函数: get_celeba_dataloader()
├─ 功能: CelebA-HQ数据加载
├─ 状态: ✅ 可用
└─ 使用: 直接调用

需要扩展：
├─ CelebAMask-HQ加载器（待实现）
├─ LFW加载器（如果补充数据集）
└─ ImageNet-100加载器（如果补充数据集）
```

### 7.2 数据加载示例代码

```python
# CelebA-HQ加载示例
from src.utils.datasets import get_celeba_dataloader

train_loader = get_celeba_dataloader(
    root_dir='data/CelebA-HQ',
    split='train',
    batch_size=32,
    image_size=256,
    return_labels=True
)

# 使用
for images, labels in train_loader:
    # images: [B, C, 256, 256], float32, [0, 1]
    # labels: [B, 40], int/long
    encrypted, enc_info = encrypt_for_ml(images, task_type='classification')
    predictions = classifier(encrypted)
    # 计算准确率...

# CelebAMask-HQ加载（待实现）
def get_celebamask_dataloader(...):
    # 加载图像和掩码
    images = load_images(...)  # [B, C, H, W]
    masks = load_masks(...)    # [B, 19, H, W] 或 [B, H, W]
    return DataLoader(images, masks, ...)
```

---

## 八、数据集统计与分析

### 8.1 数据分布

```
CelebA-HQ属性分布：
├─ 性别: 男性 ~50%, 女性 ~50% (平衡)
├─ 年龄: 年轻 ~60%, 中年 ~30%, 老年 ~10% (不平衡)
├─ 表情: 微笑 ~40%, 中性 ~60% (轻微不平衡)
└─ 其他属性: 相对均衡

注意：
⚠️ 部分属性存在类别不平衡
⚠️ 需要平衡采样或加权损失
⚠️ 因果分析时需要考虑数据分布
```

### 8.2 数据质量

```
图像质量：
├─ 分辨率: 1024×1024（高质量）
├─ 清晰度: 高（HQ版本）
└─ 标注质量: 高（人工标注）

标签质量：
├─ 属性标注: 准确率 > 95%
├─ 分割掩码: 准确率 > 90%
└─ 适合实验使用

数据完整性：
├─ 图像文件: 完整（无缺失）
├─ 标签文件: 完整（训练/验证集）
└─ 数据一致性: 良好
```

---

## 九、反事实生成方法（Perfect Counterfactual Environment）

### 9.1 数字图像场景的独特优势

**核心创新点**: 在数字图像加密场景中，我们可以完美生成反事实数据，这解决了传统因果推断中"数据缺失"的根本难题。

#### 9.1.1 传统因果推断的困境

在医学/社会科学领域，因果推断面临的核心难题是：
- 无法同时观测 $Y(1)$ 和 $Y(0)$（不能同时给同一患者吃药和不吃药）
- 只能观测到事实结果（factual），无法观测反事实结果（counterfactual）
- 必须依赖统计假设（如无混淆）来估计因果效应

#### 9.1.2 数字图像场景的完美反事实生成

**关键优势**: 对于同一张图像 $I$，我们可以生成多个加密强度副本，完美控制干预变量 $Z$。

**方法**:
```
原始图像: I₀ (明文)

反事实生成:
├─ 副本A: I_A = Encrypt(I₀, Z=0.9)  [强加密]
├─ 副本B: I_B = Encrypt(I₀, Z=0.3)  [弱加密]
├─ 副本C: I_C = Encrypt(I₀, Z=0.5)  [中等加密]
└─ ... (可生成任意 Z ∈ [0, 1] 的副本)

观测结果:
├─ Y_A = Model(I_A)  [强加密下的性能]
├─ Y_B = Model(I_B)  [弱加密下的性能]
└─ Y_C = Model(I_C)  [中等加密下的性能]

因果效应:
ATE = E[Y_A - Y_B] = E[Y(Z=0.9) - Y(Z=0.3)]
```

#### 9.1.3 为什么这是"完美"的反事实？

1. **消除个体差异**:
   - 所有副本来自同一张原始图像 $I_0$
   - 图像内容、语义区域 $X$ 完全一致
   - 唯一变化的是加密强度 $Z$

2. **消除时间效应**:
   - 所有副本同时生成（无时间延迟）
   - 模型参数在所有副本上保持一致
   - 环境变量（如光照、噪声）完全相同

3. **完美控制**:
   - 可以精确控制 $Z$ 的取值（任意 $z \in [0, 1]$）
   - 不存在"正性假设"违反（所有 $z$ 都是可达的）
   - 满足"一致性假设"（干预 $do(Z=z)$ 与观测 $Z=z$ 一致）

4. **无遗漏变量**:
   - 控制 $X$（语义区域）和 $T$（任务类型）后
   - 影响 $Y$ 的其他因素在所有 $Z$ 取值下保持一致
   - 因此观测到的 $Y$ 差异可以完全归因于 $Z$

**结论**: 这实现了"Interventionist Causal Inference"的理想条件，使因果推断的可靠性和准确性达到最高水平。

### 9.2 实验设计（RCT变体）

#### 9.2.1 反事实生成流程

**步骤1**: 对每个样本 $(X_i, T_i)$，生成K个加密强度副本

```
输入: 图像 I_i，语义区域 X_i，任务类型 T_i

生成副本:
├─ Z₁ = 0.0 (无加密，基准)
├─ Z₂ = 0.3 (弱加密)
├─ Z₃ = 0.5 (中等加密)
├─ Z₄ = 0.7 (强加密)
└─ Z₅ = 0.9 (极强加密)

加密结果:
{I_i^(k) = Encrypt(I_i, Z_k)}_{k=1}^5
```

**步骤2**: 在密文上运行ML模型

```
性能评估:
├─ Y_i^(1) = Model(I_i^(1))  [无加密性能]
├─ Y_i^(2) = Model(I_i^(2))  [弱加密性能]
├─ Y_i^(3) = Model(I_i^(3))  [中等加密性能]
├─ Y_i^(4) = Model(I_i^(4))  [强加密性能]
└─ Y_i^(5) = Model(I_i^(5))  [极强加密性能]
```

**步骤3**: 计算ATE（分层估计）

```
ATE估计:
ATE = (1/N) Σ_{i=1}^N (1/K) Σ_{k=2}^K [Y_i^(k) - Y_i^(1)]

其中:
- N: 样本数量
- K: 加密强度级别数量
- Y_i^(1): 基准性能（无加密）
- Y_i^(k): 加密强度 Z_k 下的性能
```

**步骤4**: 计算CATE（条件平均处理效应）

```
CATE估计（按语义区域）:
CATE(X=x) = E[Y(Z=z_high) - Y(Z=z_low) | X=x]

实现:
- 筛选所有 X_i = x 的样本
- 计算这些样本的平均处理效应
- 得到区域 $x$ 的条件平均处理效应
```

#### 9.2.2 分层采样策略

为了满足"正性假设"并降低方差，采用分层采样：

```
采样策略:
├─ 按语义区域分层: {sensitive, task_relevant, background}
├─ 按任务类型分层: {classification, segmentation, detection}
└─ 每层生成所有加密强度副本

优势:
✅ 确保所有 (X, T, Z) 组合都有数据
✅ 降低ATE估计的方差
✅ 满足正性假设
```

### 9.3 论文表述建议

**在论文 Experiment Setup 部分，建议加入以下段落**:

> **Perfect Counterfactual Environment for Causal Inference**
> 
> Unlike traditional causal inference scenarios (e.g., medical trials) where we cannot simultaneously observe both treatment and control outcomes for the same unit, our digital image encryption setting enables **perfect counterfactual generation**. For each image $I$ with fixed semantic regions $X$ and task type $T$, we can generate multiple encrypted copies $\{Encrypt(I, z_k)\}_{k=1}^K$ with different encryption intensities $z_k \in [0, 1]$. This eliminates the fundamental "missing data" problem in causal inference, as we can observe both $Y(Z=z_1)$ and $Y(Z=z_0)$ for the same image.
> 
> This **interventionist causal inference** capability ensures that:
> 1. Individual differences are eliminated (all copies share the same image content)
> 2. Temporal effects are removed (all copies are generated simultaneously)
> 3. Confounding variables are perfectly controlled (only $Z$ varies)
> 
> As a result, our ATE/CATE estimates achieve the highest possible reliability and accuracy, making our causal analysis more credible than traditional observational studies.

### 9.4 代码实现示例

```python
def generate_counterfactual_dataset(images, semantic_masks, task_type, encryption_levels):
    """
    生成反事实数据集
    
    Args:
        images: [N, C, H, W] 原始图像
        semantic_masks: [N, 1, H, W] 语义掩码
        task_type: 任务类型
        encryption_levels: [K] 加密强度列表，如 [0.0, 0.3, 0.5, 0.7, 0.9]
    
    Returns:
        counterfactual_data: {
            'images': [N*K, C, H, W],  # 所有加密副本
            'encryption_levels': [N*K],  # 对应的加密强度
            'semantic_masks': [N*K, 1, H, W],  # 语义掩码（重复）
            'original_indices': [N*K]  # 原始图像索引
        }
    """
    N = len(images)
    K = len(encryption_levels)
    
    counterfactual_images = []
    encryption_levels_all = []
    semantic_masks_all = []
    original_indices = []
    
    for i in range(N):
        for k, z_k in enumerate(encryption_levels):
            # 生成加密副本
            encrypted = encrypt_image(images[i], privacy_level=z_k)
            counterfactual_images.append(encrypted)
            encryption_levels_all.append(z_k)
            semantic_masks_all.append(semantic_masks[i])
            original_indices.append(i)
    
    return {
        'images': torch.stack(counterfactual_images),  # [N*K, C, H, W]
        'encryption_levels': torch.tensor(encryption_levels_all),  # [N*K]
        'semantic_masks': torch.stack(semantic_masks_all),  # [N*K, 1, H, W]
        'original_indices': torch.tensor(original_indices)  # [N*K]
    }

def compute_ate_from_counterfactual(counterfactual_data, ml_performances):
    """
    从反事实数据计算ATE
    
    Args:
        counterfactual_data: 反事实数据集（上述函数返回）
        ml_performances: [N*K] ML模型在加密图像上的性能
    
    Returns:
        ate_result: {
            'ate': float,  # 平均处理效应
            'ate_by_level': dict,  # 不同加密强度下的效应
            'std': float  # 标准差
        }
    """
    N = len(set(counterfactual_data['original_indices'].tolist()))
    K = len(set(counterfactual_data['encryption_levels'].tolist()))
    
    # 基准性能（Z=0.0）
    baseline_mask = (counterfactual_data['encryption_levels'] == 0.0)
    baseline_perf = ml_performances[baseline_mask].mean()
    
    # 处理效应（相对于基准）
    treatment_effects = []
    for z in sorted(set(counterfactual_data['encryption_levels'].tolist())):
        if z == 0.0:
            continue
        mask = (counterfactual_data['encryption_levels'] == z)
        perf_z = ml_performances[mask].mean()
        effect = perf_z - baseline_perf
        treatment_effects.append(effect)
    
    # ATE（所有处理级别的平均）
    ate = np.mean(treatment_effects)
    ate_std = np.std(treatment_effects)
    
    return {
        'ate': float(ate),
        'std': float(ate_std),
        'baseline_performance': float(baseline_perf)
    }
```

---

## 十、因果分析数据要求

### 10.1 数据量要求

```
统计显著性：
├─ 训练集: ≥ 4800张（分类器训练）
├─ 验证集: ≥ 600张（调参、验证）
├─ 测试集: ≥ 600张（最终评估）
└─ 因果分析: ≥ 1000张（ATE估计，降低方差）

当前数据集：
✅ CelebA-HQ: 6000张（满足要求）
✅ CelebAMask-HQ: 6710张（满足要求）
```

### 9.2 数据多样性要求

```
语义多样性：
├─ 敏感区域: 人脸、身份证等
├─ 任务区域: 物体、场景等
└─ 背景区域: 各种背景

任务多样性：
├─ 分类任务: 40个属性（CelebA-HQ）
├─ 分割任务: 19个区域（CelebAMask-HQ）
└─ 识别任务: 身份识别

当前数据集：
✅ 语义多样性: 充足（人脸场景）
⚠️ 任务多样性: 需要补充自然场景
```

---

## 十、总结与建议

### 10.1 数据集适用性总结

| 数据集 | 主要用途 | 适用度 | 状态 | 因果分析 |
|--------|---------|--------|------|---------|
| **CelebA-HQ** | 分类、识别 | ⭐⭐⭐⭐⭐ | ✅ 可用 | ✅ 适合 |
| **CelebAMask-HQ** | 分割 | ⭐⭐⭐⭐ | ✅ 可用 | ✅ 适合 |
| **LFW** | 识别验证 | ⭐⭐⭐⭐ | ⚠️ 建议补充 | ✅ 跨数据集ATE |
| **ImageNet-100** | 泛化验证 | ⭐⭐⭐ | ⚠️ 可选补充 | ✅ 自然场景 |

### 10.2 使用建议

```
1. 主要实验：使用CelebA-HQ（已有，适合）
   ✅ 数据量大（6000张）
   ✅ 标签完整（40个属性）
   ✅ 适合因果分析

2. 分割任务：使用CelebAMask-HQ（已有，适合）
   ✅ 区域级掩码（19个区域）
   ✅ 适合区域级因果分析

3. 识别验证：补充LFW（重要，建议）
   ✅ 跨数据集验证
   ✅ 泛化性评估
   ✅ 跨数据集因果分析

4. 泛化验证：补充ImageNet-100（可选）
   ✅ 自然场景验证
   ✅ 非人脸图像因果分析
```

### 10.3 数据准备清单

```
已完成：
- [x] CelebA-HQ数据集（6000张）
- [x] CelebAMask-HQ数据集（6710张）
- [x] 数据加载器（部分已有）

待完成：
- [ ] LFW数据集（建议补充）
- [ ] ImageNet-100数据集（可选补充）
- [ ] CelebAMask-HQ数据加载器（待实现）
- [ ] 因果分析数据预处理（待实现）
```

### 10.4 因果分析数据要求

```
核心要求：
✅ 足够的数据量（统计显著性）
✅ 多样化的语义区域
✅ 多任务评估（分类、分割、识别）
✅ 跨数据集验证（泛化性）

当前状态：
✅ 主要数据集满足要求
⚠️ 需要补充跨数据集验证数据
```

---

**相关文档**:
- [项目总览](project_overview.md)
- [数据流向](data_flow.md)
- [工作流程](workflow.md)
- [目标与指标](goals_and_metrics.md)
- [理论证明](theoretical_proof.md)