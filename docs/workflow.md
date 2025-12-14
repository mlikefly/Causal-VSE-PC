# Causal-VSE-PC 工作流程

**文档版本**: 1.0  
**创建日期**: 2025年12月  
**核心创新**: 因果推断驱动的加密策略

---

## 一、完整工作流程（含因果分析）

### 1.1 流程图

```
┌─────────────────────────────────────────────────┐
│ 阶段1: 输入准备                                  │
│ 输入: 原始图像 I, 任务类型 T, 隐私需求 P        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段2: 语义分析（U-Net，已有）                   │
│ 输出: 语义掩码 M = {敏感区域, 任务区域, 背景}    │
│ 时间: ~5ms (GPU)                                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段3: 因果分析（核心创新）                      │
│ ├─ 构建因果图: X(Semantic) → Z(Privacy) → Y(ML)│
│ ├─ 计算历史基线: E[Y | X, T]                    │
│ ├─ 生成因果建议: 隐私预算分配策略                │
│ └─ 输出: 因果报告 + 隐私预算建议                │
│ 时间: ~10ms                                      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段4: 隐私预算分配（规则-based + 因果指导）     │
│ 规则:                                            │
│   - 敏感区域（人脸）: privacy = 1.0             │
│   - 任务区域（物体）: privacy = 0.3             │
│   - 背景区域: privacy = 0.0                     │
│ 因果调整: 根据因果分析结果微调                  │
│ 输出: 隐私预算图 P(x,y)                         │
│ 时间: <1ms                                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段5: 分层加密（利用现有组件）                  │
│                                                 │
│ Layer 1: 混沌置乱（已有，5D混沌系统）           │
│   - Arnold变换: 打乱像素位置                    │
│   - 迭代次数: 固定5次                           │
│   - 5D超混沌扩散: 像素值混淆                    │
│                                                 │
│ Layer 2: 频域语义控制（已有）                   │
│   - FFT变换（比DWT快）                          │
│   - LL子带: 弱扰动（保留语义）                 │
│   - LH/HL/HH子带: 强扰动（破坏细节）           │
│   - 根据P(x,y)调整扰动强度                      │
│                                                 │
│ 输出: 加密图像 I_enc                            │
│ 时间: ~100ms (GPU)                              │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段6: 密文域ML推理（新设计）                    │
│                                                 │
│ 输入: I_enc（加密图像）                         │
│ 处理: 直接在I_enc上运行ML模型                   │
│ 输出: 预测结果（分类/分割/检测）                │
│ 时间: ~50ms (取决于模型复杂度)                  │
│                                                 │
│ 关键: 无需解密，直接推理                        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段7: 因果效应计算（后处理分析）                │
│                                                 │
│ ├─ 计算ATE: E[Y(high_privacy) - Y(low_privacy)]│
│ ├─ 计算CATE: E[Y(high) - Y(low) | X=sensitive] │
│ ├─ 生成可解释报告: "为什么这样加密？"            │
│ └─ 输出: 因果分析报告                           │
│ 时间: ~5ms                                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 阶段8: 可验证性证明（轻量级Hash Commitment）     │
│                                                 │
│ 生成: Hash Commitment + HMAC                    │
│ 证明: I_enc确实来自I，且加密正确                │
│ 验证: 无需密钥即可验证                          │
│ 时间: ~10ms                                      │
└─────────────────────────────────────────────────┘
```

---

## 二、详细工作流程说明

### 2.1 阶段3：因果分析（核心创新）

```
输入：
├─ M_semantic: [B, 1, H, W] 语义掩码
├─ T: 任务类型 {'classification', 'segmentation', 'detection'}
└─ 历史数据: 历史加密-性能对

处理步骤：

Step 1: 构建因果图 (Structural Causal Model, SCM)
├─ 节点:
│   ├─ X: 语义区域类型（敏感/任务相关/背景）
│   ├─ T: 任务类型（分类/分割/检测）
│   ├─ Z: 隐私预算分配（加密强度）
│   └─ Y: ML任务性能（准确率/mIoU/mAP）
│
├─ 边:
│   ├─ X → Z: 语义决定隐私预算
│   ├─ T → Z: 任务类型影响隐私预算
│   └─ Z → Y: 隐私预算影响ML性能
│
└─ 输出: 因果图 G

Step 2: 计算历史基线
├─ 收集: 历史加密-性能对 {Z_i, Y_i}
├─ 估计: E[Y | X, T] = 基线性能
└─ 输出: baseline_performance

Step 3: 生成因果建议
├─ 反事实推理: "如果使用high_privacy会怎样？"
├─ ATE估计: E[Y(high) - Y(low) | X, T]
├─ 建议生成: 基于ATE的隐私预算分配建议
└─ 输出: causal_suggestion

Step 4: 生成因果解释
├─ 自然语言生成:
│   "人脸区域加密的识别率下降效应为-0.95，
│    任务区域加密的分类准确率下降效应为-0.10，
│    因此建议对人脸区域使用强加密(privacy=1.0)，
│    对任务区域使用弱加密(privacy=0.3)"
│
└─ 输出: causal_report (可解释的加密策略)
```

### 2.2 阶段7：因果效应计算（后处理）

```
输入：
├─ I_encrypted: 加密图像
├─ predictions: ML预测结果
├─ ground_truth: 真实标签
└─ P(x,y): 隐私预算图

处理步骤：

Step 1: 性能评估
├─ 计算: accuracy / mIoU / mAP
└─ 输出: performance_metrics

Step 2: 计算ATE (Average Treatment Effect)
├─ 干预对比:
│   ├─ 高隐私组: privacy > 0.7
│   └─ 低隐私组: privacy < 0.3
│
├─ ATE计算:
│   ATE = E[Y(high) - Y(low)]
│       = mean(Y_high) - mean(Y_low)
│
└─ 输出: ATE值

Step 3: 计算CATE (Conditional ATE)
├─ CATE(X=sensitive):
│   CATE = E[Y(high) - Y(low) | X=sensitive]
│
├─ CATE(X=task_relevant):
│   CATE = E[Y(high) - Y(low) | X=task_relevant]
│
└─ 输出: CATE值（区域级因果效应）

Step 4: 生成因果报告
├─ 汇总: ATE, CATE, 性能指标
├─ 可视化: 因果效应图
└─ 输出: causal_analysis_report
```

---

## 三、完整流程代码框架

### 3.1 主流程函数

```python
def causal_vse_pc_pipeline(image, task_type, privacy_requirement):
    """
    Causal-VSE-PC完整流程（含因果分析）
    
    参数:
        image: 原始图像 [B, C, H, W]
        task_type: 任务类型 {'classification', 'segmentation', 'detection'}
        privacy_requirement: 隐私需求 [0.0, 1.0]
    
    返回:
        encrypted: 加密图像
        predictions: ML预测结果
        proof: 可验证性证明
        causal_report: 因果分析报告
    """
    
    # 1. 语义分析
    semantic_mask = unet_semantic_analysis(image)  # [B, 1, H, W]
    
    # 2. 因果分析（核心创新）
    causal_analyzer = CausalPrivacyAnalyzer()
    causal_suggestion = causal_analyzer.analyze_allocation(
        semantic_mask, task_type
    )
    
    # 3. 隐私预算分配（因果指导）
    privacy_allocator = AdaptivePrivacyBudget()
    privacy_map = privacy_allocator.allocate(
        semantic_mask, 
        task_type, 
        privacy_requirement,
        causal_suggestion=causal_suggestion  # 因果建议
    )  # [B, 1, H, W]
    
    # 4. 分层加密
    encrypted, enc_info = encrypt_layered(
        image,
        semantic_mask,
        privacy_map
    )
    
    # 5. 密文域ML推理
    predictions = ciphertext_ml_inference(
        encrypted,
        ml_model,
        task_type
    )
    
    # 6. 因果效应计算（后处理）
    causal_report = causal_analyzer.compute_causal_effects(
        semantic_mask,
        privacy_map,
        predictions,
        ground_truth
    )
    
    # 7. 可验证性证明
    proof = generate_verification_proof(
        image,
        encrypted,
        enc_info
    )
    
    return {
        'encrypted': encrypted,
        'predictions': predictions,
        'proof': proof,
        'causal_report': causal_report,
        'enc_info': enc_info
    }
```

### 3.2 因果分析组件接口

```python
class CausalPrivacyAnalyzer:
    """
    因果隐私分析器（核心创新）
    """
    
    def analyze_allocation(self, semantic_mask, task_type):
        """
        分析隐私预算分配的因果合理性
        
        返回:
            causal_suggestion: 因果建议（隐私预算分配策略）
        """
        # 1. 构建因果图
        causal_graph = self._build_causal_graph(semantic_mask, task_type)
        
        # 2. 计算历史基线
        baseline = self._compute_baseline(task_type)
        
        # 3. 生成因果建议
        suggestion = self._generate_suggestion(causal_graph, baseline)
        
        return suggestion
    
    def compute_causal_effects(self, semantic_mask, privacy_map, 
                                predictions, ground_truth):
        """
        计算因果效应（后处理）
        
        返回:
            causal_report: 因果分析报告
        """
        # 1. 性能评估
        performance = self._evaluate_performance(predictions, ground_truth)
        
        # 2. 计算ATE
        ate = self._compute_ate(privacy_map, performance)
        
        # 3. 计算CATE
        cate = self._compute_cate(semantic_mask, privacy_map, performance)
        
        # 4. 生成报告
        report = self._generate_report(ate, cate, performance)
        
        return report
```

---

## 四、关键工作流程对比

### 4.1 传统VSE-PC vs Causal-VSE-PC

| 阶段 | 传统VSE-PC | Causal-VSE-PC（新） |
|------|-----------|-------------------|
| **语义分析** | ✅ U-Net提取语义 | ✅ U-Net提取语义 |
| **隐私分配** | ✅ 规则-based | ✅ 规则-based + **因果指导** |
| **加密** | ✅ 分层加密 | ✅ 分层加密 |
| **ML推理** | ✅ 密文域推理 | ✅ 密文域推理 |
| **因果分析** | ❌ 无 | ✅ **因果分析（核心创新）** |
| **可解释性** | ⚠️ 有限 | ✅ **因果解释（自然语言）** |
| **可验证性** | ✅ Hash Commitment | ✅ Hash Commitment |

### 4.2 因果分析带来的优势

```
1. 可解释性提升
   - 传统: "为什么对人脸区域加密？" → "规则规定"
   - 因果: "为什么对人脸区域加密？" → 
          "因果分析显示，人脸区域加密的识别率下降效应为-0.95，
           任务区域加密的分类准确率下降效应仅为-0.10，
           因此建议对人脸区域使用强加密"

2. 策略优化
   - 传统: 固定规则
   - 因果: 基于因果效应的动态调整

3. 理论深度
   - 传统: 经验规则
   - 因果: 结构因果模型（SCM）+ ATE/CATE

4. 学术价值
   - 传统: 组合创新
   - 因果: 单点突破（因果推断驱动的加密策略）
```

---

## 五、工作流程时间分析

### 5.1 各阶段时间开销

```
阶段1: 输入准备          < 1ms
阶段2: 语义分析（U-Net）  ~5ms (GPU)
阶段3: 因果分析          ~10ms (CPU, 可优化)
阶段4: 隐私预算分配      < 1ms
阶段5: 分层加密          ~100ms (GPU)
阶段6: ML推理            ~50ms (GPU)
阶段7: 因果效应计算      ~5ms (CPU)
阶段8: 可验证性证明      ~10ms (CPU)

总计: ~181ms (约5.5 FPS)
目标: < 100ms (10+ FPS)

优化方向:
✅ 因果分析GPU加速: 10ms → 2ms
✅ 加密流程优化: 100ms → 80ms
✅ 批处理优化: 提升吞吐量

优化后预计: ~140ms (约7 FPS)
继续优化目标: ~100ms (10 FPS)
```

### 5.2 性能瓶颈

```
瓶颈1: 分层加密（~100ms）
├─ Layer 1: 5D混沌序列生成 (~20ms)
├─ Layer 1: Arnold变换 (~5ms)
├─ Layer 2: FFT变换 (~10ms)
├─ Layer 2: 频域扰动 (~30ms)
└─ Layer 2: IFFT变换 (~10ms)

优化方案:
✅ GPU向量化（已实现）
✅ 批处理优化（待实现）
✅ 预计算混沌序列（待实现）

瓶颈2: 因果分析（~10ms）
├─ 因果图构建: ~2ms
├─ 基线计算: ~5ms
└─ 建议生成: ~3ms

优化方案:
✅ GPU加速（待实现）
✅ 缓存历史数据（待实现）
✅ 简化计算（可选）
```

---

## 六、错误处理与异常情况

### 6.1 常见错误处理

```
1. 语义分析失败
   - 检查: U-Net模型是否加载
   - 处理: 使用默认语义掩码（全1）
   - 日志: 记录错误信息

2. 因果分析失败
   - 检查: 历史数据是否足够
   - 处理: 降级为纯规则-based分配
   - 日志: 记录降级原因

3. 加密失败
   - 检查: 密钥是否正确
   - 检查: 图像格式是否正确
   - 处理: 抛出异常，中断流程

4. ML推理失败
   - 检查: 输入尺寸是否匹配
   - 检查: 数值范围是否在[0,1]
   - 处理: 返回默认预测值

5. 因果效应计算失败
   - 检查: 数据是否完整
   - 处理: 返回部分计算结果
   - 日志: 记录缺失数据
```

---

## 七、工作流程总结

### 7.1 核心流程要点

```
1. **8个阶段**: 输入→语义→因果→分配→加密→推理→因果效应→验证
2. **因果分析**: 阶段3（事前）和阶段7（事后），双重保障
3. **可解释性**: 因果报告生成，自然语言解释
4. **利用现有组件**: 80%组件可直接使用
5. **核心创新**: 因果推断驱动的加密策略
```

### 7.2 关键设计决策

```
✅ **因果分析在前**: 阶段3，指导隐私预算分配
✅ **因果效应在后**: 阶段7，验证加密策略有效性
✅ **规则+因果**: 规则-based + 因果指导，兼顾效率和准确性
✅ **轻量级实现**: Hash Commitment（而非zk-SNARKs）
✅ **可解释输出**: 自然语言因果报告
```

---

**相关文档**:
- [返回总览](Causal-VSE-PC_项目总览.md)
- [数据流向](Causal-VSE-PC_数据流向.md)
- [数据集分析](Causal-VSE-PC_数据集分析与使用.md)
- [目标与指标](Causal-VSE-PC_目标与指标.md)
- [理论证明](Causal-VSE-PC_理论证明.md)
