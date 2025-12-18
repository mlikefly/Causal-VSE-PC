# Causal-VSE-PC 目标与指标

**文档版本**: 2.1.1  
**更新日期**: 2024-12-18  
**目标期刊**: T-IFS / TIP / TNNLS / IEEE Access (SCI二区)

---

## 一、研究问题

### RQ1: 语义区域级隐私预算分配与因果效应估计
我们是否能在 semantic-region 层级实现可控的隐私预算分配，并通过因果效应估计（ATE/CATE）指导预算优化，使 privacy-utility Pareto frontier 优于现有方法？

### RQ2: 自适应攻击下的鲁棒性 (A2)
在 attacker adaptivity（A2 白盒自适应）下，dual-view 架构是否仍然保持隐私优势且不牺牲 utility？

### RQ3: C-view 安全目标与可验证性
C-view 是否满足明确的密码学安全目标：机密性、完整性、抗重放？

### RQ4: 端到端可复现性
实验流程是否可以实现端到端的确定性复现？

---

## 二、核心指标矩阵

### 2.1 隐私保护指标

| 指标 | 目标值 | 当前状态 | 优先级 |
|------|--------|---------|--------|
| 识别率 (TAR@FAR=1e-3) | < 5% | ⚠️ 待验证 | P0 |
| 属性推断 AUC | < 0.6 | ⚠️ 待验证 | P0 |
| 重建 identity_similarity | < 0.3 | ⚠️ 待验证 | P0 |
| 成员推断 AUC | < 0.6 | ⚠️ 待验证 | P0 |
| 属性推断 AUC | < 0.6 | ⚠️ 待验证 | P0 |

### 2.2 任务效用指标

| 指标 | 目标值 | 当前状态 | 优先级 |
|------|--------|---------|--------|
| 分类准确率 (Z2Z, λ=0.3) | ≥ 75% P2P | ⚠️ 待验证 | P0 |
| 分类准确率 (Z2Z, λ=0.5) | ≥ 65% P2P | ⚠️ 待验证 | P0 |
| 分类准确率 (Z2Z, λ=0.7) | ≥ 55% P2P | ⚠️ 待验证 | P0 |
| 分割 mIoU (Z2Z) | ≥ 70% P2P | ⚠️ 待验证 | P1 |

### 2.3 安全性指标

| 指标 | 目标值 | 当前状态 | 优先级 |
|------|--------|---------|--------|
| NPCR | > 99.6% | ✅ 99.57% | P0 |
| UACI | 30-36% | ✅ 33.49% | P0 |
| 信息熵 | > 7.9 bits | ✅ 7.99 | P0 |
| Tamper fail_rate | ≥ 99% | ✅ 实现 | P0 |
| Replay reject_rate | = 100% | ✅ 实现 | P0 |
| NIST p_value | ≥ 0.01 | ⚠️ 待验证 | P1 |
| Avalanche flip_rate | 0.45-0.55 | ⚠️ 待验证 | P1 |

### 2.4 效率指标

| 指标 | 目标值 | 当前状态 | 优先级 |
|------|--------|---------|--------|
| 加密速度 | > 10 FPS | ⚠️ ~5.5 FPS | P1 |
| 解密PSNR | > 40 dB | ✅ 43-63 dB | P0 |
| 解密SSIM | > 0.9 | ✅ 0.99 | P0 |

---

## 三、因果效应指标

### 3.1 ATE (Average Treatment Effect)

```
定义: ATE = E[Y(high_privacy) - Y(low_privacy)]

目标:
├─ 识别任务: ATE < -0.90 (识别率下降 > 90%)
├─ 分类任务: ATE > -0.20 (分类准确率下降 < 20%)
└─ 分割任务: ATE > -0.30 (mIoU下降 < 30%)
```

### 3.2 CATE (Conditional ATE)

```
定义: CATE = E[Y(high) - Y(low) | X=region]

目标:
├─ 敏感区域 (face): CATE < -0.95 (隐私保护强)
├─ 任务区域: CATE > -0.10 (任务可用性保留)
└─ 背景区域: CATE ≈ 0 (无影响)
```

---

## 四、覆盖度要求

### 4.1 组合键定义

```
K_utility = dataset × task × method × training_mode × privacy_level × seed
K_attack = dataset × task × method × training_mode × attack_type × threat_level × privacy_level × seed
K_security = dataset × method × privacy_level × seed × test_type
```

### 4.2 覆盖度门槛

| 类型 | 门槛 | 说明 |
|------|------|------|
| 效用指标 | ≥ 98% | 缺失 → hard fail |
| 攻击指标 | ≥ 98% | 缺失 → hard fail |
| 安全指标 | ≥ 98% | 缺失 → hard fail |

---

## 五、统计要求

### 5.1 默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| seeds | ≥ 3 (建议 5) | 随机种子数 |
| n_boot | ≥ 500 (推荐 1000) | Bootstrap次数 |
| CI | 95% | 置信区间 |
| alpha | 0.05 | 显著性水平 |
| NIST alpha | 0.01 | NIST测试显著性 |
| 多重比较校正 | BH-FDR | 默认方法 |

### 5.2 必需字段

所有CSV必须包含:
- `stat_method`: 统计方法
- `n_boot`: Bootstrap次数
- `ci_low`, `ci_high`: 95% CI
- `family_id`: Family标识
- `alpha`: 显著性水平

---

## 六、R1-R10 红线检查

| # | 红线 | 检查方法 | 通过标准 |
|---|------|----------|----------|
| R1 | 协议版本一致 | protocol_version == schema_version | 必须一致 |
| R2 | 覆盖度 | CoverageChecker | ≥ 98% |
| R3 | A2存在 | attack_metrics.csv | 必须有A2记录 |
| R4 | 重放拒绝率 | replay_results.csv | = 100% |
| R5 | 篡改失败率 | security_metrics_cview.csv | ≥ 99% |
| R6 | C-view隔离 | DataLoader审计 | 训练无C-view |
| R7 | 图表可复现 | figure_manifest.json | SHA256一致 |
| R8 | Nonce唯一 | nonce_log.json | 无重用 |
| R9 | 数据分割 | ManifestBuilder | 零ID重叠 |
| R10 | Schema合规 | ResultsSchema | 字段完整 |

---

## 七、基线对比

### 7.1 必需基线

| Baseline | 类型 | 来源 |
|----------|------|------|
| InstaHide | 近5年代表作 | ICML 2020 |
| P3 | 经典基线 | CVPR 2021 |
| DP-SGD | 差分隐私 | CCS 2016 |
| Pixelation | 传统方法 | - |
| Gaussian Blur | 传统方法 | - |

### 7.2 对比维度

| 维度 | 说明 |
|------|------|
| Utility | 任务效用 |
| Privacy | 隐私保护 |
| Practical | 实用性 |
| Explainability | 可解释性 |
| Strong Security | 强安全性 |

---

## 八、消融实验

### 8.1 消融清单 (12项)

| ID | 消融项 | 描述 |
|----|--------|------|
| A1 | remove_layer1 | 去混沌层 |
| A2 | remove_layer2 | 去频域层 |
| A3 | remove_crypto_wrap | 去AEAD封装 |
| A4 | causal_to_uniform | 因果预算→均匀预算 |
| A5 | causal_to_sensitive_only | 因果预算→仅敏感区域 |
| A6 | causal_to_task_only | 因果预算→仅任务区域 |
| A7 | mask_strong_to_weak | 强监督mask→弱监督mask |
| A8 | fft_to_dwt | FFT→DWT |
| A9 | semantic_preserving_off | 语义保留关闭 |
| A10 | deterministic_nonce_to_random | 确定性nonce→随机nonce |
| A11 | budget_normalization_variantA | 预算归一策略A |
| A12 | budget_normalization_variantB | 预算归一策略B |

---

## 九、成功标准

### 9.1 技术成功

- [x] 所有核心模块实现 ✅
- [x] 协议与Schema基座完成 ✅
- [x] 5类攻击框架 + A2自适应攻击 ✅
- [x] C-view安全评估套件 ✅
- [x] 因果效应估计 (ATE/CATE) ✅
- [x] 统计引擎 (Bootstrap CI + BH-FDR) ✅
- [x] 图表生成器 (8张主图) ✅
- [x] CI集成 (smoke_test + reproduce) ✅
- [ ] 所有P0指标达标（待验证）
- [ ] R1-R10红线全部通过（待验证）
- [ ] 覆盖度 ≥ 98%（待验证）

### 9.2 学术成功

- [ ] 论文撰写完成
- [x] 因果推断理论完整 ✅
- [ ] 实验数据完整
- [x] 创新点突出 ✅

---

## 十、相关文档

- [项目总览](project_overview.md)
- [工作流程](workflow.md)
- [开发日志](development_log.md)
- [数据流向](data_flow.md)
- [数据集分析](dataset_analysis.md)
- [理论证明](theoretical_proof.md)
- [设计文档](../.kiro/specs/top-journal-experiment-suite/design.md)
