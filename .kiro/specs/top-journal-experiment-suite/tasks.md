# Implementation Plan

> **对应设计文档版本**: v2.1.1
> **任务优先级**: P0 (主证据链) → P1 (可复现) → P2 (消融/鲁棒性)

---

## T-REDLINE: 红线检查（Meta-task）

**任何一次 full run 必须满足以下 10 条红线**，由 `validate_run` 输出逐条 ✓/✗：

| # | 红线 | 检查方法 | 对应属性 |
|---|------|----------|----------|
| R1 | protocol_version == schema_version == commit_hash 中的版本 | meta/protocol_version.txt vs SCHEMA_VERSION | Property 13 |
| R2 | coverage ≥ 0.98（按 N/A 规则计算） | CoverageChecker | Property 10 |
| R3 | A2 存在且 attacker_strength=full 的记录存在 | attack_metrics.csv 检查 | Property 14 |
| R4 | replay reject_rate = 100% | tables/replay_results.csv | §9.6.1 |
| R5 | tamper fail_rate ≥ 99% | tables/security_metrics_cview.csv | Property 4 |
| R6 | c_view guard 无泄漏（训练集中无 c_view） | DataLoader 审计日志 | Property 6 |
| R7 | figure_manifest SHA256 可复现 | reports/figure_manifest.json | Property 9 |
| R8 | nonce 无重用 | meta/nonce_log.json 唯一性检查 | §5.5.3 |
| R9 | train/val/test 零 ID 重叠 | ManifestBuilder 检查 | Property 12 |
| R10 | 所有 CSV 字段完整且类型正确 | ResultsSchema 验证 | Property 2 |

**Evidence**: `reports/validate_run_onepage.md` 必须包含 R1-R10 的逐条 ✓/✗

---

## P0: 顶刊主证据链（必须先跑通 C1-C4）

### T0. 文档一致性补丁（已完成）

- [x] 0.1 修复 §10.3 旧版 nonce 派生代码
  - 改为引用 §5.5.3 NonceManager
  - _Requirements: 文档一致性_

- [x] 0.2 修复 §12.1 ProtocolManager 版本号
  - 统一为 2.1.1
  - _Requirements: 文档一致性_

- [x] 0.3 修复 ReplayCache key 碰撞风险
  - 改用 full_tag 确保零碰撞
  - _Requirements: §9.6.1.1_

- [x] 0.4 新增 CI 例外条款
  - smoke_test lite 模式与 A2 强度契约兼容
  - _Requirements: §10.4_

---

### T1. 协议与 Schema 基座

**Inputs/Outputs Contract**:
- 输入: config.yaml (完整配置)
- 输出: meta/protocol_version.txt, reports/protocol_snapshot.md, reports/schema_validation.json, reports/coverage_report.json
- 必填字段: 见 §10.1 UTILITY_FIELDS, ATTACK_FIELDS

**Evidence**: `reports/schema_validation.json`, `reports/coverage_report.json`

- [x] 1. 实现 ProtocolManager


  - [x] 1.1 实现协议版本管理


    - PROTOCOL_VERSION = "2.1.1"
    - SCHEMA_VERSION = "2.1.1"
    - 实现 write_protocol_version() 和 write_protocol_snapshot()
    - _Requirements: §8.4, Property 13_
  - [x] 1.2 实现 validate_consistency()

    - 验证 protocol_version == schema_version
    - 验证 protocol_snapshot.md 与 config.yaml 一致
    - _Requirements: Property 13_
  - [ ]* 1.3 编写 ProtocolManager 属性测试
    - **Property 13: 协议版本一致性**
    - **Validates: Requirements 1.7, 1.8, GC2**

- [x] 2. 实现 ResultsSchema


  - [x] 2.1 定义 UTILITY_FIELDS（冻结）

    - dataset, task, method, training_mode, privacy_level, seed, metric_name, metric_value, relative_to, relative_performance, ci_low, ci_high, stat_method, n_boot, family_id, alpha
    - _Requirements: §10.1_

  - [x] 2.2 定义 ATTACK_FIELDS（冻结）
    - 新增 attacker_strength 字段（lite/full）
    - lite 模式必须带 degrade_reason
    - _Requirements: §10.1, §10.4 CI例外条款_
  - [x] 2.3 实现字段验证

    - 类型检查、枚举域验证、必填检查
    - _Requirements: Property 2_
  - [ ]* 2.4 编写 ResultsSchema 属性测试
    - **Property 2: CSV Schema 合规性**
    - **Validates: Requirements 1.3, 2.6, 2.9, GC6, GC7**

- [x] 3. 实现 ValidateRun


  - [x] 3.1 实现目录结构检查

    - meta/, tables/, figures/, logs/, reports/ 必须存在
    - _Requirements: Property 1_

  - [x] 3.2 实现 Schema 检查
    - 调用 ResultsSchema 验证所有 CSV
    - _Requirements: Property 2_

  - [x] 3.3 实现覆盖度检查
    - 覆盖度 < 98% 必须 hard fail
    - 输出 missing_matrix.csv
    - _Requirements: Property 10_
  - [x] 3.4 实现审稿人一页报告（含 R1-R10 红线）

    - ✓/✗ 每项检查状态、失败原因、缺失组合、复现命令
    - _Requirements: §10.0.3, T-REDLINE_
  - [ ]* 3.5 编写 ValidateRun 属性测试
    - **Property 1: 运行目录结构完整性**
    - **Property 10: 覆盖度门槛强制执行**
    - **Validates: Requirements 1.7, 7.1-7.8, GC4**

- [x] 4. Checkpoint - 确保所有测试通过

  - Ensure all tests pass, ask the user if questions arise.

---

### T2. NonceManager + ReplayCache（C-view 安全目标落地）

**Inputs/Outputs Contract**:
- 输入: master_key (bytes), run_dir (Path), NonceDerivationInput (image_id, method, privacy_level, training_mode, purpose)
- 输出: meta/nonce_log.json, meta/replay_cache.json, tables/replay_results.csv
- 约束: nonce 必须唯一，replay reject_rate = 100%

**Evidence**: `meta/nonce_log.json`, `tables/replay_results.csv`

- [x] 5. 实现 NonceManager



  - [x] 5.1 实现 NonceDerivationInput 数据类

    - image_id, method, privacy_level, training_mode, purpose
    - _Requirements: §5.5.3_

  - [x] 5.2 实现 derive_nonce() 方法

    - nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
    - 唯一性检查 + NonceReuseError

    - _Requirements: §5.5.3, RQ3_

  - [x] 5.3 实现 nonce_log.json 落盘
    - 记录 image_id, method, privacy_level, training_mode, purpose, nonce_hex, timestamp
    - _Requirements: §5.5.3_
  - [ ]* 5.4 编写 NonceManager 属性测试
    - **Property 4: C-view 安全测试完整性（nonce 唯一性部分）**
    - **Validates: Requirements RQ3, §5.5.3**

- [x] 6. 实现 ReplayCache



  - [x] 6.1 实现 check_and_record() 方法

    - key = (key_id, nonce, full_tag)
    - 返回 True（新密文）或 False（重放）
    - _Requirements: §9.6.1.1_

  - [x] 6.2 实现 persist() 方法
    - 落盘到 meta/replay_cache.json
    - _Requirements: §9.6.1.1_
  - [ ]* 6.3 编写 ReplayCache 属性测试
    - **Property: Replay reject_rate = 100%**
    - **Validates: Requirements §9.6.1, RQ3**

- [x] 7. Checkpoint - 确保所有测试通过

  - Ensure all tests pass, ask the user if questions arise.

---

### T3. 五类攻击 + A2 强度契约执行

**Inputs/Outputs Contract**:
- 输入: z_view (Tensor), attack_type, threat_level, privacy_level, attacker_strength (lite/full)
- 输出: tables/attack_metrics.csv, tables/table_a2.csv, reports/protocol_snapshot.md (A2 契约部分)
- 必填字段: attack_type, threat_level, attack_success, attacker_strength
- 约束: A2 必须存在且 attacker_strength=full；lite 模式必须带 degrade_reason

**Evidence**: `tables/attack_metrics.csv`, `tables/table_a2.csv`, `reports/protocol_snapshot.md`

- [x] 8. 实现攻击框架




  - [x] 8.1 实现 AttackFitContext 数据类

    - run_id, dataset, task, method, training_mode, privacy_level, seed, threat_level, attacker_visible
    - _Requirements: §6.2_

  - [x] 8.2 实现 AttackBase 抽象类
    - fit(), evaluate(), get_attack_success()

    - _Requirements: §6.2_
  - [x] 8.3 实现 attack_success 映射（GC7）
    - face_verification → TAR@FAR=1e-3
    - attribute_inference → AUC
    - reconstruction → identity_similarity
    - membership_inference → AUC
    - property_inference → AUC
    - _Requirements: §7.2, Property 3_
  - [ ]* 8.4 编写攻击框架属性测试
    - **Property 3: 攻击成功率映射一致性**
    - **Validates: Requirements 2.9, GC7**

- [x] 9. 实现五类攻击



  - [x] 9.1 实现 FaceVerificationAttack

    - _Requirements: R2_

  - [x] 9.2 实现 AttributeInferenceAttack

    - _Requirements: R2_

  - [x] 9.3 实现 ReconstructionAttack
    - _Requirements: R2_

  - [x] 9.4 实现 MembershipInferenceAttack
    - Shadow Models 方法
    - _Requirements: R2.AC1_
  - [x] 9.5 实现 PropertyInferenceAttack
    - 群体属性分布推断
    - _Requirements: R2.AC2_

- [x] 10. 实现 A2 强度契约


  - [x] 10.1 实现 AdaptiveAttacker

    - design_adaptive_strategy() 方法
    - _Requirements: §6.3_

  - [x] 10.2 实现攻击族谱（3 family）
    - Reconstruction: U-Net decoder, GAN-based inversion
    - Inference: Linear probe, MLP classifier, Contrastive learning
    - Optimization: Gradient-based, Evolutionary search

    - _Requirements: §5.4.1_
  - [x] 10.3 实现攻击预算冻结

    - 100 epochs, LR 搜索 {1e-4, 1e-3, 1e-2}, ≤24h/family
    - _Requirements: §5.4.2_

  - [x] 10.4 实现 worst-case 聚合
    - worst_case_attack_success = max(attack_success) over same (dataset, task, privacy_level, threat_level)
    - _Requirements: §5.4.3_
  - [x] 10.5 实现 A2 强制存在检查
    - attack_metrics.csv 必须包含 threat_level=A2
    - 缺少 → hard fail
    - _Requirements: Property 14_
  - [ ]* 10.6 编写 A2 属性测试
    - **Property 14: A2 攻击强制存在**
    - **Validates: Requirements 16.1, 16.3, 16.5**

- [x] 11. Checkpoint - 确保所有测试通过

  - Ensure all tests pass, ask the user if questions arise.

---

### T4. attack_success 归一化 + 汇总指标

**Inputs/Outputs Contract**:
- 输入: attack_type, attack_success, threat_level, attacker_strength (必填)
- 输出: tables/normalized_metrics.csv, tables/privacy_summary.csv
- 输出必须补齐: privacy_protection, normalized_attack_success
- 约束: lite 模式必须带 degrade_reason；归一化后范围 [0, 1]

**Evidence**: `tables/normalized_metrics.csv`, `tables/privacy_summary.csv`

- [x] 12. 实现归一化系统



  - [x] 12.1 实现 AttackNormalizer


    - 按 §7.3 定义的上下界归一化
    - face_verification: (x - x_random) / (x_P2P - x_random)
    - attribute_inference: (x - 0.5) / (x_P2P - 0.5)
    - reconstruction: x / x_P2P
    - membership_inference: (x - 0.5) / (x_P2P - 0.5)
    - property_inference: (x - 0.5) / (x_P2P - 0.5)
    - _Requirements: §7.3_

  - [x] 12.2 实现 privacy_protection 计算
    - privacy_protection = 1 - normalized(attack_success)

    - _Requirements: §7.1_
  - [x] 12.3 实现汇总指标
    - avg_privacy_protection: 均匀权重
    - worst_case_privacy_protection: min
    - weighted_privacy_protection: A0:0.2, A1:0.3, A2:0.5
    - _Requirements: §7.4_
  - [x]* 12.4 编写归一化属性测试

    - **Property: 归一化后范围 [0, 1]**
    - **Validates: Requirements §7.3, §7.4**

- [x] 13. Checkpoint - 确保所有测试通过

  - Ensure all tests pass, ask the user if questions arise.

---

### T5. C-view 安全评估套件

**Inputs/Outputs Contract**:
- 输入: c_view (加密数据), key, nonce, aad
- 输出: tables/security_metrics_cview.csv, tables/replay_results.csv, reports/security_report.md
- 约束: tamper fail_rate ≥ 99%, replay reject_rate = 100%
- NIST/Avalanche 失败不 hard fail，但需说明原因

**Evidence**: `tables/security_metrics_cview.csv`, `reports/security_report.md`

- [x] 14. 实现安全目标验证（主证据）

  - [x] 14.1 实现 TamperTester

    - 3 种 tamper 类型: ciphertext, tag, aad
    - fail_rate ≥ 99%
    - _Requirements: §9.6.1, Property 4_

  - [x] 14.2 实现 ReplayTester
    - 使用 ReplayCache
    - reject_rate = 100%
    - _Requirements: §9.6.1_
  - [ ]* 14.3 编写安全目标属性测试
    - **Property 4: C-view 安全测试完整性**
    - **Validates: Requirements 3.4, 3.5, 3.6**

- [x] 15. 实现诊断证据（NIST/Avalanche）

  - [x] 15.1 实现 NISTTestRunner

    - 7 项子测试: frequency, block_frequency, runs, longest_run, fft, serial, approximate_entropy
    - p_value ≥ 0.01
    - 记录 nist_bits
    - _Requirements: §9.6.2, Property 5_

  - [x] 15.2 实现 AvalancheEffectTester
    - 3 种 flip 类型: key, nonce, plaintext
    - flip_rate ∈ [0.45, 0.55]

    - _Requirements: §9.6.2_
  - [x] 15.3 生成 security_report.md

    - 主证据 + 诊断证据汇总
    - NIST/Avalanche 失败不 hard fail，但需说明原因
    - _Requirements: §9.6.2_
  - [ ]* 15.4 编写诊断属性测试
    - **Property 5: NIST 比特流充足性**
    - **Validates: Requirements 3.1, 3.2, 3.3**

- [x] 16. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

## P1: 顶刊可复现（审稿人一键复现）

### T6. 训练模式与效用评估

**Inputs/Outputs Contract**:
- 输入: training_mode (P2P/P2Z/Z2Z/Mix2Z), split (train/val/test)
- 输出: tables/utility_metrics.csv, reports/utility_failure_analysis.md (触发时)
- 约束: 训练时请求 c_view → ViewAccessError；relative_performance < 门槛 → 生成分析报告

**Evidence**: `tables/utility_metrics.csv`, DataLoader 审计日志

- [x] 17. 实现 TrainingModeManager

  - [x] 17.1 实现 4 种训练模式

    - P2P, P2Z, Z2Z, Mix2Z
    - _Requirements: §8.1_

  - [x] 17.2 实现 C-view Guard
    - 训练 DataLoader 请求 c_view → ViewAccessError
    - _Requirements: Property 6_
  - [ ]* 17.3 编写训练模式属性测试
    - **Property 6: 训练模式数据隔离**
    - **Validates: Requirements 4.1, 4.2, 11.3, 11.6**

- [x] 18. 实现 UtilityEvaluator


  - [x] 18.1 实现 relative_performance 计算
    - relative_performance = metric_value / P2P_mean
    - relative_to = "P2P_mean"
    - _Requirements: Property 7_
  - [x] 18.2 实现门槛检查
    - 0.3 → 75% P2P, 0.5 → 65% P2P, 0.7 → 55% P2P
    - _Requirements: §9.3_
  - [x] 18.3 实现失败分析报告
    - relative_performance < 门槛 → utility_failure_analysis.md
    - _Requirements: Property 7_
  - [ ]* 18.4 编写效用评估属性测试
    - **Property 7: 效用门槛计算**
    - **Validates: Requirements 4.3, 4.4, 4.6, 4.8**

- [x] 19. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T7. 因果两阶段：干预网格 → ATE/CATE → 预算优化

**Inputs/Outputs Contract**:
- 输入: regions, beta_values, n_samples_per_cell, utility_threshold
- 输出: tables/causal_effects.csv, tables/optimal_allocation.json, figures/fig_causal_ate_cate.png
- 约束: ATE/CATE 必须含 CI；优化目标 min max_a attack_success s.t. utility >= threshold

**Evidence**: `tables/causal_effects.csv`, `tables/optimal_allocation.json`, `figures/fig_causal_ate_cate.png`

- [x] 20. 实现干预网格
  - [x] 20.1 实现 InterventionGrid 数据类
    - regions: ["face", "background", "sensitive_attr"]
    - beta_values: [0.0, 0.25, 0.5, 0.75, 1.0]
    - n_samples_per_cell: 100
    - _Requirements: §9.2_
  - [x] 20.2 实现 generate_experiments() 方法
    - 生成所有干预实验配置
    - _Requirements: §9.2_

- [x] 21. 实现因果效应估计
  - [x] 21.1 实现 CausalBudgetOptimizer.estimate_ate()
    - ATE = E[A|do(β=1)] - E[A|do(β=0)]
    - _Requirements: §2.5, §9.2_
  - [x] 21.2 实现 estimate_cate()
    - CATE = E[A|do(β=1), X=x] - E[A|do(β=0), X=x]
    - _Requirements: §2.5_
  - [x] 21.3 生成 causal_effects.csv
    - 每个区域的 ATE/CATE + CI
    - _Requirements: §9.2_

- [x] 22. 实现预算优化求解
  - [x] 22.1 实现 solve_optimal_allocation()
    - 目标: min max_a attack_success_a
    - 约束: utility >= utility_threshold
    - _Requirements: §2.5, §9.2_
  - [x] 22.2 生成 optimal_allocation.json
    - _Requirements: §9.2_
  - [ ]* 22.3 编写因果效应属性测试
    - **Property: ATE/CATE + CI 完整性**
    - **Validates: Requirements C1, §9.2**

- [x] 23. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T8. 基线矩阵与对比

**Inputs/Outputs Contract**:
- 输入: baseline_name, task, training_mode
- 输出: tables/baseline_comparison.csv, reports/baseline_matrix_report.md
- 约束: 至少包含 InstaHide + P3；N/A 不计入缺失覆盖度

**Evidence**: `tables/baseline_comparison.csv`, `reports/baseline_matrix_report.md`

- [x] 24. 实现 BaselineComparator
  - [x] 24.1 实现 5 个基线
    - InstaHide, P3, DP-SGD, Pixelation, Gaussian Blur
    - _Requirements: §9.4_
  - [x] 24.2 实现 N/A 覆盖规则
    - 不支持项记录为 "N/A"
    - N/A 不计入缺失覆盖度
    - _Requirements: §9.4_
  - [x] 24.3 生成 baseline_comparison.csv
    - 至少包含 InstaHide + P3
    - _Requirements: Property 15_
  - [ ]* 24.4 编写基线对比属性测试
    - **Property 15: 基线对比完整性**
    - **Validates: Requirements 8.1, 8.2, 8.6, 8.7**

- [x] 25. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T9. 统计引擎

**Inputs/Outputs Contract**:
- 输入: values (array), run_seed, dataset, task, metric_name, privacy_level
- 输出: 所有 CSV 补齐 ci_low, ci_high, n_boot, family_id, alpha
- 约束: n_boot ≥ 500；family_id 确定性（相同输入 → 相同输出）

**Evidence**: 所有 CSV 的统计字段完整性

- [x] 26. 实现 StatisticsEngine
  - [x] 26.1 实现 compute_ci()
    - bootstrap CI, n_boot ≥ 500
    - _Requirements: Property 11_
  - [x] 26.2 实现 multiple_comparison_correction()
    - BH-FDR 校正
    - _Requirements: R10.2_
  - [x] 26.3 实现 generate_family_id()
    - family_id = sha1(f"{dataset}|{task}|{metric_name}|{privacy_level}")[:10]
    - _Requirements: Property 16, GC9_
  - [ ]* 26.4 编写统计引擎属性测试
    - **Property 11: 统计严谨性完整性**
    - **Property 16: Family ID 确定性**
    - **Validates: Requirements 10.1, 10.4, 10.5, 10.6, GC3, GC9**

- [x] 27. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T10. 图表生成 + 字节级复现

**Inputs/Outputs Contract**:
- 输入: tables/*.csv
- 输出: figures/*.png, reports/figure_manifest.json
- 约束: 从 CSV 重建 PNG，SHA256 字节一致

**Evidence**: `reports/figure_manifest.json` (SHA256 哈希)

- [x] 28. 实现 FigureGenerator
  - [x] 28.1 实现 8 张主图生成
    - fig_utility_curve, fig_attack_curves, fig_pareto_frontier, fig_causal_ate_cate
    - fig_cview_security_summary, fig_ablation_summary, fig_efficiency, fig_robustness
    - _Requirements: §12.6_
  - [x] 28.2 实现 FigureSpecs
    - DPI=300, 字体优先级, 尺寸规格
    - _Requirements: §12.6_
  - [x] 28.3 生成 figure_manifest.json
    - 每张图的 SHA256 哈希
    - _Requirements: Property 9_
  - [ ]* 28.4 编写图表属性测试
    - **Property 9: 图表可复现性**
    - **Validates: Requirements 6.1, 6.7, 6.8**

- [x] 29. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T11. CI 集成

**Inputs/Outputs Contract**:
- 输入: config.yaml, mode (smoke_test/full)
- 输出: scripts/smoke_test.sh, scripts/reproduce.sh, ARTIFACT_CHECKLIST.md, reports/validate_run_onepage.md
- 约束: smoke_test < 20min, attacker_strength=lite；full 模式用于主证据

**Evidence**: `reports/validate_run_onepage.md` (含 R1-R10 红线检查)

- [x] 30. 实现 smoke_test
  - [x] 30.1 实现 smoke_test.sh
    - 时间预算 < 20 min
    - attacker_strength=lite
    - _Requirements: §10.4_
  - [x] 30.2 实现 lite 模式约束
    - 5 epochs, 1 实例化, 子集数据
    - 必须标记 attacker_strength=lite
    - 不用于主证据
    - _Requirements: §10.4 CI例外条款_

- [x] 31. 实现 reproduce.sh
  - [x] 31.1 实现一键复现脚本
    - 使用 full 模式
    - _Requirements: §10.4_
  - [x] 31.2 生成 ARTIFACT_CHECKLIST.md
    - _Requirements: C4_

- [x] 32. 实现数据泄漏检查
  - [x] 32.1 实现 split 泄漏检查
    - train/val/test 零 ID 重叠
    - _Requirements: Property 12_
  - [x] 32.2 实现 shadow 分割检查
    - shadow 分割与评估样本严格分离
    - _Requirements: Property 12_
  - [ ]* 32.3 编写数据泄漏属性测试
    - **Property 12: 数据泄漏防护**
    - **Validates: Requirements 11.1, 11.2, 11.5**

- [x] 33. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

## P2: 消融与鲁棒性收尾（论文附录）

### T12. 12 项消融实验

**Inputs/Outputs Contract**:
- 输入: ablation_id (A1-A12), config_override
- 输出: tables/ablation.csv, figures/fig_ablation_summary.png
- 约束: 12 项全跑，每项同时产出效用 + 攻击指标

**Evidence**: `tables/ablation.csv`, `figures/fig_ablation_summary.png`

- [x] 34. 实现消融运行器
  - [x] 34.1 实现 12 项消融配置
    - A1-A12 按 §11.1 定义
    - _Requirements: §11.1_
  - [x] 34.2 实现消融运行器
    - 每项同时产出效用 + 攻击指标
    - _Requirements: Property 8_
  - [x] 34.3 生成 ablation.csv
    - _Requirements: Property 8_
  - [x] 34.4 生成 fig_ablation_summary.png
    - _Requirements: §12.6_
  - [ ]* 34.5 编写消融属性测试
    - **Property 8: 消融实验目录完整性**
    - **Validates: Requirements 5.1-5.6**

- [x] 35. Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

### T13. 稳健性与效率图（可选）

- [x]* 36. 实现稳健性评估
  - [x]* 36.1 生成 robustness_metrics.csv
  - [x]* 36.2 生成 fig_robustness.png

- [x]* 37. 实现效率评估
  - [x]* 37.1 生成 efficiency.csv
  - [x]* 37.2 生成 fig_efficiency.png

- [x] 38. Final Checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

---

## 任务依赖图

```
T0 (文档补丁) ✓
    │
    ▼
T1 (协议/Schema) ──────────────────────────────────────┐
    │                                                   │
    ├──► T2 (Nonce/Replay) ──► T5 (安全评估)           │
    │                                                   │
    ├──► T3 (攻击框架) ──► T4 (归一化) ──► T7 (因果)   │
    │         │                                         │
    │         └──► T8 (基线对比)                        │
    │                                                   │
    └──► T6 (训练模式) ──► T7 (因果)                   │
                                                        │
T9 (统计引擎) ◄─────────────────────────────────────────┘
    │
    ▼
T10 (图表生成)
    │
    ▼
T11 (CI集成)
    │
    ▼
T12 (消融) ──► T13 (稳健性/效率)
```

---

*任务列表版本: 1.1.0*
*对应设计文档: v2.1.1*
*最后更新: 2024-12*
*增强: T-REDLINE 红线检查 + Inputs/Outputs Contract + Evidence*
