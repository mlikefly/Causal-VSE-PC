# Requirements Document

## Introduction

本需求文档定义了 Causal-VSE-PC 项目冲击顶刊（T-IFS/TIP/TNNLS/IEEE Access SCI二区）的完整实验套件升级。基于已完成的 Phase 0-5（双视图架构、数据流水线、攻击评估框架），本阶段聚焦于：

1. **协议冻结与文档规范** - 威胁模型、结果 schema、数据说明
2. **攻击评估完善** - 补充 Membership Inference 和 Property Inference 两类攻击（共5类）
3. **C-view 传统安全全套** - NIST SP800-22 完整测试、Avalanche、Tamper/Replay
4. **训练策略升级** - Train-on-Z、Mixed Training 实现 Z-view 可用性
5. **消融实验矩阵** - ≥12 项消融实验
6. **论文级输出工程化** - 一键生成图表、结果校验
7. **安全主张边界声明** - 明确混沌层与AEAD的安全边界
8. **A2白盒自适应攻击** - 覆盖最强威胁等级

## Global Constraints（全局约束）

所有 Requirements 默认继承以下约束：

### GC1: 统一结果目录（run_dir）硬约束
- 所有脚本必须写入：`results/{exp_name}/{run_id}/`
- 必须包含子目录：`meta/`、`tables/`、`figures/`、`logs/`、`reports/`
- **Failure Handling**: 缺任何一个子目录 → hard fail

### GC2: 统一协议版本
- `protocol_version` 必须写入 `meta/protocol_version.txt`
- 当 schema 或协议变更，必须 bump version
- **Failure Handling**: 未 bump version 视为不可审计 → hard fail

### GC3: 统计默认值
- seeds ≥ 3（建议 5）
- bootstrap ≥ 500（推荐 1000）
- CI = 95%
- 显著性 alpha = 0.05
- 多重比较校正默认 BH-FDR（或 Holm-Bonferroni）
- **Failure Handling**: 未按要求写 `stat_method`/`n_boot`/`p_adj` → hard fail

### GC4: 覆盖度默认值（Coverage Gate）
定义组合键：
- `K_utility = dataset × task × method × training_mode × privacy_level × seed`
- `K_attack = dataset × task × method × training_mode × attack_type × threat_level × privacy_level × seed`
- `K_security = dataset × method × privacy_level × seed × (nist_test_name/avalanche_type/tamper_test)`
- 每个 csv 必须覆盖 ≥ 98% 的期望组合
- **Failure Handling**: 覆盖不足 → hard fail 并输出 `reports/missing_matrix.csv`

### GC5: 失败记录强制落盘
任何"指标未达门槛/某子任务失败/攻击训练崩溃"必须输出：
- `reports/failure_modes.md`（包含原因分类 + 样例 id + 可视化路径）
- `reports/missing_matrix.csv`（缺失组合列表）

### GC6: privacy_level 网格固定
- 只允许 {0.0, 0.3, 0.5, 0.7, 1.0} 五档
- **Failure Handling**: 使用其他值 → hard fail

### GC7: attack_success 统一映射表（冻结）
每个 attack_type 的 attack_success 字段必须按以下映射计算（越低越好=隐私保护越强）：
| attack_type | attack_success 定义 | 方向 |
|-------------|---------------------|------|
| face_verification | TAR@FAR=1e-3 | ↓ |
| attribute_inference | AUC | ↓ |
| membership_inference | AUC（advantage 单独列） | ↓ |
| property_inference | AUC | ↓ |
| reconstruction | identity_similarity | ↓ |
- **Failure Handling**: 使用非标准映射 → hard fail

### GC8: 显著性比较 baseline 冻结
- Utility baseline：P2P（plaintext 上限）
- Privacy baseline：plaintext（或 P2Z，固定为 plaintext）
- Baseline methods 对比：InstaHide/P3 需先做 strength_param -> privacy_level 映射后再比较
- **Failure Handling**: baseline 漂移（同一 csv 中 baseline 不一致）→ hard fail

### GC9: family_id 编码规则（冻结）
- family_id 生成规则：`family_id = sha1(f"{dataset}|{task}|{metric_name}|{privacy_level}")[:10]`
- 所有 csv 必须使用相同的 family_id 生成逻辑
- **Failure Handling**: family_id 不一致 → hard fail

### GC10: Degrade 允许边界（冻结）
- **允许 degrade**：A0/A1 中计算成本过高的攻击者（需记录 attacker_strength=lite）
- **不允许 degrade（必须 hard fail）**：
  - A2 自适应攻击
  - C-view tamper/replay 测试
  - validate_run 覆盖度检查
  - 协议版本一致性检查
  - NIST 随机性测试
- **Failure Handling**: 在不允许 degrade 的场景使用 degrade → hard fail

## Glossary

- **Threat Model** - 威胁模型，定义攻击者能力等级（A0黑盒/A1灰盒/A2白盒自适应）
- **A0 (Black-box)** - 攻击者仅能访问输出，不知道算法细节
- **A1 (Gray-box)** - 攻击者知道算法但无法访问密钥或模型参数
- **A2 (White-box Adaptive)** - 攻击者知道算法与模型结构，可自适应设计攻击策略，但无密钥
- **Membership Inference Attack** - 成员推断攻击，判断样本是否在训练集中
- **Property Inference Attack** - 属性推断攻击，推断数据集的群体属性（如种族/性别分布）
- **NIST SP800-22** - 美国国家标准与技术研究院的随机性测试套件
- **Avalanche Effect** - 雪崩效应，输入微小变化导致输出大幅变化（理想≈50%比特翻转）
- **Train-on-Z (Z2Z)** - 在 Z-view 密文上训练，在 Z-view 上测试
- **Mixed Training (Mix2Z)** - 混合训练，同时使用明文和 Z-view 进行训练
- **Ablation Study** - 消融实验，逐步移除组件以验证各部分贡献
- **Pareto Frontier** - 帕累托前沿，隐私-效用权衡的最优边界
- **DoD (Definition of Done)** - 完成定义，任务验收标准
- **AEAD** - Authenticated Encryption with Associated Data，带关联数据的认证加密
- **IND-CPA/IND-CCA** - 选择明文/密文攻击下的不可区分性安全定义
- **BH-FDR** - Benjamini-Hochberg False Discovery Rate，多重比较校正方法
- **Shadow Models** - 影子模型，用于成员推断攻击的训练方法

## Requirements

### Requirement 1: 实验协议冻结

**User Story:** As a 研究者, I want to 将所有实验协议写入规范文档, so that 审稿人无法从"协议不严谨/不复现"角度质疑。

#### Acceptance Criteria

1. WHEN 定义威胁模型 THEN 系统 SHALL 在 EXPERIMENT_PROTOCOL.md 中明确 A0/A1/A2 三级攻击者能力，包括每级攻击者可见信息与不可见信息
2. WHEN 定义 Z/C 视图边界 THEN 系统 SHALL 明确 Z-view 用攻击成功率证明隐私、C-view 用传统密码学测试证明安全
3. WHEN 定义结果 schema THEN 系统 SHALL 在 RESULTS_SCHEMA.md 中冻结所有 CSV 字段、类型、单位、方向（↑好/↓好）
4. WHEN 定义数据获取 THEN 系统 SHALL 在 DATASETS.md 中说明每个数据集的许可、下载方式、预处理命令
5. WHEN 定义 privacy_level 网格 THEN 系统 SHALL 固定为 {0.0, 0.3, 0.5, 0.7, 1.0} 五档，不允许其他值
6. WHEN 定义统计方式 THEN 系统 SHALL 明确 seeds≥3、bootstrap≥500、CI=95%、alpha=0.05、NIST alpha=0.01、多重比较校正方法（BH-FDR）
7. WHEN 生成 run 目录 THEN 系统 SHALL 输出 meta/protocol_version.txt 与 reports/protocol_snapshot.md
8. WHEN validate_run 检查协议 THEN 系统 SHALL 验证 protocol_snapshot.md 与 meta/config.yaml 一致（配置冻结后不可漂移）

#### Artifacts
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SCHEMA.md`
- `docs/DATASETS.md`
- `docs/RELATED_WORK_2020_2025.md`
- 每个 run 的 `meta/protocol_version.txt`
- 每个 run 的 `reports/protocol_snapshot.md`

#### Coverage Gate
- validate_run 必须检查 protocol_snapshot.md 与 meta/config.yaml 一致
- privacy_level 只能是 {0.0, 0.3, 0.5, 0.7, 1.0}
- 每个指标/攻击都有定义与方向声明

#### Failure Handling
- 缺任何文档 → PR 不允许合并（CI hard fail）
- 协议更新必须 bump protocol_version，否则 hard fail
- privacy_level 使用非法值 → hard fail

#### Notes
- Stat family: 在 EXPERIMENT_PROTOCOL.md 中明确定义
- 每个指标/攻击都有定义与方向（↑好/↓好）
- NIST alpha=0.01，其他显著性 alpha=0.05

### Requirement 2: 攻击评估完善（5类攻击）

**User Story:** As a 研究者, I want to 实现完整的 5 类攻击评估, so that 隐私保护的有效性有充分的攻击证据支撑。

#### Acceptance Criteria

1. WHEN 评估 Membership Inference 攻击 THEN 系统 SHALL 使用 Shadow Models 方法输出 Attack AUC 和 Advantage，并确保 shadow split 与被评测样本严格分离
2. WHEN 评估 Property Inference 攻击 THEN 系统 SHALL 预测群体属性（race/gender 分布）并输出准确率/AUC + CI
3. WHEN 评估 Face Verification 攻击 THEN 系统 SHALL 在 LFW 外部验证集上报告 AUC、TAR@FAR=1e-3、EER
4. WHEN 评估 Attribute Inference 攻击 THEN 系统 SHALL 预测敏感属性并输出 AUC/Acc + CI
5. WHEN 评估 Reconstruction 攻击 THEN 系统 SHALL 输出 mask 内 PSNR/SSIM + identity similarity（使用预训练人脸识别模型）
6. WHEN 输出攻击指标 THEN 系统 SHALL 统一写入 attack_metrics.csv 并包含 threat_level（A0/A1/A2）字段
7. WHEN 攻击成功率随 privacy_level 变化 THEN 系统 SHALL 输出完整曲线并计算 95% CI
8. WHEN 运行任意攻击 THEN 系统 SHALL 记录攻击训练超参（epochs/lr/batch_size/early_stop/输入视图/攻击者可见信息）写入 meta/attack_hparams.yaml
9. WHEN 输出攻击指标 THEN 系统 SHALL 同时输出 attack_success 统一字段，并在 RESULTS_SCHEMA.md 中声明方向（越低越好=隐私保护越强）
10. WHEN 运行攻击失败 THEN 系统 SHALL 将失败样本与原因写入 reports/failure_modes.md 并在 csv 中写 status=failed（同时 validate_run hard fail，除非配置允许 degrade）

#### Artifacts
- `tables/attack_metrics.csv`
- `figures/fig_attack_curves.png`（五类攻击曲线+CI，标注 threat_level）
- `meta/attack_hparams.yaml`
- `reports/attack_report.md`（每类攻击的解释、威胁等级、训练成本、攻击者可见信息）
- `reports/missing_matrix.csv`（若缺失组合）

#### Coverage Gate
- attack_types 必须包含：face_verification, attribute_inference, reconstruction, membership_inference, property_inference
- 每个 attack_type 至少覆盖：
  - datasets ≥ 2（建议 3）
  - tasks ≥ 1（建议 2：分类+分割）
  - methods ≥ 3（主方法+2 baselines）
  - privacy_levels = 5 档全覆盖
  - seeds ≥ 3
- 覆盖度 < 98%：hard fail

#### Failure Handling
- 若某攻击因算力不可行，可允许降级（轻量攻击者），但必须：
  - 在 EXPERIMENT_PROTOCOL.md 明确降级原因
  - 在 attack_report.md 写"与强攻击的差异与风险"
  - 在结果里标注 attacker_strength=lite/full
- 攻击训练崩溃 → 写入 failure_modes.md + status=failed

#### Notes
- Stat family: dataset × task × attack_type × metric_name
- 所有攻击指标方向统一：attack_success 越低越好（隐私保护越强）
- threat_level 必须在 {A0, A1, A2} 中选择

### Requirement 3: C-view 传统安全全套

**User Story:** As a 研究者, I want to 对 C-view 执行完整的密码学随机性测试, so that 满足安全向顶刊的审稿要求。

#### Acceptance Criteria

1. WHEN 执行 NIST SP800-22 测试 THEN 系统 SHALL 至少运行 7 项测试（Frequency、BlockFrequency、Runs、LongestRun、FFT、Serial、ApproxEntropy）
2. WHEN 执行 NIST 测试 THEN 系统 SHALL 记录 bitstream 总长度（nist_bits）并确保满足各子测试最低长度要求；若不足 SHALL 自动拼接多个 ciphertext 直到达到阈值
3. WHEN 评估 Avalanche 效应 THEN 系统 SHALL 测试 key flip、nonce flip、plaintext flip 三类，期望翻转率在 45%-55%
4. WHEN 评估 Tamper 抗性 THEN 系统 SHALL 验证修改 1 bit ciphertext/tag/AAD 必须导致解密失败，报告 tamper_fail_rate（期望≈100%）并记录 tamper 类型分项
5. WHEN 执行 replay 测试 THEN 系统 SHALL 明确一种策略：(a) 系统检测 nonce 重用并拒绝；或(b) 协议声明 nonce 重用的影响范围与上层防重放机制，并写入 security_metrics_cview.csv 的 replay_behavior 字段
6. WHEN 声称 C-view 安全 THEN 系统 SHALL 明确声明：安全主张继承自 AEAD（IND-CPA/CCA），混沌层不单独宣称语义安全（写进 EXPERIMENT_PROTOCOL.md）
7. WHEN 输出安全指标 THEN 系统 SHALL 写入 security_metrics_cview.csv 包含所有测试结果、nist_bits、replay_behavior 字段
8. WHEN NIST 子测试 p-value < 0.01 THEN 系统 SHALL 标记该测试为 fail 并在报告中说明

#### Artifacts
- `tables/security_metrics_cview.csv`
- `figures/fig_cview_security_summary.png`
- `reports/security_report.md`（解释每项测试、阈值、样本长度、安全边界声明）
- `meta/security_hparams.yaml`（alpha=0.01、NIST 子集列表、拼接策略、最小 bitstream 长度）

#### Coverage Gate
- NIST 子测试必须 ≥ 7 项（Frequency、BlockFrequency、Runs、LongestRun、FFT、Serial、ApproxEntropy）
- avalanche 必须 ≥ 3 类（key/nonce/plaintext）
- tamper 必须 ≥ 3 类（ciphertext/tag/aad）
- privacy_levels 全覆盖（至少在一个主数据集上）

#### Failure Handling
- 任一 NIST 子测试无法运行（输入不足）→ 必须自动拼接；仍不足则 hard fail 并在报告写明原因（数据量不足）
- tamper_fail_rate < 99% → hard fail
- avalanche flip_rate 不在 [0.45, 0.55] → 警告但不 hard fail（需在报告中解释）

#### Notes
- Security boundary: C-view 安全性继承自标准 AEAD（IND-CPA/IND-CCA），混沌层仅作为混淆/扩散，不单独宣称语义安全
- NIST alpha=0.01（每项 p≥0.01 视为 pass）
- 拼接策略（避免序列相关性）：按 SHA256(image_id) 哈希排序拼接 ciphertext bytes，固定 seed=42
- tamper 测试次数：每个 privacy_level 至少 N=1000 次（或抽样 200 条 × 5 tamper 类型）
- replay_behavior 必填：选定策略后必须写死（reject 或 warn-only），不允许运行时切换

### Requirement 4: Z-view 训练策略升级

**User Story:** As a 研究者, I want to 实现 Train-on-Z 和 Mixed Training 策略, so that Z-view 的任务效用能达到顶刊门槛。

#### Acceptance Criteria

1. WHEN 执行 Z2Z 训练 THEN 系统 SHALL 在 Z-view 上训练模型并在 Z-view 上测试
2. WHEN 执行 Mixed Training (Mix2Z) THEN 系统 SHALL 混合使用明文和 Z-view 数据进行训练，在 Z-view 上测试
3. WHEN 评估效用 THEN 系统 SHALL 在 privacy_level=0.3 时达到 ≥0.75×plaintext 性能
4. WHEN 评估效用 THEN 系统 SHALL 在 privacy_level=0.5 时达到 ≥0.65×plaintext 性能
5. WHEN 输出效用指标 THEN 系统 SHALL 包含 training_mode 字段（P2P/P2Z/Z2Z/Mix2Z）
6. WHEN 评估效用门槛 THEN 系统 SHALL 以 P2P 的 mean（across seeds）为基准计算比例，并在 csv 记录 relative_to=P2P_mean 与 relative_performance 字段
7. WHEN 评估效用 THEN 系统 SHALL 输出 group-wise 指标（至少 FairFace：race/gender 分组）写入 utility_group_metrics.csv
8. WHEN 未达到门槛 THEN 系统 SHALL 自动生成 failure analysis 写入 reports/failure_modes.md，包含：
   - 训练曲线（loss/metric vs epoch）
   - 样例可视化（Z-view vs P 对比）
   - 原因分类（domain shift / mask过强 / 预算过度 / 其他）

#### Artifacts
- `tables/utility_metrics.csv`
- `tables/utility_group_metrics.csv`（group-wise 指标：race/gender）
- `figures/fig_utility_curve.png`
- `reports/utility_report.md`（训练模式对比、门槛达成情况）
- `meta/train_hparams.yaml`（lr/batch_size/epochs/optimizer/scheduler 等）

#### Coverage Gate
- training_mode 必须覆盖：P2P/P2Z/Z2Z/Mix2Z（至少 4 个）
- privacy_levels 全覆盖（5 档）
- seeds ≥ 3
- 每个 csv 覆盖度 ≥ 98%

#### Failure Handling
- 未达 0.75×/0.65× 门槛时必须生成 failure 分析，不允许沉默
- 若所有 training_mode 都未达标 → 在 reports/failure_modes.md 详细说明原因与改进方向
- 训练崩溃 → 写入 failure_modes.md + status=failed

#### Notes
- Stat family: dataset × task × training_mode × metric_name
- 门槛计算口径：relative_performance = metric_value / P2P_mean
- group-wise 指标用于检测公平性问题

### Requirement 5: 消融实验矩阵

**User Story:** As a 研究者, I want to 执行 ≥12 项消融实验, so that 能解释"每个组件贡献了什么"。

#### Acceptance Criteria

1. WHEN 消融加密层 THEN 系统 SHALL 分别测试去 Layer1（混沌层）、去 Layer2（频域层）、去 crypto wrap（AEAD封装）的效果
2. WHEN 消融预算策略 THEN 系统 SHALL 对比 causal vs uniform vs sensitive-only vs task-only
3. WHEN 消融 mask 来源 THEN 系统 SHALL 对比强监督 vs 弱监督 mask
4. WHEN 消融频域变换 THEN 系统 SHALL 对比 FFT vs DWT
5. WHEN 输出消融结果 THEN 系统 SHALL 写入 ablation.csv 并生成对比图表
6. WHEN 执行消融实验 THEN 系统 SHALL 按 EXPERIMENT_PROTOCOL.md 中的 Ablation Catalog 固定 ≥12 项，禁止临时更改（更改需 bump protocol_version）
7. WHEN 消融结果与预期相反 THEN 系统 SHALL 在 ablation_report.md 中解释原因

#### Ablation Catalog（冻结清单，≥12 项）
| ID | 消融项 | 描述 |
|----|--------|------|
| A1 | remove_layer1 | 去混沌层 |
| A2 | remove_layer2 | 去频域层 |
| A3 | remove_crypto_wrap | 去 AEAD 封装 |
| A4 | causal_to_uniform | 因果预算→均匀预算 |
| A5 | causal_to_sensitive_only | 因果预算→仅敏感区域 |
| A6 | causal_to_task_only | 因果预算→仅任务区域 |
| A7 | mask_strong_to_weak | 强监督 mask→弱监督 mask |
| A8 | fft_to_dwt | FFT→DWT |
| A9 | semantic_preserving_off | 语义保留关闭 |
| A10 | deterministic_nonce_to_random | 确定性 nonce→随机 nonce |
| A11 | budget_normalization_variantA | 预算归一策略 A |
| A12 | budget_normalization_variantB | 预算归一策略 B |

#### Artifacts
- `tables/ablation.csv`（包含 ablation_id、utility_metric、attack_metric、delta_vs_full）
- `figures/fig_ablation_summary.png`
- `reports/ablation_report.md`（每个消融项的解释与结论）

#### Coverage Gate
- 每个消融至少在 1 dataset × 1 task × 5 levels × 3 seeds 跑全
- 输出覆盖度不足 → hard fail
- 必须同时输出 utility + attack 两类指标

#### Failure Handling
- 某消融项无法运行（如 DWT 未实现）→ 在 ablation_report.md 说明原因，标注 status=skipped
- 消融结果与预期相反 → 必须在报告中解释
- 消融清单变更 → 必须 bump protocol_version

#### Notes
- 消融清单变更需要 bump protocol_version
- 每个消融项输出 utility + attack 两类指标
- delta_vs_full = ablation_metric - full_method_metric

### Requirement 6: 论文级输出工程化

**User Story:** As a 研究者, I want to 一键生成所有论文图表, so that 减少手工整理的工作量和错误。

#### Acceptance Criteria

1. WHEN 执行 make_figures 脚本 THEN 系统 SHALL 从 tables/*.csv 生成所有主图和补充图
2. WHEN 执行 validate_run 脚本 THEN 系统 SHALL 检查结果目录的完整性（必须文件、字段覆盖度）
3. WHEN 生成 Pareto 前沿图 THEN 系统 SHALL 包含你的方法 vs InstaHide vs P3 的对比
4. WHEN 生成 ATE/CATE 图 THEN 系统 SHALL 包含置信区间和 group-wise 分析
5. WHEN 提供复现脚本 THEN 系统 SHALL 创建 reproduce.sh 或 Makefile 实现一键复现
6. WHEN 生成图表 THEN 系统 SHALL 使用 figure_specs 冻结规格：
   - 尺寸：单栏 3.5in，双栏 7in
   - dpi=300
   - 字体：Arial/Helvetica，字号 ≥8pt
   - 图例规范统一
7. WHEN 生成图表 THEN 系统 SHALL 记录规格到 reports/figure_manifest.json
8. WHEN 生成图表 THEN 系统 SHALL 禁止手工后处理；所有图必须可由脚本从 tables/ 重建（validate_run 检查）

#### Artifacts
- `scripts/run/make_figures.py`
- `src/plotting/figure_specs.py`（冻结图表规格）
- `figures/*.png`（主图+补充图）
- `reports/figure_manifest.json`（记录每张图的来源 csv、生成时间、规格）
- `reproduce.sh` 或 `Makefile`

#### Figure List（必须生成，共 8 张主图）
| 图名 | 描述 | 来源 csv |
|------|------|----------|
| fig_utility_curve.png | 效用随 privacy_level 变化 | utility_metrics.csv |
| fig_attack_curves.png | 五类攻击曲线+CI | attack_metrics.csv |
| fig_pareto_frontier.png | 隐私-效用 Pareto 前沿 | utility_metrics.csv + attack_metrics.csv |
| fig_causal_ate_cate.png | ATE/CATE + CI | causal_effects.csv |
| fig_cview_security_summary.png | C-view 安全指标汇总 | security_metrics_cview.csv |
| fig_ablation_summary.png | 消融实验对比 | ablation.csv |
| fig_efficiency.png | 效率对比 | efficiency.csv |
| fig_robustness.png | 鲁棒性测试结果 | robustness_metrics.csv |

#### Coverage Gate
- 所有 8 张主图必须存在
- 每张图必须可从 tables/*.csv 重建
- figure_manifest.json 必须记录所有图

#### Failure Handling
- 缺任何主图 → hard fail
- 图与 csv 数据不一致 → hard fail
- figure_manifest.json 缺失 → hard fail

#### Notes
- dpi=300，尺寸按期刊要求
- 字体优先级：Arial > Helvetica > DejaVu Sans（Linux fallback）
- 字号 ≥8pt
- 字体 fallback 规则：若首选字体不存在，自动 fallback 并在 figure_manifest.json 记录 actual_font
- 禁止手工后处理，确保可复现

### Requirement 7: 结果目录规范

**User Story:** As a 研究者, I want to 所有实验结果按统一规范存储, so that 结果可审计、可复现。

#### Acceptance Criteria

1. WHEN 创建 run 目录 THEN 系统 SHALL 包含 meta/ 子目录，必须包含：
   - config.yaml（实验配置）
   - git_commit.txt（代码版本）
   - seed.txt（随机种子）
   - env.txt（环境信息）
   - dataset_manifest_hash.txt（数据集 manifest 的 SHA256）
   - protocol_version.txt（协议版本）
2. WHEN 输出表格 THEN 系统 SHALL 写入 tables/ 子目录，必须包含：
   - utility_metrics.csv
   - attack_metrics.csv
   - causal_effects.csv
   - security_metrics_cview.csv
   - ablation.csv
   - efficiency.csv
   - robustness_metrics.csv
   - baseline_comparison.csv
3. WHEN 输出图表 THEN 系统 SHALL 写入 figures/（所有 R6 定义的 8 张主图）
4. WHEN 输出日志 THEN 系统 SHALL 写入 logs/（stdout.log、errors.log）
5. WHEN 输出报告 THEN 系统 SHALL 写入 reports/（failure_modes.md、missing_matrix.csv、protocol_snapshot.md）
6. WHEN 校验覆盖度 THEN 系统 SHALL 确保每个 csv 覆盖 ≥ 98% 期望组合
7. WHEN 写入 tables THEN 系统 SHALL 使用统一 schema（RESULTS_SCHEMA.md）并记录 schema_version
8. WHEN validate_run THEN 系统 SHALL 检查覆盖度并输出 missing_matrix.csv

#### Artifacts
- `meta/config.yaml`
- `meta/git_commit.txt`
- `meta/seed.txt`
- `meta/env.txt`
- `meta/dataset_manifest_hash.txt`
- `meta/protocol_version.txt`
- `meta/hardware.json`
- `reports/missing_matrix.csv`
- `reports/protocol_snapshot.md`

#### Coverage Gate
- meta/ 必须包含 7 个文件（含 hardware.json）
- tables/ 必须包含 8 个 csv
- figures/ 必须包含 8 张主图
- 每个 csv 覆盖度 ≥ 98%

#### Failure Handling
- 缺任何必须文件 → hard fail
- 覆盖度不足 → hard fail 并输出 missing_matrix.csv
- schema_version 与 protocol_version 不一致 → hard fail

#### Notes
- schema_version 与 protocol_version 必须一致
- 所有路径使用相对路径
- dataset_manifest_hash 用于验证数据一致性

### Requirement 8: 近 5 年代表作 + 经典系统基线对标

**User Story:** As a 研究者, I want to 明确与近 5 年顶刊工作及经典基线的对标关系, so that 论文定位清晰、审稿人认可。

#### Acceptance Criteria

1. WHEN 对标 InstaHide THEN 系统 SHALL 在相同协议下比较隐私保护强度和任务效用（近5年代表作）
2. WHEN 对标 P3 THEN 系统 SHALL 比较公有/私有分离策略与语义差异化加密（经典系统基线，非近5年）
3. WHEN 撰写 Related Work THEN 系统 SHALL 覆盖 InstaHide/InfoScrub/Fawkes/OPOM/PSIC 等代表作
4. WHEN 定位方法 THEN 系统 SHALL 强调 Dual-View + 语义区域预算 + 因果可解释 + 可验证性的组合创新
5. WHEN 输出对标表格 THEN 系统 SHALL 包含 Utility/Privacy/Practical/Explainability/Strong Security 五维度
6. WHEN 对标基线 THEN 系统 SHALL 在 baseline_comparison.csv 输出五维度汇总表，并标注 implementation_type（reproduce/reuse/official-tool）
7. WHEN 定义等效强度 THEN 系统 SHALL 在 docs/baselines/{method}.md 写清 strength_param -> privacy_level 映射规则，并在结果里输出 mapped_privacy_level

#### Artifacts
- `tables/baseline_comparison.csv`（五维度 + implementation_type + mapped_privacy_level）
- `docs/baselines/instahide.md`（含 strength_param 映射规则）
- `docs/baselines/p3.md`（含 strength_param 映射规则）
- `docs/baselines/infoscrub.md`（可选）
- `figures/fig_pareto_frontier.png`

#### Coverage Gate
- 至少对标 2 个基线（InstaHide + P3）
- 每个基线在同一评测协议下运行
- 每个基线必须有 strength_param -> privacy_level 映射文档

#### Failure Handling
- 基线无法运行 → 在 baseline_comparison.csv 标注 status=failed，并在报告说明原因
- 映射规则缺失 → hard fail

#### Notes
- P3 为经典基线（非近5年），需在文档中明确声明
- 等效强度映射原则：按 attack_success 风险对齐优先
- implementation_type 必须在 {reproduce, reuse, official-tool} 中选择

### Requirement 9: 近 5 年高质量论文对标落地到可执行对比

**User Story:** As a 研究者, I want to 按研究链条分类对标近5年工作, so that 每条线都能对齐到实验协议里。

#### Acceptance Criteria

1. WHEN 撰写 Related Work THEN 系统 SHALL 在 RELATED_WORK_2020_2025.md 中按 5 条研究线分类：
   - 训练阶段保护 (InstaHide)
   - 属性隐私混淆 (InfoScrub)
   - 人脸披风 (Fawkes/OPOM)
   - 生成式匿名化 (DeepPrivacy2)
   - 压缩链路防VLP (PSIC)
2. WHEN 输出对标表格 THEN 系统 SHALL 用同一套 Utility/Privacy/Practical/Explainability/Strong Security 五维度字段
3. WHEN 标注基线实现方式 THEN 系统 SHALL 明确哪些是"复现实现"、哪些是"复用公开实现/权重/官方工具"
4. WHEN 对比实验 THEN 系统 SHALL 确保所有基线在同一评测协议下运行
5. WHEN 对标近 5 年工作 THEN 系统 SHALL 至少对齐 2 条研究线的"可执行对比"（例如：InstaHide + InfoScrub 或 Fawkes/OPOM + DeepPrivacy2），并在同一攻击协议下产出 attack_metrics.csv

#### Artifacts
- `docs/RELATED_WORK_2020_2025.md`（按 5 条研究线分类）
- 对标表格（在文档中，五维度）

#### Coverage Gate
- 至少覆盖 2 条研究线的可执行对比
- 避免 Related Work 写得很豪华但实验只跑了 1 个基线
- 每条可执行对比的研究线必须在 attack_metrics.csv 中有记录

#### Failure Handling
- 某研究线无法对标 → 在文档中说明原因（如代码未公开、数据不兼容）
- 只有 1 条研究线可执行对比 → 警告但不 hard fail

#### Notes
- 5 条研究线：训练阶段保护、属性隐私混淆、人脸披风、生成式匿名化、压缩链路防VLP
- 可执行对比 = 在同一评测协议下运行并产出 attack_metrics.csv

### Requirement 10: 统计严谨性与显著性规范

**User Story:** As a 研究者, I want to 统一统计检验与显著性报告, so that 结果可信且不被质疑为 cherry-picking。

#### Acceptance Criteria

1. WHEN 计算置信区间 THEN 系统 SHALL 对所有核心曲线点输出 95% CI（bootstrap≥500，推荐 1000）
2. WHEN 比较多个方法/多个 privacy_level THEN 系统 SHALL 使用 Holm-Bonferroni 或 BH-FDR 做多重比较校正，并在表格里记录 p_adj
3. WHEN 声称改进 THEN 系统 SHALL 同时报告 effect size（例如 ΔAcc、ΔAUC）与 CI/显著性（CI 不跨 0 或 p_adj < 0.05）
4. WHEN 输出 csv THEN 系统 SHALL 写入 stat_method 字段（bootstrap/t-interval 等）与 n_boot
5. WHEN 执行多重比较校正 THEN 系统 SHALL 明确 family 定义并将 family_id 写入 csv：
   - family = dataset × task × metric_name × privacy_level（或 × method，二选一但要写死）
6. WHEN 输出显著性 THEN 系统 SHALL 固定 alpha=0.05，并输出 alpha 字段（便于审计）

#### Artifacts
- 所有 csv 中必须包含以下统计字段：
  - stat_method（统计方法）
  - n_boot（bootstrap 次数）
  - ci_low、ci_high（95% CI）
  - p_value（原始 p 值）
  - p_adj（校正后 p 值）
  - effect_size（效应量）
  - alpha（显著性水平）
  - family_id（family 标识）
- `tables/stats_summary.csv`（汇总每个 family 的比较与 p_adj，强烈推荐）

#### Coverage Gate
- 所有核心指标必须有 CI
- 所有方法对比必须有 p_adj
- family_id 必须在所有 csv 中一致

#### Failure Handling
- 缺 stat_method/n_boot/p_adj → hard fail
- CI 计算失败 → 在 failure_modes.md 记录原因
- family 定义不一致 → hard fail

#### Notes
- Stat family 定义：dataset × task × metric_name × privacy_level（默认）
- 默认多重比较校正：BH-FDR
- alpha=0.05（NIST 测试除外，NIST alpha=0.01）
- effect_size 计算：ΔMetric = method_metric - baseline_metric
- baseline 定义（见 GC8）：Utility 用 P2P，Privacy 用 plaintext

### Requirement 11: 数据泄漏与协议一致性防护

**User Story:** As a 研究者, I want to 自动检查数据泄漏与协议一致性, so that 任何人复现实验不会无意中泄漏 test 信息或混用视图。

#### Acceptance Criteria

1. WHEN 构建 manifest THEN 系统 SHALL 检查 train/val/test 之间无 ID 重叠（hash/filename）
2. WHEN 运行攻击 THEN 系统 SHALL 检查攻击训练集与被评测样本严格分离（尤其 membership inference 的 shadow split）
3. WHEN 生成 Z/C 视图 THEN 系统 SHALL 记录 view_version 与 enc_info，并确保 C-view 数据永不进入训练 DataLoader（仅进入 security evaluator）
4. WHEN validate_run THEN 系统 SHALL 对上述规则做硬失败（hard fail）
5. WHEN validate_splits 发现泄漏 THEN 系统 SHALL 输出 reports/split_leakage_report.md 并 hard fail
6. WHEN C-view 进入训练管线 THEN 系统 SHALL hard fail（通过 DataLoader guard 实现）

#### Artifacts
- `scripts/validate/validate_splits.py`
- `reports/split_leakage_report.md`（若发现泄漏，包含重叠 ID 列表）
- `src/data/manifest_dataset.py`（含 C-view guard）

#### Coverage Gate
- 所有 split 必须无重叠
- C-view 永不进入训练
- shadow split 与被评测样本严格分离

#### Failure Handling
- split 泄漏 → hard fail + 输出详细报告（重叠 ID 列表）
- C-view 进入训练 → hard fail + 错误信息清晰
- shadow split 泄漏 → hard fail

#### Notes
- DataLoader guard 实现：训练 loader 只能取 plaintext 或 z_view；若取 c_view → raise ViewAccessError
- validate_splits.py 应在 CI 中运行

### Requirement 12: Practicality/工程指标

**User Story:** As a 研究者, I want to 报告端到端成本与系统开销, so that 方法可落地且 trade-off 清晰。

#### Acceptance Criteria

1. WHEN 运行 benchmark THEN 系统 SHALL 输出加密/解密吞吐（images/s）、端到端延迟（P50/P95/P99）
2. WHEN 评估存储开销 THEN 系统 SHALL 报告 bytes/image（Z 与 C 分别）
3. WHEN 评估资源占用 THEN 系统 SHALL 报告显存/CPU 占用
4. WHEN 生成效率图 THEN 系统 SHALL 在 fig_efficiency.png 中对比你方法与 baselines 的成本
5. WHEN 输出结果 THEN 系统 SHALL 写入 runtime.json 与 efficiency.csv
6. WHEN 记录效率指标 THEN 系统 SHALL 记录硬件信息并输出 meta/hardware.json，包含：
   - GPU 型号/数量
   - CPU 型号/核数
   - 内存大小
   - batch_size
   - 重复测量次数 n_runs
7. WHEN 报告延迟 THEN 系统 SHALL 固定测量协议写入 EXPERIMENT_PROTOCOL.md：
   - warmup 次数（默认 10）
   - 统计窗口（默认 100 次）
   - 统计量（mean±std）

#### Artifacts
- `tables/efficiency.csv`（包含 stage/throughput/latency_p50/p95/p99/bytes_per_image）
- `meta/hardware.json`
- `meta/runtime.json`
- `figures/fig_efficiency.png`
- `reports/efficiency_report.md`（解释测量协议、硬件环境）

#### Coverage Gate
- 必须覆盖：encrypt/decrypt/train/infer 四个 stage
- 必须对比至少 2 个 baselines
- 必须记录硬件信息

#### Failure Handling
- 效率测量失败 → 在 efficiency_report.md 说明原因
- 硬件信息缺失 → hard fail

#### Notes
- 测量协议：warmup=10, n_runs=100, 统计窗口=mean±std
- 硬件信息必须记录以便复现
- bytes_per_image 分别记录 Z-view 和 C-view

### Requirement 13: 鲁棒性与非理想条件测试

**User Story:** As a 研究者, I want to 覆盖非理想条件, so that 防御不是只在理想输入上成立。

#### Acceptance Criteria

1. WHEN 评估 Z-view THEN 系统 SHALL 在至少 3 种扰动下复测，并报告 utility 与 attack 的变化
2. WHEN 评估 C-view THEN 系统 SHALL 对 tamper（1-bit flip）、截断、重排等异常输入验证认证失败率≈100%
3. WHEN 评估跨域泛化 THEN 系统 SHALL 至少在 1 个外部集（如 LFW/FFHQ）上进行 face verification 攻击评测
4. WHEN 运行鲁棒性测试 THEN 系统 SHALL 固定 perturbation 组并在 csv 写 perturbation_type/strength
5. WHEN 输出鲁棒性结果 THEN 系统 SHALL 生成 tables/robustness_metrics.csv 与 figures/fig_robustness.png

#### Perturbation Catalog（冻结清单）
| ID | 扰动类型 | 强度参数 |
|----|----------|----------|
| P1 | JPEG compression | quality ∈ {90, 70, 50} |
| P2 | resize+crop | ratio ∈ {0.9, 0.8} |
| P3 | Gaussian noise | σ ∈ {2, 5}（8-bit 标度） |

#### Artifacts
- `tables/robustness_metrics.csv`（包含 perturbation_type/strength/utility_delta/attack_delta）
- `figures/fig_robustness.png`
- `reports/robustness_report.md`（解释每种扰动的影响）

#### Coverage Gate
- 至少 3 种扰动类型（JPEG/resize+crop/noise）
- 至少 1 个外部验证集（LFW 或 FFHQ）
- 每种扰动至少 2 个强度级别

#### Failure Handling
- 某扰动无法运行 → 在 robustness_report.md 说明原因，标注 status=skipped
- 跨域泛化失败 → 必须分析原因并写入报告
- 扰动后 utility 下降 >30% → 警告并在报告中解释

#### Notes
- 扰动强度范围固定，不允许临时更改（更改需 bump protocol_version）
- 外部集建议：LFW（face verification）或 FFHQ（重建攻击）
- **外部集使用限制**：外部集只能用于 run_attacks 的评测阶段（A0/A1/A2），不得进入 utility 训练数据，避免被质疑"用外部集调参"
- utility_delta = perturbed_utility - clean_utility
- attack_delta = perturbed_attack_success - clean_attack_success

### Requirement 14: Artifact/CI/质量门槛

**User Story:** As a 研究者, I want to 实验产线可持续运行, so that 最后冲刺不会被工程问题拖垮。

#### Acceptance Criteria

1. WHEN 合并代码 THEN 系统 SHALL 在 CI 中运行：validate_manifest、validate_run、核心单测（≥12）
2. WHEN 发布 artifact THEN 系统 SHALL 提供 reproduce.sh/Makefile，并保证从 0 到生成主图主表可以一键完成（数据需用户自行下载）
3. WHEN 依赖更新 THEN 系统 SHALL 锁定关键版本（torch/cuda/arcface 等），并在 env.txt 记录
4. WHEN 在 CI 运行 THEN 系统 SHALL 提供 --smoke_test 模式：
   - 小样本（每数据集 32–128 张）
   - 跑 validate_manifest、validate_run
   - 最小 utility/attack/security 1–2 个 level
   - 时间预算 < 20 min
5. WHEN 发布 artifact THEN 系统 SHALL 提供 ARTIFACT_CHECKLIST.md（列出所有必须文件与复现步骤）

#### Artifacts
- `scripts/run/run_benchmark.py --smoke_test`
- `docs/ARTIFACT_CHECKLIST.md`（复现检查清单）
- `reproduce.sh` 或 `Makefile`
- `.github/workflows/ci.yml`（或等效 CI 配置）

#### CI Smoke Test 规格
| 检查项 | 时间预算 | 样本量 |
|--------|----------|--------|
| validate_manifest | < 1 min | 全量 manifest |
| validate_run | < 1 min | 检查目录结构 |
| 核心单测 | < 5 min | ≥12 个测试 |
| smoke_test utility | < 5 min | 32-128 张/数据集 |
| smoke_test attack | < 5 min | 32-128 张/数据集 |
| smoke_test security | < 3 min | 1-2 个 level |
| **总计** | **< 20 min** | - |

#### Coverage Gate
- CI 必须通过才能 merge
- smoke_test 必须覆盖 utility/attack/security 三类
- 核心单测 ≥ 12 个

#### Failure Handling
- CI 失败 → PR 不允许合并
- smoke_test 超时 → 警告但不 hard fail（需优化）
- 依赖版本漂移 → 警告

#### Notes
- smoke_test 用于快速验证，不替代完整实验
- ARTIFACT_CHECKLIST.md 应包含：必须文件列表、复现命令、预期输出
- 关键依赖版本锁定：torch、cuda、arcface、numpy、scipy

### Requirement 15: 安全主张边界与声明（新增必需）

**User Story:** As a 研究者, I want to 明确安全主张边界, so that 自定义混沌层不会被误解为语义安全声明而被审稿人击穿。

#### Acceptance Criteria

1. WHEN 描述 C-view 安全 THEN 系统 SHALL 声明安全性继承自标准 AEAD（IND-CPA/IND-CCA 由标准算法保证）
2. WHEN 描述混沌/频域扰动 THEN 系统 SHALL 明确其角色为"混淆/扩散/分布扰动"，不单独宣称语义安全
3. WHEN 输出文档 THEN 系统 SHALL 在 EXPERIMENT_PROTOCOL.md 与 README 中各写一次该边界声明
4. WHEN 声称隐私保护 THEN 系统 SHALL 区分：
   - Z-view 隐私：通过攻击成功率实证证明
   - C-view 安全：通过 AEAD 标准保证 + 传统密码学测试验证
5. WHEN 描述威胁模型 THEN 系统 SHALL 明确不防御的攻击类型（如侧信道、物理攻击）

#### Security Boundary Statement（标准声明模板）
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

#### Artifacts
- `docs/EXPERIMENT_PROTOCOL.md` 更新段落（Security Boundary 小节）
- `docs/security_claims.md`（独立安全声明文档）
- `README.md` 更新段落（Security Notes 小节）

#### Coverage Gate
- EXPERIMENT_PROTOCOL.md 必须包含 Security Boundary 小节
- README.md 必须包含 Security Notes 小节
- security_claims.md 必须存在

#### Failure Handling
- 缺任何安全边界声明 → hard fail
- 声明与实际实现不一致 → hard fail

#### Notes
- 此声明用于防止审稿人从"混沌层不是语义安全"角度攻击
- 明确区分 Z-view（实证隐私）和 C-view（密码学安全）
- 不防御的攻击类型必须明确列出

### Requirement 16: 自适应攻击者与最强威胁等级样例（A2）（新增必需）

**User Story:** As a 研究者, I want to 覆盖至少一组 A2 白盒自适应攻击, so that 审稿人无法质疑"只防弱攻击"。

#### Acceptance Criteria

1. WHEN 评估隐私 THEN 系统 SHALL 至少对 1 个数据集、1 个任务、2 个 privacy_level（0.5/0.7）运行 A2 自适应攻击
2. WHEN 定义 A2 攻击者 THEN 系统 SHALL 明确攻击者能力：
   - 知道完整算法（加密流程、mask 生成、预算分配）
   - 知道模型结构（但无训练权重）
   - 无法访问加密密钥
   - 可自适应设计攻击策略
3. WHEN 输出结果 THEN 系统 SHALL 在 attack_metrics.csv 标注 threat_level=A2
4. WHEN 输出报告 THEN 系统 SHALL 在 attack_report.md 描述自适应策略，包括：
   - 攻击者利用的算法知识
   - 自适应策略设计
   - 与 A0/A1 攻击的对比
5. WHEN A2 运行失败 THEN 系统 SHALL hard fail（因为这是顶刊防线）

#### A2 Adaptive Attack Specification
| 攻击类型 | 自适应策略 | 攻击者可见信息 |
|----------|------------|----------------|
| Reconstruction | 利用 mask 结构设计 loss | 算法、mask 生成逻辑 |
| Attribute Inference | 利用预算分配规则 | 算法、预算分配逻辑 |
| Face Verification | 利用语义保留区域 | 算法、语义保留策略 |

#### Artifacts
- `attack_metrics.csv` 中 threat_level=A2 的记录
- `reports/attack_report.md` 增补 A2 小节（自适应策略描述）
- `figures/fig_attack_curves.png` 中标注 A2 曲线或点
- `src/evaluation/attacks/adaptive_attacker.py`（A2 攻击实现）

#### Coverage Gate
- 至少 1 个数据集（建议 CelebA-HQ）
- 至少 1 个任务（建议 attribute_inference 或 reconstruction）
- 至少 2 个 privacy_level（0.5 和 0.7）
- attack_metrics.csv 中必须有 threat_level=A2 的记录

#### Failure Handling
- A2 攻击未运行 → hard fail
- A2 攻击运行失败 → hard fail（必须修复）
- A2 攻击成功率过高（>0.8）→ 警告并在报告中分析原因

#### Notes
- A2 是顶刊安全论文的必要防线
- 自适应攻击者可以利用算法知识设计更强的攻击
- 即使 A2 攻击成功率较高，也必须报告（诚实报告比隐藏更好）
- A2 攻击结果应与 A0/A1 对比，展示防御在不同威胁等级下的表现
