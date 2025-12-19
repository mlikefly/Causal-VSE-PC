# Causal‑VSE‑PC 顶刊冲刺 V2：最“吹毛求疵”的全实验闭环计划（传统 + 非传统，全覆盖）

> 角色设定：把我当成一个延毕博士生——这是最后一次冲顶刊的机会。  
> 目标：不仅要“能跑、指标好看”，更要让审稿人**找不到明显缺口**：威胁模型闭合、协议标准、传统密码学测试齐全、对标充分、统计严谨、结果可复现、结论可解释。

---

## A. 最终交付（Final Deliverables）与“实验完全完成”的硬性验收

### A1. 论文级产物（必须全部可一键重生成）
**主文 6 图 + 4 表（建议）**
- Fig1：系统框架（Dual‑View + 语义区域 + 因果预算 + 三层）
- Fig2：Utility–Privacy 曲线（至少 2 任务 × ≥3 seeds mean±CI）
- Fig3：Attack–Privacy 曲线（三类核心攻击 + 2 类补充攻击）
- Fig4：Pareto frontier（你 vs baselines）
- Fig5：Causal ATE/CATE（含 CI，含 group‑wise）
- Fig6：效率/开销（吞吐、延迟、存储、显存）  
- Table1：Datasets/Tasks/Splits/License/Preprocess 规格
- Table2：Baselines 对标总表（同协议）
- Table3：Ablation 矩阵（≥12 项）
- Table4：Fairness/Group‑wise（FairFace）与稳定性（std、CI）

**补充材料（Supp）**
- S‑Fig：C‑view 密码学/随机性传统测试汇总（NIST/χ²/Dieharder 选做）
- S‑Fig：重建攻击可视化（mask 内 vs 全图）
- S‑Table：攻击训练成本、攻击者假设、参数表、超参表
- Artifact：`reproduce.sh`/`Makefile` + `RESULTS_SCHEMA.md` + `EXPERIMENT_PROTOCOL.md` + `DATASETS.md`

### A2. 强制复现性（Reproducibility Gate）
每个 run 必须自动落盘：
- `meta/config.yaml`（最终解析后配置）
- `meta/git_commit.txt`
- `meta/seed.txt`
- `meta/env.txt`（pip freeze + cuda + torch 版本）
- `meta/dataset_manifest_hash.txt`（hash(all.jsonl)）
- `meta/protocol_version.txt`（协议版本号）
- `tables/*.csv` + `figures/*.png` + `logs/*.log`

**验收**：在干净环境（新机器/新 conda）执行 `reproduce.sh`，可生成结构一致的结果目录；核心曲线趋势一致（数值允许轻微漂移）。

---

## B. 威胁模型（Threat Model）——审稿人最爱抓的“矛盾点”先封死

### B1. 两视图的能力边界（必须写进协议，避免混用）
- **C‑view（强密文）**：用于传输/存储；攻击者可获得 ciphertext、可篡改、可重放；安全目标是**机密性 + 完整性**；不要求可推理。
- **Z‑view（可用密文表示）**：用于云端/第三方推理；攻击者可获得 Z 及模型输出，甚至可能知道算法（Kerckhoffs）；安全目标是**在可用性约束下抑制隐私泄漏**（通过攻击成功率衡量，不靠 NIST）。

> 关键口径：**传统密码学测试只对 C‑view 做主张**；Z‑view 的隐私靠攻击族证明。

### B2. 攻击者知识与能力分级（实验必须覆盖至少 3 档）
- **A0：黑盒**：只见 Z、见输出，不见模型参数；可 query 多次。
- **A1：灰盒**：知道防御算法与超参范围，但不持有密钥；可训练替代模型。
- **A2：白盒（最严）**：知道防御算法、已知模型结构/参数；仍不持有密钥。  
（对 C‑view：允许 chosen‑ciphertext / tamper；对 Z‑view：允许 adaptive attacker。）

**验收**：每项攻击在报告中标注威胁等级（A0/A1/A2）。

---

## C. 数据与任务：选择“顶刊最稳”的组合（至少 3 域）

### C1. 数据集（建议最小组合）
1) **CelebA（属性分类）**：任务稳定、攻击丰富、文献对标多  
2) **FairFace（公平性）**：提供 group‑wise 证据（race/gender/age）  
3) **OpenImages PII 子集**：开放域泛化（person/face/plate 至少覆盖前两类）  
**外部验证加分项**：LFW（只评测）、FFHQ（只评测或轻量评测）

### C2. 任务（至少 2 类，建议 3 类）
- **T1 属性分类（必做）**：Acc、Macro‑F1、AUC（敏感属性/任务属性分开）
- **T2 分割（强烈建议）**：mIoU、Dice（mask 内/全图分别报告）
- **T3 检测（可选）**：mAP@0.5、mAP@[.5:.95]

### C3. 数据规模门槛（避免小样本“虚高”）
- 每个主数据集 **test ≥ 1,000**（强烈建议 ≥ 5,000）
- 每个 privacy_level 点评测样本数 **≥ 1,000**
- seeds：**≥ 3**（顶刊更稳：5）
- bootstrap：**≥ 1,000 resamples**（算力不够可 200–500，但必须说明）

---

## D. 指标体系（Metrics）：统一口径 + 公式级标准 + 门槛

> 注意：门槛是“目标线/验收线”，不是保证一定达到；达不到就要调整方法或在论文中诚实说明 trade‑off。

### D1. Utility 指标（Z‑view）
**分类（属性/多分类）**
- Acc、Macro‑F1（主）
- AUC（敏感属性推断/二分类）
- ECE（校准，可选加分）
- 统计：mean±95% CI（bootstrap 或 t‑interval），并报告 std

**分割**
- mIoU、Dice（主）
- 辅助：Boundary F1（可选）

**门槛建议（相对明文）**
- privacy_level=0.3：≥ 0.75×plaintext（或绝对指标：Acc≥75% / mIoU≥0.55，按任务调）
- privacy_level=0.5：≥ 0.65×plaintext
- privacy_level=0.7：趋势可用（≥0.5×plaintext），并把 Pareto 曲线讲清楚

### D2. Privacy 指标（Z‑view）：攻击成功率为核心
你至少要做 **3 主攻 + 2 补攻**，并统一输出为 `attack_success`（越低越好）。

**主攻 1：身份验证/识别（Face Verification/Re‑ID）**
- 指标：Verification Acc、TAR@FAR=1e‑3（可选）、ROC‑AUC
- 门槛：privacy_level 上升应显著下降；在中等隐私强度下应接近随机/显著低于明文

**主攻 2：属性推断（Sensitive Attribute Inference）**
- 指标：Acc/AUC（敏感属性）
- 门槛：应逼近随机（多分类≈1/K；二分类≈50%）或显著低于明文

**主攻 3：重建攻击（Reconstruction / Model Inversion）**
- 指标（敏感区域 mask 内必须报告）：PSNR、SSIM、LPIPS（可选）、ArcFace/FaceNet 相似度（身份泄漏）
- 门槛：mask 内 PSNR/SSIM 显著下降；身份 embedding 相似度显著下降

**补攻 A：Membership Inference（成员推断）**
- 协议：Shadow models 或 LiRA 风格（轻量版也可）
- 指标：Attack AUC、Advantage
- 门槛：接近 0.5 / advantage 低

**补攻 B：Property Inference / Group Leakage（群体属性/数据集属性泄漏）**
- 指标：攻击准确率
- 门槛：显著低于明文

### D3. Security 指标（C‑view）：传统密码学/随机性必须“全套做齐”
> 这些是“传统测试”，属于顶刊（尤其安全/密码学审稿）会要求的项目。

#### D3.1 统计/图像加密传统指标（可快速跑，必须）
- 信息熵（Entropy）：目标 ≥ 7.95（8bit 图像）
- 直方图均匀性：χ² 检验（p≥0.01 视作不过拟合拒绝均匀）
- 像素相关性（H/V/D）：|corr| ≤ 0.01（目标）或 ≤0.02（最低线）
- NPCR：≥ 99.5%（目标 ≥ 99.6%）
- UACI：≈ 33%（允许 31%–35%）
- Differential key sensitivity：微小 key/nonce 变化导致输出大幅变化（见 D3.3）
- Robustness：裁剪/噪声/压缩后解密失败（AEAD）或认证失败率≈100%

#### D3.2 NIST SP 800‑22（必须，至少 7–15 项）
对密钥流/密文比特流做 NIST 统计测试（最少集：Frequency、BlockFrequency、Runs、LongestRun、FFT、Serial、ApproxEntropy）。  
- 门槛：每项 p‑value ≥ 0.01；多项测试通过率达到预设阈值（例如 ≥ 6/7 或 ≥ 12/15）
- 注意：样本长度要足够（报告每项输入 bits 长度）

#### D3.3 Avalanche / Key Sensitivity / Plaintext Sensitivity（必须写清协议）
- 固定 plaintext，翻转 1 bit 的 key 或 nonce：
  - 期望密文比特翻转率 ≈ 50%（理想区间 45%–55%）
- 固定 key，翻转 plaintext 1 bit：
  - 密文比特翻转率同上  
- 产出：`avalanche.csv`（flip_rate、std）

#### D3.4 形式化/密码学语义安全（尽你所能做到“可审查”）
- 如果 C‑view 使用 AEAD（如 ChaCha20‑Poly1305 / AES‑GCM）：
  - 明确安全声明：IND‑CPA/IND‑CCA（基于标准 AEAD）
  - 明确 nonce 规则：唯一性/确定性派生策略，及冲突概率估算
- 如果有自定义混沌层：只作为**前置混淆**，最终安全依赖 AEAD；不要对混沌层单独宣称语义安全。

#### D3.5 额外传统测试（选做但“顶刊加分”）
- Dieharder / TestU01（SmallCrush）对密钥流/密文流测试
- 重放攻击/篡改攻击：认证必须失败
- 密钥空间估算：≥2^128 等级，并说明密钥派生结构

---

## E. 实验矩阵（Experiment Matrix）：把所有跑的东西“制度化”

### E1. privacy_level 网格（统一）
- L = {0.0, 0.3, 0.5, 0.7, 1.0}
- 对比方法必须在同一网格或提供等效映射（写清）

### E2. 方法组（至少 6 组）
1) Plaintext（上限）
2) Z‑view（你的方法：causal budget）
3) Z‑view（uniform budget）
4) Z‑view（sensitive‑only）
5) Baseline‑InstaHide
6) Baseline‑P3
（可选：Blur/Pixelate/JPEG/DP noise 作为弱基线）

### E3. 消融组（至少 12 项，形成表格）
- 去 Layer1 / 去 Layer2 / 去 crypto‑wrap / 仅 crypto‑wrap
- causal → uniform / heuristic / sensitive‑only / task‑only
- mask：强监督 → 弱监督
- FFT→DWT
- semantic_preserving：on/off
- key/nonce：deterministic vs randomized（如果你支持）
- Z 表示：不同频域扰动强度、不同预算归一策略

**验收**：每个组必须产出同字段 csv 与同规格图。

---

## F. 训练与评测协议（严谨到审稿人挑不出）

### F1. 训练策略（Z‑view 必须做 Domain‑aware）
- P→Z（对照）：train plaintext, test Z
- Z→Z（必须）：train Z, test Z
- Mixed（推荐）：train {P,Z}, test Z
- 可选：domain adaptation（CORAL/MMD/BN‑adapt）

**统计要求**
- seeds ≥ 3（建议 5）
- 每个 seed 记录：最终 best epoch、val 指标、早停条件
- 报告：mean±95% CI（bootstrap over samples 或 across seeds）

### F2. 多重比较校正（顶刊常见）
当你对很多方法/很多 level 做显著性检验时：
- Holm‑Bonferroni 或 Benjamini‑Hochberg（FDR）  
至少在论文里说明你采用了哪种。

### F3. 失败模式记录（必须）
对每个数据集/任务记录：
- Z 在高 privacy_level 崩溃的原因（可视化/分布漂移）
- 攻击在某些 level 反常的原因（攻击过拟合/输入长度不足）

---

## G. 三类主攻击 + 两类补攻击：脚本化、可复现、可审计

### G1. Face Verification / Re‑ID（主攻）
- 数据：LFW（只评测，避免训练数据污染）
- 模型：ArcFace/FaceNet（固定版本；写入 env）
- 指标：Acc、TAR@FAR、AUC
- 输出：按 privacy_level 的曲线 + 置信区间

### G2. Attribute Inference（主攻）
- 攻击者：轻量 CNN/MLP + 训练集来自 Z‑view（A1/A2 假设）
- 指标：Acc/AUC；并报告 attack calibration（可选）

### G3. Reconstruction（主攻）
- 攻击者：U‑Net / VAE（先轻后强）
- 指标：mask 内 PSNR/SSIM/LPIPS + identity similarity

### G4. Membership Inference（补攻）
- Shadow models + thresholding
- 指标：AUC、Advantage
- 需要明确 attacker 看到的输出类型（logits/probs/labels）

### G5. Property Inference / Group Leakage（补攻）
- 攻击者预测 group（race/gender）或数据集属性
- 指标：Acc/AUC

**统一输出表字段（attack_metrics.csv）**
- dataset, task, method, training_mode, attack_type, threat_level, privacy_level, metric_name, metric_value, seed, n_samples

---

## H. 因果闭环（Causal）：从“讲故事”到“可检验”

### H1. 定义（写进 protocol）
- Treatment：`T = privacy_level` 或 `(b_s, b_t, b_b)` 预算向量
- Outcome：`Y_util`、`Y_priv`（三攻击分别作为 outcome）
- Covariates：group、mask area、亮度、姿态等可观测量

### H2. 三层稳健性（顶刊最喜欢）
1) **Balance check**：倾向评分/加权后协变量平衡（SMD < 0.1）
2) **Doubly Robust**：AIPW / DR estimator（减少模型错设风险）
3) **Sensitivity**：对未观测混杂的敏感性分析（Rosenbaum bounds 或简化版）

### H3. 输出与门槛
- ATE/CATE + 95% CI（bootstrap）
- 结论必须是“可证伪”的：例如 causal 策略在同等 utility 下能显著降低某攻击成功率（CI 不跨 0）

---

## I. Baselines：不怕麻烦，审稿人最服“强对标”

### I1. 强基线（必做）
- InstaHide
- P3

### I2. 传统弱基线（建议加，便于解释 trade‑off）
- Gaussian noise / Laplace noise（DP 风格）
- Blur / Pixelate
- JPEG 强压缩
- Random permute（简单置乱）

### I3. 对齐规则（必须写清）
- 同数据集、同任务、同 splits、同评测脚本
- privacy_level 的等效映射（若 baseline 没有 level 概念，要定义“强度参数→level”的映射曲线）

---

## J. Robustness / Practicality（很多顶刊会问）

### J1. 传输/存储开销
- C‑view：ciphertext 大小、编码开销、解密吞吐
- Z‑view：表示大小、推理吞吐

### J2. 运行时性能
- 加密/解密 FPS（batch 与单张）
- 端到端延迟（P95、P99）
- 显存占用

### J3. 失真与可视化（必须有）
- Z‑view 可视化（审稿人直觉检查）
- mask 内 vs mask 外对比图（sensitive 保密、task 保留）

---

## K. 结果目录与文件“必须到什么程度”（超具体的 DoD）

### K1. 结果目录中必须存在的文件清单（硬校验）
- `tables/utility_metrics.csv`（≥ N_rows：datasets×tasks×methods×levels×seeds）
- `tables/attack_metrics.csv`（包含 5 种 attack_type）
- `tables/causal_effects.csv`（ATE/CATE + CI）
- `tables/security_metrics_cview.csv`（含 NIST + avalanche + NPCR/UACI 等）
- `tables/ablation.csv`（≥12 ablations）
- `figures/*.png`（至少 10 张：主 6 + 补 4）
- `reports/causal_report.md`
- `reports/protocol_snapshot.md`（把关键协议参数自动 dump）

### K2. 文件行数/覆盖度的“最低线”
- `utility_metrics.csv`：至少 3 datasets × 2 tasks × 6 methods × 5 levels × 3 seeds = 540 行（每行一个 metric 或每 metric 一行，看你定义；但要能覆盖全矩阵）
- `attack_metrics.csv`：同上再乘 attack_type（≥5）= 至少 2,700 行级别
- `ablation.csv`：12 ablations × 5 levels × ≥1 dataset × ≥1 task × ≥3 seeds

> 这不是为了“凑行数”，而是为了保证你的实验矩阵真的完整。

---

## L. 你现在到“实验完全完成”的分阶段计划（每步都有“验收产出”）

### L1. 阶段 1：协议冻结 + 结果目录规范（DoD：能自动校验）
- 产出：`EXPERIMENT_PROTOCOL.md` + `RESULTS_SCHEMA.md`
- 改动：所有脚本输出统一写入 run 目录
- 验收：提供 `python scripts/validate_run.py --run_dir ...` 一键检查缺失文件/字段

### L2. 阶段 2：数据闭环与 manifest 完整（DoD：任意人能准备数据）
- 产出：`DATASETS.md`
- 产出：`data/manifests/all.jsonl`（schema validated）
- 验收：新机器从 0 到生成 all.jsonl 与 masks 成功

### L3. 阶段 3：Z‑view 训练闭环（DoD：utility 曲线稳定）
- 产出：`utility_metrics.csv` + Fig2
- 验收：3 seeds，曲线趋势一致，达到 D1 门槛（或给出原因与改进）

### L4. 阶段 4：攻击闭环（DoD：三主攻+两补攻齐全）
- 产出：`attack_metrics.csv` + Fig3
- 验收：攻击成功率随 privacy_level 上升显著下降；同时 utility 不“全崩”

### L5. 阶段 5：C‑view 传统安全全套（DoD：NIST+avalanche+鲁棒）
- 产出：`security_metrics_cview.csv` + S‑Fig
- 验收：D3.1–D3.4 达标；篡改/重放必须失败

### L6. 阶段 6：因果闭环（DoD：ATE/CATE+CI+稳健性）
- 产出：`causal_effects.csv` + Fig5 + `causal_report.md`
- 验收：balance check + DR estimator + CI 报告齐全

### L7. 阶段 7：Baselines（DoD：InstaHide+P3 同协议跑完）
- 产出：Table2 + Fig4
- 验收：至少一个数据集/任务上你方法 Pareto 更优（或解释为何）

### L8. 阶段 8：消融矩阵（DoD：≥12 ablations 完整）
- 产出：Table3 + S‑Fig
- 验收：每个 ablation 都能解释“哪一层贡献了什么”

### L9. 阶段 9：最终打包（DoD：一键 reproducible + paper-ready）
- 产出：`reproduce.sh`/`Makefile`
- 产出：最终 figures/tables 全自动生成脚本
- 验收：从 0 到最终 PDF 所需的所有图表都能自动生成

---

## M. 你要求的“传统测试 + 非传统测试”完整清单（Checklist）

### M1. 传统安全（C‑view）
- AEAD 正确性：decrypt( encrypt(m) ) = m
- 完整性：tamper 必须失败（100%）
- Nonce 唯一性/冲突概率说明
- Key space ≥ 2^128
- Entropy / Corr / NPCR / UACI
- χ² histogram uniformity
- NIST SP800‑22（≥7 项，推荐 15 项）
- Avalanche（key/nonce/plaintext bit flip ≈50%）
- 抗噪/裁剪/压缩：认证失败率≈100%（若 AEAD）
- 侧信道不做强声称，但至少报告运行时稳定性

### M2. 非传统隐私（Z‑view）
- Face verification / Re‑ID
- Attribute inference
- Reconstruction / inversion
- Membership inference
- Property inference / group leakage
- Adaptive attacker（A2 白盒）实验一组
- Cross‑dataset generalization（train dataset A，attack/test dataset B）

### M3. 系统层（Practical）
- 端到端延迟（P95/P99）
- 吞吐（FPS / images/s）
- 存储开销（bytes/image）
- 失败模式分析（高隐私强度下崩溃点）

---

## N. “如果审稿人刁钻追问”你要准备的备选证据

- 为什么 Z‑view 不做 NIST：因为目标不同；Z 以攻击成功率为证据，C 才做传统随机性。
- 为什么混沌层不单独声称安全：最终安全依赖 AEAD，混沌层是混淆/分布扰动。
- 如果 utility 掉得快：展示 train‑on‑Z/mixed training 的恢复效果；并用因果解释预算合理性。
- 如果攻击没降：说明 threat model 是否过强；增加 adaptive attacker 讨论；或者调整 Z 的扰动结构与预算策略。
- 如果 baseline 很强：强调解释性（ATE/CATE）、公平性一致性、以及 C‑view 的完整性保证。

---

## O. 你接下来“最该马上做”的三件事（因为它们决定后面全是否顺）

1) **冻结协议文件**（EXPERIMENT_PROTOCOL.md）——所有曲线/表格的字段、定义、阈值先写死  
2) **统一 run 目录 + csv schema**（RESULTS_SCHEMA.md + validate_run.py）  
3) **把攻击脚本变成标准化产线**（attack_metrics.csv 字段固定，自动出图）

只要这三件做完，后面跑再多实验也不会“散、乱、不可复现”。

---

> 如果你愿意，我可以把这份 V2 计划继续升级到 V3：**逐路径逐文件的“函数级任务清单”**（例如 `scripts/run_benchmark.py` 增哪些 flag、`src/evaluation/*` 要新增哪些类、每个 csv 字段定义与单位、每个图的绘图脚本输入输出），做到你可以把它当作项目管理的唯一真源（single source of truth）。
