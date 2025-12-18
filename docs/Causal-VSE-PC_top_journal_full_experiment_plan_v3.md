# Causal‑VSE‑PC 顶刊冲刺 V3（终极版）：逐文件/逐接口/逐指标/逐产出清单 + 近 5 年高质量论文对标路线图

> 你要的是“**最后一次冲顶刊**”级别的执行文档：照着做，跑完就是 paper‑ready；缺一项就知道缺在哪。  
> V3 的核心：**把“实验”变成可审计的产线**——每个脚本、每个 CSV、每个图、每个阈值、每个 baseline、每个攻击都能定位到仓库路径与 DoD。

---

## 0. 北极星（North Star）：你要投顶刊时，审稿人只能从“创新性/边界条件”质疑，而不是从“协议不严谨/不复现/不对标/指标不标准”打回

### 0.1 论文主张（必须写进 `EXPERIMENT_PROTOCOL.md`，全项目统一口径）
- **C‑view（强密文）**：用于存储/传输安全；安全主张基于标准 AEAD（如 ChaCha20‑Poly1305/AES‑GCM），并用传统统计与密码学测试作为补充证据。
- **Z‑view（可用密文表示）**：用于云端/第三方推理；隐私主张必须靠**攻击成功率**证明（身份识别/属性推断/重建/成员推断/属性群体泄漏等），而不是用 NIST 去“证明 Z 随机”。  
（这能避免审稿人抓“你把目标不同的指标混用”的漏洞。）

### 0.2 你要对标的“近 5 年（2020–2025）高质量工作”分组（Related Work Map）
> 这部分会直接决定你在顶刊里“站在哪一条研究链上”。（下面每条都给了可引用的代表作。）

**A. 训练阶段/实例隐藏（Instance Hiding / Data Mixing）**
- InstaHide（2020/2021）：私有数据训练时混合与遮罩，强调“对现有训练管线的可插拔”。citeturn0search1turn0search5

**B. 目标属性隐私的图像混淆（Targeted Attribute Obfuscation）**
- InfoScrub（CVPRW 2021）：针对属性推断做“最大化不确定性”的混淆，同时尽量保持画面保真。citeturn1search1turn1search9

**C. 对抗式“披风/隐身衣”（Cloaking against Face Recognition）**
- Fawkes（USENIX Security 2020）：通过不可见扰动使人脸识别训练出来的模型“识别错人”。citeturn2search0turn2search4  
- OPOM（TPAMI 2022，代码公开）：一人一 mask 的通用披风思路（审稿人很熟）。citeturn2search10

**D. 人脸匿名化（Face Anonymization / GAN‑based）**
- DeepPrivacy（2019）及其后续 toolbox（DeepPrivacy2）：生成式匿名化，强调身份移除与视觉质量。citeturn2search1turn2search17

**E. “压缩/编码阶段防 VLP/视觉模型滥用”（Privacy‑aware Compression / Packaging）**
- PSIC（2025 arXiv）：在压缩码流层面引入“可多种解码选项”的隐私屏蔽思路，面向 VLP/通用视觉分析防护。citeturn1search18

**F. 视觉隐私攻击与防御综述（写 Related Work/Threat Model 的底盘）**
- Visual privacy attacks and defenses survey（2021）：把攻击面/防御面体系化，便于你写 threat model 与攻击族。citeturn0search10

> 你的定位建议：**你不是单纯“图像加密”**，也不是单纯“匿名化/披风”，而是 **Dual‑View（Z 可用 + C 强安全）+ 语义区域预算 + 因果可解释预算分配 + 可验证性** 的组合拳。  
> 顶刊写法：把你放到 A/B/C/D/E 各链条的交叉点，同时声明你解决的是“**可用密文推理**”与“**强安全存储/传输**”在同一系统中的闭合问题。

---

## 1. 仓库结构的“顶刊化改造”蓝图（可以大改，但要有纪律）

你的 README 已经把核心模块列出来了（SCNECipherAPI、pipeline、评估器等）。citeturn3view0  
V3 要做的是：把它升级为“产线级”——**manifest 驱动、协议冻结、统一输出 schema、任何评测都走同一入口**。

### 1.1 新增目录（建议一次性加齐）
```
docs/
  EXPERIMENT_PROTOCOL.md
  RESULTS_SCHEMA.md
  DATASETS.md
  RELATED_WORK_2020_2025.md
  baselines/
    instahide.md
    p3.md
    infoscrub.md
    fawkes_opom.md
  attacks/
    face_verification.md
    attribute_inference.md
    reconstruction.md
    membership_inference.md
    property_inference.md

src/
  data/
    manifest_schema.py
    manifest_io.py
    dataset_registry.py
  evaluation/
    attacks/
      face_verification.py
      attribute_inference.py
      reconstruction.py
      membership_inference.py
      property_inference.py
    nist_sp800_22.py
    avalanche.py
  plotting/
    make_figures.py
    figure_specs.py
  baselines/
    instahide.py
    p3.py
    infoscrub.py   # 只做对齐的“评测包装”，不一定要复现训练
scripts/
  prepare/
    prepare_celeba.py
    prepare_fairface.py
    prepare_openimages_pii.py
  run/
    run_benchmark.py
    run_attacks.py
    run_security_cview.py
    run_causal.py
    run_baselines.py
    make_figures.py
  validate/
    validate_run.py
    validate_manifest.py
```

---

## 2. V3 的“单一真源”（Single Source of Truth）：三份协议文档（必须先写死）

### 2.1 `docs/EXPERIMENT_PROTOCOL.md`（冻结口径）
必须包含：
1) Threat model（A0/A1/A2 分级）  
2) Z/C 的边界与安全主张  
3) 数据集、split、预处理  
4) 任务定义（分类/分割/检测）  
5) privacy_level 网格（固定为 L={0,0.3,0.5,0.7,1.0}，除非论文里声明改网格）  
6) 攻击协议（输入、攻击者能力、模型、训练轮数、early stop、指标）  
7) 统计方式（seeds、CI、显著性、多重比较校正）

### 2.2 `docs/RESULTS_SCHEMA.md`（冻结输出）
定义所有 CSV 的字段、单位、粒度、允许缺失值、写入规则（append vs overwrite）。  
并且定义每张图的文件名、分辨率（例如 300 dpi）、曲线平滑规则（不允许偷偷平滑）。

### 2.3 `docs/DATASETS.md`（冻结数据闭环）
- 不上传数据；只写下载/许可/预处理命令。  
- 每个数据集给出：目录结构、split 生成方式、manifest 字段映射、敏感属性/任务属性定义。  
（这一点是顶刊 artifact review 的生死线。）

---

## 3. Manifest 驱动（强制）：从“读目录”升级为“读协议化数据表”

### 3.1 新文件：`src/data/manifest_schema.py`
**必须提供：**
- `ManifestRecord`（dataclass/pydantic）
- `validate_record(record) -> List[str]`
- `validate_manifest(path) -> (n_ok, n_bad, bad_examples)`

**字段最低集（必须）**
- `id, dataset, split, image_path`
- `labels_task`（任务标签）与 `labels_sensitive`（敏感属性标签）
- `group`（FairFace：race/gender/age；无则空）
- `mask_sensitive_path, mask_task_path, mask_bg_path`
- `privacy_level_default`（float）
- `privacy_map_path`（可空）
- `z_path, c_path, enc_info_path`（可空）
- `sha1_image, sha1_masks, sha1_views`（可分开）

### 3.2 新脚本：`scripts/validate/validate_manifest.py`
- 输入：manifest.jsonl
- 输出：0/1 return code（CI 用）
- 失败：打印前 20 条错误样例（字段缺失/路径不存在/哈希不一致）

**DoD**
- `python scripts/validate/validate_manifest.py --manifest data/manifests/all.jsonl` 必须 exit 0。

---

## 4. “统一入口”与 run 目录规范：把实验变成一条流水线

你当前已经有 `scripts/run_benchmark.py` 作为统一入口的雏形。citeturn4view0  
V3 要把它升级为“顶刊版 runner”。

### 4.1 重构：`scripts/run/run_benchmark.py`（或保留原名但路径统一）
**要求：一个命令跑完整闭环**
- Utility（Z‑view）
- Attacks（Z‑view）
- Causal（ATE/CATE）
- Security（C‑view）
- Baselines（可选 `--with_baselines`）

**必须参数**
- `--manifest`
- `--config`（yaml）
- `--seeds 0 1 2`
- `--privacy_levels 0 0.3 0.5 0.7 1.0`
- `--methods plaintext vsepc_causal vsepc_uniform vsepc_sensitive_only instahide p3 ...`
- `--tasks attr_cls segm ...`
- `--output_root results/`

### 4.2 新脚本：`scripts/validate/validate_run.py`
输入 `--run_dir`，检查：
- meta/ 必须文件是否齐
- tables/ 必须 CSV 是否存在且字段齐全
- figures/ 必须图是否存在且尺寸合规
- 所有 CSV 行数是否覆盖实验矩阵（见 12.2 的“最低行数门槛”）

**DoD**
- 任意 run 都能通过 validate；否则拒绝生成论文图表。

---

## 5. Z‑view 可用性（Utility）：必须“训练域适配”，否则顶刊站不住

你 README 已强调“密文域 ML”。citeturn3view0  
顶刊审稿人的第一问：**你是 train‑on‑Z 还是拿明文模型硬推？**

### 5.1 新脚本：`scripts/run/run_utility.py`
支持三种训练模式：
- `P2P`：train P, test P（上限基准）
- `P2Z`：train P, test Z（对照，通常会掉）
- `Z2Z`：train Z, test Z（必须）
- `Mix2Z`：train (P+Z), test Z（强烈推荐）

**输出：`tables/utility_metrics.csv`**
字段（冻结）：
- `dataset, split, task, method, training_mode, privacy_level, seed`
- `metric_name`（acc/macro_f1/miou/dice…）
- `metric_value`
- `n_samples`
- `ci_low, ci_high`（95% CI）
- `time_train_s, time_eval_s`

### 5.2 Utility 门槛（建议写进 protocol，达不到就必须解释或改方法）
- privacy_level=0.3：`Utility(Z2Z) >= 0.75 * Utility(P2P)`
- privacy_level=0.5：`>= 0.65 *`
- privacy_level=0.7：`>= 0.50 *`（允许更低，但要把 Pareto 讲清楚）
- std（across seeds）建议 ≤ 0.03（分类 acc）或 ≤ 0.05（复杂任务）

---

## 6. Z‑view 隐私（Privacy）：必须用攻击成功率闭合（至少 3 主攻 + 2 补攻）

你项目已有 AttackEvaluator 框架（四类攻击）。citeturn5view2  
V3 要补齐：**威胁等级标注 + 两个补攻 + 统一输出 + 跨数据集验证**。

### 6.1 攻击清单（2020–2025 顶刊/顶会常见）
**主攻 1：Face Verification / Re‑ID**
- 外部验证集 LFW（只评测）  
- 预训练识别器（ArcFace/FaceNet）固定版本  
- 指标：AUC、TAR@FAR=1e‑3、EER  
（Fawkes/OPOM/匿名化论文都会被拿来对照 threat model。citeturn2search4turn2search10）

**主攻 2：Sensitive Attribute Inference**
- 攻击者在 Z 上训练分类器预测敏感属性  
- 指标：AUC/Acc，multi‑class 用 macro‑F1  
（InfoScrub 这条线你必须对齐。citeturn1search1）

**主攻 3：Reconstruction / Model Inversion**
- 攻击者训练 U‑Net/VAE: Z→P（或 Z→sensitive‑region）  
- 指标（必须 mask 内）：PSNR/SSIM + identity similarity（embedding 相似度）  
（匿名化/反匿名化审稿人会盯这个。citeturn2search17）

**补攻 A：Membership Inference（成员推断）**
- Shadow models 或 LiRA（轻量版也可）  
- 指标：Attack AUC / advantage  
（这是近几年隐私论文“必问项”，哪怕你做轻量版也比不做强。）

**补攻 B：Property Inference / Group Leakage**
- 攻击者预测 group（FairFace：race/gender/age）或数据集属性  
- 指标：Acc/AUC  
（用来证明你不是“牺牲某些群体的隐私来换整体指标”。）

### 6.2 新模块：`src/evaluation/attacks/*.py`
每个攻击一个文件，必须实现统一接口：
- `fit(train_loader, **kwargs)`
- `evaluate(test_loader) -> Dict[str,float]`
- `threat_level`（A0/A1/A2）

### 6.3 输出：`tables/attack_metrics.csv`
字段（冻结）：
- `dataset, task, method, training_mode, attack_type, threat_level`
- `privacy_level, seed, metric_name, metric_value, n_samples, ci_low, ci_high`

**门槛（方向性 + 最低线）**
- Face verification：privacy_level 上升，AUC 显著下降；中等强度下明显低于明文（并给出 CI）
- Attribute inference：逼近随机（或显著低于明文）
- Reconstruction：敏感区域内 PSNR/SSIM 显著下降；identity similarity 显著下降
- Membership/property：AUC 接近 0.5

---

## 7. C‑view 传统安全（Traditional Security）：必须做“全套”，否则安全向顶刊会不认

你 README 已列 entropy/NPCR/UACI/相关性/χ² 等标准。citeturn3view0  
你脚本里也有 security 评测入口。citeturn4view3  
V3 要补齐：**NIST SP800‑22 + avalanche + tamper/replay + nonce 规则说明**。

### 7.1 新模块：`src/evaluation/nist_sp800_22.py`
实现最小集（7 项）：
- Frequency, BlockFrequency, Runs, LongestRun, FFT, Serial, ApproxEntropy  
**DoD**
- 允许输入 bitstream（从 ciphertext bytes 拼接）
- 输出每项 p‑value + pass/fail（阈值 0.01）
- 报告输入长度（bits）

### 7.2 新模块：`src/evaluation/avalanche.py`
三类 avalanche：
- key flip（模拟 1 bit 改动）
- nonce flip
- plaintext flip（1 bit / 1 pixel）
输出 flip_rate（期望≈50%）与 std。

### 7.3 Tamper/Replay 测试（AEAD 的“生命线”）
在 `src/crypto/key_system.py` 或 crypto wrap 实现处，必须提供：
- `encrypt(m, aad) -> (c, nonce, tag)`
- `decrypt(c, nonce, tag, aad) -> m | raise`
实验：
- 改 1 bit 的 ciphertext 必须失败
- 改 1 bit 的 tag 必须失败
- AAD 改动必须失败
- nonce 重放必须在协议层被识别（至少写明“允许重放但不影响机密性/完整性”或“上层拒绝重复 nonce”）

### 7.4 输出：`tables/security_metrics_cview.csv`
字段（冻结）：
- `dataset, method, privacy_level, seed`
- `entropy, npcr, uaci, corr_h, corr_v, corr_d, chi2_p`
- `nist_test_name, nist_p, nist_pass`
- `avalanche_type, flip_rate, flip_std`
- `tamper_success_rate`（必须接近 0，或认证失败率接近 1）

---

## 8. 因果闭环（Causal）：从“讲解释”升级为“可检验 + 可稳健”

你的 pipeline 里已体现“allocate→analyze_allocation→proof”流程。citeturn3view1  
V3 目标：让审稿人相信：**预算分配不是拍脑袋，而是能在可观测数据下被检验的因果效应估计**。

### 8.1 新模块：`src/vse_pc/reporting.py`
提供：
- `make_causal_report(ate, cate, ci, balance_stats, sensitivity) -> markdown`

### 8.2 因果估计最小闭环（必须）
- ATE：`privacy_level` 对 `utility` 与 `attack_success` 的平均效应
- CATE：按 `group`（FairFace）与 `mask_area_bin` 分层
- CI：bootstrap（≥500，建议 1000）
- Balance check：SMD < 0.1（倾向评分/加权后）
- Sensitivity（简化版也行）：说明未观测混杂会怎样影响结论

### 8.3 输出：`tables/causal_effects.csv` + `reports/causal_report.md`

---

## 9. Baselines（近 5 年链条里必须对齐的基线）与“等效强度映射”

### 9.1 必做强基线
- **InstaHide**：训练阶段实例隐藏。citeturn0search5  
- **P3**：公开/私有分解的 photo sharing 思路（虽然旧，但是经典“部分加密/可变换”参照物）。citeturn1search0

### 9.2 强相关对比（可做但至少要在 related work 里讨论）
- InfoScrub（目标属性混淆）。citeturn1search1  
- Fawkes/OPOM（披风）。citeturn2search4turn2search10  
- DeepPrivacy2（生成式匿名化）。citeturn2search17  
- PSIC（压缩阶段防视觉模型滥用）。citeturn1search18

> 注意：你不必都“复现训练”，但必须至少做到：同一任务/数据/攻击协议下的“可比较输出”（哪怕是复用作者模型或公开实现）。

### 9.3 等效强度映射（必须写进 baseline 文档）
如果 baseline 没有 privacy_level：
- 定义 `strength_param -> privacy_level` 的单调映射（用攻击成功率/失真做对齐）
- 例如：在同等 Face‑Verification AUC 降幅下对齐为同一 privacy_level  
（顶刊审稿人会接受“按风险对齐”，比按噪声幅度对齐更合理。）

---

## 10. 近 5 年论文对标“落地到仓库”的方式（你要的：不是口头引用，是可执行对比）

### 10.1 新文档：`docs/RELATED_WORK_2020_2025.md`
必须包含一个表（每行一个工作），字段：
- Year / Venue / Task / Threat model / Defense type / Utility metric / Privacy metric / Notes / Repro link
并把下面这些关键引用填进去（至少）：
- InstaHide（2020/2021）citeturn0search1turn0search5
- InfoScrub（2021）citeturn1search1
- Fawkes（2020）citeturn2search4
- OPOM（TPAMI 2022）citeturn2search10
- PSIC（2025）citeturn1search18
- Visual privacy survey（2021）citeturn0search10

### 10.2 新脚本：`scripts/lit/update_related_work.py`（可选但很强）
- 输入：`bibtex` 或 `papers.yaml`
- 输出：自动生成的 related‑work 表格（markdown）
（这样你投稿前改版本不会乱。）

---

## 11. 绘图与表格：把“paper-ready 输出”工程化（避免最后手工拼表翻车）

### 11.1 新模块：`src/plotting/figure_specs.py`
冻结：
- 每张图的标题、轴、legend、线型、dpi、字体、尺寸（例如 6.5in 宽）
- 不允许在 notebook 里手工改线条（要可复现）

### 11.2 新脚本：`scripts/run/make_figures.py`
- 输入：run_dir 或多个 run_dir
- 输出：figures/ 下所有主图与补充图

**DoD**
- 给任何 run_dir，都能生成：
  - `fig_utility_curve.png`
  - `fig_attack_curves.png`
  - `fig_pareto_frontier.png`
  - `fig_causal_ate_cate.png`
  - `fig_cview_security_summary.png`
  - `fig_ablation_summary.png`
  - `fig_efficiency.png`

---

## 12. 你要的“非常清晰”的量化验收（数值阈值 + 覆盖度阈值）

### 12.1 关键阈值（建议写进 protocol，作为 gate）
**C‑view（传统安全）最低线**
- entropy ≥ 7.95  
- NPCR ≥ 99.5%  
- UACI ∈ [31, 35]（经验区间）  
- |corr_h|,|corr_v|,|corr_d| ≤ 0.02  
- χ² p ≥ 0.01  
- NIST：最小 7 项中 pass ≥ 6（每项 p≥0.01）  
- Avalanche：flip_rate ∈ [45%, 55%]（key/nonce/plaintext）

**Z‑view（隐私）方向性线**
- Face verification AUC 随 privacy_level 单调下降（至少总体趋势），并提供 CI
- Attribute inference AUC 接近 0.5（或显著低于明文）
- Membership inference AUC 接近 0.5
- Reconstruction：敏感区域内 PSNR/SSIM 显著下降 + identity similarity 显著下降

**Z‑view（效用）最低线**
- privacy_level=0.3：≥ 0.75×plaintext（Z2Z / Mix2Z）
- privacy_level=0.5：≥ 0.65×plaintext

### 12.2 覆盖度阈值（防“漏跑”）
假设你做：
- datasets=3（CelebA/FairFace/OpenImages‑PII）
- tasks=2（attr_cls + seg）
- methods=6（P2P、VSEPC‑causal、uniform、sensitive‑only、InstaHide、P3）
- levels=5
- seeds=3

那么：
- utility 行数最低应 ≥ 3×2×6×5×3 = **540**（每个组合至少一个主指标；如果每个指标一行会更多）
- attack 行数最低应 ≥ 540×(attack_types=5) = **2700**
- security 行数最低应 ≥ 3×1×(methods含 C‑view 的至少 1–2 个)×5×3 ×(nist_tests/avalanche 展开)  
（你可以定义为“宽表”或“长表”，但 validate_run 必须能检查覆盖。）

---

## 13. 逐文件/逐接口执行清单（你要的“V3 真正核心”）

> 说明：下面每条都是可以直接开 PR 的粒度。  
> 标注：
> - **[MOD]** 修改现有文件
> - **[NEW]** 新增文件
> - **DoD**：完成到什么程度算完成（可验收）

---

### 13.1 根目录与文档

**[NEW] `docs/EXPERIMENT_PROTOCOL.md`**  
- 写 threat model（A0/A1/A2）  
- 写 Z/C 边界  
- 写 privacy_level 网格  
- 写每项攻击协议与训练设置  
**DoD**：文档里每个指标都有公式/定义/单位；每个攻击都有输入输出与威胁等级。

**[NEW] `docs/RESULTS_SCHEMA.md`**  
- 定义所有 csv 字段、类型、单位、是否必填  
- 定义结果目录树  
**DoD**：validate_run 以此为标准进行严格检查。

**[NEW] `docs/DATASETS.md`**  
- CelebA/FairFace/OpenImages‑PII 获取与许可说明  
- prepare 脚本使用说明  
**DoD**：第三方按文档可从 0 生成 manifest + masks。

**[NEW] `docs/RELATED_WORK_2020_2025.md`**  
- 按 A/B/C/D/E/F 分类写表格与点评  
**DoD**：至少覆盖 InstaHide/InfoScrub/Fawkes/OPOM/PSIC/Survey（并给链接/引用）。citeturn0search5turn1search1turn2search4turn2search10turn1search18turn0search10

---

### 13.2 数据层（manifest）

**[NEW] `src/data/manifest_schema.py`**  
- `ManifestRecord`  
- `validate_*`  
**DoD**：对缺字段/路径错/哈希错能输出可读错误。

**[NEW] `src/data/manifest_io.py`**  
- `read_jsonl(path) -> Iterable[Record]`  
- `write_jsonl(records, path)`  
**DoD**：对大文件流式处理，不爆内存。

**[NEW] `scripts/validate/validate_manifest.py`**  
**DoD**：exit code 可用于 CI。

**[MOD] `src/utils/datasets.py`**（你已有 CelebA/CelebAMask/Attributes loader）citeturn5view4  
- 新增：FairFace dataset loader（含 group label）  
- 新增：OpenImages‑PII 子集 loader（person/face/plate）  
- 新增：`ManifestDataset(Dataset)`：从 manifest 读样本  
**DoD**：所有训练/攻击/评测统一走 ManifestDataset（杜绝“脚本各读各的目录”）。

---

### 13.3 核心流水线（Z/C 生成）

**[MOD] `src/vse_pc/pipeline.py`**（当前 mask 为模拟中心区域）citeturn3view1  
- 去掉 `unet_mock`：改为注入真实 `SemanticMaskGenerator`（见下）  
- `_derive_keys`：把 nonce/region_key 的规则写清并可配置  
- 增加 `mode='z'|'c'`：同一个 pipeline 可导出 Z 或 C  
- 输出结构：必须包含 `enc_info_path` 可落盘（nonce/tag/aad/version）

**DoD**
- 给同一 id：能导出 Z 与 C 两份产物；Z 可喂模型，C 可 AEAD 验证失败/成功。

**[MOD] `src/cipher/scne_cipher.py`**（主 API）citeturn3view0  
- 增加：`encrypt(image, privacy_level, semantic_masks, mode)`  
- 明确：Z 的 crypto_wrap 是否开启、C 必须开启  
**DoD**：Z/C 的 API 路径一致，避免“两个体系各自维护”。

**[NEW] `src/data/semantic_mask_generator.py`（或升级你已有版本）**  
- 支持：强监督 mask、弱监督 U‑Net mask、规则 mask  
**DoD**：给一张图，输出三 mask：sensitive/task/background，并能统计面积分布。

---

### 13.4 Utility 评测（任务性能）

**[MOD] `src/evaluation/utility_evaluator.py`**（已有 3 类任务框架）citeturn5view6  
- 增加：CI 计算（bootstrap）  
- 增加：group‑wise（FairFace）统计输出  
- 增加：训练模式记录字段（P2Z/Z2Z/Mix2Z）

**[NEW] `scripts/run/run_utility.py`**  
**DoD**：生成 `tables/utility_metrics.csv` 且字段齐全。

---

### 13.5 攻击评测（隐私）

**[MOD] `src/evaluation/attack_evaluator.py`**（已有攻击评估器框架）citeturn5view2  
- 把攻击实现拆到 `src/evaluation/attacks/*`  
- 新增：membership/property inference  
- 每个攻击必须带 threat_level 字段

**[NEW] `src/evaluation/attacks/face_verification.py`**  
- LFW eval pipeline（只评测）  
**DoD**：输出 AUC、TAR@FAR、EER。

**[NEW] `src/evaluation/attacks/attribute_inference.py`**  
- 训练攻击模型并评测（A1/A2）  
**DoD**：输出 AUC/Acc + CI。

**[NEW] `src/evaluation/attacks/reconstruction.py`**  
- U‑Net/VAE 重建  
**DoD**：输出 mask 内 PSNR/SSIM + identity similarity。

**[NEW] `src/evaluation/attacks/membership_inference.py`**  
- shadow model 轻量版  
**DoD**：输出 AUC/advantage。

**[NEW] `src/evaluation/attacks/property_inference.py`**  
- group leakage  
**DoD**：输出 AUC/Acc。

**[NEW] `scripts/run/run_attacks.py`**  
**DoD**：生成 `tables/attack_metrics.csv` 并能画曲线。

---

### 13.6 C‑view 传统安全（NIST/avalanche/tamper）

**[MOD] `src/evaluation/security_metrics.py`**（现有实现但需补齐）citeturn4view1  
- 确保：entropy/NPCR/UACI/corr/chi2 计算标准一致  
- 增加：输入位流导出（供 NIST）

**[NEW] `src/evaluation/nist_sp800_22.py`**  
**DoD**：最小 7 项可跑、结果写入 csv。

**[NEW] `src/evaluation/avalanche.py`**  
**DoD**：flip_rate 输出 45%–55% 区间并可统计。

**[MOD] `scripts/evaluation/security.py`**（已有入口）citeturn4view3  
- 改为调用 nist/avalanche/tamper  
**DoD**：生成 `tables/security_metrics_cview.csv`。

---

### 13.7 因果与报告

**[MOD] `src/vse_pc/causal_analysis.py` + `privacy_budget.py`**  
- 输出：ATE/CATE + CI  
- 增加：balance check 与 sensitivity（简化版）  
**DoD**：`tables/causal_effects.csv` + `reports/causal_report.md` 自动生成。

---

### 13.8 Baselines

**[NEW] `src/baselines/instahide.py`**  
- 目标：最小可运行实现或包装公开实现（优先可复现）citeturn0search5  
**DoD**：能生成等效的 Z‑like 输入并走同一评测脚本。

**[NEW] `src/baselines/p3.py`**  
- 目标：实现 public/private split（频带/系数分解）citeturn1search0  
**DoD**：同上。

**[NEW] `scripts/run/run_baselines.py`**  
**DoD**：baseline 也产出同 schema 的 csv，能进 Fig4。

---

### 13.9 绘图与表格

**[NEW] `src/plotting/make_figures.py` + `figure_specs.py`**  
**DoD**：从 tables/*.csv 一键生成所有 paper 图。

**[NEW] `scripts/run/make_figures.py`**  
**DoD**：`python scripts/run/make_figures.py --run_dir results/...` 生成全套图。

---

### 13.10 验证与 CI

**[NEW] `scripts/validate/validate_run.py`**  
**DoD**：严格检查缺文件/缺列/覆盖不足/图像缺失即失败。

**[MOD] `tests/`（新增至少 12 个单测）**
- manifest validate  
- deterministic nonce（如支持）  
- tamper 必失败  
- Z/C 导出一致性  
- nist/avalanche 输出格式  
- plotting 产物存在性

---

## 14. 你最关心的“近 5 年顶刊/顶会对标怎么写进实验”：一页纸策略

### 14.1 你必须在论文里明确对比的“主线问题”
1) **训练阶段保护 vs 推理阶段保护**：InstaHide 属于前者，你属于后者（同时兼顾存储/传输 C‑view）。citeturn0search5  
2) **目标属性混淆**：InfoScrub 属于“属性隐私‑保真权衡”，你要对齐“属性推断攻击”协议。citeturn1search1  
3) **披风**：Fawkes/OPOM 重点是“破坏识别器特征空间”，你需要在 face verification 任务下给出直接对比或讨论。citeturn2search4turn2search10  
4) **匿名化**：DeepPrivacy2 强调视觉可用，隐私通常靠识别下降；你强调可用密文推理 + 强安全 C‑view。citeturn2search17  
5) **压缩阶段隐私**：PSIC 代表“在系统链路更前端做隐私屏蔽”，你要在 “系统落地性/多解码选项” 上讨论。citeturn1search18

### 14.2 你必须给出的“对标维度表”（建议做成 Table2）
- Utility：分类/分割指标
- Privacy：face verification / attribute inference / reconstruction / membership / property  
- Practical：延迟/吞吐/存储开销  
- Explainability：是否给出因果解释（你是核心卖点）  
- Strong security：是否提供 AEAD + tamper protection（你在 C‑view 上要明确强）

---

## 15. 最终：从 0 到 paper‑ready 的“一键命令”与 Makefile（建议）

**[NEW] `Makefile`**
- `make prepare_data`
- `make build_manifest`
- `make gen_masks`
- `make export_views`
- `make run_all`（utility+attacks+causal+security+baselines）
- `make validate`
- `make figures`

**DoD**
- 任意人 clone 后按 README/Makefile 可复现主图主表。

---

# 结语：你现在最该“立刻开干”的 5 个 PR（决定成败）

1) 协议冻结：`EXPERIMENT_PROTOCOL.md` + `RESULTS_SCHEMA.md` + `validate_run.py`  
2) Manifest 驱动：`manifest_schema.py` + `ManifestDataset` + `validate_manifest.py`  
3) 攻击产线：把攻击实现拆模块 + 输出 attack_metrics.csv（5 攻齐）  
4) C‑view 传统安全全套：NIST + avalanche + tamper + 统一输出  
5) Baseline 对齐：InstaHide + P3 先跑通小规模，再扩规模

> 你说“加油冲顶刊”，那 V3 的逻辑就是：**把所有审稿人可能问的“你有没有做 X？”都提前做到，并且做到可复现、可审计、可对标。**
