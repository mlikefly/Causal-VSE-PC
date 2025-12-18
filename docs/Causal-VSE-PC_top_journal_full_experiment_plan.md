# Causal‑VSE‑PC 顶刊实验「全闭环」工作计划（从现在到实验完全完成）

> 目标：把当前“系统能跑 + C 视图安全指标过线”的原型，升级为**顶刊可审、可复现、可对标、可解释**的完整实验闭环。  
> 核心口径：**Z‑view 负责可用密文域推理（utility + attacks 主证据）**；**C‑view 负责存储/传输的强安全与完整性（统计/密码学类指标作为补充证据）**。

---

## 0. 全局定义：什么叫“实验完全完成”（Definition of Done）

当且仅当以下 8 类产出全部齐备且达标，你的“实验部分”才算完成：

1) **复现性**  
- 任意新机器、按 README 一键命令，可在无交互情况下跑出同样结构的结果目录（允许数值轻微浮动但曲线趋势一致）。
- 每次实验输出必须包含：`config.yaml`、`git_commit.txt`、`seed.txt`、`env.txt`、`dataset_manifest_hash.txt`、`runtime.json`。

2) **数据闭环**（不上传原始数据到 Git）  
- 仓库提供 `DATASETS.md` + `scripts/prepare_*.py`（用户自行下载后，一键生成 manifest / masks / views）。

3) **Z‑view 可用性曲线**（主战场）  
- 至少 2 个任务（推荐：属性分类 + 分割；检测可选）  
- 形成 `privacy_level ∈ {0.0, 0.3, 0.5, 0.7, 1.0}` 的完整曲线（mean±std，≥3 seeds）。

4) **三类攻击评测**（主战场）  
- 身份验证/识别（LFW 等）  
- 属性推断（敏感属性：性别/种族/年龄等）  
- 重建攻击（U‑Net/VAE；可升级到 GAN/扩散）  
- 输出：`attack_success vs privacy_level` 曲线 + 表格。

5) **因果证据链**（论文卖点）  
- 明确 treatment：`do(privacy_level)` 或 `do(privacy_map)`  
- 输出 ATE/CATE（区域敏感/任务/背景）+ 置信区间（bootstrap）+ 与启发式策略对比。

6) **强基线对标**  
- 至少 2 个：InstaHide、P3  
- 对齐同一任务/数据/协议，输出同样的曲线与表格。

7) **消融实验**（审稿人必问项）  
- 三层结构逐层去掉  
- 因果预算 vs 启发式预算  
- 语义 mask 来源（强监督 vs 弱监督）  
- FFT vs DWT / semantic_preserving 开关等

8) **论文可直接引用的产物**  
- 4 张主图 + 2 张补充图 + 3 张主表 + 1 张补充表（见第 8 节模板）  
- 每张图/表对应脚本可一键重生成。

---

## 1. 里程碑总览（不写时间，只写顺序与验收）

- **M0（现在）**：原型能跑 + C‑view 安全指标过线 ✅（你已完成）
- **M1（复现地基）**：manifest 驱动 + 结果目录元信息齐全 + 一键命令跑通
- **M2（Z‑view 可用性）**：train‑on‑Z / mixed‑training 曲线稳定（≥3 seeds）
- **M3（攻击闭环）**：3 类攻击曲线 + 协议固定
- **M4（因果闭环）**：ATE/CATE + CI + 显著性 + 与启发式对比
- **M5（基线对标）**：InstaHide + P3 同协议跑通（先小规模，后大规模）
- **M6（消融矩阵）**：关键 ablation 全部跑完
- **M7（论文级打包）**：paper‑ready figures/tables + artifact release 清单

---

## 2. 仓库“应达到的文件状态”（新增/改动清单 + 数量）

### 2.1 必须新增的文档（≥ 3 个）
- `DATASETS.md`（数据许可、获取方式、目录结构、预处理命令、一键复现说明）
- `EXPERIMENT_PROTOCOL.md`（威胁模型、任务定义、攻击协议、指标定义、统计方式）
- `RESULTS_SCHEMA.md`（结果目录结构、csv 字段、图表命名规范）

### 2.2 必须新增的脚本（建议新增 12–18 个）
放在 `scripts/` 下，保持“单一职责 + 可组合”：
- 数据准备：`scripts/prepare_{celeba,celeba_hq,fairface,ffhq,openimages}.py`（3–5 个）
- manifest：`scripts/build_manifest.py`（你已有，可升级）
- mask：`scripts/gen_semantic_masks.py`（你已有，可升级）
- privacy map：`scripts/gen_privacy_maps.py`（你已有，可升级）
- 视图导出：`scripts/export_zview_dataset.py`、`scripts/export_cview_dataset.py`（你已有，可升级）
- 训练：`scripts/training/train_task_model.py`、`scripts/training/train_on_z.py`（新增 1–2 个）
- 攻击：`scripts/evaluation/attack_face_verification_lfw.py`、`attack_attribute_inference.py`、`attack_reconstruction.py`（新增 3 个）
- baseline：`scripts/baselines/run_instahide.py`、`run_p3.py`（新增 2 个）
- 汇总：`scripts/run_benchmark.py`（你已有，可升级为单入口）

> 目标：新增脚本总数 **≥12** 且每个脚本都能独立运行、写入结果目录。

### 2.3 必须新增/调整的模块（建议新增 6–10 个 py 文件）
- `src/data/manifest_schema.py`（定义 JSONL schema + 校验）
- `src/data/dataset_registry.py`（统一 dataset 名称、split、标签映射）
- `src/evaluation/attacks/{face_verification,attribute_inference,reconstruction}.py`
- `src/plotting/make_figures.py`（一键从 csv 生成论文图）
- `src/vse_pc/reporting.py`（ATE/CATE 报告模板生成）

> 目标：新增模块文件 **≥6**，并新增单元测试覆盖关键接口（见第 7 节）。

---

## 3. 复现地基：manifest 驱动 + 结果目录规范（M1）

### 3.1 Manifest 规范（JSONL，一行一条样本）

**必须字段（最低可复现集）**
- `id`: 唯一 ID（dataset + split + index）
- `dataset`: `celeba|celeba_hq|fairface|ffhq|openimages_pii`
- `split`: `train|val|test`
- `image_path`
- `labels`: 任务标签（属性/分割/检测）
- `group`: 公平性分组（FairFace 必须：race/gender/age）
- `masks`: `sensitive|task|background` 三类 mask 路径（可空但必须字段存在）
- `privacy_level`: 默认全局（也可为 list）
- `privacy_map_path`: region‑wise map 的路径（可空）
- `z_path`: Z‑view 输出路径（可空）
- `c_path`: C‑view 输出路径（可空）
- `enc_info_path`: nonce/tag/version/commitment 等元信息路径（可空）
- `hash`: `sha1`（image + mask + views 可分别存）

**验收标准**
- `python scripts/build_manifest.py ...` 产出：  
  - `data/manifests/{dataset}.jsonl`  
  - `data/manifests/all.jsonl`（统一入口）  
  - `data/manifests/schema.json`（schema 版本号+字段解释）  
- 校验脚本：`python -m src.data.manifest_builder --validate data/manifests/all.jsonl` 必须通过

### 3.2 结果目录结构（强制）

每次运行创建唯一目录：`scripts/results/{experiment_name}/{run_id}/`

**必须包含**
- `meta/`
  - `config.yaml`
  - `git_commit.txt`
  - `seed.txt`
  - `env.txt`（pip freeze + cuda）
  - `dataset_manifest_hash.txt`
  - `protocol_version.txt`
- `tables/`
  - `utility_metrics.csv`
  - `attack_metrics.csv`
  - `causal_effects.csv`
  - `security_metrics_cview.csv`
- `figures/`
  - `fig_utility_curve.png`
  - `fig_attack_curves.png`
  - `fig_causal_ate_cate.png`
  - `fig_pareto_frontier.png`
- `logs/`
  - `stdout.log`
  - `errors.log`

**验收标准**
- 任意脚本跑完后都能在对应 `run_id` 下追加表格，不互相覆盖。
- 同一 `run_id` 允许多次 rerun，但必须写入 `meta/rerun_count.txt` 或生成新的 run_id。

---

## 4. 数据集与任务闭环（M1→M2 的前置）

### 4.1 数据集组合（顶刊三角验证）

- **人脸域**：CelebA / CelebA‑HQ（属性分类 + 重建攻击 + 可视化）
- **公平性域**：FairFace（分组一致性：group‑wise utility/privacy）
- **开放域 PII**：OpenImages 子集（person/face/plate 等，至少 person/face）

**外部验证加分项（建议）**
- **FFHQ**：跨数据集泛化（train 在 CelebA/FairFace，test 在 FFHQ/LFW）

### 4.2 任务选择（至少 2 类）

**Task‑1：属性分类（必做，最稳）**
- CelebA：选 10–20 个常用属性（Smiling, Eyeglasses, Male 等）
- FairFace：Gender / Age bucket / Race

指标：
- Acc / Macro‑F1（主）
- Calibration（ECE，可选，加分）

**Task‑2：分割（强烈建议）**
- CelebAMask‑HQ（如果你接得上）：face/skin/hair 等
- 或 person mask（OpenImages/通用分割）

指标：
- mIoU / Dice（主）

**Task‑3：检测（可选，资源允许再做）**
- OpenImages：person/face/plate 的 mAP@0.5

### 4.3 数据规模门槛（避免小样本虚高）

- 每个主数据集 test **≥ 1,000**（建议 ≥ 5,000）
- train/val/test 固定 split，且 manifest 记录 split hash
- 每条曲线点（privacy_level）至少 **≥ 1,000** 张参与评估

---

## 5. 语义 Mask 与 privacy_map：从“能跑”到“可发表”（M1）

### 5.1 Mask 来源双路线（主实验 + 消融）

**A. 强监督/强规则（主实验）**
- 人脸：检测框/landmark → face ellipse mask（可复现）
- 有分割标注：直接用 GT segmentation 合成 sensitive/task/background

**B. 弱监督（消融）**
- 你集成的 U‑Net 显著性/分割器输出（阈值固定）

**验收标准**
- `scripts/gen_semantic_masks.py --manifest ...` 输出：
  - `data/masks/{dataset}/{id}_s.png`
  - `data/masks/{dataset}/{id}_t.png`
  - `data/masks/{dataset}/{id}_b.png`
  - `tables/mask_stats.csv`（区域面积占比直方图）

### 5.2 privacy_map 生成（规则 init + 因果建议）

- 先生成 `privacy_map_init`（三类区域常数预算）
  - 默认示例：sensitive=0.9, task=0.3, background=0.0
- 再调用因果分析模块给出建议（建议输出到 `reports/causal_budget_report.md`）

**验收标准**
- `scripts/gen_privacy_maps.py` 产出：
  - `data/privacy_maps/{id}.npy` 或 `.png`
  - `tables/privacy_map_stats.csv`（均值/方差/区域预算分布）
  - `reports/causal_budget_report.md`（模板化说明）

---

## 6. 双视图生成与评估协议固化（M1→M2）

### 6.1 Z‑view 与 C‑view 的生成规则（写进协议文档）

- **Z‑view（ML 可用）**
  - 保留结构/可学习特征
  - crypto wrap 可弱化/关闭（或使用“可逆但不追求 NIST”的模式）
  - 评估：任务性能 + 攻击成功率 + 视觉/结构指标

- **C‑view（存储/传输强安全）**
  - 使用 AEAD（例如 ChaCha20‑Poly1305）
  - enc_info 必须包含：nonce、tag、版本、AAD 绑定、commitment
  - 评估：熵/NPCR/UACI/相关性/χ²/NIST/完整性校验

### 6.2 导出脚本验收

- `scripts/export_zview_dataset.py`
  - 输入：manifest + privacy_level 列表
  - 输出：z 文件（png/pt）+ `z_index.csv`
- `scripts/export_cview_dataset.py`
  - 输出：c.bin + enc_info.json（nonce/tag/version）
  - 并生成 `c_index.csv`

**验收标准**
- 同一张图，生成 Z 与 C 两份密文；Z 能进训练 DataLoader，C 不进入训练。
- C‑view 安全指标维持你当前水平（可作为门槛）：
  - entropy ≥ 7.95
  - NPCR ≥ 99.5%
  - UACI 在 33% 左右（±2%）
  - χ² p‑value ≥ 0.01（通过“非显著偏离均匀”）
  - NIST 子集测试 pass 率 ≥ 6/7（或你定义的固定集合全过）

---

## 7. Z‑view 可用性：训练策略升级与指标门槛（M2）

### 7.1 训练策略（必须做，否则 Z‑view 很难顶刊站住）

**Baseline‑A（对照）**：train on plaintext → test on Z  
**Baseline‑B（必须）**：train on Z → test on Z  
**Baseline‑C（推荐）**：mixed training（plaintext + Z）→ test on Z（或 domain adaptation）

**验收标准（建议门槛，可按资源调整）**
- 在 test≥1k 的条件下：
  - privacy_level=0.3：主任务性能 ≥ 0.75×plaintext（或绝对 Acc≥75%）
  - privacy_level=0.5：主任务性能 ≥ 0.65×plaintext
- 至少 3 个 seeds，报告 mean±std，std 不能“离谱”（例如 >5% 需解释）

### 7.2 输出文件（必须）

- `tables/utility_metrics.csv` 字段：
  - dataset, task, model, training_mode, privacy_level
  - metric_name, metric_value, seed, n_samples
- `figures/fig_utility_curve.png`
- `figures/fig_groupwise_utility.png`（FairFace 必做：按 race/gender/age 分组）

---

## 8. 三类攻击闭环（M3）

### 8.1 身份验证/识别攻击（Face Verification）

**协议**
- 数据：LFW（只评测不训练），或你定义的外部验证集
- 攻击者：ArcFace/FaceNet embedding + cosine similarity
- 评估：verification accuracy / TAR@FAR=1e‑3（可选）

**产出**
- `tables/attack_metrics.csv`（attack=face_verification）
- `figures/fig_attack_face_verification.png`

**门槛（方向性要求）**
- privacy_level 上升时，验证准确率显著下降
- 需对比 plaintext 与 baseline 方法

### 8.2 属性推断攻击（Sensitive Attribute Inference）

**协议**
- 攻击者在 Z‑view 上训练分类器预测敏感属性（gender/race 等）
- 评估：敏感属性 Acc / AUC
- 同时报告主任务性能，体现 trade‑off

**门槛**
- 敏感属性推断接近随机（例如二分类≈50%，多分类≈1/K，或显著低于 plaintext）
- 主任务性能满足第 7 节门槛

### 8.3 重建攻击（Reconstruction）

**协议**
- 攻击者训练 U‑Net/VAE：Z → plaintext（或 Z → sensitive region）
- 评估：
  - mask 内 PSNR/SSIM（敏感区域）
  - LPIPS（可选）
  - identity similarity drop（embedding similarity）

**门槛**
- 在敏感区域：重建质量显著下降（PSNR/SSIM 显著低于对照）
- 同时主任务性能仍可用（避免“全部毁掉”）

---

## 9. 因果闭环：ATE/CATE + CI + 与启发式对比（M4）

### 9.1 因果问题定义（写进协议）

- Treatment：
  - 方案 1：`T = privacy_level`
  - 方案 2：`T = privacy_map` 的区域预算向量（s,t,b）
- Outcome：
  - `Y_util`：任务性能（Acc/mIoU）
  - `Y_priv`：攻击成功率（verification / inference / recon quality）
- Confounders：
  - 图像难度、光照、姿态、群体属性等（尽量从 labels/group 进入 X）

### 9.2 估计器与统计检验

- ATE：基于 backdoor adjustment 的回归/分层估计
- CATE：按 group / mask area bin 计算
- CI：bootstrap（≥1,000 resamples，或你能承担的规模）
- 显著性：差异的 CI 不跨 0 / permutation test（可选）

### 9.3 必须对比的预算策略（审稿人最爱问）

在相同总隐私强度下，对比：
- causal（你的方法）
- uniform（全图常数）
- sensitive‑only
- task‑only
- heuristic (s=0.9,t=0.3,b=0.0) 作为规则 baseline

**产出**
- `tables/causal_effects.csv`（ATE/CATE + CI）
- `figures/fig_causal_ate_cate.png`
- `reports/causal_report.md`（自动生成：一张图+一句话结论模板）

**门槛（方向性）**
- causal 策略将 Pareto 前沿推向更优（同等 utility 下攻击更低，或同等 attack 下 utility 更高）
- 在 FairFace 分组上，CATE 显示分配策略具备解释性（而非随机）

---

## 10. 强基线对标（M5）

### 10.1 InstaHide（建议先跑小规模通管线）

- 实现方式：复用公开实现或最小自实现（混合多图 + 随机符号 mask）
- 对齐：
  - 同一数据集/任务
  - 同样的 privacy_level 或等效强度刻度（写清映射方式）
  - 同样攻击协议（ArcFace / attribute / recon）

### 10.2 P3（public/private split）

- 实现方式：按频带或像素块划分 public/private
- 对齐：同上

**产出**
- `tables/baseline_comparison.csv`
- `figures/fig_baseline_pareto.png`
- `figures/fig_baseline_attack.png`

**门槛**
- 至少在 1 个数据集 + 1 个任务上，证明你方法优于两基线（或在解释性上明显更强）

---

## 11. 消融矩阵（M6）

必须包含（最少 8 组）：

1) 去掉 Layer1（只用频域/crypto）
2) 去掉 Layer2（只用混沌/crypto）
3) 去掉 crypto wrap（只 Z‑style）
4) 因果预算 → uniform
5) 因果预算 → sensitive‑only
6) mask 来源：强监督 → 弱监督
7) FFT → DWT
8) semantic_preserving 开关：on/off

**产出**
- `tables/ablation.csv`
- `figures/fig_ablation_summary.png`

---

## 12. 论文级图表与表格模板（M7）

### 12.1 主图（4 张）
- Fig‑1：系统框架图（Dual‑View + 三层 + 因果预算）
- Fig‑2：Utility 曲线（mean±std），含至少 2 个任务
- Fig‑3：Attack 曲线（三攻击合并或分面）
- Fig‑4：Causal（ATE/CATE + 解释文本）

### 12.2 补充图（≥2 张）
- S‑1：C‑view 安全统计指标（熵/NPCR/UACI/χ²/NIST）
- S‑2：Ablation 总览（8 组对比）

### 12.3 主表（≥3 张）
- Table‑1：Datasets & tasks & splits（含 license 说明简表）
- Table‑2：Baseline 对比（你 vs InstaHide vs P3）
- Table‑3：Group‑wise fairness（FairFace 分组）

### 12.4 补充表（≥1 张）
- S‑Table：运行成本（FPS/延迟/吞吐/显存），以及攻击训练成本

---

## 13. 质量与测试：顶刊工程“最低不翻车线”

### 13.1 单测覆盖（建议新增 8–12 个 test）

必须覆盖：
- manifest schema validate（字段缺失/路径不存在/哈希不一致）
- deterministic_nonce 与 non‑deterministic 模式切换
- Z/C 导出一致性（同 id 输出两个视图，且可读）
- AAD binding / HMAC 验证（tamper 必须失败）
- attacks 的输入输出维度检查（防 silent bug）
- plotting 输出存在性（csv→png）

### 13.2 代码质量门槛
- `ruff/flake8` 任选其一 + `black`（或你习惯的格式化）
- `pytest -q` 全过
- `python scripts/run_benchmark.py --dry_run` 能检查依赖与路径

---

## 14. 发布与合规（顶刊/开源必须）

- 不上传数据；只提供下载/预处理脚本（DATASETS.md）
- 每个数据集明确许可摘要与用户需要的动作（手动下载/同意条款）
- 结果可复现：给 `reproduce.sh` 或 `Makefile`（一键跑 M1→M7）

---

# 附：你可以直接照抄的“顶刊实验一键命令”模板

```bash
# 1) 准备数据（用户自行下载后）
python scripts/prepare_celeba.py --manual_zip /path/to/celeba.zip
python scripts/prepare_fairface.py --manual_dir /path/to/fairface/
python scripts/prepare_openimages.py --subset pii_v1

# 2) manifest
python scripts/build_manifest.py --datasets celeba fairface openimages_pii --out data/manifests/all.jsonl

# 3) masks + privacy maps
python scripts/gen_semantic_masks.py --manifest data/manifests/all.jsonl
python scripts/gen_privacy_maps.py   --manifest data/manifests/all.jsonl --policy causal

# 4) export views（可选：也可训练时 on-the-fly）
python scripts/export_zview_dataset.py --manifest data/manifests/all.jsonl --levels 0 0.3 0.5 0.7 1.0
python scripts/export_cview_dataset.py --manifest data/manifests/all.jsonl --levels 0 0.3 0.5 0.7 1.0

# 5) run benchmark（单入口）
python scripts/run_benchmark.py --config configs/benchmark.yaml --seeds 0 1 2
```

---

# 附：你下一步最先做的 10 个“可提交 PR 的任务”（按收益排序）

1) 写 `DATASETS.md` + 把 README 的数据复现入口改成“脚本化”
2) 把 `run_benchmark.py` 升级成唯一入口：统一调用 utility/attacks/causal/security
3) 强制结果目录 schema：meta/tables/figures/logs
4) manifest schema 校验模块 + 单测
5) Z/C 生成规则写死到 `EXPERIMENT_PROTOCOL.md`（防止审稿人抓矛盾）
6) train‑on‑Z / mixed training 脚本 + utility 曲线输出
7) LFW face verification 攻击脚本 + 曲线输出
8) attribute inference 攻击脚本 + 曲线输出
9) reconstruction 攻击脚本（U‑Net/VAE）+ mask 内指标输出
10) baseline：InstaHide + P3（先小规模跑通，再扩规模）

---

> 如果你要我继续“把这份计划落到你仓库每个文件该怎么改、函数签名怎么定、csv 字段怎么对齐、每张图怎么画”，我可以在同一份文档里再追加一节：**逐文件改动点清单（按仓库路径）**，做到你照着敲就能完成 M1→M7。
