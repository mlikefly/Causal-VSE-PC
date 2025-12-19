# 顶刊实验执行路线图

> 生成时间: 2024-12-18
> 版本: v1.0
> 状态: 准备执行

## 一、项目状态总结

### 1.1 Spec 状态

| Spec | 状态 | 说明 |
|------|------|------|
| top-journal-experiment-suite | ✅ 完成 | 当前主版本 v2.1.1，T0-T13 全部完成 |

### 1.2 代码实现完整性

已实现的核心模块（24个测试文件覆盖）：

- ✅ `src/protocol/` - ProtocolManager, ResultsSchema, ValidateRun
- ✅ `src/core/` - NonceManager, ReplayCache, 加密核心
- ✅ `src/cipher/` - DualViewEncryptionEngine
- ✅ `src/data/` - ManifestBuilder, SemanticMaskGenerator, CausalBudgetAllocator
- ✅ `src/evaluation/` - 攻击评估、安全指标、统计引擎、图表生成（17个模块）
- ✅ `src/training/` - TrainingModeManager

### 1.3 模块关系说明

| 模块对 | 关系 | 说明 |
|--------|------|------|
| attack_evaluator.py / attack_framework.py | 演进 | framework 是新标准，evaluator 提供具体实现 |
| baseline_comparator.py / baseline_matrix.py | 演进 | matrix 是升级版，支持5基线+N/A规则 |

---

## 二、执行计划

### 阶段 1：验证与清理 (1-2天)

#### 1.1 运行完整测试套件
```bash
# 运行所有测试
pytest tests/ -v --tb=short

# 运行特定模块测试
pytest tests/test_protocol_manager.py -v
pytest tests/test_attack_framework.py -v
pytest tests/test_baseline_comparator.py -v
```

#### 1.2 运行 Smoke Test
```bash
python scripts/run_benchmark.py --smoke_test
```

#### 1.3 验证红线检查 (R1-R10)
```python
from src.protocol import ValidateRun
validator = ValidateRun(run_dir="results/test_run")
validator.validate_all()
```

### 阶段 2：数据准备 (2-3天)

#### 2.1 构建 Manifest
```bash
python scripts/build_manifest.py --data_root data/ --output data/manifest.jsonl
```

#### 2.2 生成语义掩码
```bash
python scripts/gen_semantic_masks.py --manifest data/manifest.jsonl
```

#### 2.3 生成隐私预算图
```bash
python scripts/gen_privacy_maps.py --manifest data/manifest.jsonl
```

### 阶段 3：核心实验 (1-2周)

#### 3.1 导出 Z-view 数据集
```bash
python scripts/export_zview_dataset.py --privacy_levels 0.0,0.3,0.5,0.7,1.0
```

#### 3.2 运行完整 Benchmark
```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

#### 3.3 验证覆盖度
- 目标: ≥98% 覆盖度
- 检查: attack_metrics.csv 包含 A2 威胁等级
- 检查: baseline_comparison.csv 包含 InstaHide + P3

### 阶段 4：论文输出 (3-5天)

#### 4.1 生成主图 (8张)
```python
from src.evaluation import FigureGenerator
fg = FigureGenerator(run_dir="results/full_run")
fg.generate_all_figures()
```

#### 4.2 生成 LaTeX 表格
```python
from src.evaluation import StatisticsEngine
se = StatisticsEngine(run_dir="results/full_run")
se.generate_latex_tables()
```

#### 4.3 验证 figure_manifest.json
```bash
python -c "from src.evaluation import validate_figure_manifest; validate_figure_manifest('results/full_run')"
```

---

## 三、关键检查点

### 红线检查 (R1-R10)

| ID | 检查项 | 阈值 |
|----|--------|------|
| R1 | 攻击覆盖度 | ≥98% |
| R2 | A2 威胁等级存在 | 必须 |
| R3 | 基线覆盖 (InstaHide+P3) | 必须 |
| R4 | 统计显著性 | p<0.05 |
| R5 | CI 宽度 | <0.1 |
| R6 | 效用保持 | >0.8 |
| R7 | 隐私保护 | >0.7 |
| R8 | 因果效应显著 | ATE>0 |
| R9 | 消融完整性 | 12项 |
| R10 | 图表完整性 | 8张 |

### 输出文件清单

```
results/{run_id}/
├── tables/
│   ├── attack_metrics.csv
│   ├── baseline_comparison.csv
│   ├── utility_metrics.csv
│   └── ablation_results.csv
├── figures/
│   ├── fig1_privacy_utility_tradeoff.pdf
│   ├── fig2_attack_success_heatmap.pdf
│   ├── fig3_causal_effects.pdf
│   ├── fig4_baseline_comparison.pdf
│   ├── fig5_ablation_study.pdf
│   ├── fig6_robustness_analysis.pdf
│   ├── fig7_efficiency_comparison.pdf
│   └── fig8_pareto_frontier.pdf
├── reports/
│   ├── baseline_matrix_report.md
│   ├── statistical_analysis.md
│   └── validation_report.md
└── figure_manifest.json
```

---

## 四、故障排除

### 常见问题

1. **测试失败**: 检查 PyTorch 版本和 CUDA 可用性
2. **内存不足**: 减少 batch_size 或使用 --smoke_test
3. **覆盖度不足**: 检查数据集完整性和 manifest 配置
4. **A2 缺失**: 确保 attack_framework 正确配置威胁等级

### 联系方式

如有问题，请参考：
- `.kiro/specs/top-journal-experiment-suite/design.md` - 设计文档
- `.kiro/specs/top-journal-experiment-suite/tasks.md` - 任务清单
- `docs/implementation_plan.md` - 实现计划

---

*本文档由项目分析自动生成*
