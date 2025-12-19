# 文档整理说明

**整理日期**: 2024-12-18  
**最后更新**: 2024-12-18  
**整理原因**: 项目迭代过程中产生了多个版本的文档，需要清理以避免混淆

---

## 一、Kiro Specs 整理

### 当前有效版本
- **`.kiro/specs/top-journal-experiment-suite/`** - v2.1.1 (当前版本)
  - `requirements.md` - 16个需求 + 10个全局约束
  - `design.md` - 完整设计文档
  - `tasks.md` - T0-T13 任务清单

### 已清理版本 ✅
- **`.kiro/specs/top-tier-journal-upgrade/`** - 已删除 (2024-12-18)
  - 内容已整合到 `top-journal-experiment-suite`
  - Phase 0-5 的任务已完成并迁移

---

## 二、实验计划文档整理

### 已归档版本 ✅
以下文件已移至 `docs/archive/` (2024-12-18)：

1. `docs/archive/Causal-VSE-PC_top_journal_full_experiment_plan_v1.md` - V1
2. `docs/archive/Causal-VSE-PC_top_journal_full_experiment_plan_v2.md` - V2
3. `docs/archive/Causal-VSE-PC_top_journal_full_experiment_plan_v3.md` - V3

### 当前有效文档
- `docs/project_overview.md` - 项目总览
- `docs/development_log.md` - 开发日志
- `docs/workflow.md` - 工作流程
- `docs/goals_and_metrics.md` - 目标与指标
- `docs/data_flow.md` - 数据流向
- `docs/dataset_analysis.md` - 数据集分析
- `docs/theoretical_proof.md` - 理论证明
- `docs/implementation_plan.md` - 实现计划
- `docs/literature_review_2015_2025.md` - 文献综述

---

## 三、清理操作记录

### 2024-12-18 清理操作 ✅
```bash
# 已执行：移动历史计划文档到归档目录
mv docs/Causal-VSE-PC_top_journal_full_experiment_plan.md docs/archive/Causal-VSE-PC_top_journal_full_experiment_plan_v1.md
mv docs/Causal-VSE-PC_top_journal_full_experiment_plan_v2.md docs/archive/
mv docs/Causal-VSE-PC_top_journal_full_experiment_plan_v3.md docs/archive/

# 已执行：删除过时的 spec
rm -rf .kiro/specs/top-tier-journal-upgrade/
```

---

## 四、文档版本对应关系

| 历史文档 | 对应的当前内容 | 状态 |
|----------|----------------|------|
| V1 计划 | `tasks.md` T0-T5 | ✅ 已归档 |
| V2 计划 | `tasks.md` T6-T11 + `requirements.md` GC1-GC10 | ✅ 已归档 |
| V3 计划 | `design.md` 完整设计 + `tasks.md` T12-T13 | ✅ 已归档 |
| `top-tier-journal-upgrade/` | `top-journal-experiment-suite/` Phase 0-5 | ✅ 已删除 |

---

## 五、当前项目状态

- **协议版本**: 2.1.1
- **核心实现**: 100% 完成
- **Spec 状态**: 仅保留 `top-journal-experiment-suite`
- **下一步**: 测试验证 → 数据准备 → 完整实验

---

## 六、代码模块关系分析

### 2024-12-18 分析结果

| 模块对 | 关系 | 说明 |
|--------|------|------|
| `attack_evaluator.py` / `attack_framework.py` | 演进 | framework 是新标准框架，evaluator 提供具体实现 |
| `baseline_comparator.py` / `baseline_matrix.py` | 演进 | matrix 是升级版，支持5基线+N/A规则 |

**结论**: 这些模块不是重复，而是合理的层次结构，无需删除。

---

## 七、相关文档

- **执行路线图**: `docs/EXECUTION_ROADMAP.md` - 完整的执行计划和检查点
- **设计文档**: `.kiro/specs/top-journal-experiment-suite/design.md`
- **任务清单**: `.kiro/specs/top-journal-experiment-suite/tasks.md`

---

*整理人: Kiro*
