# 脚本说明

## 目录结构

```
scripts/
├── experiments/              # 实验脚本
│   └── vse_pc/               # Causal-VSE-PC实验
│       ├── test_causal_e2e_full.py   # 完整端到端测试（主要）
│       ├── test_decrypt_layers.py    # 分层解密测试
│       ├── test_deterministic.py     # 确定性验证
│       └── ...
│
├── evaluation/               # 评估脚本
│   ├── security.py           # 安全性评估
│   ├── benchmark.py          # 性能基准测试
│   └── attacks.py            # 攻击测试
│
└── training/                 # 训练脚本
```

## 主要脚本

### 完整端到端测试

```bash
python scripts/experiments/vse_pc/test_causal_e2e_full.py
```

功能：
- 加载CelebA-HQ数据
- 语义分析与因果推断
- 多隐私级别加密测试
- 安全指标评估
- 解密验证
- 生成可视化报告

### 分层解密测试

```bash
python scripts/experiments/vse_pc/test_decrypt_layers.py
```

功能：
- 单独测试Layer 1（混沌加密）
- 单独测试Layer 2（频域加密）
- 测试组合加密
- 验证各层可逆性

### 确定性验证

```bash
python scripts/experiments/vse_pc/test_deterministic.py
```

功能：
- 验证相同密码产生相同加密结果
- 验证加密-解密可逆性
- 验证不同隐私级别的MAE差异

## 输出目录

结果保存在 `scripts/results/` 目录：
- `causal_analysis_full/` - 完整测试结果
  - `encryption_comparison.png` - 加密效果对比图
  - `histogram_analysis.png` - 直方图分析
  - `causal_effects.png` - 因果效应图
  - `security_report.md` - 安全评估报告
