# Causal-VSE-PC: 因果推断驱动的可验证语义加密

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于因果推断的隐私保护图像加密框架，支持语义感知的差异化加密和密文域机器学习。

## 项目概述

Causal-VSE-PC (Causal Verifiable Semantic Encryption with Privacy Control) 是一个创新的图像加密框架，核心特点：

- **因果推断驱动**：使用ATE/CATE分析隐私-效用权衡，自动优化隐私预算分配
- **语义感知加密**：基于U-Net的显著性检测，对敏感区域实施差异化加密
- **三层加密架构**：
  - Layer 1: 空域混沌置乱（Arnold + 5D超混沌）
  - Layer 2: 频域语义控制（FFT/DWT分层扰动）
  - Layer 3: 字节级流加密（ChaCha20）
- **密文域ML**：加密图像可直接用于机器学习推理

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8+ 推荐)
- 8GB+ GPU显存（推荐）

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/Causal-VSE-PC.git
cd Causal-VSE-PC

# 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

项目使用 CelebA-HQ 数据集进行测试：

```bash
# 下载数据集到 data/ 目录
# 目录结构：
# data/
# ├── CelebA-HQ/           # 图像文件
# │   ├── train/
# │   └── test/
# └── CelebA-HQ-labels/    # 标签文件（可选）
```

### 基础使用

```python
from src.cipher.scne_cipher import SCNECipherAPI
import torch

# 初始化加密器
cipher = SCNECipherAPI(
    password="your_password",
    image_size=256,
    device="cuda"
)

# 加载图像 [B, 1, H, W]，值域 [0, 1]
image = torch.rand(1, 1, 256, 256).cuda()

# 加密
encrypted, enc_info = cipher.encrypt_simple(
    image,
    privacy_level=0.7,  # 隐私级别 [0, 1]
    semantic_preserving=False
)

# 解密
mask = torch.ones_like(image)
decrypted = cipher.cipher.decrypt(encrypted, enc_info, mask, password="your_password")
```

### 运行完整评测

```bash
# 端到端测试（含安全评估）
python scripts/experiments/vse_pc/test_causal_e2e_full.py

# 解密验证测试
python scripts/experiments/vse_pc/test_decrypt_layers.py

# 确定性验证
python scripts/experiments/vse_pc/test_deterministic.py
```

## 项目结构

```
Causal-VSE-PC/
├── src/                          # 源代码
│   ├── cipher/                   # 加密器
│   │   └── scne_cipher.py        # SCNE主加密器
│   ├── core/                     # 核心算法
│   │   ├── chaotic_encryptor.py  # 混沌加密（Layer 1）
│   │   ├── frequency_cipher.py   # 频域加密（Layer 2）
│   │   └── chaos_systems.py      # 混沌系统
│   ├── crypto/                   # 密码学组件
│   │   └── key_system.py         # 分层密钥系统
│   ├── vse_pc/                   # 因果推断模块
│   │   ├── causal_analysis.py    # ATE/CATE分析
│   │   ├── privacy_budget.py     # 隐私预算分配
│   │   └── pipeline.py           # 完整流水线
│   ├── neural/                   # 神经网络
│   │   └── unet.py               # U-Net显著性检测
│   ├── evaluation/               # 评估模块
│   │   └── security_metrics.py   # 安全指标
│   └── utils/                    # 工具函数
│       └── datasets.py           # 数据加载
├── scripts/                      # 脚本
│   ├── experiments/vse_pc/       # 实验脚本
│   └── evaluation/               # 评估脚本
├── configs/                      # 配置文件
├── data/                         # 数据目录
├── docs/                         # 文档
└── results/                      # 结果输出
```

## 配置说明

配置文件位于 `configs/` 目录：

```yaml
# configs/default.yaml
encryption:
  use_frequency: true      # 启用频域加密
  use_fft: true            # 使用FFT（比DWT快）
  enable_crypto_wrap: true # 启用字节级加密
  
privacy:
  default_level: 0.7       # 默认隐私级别
  
data:
  image_size: 256
  batch_size: 32
```

## 安全指标

项目评估以下安全指标：

| 指标 | 标准 | 说明 |
|------|------|------|
| 信息熵 | ≥ 7.9 bits | 加密图像随机性 |
| NPCR | ≥ 99.5% | 像素变化率 |
| UACI | 30-36% | 平均强度变化 |
| 相关性 | \|r\| < 0.1 | 相邻像素相关性 |
| Chi-square | p > 0.05 | 直方图均匀性 |

## 主要API

### SCNECipherAPI

```python
class SCNECipherAPI:
    def __init__(self, password, image_size=256, device=None, 
                 use_frequency=True, use_fft=True, enable_crypto_wrap=True)
    
    def encrypt_simple(self, image, privacy_level=1.0, 
                       semantic_preserving=False, mask=None)
    
    def cipher.decrypt(self, encrypted, enc_info, mask, password)
```

### CausalPrivacyAnalyzer

```python
class CausalPrivacyAnalyzer:
    def analyze_allocation(self, semantic_mask, task_type, privacy_map)
    
    def compute_causal_effects(self, semantic_mask, privacy_map,
                               performance_encrypted, performance_original,
                               task_type, conf_interval=True)
```

## 文档

详细文档位于 `docs/` 目录：

- [项目总览](docs/Causal-VSE-PC_项目总览.md)
- [工作流程](docs/Causal-VSE-PC_工作流程.md)
- [理论证明](docs/Causal-VSE-PC_理论证明.md)
- [数据集说明](docs/Causal-VSE-PC_数据集分析与使用.md)

## 开发日志

- [项目开发日志](docs/项目开发日志.md)
- [问题诊断与解决方案](docs/问题诊断与解决方案_20251211.md)

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{causal-vse-pc-2024,
  title={Causal-VSE-PC: Causal Inference Driven Verifiable Semantic Encryption with Privacy Control},
  author={...},
  journal={IEEE Access},
  year={2024}
}
```

## License

MIT License
