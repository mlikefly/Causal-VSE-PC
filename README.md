# Causal-VSE-PC

**Causal Visual Semantic Encryption with Privacy Control**

[![Protocol Version](https://img.shields.io/badge/Protocol-v2.1.1-blue.svg)](docs/project_overview.md)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Causal-VSE-PC is a privacy-preserving image encryption system designed for top-tier journal publication (T-IFS/TIP/TNNLS). The system implements a **dual-view architecture** with **causal privacy budget allocation**.

### Core Contributions

| # | Contribution | Evidence |
|---|--------------|----------|
| C1 | **Causal Privacy Budget Allocation** - ATE/CATE-guided semantic region budget optimization | Pareto curves + causal effects |
| C2 | **Dual-View Architecture** - Z-view (utility) + C-view (crypto) separation with A0/A1/A2 threat levels | Attack curves + worst-case aggregation |
| C3 | **Comprehensive Attack Evaluation** - 5 attack types + A2 adaptive attacks | Attack metrics + statistical significance |
| C4 | **Auditable AEAD Security** - Confidentiality/Integrity/Replay resistance | Security validation + diagnostics |
| C5 | **Reproducible Protocol** - Frozen protocol, coverage verification, byte-level figure reproduction | Artifact checklist + CI results |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Causal-VSE-PC Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image → Semantic Mask → Causal Budget → Dual-View Encrypt │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Z-view    │    │   C-view    │    │  Evaluation │          │
│  │ (Utility)   │    │  (Crypto)   │    │  (5 Attacks)│          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                  │
│  Training Modes: P2P / P2Z / Z2Z / Mix2Z                        │
│  Threat Levels: A0 (Black-box) / A1 (Gray-box) / A2 (Adaptive)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mlikefly/Causal-VSE-PC.git
cd Causal-VSE-PC

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.cipher.dual_view_engine import DualViewEngine

# Initialize encryption engine
engine = DualViewEngine(master_key=your_key)

# Encrypt image with dual views
z_view, c_view, enc_info = engine.encrypt(
    image,
    privacy_level=0.5
)

# Z-view: Use for ML inference (preserves semantics)
# C-view: Use for secure storage (AEAD wrapped)
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run smoke test (< 20 min)
python scripts/run_benchmark.py --smoke_test

# Run full experiments
python scripts/run_benchmark.py --full
```

---

## 📊 Metrics

### Security Metrics

| Metric | Target | Current |
|--------|--------|---------|
| NPCR | > 99.6% | ✅ 99.57% |
| UACI | 30-36% | ✅ 33.49% |
| Entropy | > 7.9 bits | ✅ 7.99 |
| Tamper fail_rate | ≥ 99% | ✅ Implemented |
| Replay reject_rate | = 100% | ✅ Implemented |

### Utility Thresholds

| Privacy Level | Threshold |
|---------------|-----------|
| λ = 0.3 | ≥ 75% P2P |
| λ = 0.5 | ≥ 65% P2P |
| λ = 0.7 | ≥ 55% P2P |

---

## 🔒 Security Boundary

```
Security Boundary Declaration:
1. C-view security inherits from standard AEAD (AES-GCM/ChaCha20-Poly1305), 
   providing IND-CPA and IND-CCA guarantees.
2. Chaotic/frequency domain transformations serve as confusion/diffusion 
   layers and do NOT independently claim semantic security.
3. Z-view privacy is empirically demonstrated through attack success rate 
   reduction, not through cryptographic proofs.
4. This system does not defend against: side-channel attacks, physical 
   attacks, or attacks with access to the encryption key.
```

---

## 📁 Project Structure

```
Causal-VSE-PC/
├── src/
│   ├── cipher/           # Encryption engines
│   ├── core/             # Core algorithms (chaos, frequency)
│   ├── crypto/           # Cryptographic components
│   ├── data/             # Data pipeline
│   ├── protocol/         # Protocol & validation
│   ├── training/         # Training modes
│   └── evaluation/       # Evaluation framework
├── tests/                # Unit tests
├── scripts/              # Utility scripts
├── configs/              # Configuration files
├── docs/                 # Documentation
└── .kiro/specs/          # Design specifications
```

---

## 📚 Documentation

- [Project Overview](docs/project_overview.md)
- [Workflow](docs/workflow.md)
- [Goals & Metrics](docs/goals_and_metrics.md)
- [Development Log](docs/development_log.md)
- [Source Code Guide](src/README.md)
- [Design Document](.kiro/specs/top-journal-experiment-suite/design.md)

---

## 🧪 Attack Evaluation

### 5 Attack Types

| Attack | Metric | Direction |
|--------|--------|-----------|
| Face Verification | TAR@FAR=1e-3 | ↓ lower is better |
| Attribute Inference | AUC | ↓ lower is better |
| Reconstruction | identity_similarity | ↓ lower is better |
| Membership Inference | AUC | ↓ lower is better |
| Property Inference | AUC | ↓ lower is better |

### Threat Levels

| Level | Knowledge | Capability |
|-------|-----------|------------|
| A0 | Z-view output only | Output-based inference |
| A1 | Algorithm + architecture | Targeted attack models |
| A2 | Mask + budget allocation | Adaptive attack strategies |

---

## 📈 Outputs

### 8 Main Figures

1. `fig_utility_curve.png` - Utility vs privacy_level
2. `fig_attack_curves.png` - 5 attack curves + CI
3. `fig_pareto_frontier.png` - Privacy-utility Pareto frontier
4. `fig_causal_ate_cate.png` - ATE/CATE + CI
5. `fig_cview_security_summary.png` - C-view security summary
6. `fig_ablation_summary.png` - Ablation comparison
7. `fig_efficiency.png` - Efficiency comparison
8. `fig_robustness.png` - Robustness results

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

- **mlikefly** - [GitHub](https://github.com/mlikefly)
- Email: 1392792307@qq.com

---

## 🙏 Acknowledgments

- CelebA-HQ Dataset
- FairFace Dataset
- OpenImages Dataset
