"""
Pytest 配置文件 - Hypothesis 属性测试配置

配置 Hypothesis 库进行属性测试，每个属性测试运行至少 100 次迭代。
"""

import pytest
import torch
import numpy as np
import random
import os
import sys

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hypothesis 配置
from hypothesis import settings, Verbosity, Phase

# 默认配置：100次迭代
settings.register_profile(
    "default",
    max_examples=100,
    verbosity=Verbosity.normal,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    deadline=None,  # 禁用超时（加密操作可能较慢）
)

# CI 配置：50次迭代（更快）
settings.register_profile(
    "ci",
    max_examples=50,
    verbosity=Verbosity.normal,
    deadline=None,
)

# 开发配置：10次迭代（快速验证）
settings.register_profile(
    "dev",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
)

# 详细配置：200次迭代（更彻底）
settings.register_profile(
    "thorough",
    max_examples=200,
    verbosity=Verbosity.normal,
    deadline=None,
)

# 根据环境变量选择配置
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


# ============ 共享 Fixtures ============

@pytest.fixture(scope="session")
def device():
    """返回可用的计算设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="session")
def test_password():
    """测试用密码"""
    return "test_password_2025"


@pytest.fixture
def random_seed():
    """固定随机种子以确保可复现性"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def small_image(device):
    """小尺寸测试图像 (64x64)"""
    return torch.rand(1, 1, 64, 64, device=device)


@pytest.fixture
def standard_image(device):
    """标准尺寸测试图像 (256x256)"""
    return torch.rand(1, 1, 256, 256, device=device)


@pytest.fixture
def batch_images(device):
    """批量测试图像 (4x1x256x256)"""
    return torch.rand(4, 1, 256, 256, device=device)


@pytest.fixture
def uniform_mask(device):
    """均匀掩码"""
    return torch.ones(1, 1, 256, 256, device=device)


@pytest.fixture
def random_mask(device):
    """随机掩码"""
    return torch.rand(1, 1, 256, 256, device=device)


@pytest.fixture
def semantic_mask(device):
    """三值语义掩码"""
    mask = torch.zeros(1, 1, 256, 256, device=device)
    # 敏感区域 (中心)
    mask[:, :, 64:192, 64:192] = 1.0
    # 任务区域 (边缘)
    mask[:, :, 32:64, :] = 0.5
    mask[:, :, 192:224, :] = 0.5
    return mask


@pytest.fixture
def default_chaos_params(device):
    """默认混沌参数"""
    params = torch.rand(1, 19, 3, device=device)
    scale = torch.tensor([10.0, 5.0, 1.0], device=device)
    return params * scale


@pytest.fixture(scope="session")
def scne_api(test_password, device):
    """SCNE API 实例（会话级别，避免重复初始化）"""
    from src.cipher.scne_cipher import SCNECipherAPI
    return SCNECipherAPI(
        password=test_password,
        image_size=256,
        deterministic=True,
        device=device
    )


@pytest.fixture(scope="session")
def scne_cipher(test_password, device):
    """SCNE Cipher 实例"""
    from src.cipher.scne_cipher import SCNECipher
    cipher = SCNECipher(
        image_size=256,
        use_frequency=True,
        use_fft=True,
        password=test_password,
        enable_crypto_wrap=True,
        deterministic=True
    )
    return cipher.to(device)


@pytest.fixture(scope="session")
def key_system(test_password):
    """密钥系统实例"""
    from src.crypto.key_system import HierarchicalKeySystem
    return HierarchicalKeySystem(test_password, iterations=10000)


# ============ Hypothesis 自定义策略 ============

def privacy_levels():
    """隐私级别策略"""
    from hypothesis import strategies as st
    return st.sampled_from([0.0, 0.3, 0.5, 0.7, 1.0])


def task_types():
    """任务类型策略"""
    from hypothesis import strategies as st
    return st.sampled_from(['classification', 'detection', 'segmentation'])


def datasets():
    """数据集策略"""
    from hypothesis import strategies as st
    return st.sampled_from(['celeba', 'celebahq', 'fairface', 'openimages'])


def splits():
    """数据划分策略"""
    from hypothesis import strategies as st
    return st.sampled_from(['train', 'val', 'test'])


def image_ids():
    """图像ID策略"""
    from hypothesis import strategies as st
    return st.text(
        alphabet='abcdefghijklmnopqrstuvwxyz0123456789_',
        min_size=8,
        max_size=32
    )


def small_images():
    """小图像策略（用于快速测试）"""
    from hypothesis import strategies as st
    return st.builds(
        lambda: torch.rand(1, 1, 64, 64),
    )


def mask_values():
    """掩码值策略（三值）"""
    from hypothesis import strategies as st
    return st.sampled_from([0.0, 0.5, 1.0])
