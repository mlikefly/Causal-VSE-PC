# -*- coding: utf-8 -*-
"""
实验配置记录器 (Experiment Tracker)

自动记录实验配置、数据版本、代码 commit hash、模型权重哈希等

Requirements: 9.3, 9.4, 9.5
"""

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import yaml


class ExperimentTracker:
    """
    实验配置记录器
    
    功能：
    1. 记录实验配置
    2. 记录数据版本
    3. 记录代码 commit hash
    4. 记录模型权重 SHA256 哈希
    5. 固定随机种子
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "results/experiments",
        config_path: str = None
    ):
        """
        初始化实验记录器
        
        Args:
            experiment_name: 实验名称
            output_dir: 输出目录
            config_path: 配置文件路径
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'git_branch': self._get_git_branch(),
            'python_version': self._get_python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        return self.config
    
    def set_seed(self, seed: int = None):
        """
        固定随机种子
        
        **Requirements 9.3**: PyTorch/NumPy/Python random 统一固定
        
        Args:
            seed: 随机种子
        """
        if seed is None:
            seed = self.config.get('reproducibility', {}).get('seed', 42)
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # CUDNN 确定性
        if self.config.get('reproducibility', {}).get('deterministic_cudnn', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.metadata['seed'] = seed
        print(f"随机种子已固定: {seed}")
    
    def record_data_version(self, data_paths: List[str]) -> Dict[str, str]:
        """
        记录数据版本（使用文件哈希）
        
        **Requirements 9.4**: 自动记录数据版本
        
        Args:
            data_paths: 数据文件/目录路径列表
        
        Returns:
            数据版本哈希字典
        """
        data_versions = {}
        
        for path in data_paths:
            path = Path(path)
            if path.exists():
                if path.is_file():
                    hash_val = self._compute_file_hash(path)
                else:
                    hash_val = self._compute_dir_hash(path)
                data_versions[str(path)] = hash_val
        
        self.metadata['data_versions'] = data_versions
        return data_versions
    
    def record_model_weights(self, model_paths: List[str]) -> Dict[str, str]:
        """
        记录模型权重 SHA256 哈希
        
        **Requirements 9.5**: 记录模型权重 SHA256 哈希
        
        Args:
            model_paths: 模型权重文件路径列表
        
        Returns:
            模型权重哈希字典
        """
        weight_hashes = {}
        
        for path in model_paths:
            path = Path(path)
            if path.exists():
                hash_val = self._compute_file_hash(path)
                weight_hashes[str(path)] = hash_val
        
        self.metadata['model_weights'] = weight_hashes
        return weight_hashes
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件 SHA256 哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    def _compute_dir_hash(self, dir_path: Path, max_files: int = 100) -> str:
        """计算目录哈希（基于文件列表和大小）"""
        files = sorted(dir_path.rglob('*'))[:max_files]
        content = ""
        for f in files:
            if f.is_file():
                content += f"{f.name}:{f.stat().st_size};"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_git_commit(self) -> Optional[str]:
        """获取 Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None
    
    def _get_git_branch(self) -> Optional[str]:
        """获取 Git 分支名"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_python_version(self) -> str:
        """获取 Python 版本"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def save_experiment(self, results: Dict = None):
        """
        保存实验记录
        
        Args:
            results: 实验结果
        """
        record = {
            'metadata': self.metadata,
            'config': self.config,
            'results': results or {}
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"实验记录已保存: {output_path}")
        return output_path
    
    def print_summary(self):
        """打印实验配置摘要"""
        print("\n" + "=" * 70)
        print("实验配置摘要")
        print("=" * 70)
        
        print(f"\n实验名称: {self.experiment_name}")
        print(f"时间戳: {self.metadata['timestamp']}")
        print(f"Git Commit: {self.metadata.get('git_commit', 'N/A')}")
        print(f"Git Branch: {self.metadata.get('git_branch', 'N/A')}")
        print(f"PyTorch: {self.metadata['pytorch_version']}")
        print(f"CUDA: {self.metadata.get('cuda_version', 'N/A')}")
        print(f"随机种子: {self.metadata.get('seed', 'Not set')}")
        
        if 'data_versions' in self.metadata:
            print(f"\n数据版本:")
            for path, hash_val in self.metadata['data_versions'].items():
                print(f"  {path}: {hash_val}")
        
        if 'model_weights' in self.metadata:
            print(f"\n模型权重:")
            for path, hash_val in self.metadata['model_weights'].items():
                print(f"  {path}: {hash_val}")
        
        print("=" * 70)


def set_global_seed(seed: int = 42):
    """
    全局设置随机种子
    
    **Requirements 9.3**: PyTorch/NumPy/Python random 统一固定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"全局随机种子已设置: {seed}")
