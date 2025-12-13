#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混沌系统 - 完整实现
包含高维、中维、低维混沌系统
"""

import numpy as np
try:
    from scipy.integrate import solve_ivp  # 可选依赖，默认不使用
    _SCIPY_AVAILABLE = True
except Exception:
    solve_ivp = None
    _SCIPY_AVAILABLE = False


class ChaosSystem:
    """混沌系统基类"""
    
    def __init__(self, complexity: str = 'high', use_scipy: bool = False):
        self.complexity = complexity
        self.state_dim = self._get_state_dim()
        self.params = self._init_params()
        # 默认禁用SciPy积分，使用GPU/向量化Logistic近似加速
        self.use_scipy = bool(use_scipy and _SCIPY_AVAILABLE)
    
    def _get_state_dim(self) -> int:
        if self.complexity == 'high':
            return 5  # 5维超混沌
        elif self.complexity == 'medium':
            return 3  # 3维Lorenz
        else:
            return 1  # 1维Logistic
    
    def _init_params(self) -> dict:
        if self.complexity == 'high':
            # 5维Chen-Logistic超混沌系统
            return {
                'a': 35.0,
                'b': 3.0,
                'c': 28.0,
                'mu': 3.99,
                'alpha': 0.5,
                'beta': 0.8,
                'gamma': 1.2,
                'eta': 0.5
            }
        elif self.complexity == 'medium':
            # 3维Lorenz系统
            return {
                'sigma': 10.0,
                'rho': 28.0,
                'beta': 8.0 / 3.0
            }
        else:
            # 1维Logistic映射
            return {
                'r': 3.99
            }
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """混沌系统方程"""
        if self.complexity == 'high':
            return self._high_dim_equations(t, state)
        elif self.complexity == 'medium':
            return self._medium_dim_equations(t, state)
        else:
            return self._low_dim_equations(t, state)
    
    def _high_dim_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """5维超混沌系统"""
        x, y, z, u, w = state
        p = self.params
        
        dx = p['a'] * (y - x) + p['alpha'] * w
        dy = (p['c'] - p['a']) * x - x * z + p['c'] * y
        dz = x * y - p['b'] * z
        
        # Logistic映射部分
        u_clip = np.clip(u, 0.0, 1.0)
        du = (p['mu'] * u_clip * (1 - u_clip) + 
              p['gamma'] * np.sin(y / 8.0) + 
              p['eta'] * (0.5 - u_clip))
        
        # 耦合项
        dw = p['beta'] * (x + y - 4.5 * w)
        
        return np.array([dx, dy, dz, du, dw])
    
    def _medium_dim_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """3维Lorenz系统"""
        x, y, z = state
        p = self.params
        
        dx = p['sigma'] * (y - x)
        dy = x * (p['rho'] - z) - y
        dz = x * y - p['beta'] * z
        
        return np.array([dx, dy, dz])
    
    def _low_dim_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """1维Logistic映射（迭代形式）"""
        return np.array([self.params['r'] * state[0] * (1 - state[0])])
    
    def generate_trajectory(self, initial_state: np.ndarray, length: int) -> np.ndarray:
        """
        生成混沌轨迹
        
        Args:
            initial_state: 初始状态
            length: 轨迹长度
        
        Returns:
            trajectory: (length, state_dim)
        """
        # 统一使用向量化Logistic近似（GPU/CPU均可快速运行），确保确定性与可逆扩散使用
        # 对于 high/medium 维度，按维度独立Logistic迭代并做轻度耦合混合
        state_dim = self.state_dim
        traj = np.zeros((length, state_dim), dtype=np.float64)
        # 将初值映射到(0,1)以适配Logistic映射
        x = 1.0 / (1.0 + np.exp(-initial_state.astype(np.float64)))
        # 不同维度设置不同r以增加混沌性
        if self.complexity == 'low':
            r_vec = np.array([self.params.get('r', 3.99)], dtype=np.float64)
        elif self.complexity == 'medium':
            r_vec = np.array([3.89, 3.93, 3.99], dtype=np.float64)
        else:
            r_vec = np.array([3.91, 3.93, 3.95, 3.97, 3.99], dtype=np.float64)[:state_dim]
        for i in range(length):
            traj[i] = x
            # Logistic更新 + 轻度跨维混合
            x = r_vec * x * (1.0 - x)
            # 简单线性混合，避免各维完全独立
            x = 0.85 * x + 0.15 * np.roll(x, 1)
            # 防止数值扎堆到边界
            x = np.clip(x, 1e-9, 1 - 1e-9)
        return traj
    
    def derive_initial_state(self, seed: bytes) -> np.ndarray:
        """从种子派生初始状态"""
        import hashlib
        
        hash_val = hashlib.sha3_512(seed).digest()
        seed_int = int.from_bytes(hash_val, 'big')
        
        if self.complexity == 'high':
            # 5维初始状态
            return np.array([
                ((seed_int >> 0) & 0xFFFFFFFF) / 2**32 * 20 - 10,
                ((seed_int >> 32) & 0xFFFFFFFF) / 2**32 * 20 - 10,
                ((seed_int >> 64) & 0xFFFFFFFF) / 2**32 * 30,
                ((seed_int >> 96) & 0xFFFFFFFF) / 2**32,
                ((seed_int >> 128) & 0xFFFFFFFF) / 2**32 * 10 - 5,
            ], dtype=np.float64)
        elif self.complexity == 'medium':
            # 3维初始状态
            return np.array([
                ((seed_int >> 0) & 0xFFFFFFFF) / 2**32 * 20 - 10,
                ((seed_int >> 32) & 0xFFFFFFFF) / 2**32 * 20 - 10,
                ((seed_int >> 64) & 0xFFFFFFFF) / 2**32 * 30,
            ], dtype=np.float64)
        else:
            # 1维初始状态
            return np.array([
                ((seed_int >> 0) & 0xFFFFFFFF) / 2**32
            ], dtype=np.float64)


class AdaptiveChaosSystem:
    """自适应混沌系统（带GNN监控）"""
    
    def __init__(
        self,
        complexity: str = 'high',
        monitor_window: int = 64,
        quality_threshold: float = 0.6
    ):
        self.chaos = ChaosSystem(complexity)
        self.monitor_window = monitor_window
        self.quality_threshold = quality_threshold
        
        # 可选：加载GNN监控器
        self.monitor = None
        try:
            from src.neural.gnn_monitor import ChaosStateMonitor
            self.monitor = ChaosStateMonitor(self.chaos.state_dim)
            print(f"✓ GNN监控器已加载（{complexity}复杂度）")
        except Exception as e:
            print(f"⚠ GNN监控器未加载: {e}")
    
    def generate_and_monitor(
        self,
        initial_state: np.ndarray,
        length: int
    ) -> tuple:
        """
        生成轨迹并监控质量
        
        Returns:
            trajectory: 混沌轨迹
            quality_scores: 质量评分列表
            adjustments: 参数调整记录
        """
        # 生成轨迹
        trajectory = self.chaos.generate_trajectory(initial_state, length)
        
        # 如果没有监控器，直接返回
        if self.monitor is None:
            return trajectory, [], []
        
        # 监控质量
        quality_scores = []
        adjustments = []
        
        window = self.monitor_window
        for start in range(0, len(trajectory) - window + 1, window // 2):
            window_slice = trajectory[start:start + window]
            scores = self.monitor.assess(window_slice)
            quality_scores.append(scores)
            
            # 如果质量不足，调整参数
            if scores['quality'] < self.quality_threshold:
                adj = self._adjust_parameters(scores)
                if adj:
                    adjustments.append(adj)
        
        return trajectory, quality_scores, adjustments
    
    def _adjust_parameters(self, scores: dict) -> dict:
        """根据评分调整参数"""
        adjustment = {}
        
        if self.chaos.complexity == 'high':
            if scores['entropy'] < 0.5:
                self.chaos.params['mu'] = min(self.chaos.params['mu'] * 1.01, 3.999)
                adjustment['mu'] = 1.01
            if scores['lyapunov'] < 0.5:
                self.chaos.params['a'] *= 1.02
                adjustment['a'] = 1.02
        elif self.chaos.complexity == 'medium':
            if scores['entropy'] < 0.5:
                self.chaos.params['rho'] *= 1.01
                adjustment['rho'] = 1.01
        else:
            if scores['entropy'] < 0.5:
                self.chaos.params['r'] = min(self.chaos.params['r'] * 1.001, 4.0)
                adjustment['r'] = 1.001
        
        return adjustment





