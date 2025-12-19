"""
Nonce 管理器

实现确定性 nonce 派生与唯一性约束。
对应 design.md §5.5.3。

**验证: 属性 4 - C-view 安全测试完整性（nonce 唯一性部分）**
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class NonceReuseError(Exception):
    """检测到 nonce 重用时抛出此异常"""
    pass


@dataclass
class NonceDerivationInput:
    """
    Nonce 派生输入元组（按 §5.5.3 冻结）
    
    所有字段都是确定性 nonce 派生所必需的。
    组合在单次运行中必须唯一。
    """
    image_id: str
    method: str           # 例如 "causal_vse_pc"
    privacy_level: float  # 例如 0.5
    training_mode: str    # 例如 "Z2Z"
    purpose: str          # 例如 "c_view_encrypt", "tamper_test", "avalanche_test"
    
    # 有效的 purpose 值（冻结）
    VALID_PURPOSES = frozenset([
        'c_view_encrypt',
        'tamper_test',
        'avalanche_test',
        'replay_test',
    ])
    
    def __post_init__(self):
        """验证输入字段"""
        if self.purpose not in self.VALID_PURPOSES:
            raise ValueError(
                f"无效的 purpose '{self.purpose}'。"
                f"必须是以下之一: {self.VALID_PURPOSES}"
            )
    
    def to_derivation_string(self) -> str:
        """转换为用于哈希的派生字符串"""
        return f"{self.image_id}|{self.method}|{self.privacy_level}|{self.training_mode}|{self.purpose}"
    
    def to_dict(self) -> Dict:
        """转换为字典用于日志记录"""
        return {
            'image_id': self.image_id,
            'method': self.method,
            'privacy_level': self.privacy_level,
            'training_mode': self.training_mode,
            'purpose': self.purpose,
        }


@dataclass
class NonceManager:
    """
    Nonce 管理器 - 确保唯一性
    
    使用协议唯一元组实现确定性 nonce 派生：
    nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
    
    管理器跟踪单次运行中所有已使用的 nonce，如果检测到重复则抛出 NonceReuseError。
    
    属性:
        master_key: 用于 nonce 派生的主密钥
        run_dir: 日志记录的运行目录路径
    """
    
    master_key: bytes
    run_dir: Path
    used_nonces: Set[bytes] = field(default_factory=set, init=False)
    log_entries: List[Dict] = field(default_factory=list, init=False)
    nonce_log_path: Path = field(init=False)
    
    def __post_init__(self):
        """初始化路径"""
        if isinstance(self.run_dir, str):
            self.run_dir = Path(self.run_dir)
        self.nonce_log_path = self.run_dir / "meta" / "nonce_log.json"
    
    def derive_nonce(self, input: NonceDerivationInput) -> bytes:
        """
        派生确定性 nonce 并检查唯一性
        
        nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
        
        Args:
            input: 包含所有必需字段的 NonceDerivationInput
            
        Returns:
            12 字节 nonce（96 位，用于 AES-GCM）
            
        Raises:
            NonceReuseError: 如果 nonce 在本次运行中已被使用
        """
        derivation_string = input.to_derivation_string()
        
        # 计算 nonce: H(master_key || derivation_string)[:12]
        nonce = hashlib.sha256(
            self.master_key + derivation_string.encode('utf-8')
        ).digest()[:12]
        
        # 检查唯一性
        if nonce in self.used_nonces:
            raise NonceReuseError(
                f"检测到 nonce 重用，输入: {input.to_dict()}"
            )
        
        # 记录 nonce
        self.used_nonces.add(nonce)
        self._log_nonce(input, nonce)
        
        return nonce
    
    def _log_nonce(self, input: NonceDerivationInput, nonce: bytes) -> None:
        """记录 nonce 使用情况用于审计"""
        entry = input.to_dict()
        entry['nonce_hex'] = nonce.hex()
        entry['timestamp'] = datetime.now().isoformat()
        self.log_entries.append(entry)
    
    def persist(self) -> Path:
        """
        将 nonce 日志持久化到磁盘
        
        Returns:
            写入的 nonce_log.json 路径
        """
        self.nonce_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.nonce_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_entries, f, indent=2)
        
        return self.nonce_log_path
    
    def get_nonce_count(self) -> int:
        """获取已生成的 nonce 数量"""
        return len(self.used_nonces)
    
    def check_uniqueness(self) -> bool:
        """
        验证所有已记录的 nonce 是否唯一
        
        Returns:
            如果所有 nonce 都唯一则返回 True
        """
        nonces = [e.get('nonce_hex') for e in self.log_entries]
        return len(nonces) == len(set(nonces))
    
    @classmethod
    def load_from_log(cls, nonce_log_path: Path, master_key: bytes) -> 'NonceManager':
        """
        从现有日志文件加载 NonceManager
        
        Args:
            nonce_log_path: nonce_log.json 的路径
            master_key: 主密钥（用于验证）
            
        Returns:
            加载状态后的 NonceManager
        """
        run_dir = nonce_log_path.parent.parent
        manager = cls(master_key=master_key, run_dir=run_dir)
        
        if nonce_log_path.exists():
            with open(nonce_log_path, 'r', encoding='utf-8') as f:
                manager.log_entries = json.load(f)
            
            # 重建 used_nonces 集合
            for entry in manager.log_entries:
                nonce_hex = entry.get('nonce_hex')
                if nonce_hex:
                    manager.used_nonces.add(bytes.fromhex(nonce_hex))
        
        return manager
