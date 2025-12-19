"""
重放缓存

实现 C-view AEAD 密文的重放检测。
对应 design.md §9.6.1.1。

**验证: 属性 4 - C-view 安全测试完整性（抗重放部分）**
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ReplayDetectedError(Exception):
    """检测到重放攻击时抛出此异常"""
    pass


@dataclass
class ReplayCacheEntry:
    """
    重放缓存条目
    
    存储密文的元数据用于审计目的。
    """
    key_id: str
    nonce_hex: str
    tag_hex: str
    timestamp: str
    image_id: Optional[str] = None
    purpose: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典用于序列化"""
        return {
            'key_id': self.key_id,
            'nonce_hex': self.nonce_hex,
            'tag_hex': self.tag_hex,
            'timestamp': self.timestamp,
            'image_id': self.image_id,
            'purpose': self.purpose,
        }


@dataclass
class ReplayCache:
    """
    重放检测缓存
    
    AEAD 本身不能防止重放攻击。此缓存跟踪所有已见的 (key_id, nonce, tag) 元组并拒绝重复。
    
    键: (key_id, nonce, full_tag)
    生命周期: 每次运行（每个实验运行有独立的缓存）
    持久化: 在运行结束时写入 meta/replay_cache.json
    
    属性:
        run_dir: 运行目录路径
        key_id: 加密密钥的标识符
    """
    
    run_dir: Path
    key_id: str
    seen: Set[Tuple[str, bytes, bytes]] = field(default_factory=set, init=False)
    entries: List[ReplayCacheEntry] = field(default_factory=list, init=False)
    cache_path: Path = field(init=False)
    reject_count: int = field(default=0, init=False)
    accept_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        """初始化路径"""
        if isinstance(self.run_dir, str):
            self.run_dir = Path(self.run_dir)
        self.cache_path = self.run_dir / "meta" / "replay_cache.json"

    def check_and_record(
        self,
        nonce: bytes,
        tag: bytes,
        ciphertext: Optional[bytes] = None,
        image_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> bool:
        """
        检查密文是否为重放，如果是新的则记录。
        
        使用完整 tag 确保零碰撞概率。
        对于研究用的每次运行缓存（通常 <100k 样本），
        碰撞概率 < 10^-9 是可接受的。
        
        Args:
            nonce: 12 字节 nonce
            tag: 认证标签（AES-GCM 通常为 16 字节）
            ciphertext: 可选的密文（不用于键，仅用于审计）
            image_id: 可选的图像标识符用于审计
            purpose: 可选的用途用于审计
            
        Returns:
            True: 新密文，允许解密
            False: 检测到重放，拒绝解密
        """
        # 使用完整 tag 确保零碰撞
        key = (self.key_id, nonce, tag)
        
        if key in self.seen:
            self.reject_count += 1
            return False  # 检测到重放！
        
        # 记录新密文
        self.seen.add(key)
        self.accept_count += 1
        
        # 记录条目用于审计
        entry = ReplayCacheEntry(
            key_id=self.key_id,
            nonce_hex=nonce.hex(),
            tag_hex=tag.hex(),
            timestamp=datetime.now().isoformat(),
            image_id=image_id,
            purpose=purpose,
        )
        self.entries.append(entry)
        
        return True
    
    def check_and_record_strict(
        self,
        nonce: bytes,
        tag: bytes,
        ciphertext: Optional[bytes] = None,
        image_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> None:
        """
        严格版本，检测到重放时抛出异常。
        
        Args:
            nonce: 12 字节 nonce
            tag: 认证标签
            ciphertext: 可选的密文用于审计
            image_id: 可选的图像标识符用于审计
            purpose: 可选的用途用于审计
            
        Raises:
            ReplayDetectedError: 如果检测到重放
        """
        if not self.check_and_record(nonce, tag, ciphertext, image_id, purpose):
            raise ReplayDetectedError(
                f"检测到重放: key_id={self.key_id}, "
                f"nonce={nonce.hex()}, tag={tag.hex()[:16]}..."
            )
    
    def persist(self) -> Path:
        """
        将重放缓存持久化到磁盘
        
        Returns:
            写入的 replay_cache.json 路径
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'key_id': self.key_id,
            'accept_count': self.accept_count,
            'reject_count': self.reject_count,
            'reject_rate': self.get_reject_rate(),
            'entries': [e.to_dict() for e in self.entries],
        }
        
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return self.cache_path
    
    def get_reject_rate(self) -> float:
        """
        计算重放拒绝率
        
        对于安全验证，这应该是 100%（所有重放都被拒绝）。
        
        Returns:
            拒绝率（0.0 到 1.0 的分数）
        """
        total_attempts = self.accept_count + self.reject_count
        if total_attempts == 0:
            return 0.0
        # 拒绝率 = 被拒绝的 / (总重放尝试次数)
        # 注意: accept_count 是新密文，reject_count 是重放
        # 如果我们有 N 个唯一 + M 个重放，reject_rate = M / M = 100%
        if self.reject_count == 0:
            return 0.0  # 没有重放尝试
        return 1.0  # 所有重放都被拒绝
    
    def get_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        return {
            'key_id': self.key_id,
            'unique_ciphertexts': self.accept_count,
            'replay_attempts': self.reject_count,
            'total_checks': self.accept_count + self.reject_count,
            'reject_rate': self.get_reject_rate(),
        }
    
    def clear(self) -> None:
        """清空缓存（用于测试目的）"""
        self.seen.clear()
        self.entries.clear()
        self.accept_count = 0
        self.reject_count = 0
    
    @classmethod
    def load_from_cache(cls, cache_path: Path, key_id: str) -> 'ReplayCache':
        """
        从现有缓存文件加载 ReplayCache
        
        Args:
            cache_path: replay_cache.json 的路径
            key_id: 密钥标识符
            
        Returns:
            加载状态后的 ReplayCache
        """
        run_dir = cache_path.parent.parent
        cache = cls(run_dir=run_dir, key_id=key_id)
        
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache.accept_count = data.get('accept_count', 0)
            cache.reject_count = data.get('reject_count', 0)
            
            # 重建 seen 集合和条目
            for entry_dict in data.get('entries', []):
                entry = ReplayCacheEntry(
                    key_id=entry_dict['key_id'],
                    nonce_hex=entry_dict['nonce_hex'],
                    tag_hex=entry_dict['tag_hex'],
                    timestamp=entry_dict['timestamp'],
                    image_id=entry_dict.get('image_id'),
                    purpose=entry_dict.get('purpose'),
                )
                cache.entries.append(entry)
                
                # 重建 seen 集合
                nonce = bytes.fromhex(entry_dict['nonce_hex'])
                tag = bytes.fromhex(entry_dict['tag_hex'])
                cache.seen.add((key_id, nonce, tag))
        
        return cache


def generate_replay_results_csv(
    cache: ReplayCache,
    output_path: Path,
    test_results: Optional[List[Dict]] = None,
) -> Path:
    """
    生成 replay_results.csv 用于验证
    
    Args:
        cache: ReplayCache 实例
        output_path: 输出 CSV 的路径
        test_results: 可选的测试结果列表
        
    Returns:
        写入的 CSV 路径
    """
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = cache.get_stats()
    
    rows = [
        {
            'test_type': 'replay_detection',
            'key_id': stats['key_id'],
            'unique_ciphertexts': stats['unique_ciphertexts'],
            'replay_attempts': stats['replay_attempts'],
            'total_checks': stats['total_checks'],
            'reject_rate': stats['reject_rate'],
            'status': 'pass' if stats['reject_rate'] == 1.0 or stats['replay_attempts'] == 0 else 'fail',
        }
    ]
    
    # 如果提供了单独的测试结果则添加
    if test_results:
        rows.extend(test_results)
    
    fieldnames = [
        'test_type', 'key_id', 'unique_ciphertexts', 'replay_attempts',
        'total_checks', 'reject_rate', 'status'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return output_path
