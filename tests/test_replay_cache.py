"""
重放缓存测试模块。

验证: 属性 4 - C-view 安全测试完整性（抗重放部分）
验证: 需求 §9.6.1 - replay reject_rate = 100%
"""

import os
import json
import tempfile
from pathlib import Path

import pytest

from src.core.replay_cache import (
    ReplayCache,
    ReplayCacheEntry,
    ReplayDetectedError,
    generate_replay_results_csv,
)


class TestReplayCacheEntry:
    """ReplayCacheEntry 数据类测试。"""
    
    def test_entry_creation(self):
        """测试基本条目创建。"""
        entry = ReplayCacheEntry(
            key_id="test_key",
            nonce_hex="0" * 24,
            tag_hex="0" * 32,
            timestamp="2024-01-01T00:00:00",
        )
        assert entry.key_id == "test_key"
        assert entry.nonce_hex == "0" * 24
        assert entry.tag_hex == "0" * 32
    
    def test_entry_to_dict(self):
        """测试条目序列化。"""
        entry = ReplayCacheEntry(
            key_id="test_key",
            nonce_hex="aabbcc",
            tag_hex="ddeeff",
            timestamp="2024-01-01T00:00:00",
            image_id="img_001",
            purpose="c_view_encrypt",
        )
        d = entry.to_dict()
        assert d['key_id'] == "test_key"
        assert d['image_id'] == "img_001"
        assert d['purpose'] == "c_view_encrypt"


class TestReplayCache:
    """ReplayCache 测试。"""
    
    @pytest.fixture
    def temp_run_dir(self):
        """创建临时运行目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            yield run_dir
    
    @pytest.fixture
    def cache(self, temp_run_dir):
        """创建 ReplayCache 实例。"""
        return ReplayCache(run_dir=temp_run_dir, key_id="test_key_001")
    
    def test_cache_creation(self, cache, temp_run_dir):
        """测试缓存初始化。"""
        assert cache.key_id == "test_key_001"
        assert cache.run_dir == temp_run_dir
        assert cache.accept_count == 0
        assert cache.reject_count == 0
    
    def test_check_and_record_new_ciphertext(self, cache):
        """测试接受新密文。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        result = cache.check_and_record(nonce, tag)
        
        assert result is True
        assert cache.accept_count == 1
        assert cache.reject_count == 0
    
    def test_check_and_record_replay_detected(self, cache):
        """测试拒绝重放。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # 第一次 - 接受
        result1 = cache.check_and_record(nonce, tag)
        assert result1 is True
        
        # 第二次 - 拒绝（重放）
        result2 = cache.check_and_record(nonce, tag)
        assert result2 is False
        
        assert cache.accept_count == 1
        assert cache.reject_count == 1
    
    def test_different_nonces_accepted(self, cache):
        """测试不同 nonce 被接受。"""
        tag = bytes.fromhex("0" * 32)
        
        for i in range(10):
            nonce = bytes([i] * 12)
            result = cache.check_and_record(nonce, tag)
            assert result is True
        
        assert cache.accept_count == 10
        assert cache.reject_count == 0
    
    def test_different_tags_accepted(self, cache):
        """测试相同 nonce 不同 tag 被接受。"""
        nonce = bytes.fromhex("0" * 24)
        
        for i in range(10):
            tag = bytes([i] * 16)
            result = cache.check_and_record(nonce, tag)
            assert result is True
        
        assert cache.accept_count == 10
        assert cache.reject_count == 0

    def test_check_and_record_strict_raises(self, cache):
        """测试严格模式在重放时抛出异常。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # 第一次 - 无异常
        cache.check_and_record_strict(nonce, tag)
        
        # 第二次 - 应该抛出异常
        with pytest.raises(ReplayDetectedError):
            cache.check_and_record_strict(nonce, tag)
    
    def test_persist_and_load(self, cache, temp_run_dir):
        """测试缓存持久化和加载。"""
        nonce1 = bytes.fromhex("aa" * 12)
        tag1 = bytes.fromhex("bb" * 16)
        nonce2 = bytes.fromhex("cc" * 12)
        tag2 = bytes.fromhex("dd" * 16)
        
        cache.check_and_record(nonce1, tag1, image_id="img_001")
        cache.check_and_record(nonce2, tag2, image_id="img_002")
        
        # 持久化
        cache_path = cache.persist()
        assert cache_path.exists()
        
        # 加载
        loaded = ReplayCache.load_from_cache(cache_path, "test_key_001")
        assert loaded.accept_count == 2
        assert len(loaded.entries) == 2
        assert len(loaded.seen) == 2
        
        # 验证加载的缓存拒绝重放
        result = loaded.check_and_record(nonce1, tag1)
        assert result is False
    
    def test_get_stats(self, cache):
        """测试统计信息获取。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        cache.check_and_record(nonce, tag)
        cache.check_and_record(nonce, tag)  # 重放
        
        stats = cache.get_stats()
        
        assert stats['unique_ciphertexts'] == 1
        assert stats['replay_attempts'] == 1
        assert stats['total_checks'] == 2
        assert stats['reject_rate'] == 1.0
    
    def test_reject_rate_100_percent(self, cache):
        """测试重放尝试时拒绝率为 100%。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # 首次接受
        cache.check_and_record(nonce, tag)
        
        # 多次重放尝试
        for _ in range(10):
            cache.check_and_record(nonce, tag)
        
        # 所有重放应被拒绝
        assert cache.reject_count == 10
        assert cache.get_reject_rate() == 1.0
    
    def test_clear(self, cache):
        """测试缓存清除。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        cache.check_and_record(nonce, tag)
        assert cache.accept_count == 1
        
        cache.clear()
        
        assert cache.accept_count == 0
        assert cache.reject_count == 0
        assert len(cache.seen) == 0
        assert len(cache.entries) == 0
        
        # 清除后，相同密文应被接受
        result = cache.check_and_record(nonce, tag)
        assert result is True
    
    def test_metadata_logging(self, cache):
        """测试元数据正确记录。"""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        cache.check_and_record(
            nonce, tag,
            image_id="img_001",
            purpose="c_view_encrypt"
        )
        
        assert len(cache.entries) == 1
        entry = cache.entries[0]
        assert entry.image_id == "img_001"
        assert entry.purpose == "c_view_encrypt"
        assert entry.nonce_hex == "0" * 24
        assert entry.tag_hex == "0" * 32


class TestGenerateReplayResultsCsv:
    """CSV 生成测试。"""
    
    @pytest.fixture
    def temp_run_dir(self):
        """创建临时运行目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            (run_dir / "tables").mkdir()
            yield run_dir
    
    def test_generate_csv(self, temp_run_dir):
        """测试 CSV 生成。"""
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # 添加一些数据
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        cache.check_and_record(nonce, tag)
        cache.check_and_record(nonce, tag)  # 重放
        
        output_path = temp_run_dir / "tables" / "replay_results.csv"
        result_path = generate_replay_results_csv(cache, output_path)
        
        assert result_path.exists()
        
        # 读取并验证
        import csv
        with open(result_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]['test_type'] == 'replay_detection'
        assert rows[0]['reject_rate'] == '1.0'
        assert rows[0]['status'] == 'pass'


class TestReplayCacheProperty:
    """ReplayCache 属性测试。
    
    **功能: top-journal-experiment-suite, 属性: 重放 reject_rate = 100%**
    **验证: 需求 §9.6.1, RQ3**
    """
    
    @pytest.fixture
    def temp_run_dir(self):
        """创建临时运行目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            yield run_dir
    
    def test_property_all_replays_rejected(self, temp_run_dir):
        """
        属性: 对于任意密文序列，所有重放尝试必须被拒绝（reject_rate = 100%）。
        
        **功能: top-journal-experiment-suite, 属性: 重放 reject_rate = 100%**
        **验证: 需求 §9.6.1, RQ3**
        """
        import random
        
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # 生成随机唯一密文
        unique_ciphertexts = []
        for i in range(50):
            nonce = bytes([random.randint(0, 255) for _ in range(12)])
            tag = bytes([random.randint(0, 255) for _ in range(16)])
            unique_ciphertexts.append((nonce, tag))
        
        # 第一轮: 全部应被接受
        for nonce, tag in unique_ciphertexts:
            result = cache.check_and_record(nonce, tag)
            assert result is True, "新密文应被接受"
        
        # 第二轮: 全部应被拒绝（重放）
        for nonce, tag in unique_ciphertexts:
            result = cache.check_and_record(nonce, tag)
            assert result is False, "重放应被拒绝"
        
        # 验证拒绝率
        assert cache.accept_count == 50
        assert cache.reject_count == 50
        assert cache.get_reject_rate() == 1.0
    
    def test_property_unique_ciphertexts_always_accepted(self, temp_run_dir):
        """
        属性: 对于任意唯一 (nonce, tag) 对集合，全部应被接受。
        
        **功能: top-journal-experiment-suite, 属性: 唯一接受**
        **验证: 需求 §9.6.1**
        """
        import random
        
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # 生成唯一密文
        seen_pairs = set()
        accepted = 0
        
        for _ in range(100):
            # 生成唯一对
            while True:
                nonce = bytes([random.randint(0, 255) for _ in range(12)])
                tag = bytes([random.randint(0, 255) for _ in range(16)])
                pair = (nonce, tag)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    break
            
            result = cache.check_and_record(nonce, tag)
            if result:
                accepted += 1
        
        # 所有唯一密文应被接受
        assert accepted == 100
        assert cache.reject_count == 0
