"""
Tests for ReplayCache.

Validates: Property 4 - C-view 安全测试完整性（抗重放部分）
Validates: Requirements §9.6.1 - replay reject_rate = 100%
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
    """Tests for ReplayCacheEntry dataclass."""
    
    def test_entry_creation(self):
        """Test basic entry creation."""
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
        """Test entry serialization."""
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
    """Tests for ReplayCache."""
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            yield run_dir
    
    @pytest.fixture
    def cache(self, temp_run_dir):
        """Create ReplayCache instance."""
        return ReplayCache(run_dir=temp_run_dir, key_id="test_key_001")
    
    def test_cache_creation(self, cache, temp_run_dir):
        """Test cache initialization."""
        assert cache.key_id == "test_key_001"
        assert cache.run_dir == temp_run_dir
        assert cache.accept_count == 0
        assert cache.reject_count == 0
    
    def test_check_and_record_new_ciphertext(self, cache):
        """Test accepting new ciphertext."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        result = cache.check_and_record(nonce, tag)
        
        assert result is True
        assert cache.accept_count == 1
        assert cache.reject_count == 0
    
    def test_check_and_record_replay_detected(self, cache):
        """Test rejecting replay."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # First time - accept
        result1 = cache.check_and_record(nonce, tag)
        assert result1 is True
        
        # Second time - reject (replay)
        result2 = cache.check_and_record(nonce, tag)
        assert result2 is False
        
        assert cache.accept_count == 1
        assert cache.reject_count == 1
    
    def test_different_nonces_accepted(self, cache):
        """Test different nonces are accepted."""
        tag = bytes.fromhex("0" * 32)
        
        for i in range(10):
            nonce = bytes([i] * 12)
            result = cache.check_and_record(nonce, tag)
            assert result is True
        
        assert cache.accept_count == 10
        assert cache.reject_count == 0
    
    def test_different_tags_accepted(self, cache):
        """Test different tags with same nonce are accepted."""
        nonce = bytes.fromhex("0" * 24)
        
        for i in range(10):
            tag = bytes([i] * 16)
            result = cache.check_and_record(nonce, tag)
            assert result is True
        
        assert cache.accept_count == 10
        assert cache.reject_count == 0

    def test_check_and_record_strict_raises(self, cache):
        """Test strict mode raises exception on replay."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # First time - no exception
        cache.check_and_record_strict(nonce, tag)
        
        # Second time - should raise
        with pytest.raises(ReplayDetectedError):
            cache.check_and_record_strict(nonce, tag)
    
    def test_persist_and_load(self, cache, temp_run_dir):
        """Test cache persistence and loading."""
        nonce1 = bytes.fromhex("aa" * 12)
        tag1 = bytes.fromhex("bb" * 16)
        nonce2 = bytes.fromhex("cc" * 12)
        tag2 = bytes.fromhex("dd" * 16)
        
        cache.check_and_record(nonce1, tag1, image_id="img_001")
        cache.check_and_record(nonce2, tag2, image_id="img_002")
        
        # Persist
        cache_path = cache.persist()
        assert cache_path.exists()
        
        # Load
        loaded = ReplayCache.load_from_cache(cache_path, "test_key_001")
        assert loaded.accept_count == 2
        assert len(loaded.entries) == 2
        assert len(loaded.seen) == 2
        
        # Verify loaded cache rejects replays
        result = loaded.check_and_record(nonce1, tag1)
        assert result is False
    
    def test_get_stats(self, cache):
        """Test statistics retrieval."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        cache.check_and_record(nonce, tag)
        cache.check_and_record(nonce, tag)  # Replay
        
        stats = cache.get_stats()
        
        assert stats['unique_ciphertexts'] == 1
        assert stats['replay_attempts'] == 1
        assert stats['total_checks'] == 2
        assert stats['reject_rate'] == 1.0
    
    def test_reject_rate_100_percent(self, cache):
        """Test reject rate is 100% when replays are attempted."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        # Accept first
        cache.check_and_record(nonce, tag)
        
        # Multiple replay attempts
        for _ in range(10):
            cache.check_and_record(nonce, tag)
        
        # All replays should be rejected
        assert cache.reject_count == 10
        assert cache.get_reject_rate() == 1.0
    
    def test_clear(self, cache):
        """Test cache clearing."""
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        
        cache.check_and_record(nonce, tag)
        assert cache.accept_count == 1
        
        cache.clear()
        
        assert cache.accept_count == 0
        assert cache.reject_count == 0
        assert len(cache.seen) == 0
        assert len(cache.entries) == 0
        
        # After clear, same ciphertext should be accepted
        result = cache.check_and_record(nonce, tag)
        assert result is True
    
    def test_metadata_logging(self, cache):
        """Test metadata is logged correctly."""
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
    """Tests for CSV generation."""
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            (run_dir / "tables").mkdir()
            yield run_dir
    
    def test_generate_csv(self, temp_run_dir):
        """Test CSV generation."""
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # Add some data
        nonce = bytes.fromhex("0" * 24)
        tag = bytes.fromhex("0" * 32)
        cache.check_and_record(nonce, tag)
        cache.check_and_record(nonce, tag)  # Replay
        
        output_path = temp_run_dir / "tables" / "replay_results.csv"
        result_path = generate_replay_results_csv(cache, output_path)
        
        assert result_path.exists()
        
        # Read and verify
        import csv
        with open(result_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]['test_type'] == 'replay_detection'
        assert rows[0]['reject_rate'] == '1.0'
        assert rows[0]['status'] == 'pass'


class TestReplayCacheProperty:
    """Property-based tests for ReplayCache.
    
    **Feature: top-journal-experiment-suite, Property: Replay reject_rate = 100%**
    **Validates: Requirements §9.6.1, RQ3**
    """
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)
            (run_dir / "meta").mkdir()
            yield run_dir
    
    def test_property_all_replays_rejected(self, temp_run_dir):
        """
        Property: For any sequence of ciphertexts, all replay attempts
        must be rejected (reject_rate = 100%).
        
        **Feature: top-journal-experiment-suite, Property: Replay reject_rate = 100%**
        **Validates: Requirements §9.6.1, RQ3**
        """
        import random
        
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # Generate random unique ciphertexts
        unique_ciphertexts = []
        for i in range(50):
            nonce = bytes([random.randint(0, 255) for _ in range(12)])
            tag = bytes([random.randint(0, 255) for _ in range(16)])
            unique_ciphertexts.append((nonce, tag))
        
        # First pass: all should be accepted
        for nonce, tag in unique_ciphertexts:
            result = cache.check_and_record(nonce, tag)
            assert result is True, "New ciphertext should be accepted"
        
        # Second pass: all should be rejected (replays)
        for nonce, tag in unique_ciphertexts:
            result = cache.check_and_record(nonce, tag)
            assert result is False, "Replay should be rejected"
        
        # Verify reject rate
        assert cache.accept_count == 50
        assert cache.reject_count == 50
        assert cache.get_reject_rate() == 1.0
    
    def test_property_unique_ciphertexts_always_accepted(self, temp_run_dir):
        """
        Property: For any set of unique (nonce, tag) pairs,
        all should be accepted.
        
        **Feature: top-journal-experiment-suite, Property: Unique acceptance**
        **Validates: Requirements §9.6.1**
        """
        import random
        
        cache = ReplayCache(run_dir=temp_run_dir, key_id="test_key")
        
        # Generate unique ciphertexts
        seen_pairs = set()
        accepted = 0
        
        for _ in range(100):
            # Generate unique pair
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
        
        # All unique ciphertexts should be accepted
        assert accepted == 100
        assert cache.reject_count == 0
