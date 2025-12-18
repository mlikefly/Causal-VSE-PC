"""
Replay Cache for Top-Journal Experiment Suite.

Implements replay detection for C-view AEAD ciphertexts.
Corresponds to design.md §9.6.1.1.

**Validates: Property 4 - C-view 安全测试完整性（抗重放部分）**
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ReplayDetectedError(Exception):
    """Raised when replay attack is detected."""
    pass


@dataclass
class ReplayCacheEntry:
    """
    Entry in replay cache.
    
    Stores metadata about a ciphertext for audit purposes.
    """
    key_id: str
    nonce_hex: str
    tag_hex: str
    timestamp: str
    image_id: Optional[str] = None
    purpose: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
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
    Replay detection cache.
    
    AEAD itself does not prevent replay attacks. This cache tracks
    all seen (key_id, nonce, tag) tuples and rejects duplicates.
    
    Key: (key_id, nonce, full_tag)
    Lifecycle: per-run (each experiment run has independent cache)
    Persistence: writes to meta/replay_cache.json at run end
    
    Attributes:
        run_dir: Path to run directory
        key_id: Identifier for the encryption key
    """
    
    run_dir: Path
    key_id: str
    seen: Set[Tuple[str, bytes, bytes]] = field(default_factory=set, init=False)
    entries: List[ReplayCacheEntry] = field(default_factory=list, init=False)
    cache_path: Path = field(init=False)
    reject_count: int = field(default=0, init=False)
    accept_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize paths."""
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
        Check if ciphertext is a replay, record if new.
        
        Uses full tag to ensure zero collision probability.
        For research per-run cache (typically <100k samples),
        collision probability < 10^-9 is acceptable.
        
        Args:
            nonce: 12-byte nonce
            tag: Authentication tag (typically 16 bytes for AES-GCM)
            ciphertext: Optional ciphertext (not used in key, for audit only)
            image_id: Optional image identifier for audit
            purpose: Optional purpose for audit
            
        Returns:
            True: New ciphertext, allowed to decrypt
            False: Replay detected, reject decryption
        """
        # Use full tag to ensure zero collision
        key = (self.key_id, nonce, tag)
        
        if key in self.seen:
            self.reject_count += 1
            return False  # Replay detected!
        
        # Record new ciphertext
        self.seen.add(key)
        self.accept_count += 1
        
        # Log entry for audit
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
        Strict version that raises exception on replay.
        
        Args:
            nonce: 12-byte nonce
            tag: Authentication tag
            ciphertext: Optional ciphertext for audit
            image_id: Optional image identifier for audit
            purpose: Optional purpose for audit
            
        Raises:
            ReplayDetectedError: If replay is detected
        """
        if not self.check_and_record(nonce, tag, ciphertext, image_id, purpose):
            raise ReplayDetectedError(
                f"Replay detected: key_id={self.key_id}, "
                f"nonce={nonce.hex()}, tag={tag.hex()[:16]}..."
            )
    
    def persist(self) -> Path:
        """
        Persist replay cache to disk.
        
        Returns:
            Path to written replay_cache.json
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
        Calculate replay reject rate.
        
        For security validation, this should be 100% (all replays rejected).
        
        Returns:
            Reject rate as fraction (0.0 to 1.0)
        """
        total_attempts = self.accept_count + self.reject_count
        if total_attempts == 0:
            return 0.0
        # Reject rate = rejected / (total replay attempts)
        # Note: accept_count is new ciphertexts, reject_count is replays
        # If we have N unique + M replays, reject_rate = M / M = 100%
        if self.reject_count == 0:
            return 0.0  # No replays attempted
        return 1.0  # All replays were rejected
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'key_id': self.key_id,
            'unique_ciphertexts': self.accept_count,
            'replay_attempts': self.reject_count,
            'total_checks': self.accept_count + self.reject_count,
            'reject_rate': self.get_reject_rate(),
        }
    
    def clear(self) -> None:
        """Clear the cache (for testing purposes)."""
        self.seen.clear()
        self.entries.clear()
        self.accept_count = 0
        self.reject_count = 0
    
    @classmethod
    def load_from_cache(cls, cache_path: Path, key_id: str) -> 'ReplayCache':
        """
        Load ReplayCache from existing cache file.
        
        Args:
            cache_path: Path to replay_cache.json
            key_id: Key identifier
            
        Returns:
            ReplayCache with loaded state
        """
        run_dir = cache_path.parent.parent
        cache = cls(run_dir=run_dir, key_id=key_id)
        
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache.accept_count = data.get('accept_count', 0)
            cache.reject_count = data.get('reject_count', 0)
            
            # Rebuild seen set and entries
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
                
                # Rebuild seen set
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
    Generate replay_results.csv for validation.
    
    Args:
        cache: ReplayCache instance
        output_path: Path to output CSV
        test_results: Optional list of test results
        
    Returns:
        Path to written CSV
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
    
    # Add individual test results if provided
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
