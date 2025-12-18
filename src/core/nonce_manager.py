"""
Nonce Manager for Top-Journal Experiment Suite.

Implements deterministic nonce derivation with uniqueness constraints.
Corresponds to design.md §5.5.3.

**Validates: Property 4 - C-view 安全测试完整性（nonce 唯一性部分）**
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class NonceReuseError(Exception):
    """Raised when nonce reuse is detected."""
    pass


@dataclass
class NonceDerivationInput:
    """
    Nonce derivation input tuple (frozen per §5.5.3).
    
    All fields are required for deterministic nonce derivation.
    The combination must be unique within a run.
    """
    image_id: str
    method: str           # e.g., "causal_vse_pc"
    privacy_level: float  # e.g., 0.5
    training_mode: str    # e.g., "Z2Z"
    purpose: str          # e.g., "c_view_encrypt", "tamper_test", "avalanche_test"
    
    # Valid purpose values (frozen)
    VALID_PURPOSES = frozenset([
        'c_view_encrypt',
        'tamper_test',
        'avalanche_test',
        'replay_test',
    ])
    
    def __post_init__(self):
        """Validate input fields."""
        if self.purpose not in self.VALID_PURPOSES:
            raise ValueError(
                f"Invalid purpose '{self.purpose}'. "
                f"Must be one of: {self.VALID_PURPOSES}"
            )
    
    def to_derivation_string(self) -> str:
        """Convert to derivation string for hashing."""
        return f"{self.image_id}|{self.method}|{self.privacy_level}|{self.training_mode}|{self.purpose}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
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
    Nonce manager - ensures uniqueness.
    
    Implements deterministic nonce derivation using protocol-unique tuple:
    nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
    
    The manager tracks all used nonces within a run and raises NonceReuseError
    if a duplicate is detected.
    
    Attributes:
        master_key: Master key for nonce derivation
        run_dir: Path to run directory for logging
    """
    
    master_key: bytes
    run_dir: Path
    used_nonces: Set[bytes] = field(default_factory=set, init=False)
    log_entries: List[Dict] = field(default_factory=list, init=False)
    nonce_log_path: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize paths."""
        if isinstance(self.run_dir, str):
            self.run_dir = Path(self.run_dir)
        self.nonce_log_path = self.run_dir / "meta" / "nonce_log.json"
    
    def derive_nonce(self, input: NonceDerivationInput) -> bytes:
        """
        Derive deterministic nonce with uniqueness check.
        
        nonce = H(master_key, image_id, method, privacy_level, training_mode, purpose)[:12]
        
        Args:
            input: NonceDerivationInput with all required fields
            
        Returns:
            12-byte nonce (96-bit for AES-GCM)
            
        Raises:
            NonceReuseError: If nonce has already been used in this run
        """
        derivation_string = input.to_derivation_string()
        
        # Compute nonce: H(master_key || derivation_string)[:12]
        nonce = hashlib.sha256(
            self.master_key + derivation_string.encode('utf-8')
        ).digest()[:12]
        
        # Check uniqueness
        if nonce in self.used_nonces:
            raise NonceReuseError(
                f"Nonce reuse detected for input: {input.to_dict()}"
            )
        
        # Record nonce
        self.used_nonces.add(nonce)
        self._log_nonce(input, nonce)
        
        return nonce
    
    def _log_nonce(self, input: NonceDerivationInput, nonce: bytes) -> None:
        """Record nonce usage for audit."""
        entry = input.to_dict()
        entry['nonce_hex'] = nonce.hex()
        entry['timestamp'] = datetime.now().isoformat()
        self.log_entries.append(entry)
    
    def persist(self) -> Path:
        """
        Persist nonce log to disk.
        
        Returns:
            Path to written nonce_log.json
        """
        self.nonce_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.nonce_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_entries, f, indent=2)
        
        return self.nonce_log_path
    
    def get_nonce_count(self) -> int:
        """Get number of nonces generated."""
        return len(self.used_nonces)
    
    def check_uniqueness(self) -> bool:
        """
        Verify all logged nonces are unique.
        
        Returns:
            True if all nonces are unique
        """
        nonces = [e.get('nonce_hex') for e in self.log_entries]
        return len(nonces) == len(set(nonces))
    
    @classmethod
    def load_from_log(cls, nonce_log_path: Path, master_key: bytes) -> 'NonceManager':
        """
        Load NonceManager from existing log file.
        
        Args:
            nonce_log_path: Path to nonce_log.json
            master_key: Master key (for verification)
            
        Returns:
            NonceManager with loaded state
        """
        run_dir = nonce_log_path.parent.parent
        manager = cls(master_key=master_key, run_dir=run_dir)
        
        if nonce_log_path.exists():
            with open(nonce_log_path, 'r', encoding='utf-8') as f:
                manager.log_entries = json.load(f)
            
            # Rebuild used_nonces set
            for entry in manager.log_entries:
                nonce_hex = entry.get('nonce_hex')
                if nonce_hex:
                    manager.used_nonces.add(bytes.fromhex(nonce_hex))
        
        return manager
