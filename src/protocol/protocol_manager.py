"""
Protocol Manager for Top-Journal Experiment Suite.

Implements protocol version management, snapshot generation, and consistency validation.
Corresponds to design.md §8.4 and §12.1.

**Validates: Property 13 - 协议版本一致性**
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ProtocolError(Exception):
    """Protocol-related errors."""
    pass


class ProtocolVersionMismatchError(ProtocolError):
    """Raised when protocol_version != schema_version."""
    pass


@dataclass
class ProtocolManager:
    """
    Protocol version manager.
    
    Manages protocol versioning, snapshot generation, and consistency validation.
    Version numbers follow semantic versioning and must match schema_version.
    
    Attributes:
        run_dir: Path to the experiment run directory
        
    Version bump rules (frozen per §8.4):
        - Major (X.0.0): Schema field additions/deletions
        - Minor (0.X.0): Protocol logic changes
        - Patch (0.0.X): Bug fixes/documentation clarifications
    """
    
    run_dir: Path
    meta_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    
    # Frozen version numbers - must match design.md v2.1.1 (class variables)
    PROTOCOL_VERSION: str = field(default="2.1.1", init=False)
    SCHEMA_VERSION: str = field(default="2.1.1", init=False)
    
    def __post_init__(self):
        """Initialize directory paths."""
        if isinstance(self.run_dir, str):
            self.run_dir = Path(self.run_dir)
        self.meta_dir = self.run_dir / "meta"
        self.reports_dir = self.run_dir / "reports"
    
    def write_protocol_version(self) -> Path:
        """
        Write protocol version to meta/protocol_version.txt.
        
        Returns:
            Path to the written file
        """
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        version_file = self.meta_dir / "protocol_version.txt"
        version_file.write_text(self.PROTOCOL_VERSION)
        return version_file

    def write_protocol_snapshot(self, config: Dict[str, Any]) -> Path:
        """
        Write protocol snapshot to reports/protocol_snapshot.md.
        
        The snapshot contains:
        - Protocol and schema versions
        - Git commit hash
        - Timestamp
        - Frozen configuration parameters
        - A2 strength contract summary
        
        Args:
            config: Complete experiment configuration
            
        Returns:
            Path to the written snapshot file
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.reports_dir / "protocol_snapshot.md"
        
        snapshot_content = self._generate_snapshot(config)
        snapshot_path.write_text(snapshot_content, encoding='utf-8')
        return snapshot_path
    
    def _generate_snapshot(self, config: Dict[str, Any]) -> str:
        """Generate protocol snapshot markdown content."""
        git_commit = self._get_git_commit()
        timestamp = datetime.now().isoformat()
        config_hash = self._compute_config_hash(config)
        
        snapshot = f"""# Protocol Snapshot

> **Generated**: {timestamp}
> **Protocol Version**: {self.PROTOCOL_VERSION}
> **Schema Version**: {self.SCHEMA_VERSION}
> **Git Commit**: {git_commit}
> **Config Hash**: {config_hash}

## Version Consistency

| Component | Version |
|-----------|---------|
| Protocol | {self.PROTOCOL_VERSION} |
| Schema | {self.SCHEMA_VERSION} |
| Design Doc | v{self.PROTOCOL_VERSION} |

## Frozen Parameters

### Privacy Levels (§2.3)
```
λ ∈ {{0.0, 0.3, 0.5, 0.7, 1.0}}
```

### Threat Levels (§2.4)
```
threat_level ∈ {{A0, A1, A2}}
```

### Training Modes (§8.1)
```
training_mode ∈ {{P2P, P2Z, Z2Z, Mix2Z}}
```

### Attack Types (§6.1)
```
attack_type ∈ {{face_verification, attribute_inference, reconstruction, membership_inference, property_inference}}
```

## A2 Strength Contract (§5.4)

### Attack Families
| Family | Min Instances |
|--------|---------------|
| Reconstruction | 2 |
| Inference | 3 |
| Optimization | 2 |

### Attack Budget
| Parameter | Value |
|-----------|-------|
| Training epochs | 100 |
| LR search | {{1e-4, 1e-3, 1e-2}} |
| Max GPU time | ≤24h/family |

### Worst-case Aggregation
```
worst_case_attack_success = max(attack_success) over same (dataset, task, privacy_level, threat_level)
```

## Configuration Summary

```yaml
{self._format_config_summary(config)}
```

---
*This snapshot is frozen. Any modification requires protocol_version bump.*
"""
        return snapshot
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.run_dir.parent if self.run_dir.exists() else '.'
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return "unknown"
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _format_config_summary(self, config: Dict[str, Any]) -> str:
        """Format configuration for snapshot."""
        # Extract key parameters
        summary_keys = [
            'dataset', 'task', 'method', 'seed', 
            'privacy_levels', 'training_modes', 'attack_types'
        ]
        summary = {}
        for key in summary_keys:
            if key in config:
                summary[key] = config[key]
        
        # Format as YAML-like string
        lines = []
        for key, value in summary.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        return '\n'.join(lines) if lines else "# No configuration provided"

    def validate_consistency(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate configuration and protocol snapshot consistency.
        
        Checks:
        1. protocol_version == schema_version
        2. protocol_snapshot.md exists and matches config (if provided)
        3. meta/protocol_version.txt matches PROTOCOL_VERSION
        
        Args:
            config: Optional configuration to validate against snapshot
            
        Returns:
            True if all consistency checks pass
            
        Raises:
            ProtocolVersionMismatchError: If versions don't match
            ProtocolError: If snapshot is missing or inconsistent
        """
        # Check 1: protocol_version == schema_version
        if self.PROTOCOL_VERSION != self.SCHEMA_VERSION:
            raise ProtocolVersionMismatchError(
                f"Protocol version ({self.PROTOCOL_VERSION}) != "
                f"Schema version ({self.SCHEMA_VERSION})"
            )
        
        # Check 2: protocol_version.txt exists and matches
        version_file = self.meta_dir / "protocol_version.txt"
        if version_file.exists():
            stored_version = version_file.read_text().strip()
            if stored_version != self.PROTOCOL_VERSION:
                raise ProtocolVersionMismatchError(
                    f"Stored protocol version ({stored_version}) != "
                    f"Current version ({self.PROTOCOL_VERSION})"
                )
        
        # Check 3: protocol_snapshot.md exists
        snapshot_path = self.reports_dir / "protocol_snapshot.md"
        if not snapshot_path.exists():
            raise ProtocolError(
                f"Missing protocol_snapshot.md at {snapshot_path}"
            )
        
        # Check 4: If config provided, verify hash matches
        if config is not None:
            snapshot_content = snapshot_path.read_text(encoding='utf-8')
            current_hash = self._compute_config_hash(config)
            
            # Extract hash from snapshot
            if f"Config Hash**: {current_hash}" not in snapshot_content:
                # Hash mismatch - config has changed
                raise ProtocolError(
                    f"Configuration hash mismatch. "
                    f"Current: {current_hash}, "
                    f"Snapshot may be outdated."
                )
        
        return True
    
    def get_version_info(self) -> Dict[str, str]:
        """
        Get version information dictionary.
        
        Returns:
            Dictionary with protocol_version, schema_version, git_commit
        """
        return {
            "protocol_version": self.PROTOCOL_VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "git_commit": self._get_git_commit(),
        }
    
    def ensure_directories(self) -> None:
        """Create all required run directories."""
        required_dirs = [
            self.meta_dir,
            self.run_dir / "tables",
            self.run_dir / "figures",
            self.run_dir / "logs",
            self.reports_dir,
        ]
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def check_version_bump_needed(cls, change_type: str) -> str:
        """
        Determine version bump type needed for a change.
        
        Args:
            change_type: One of 'schema_field', 'protocol_logic', 'bugfix'
            
        Returns:
            Description of required version bump
        """
        bump_rules = {
            'schema_field': 'Major (X.0.0) - Schema field additions/deletions',
            'protocol_logic': 'Minor (0.X.0) - Protocol logic changes',
            'bugfix': 'Patch (0.0.X) - Bug fixes/documentation clarifications',
        }
        return bump_rules.get(change_type, 'Unknown change type')
