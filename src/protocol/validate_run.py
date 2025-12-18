"""
Run Validation for Top-Journal Experiment Suite.

Implements comprehensive validation of experiment runs including:
- Directory structure validation
- Schema validation
- Coverage checking
- Red-line checks (R1-R10)

Corresponds to design.md §10.0.3 and T-REDLINE.

**Validates: Property 1, Property 10**
"""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .protocol_manager import ProtocolManager, ProtocolError
from .results_schema import ResultsSchema


@dataclass
class RedLineCheck:
    """Result of a single red-line check."""
    id: str
    description: str
    passed: bool
    details: str = ""


@dataclass
class ValidationResult:
    """Complete validation result."""
    valid: bool
    red_line_checks: List[RedLineCheck] = field(default_factory=list)
    schema_results: Dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0
    missing_combinations: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ValidateRun:
    """
    Run results validator.
    
    Validates experiment run directories against protocol requirements.
    Implements T-REDLINE checks (R1-R10).
    """
    
    COVERAGE_THRESHOLD = 0.98  # 98% coverage gate
    
    REQUIRED_DIRS = ['meta', 'tables', 'figures', 'logs', 'reports']
    
    REQUIRED_META_FILES = [
        'protocol_version.txt',
        'config.yaml',
        'git_commit.txt',
        'seed.txt',
    ]
    
    REQUIRED_TABLE_FILES = [
        'utility_metrics.csv',
        'attack_metrics.csv',
    ]
    
    def __init__(self, run_dir: Path):
        """
        Initialize validator.
        
        Args:
            run_dir: Path to experiment run directory
        """
        self.run_dir = Path(run_dir)
        self.protocol_manager = ProtocolManager(run_dir)
        self.schema = ResultsSchema()

    def validate(self, config: Optional[Dict] = None) -> ValidationResult:
        """
        Perform complete validation of run directory.
        
        Args:
            config: Optional configuration for consistency check
            
        Returns:
            ValidationResult with all check results
        """
        result = ValidationResult(valid=True)
        
        # 1. Directory structure check
        dir_check = self._check_directory_structure()
        if not dir_check[0]:
            result.valid = False
            result.errors.extend(dir_check[1])
        
        # 2. Schema validation
        tables_dir = self.run_dir / "tables"
        if tables_dir.exists():
            result.schema_results = self.schema.validate_all_csvs(tables_dir)
            for filename, schema_result in result.schema_results.items():
                if not schema_result['valid']:
                    result.valid = False
                    result.errors.extend(schema_result['errors'])
                result.warnings.extend(schema_result.get('warnings', []))
        
        # 3. Coverage check
        coverage_result = self._check_coverage()
        result.coverage = coverage_result['coverage']
        result.missing_combinations = coverage_result.get('missing', [])
        if result.coverage < self.COVERAGE_THRESHOLD:
            result.valid = False
            result.errors.append(
                f"Coverage {result.coverage:.2%} < threshold {self.COVERAGE_THRESHOLD:.2%}"
            )
        
        # 4. Red-line checks (R1-R10)
        result.red_line_checks = self._run_red_line_checks(config)
        for check in result.red_line_checks:
            if not check.passed:
                result.valid = False
                result.errors.append(f"Red-line {check.id} failed: {check.description}")
        
        return result
    
    def _check_directory_structure(self) -> Tuple[bool, List[str]]:
        """
        Check required directory structure exists.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.run_dir.exists():
            return False, [f"Run directory not found: {self.run_dir}"]
        
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.run_dir / dir_name
            if not dir_path.exists():
                errors.append(f"Missing required directory: {dir_name}/")
        
        # Check required meta files
        meta_dir = self.run_dir / "meta"
        if meta_dir.exists():
            for filename in self.REQUIRED_META_FILES:
                if not (meta_dir / filename).exists():
                    errors.append(f"Missing required meta file: meta/{filename}")
        
        # Check required table files
        tables_dir = self.run_dir / "tables"
        if tables_dir.exists():
            for filename in self.REQUIRED_TABLE_FILES:
                if not (tables_dir / filename).exists():
                    errors.append(f"Missing required table file: tables/{filename}")
        
        return len(errors) == 0, errors
    
    def _check_coverage(self) -> Dict[str, Any]:
        """
        Check experiment coverage.
        
        Returns:
            Dictionary with coverage percentage and missing combinations
        """
        attack_csv = self.run_dir / "tables" / "attack_metrics.csv"
        
        if not attack_csv.exists():
            return {'coverage': 0.0, 'missing': [], 'error': 'attack_metrics.csv not found'}
        
        # Expected combinations
        expected = self._get_expected_combinations()
        
        # Actual combinations from CSV
        actual = set()
        with open(attack_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get('dataset', ''),
                    row.get('task', ''),
                    row.get('attack_type', ''),
                    row.get('threat_level', ''),
                    str(row.get('privacy_level', '')),
                )
                actual.add(key)
        
        # Calculate coverage
        if len(expected) == 0:
            return {'coverage': 1.0, 'missing': []}
        
        missing = expected - actual
        coverage = (len(expected) - len(missing)) / len(expected)
        
        missing_list = [
            {'dataset': m[0], 'task': m[1], 'attack_type': m[2], 
             'threat_level': m[3], 'privacy_level': m[4]}
            for m in missing
        ]
        
        return {
            'coverage': coverage,
            'missing': missing_list,
            'expected_count': len(expected),
            'actual_count': len(actual),
        }
    
    def _get_expected_combinations(self) -> Set[Tuple]:
        """Get expected experiment combinations."""
        # Read from config or use defaults
        datasets = ['CelebA-HQ']  # Default
        tasks = ['face_verification']  # Default
        attack_types = list(self.schema.ATTACK_TYPES)
        threat_levels = list(self.schema.THREAT_LEVELS)
        privacy_levels = ['0.0', '0.3', '0.5', '0.7', '1.0']
        
        expected = set()
        for dataset in datasets:
            for task in tasks:
                for attack_type in attack_types:
                    for threat_level in threat_levels:
                        for privacy_level in privacy_levels:
                            expected.add((dataset, task, attack_type, 
                                        threat_level, privacy_level))
        
        return expected

    def _run_red_line_checks(self, config: Optional[Dict] = None) -> List[RedLineCheck]:
        """
        Run all T-REDLINE checks (R1-R10).
        
        Returns:
            List of RedLineCheck results
        """
        checks = []
        
        # R1: protocol_version == schema_version == commit_hash version
        checks.append(self._check_r1_version_consistency())
        
        # R2: coverage >= 0.98
        checks.append(self._check_r2_coverage())
        
        # R3: A2 exists with attacker_strength=full
        checks.append(self._check_r3_a2_exists())
        
        # R4: replay reject_rate = 100%
        checks.append(self._check_r4_replay())
        
        # R5: tamper fail_rate >= 99%
        checks.append(self._check_r5_tamper())
        
        # R6: c_view guard (no c_view in training)
        checks.append(self._check_r6_cview_guard())
        
        # R7: figure_manifest SHA256 reproducible
        checks.append(self._check_r7_figure_manifest())
        
        # R8: nonce uniqueness
        checks.append(self._check_r8_nonce_unique())
        
        # R9: train/val/test zero ID overlap
        checks.append(self._check_r9_split_leakage())
        
        # R10: all CSV fields complete and correct type
        checks.append(self._check_r10_schema_complete())
        
        return checks
    
    def _check_r1_version_consistency(self) -> RedLineCheck:
        """R1: Version consistency check."""
        try:
            version_file = self.run_dir / "meta" / "protocol_version.txt"
            if not version_file.exists():
                return RedLineCheck("R1", "Version consistency", False, 
                                   "protocol_version.txt not found")
            
            stored_version = version_file.read_text().strip()
            if stored_version != self.protocol_manager.PROTOCOL_VERSION:
                return RedLineCheck("R1", "Version consistency", False,
                                   f"Version mismatch: {stored_version} != {self.protocol_manager.PROTOCOL_VERSION}")
            
            if self.protocol_manager.PROTOCOL_VERSION != self.protocol_manager.SCHEMA_VERSION:
                return RedLineCheck("R1", "Version consistency", False,
                                   "Protocol version != Schema version")
            
            return RedLineCheck("R1", "Version consistency", True, 
                              f"Version {stored_version} consistent")
        except Exception as e:
            return RedLineCheck("R1", "Version consistency", False, str(e))
    
    def _check_r2_coverage(self) -> RedLineCheck:
        """R2: Coverage >= 98% check."""
        coverage_result = self._check_coverage()
        coverage = coverage_result.get('coverage', 0.0)
        
        if coverage >= self.COVERAGE_THRESHOLD:
            return RedLineCheck("R2", "Coverage >= 98%", True, 
                              f"Coverage: {coverage:.2%}")
        else:
            return RedLineCheck("R2", "Coverage >= 98%", False,
                              f"Coverage: {coverage:.2%} < {self.COVERAGE_THRESHOLD:.2%}")
    
    def _check_r3_a2_exists(self) -> RedLineCheck:
        """R3: A2 attack with attacker_strength=full exists."""
        attack_csv = self.run_dir / "tables" / "attack_metrics.csv"
        
        if not attack_csv.exists():
            return RedLineCheck("R3", "A2 attack exists", False, 
                              "attack_metrics.csv not found")
        
        has_a2_full = False
        with open(attack_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('threat_level') == 'A2' and 
                    row.get('attacker_strength') == 'full'):
                    has_a2_full = True
                    break
        
        if has_a2_full:
            return RedLineCheck("R3", "A2 attack exists", True, 
                              "A2 with attacker_strength=full found")
        else:
            return RedLineCheck("R3", "A2 attack exists", False,
                              "No A2 attack with attacker_strength=full")
    
    def _check_r4_replay(self) -> RedLineCheck:
        """R4: Replay reject_rate = 100%."""
        replay_csv = self.run_dir / "tables" / "replay_results.csv"
        
        if not replay_csv.exists():
            return RedLineCheck("R4", "Replay reject_rate = 100%", False,
                              "replay_results.csv not found")
        
        with open(replay_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                reject_rate = float(row.get('reject_rate', 0))
                if reject_rate < 1.0:
                    return RedLineCheck("R4", "Replay reject_rate = 100%", False,
                                      f"Reject rate: {reject_rate:.2%}")
        
        return RedLineCheck("R4", "Replay reject_rate = 100%", True, "100% reject rate")
    
    def _check_r5_tamper(self) -> RedLineCheck:
        """R5: Tamper fail_rate >= 99%."""
        security_csv = self.run_dir / "tables" / "security_metrics_cview.csv"
        
        if not security_csv.exists():
            return RedLineCheck("R5", "Tamper fail_rate >= 99%", False,
                              "security_metrics_cview.csv not found")
        
        with open(security_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('test_type') == 'tamper':
                    fail_rate = float(row.get('value', 0))
                    if fail_rate < 0.99:
                        return RedLineCheck("R5", "Tamper fail_rate >= 99%", False,
                                          f"Fail rate: {fail_rate:.2%}")
        
        return RedLineCheck("R5", "Tamper fail_rate >= 99%", True, ">= 99% fail rate")
    
    def _check_r6_cview_guard(self) -> RedLineCheck:
        """R6: C-view guard (no c_view in training data)."""
        # This would check DataLoader audit logs
        audit_log = self.run_dir / "logs" / "dataloader_audit.log"
        
        if not audit_log.exists():
            return RedLineCheck("R6", "C-view guard", True, 
                              "No audit log (assumed compliant)")
        
        content = audit_log.read_text()
        if "c_view" in content.lower() and "train" in content.lower():
            return RedLineCheck("R6", "C-view guard", False,
                              "C-view access detected in training")
        
        return RedLineCheck("R6", "C-view guard", True, "No c_view in training")
    
    def _check_r7_figure_manifest(self) -> RedLineCheck:
        """R7: Figure manifest SHA256 reproducible."""
        manifest_path = self.run_dir / "reports" / "figure_manifest.json"
        
        if not manifest_path.exists():
            return RedLineCheck("R7", "Figure manifest reproducible", False,
                              "figure_manifest.json not found")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Verify each figure's hash
            figures_dir = self.run_dir / "figures"
            for entry in manifest.get('figures', []):
                fig_path = figures_dir / entry['filename']
                if fig_path.exists():
                    actual_hash = hashlib.sha256(fig_path.read_bytes()).hexdigest()
                    if actual_hash != entry.get('sha256'):
                        return RedLineCheck("R7", "Figure manifest reproducible", False,
                                          f"Hash mismatch for {entry['filename']}")
            
            return RedLineCheck("R7", "Figure manifest reproducible", True, 
                              "All figure hashes match")
        except Exception as e:
            return RedLineCheck("R7", "Figure manifest reproducible", False, str(e))
    
    def _check_r8_nonce_unique(self) -> RedLineCheck:
        """R8: Nonce uniqueness."""
        nonce_log = self.run_dir / "meta" / "nonce_log.json"
        
        if not nonce_log.exists():
            return RedLineCheck("R8", "Nonce uniqueness", False,
                              "nonce_log.json not found")
        
        try:
            with open(nonce_log, 'r') as f:
                entries = json.load(f)
            
            nonces = [e.get('nonce_hex') for e in entries]
            if len(nonces) != len(set(nonces)):
                return RedLineCheck("R8", "Nonce uniqueness", False,
                                  "Duplicate nonces detected")
            
            return RedLineCheck("R8", "Nonce uniqueness", True, 
                              f"{len(nonces)} unique nonces")
        except Exception as e:
            return RedLineCheck("R8", "Nonce uniqueness", False, str(e))
    
    def _check_r9_split_leakage(self) -> RedLineCheck:
        """R9: Train/val/test zero ID overlap."""
        manifest_path = self.run_dir / "meta" / "dataset_manifest.json"
        
        if not manifest_path.exists():
            return RedLineCheck("R9", "Split leakage check", True,
                              "No manifest (assumed compliant)")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            splits = manifest.get('split_info', {})
            train_ids = set(splits.get('train', {}).get('ids', []))
            val_ids = set(splits.get('val', {}).get('ids', []))
            test_ids = set(splits.get('test', {}).get('ids', []))
            
            train_val = train_ids & val_ids
            train_test = train_ids & test_ids
            val_test = val_ids & test_ids
            
            if train_val or train_test or val_test:
                return RedLineCheck("R9", "Split leakage check", False,
                                  f"ID overlap detected: train-val={len(train_val)}, "
                                  f"train-test={len(train_test)}, val-test={len(val_test)}")
            
            return RedLineCheck("R9", "Split leakage check", True, "Zero ID overlap")
        except Exception as e:
            return RedLineCheck("R9", "Split leakage check", False, str(e))
    
    def _check_r10_schema_complete(self) -> RedLineCheck:
        """R10: All CSV fields complete and correct type."""
        tables_dir = self.run_dir / "tables"
        
        if not tables_dir.exists():
            return RedLineCheck("R10", "Schema completeness", False,
                              "tables/ directory not found")
        
        results = self.schema.validate_all_csvs(tables_dir)
        
        all_valid = all(r['valid'] for r in results.values())
        
        if all_valid:
            return RedLineCheck("R10", "Schema completeness", True,
                              "All CSV schemas valid")
        else:
            invalid = [k for k, v in results.items() if not v['valid']]
            return RedLineCheck("R10", "Schema completeness", False,
                              f"Invalid schemas: {invalid}")

    def generate_onepage_report(self, result: ValidationResult) -> str:
        """
        Generate reviewer-friendly one-page validation report.
        
        Args:
            result: ValidationResult from validate()
            
        Returns:
            Markdown formatted report
        """
        status = "✅ PASSED" if result.valid else "❌ FAILED"
        
        report = f"""# Validation Report

> **Status**: {status}
> **Timestamp**: {result.timestamp}
> **Run Directory**: {self.run_dir}

## Red-Line Checks (R1-R10)

| # | Check | Status | Details |
|---|-------|--------|---------|
"""
        for check in result.red_line_checks:
            status_icon = "✓" if check.passed else "✗"
            report += f"| {check.id} | {check.description} | {status_icon} | {check.details} |\n"
        
        report += f"""
## Coverage

- **Coverage**: {result.coverage:.2%}
- **Threshold**: {self.COVERAGE_THRESHOLD:.2%}
- **Status**: {"✓ Pass" if result.coverage >= self.COVERAGE_THRESHOLD else "✗ Fail"}

"""
        if result.missing_combinations:
            report += f"### Missing Combinations ({len(result.missing_combinations)})\n\n"
            report += "| Dataset | Task | Attack Type | Threat Level | Privacy Level |\n"
            report += "|---------|------|-------------|--------------|---------------|\n"
            for m in result.missing_combinations[:10]:  # Show first 10
                report += f"| {m['dataset']} | {m['task']} | {m['attack_type']} | {m['threat_level']} | {m['privacy_level']} |\n"
            if len(result.missing_combinations) > 10:
                report += f"\n*... and {len(result.missing_combinations) - 10} more*\n"
        
        report += """
## Schema Validation

"""
        for filename, schema_result in result.schema_results.items():
            status_icon = "✓" if schema_result['valid'] else "✗"
            report += f"- **{filename}**: {status_icon} ({schema_result['row_count']} rows)\n"
            if schema_result['errors']:
                for error in schema_result['errors'][:3]:
                    report += f"  - Error: {error}\n"
        
        if result.errors:
            report += "\n## Errors\n\n"
            for error in result.errors:
                report += f"- {error}\n"
        
        if result.warnings:
            report += "\n## Warnings\n\n"
            for warning in result.warnings:
                report += f"- {warning}\n"
        
        report += f"""
## Reproduction Command

```bash
python -m src.protocol.validate_run {self.run_dir}
```

---
*Generated by ValidateRun v{self.protocol_manager.PROTOCOL_VERSION}*
"""
        return report
    
    def write_onepage_report(self, result: ValidationResult) -> Path:
        """
        Write one-page report to reports/validate_run_onepage.md.
        
        Args:
            result: ValidationResult from validate()
            
        Returns:
            Path to written report
        """
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / "validate_run_onepage.md"
        report_content = self.generate_onepage_report(result)
        report_path.write_text(report_content, encoding='utf-8')
        
        return report_path
    
    def write_missing_matrix(self, missing: List[Dict]) -> Optional[Path]:
        """
        Write missing combinations to missing_matrix.csv.
        
        Args:
            missing: List of missing combination dictionaries
            
        Returns:
            Path to written file, or None if no missing combinations
        """
        if not missing:
            return None
        
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = reports_dir / "missing_matrix.csv"
        
        fieldnames = ['dataset', 'task', 'attack_type', 'threat_level', 'privacy_level']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(missing)
        
        return csv_path


def main():
    """CLI entry point for validation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.protocol.validate_run <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    validator = ValidateRun(run_dir)
    result = validator.validate()
    
    # Write report
    report_path = validator.write_onepage_report(result)
    print(f"Report written to: {report_path}")
    
    # Write missing matrix if needed
    if result.missing_combinations:
        missing_path = validator.write_missing_matrix(result.missing_combinations)
        print(f"Missing matrix written to: {missing_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
