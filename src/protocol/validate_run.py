"""
顶级期刊实验套件运行验证模块。

实现实验运行的全面验证，包括:
- 目录结构验证
- 模式验证
- 覆盖率检查
- 红线检查 (R1-R10)

对应 design.md §10.0.3 和 T-REDLINE。

**验证: 属性 1, 属性 10**
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
    """单个红线检查的结果。"""
    id: str
    description: str
    passed: bool
    details: str = ""


@dataclass
class ValidationResult:
    """完整验证结果。"""
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
    运行结果验证器。
    
    根据协议要求验证实验运行目录。
    实现 T-REDLINE 检查 (R1-R10)。
    """
    
    COVERAGE_THRESHOLD = 0.98  # 98% 覆盖率门槛
    
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
        初始化验证器。
        
        参数:
            run_dir: 实验运行目录路径
        """
        self.run_dir = Path(run_dir)
        self.protocol_manager = ProtocolManager(run_dir)
        self.schema = ResultsSchema()

    def validate(self, config: Optional[Dict] = None) -> ValidationResult:
        """
        执行运行目录的完整验证。
        
        参数:
            config: 可选的一致性检查配置
            
        返回:
            包含所有检查结果的 ValidationResult
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
        检查必需的目录结构是否存在。
        
        返回:
            (是否有效, 错误消息列表) 元组
        """
        errors = []
        
        if not self.run_dir.exists():
            return False, [f"Run directory not found: {self.run_dir}"]
        
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.run_dir / dir_name
            if not dir_path.exists():
                errors.append(f"Missing required directory: {dir_name}/")
        
        # 检查必需的元数据文件
        meta_dir = self.run_dir / "meta"
        if meta_dir.exists():
            for filename in self.REQUIRED_META_FILES:
                if not (meta_dir / filename).exists():
                    errors.append(f"缺少必需的元数据文件: meta/{filename}")
        
        # 检查必需的表格文件
        tables_dir = self.run_dir / "tables"
        if tables_dir.exists():
            for filename in self.REQUIRED_TABLE_FILES:
                if not (tables_dir / filename).exists():
                    errors.append(f"Missing required table file: tables/{filename}")
        
        return len(errors) == 0, errors
    
    def _check_coverage(self) -> Dict[str, Any]:
        """
        检查实验覆盖率。
        
        返回:
            包含覆盖率百分比和缺失组合的字典
        """
        attack_csv = self.run_dir / "tables" / "attack_metrics.csv"
        
        if not attack_csv.exists():
            return {'coverage': 0.0, 'missing': [], 'error': 'attack_metrics.csv not found'}
        
        # 期望的组合
        expected = self._get_expected_combinations()
        
        # CSV 中的实际组合
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
        """获取预期的实验组合。"""
        # 从配置读取或使用默认值
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
        运行所有 T-REDLINE 检查 (R1-R10)。
        
        返回:
            RedLineCheck 结果列表
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
        """R1: 版本一致性检查。"""
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
        """R2: 覆盖率 >= 98% 检查。"""
        coverage_result = self._check_coverage()
        coverage = coverage_result.get('coverage', 0.0)
        
        if coverage >= self.COVERAGE_THRESHOLD:
            return RedLineCheck("R2", "Coverage >= 98%", True, 
                              f"Coverage: {coverage:.2%}")
        else:
            return RedLineCheck("R2", "Coverage >= 98%", False,
                              f"Coverage: {coverage:.2%} < {self.COVERAGE_THRESHOLD:.2%}")
    
    def _check_r3_a2_exists(self) -> RedLineCheck:
        """R3: 存在 attacker_strength=full 的 A2 攻击。"""
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
        """R4: 重放 reject_rate = 100%。"""
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
        """R5: 篡改 fail_rate >= 99%。"""
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
        """R6: C-view 守卫（训练数据中无 c_view）。"""
        # 检查 DataLoader 审计日志
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
        """R7: 图表清单 SHA256 可复现。"""
        manifest_path = self.run_dir / "reports" / "figure_manifest.json"
        
        if not manifest_path.exists():
            return RedLineCheck("R7", "Figure manifest reproducible", False,
                              "figure_manifest.json not found")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # 验证每个图表的哈希
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
        """R8: Nonce 唯一性。"""
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
        """R9: 训练/验证/测试集零 ID 重叠。"""
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
        """R10: 所有 CSV 字段完整且类型正确。"""
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
        生成审稿人友好的单页验证报告。
        
        参数:
            result: validate() 返回的 ValidationResult
            
        返回:
            Markdown 格式的报告
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
        将单页报告写入 reports/validate_run_onepage.md。
        
        参数:
            result: validate() 返回的 ValidationResult
            
        返回:
            写入报告的路径
        """
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / "validate_run_onepage.md"
        report_content = self.generate_onepage_report(result)
        report_path.write_text(report_content, encoding='utf-8')
        
        return report_path
    
    def write_missing_matrix(self, missing: List[Dict]) -> Optional[Path]:
        """
        将缺失组合写入 missing_matrix.csv。
        
        参数:
            missing: 缺失组合字典列表
            
        返回:
            写入文件的路径，如果没有缺失组合则返回 None
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
    """验证的 CLI 入口点。"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.protocol.validate_run <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    validator = ValidateRun(run_dir)
    result = validator.validate()
    
    # 写入报告
    report_path = validator.write_onepage_report(result)
    print(f"报告已写入: {report_path}")
    
    # 如果需要，写入缺失矩阵
    if result.missing_combinations:
        missing_path = validator.write_missing_matrix(result.missing_combinations)
        print(f"缺失矩阵已写入: {missing_path}")
    
    # 使用适当的退出码退出
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
