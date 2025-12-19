"""
顶级期刊实验套件结果模式定义

定义并验证 utility_metrics.csv、attack_metrics.csv 和 security_metrics_cview.csv 的 CSV 模式。

对应 design.md §10.1。

**验证: 属性 2 - CSV Schema 合规性**
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class SchemaValidationError(Exception):
    """模式验证错误。"""
    pass


class FieldType(Enum):
    """支持的字段类型。"""
    STRING = "str"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


@dataclass
class FieldDefinition:
    """CSV 字段定义。"""
    name: str
    field_type: FieldType
    required: bool = True
    allowed_values: Optional[Set[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        验证字段值。
        
        返回:
            (是否有效, 错误消息) 元组
        """
        # Check required
        if value is None or (isinstance(value, str) and value.strip() == ''):
            if self.required:
                return False, f"Required field '{self.name}' is empty"
            return True, None
        
        # Type validation
        try:
            if self.field_type == FieldType.STRING:
                str_value = str(value)
            elif self.field_type == FieldType.FLOAT:
                float_value = float(value)
                if self.min_value is not None and float_value < self.min_value:
                    return False, f"Field '{self.name}' value {float_value} < min {self.min_value}"
                if self.max_value is not None and float_value > self.max_value:
                    return False, f"Field '{self.name}' value {float_value} > max {self.max_value}"
            elif self.field_type == FieldType.INT:
                int_value = int(float(value))
            elif self.field_type == FieldType.BOOL:
                if str(value).lower() not in ('true', 'false', '1', '0'):
                    return False, f"Field '{self.name}' invalid bool: {value}"
        except (ValueError, TypeError) as e:
            return False, f"Field '{self.name}' type error: {e}"
        
        # Enum validation
        if self.allowed_values is not None:
            str_value = str(value)
            if str_value not in self.allowed_values:
                return False, f"Field '{self.name}' value '{str_value}' not in {self.allowed_values}"
        
        return True, None


@dataclass
class ResultsSchema:
    """
    结果模式定义（按 §10.1 冻结）。
    
    定义以下文件的必需字段:
    - utility_metrics.csv
    - attack_metrics.csv
    - security_metrics_cview.csv
    """
    
    # 冻结的枚举值
    PRIVACY_LEVELS: Set[str] = field(default_factory=lambda: {'0.0', '0.3', '0.5', '0.7', '1.0'})
    THREAT_LEVELS: Set[str] = field(default_factory=lambda: {'A0', 'A1', 'A2'})
    TRAINING_MODES: Set[str] = field(default_factory=lambda: {'P2P', 'P2Z', 'Z2Z', 'Mix2Z'})
    ATTACK_TYPES: Set[str] = field(default_factory=lambda: {
        'face_verification', 'attribute_inference', 'reconstruction',
        'membership_inference', 'property_inference'
    })
    ATTACKER_STRENGTHS: Set[str] = field(default_factory=lambda: {'lite', 'full'})
    STATUS_VALUES: Set[str] = field(default_factory=lambda: {'success', 'failed', 'skipped'})
    
    # 模式版本（必须与 ProtocolManager 匹配）
    SCHEMA_VERSION: str = "2.1.1"
    
    def __post_init__(self):
        """初始化字段定义。"""
        self._init_utility_fields()
        self._init_attack_fields()
        self._init_security_fields()
    
    def _init_utility_fields(self):
        """初始化 utility_metrics.csv 字段定义。"""
        self.UTILITY_FIELDS: List[FieldDefinition] = [
            FieldDefinition("dataset", FieldType.STRING, required=True),
            FieldDefinition("task", FieldType.STRING, required=True),
            FieldDefinition("method", FieldType.STRING, required=True),
            FieldDefinition("training_mode", FieldType.STRING, required=True, 
                          allowed_values=self.TRAINING_MODES),
            FieldDefinition("privacy_level", FieldType.FLOAT, required=True,
                          min_value=0.0, max_value=1.0),
            FieldDefinition("seed", FieldType.INT, required=True),
            FieldDefinition("metric_name", FieldType.STRING, required=True),
            FieldDefinition("metric_value", FieldType.FLOAT, required=True),
            FieldDefinition("relative_to", FieldType.STRING, required=True),
            FieldDefinition("relative_performance", FieldType.FLOAT, required=True),
            FieldDefinition("ci_low", FieldType.FLOAT, required=True),
            FieldDefinition("ci_high", FieldType.FLOAT, required=True),
            FieldDefinition("stat_method", FieldType.STRING, required=True),
            FieldDefinition("n_boot", FieldType.INT, required=True),
            FieldDefinition("family_id", FieldType.STRING, required=True),
            FieldDefinition("alpha", FieldType.FLOAT, required=True),
        ]
    
    def _init_attack_fields(self):
        """初始化 attack_metrics.csv 字段定义。"""
        self.ATTACK_FIELDS: List[FieldDefinition] = [
            FieldDefinition("dataset", FieldType.STRING, required=True),
            FieldDefinition("task", FieldType.STRING, required=True),
            FieldDefinition("method", FieldType.STRING, required=True),
            FieldDefinition("training_mode", FieldType.STRING, required=True,
                          allowed_values=self.TRAINING_MODES),
            FieldDefinition("attack_type", FieldType.STRING, required=True,
                          allowed_values=self.ATTACK_TYPES),
            FieldDefinition("threat_level", FieldType.STRING, required=True,
                          allowed_values=self.THREAT_LEVELS),
            FieldDefinition("privacy_level", FieldType.FLOAT, required=True,
                          min_value=0.0, max_value=1.0),
            FieldDefinition("seed", FieldType.INT, required=True),
            FieldDefinition("attack_success", FieldType.FLOAT, required=True,
                          min_value=0.0, max_value=1.0),
            FieldDefinition("metric_name", FieldType.STRING, required=True),
            FieldDefinition("metric_value", FieldType.FLOAT, required=True),
            FieldDefinition("ci_low", FieldType.FLOAT, required=True),
            FieldDefinition("ci_high", FieldType.FLOAT, required=True),
            FieldDefinition("attacker_strength", FieldType.STRING, required=True,
                          allowed_values=self.ATTACKER_STRENGTHS),
            FieldDefinition("degrade_reason", FieldType.STRING, required=False),  # 如果是 lite 则必需
            FieldDefinition("status", FieldType.STRING, required=True,
                          allowed_values=self.STATUS_VALUES),
            FieldDefinition("stat_method", FieldType.STRING, required=True),
            FieldDefinition("n_boot", FieldType.INT, required=True),
            FieldDefinition("family_id", FieldType.STRING, required=True),
            FieldDefinition("alpha", FieldType.FLOAT, required=True),
        ]
    
    def _init_security_fields(self):
        """初始化 security_metrics_cview.csv 字段定义。"""
        self.SECURITY_FIELDS: List[FieldDefinition] = [
            FieldDefinition("test_type", FieldType.STRING, required=True),
            FieldDefinition("test_name", FieldType.STRING, required=True),
            FieldDefinition("result", FieldType.STRING, required=True),
            FieldDefinition("value", FieldType.FLOAT, required=True),
            FieldDefinition("threshold", FieldType.FLOAT, required=False),
            FieldDefinition("pass", FieldType.BOOL, required=True),
            FieldDefinition("n_samples", FieldType.INT, required=True),
            FieldDefinition("details", FieldType.STRING, required=False),
        ]

    def validate_csv(self, csv_path: Path, schema_type: str) -> Dict[str, Any]:
        """
        根据模式验证 CSV 文件。
        
        参数:
            csv_path: CSV 文件路径
            schema_type: 'utility'、'attack' 或 'security' 之一
            
        返回:
            验证结果字典，包含:
            - valid: 布尔值
            - errors: 错误消息列表
            - warnings: 警告消息列表
            - row_count: 验证的行数
            
        异常:
            SchemaValidationError: 如果 schema_type 无效
        """
        schema_map = {
            'utility': self.UTILITY_FIELDS,
            'attack': self.ATTACK_FIELDS,
            'security': self.SECURITY_FIELDS,
        }
        
        if schema_type not in schema_map:
            raise SchemaValidationError(f"Unknown schema type: {schema_type}")
        
        fields = schema_map[schema_type]
        field_names = {f.name for f in fields}
        field_map = {f.name: f for f in fields}
        
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'row_count': 0,
        }
        
        if not csv_path.exists():
            result['valid'] = False
            result['errors'].append(f"CSV file not found: {csv_path}")
            return result
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # 检查表头字段
            if reader.fieldnames is None:
                result['valid'] = False
                result['errors'].append("CSV 没有表头")
                return result
            
            csv_fields = set(reader.fieldnames)
            
            # 检查缺失的必需字段
            missing_fields = field_names - csv_fields
            required_missing = [f for f in missing_fields 
                              if field_map.get(f, FieldDefinition(f, FieldType.STRING, False)).required]
            if required_missing:
                result['valid'] = False
                result['errors'].append(f"缺少必需字段: {required_missing}")
            
            # 检查额外字段（仅警告）
            extra_fields = csv_fields - field_names
            if extra_fields:
                result['warnings'].append(f"额外字段（已忽略）: {extra_fields}")
            
            # 验证每一行
            for row_idx, row in enumerate(reader, start=2):  # 从 2 开始（表头是第 1 行）
                result['row_count'] += 1
                
                for field_def in fields:
                    if field_def.name not in row:
                        continue
                    
                    value = row[field_def.name]
                    is_valid, error_msg = field_def.validate(value)
                    
                    if not is_valid:
                        result['valid'] = False
                        result['errors'].append(f"Row {row_idx}: {error_msg}")
                
                # Special validation: lite mode must have degrade_reason
                if schema_type == 'attack':
                    if row.get('attacker_strength') == 'lite':
                        if not row.get('degrade_reason'):
                            result['valid'] = False
                            result['errors'].append(
                                f"Row {row_idx}: attacker_strength=lite requires degrade_reason"
                            )
        
        return result
    
    def get_required_fields(self, schema_type: str) -> List[str]:
        """获取指定模式类型的必需字段名列表。"""
        schema_map = {
            'utility': self.UTILITY_FIELDS,
            'attack': self.ATTACK_FIELDS,
            'security': self.SECURITY_FIELDS,
        }
        fields = schema_map.get(schema_type, [])
        return [f.name for f in fields if f.required]
    
    def get_all_fields(self, schema_type: str) -> List[str]:
        """获取指定模式类型的所有字段名列表。"""
        schema_map = {
            'utility': self.UTILITY_FIELDS,
            'attack': self.ATTACK_FIELDS,
            'security': self.SECURITY_FIELDS,
        }
        fields = schema_map.get(schema_type, [])
        return [f.name for f in fields]
    
    def validate_all_csvs(self, tables_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        验证 tables 目录中的所有 CSV 文件。
        
        参数:
            tables_dir: tables/ 目录路径
            
        返回:
            文件名到验证结果的字典映射
        """
        results = {}
        
        csv_schema_map = {
            'utility_metrics.csv': 'utility',
            'attack_metrics.csv': 'attack',
            'security_metrics_cview.csv': 'security',
        }
        
        for filename, schema_type in csv_schema_map.items():
            csv_path = tables_dir / filename
            if csv_path.exists():
                results[filename] = self.validate_csv(csv_path, schema_type)
            else:
                results[filename] = {
                    'valid': False,
                    'errors': [f"File not found: {csv_path}"],
                    'warnings': [],
                    'row_count': 0,
                }
        
        return results
