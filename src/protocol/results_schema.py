"""
Results Schema for Top-Journal Experiment Suite.

Defines and validates CSV schema for utility_metrics.csv, attack_metrics.csv,
and security_metrics_cview.csv.

Corresponds to design.md §10.1.

**Validates: Property 2 - CSV Schema 合规性**
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class SchemaValidationError(Exception):
    """Schema validation errors."""
    pass


class FieldType(Enum):
    """Supported field types."""
    STRING = "str"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


@dataclass
class FieldDefinition:
    """Definition of a CSV field."""
    name: str
    field_type: FieldType
    required: bool = True
    allowed_values: Optional[Set[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a field value.
        
        Returns:
            Tuple of (is_valid, error_message)
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
    Results Schema definition (frozen per §10.1).
    
    Defines required fields for:
    - utility_metrics.csv
    - attack_metrics.csv
    - security_metrics_cview.csv
    """
    
    # Frozen enums
    PRIVACY_LEVELS: Set[str] = field(default_factory=lambda: {'0.0', '0.3', '0.5', '0.7', '1.0'})
    THREAT_LEVELS: Set[str] = field(default_factory=lambda: {'A0', 'A1', 'A2'})
    TRAINING_MODES: Set[str] = field(default_factory=lambda: {'P2P', 'P2Z', 'Z2Z', 'Mix2Z'})
    ATTACK_TYPES: Set[str] = field(default_factory=lambda: {
        'face_verification', 'attribute_inference', 'reconstruction',
        'membership_inference', 'property_inference'
    })
    ATTACKER_STRENGTHS: Set[str] = field(default_factory=lambda: {'lite', 'full'})
    STATUS_VALUES: Set[str] = field(default_factory=lambda: {'success', 'failed', 'skipped'})
    
    # Schema version (must match ProtocolManager)
    SCHEMA_VERSION: str = "2.1.1"
    
    def __post_init__(self):
        """Initialize field definitions."""
        self._init_utility_fields()
        self._init_attack_fields()
        self._init_security_fields()
    
    def _init_utility_fields(self):
        """Initialize utility_metrics.csv field definitions."""
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
        """Initialize attack_metrics.csv field definitions."""
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
            FieldDefinition("degrade_reason", FieldType.STRING, required=False),  # Required if lite
            FieldDefinition("status", FieldType.STRING, required=True,
                          allowed_values=self.STATUS_VALUES),
            FieldDefinition("stat_method", FieldType.STRING, required=True),
            FieldDefinition("n_boot", FieldType.INT, required=True),
            FieldDefinition("family_id", FieldType.STRING, required=True),
            FieldDefinition("alpha", FieldType.FLOAT, required=True),
        ]
    
    def _init_security_fields(self):
        """Initialize security_metrics_cview.csv field definitions."""
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
        Validate a CSV file against its schema.
        
        Args:
            csv_path: Path to CSV file
            schema_type: One of 'utility', 'attack', 'security'
            
        Returns:
            Validation result dictionary with:
            - valid: bool
            - errors: List of error messages
            - warnings: List of warning messages
            - row_count: Number of rows validated
            
        Raises:
            SchemaValidationError: If schema_type is invalid
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
            
            # Check header fields
            if reader.fieldnames is None:
                result['valid'] = False
                result['errors'].append("CSV has no header")
                return result
            
            csv_fields = set(reader.fieldnames)
            
            # Check for missing required fields
            missing_fields = field_names - csv_fields
            required_missing = [f for f in missing_fields 
                              if field_map.get(f, FieldDefinition(f, FieldType.STRING, False)).required]
            if required_missing:
                result['valid'] = False
                result['errors'].append(f"Missing required fields: {required_missing}")
            
            # Check for extra fields (warning only)
            extra_fields = csv_fields - field_names
            if extra_fields:
                result['warnings'].append(f"Extra fields (ignored): {extra_fields}")
            
            # Validate each row
            for row_idx, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
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
        """Get list of required field names for a schema type."""
        schema_map = {
            'utility': self.UTILITY_FIELDS,
            'attack': self.ATTACK_FIELDS,
            'security': self.SECURITY_FIELDS,
        }
        fields = schema_map.get(schema_type, [])
        return [f.name for f in fields if f.required]
    
    def get_all_fields(self, schema_type: str) -> List[str]:
        """Get list of all field names for a schema type."""
        schema_map = {
            'utility': self.UTILITY_FIELDS,
            'attack': self.ATTACK_FIELDS,
            'security': self.SECURITY_FIELDS,
        }
        fields = schema_map.get(schema_type, [])
        return [f.name for f in fields]
    
    def validate_all_csvs(self, tables_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Validate all CSV files in a tables directory.
        
        Args:
            tables_dir: Path to tables/ directory
            
        Returns:
            Dictionary mapping filename to validation result
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
