"""
协议管理器

实现协议版本管理、快照生成和一致性验证。
对应 design.md §8.4 和 §12.1。

**验证: 属性 13 - 协议版本一致性**
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ProtocolError(Exception):
    """协议相关错误"""
    pass


class ProtocolVersionMismatchError(ProtocolError):
    """当 protocol_version != schema_version 时抛出"""
    pass


@dataclass
class ProtocolManager:
    """
    协议版本管理器
    
    管理协议版本控制、快照生成和一致性验证。
    版本号遵循语义化版本控制，必须与 schema_version 匹配。
    
    属性:
        run_dir: 实验运行目录的路径
        
    版本升级规则（按 §8.4 冻结）:
        - 主版本 (X.0.0): Schema 字段添加/删除
        - 次版本 (0.X.0): 协议逻辑变更
        - 补丁版本 (0.0.X): Bug 修复/文档澄清
    """
    
    run_dir: Path
    meta_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    
    # 冻结的版本号 - 必须与 design.md v2.1.1 匹配（类变量）
    PROTOCOL_VERSION: str = field(default="2.1.1", init=False)
    SCHEMA_VERSION: str = field(default="2.1.1", init=False)
    
    def __post_init__(self):
        """初始化目录路径"""
        if isinstance(self.run_dir, str):
            self.run_dir = Path(self.run_dir)
        self.meta_dir = self.run_dir / "meta"
        self.reports_dir = self.run_dir / "reports"
    
    def write_protocol_version(self) -> Path:
        """
        将协议版本写入 meta/protocol_version.txt
        
        Returns:
            写入文件的路径
        """
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        version_file = self.meta_dir / "protocol_version.txt"
        version_file.write_text(self.PROTOCOL_VERSION)
        return version_file

    def write_protocol_snapshot(self, config: Dict[str, Any]) -> Path:
        """
        将协议快照写入 reports/protocol_snapshot.md
        
        快照包含:
        - 协议和 schema 版本
        - Git 提交哈希
        - 时间戳
        - 冻结的配置参数
        - A2 强度契约摘要
        
        Args:
            config: 完整的实验配置
            
        Returns:
            写入的快照文件路径
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.reports_dir / "protocol_snapshot.md"
        
        snapshot_content = self._generate_snapshot(config)
        snapshot_path.write_text(snapshot_content, encoding='utf-8')
        return snapshot_path
    
    def _generate_snapshot(self, config: Dict[str, Any]) -> str:
        """生成协议快照 markdown 内容"""
        git_commit = self._get_git_commit()
        timestamp = datetime.now().isoformat()
        config_hash = self._compute_config_hash(config)
        
        snapshot = f"""# 协议快照

> **生成时间**: {timestamp}
> **协议版本**: {self.PROTOCOL_VERSION}
> **Schema 版本**: {self.SCHEMA_VERSION}
> **Git 提交**: {git_commit}
> **配置哈希**: {config_hash}

## 版本一致性

| 组件 | 版本 |
|-----------|---------|
| 协议 | {self.PROTOCOL_VERSION} |
| Schema | {self.SCHEMA_VERSION} |
| 设计文档 | v{self.PROTOCOL_VERSION} |

## 冻结参数

### 隐私级别 (§2.3)
```
λ ∈ {{0.0, 0.3, 0.5, 0.7, 1.0}}
```

### 威胁级别 (§2.4)
```
threat_level ∈ {{A0, A1, A2}}
```

### 训练模式 (§8.1)
```
training_mode ∈ {{P2P, P2Z, Z2Z, Mix2Z}}
```

### 攻击类型 (§6.1)
```
attack_type ∈ {{face_verification, attribute_inference, reconstruction, membership_inference, property_inference}}
```

## A2 强度契约 (§5.4)

### 攻击族
| 族 | 最小实例数 |
|--------|---------------|
| 重建 | 2 |
| 推断 | 3 |
| 优化 | 2 |

### 攻击预算
| 参数 | 值 |
|-----------|-------|
| 训练轮数 | 100 |
| 学习率搜索 | {{1e-4, 1e-3, 1e-2}} |
| 最大 GPU 时间 | ≤24h/族 |

### 最坏情况聚合
```
worst_case_attack_success = max(attack_success) over same (dataset, task, privacy_level, threat_level)
```

## 配置摘要

```yaml
{self._format_config_summary(config)}
```

---
*此快照已冻结。任何修改都需要升级 protocol_version。*
"""
        return snapshot
    
    def _get_git_commit(self) -> str:
        """获取当前 git 提交哈希"""
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
        """计算配置的 SHA256 哈希"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _format_config_summary(self, config: Dict[str, Any]) -> str:
        """格式化配置用于快照"""
        # 提取关键参数
        summary_keys = [
            'dataset', 'task', 'method', 'seed', 
            'privacy_levels', 'training_modes', 'attack_types'
        ]
        summary = {}
        for key in summary_keys:
            if key in config:
                summary[key] = config[key]
        
        # 格式化为类 YAML 字符串
        lines = []
        for key, value in summary.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        return '\n'.join(lines) if lines else "# 未提供配置"

    def validate_consistency(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置和协议快照的一致性
        
        检查:
        1. protocol_version == schema_version
        2. protocol_snapshot.md 存在且与配置匹配（如果提供）
        3. meta/protocol_version.txt 与 PROTOCOL_VERSION 匹配
        
        Args:
            config: 可选的配置，用于与快照验证
            
        Returns:
            如果所有一致性检查通过则返回 True
            
        Raises:
            ProtocolVersionMismatchError: 如果版本不匹配
            ProtocolError: 如果快照缺失或不一致
        """
        # 检查 1: protocol_version == schema_version
        if self.PROTOCOL_VERSION != self.SCHEMA_VERSION:
            raise ProtocolVersionMismatchError(
                f"协议版本 ({self.PROTOCOL_VERSION}) != "
                f"Schema 版本 ({self.SCHEMA_VERSION})"
            )
        
        # 检查 2: protocol_version.txt 存在且匹配
        version_file = self.meta_dir / "protocol_version.txt"
        if version_file.exists():
            stored_version = version_file.read_text().strip()
            if stored_version != self.PROTOCOL_VERSION:
                raise ProtocolVersionMismatchError(
                    f"存储的协议版本 ({stored_version}) != "
                    f"当前版本 ({self.PROTOCOL_VERSION})"
                )
        
        # 检查 3: protocol_snapshot.md 存在
        snapshot_path = self.reports_dir / "protocol_snapshot.md"
        if not snapshot_path.exists():
            raise ProtocolError(
                f"缺少 protocol_snapshot.md: {snapshot_path}"
            )
        
        # 检查 4: 如果提供了配置，验证哈希匹配
        if config is not None:
            snapshot_content = snapshot_path.read_text(encoding='utf-8')
            current_hash = self._compute_config_hash(config)
            
            # 从快照中提取哈希
            if f"Config Hash**: {current_hash}" not in snapshot_content:
                # 哈希不匹配 - 配置已更改
                raise ProtocolError(
                    f"配置哈希不匹配。"
                    f"当前: {current_hash}，"
                    f"快照可能已过期。"
                )
        
        return True
    
    def get_version_info(self) -> Dict[str, str]:
        """
        获取版本信息字典
        
        Returns:
            包含 protocol_version、schema_version、git_commit 的字典
        """
        return {
            "protocol_version": self.PROTOCOL_VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "git_commit": self._get_git_commit(),
        }
    
    def ensure_directories(self) -> None:
        """创建所有必需的运行目录"""
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
        确定变更所需的版本升级类型
        
        Args:
            change_type: 'schema_field'、'protocol_logic' 或 'bugfix' 之一
            
        Returns:
            所需版本升级的描述
        """
        bump_rules = {
            'schema_field': '主版本 (X.0.0) - Schema 字段添加/删除',
            'protocol_logic': '次版本 (0.X.0) - 协议逻辑变更',
            'bugfix': '补丁版本 (0.0.X) - Bug 修复/文档澄清',
        }
        return bump_rules.get(change_type, '未知变更类型')
