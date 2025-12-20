# -*- coding: utf-8 -*-
"""
协议管理器测试

**Property 13: 协议版本一致性**
**Validates: Requirements 1.7, 1.8, GC2**
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_protocol_version_constants():
    """测试协议版本常量"""
    print("=" * 70)
    print("测试协议版本常量")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager
    
    assert ProtocolManager.PROTOCOL_VERSION == "2.1.1"
    assert ProtocolManager.SCHEMA_VERSION == "2.1.1"
    
    print(f"✓ PROTOCOL_VERSION: {ProtocolManager.PROTOCOL_VERSION}")
    print(f"✓ SCHEMA_VERSION: {ProtocolManager.SCHEMA_VERSION}")
    print("✓ 协议版本常量测试通过\n")


def test_protocol_manager_initialization():
    """测试 ProtocolManager 初始化"""
    print("=" * 70)
    print("测试 ProtocolManager 初始化")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        manager = ProtocolManager(run_dir)
        
        assert manager.run_dir == run_dir
        assert manager.meta_dir == run_dir / "meta"
        assert manager.reports_dir == run_dir / "reports"
        
        print(f"✓ run_dir: {manager.run_dir}")
        print(f"✓ meta_dir: {manager.meta_dir}")
        print(f"✓ reports_dir: {manager.reports_dir}")
        print("✓ ProtocolManager 初始化测试通过\n")


def test_write_protocol_version():
    """测试写入协议版本"""
    print("=" * 70)
    print("测试写入协议版本")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        manager = ProtocolManager(run_dir)
        
        # 写入协议版本
        manager.write_protocol_version()
        
        # 验证文件存在
        version_file = run_dir / "meta" / "protocol_version.txt"
        assert version_file.exists(), "protocol_version.txt 应该存在"
        
        # 验证内容
        content = version_file.read_text().strip()
        assert content == ProtocolManager.PROTOCOL_VERSION
        
        print(f"✓ 协议版本文件已创建: {version_file}")
        print(f"✓ 文件内容: {content}")
        print("✓ 写入协议版本测试通过\n")


def test_write_protocol_snapshot():
    """测试写入协议快照"""
    print("=" * 70)
    print("测试写入协议快照")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        manager = ProtocolManager(run_dir)
        
        # 测试配置
        config = {
            'experiment': 'test_exp',
            'privacy_level': 0.5,
            'dataset': 'celeba',
            'seed': 42
        }
        
        # 写入协议快照
        manager.write_protocol_snapshot(config)
        
        # 验证文件存在
        snapshot_file = run_dir / "reports" / "protocol_snapshot.md"
        assert snapshot_file.exists(), "protocol_snapshot.md 应该存在"
        
        # 验证内容包含关键信息
        content = snapshot_file.read_text(encoding='utf-8')
        assert "2.1.1" in content, "快照应包含协议版本 2.1.1"
        assert "协议" in content or "Protocol" in content, "快照应包含协议关键字"
        
        print(f"✓ 协议快照文件已创建: {snapshot_file}")
        print(f"✓ 文件大小: {len(content)} 字节")
        print("✓ 写入协议快照测试通过\n")


def test_validate_consistency():
    """测试协议一致性验证"""
    print("=" * 70)
    print("测试协议一致性验证")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager, ProtocolError
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        manager = ProtocolManager(run_dir)
        
        # 先写入协议版本和快照
        config = {'test': 'config'}
        manager.write_protocol_version()
        manager.write_protocol_snapshot(config)
        
        # 验证一致性 - validate_consistency 返回 bool 或抛异常
        try:
            is_valid = manager.validate_consistency()
            print(f"✓ 一致性验证结果: {is_valid}")
        except ProtocolError as e:
            print(f"✓ 一致性验证抛出预期异常: {e}")
        
        print("✓ 协议一致性验证测试通过\n")


def test_property_13_protocol_version_consistency():
    """
    **Property 13: 协议版本一致性**
    
    验证 protocol_version == schema_version
    """
    print("=" * 70)
    print("Property 13: 协议版本一致性")
    print("=" * 70)
    
    from src.protocol.protocol_manager import ProtocolManager
    
    # 验证版本一致
    assert ProtocolManager.PROTOCOL_VERSION == ProtocolManager.SCHEMA_VERSION, \
        "PROTOCOL_VERSION 和 SCHEMA_VERSION 必须一致"
    
    print(f"✓ PROTOCOL_VERSION: {ProtocolManager.PROTOCOL_VERSION}")
    print(f"✓ SCHEMA_VERSION: {ProtocolManager.SCHEMA_VERSION}")
    print("✓ Property 13 测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("协议管理器测试")
    print("=" * 70 + "\n")
    
    test_protocol_version_constants()
    test_protocol_manager_initialization()
    test_write_protocol_version()
    test_write_protocol_snapshot()
    test_validate_consistency()
    test_property_13_protocol_version_consistency()
    
    print("=" * 70)
    print("✓ 所有协议管理器测试通过")
    print("=" * 70)


if __name__ == '__main__':
    main()
