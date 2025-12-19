#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal-VSE-PC 测试运行脚本

用法:
    python scripts/run_tests.py              # 运行所有测试
    python scripts/run_tests.py --quick      # 快速测试（核心模块）
    python scripts/run_tests.py --module X   # 运行指定模块测试
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_single_test(test_file: str) -> bool:
    """运行单个测试文件"""
    print(f"\n{'='*70}")
    print(f"运行测试: {test_file}")
    print('='*70)
    
    try:
        # 动态导入并运行
        test_path = project_root / "tests" / test_file
        if not test_path.exists():
            print(f"✗ 测试文件不存在: {test_path}")
            return False
        
        # 使用 exec 运行测试
        with open(test_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 创建独立的命名空间
        namespace = {'__name__': '__main__', '__file__': str(test_path)}
        exec(code, namespace)
        
        print(f"✓ {test_file} 测试通过")
        return True
        
    except Exception as e:
        print(f"✗ {test_file} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_tests():
    """运行快速测试（核心模块）"""
    print("\n" + "="*70)
    print("Causal-VSE-PC 快速测试")
    print("="*70)
    
    quick_tests = [
        "test_encryption.py",
        "test_protocol_manager.py",
        "test_replay_cache.py",
    ]
    
    passed = 0
    failed = 0
    
    for test_file in quick_tests:
        if run_single_test(test_file):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print(f"快速测试结果: {passed} 通过, {failed} 失败")
    print("="*70)
    
    return failed == 0


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Causal-VSE-PC 完整测试套件")
    print("="*70)
    
    tests_dir = project_root / "tests"
    test_files = sorted([
        f.name for f in tests_dir.glob("test_*.py")
        if f.name != "__init__.py"
    ])
    
    print(f"\n发现 {len(test_files)} 个测试文件:")
    for f in test_files:
        print(f"  - {f}")
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_file in test_files:
        if run_single_test(test_file):
            passed += 1
        else:
            failed += 1
            failed_tests.append(test_file)
    
    print("\n" + "="*70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    if failed_tests:
        print(f"失败的测试: {', '.join(failed_tests)}")
    print("="*70)
    
    return failed == 0


def run_module_test(module_name: str):
    """运行指定模块的测试"""
    test_file = f"test_{module_name}.py"
    return run_single_test(test_file)


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    required = ['torch', 'numpy', 'scipy', 'cryptography']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (缺失)")
            missing.append(pkg)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✓ 所有依赖已安装\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Causal-VSE-PC 测试运行器')
    parser.add_argument('--quick', action='store_true', help='运行快速测试')
    parser.add_argument('--module', type=str, help='运行指定模块测试')
    parser.add_argument('--check', action='store_true', help='仅检查依赖')
    parser.add_argument('--list', action='store_true', help='列出所有测试')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check:
        check_dependencies()
        return
    
    # 列出测试
    if args.list:
        tests_dir = project_root / "tests"
        test_files = sorted([
            f.name for f in tests_dir.glob("test_*.py")
            if f.name != "__init__.py"
        ])
        print(f"可用测试 ({len(test_files)} 个):")
        for f in test_files:
            print(f"  - {f}")
        return
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖后再运行测试")
        sys.exit(1)
    
    # 运行测试
    if args.module:
        success = run_module_test(args.module)
    elif args.quick:
        success = run_quick_tests()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
