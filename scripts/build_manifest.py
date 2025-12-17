#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manifest 构建脚本

命令行接口，用于扫描数据集目录并生成 manifest.jsonl 文件。

使用方法:
    # 扫描所有数据集
    python scripts/build_manifest.py --data-root data --output manifest.jsonl
    
    # 只扫描特定数据集
    python scripts/build_manifest.py --data-root data --output manifest.jsonl --datasets celebahq fairface
    
    # 只扫描特定分割
    python scripts/build_manifest.py --data-root data --output manifest.jsonl --splits train val

Requirements: 2.1, 8.1
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.manifest_builder import ManifestBuilder, build_manifest


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='构建数据集 Manifest 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 扫描所有数据集和分割
    python scripts/build_manifest.py --data-root data --output manifest.jsonl
    
    # 只扫描 CelebA-HQ 和 FairFace
    python scripts/build_manifest.py -d data -o manifest.jsonl --datasets celebahq fairface
    
    # 只扫描训练集和验证集
    python scripts/build_manifest.py -d data -o manifest.jsonl --splits train val
    
    # 指定掩码输出目录
    python scripts/build_manifest.py -d data -o manifest.jsonl --mask-dir masks
        """
    )
    
    parser.add_argument(
        '-d', '--data-root',
        type=str,
        default='data',
        help='数据集根目录 (默认: data)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='manifest.jsonl',
        help='输出 manifest 文件路径 (默认: manifest.jsonl)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['celeba', 'celebahq', 'fairface', 'openimages'],
        default=None,
        help='要扫描的数据集列表 (默认: 全部)'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'val', 'test'],
        default=None,
        help='要扫描的分割列表 (默认: 全部)'
    )
    
    parser.add_argument(
        '--mask-dir',
        type=str,
        default=None,
        help='掩码输出目录 (可选)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("Manifest 构建器")
    print("=" * 60)
    
    # 显示配置
    print(f"\n配置:")
    print(f"  数据根目录: {args.data_root}")
    print(f"  输出文件: {args.output}")
    print(f"  数据集: {args.datasets or '全部'}")
    print(f"  分割: {args.splits or '全部'}")
    if args.mask_dir:
        print(f"  掩码目录: {args.mask_dir}")
    
    # 检查数据目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"\n❌ 错误: 数据目录不存在: {data_root}")
        sys.exit(1)
    
    # 构建 manifest
    print("\n开始扫描数据集...")
    
    try:
        builder = ManifestBuilder(
            data_root=str(data_root),
            output_path=args.output,
            mask_output_dir=args.mask_dir
        )
        
        count = builder.build(
            datasets=args.datasets,
            splits=args.splits
        )
        
        print("\n" + "=" * 60)
        print(f"✓ 完成！共生成 {count} 条记录")
        print(f"  输出文件: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
