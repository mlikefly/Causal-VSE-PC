#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义掩码批量生成脚本

读取 manifest，批量生成三值语义掩码，并更新 manifest 中的掩码路径。

使用方法:
    # 生成所有掩码
    python scripts/gen_semantic_masks.py --manifest manifest.jsonl --output-dir masks
    
    # 只处理特定数据集
    python scripts/gen_semantic_masks.py --manifest manifest.jsonl --output-dir masks --datasets celebahq
    
    # 限制处理数量
    python scripts/gen_semantic_masks.py --manifest manifest.jsonl --output-dir masks --limit 100

Requirements: 8.2
"""

import argparse
import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.manifest_builder import ManifestBuilder, ManifestRecord
from src.data.semantic_mask_generator import SemanticMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='批量生成语义掩码')
    
    parser.add_argument(
        '-m', '--manifest',
        type=str,
        required=True,
        help='输入 manifest.jsonl 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='masks',
        help='掩码输出目录 (默认: masks)'
    )
    
    parser.add_argument(
        '-d', '--data-root',
        type=str,
        default='data',
        help='数据集根目录 (默认: data)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='只处理指定数据集'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        choices=['opencv', 'retinaface', 'mtcnn'],
        default='opencv',
        help='人脸检测器类型 (默认: opencv)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理数量'
    )
    
    parser.add_argument(
        '--update-manifest',
        action='store_true',
        help='更新 manifest 文件中的掩码路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("语义掩码批量生成")
    print("=" * 60)
    
    # 检查输入文件
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"❌ Manifest 文件不存在: {manifest_path}")
        sys.exit(1)
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化生成器
    celebamask_root = data_root / 'CelebAMask-HQ'
    gen = SemanticMaskGenerator(
        detector_type=args.detector,
        celebamask_root=str(celebamask_root) if celebamask_root.exists() else None
    )
    
    print(f"\n配置:")
    print(f"  Manifest: {manifest_path}")
    print(f"  数据根目录: {data_root}")
    print(f"  输出目录: {output_dir}")
    print(f"  检测器: {args.detector}")
    print(f"  CelebAMask: {'✓' if celebamask_root.exists() else '✗'}")
    
    # 加载 manifest
    records = ManifestBuilder.load_manifest(str(manifest_path))
    print(f"\n加载 {len(records)} 条记录")
    
    # 过滤数据集
    if args.datasets:
        records = [r for r in records if r.dataset in args.datasets]
        print(f"过滤后: {len(records)} 条记录 (datasets={args.datasets})")
    
    # 限制数量
    if args.limit:
        records = records[:args.limit]
        print(f"限制后: {len(records)} 条记录")
    
    # 统计
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    updated_records = []
    
    print("\n开始生成掩码...")
    
    for record in tqdm(records, desc="生成掩码"):
        try:
            # 构建图像路径
            image_path = data_root / record.image_path
            
            if not image_path.exists():
                if args.verbose:
                    print(f"⚠️ 图像不存在: {image_path}")
                stats['skipped'] += 1
                updated_records.append(record)
                continue
            
            # 加载图像
            image = np.array(Image.open(image_path).convert('RGB'))
            
            # 提取 image_id
            image_id = Path(record.image_path).stem
            
            # 生成掩码
            mask = gen.generate(
                image,
                dataset=record.dataset,
                image_id=image_id,
                annotations=record.labels if 'bbox' in record.labels else None
            )
            
            # 保存掩码
            mask_subdir = output_dir / record.dataset / record.split
            mask_subdir.mkdir(parents=True, exist_ok=True)
            
            mask_filename = f"{image_id}.png"
            mask_path = mask_subdir / mask_filename
            
            # 转换为 uint8 保存 (0, 127, 255 对应 0.0, 0.5, 1.0)
            mask_uint8 = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(mask_path)
            
            # 更新记录
            rel_mask_path = str(mask_path.relative_to(Path.cwd()))
            record_dict = record.to_dict()
            record_dict['semantic_mask_path'] = rel_mask_path
            record_dict['sensitive_mask_path'] = rel_mask_path
            record_dict['task_mask_path'] = rel_mask_path
            updated_records.append(ManifestRecord.from_dict(record_dict))
            
            stats['processed'] += 1
            
        except Exception as e:
            if args.verbose:
                print(f"❌ 处理失败 {record.sample_id}: {e}")
            stats['errors'] += 1
            updated_records.append(record)
    
    # 更新 manifest
    if args.update_manifest and updated_records:
        output_manifest = manifest_path.with_suffix('.updated.jsonl')
        with open(output_manifest, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
        print(f"\n✓ 更新后的 Manifest: {output_manifest}")
    
    # 打印统计
    print("\n" + "=" * 60)
    print("统计:")
    print(f"  处理成功: {stats['processed']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  错误: {stats['errors']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
