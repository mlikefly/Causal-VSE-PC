#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隐私预算图生成脚本

读取 manifest 和语义掩码，为每张图像生成三种任务类型的隐私预算图。

Usage:
    python scripts/gen_privacy_maps.py --manifest data/manifest.jsonl --output data/privacy_maps
    python scripts/gen_privacy_maps.py --manifest data/manifest.jsonl --output data/privacy_maps --global-privacy 0.8
    python scripts/gen_privacy_maps.py --manifest data/manifest.jsonl --output data/privacy_maps --tasks classification detection

Requirements: 8.3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.causal_budget_allocator import CausalBudgetAllocator


def load_manifest(manifest_path: Path) -> List[Dict]:
    """加载 manifest 文件"""
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_semantic_mask(mask_path: Path) -> Optional[np.ndarray]:
    """加载语义掩码"""
    if not mask_path.exists():
        return None
    
    # 支持多种格式
    if mask_path.suffix == '.npy':
        return np.load(mask_path)
    elif mask_path.suffix in ['.png', '.jpg', '.jpeg']:
        img = Image.open(mask_path).convert('L')
        mask = np.array(img, dtype=np.float32) / 255.0
        return mask
    else:
        return None


def save_privacy_map(
    privacy_map: np.ndarray,
    output_path: Path,
    format: str = 'npy'
) -> None:
    """保存隐私预算图"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npy':
        np.save(output_path.with_suffix('.npy'), privacy_map)
    elif format == 'png':
        # 转换为 0-255 灰度图
        img = Image.fromarray((privacy_map * 255).astype(np.uint8), mode='L')
        img.save(output_path.with_suffix('.png'))


def generate_privacy_maps_for_manifest(
    manifest_path: Path,
    output_dir: Path,
    data_root: Path,
    global_privacy: float = 1.0,
    task_types: Optional[List[str]] = None,
    save_format: str = 'npy',
    save_reports: bool = True,
    max_samples: Optional[int] = None
) -> Dict:
    """
    为 manifest 中的所有样本生成隐私预算图
    
    Args:
        manifest_path: manifest 文件路径
        output_dir: 输出目录
        data_root: 数据根目录
        global_privacy: 全局隐私级别
        task_types: 任务类型列表
        save_format: 保存格式 ('npy' 或 'png')
        save_reports: 是否保存因果报告
        max_samples: 最大处理样本数
    
    Returns:
        stats: 处理统计
    """
    # 加载 manifest
    records = load_manifest(manifest_path)
    print(f"加载 {len(records)} 条记录")
    
    if max_samples:
        records = records[:max_samples]
        print(f"限制处理 {max_samples} 条记录")
    
    # 初始化分配器
    allocator = CausalBudgetAllocator()
    
    if task_types is None:
        task_types = allocator.TASK_TYPES
    
    # 创建输出目录
    output_dir = Path(output_dir)
    maps_dir = output_dir / 'maps'
    reports_dir = output_dir / 'reports'
    maps_dir.mkdir(parents=True, exist_ok=True)
    if save_reports:
        reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {
        'total': len(records),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'by_dataset': {}
    }
    
    # 更新后的 manifest 记录
    updated_records = []
    
    for record in tqdm(records, desc="生成隐私预算图"):
        sample_id = record['sample_id']
        dataset = record['dataset']
        
        # 初始化数据集统计
        if dataset not in stats['by_dataset']:
            stats['by_dataset'][dataset] = {'processed': 0, 'skipped': 0}
        
        try:
            # 获取语义掩码路径
            semantic_mask_path = record.get('semantic_mask_path')
            
            if not semantic_mask_path:
                # 如果没有语义掩码，尝试从图像生成默认掩码
                image_path = data_root / record['image_path']
                if image_path.exists():
                    img = Image.open(image_path)
                    H, W = img.size[1], img.size[0]
                    # 默认全图为任务区域
                    semantic_mask = np.full((H, W), 0.5, dtype=np.float32)
                else:
                    stats['skipped'] += 1
                    stats['by_dataset'][dataset]['skipped'] += 1
                    updated_records.append(record)
                    continue
            else:
                # 加载语义掩码
                mask_path = data_root / semantic_mask_path
                semantic_mask = load_semantic_mask(mask_path)
                
                if semantic_mask is None:
                    stats['skipped'] += 1
                    stats['by_dataset'][dataset]['skipped'] += 1
                    updated_records.append(record)
                    continue
            
            # 生成隐私预算图
            privacy_maps = allocator.generate_privacy_maps(
                semantic_mask,
                global_privacy=global_privacy,
                task_types=task_types
            )
            
            # 保存隐私预算图
            privacy_map_paths = {}
            for task_type, privacy_map in privacy_maps.items():
                map_filename = f"{sample_id}_{task_type}"
                map_path = maps_dir / dataset / map_filename
                save_privacy_map(privacy_map, map_path, format=save_format)
                
                # 记录相对路径
                rel_path = str(map_path.relative_to(output_dir))
                if save_format == 'npy':
                    rel_path += '.npy'
                else:
                    rel_path += '.png'
                privacy_map_paths[task_type] = rel_path
            
            # 更新记录
            record['privacy_map_paths'] = privacy_map_paths
            
            # 生成并保存因果报告
            if save_reports:
                report = allocator.generate_causal_report(
                    semantic_mask,
                    privacy_maps,
                    sample_id=sample_id,
                    dataset=dataset
                )
                
                report_path = reports_dir / dataset / f"{sample_id}_causal_report.json"
                allocator.save_report(report, report_path)
                record['causal_report_path'] = str(report_path.relative_to(output_dir))
            
            stats['processed'] += 1
            stats['by_dataset'][dataset]['processed'] += 1
            
        except Exception as e:
            print(f"\n错误处理 {sample_id}: {e}")
            stats['errors'] += 1
        
        updated_records.append(record)
    
    # 保存更新后的 manifest
    updated_manifest_path = output_dir / 'manifest_with_privacy.jsonl'
    with open(updated_manifest_path, 'w', encoding='utf-8') as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n更新后的 manifest 已保存到: {updated_manifest_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='为 manifest 中的样本生成隐私预算图'
    )
    parser.add_argument(
        '--manifest', '-m',
        type=str,
        required=True,
        help='Manifest 文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default='data',
        help='数据根目录 (默认: data)'
    )
    parser.add_argument(
        '--global-privacy', '-g',
        type=float,
        default=1.0,
        help='全局隐私级别 [0.0, 1.0] (默认: 1.0)'
    )
    parser.add_argument(
        '--tasks', '-t',
        nargs='+',
        choices=['classification', 'detection', 'segmentation'],
        default=None,
        help='要生成的任务类型 (默认: 全部)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['npy', 'png'],
        default='npy',
        help='保存格式 (默认: npy)'
    )
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='不保存因果报告'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='最大处理样本数'
    )
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    data_root = Path(args.data_root)
    
    if not manifest_path.exists():
        print(f"错误: Manifest 文件不存在: {manifest_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("隐私预算图生成")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"输出目录: {output_dir}")
    print(f"数据根目录: {data_root}")
    print(f"全局隐私级别: {args.global_privacy}")
    print(f"任务类型: {args.tasks or '全部'}")
    print(f"保存格式: {args.format}")
    print(f"保存报告: {not args.no_reports}")
    print("=" * 60)
    
    stats = generate_privacy_maps_for_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        data_root=data_root,
        global_privacy=args.global_privacy,
        task_types=args.tasks,
        save_format=args.format,
        save_reports=not args.no_reports,
        max_samples=args.max_samples
    )
    
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"总记录数: {stats['total']}")
    print(f"已处理: {stats['processed']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"错误: {stats['errors']}")
    
    print("\n按数据集统计:")
    for dataset, ds_stats in stats['by_dataset'].items():
        print(f"  {dataset}: 处理 {ds_stats['processed']}, 跳过 {ds_stats['skipped']}")
    
    print("\n✓ 完成")


if __name__ == '__main__':
    main()
