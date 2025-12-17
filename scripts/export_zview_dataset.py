#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Z-view 数据集导出脚本

按 privacy_level 档位批量导出 Z-view 加密图像。
支持 {0.0, 0.3, 0.5, 0.7, 1.0} 五档隐私级别。

Usage:
    python scripts/export_zview_dataset.py --manifest data/manifest.jsonl --output data/zview --password "your_password"
    python scripts/export_zview_dataset.py --manifest data/manifest.jsonl --output data/zview --password "your_password" --levels 0.3 0.7 1.0
    python scripts/export_zview_dataset.py --manifest data/manifest.jsonl --output data/zview --password "your_password" --task-type classification

Requirements: 8.4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cipher.dual_view_engine import DualViewEncryptionEngine


SUPPORTED_PRIVACY_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]


def load_manifest(manifest_path: Path) -> List[Dict]:
    """加载 manifest 文件"""
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image(image_path: Path) -> Optional[torch.Tensor]:
    """加载图像为张量 [1, 3, H, W]"""
    if not image_path.exists():
        return None
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # [H, W, 3] -> [1, 3, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def load_privacy_map(privacy_map_path: Path, shape: tuple) -> torch.Tensor:
    """加载隐私预算图"""
    H, W = shape
    
    if privacy_map_path and privacy_map_path.exists():
        if privacy_map_path.suffix == '.npy':
            privacy_map = np.load(privacy_map_path)
        else:
            img = Image.open(privacy_map_path).convert('L')
            privacy_map = np.array(img, dtype=np.float32) / 255.0
        
        # 调整大小
        if privacy_map.shape != (H, W):
            from PIL import Image as PILImage
            pil_map = PILImage.fromarray((privacy_map * 255).astype(np.uint8))
            pil_map = pil_map.resize((W, H), PILImage.NEAREST)
            privacy_map = np.array(pil_map, dtype=np.float32) / 255.0
        
        return torch.from_numpy(privacy_map).unsqueeze(0).unsqueeze(0)
    else:
        # 默认全图中等隐私
        return torch.ones(1, 1, H, W) * 0.5


def save_zview_image(z_view: torch.Tensor, output_path: Path) -> None:
    """保存 Z-view 为图像"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # [1, 3, H, W] -> [H, W, 3]
    img_np = z_view.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_np)
    img.save(output_path)


def export_zview_dataset(
    manifest_path: Path,
    output_dir: Path,
    data_root: Path,
    password: str,
    privacy_levels: List[float] = None,
    task_type: str = 'classification',
    device: str = 'cpu',
    max_samples: Optional[int] = None,
    image_size: int = 256
) -> Dict:
    """
    导出 Z-view 数据集
    
    Args:
        manifest_path: manifest 文件路径
        output_dir: 输出目录
        data_root: 数据根目录
        password: 加密密码
        privacy_levels: 隐私级别列表
        task_type: 任务类型
        device: 计算设备
        max_samples: 最大样本数
        image_size: 图像尺寸
    
    Returns:
        stats: 导出统计
    """
    if privacy_levels is None:
        privacy_levels = SUPPORTED_PRIVACY_LEVELS
    
    # 验证隐私级别
    for level in privacy_levels:
        if level not in SUPPORTED_PRIVACY_LEVELS:
            print(f"警告: 隐私级别 {level} 不在支持列表中，将使用最接近的值")
    
    # 加载 manifest
    records = load_manifest(manifest_path)
    print(f"加载 {len(records)} 条记录")
    
    if max_samples:
        records = records[:max_samples]
        print(f"限制处理 {max_samples} 条记录")
    
    # 初始化加密引擎
    engine = DualViewEncryptionEngine(
        password=password,
        image_size=image_size,
        device=device,
        deterministic=True,
        use_frequency=False  # 简化处理，仅使用空域加密
    )
    
    # 统计
    stats = {
        'total': len(records),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'by_level': {level: 0 for level in privacy_levels},
        'by_dataset': {}
    }
    
    # 为每个隐私级别创建输出目录
    for level in privacy_levels:
        level_dir = output_dir / f"level_{level:.1f}"
        level_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每条记录
    for record in tqdm(records, desc="导出 Z-view"):
        sample_id = record['sample_id']
        dataset = record['dataset']
        split = record.get('split', 'unknown')
        
        # 初始化数据集统计
        if dataset not in stats['by_dataset']:
            stats['by_dataset'][dataset] = {'processed': 0, 'skipped': 0}
        
        try:
            # 加载图像
            image_path = data_root / record['image_path']
            image = load_image(image_path)
            
            if image is None:
                stats['skipped'] += 1
                stats['by_dataset'][dataset]['skipped'] += 1
                continue
            
            # 调整图像大小
            if image.shape[2] != image_size or image.shape[3] != image_size:
                image = torch.nn.functional.interpolate(
                    image, size=(image_size, image_size), mode='bilinear', align_corners=False
                )
            
            # 加载隐私预算图
            privacy_map_path = record.get('privacy_map_paths', {}).get(task_type)
            if privacy_map_path:
                privacy_map_path = data_root / privacy_map_path
            else:
                privacy_map_path = None
            
            privacy_map = load_privacy_map(privacy_map_path, (image_size, image_size))
            
            # 为每个隐私级别生成 Z-view
            for level in privacy_levels:
                z_view, z_info = engine.encrypt_zview_only(
                    images=image,
                    privacy_map=privacy_map,
                    privacy_level=level,
                    image_id=sample_id,
                    task_type=task_type,
                    semantic_preserving=(level < 0.5)
                )
                
                # 保存 Z-view
                level_dir = output_dir / f"level_{level:.1f}" / dataset / split
                output_path = level_dir / f"{sample_id}.png"
                save_zview_image(z_view, output_path)
                
                stats['by_level'][level] += 1
            
            stats['processed'] += 1
            stats['by_dataset'][dataset]['processed'] += 1
            
        except Exception as e:
            print(f"\n错误处理 {sample_id}: {e}")
            stats['errors'] += 1
    
    # 保存导出信息
    export_info = {
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
        'password_hash': hash(password) % (10 ** 8),  # 不保存明文密码
        'privacy_levels': privacy_levels,
        'task_type': task_type,
        'image_size': image_size,
        'stats': stats
    }
    
    info_path = output_dir / 'export_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(export_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n导出信息已保存到: {info_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='导出 Z-view 加密数据集'
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
        '--password', '-p',
        type=str,
        required=True,
        help='加密密码'
    )
    parser.add_argument(
        '--levels', '-l',
        nargs='+',
        type=float,
        default=None,
        help=f'隐私级别列表 (默认: {SUPPORTED_PRIVACY_LEVELS})'
    )
    parser.add_argument(
        '--task-type', '-t',
        choices=['classification', 'detection', 'segmentation'],
        default='classification',
        help='任务类型 (默认: classification)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='计算设备 (默认: cpu)'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='最大处理样本数'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='图像尺寸 (默认: 256)'
    )
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    data_root = Path(args.data_root)
    
    if not manifest_path.exists():
        print(f"错误: Manifest 文件不存在: {manifest_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Z-view 数据集导出")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"输出目录: {output_dir}")
    print(f"数据根目录: {data_root}")
    print(f"隐私级别: {args.levels or SUPPORTED_PRIVACY_LEVELS}")
    print(f"任务类型: {args.task_type}")
    print(f"设备: {args.device}")
    print(f"图像尺寸: {args.image_size}")
    print("=" * 60)
    
    stats = export_zview_dataset(
        manifest_path=manifest_path,
        output_dir=output_dir,
        data_root=data_root,
        password=args.password,
        privacy_levels=args.levels,
        task_type=args.task_type,
        device=args.device,
        max_samples=args.max_samples,
        image_size=args.image_size
    )
    
    print("\n" + "=" * 60)
    print("导出统计")
    print("=" * 60)
    print(f"总记录数: {stats['total']}")
    print(f"已处理: {stats['processed']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"错误: {stats['errors']}")
    
    print("\n按隐私级别统计:")
    for level, count in stats['by_level'].items():
        print(f"  level_{level:.1f}: {count} 张")
    
    print("\n按数据集统计:")
    for dataset, ds_stats in stats['by_dataset'].items():
        print(f"  {dataset}: 处理 {ds_stats['processed']}, 跳过 {ds_stats['skipped']}")
    
    print("\n✓ 完成")


if __name__ == '__main__':
    main()
