#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C-view 数据集导出脚本

导出 AEAD 封装的 C-view 二进制文件和 enc_info JSON。

Usage:
    python scripts/export_cview_dataset.py --manifest data/manifest.jsonl --output data/cview --password "your_password"
    python scripts/export_cview_dataset.py --manifest data/manifest.jsonl --output data/cview --password "your_password" --task-type detection

Requirements: 8.5
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


def export_cview_dataset(
    manifest_path: Path,
    output_dir: Path,
    data_root: Path,
    password: str,
    privacy_level: float = 1.0,
    task_type: str = 'classification',
    device: str = 'cpu',
    max_samples: Optional[int] = None,
    image_size: int = 256
) -> Dict:
    """
    导出 C-view 数据集
    
    Args:
        manifest_path: manifest 文件路径
        output_dir: 输出目录
        data_root: 数据根目录
        password: 加密密码
        privacy_level: 隐私级别
        task_type: 任务类型
        device: 计算设备
        max_samples: 最大样本数
        image_size: 图像尺寸
    
    Returns:
        stats: 导出统计
    """
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
        use_frequency=False
    )
    
    # 创建输出目录
    ciphertext_dir = output_dir / 'ciphertext'
    enc_info_dir = output_dir / 'enc_info'
    ciphertext_dir.mkdir(parents=True, exist_ok=True)
    enc_info_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {
        'total': len(records),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_bytes': 0,
        'by_dataset': {}
    }
    
    # 处理每条记录
    for record in tqdm(records, desc="导出 C-view"):
        sample_id = record['sample_id']
        dataset = record['dataset']
        split = record.get('split', 'unknown')
        
        # 初始化数据集统计
        if dataset not in stats['by_dataset']:
            stats['by_dataset'][dataset] = {'processed': 0, 'skipped': 0, 'bytes': 0}
        
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
            
            # 执行双视图加密
            result = engine.encrypt(
                images=image,
                privacy_map=privacy_map,
                privacy_level=privacy_level,
                image_id=sample_id,
                task_type=task_type,
                dataset=dataset,
                split=split
            )
            
            # 打包 C-view 为二进制存储格式
            binary_pack = engine.pack_cview_for_storage(result.c_view, result.enc_info)
            
            # 保存密文（二进制）
            ct_dir = ciphertext_dir / dataset / split
            ct_dir.mkdir(parents=True, exist_ok=True)
            ct_path = ct_dir / f"{sample_id}.bin"
            
            # 将密文列表合并为单个二进制文件
            ciphertext_bytes = bytes.fromhex(binary_pack['ciphertext'][0])
            with open(ct_path, 'wb') as f:
                f.write(ciphertext_bytes)
            
            stats['total_bytes'] += len(ciphertext_bytes)
            stats['by_dataset'][dataset]['bytes'] += len(ciphertext_bytes)
            
            # 保存 enc_info（JSON）
            info_dir = enc_info_dir / dataset / split
            info_dir.mkdir(parents=True, exist_ok=True)
            info_path = info_dir / f"{sample_id}.json"
            
            # 构建完整的 enc_info
            enc_info_to_save = {
                'version': binary_pack['version'],
                'format': binary_pack['format'],
                'shape': binary_pack['shape'],
                'mode': binary_pack['mode'],
                'wrap_bits': binary_pack['wrap_bits'],
                'nonces': binary_pack['nonces'],
                'tags': binary_pack['tags'],
                'affine_min': binary_pack['affine_min'],
                'affine_scale': binary_pack['affine_scale'],
                # 元数据
                'sample_id': sample_id,
                'dataset': dataset,
                'split': split,
                'task_type': task_type,
                'privacy_level': privacy_level,
                # 审计信息
                'image_id': binary_pack.get('image_id'),
                'privacy_map_hash': binary_pack.get('privacy_map_hash'),
                'z_view_hash': binary_pack.get('z_view_hash'),
                # 文件路径
                'ciphertext_path': str(ct_path.relative_to(output_dir)),
                'original_image_path': record['image_path']
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(enc_info_to_save, f, ensure_ascii=False, indent=2)
            
            stats['processed'] += 1
            stats['by_dataset'][dataset]['processed'] += 1
            
        except Exception as e:
            print(f"\n错误处理 {sample_id}: {e}")
            stats['errors'] += 1
    
    # 保存导出摘要
    summary = {
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
        'password_hash': hash(password) % (10 ** 8),
        'privacy_level': privacy_level,
        'task_type': task_type,
        'image_size': image_size,
        'stats': stats
    }
    
    summary_path = output_dir / 'export_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n导出摘要已保存到: {summary_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='导出 C-view 加密数据集'
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
        '--privacy-level', '-l',
        type=float,
        default=1.0,
        help='隐私级别 (默认: 1.0)'
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
    print("C-view 数据集导出")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"输出目录: {output_dir}")
    print(f"数据根目录: {data_root}")
    print(f"隐私级别: {args.privacy_level}")
    print(f"任务类型: {args.task_type}")
    print(f"设备: {args.device}")
    print(f"图像尺寸: {args.image_size}")
    print("=" * 60)
    
    stats = export_cview_dataset(
        manifest_path=manifest_path,
        output_dir=output_dir,
        data_root=data_root,
        password=args.password,
        privacy_level=args.privacy_level,
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
    print(f"总字节数: {stats['total_bytes']:,} bytes ({stats['total_bytes'] / 1024 / 1024:.2f} MB)")
    
    print("\n按数据集统计:")
    for dataset, ds_stats in stats['by_dataset'].items():
        print(f"  {dataset}: 处理 {ds_stats['processed']}, 跳过 {ds_stats['skipped']}, "
              f"大小 {ds_stats['bytes'] / 1024:.1f} KB")
    
    print("\n✓ 完成")


if __name__ == '__main__':
    main()
