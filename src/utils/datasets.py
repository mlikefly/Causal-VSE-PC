# -*- coding: utf-8 -*-
"""
数据集加载工具
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Tuple


def _resolve_celeba_root(root_dir: str) -> Path:
    """自动解析 CelebA-HQ 根目录，兼容多种运行路径。

    优先顺序：
    1) 传入的 root_dir
    2) CWD/data/CelebA-HQ
    3) CWD../data/CelebA-HQ
    4) 基于本文件向上回溯，自动寻找包含 data/CelebA-HQ 的仓库根目录
    """
    rd = Path(root_dir)
    candidates = []

    # 1) 若传入的是绝对路径，最高优先
    if rd.is_absolute():
        candidates.append(rd)

    # 2) 基于本文件向上回溯，优先锁定当前项目下的 data/CelebA-HQ
    try:
        here = Path(__file__).resolve()
        for p in here.parents:
            candidates.append(p / 'data' / 'CelebA-HQ')
    except Exception:
        pass

    # 3) 当前工作目录及其上一级（兼容从其他目录启动脚本的情况）
    cwd = Path.cwd().resolve()
    candidates.append(cwd / 'data' / 'CelebA-HQ')
    candidates.append(cwd.parent / 'data' / 'CelebA-HQ')

    # 4) 最后才考虑相对的 root_dir（例如 'data/CelebA-HQ'）
    if not rd.is_absolute():
        candidates.append(rd)

    for c in candidates:
        if (c / 'train').exists() or (c / 'val').exists() or (c / 'test').exists():
            return c.resolve()

    # 如果均不存在，返回原始 root_dir（后续由上层抛错）
    return rd


class CelebADataset(Dataset):
    """CelebA-HQ 数据集"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        return_labels: bool = False,
        transform=None
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            image_size: 图像大小
            return_labels: 是否返回标签
            transform: 图像变换
        """
        # 解析实际根目录
        self.root_dir = _resolve_celeba_root(root_dir)
        self.split = split
        self.image_size = image_size
        self.return_labels = return_labels
        self.transform = transform
        
        # 图像目录
        self.image_dir = self.root_dir / split
        
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"图像目录不存在: {self.image_dir} (解析根目录: {self.root_dir})"
            )
        
        # 获取所有图像文件
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + 
                                   list(self.image_dir.glob('*.jpg')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        
        # 标签目录（如果需要）
        if return_labels:
            label_dir = self.root_dir.parent / 'CelebA-HQ-labels' / split
            if label_dir.exists():
                self.label_dir = label_dir
                self.has_labels = True
            else:
                print(f"⚠️ 标签目录不存在: {label_dir}，将返回零标签")
                self.has_labels = False
        else:
            self.has_labels = False
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 调整大小
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # 转换为张量
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        # 返回标签（如果需要）
        if self.return_labels:
            if self.has_labels:
                # 尝试加载对应的标签文件
                label_file = self.label_dir / f"{img_path.stem}.npy"
                if label_file.exists():
                    labels = np.load(label_file)
                    labels = torch.from_numpy(labels).float()
                else:
                    # 如果没有标签文件，返回零标签
                    labels = torch.zeros(40)
            else:
                # 没有标签，返回零标签
                labels = torch.zeros(40)
            
            return image, labels
        else:
            return image


def get_celeba_dataloader(
    root_dir: str = 'data/CelebA-HQ',
    split: str = 'train',
    batch_size: int = 16,
    image_size: int = 256,
    return_labels: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    获取 CelebA-HQ 数据加载器
    
    Args:
        root_dir: 数据集根目录
        split: 'train', 'val', 或 'test'
        batch_size: 批次大小
        image_size: 图像大小
        return_labels: 是否返回标签
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        pin_memory: 是否使用固定内存
    
    Returns:
        DataLoader 对象
    """
    resolved_root = _resolve_celeba_root(root_dir)

    dataset = CelebADataset(
        root_dir=str(resolved_root),
        split=split,
        image_size=image_size,
        return_labels=return_labels
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return dataloader


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ 分割数据集（返回 image, mask）。
    
    - 优先从常见目录推断 mask 路径：
      1) root.parent / 'CelebAMask-HQ' / 'masks' / split / filename
      2) root.parent / 'CelebAMask-HQ' / split / filename
      3) root / 'masks' / split / filename
      4) root / 'masks' / filename
    - 若未找到，回退为伪 mask（基于灰度阈值）。
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'test',
        image_size: int = 256,
        threshold: float = 0.5,
    ):
        self.root_dir = _resolve_celeba_root(root_dir)
        self.split = split
        self.image_size = image_size
        self.threshold = threshold
        self.image_dir = self.root_dir / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        # 预计算候选 mask 根目录
        self._mask_roots = [
            self.root_dir.parent / 'CelebAMask-HQ' / 'masks' / self.split,
            self.root_dir.parent / 'CelebAMask-HQ' / self.split,
            self.root_dir / 'masks' / self.split,
            self.root_dir / 'masks',
        ]
        self.is_mask_dataset = True  # 供上层识别

    def _resolve_mask_path(self, img_path: Path) -> Path:
        fname = img_path.name
        cand = []
        for r in self._mask_roots:
            cand.append(r / fname)
            # 常见掩码为 png，若原图是 jpg 也尝试 png
            if fname.lower().endswith('.jpg'):
                cand.append(r / (img_path.stem + '.png'))
        for c in cand:
            if c.exists():
                return c
        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)

        mask_path = self._resolve_mask_path(img_path)
        if mask_path is not None and mask_path.exists():
            try:
                m = Image.open(mask_path).convert('L').resize((self.image_size, self.image_size), Image.NEAREST)
                mask_np = (np.array(m).astype(np.float32) / 255.0)
                mask_np = (mask_np > self.threshold).astype(np.float32)
                mask_t = torch.from_numpy(mask_np)[None, ...]
            except Exception:
                # 回退伪 mask
                gray = image_t.mean(dim=0, keepdim=True)
                mask_t = (gray > self.threshold).float()
        else:
            gray = image_t.mean(dim=0, keepdim=True)
            mask_t = (gray > self.threshold).float()
        return image_t, mask_t


def get_celeba_mask_dataloader(
    root_dir: str = 'data/CelebA-HQ',
    split: str = 'test',
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    dataset = CelebAMaskHQDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


class CelebAAttributesDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'test',
        image_size: int = 256,
        attr_file: Optional[str] = None,
    ):
        self.root_dir = _resolve_celeba_root(root_dir)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.root_dir / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        # 解析属性文件
        self.attr_map = self._load_attributes(attr_file)

    def _load_attributes(self, attr_file: Optional[str]):
        # 常见位置候选
        cands = []
        if attr_file:
            p = Path(attr_file)
            if p.exists():
                cands.append(p)
        base = self.root_dir.parent
        cands.extend([
            base / 'CelebA' / 'list_attr_celeba.txt',
            base / 'CelebA' / 'list_attr_celeba.csv',
            base / 'CelebA-HQ' / 'list_attr_celeba.txt',
            base / 'CelebA-HQ' / 'list_attr_celeba.csv',
            self.root_dir / 'list_attr_celeba.txt',
            self.root_dir / 'list_attr_celeba.csv',
        ])
        for c in cands:
            if c.exists():
                try:
                    if c.suffix.lower() == '.txt':
                        return self._parse_attr_txt(c)
                    else:
                        return self._parse_attr_csv(c)
                except Exception:
                    continue
        return {}

    def _parse_attr_txt(self, path: Path):
        m = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 3:
            return m
        # 第二行是属性名
        attr_names = lines[1].split()
        # 后续行为 文件名 + 40 个 -1/1
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) != 1 + len(attr_names):
                continue
            fname = parts[0]
            vals = [(1 if int(x) > 0 else 0) for x in parts[1:]]
            m[fname] = torch.tensor(vals, dtype=torch.float32)
        return m

    def _parse_attr_csv(self, path: Path):
        import csv
        m = {}
        with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            # 允许字段名包含 image_id 或 filename
            for row in reader:
                fname = row.get('image_id') or row.get('filename') or row.get('file')
                if not fname:
                    continue
                vals = []
                for k, v in row.items():
                    if k in ('image_id', 'filename', 'file'):
                        continue
                    try:
                        x = int(v)
                        vals.append(1.0 if x > 0 else 0.0)
                    except Exception:
                        try:
                            x = float(v)
                            vals.append(1.0 if x > 0 else 0.0)
                        except Exception:
                            pass
                if len(vals) == 40:
                    m[fname] = torch.tensor(vals, dtype=torch.float32)
        return m

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        # 匹配属性
        key = img_path.name
        labels = self.attr_map.get(key)
        if labels is None and key.lower().endswith('.jpg'):
            alt = img_path.stem + '.png'
            labels = self.attr_map.get(alt)
        if labels is None and key.lower().endswith('.png'):
            alt = img_path.stem + '.jpg'
            labels = self.attr_map.get(alt)
        if labels is None:
            labels = torch.zeros(40, dtype=torch.float32)
        return image_t, labels


def get_celeba_attr_dataloader(
    root_dir: str = 'data/CelebA-HQ',
    split: str = 'test',
    batch_size: int = 8,
    image_size: int = 256,
    attr_file: Optional[str] = None,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    dataset = CelebAAttributesDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        attr_file=attr_file,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


__all__ = [
    'CelebADataset',
    'get_celeba_dataloader',
    'CelebAMaskHQDataset',
    'get_celeba_mask_dataloader',
    'CelebAAttributesDataset',
    'get_celeba_attr_dataloader',
]













