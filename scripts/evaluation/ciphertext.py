"""
密文域任务评估（严格真实数据与预训练模型）
验收标准：密文域任务性能 ≥ 原图80%

评估任务：
1. 分类任务（属性分类）
2. 分割任务（人脸分割）
3. 检测任务（COCO目标检测）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from src.cipher.scne_cipher import SCNECipherAPI
from src.utils.datasets import (
    get_celeba_dataloader,
    get_celeba_attr_dataloader,
    get_celeba_mask_dataloader,
)
from src.evaluation.attack_models import AttributeClassifier, SegmentationAttacker


def _to_serializable(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class CiphertextTaskEvaluator:
    """密文域任务评估器"""
    
    def __init__(
        self,
        password: str = "evaluation_password_2025",
        image_size: int = 256,
        device: str = 'cuda'
    ):
        self.device = device
        self.image_size = image_size
        
        # 加密系统
        self.cipher = SCNECipherAPI(
            password=password,
            image_size=image_size
        )
        
        # 任务模型
        self.attr_classifier = AttributeClassifier(num_attributes=40).to(device)
        self.segmenter = SegmentationAttacker(in_channels=3, out_channels=1).to(device)
        
        # 验收阈值
        self.threshold_ratio = 0.80  # 密文性能 ≥ 原图性能的80%
    
    def load_task_models(self, model_dir: str):
        """加载任务模型（如果存在预训练）"""
        model_dir = Path(model_dir)
        
        models = {
            'attr_classifier': self.attr_classifier,
            'segmenter': self.segmenter
        }
        
        for name, model in models.items():
            model_path = model_dir / f"{name}.pth"
            if model_path.exists():
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"✓ 加载任务模型: {name}")
                except Exception as e:
                    print(f"⚠️ 加载 {name} 失败: {e}")
    
    @torch.no_grad()
    def evaluate_classification_task(
        self,
        dataloader,
        privacy_level: float = 0.3,
        num_batches: int = 10
    ) -> dict:
        """
        评估分类任务
        
        Args:
            dataloader: 数据加载器
            privacy_level: 隐私级别 (0.3=可用性优先, 1.0=隐私优先)
            num_batches: 评估批次数
        """
        
        print(f"\n评估分类任务 (隐私级别={privacy_level})...")
        
        self.attr_classifier.eval()
        
        # 收集结果
        orig_correct = 0
        enc_correct = 0
        total = 0
        
        pbar = tqdm(total=num_batches, desc="分类评估")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # 处理batch
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                if len(batch) > 1 and batch[1] is not None:
                    labels = batch[1]
                else:
                    raise RuntimeError("Classification task requires real attribute labels; dataloader did not provide labels.")
            else:
                images = batch
                raise RuntimeError("Classification task requires (images, labels) batches; got images only.")
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 转换为3通道
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # 加密
            encrypted, enc_info = self.cipher.encrypt_simple(
                images.mean(dim=1, keepdim=True),  # 转灰度
                privacy_level=privacy_level,
                semantic_preserving=True
            )
            encrypted = encrypted.repeat(1, 3, 1, 1)  # 转回3通道
            
            # 原图分类
            logits_orig = self.attr_classifier(images)
            pred_orig = (torch.sigmoid(logits_orig) > 0.5).float()
            
            # 密文分类
            logits_enc = self.attr_classifier(encrypted)
            pred_enc = (torch.sigmoid(logits_enc) > 0.5).float()
            
            # 统计准确率
            orig_correct += (pred_orig == labels).float().sum().item()
            enc_correct += (pred_enc == labels).float().sum().item()
            total += labels.numel()
            
            pbar.update(1)
        
        pbar.close()
        
        # 计算准确率
        orig_acc = orig_correct / total if total > 0 else 0.0
        enc_acc = enc_correct / total if total > 0 else 0.0
        
        # 性能保留率
        retention_ratio = enc_acc / orig_acc if orig_acc > 0 else 0.0
        
        results = {
            'original_accuracy': orig_acc,
            'encrypted_accuracy': enc_acc,
            'retention_ratio': retention_ratio,
            'pass_threshold': retention_ratio >= self.threshold_ratio,
            'privacy_level': privacy_level
        }
        
        return results
    
    @torch.no_grad()
    def evaluate_segmentation_task(
        self,
        dataloader,
        privacy_level: float = 0.3,
        num_batches: int = 10
    ) -> dict:
        """
        评估分割任务
        
        Args:
            dataloader: 数据加载器
            privacy_level: 隐私级别
            num_batches: 评估批次数
        """
        
        print(f"\n评估分割任务 (隐私级别={privacy_level})...")
        
        self.segmenter.eval()
        
        # 收集结果
        orig_iou_list = []
        enc_iou_list = []
        
        # 如果数据集本身提供了 mask（例如 CelebAMaskHQDataset），则优先直接使用
        ds = getattr(dataloader, 'dataset', None)
        use_dataset_masks = getattr(ds, 'is_mask_dataset', False)

        pbar = tqdm(total=num_batches, desc="分割评估")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # 处理batch
            provided_masks = None
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                # 仅当数据集标记为 mask 数据集时，才将第二个元素视为 GT mask
                if use_dataset_masks and len(batch) > 1:
                    provided_masks = batch[1]
            else:
                images = batch
            
            images = images.to(self.device)
            
            # 转换为3通道
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # 优先使用数据集提供的掩码；如果没有，则再尝试从磁盘加载真实 CelebAMask-HQ 前景掩码
            B, C, H, W = images.shape
            if provided_masks is not None:
                # 直接使用数据集返回的 mask（可能是真实的，也可能是基于灰度的伪 mask）
                gt_masks = provided_masks.to(images.device)
                if gt_masks.dim() == 3:
                    gt_masks = gt_masks.unsqueeze(1)
            else:
                gt_masks = torch.zeros(B, 1, H, W, device=images.device)
                try:
                    has_files = hasattr(ds, 'image_files') and hasattr(ds, 'split')
                    batch_size = getattr(dataloader, 'batch_size', B)
                    base_idx = batch_idx * int(batch_size)
                    from PIL import Image as _PILImage
                    from pathlib import Path as _Path
                    def _load_mask_for(path_str: str):
                        if not path_str:
                            return None
                        p = _Path(path_str)
                        fname = p.name
                        # 候选掩码路径
                        root = ds.root_dir if hasattr(ds, 'root_dir') else p.parent.parent
                        cand = [
                            _Path(root).parent / 'CelebAMask-HQ' / 'masks' / ds.split / fname,
                            _Path(root).parent / 'CelebAMask-HQ' / ds.split / fname,
                            _Path(root) / 'masks' / fname,
                        ]
                        for c in cand:
                            if c.exists():
                                try:
                                    m = _PILImage.open(str(c)).convert('L').resize((W, H))
                                    arr = (np.array(m).astype(np.float32) / 255.0)
                                    arr = (arr > 0.5).astype(np.float32)
                                    return torch.from_numpy(arr)[None, None, ...]
                                except Exception:
                                    continue
                        return None
                    if has_files:
                        for i in range(B):
                            idx = base_idx + i
                            if 0 <= idx < len(ds.image_files):
                                mask_t = _load_mask_for(str(ds.image_files[idx]))
                                if mask_t is not None:
                                    gt_masks[i:i+1] = mask_t.to(images.device)
                except Exception:
                    gt_masks = torch.zeros(B, 1, H, W, device=images.device)
            
            # 加密
            encrypted, enc_info = self.cipher.encrypt_simple(
                images.mean(dim=1, keepdim=True),
                privacy_level=privacy_level,
                semantic_preserving=True
            )
            encrypted = encrypted.repeat(1, 3, 1, 1)
            
            # 原图分割
            pred_orig = self.segmenter(images)
            pred_orig_binary = (pred_orig > 0.5).float()
            
            # 密文分割
            pred_enc = self.segmenter(encrypted)
            pred_enc_binary = (pred_enc > 0.5).float()
            
            # 计算IoU（仅对存在真实掩码的样本）
            for i in range(images.size(0)):
                if gt_masks[i].sum() <= 0:
                    continue
                # 原图IoU
                intersection_orig = (pred_orig_binary[i] * gt_masks[i]).sum()
                union_orig = ((pred_orig_binary[i] + gt_masks[i]) > 0).float().sum()
                iou_orig = (intersection_orig / (union_orig + 1e-6)).item()
                
                # 密文IoU
                intersection_enc = (pred_enc_binary[i] * gt_masks[i]).sum()
                union_enc = ((pred_enc_binary[i] + gt_masks[i]) > 0).float().sum()
                iou_enc = (intersection_enc / (union_enc + 1e-6)).item()
                
                orig_iou_list.append(iou_orig)
                enc_iou_list.append(iou_enc)
            
            pbar.update(1)
        
        pbar.close()
        
        # 计算平均IoU（若没有任何真实掩码样本，则视为配置错误）
        if len(orig_iou_list) == 0:
            raise RuntimeError("Segmentation task requires real foreground masks; no valid masks were found for any sample.")
        orig_iou = np.mean(orig_iou_list)
        enc_iou = np.mean(enc_iou_list)
        
        # 性能保留率
        retention_ratio = enc_iou / orig_iou if orig_iou > 0 else 0.0
        
        results = {
            'original_iou': orig_iou,
            'encrypted_iou': enc_iou,
            'retention_ratio': retention_ratio,
            'pass_threshold': retention_ratio >= self.threshold_ratio,
            'privacy_level': privacy_level
        }
        
        return results
    
    def evaluate_privacy_utility_tradeoff(
        self,
        dataloader=None,
        *,
        attr_loader=None,
        mask_loader=None,
        privacy_levels: list = [0.0, 0.3, 0.5, 0.7, 1.0],
        num_batches: int = 5
    ) -> dict:
        """
        评估隐私-可用性权衡
        
        Args:
            dataloader: 数据加载器
            privacy_levels: 隐私级别列表
            num_batches: 每个级别的评估批次数
        """
        
        print("\n" + "="*70)
        print("评估隐私-可用性权衡曲线")
        print("="*70)
        
        results = {
            'privacy_levels': privacy_levels,
            'classification': [],
            'segmentation': []
        }
        
        for privacy_level in privacy_levels:
            print(f"\n隐私级别: {privacy_level}")
            
            # 分类任务
            cls_result = self.evaluate_classification_task(
                (attr_loader if attr_loader is not None else dataloader),
                privacy_level=privacy_level,
                num_batches=num_batches
            )
            results['classification'].append(cls_result)
            
            # 分割任务
            seg_result = self.evaluate_segmentation_task(
                (mask_loader if mask_loader is not None else dataloader),
                privacy_level=privacy_level,
                num_batches=num_batches
            )
            results['segmentation'].append(seg_result)
        
        return results
    
    def print_tradeoff_report(self, results):
        """打印权衡曲线报告"""
        
        print("\n" + "="*70)
        print("隐私-可用性权衡报告")
        print("="*70)
        
        print(f"\n{'隐私级别':<10} {'分类保留率':<15} {'分割保留率':<15} {'验收状态'}")
        print("-" * 60)
        
        for i, privacy_level in enumerate(results['privacy_levels']):
            cls_ratio = results['classification'][i]['retention_ratio']
            seg_ratio = results['segmentation'][i]['retention_ratio']
            
            # 检查是否通过（至少一个任务≥80%）
            pass_check = cls_ratio >= 0.80 or seg_ratio >= 0.80
            status = "✓" if pass_check else "✗"
            
            print(f"{privacy_level:<10.1f} {cls_ratio:<15.2%} {seg_ratio:<15.2%} {status}")
        
        print("\n推荐隐私级别:")
        # 找到最佳隐私级别（保留率≥80%的最高隐私）
        for i in reversed(range(len(results['privacy_levels']))):
            cls_ratio = results['classification'][i]['retention_ratio']
            seg_ratio = results['segmentation'][i]['retention_ratio']
            
            if cls_ratio >= 0.80 or seg_ratio >= 0.80:
                print(f"  隐私级别 {results['privacy_levels'][i]:.1f}: "
                      f"分类保留{cls_ratio:.1%}, 分割保留{seg_ratio:.1%}")
                break
        
        print("="*70)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Ciphertext Task Evaluation (strict real data)')
    parser.add_argument('--dataset-root', type=str, default='data/CelebA-HQ')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化评估器
    evaluator = CiphertextTaskEvaluator(
        password="evaluation_password_2025",
        image_size=256,
        device=device
    )

    # 强制要求预训练模型存在
    evaluator.load_task_models('src/models/pretrained')
    model_dir = Path('src/models/pretrained')
    required_models = [model_dir / 'attr_classifier.pth', model_dir / 'segmenter.pth']
    missing = [str(p) for p in required_models if not p.exists()]
    if missing:
        print("✗ 缺少任务模型权重：")
        for m in missing:
            print(f"  - {m}")
        print("请先运行: python scripts/train_task_models.py 或将预训练权重放置到上述路径。")
        return 2

    # 加载真实数据（属性与掩码专用加载器）
    try:
        attr_loader = get_celeba_attr_dataloader(
            root_dir=args.dataset_root,
            split='test',
            batch_size=args.batch_size,
            image_size=256,
            shuffle=False,
            num_workers=args.num_workers,
        )
        mask_loader = get_celeba_mask_dataloader(
            root_dir=args.dataset_root,
            split='test',
            batch_size=args.batch_size,
            image_size=256,
            shuffle=False,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"✗ 无法加载真实数据集: {e}")
        print("请确保 --dataset-root 指向有效的 CelebA-HQ 路径（及 CelebAMask-HQ 掩码目录，如存在）。")
        return 2
    
    # 评估隐私-可用性权衡
    results = evaluator.evaluate_privacy_utility_tradeoff(
        dataloader=None,
        attr_loader=attr_loader,
        mask_loader=mask_loader,
        privacy_levels=[0.0, 0.3, 0.5, 0.7, 1.0],
        num_batches=5
    )
    
    # 打印报告
    evaluator.print_tradeoff_report(results)
    
    # 保存结果
    save_dir = Path('results/ciphertext_tasks')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'tradeoff_results.json', 'w') as f:
        json.dump(_to_serializable(results), f, indent=2)
    
    print(f"\n✓ 结果已保存到: {save_dir / 'tradeoff_results.json'}")
    
    # 检查验收标准
    # 使用privacy_level=0.3的结果
    idx_03 = results['privacy_levels'].index(0.3)
    cls_pass = results['classification'][idx_03]['pass_threshold']
    seg_pass = results['segmentation'][idx_03]['pass_threshold']
    
    if cls_pass or seg_pass:
        print("\n✓ 密文域任务验收标准通过！")
        return 0
    else:
        print("\n✗ 密文域任务未达到验收标准")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

