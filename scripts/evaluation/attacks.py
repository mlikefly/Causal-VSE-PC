import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.cipher.scne_cipher import SCNECipherAPI
from src.evaluation.attack_models import AttackEvaluator
from src.utils.datasets import get_celeba_attr_dataloader


def _bce_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return loss.mean(dim=1)


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-scores)
    scores = scores[idx]
    labels = labels[idx].astype(np.int32)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    last_score = None
    for s, y in zip(scores, labels):
        if last_score is None:
            last_score = s
        if s != last_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
            last_score = s
        if y == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / P)
    fpr.append(fp / N)
    return np.array(fpr), np.array(tpr)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = _roc_curve(scores, labels)
    auc = np.trapz(tpr, fpr)
    return float(max(0.0, min(1.0, auc)))


def _best_threshold_accuracy(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(scores)
    uniq = np.unique(scores[order])
    best_acc = 0.0
    best_thr = 0.0
    for thr in uniq:
        pred = (scores >= thr).astype(np.int32)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    return best_thr, best_acc


def encrypt_to_3ch(cipher: SCNECipherAPI, images: torch.Tensor, privacy_level: float) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    enc, _ = cipher.encrypt_simple(gray, privacy_level=privacy_level, semantic_preserving=True)
    return enc.repeat(1, 3, 1, 1)


def collect_mia_scores(evaluator: AttackEvaluator, loader, cipher: SCNECipherAPI, privacy_level: float, device: str,
                        max_batches: int, use_encrypted: bool, require_labels: bool = True) -> Tuple[List[float], List[float]]:
    scores, labels_all = [], []
    seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = None
        images = images.to(device)
        if require_labels and labels is not None:
            labels = labels.to(device)
        with torch.no_grad():
            x = encrypt_to_3ch(cipher, images, privacy_level) if use_encrypted else images
            logits = evaluator.attr_classifier(x)
            if require_labels and labels is not None:
                loss = _bce_loss_per_sample(logits, labels)
                score = (-loss).detach().cpu().numpy()
            else:
                p = torch.sigmoid(logits)
                conf = torch.maximum(p, 1 - p).mean(dim=1)
                score = conf.detach().cpu().numpy()
        scores.extend(score.tolist())
        if require_labels:
            labels_all.extend([1] * len(score))  # caller will set non-member labels later
        seen += 1
        if seen >= max_batches:
            break
    return scores, labels_all


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extended Privacy Attacks Evaluation: inversion / attribute / membership inference')
    parser.add_argument('--dataset-root', type=str, default='data/CelebA-HQ')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--privacy-level', type=float, default=0.3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluator = AttackEvaluator(device=device)
    evaluator.load_pretrained('src/models/pretrained')

    cipher = SCNECipherAPI(
        password='evaluation_password_2025',
        image_size=256,
        device=device,
    )

    # dataloaders: train = members, test = non-members
    try:
        train_loader = get_celeba_attr_dataloader(
            root_dir=args.dataset_root,
            split='train',
            batch_size=args.batch_size,
            image_size=256,
            shuffle=False,
            num_workers=args.num_workers,
        )
    except Exception:
        train_loader = None
    try:
        test_loader = get_celeba_attr_dataloader(
            root_dir=args.dataset_root,
            split='test',
            batch_size=args.batch_size,
            image_size=256,
            shuffle=False,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"cannot load test split: {e}")
        return 2

    # Attribute inference (on test split)
    attr_results = {"original_acc": [], "encrypted_acc": []}
    tb = 0
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0].to(device), batch[1].to(device)
        else:
            images = batch.to(device)
            labels = None
        with torch.no_grad():
            logits_o = evaluator.attr_classifier(images)
            enc = encrypt_to_3ch(cipher, images, args.privacy_level)
            logits_e = evaluator.attr_classifier(enc)
            if labels is not None:
                acc_o = ((torch.sigmoid(logits_o) > 0.5) == labels).float().mean().item()
                acc_e = ((torch.sigmoid(logits_e) > 0.5) == labels).float().mean().item()
            else:
                acc_o, acc_e = 0.0, 0.0
        attr_results["original_acc"].append(acc_o)
        attr_results["encrypted_acc"].append(acc_e)
        tb += 1
        if tb >= args.num_batches:
            break
    attr_summary = {
        "original_accuracy": float(np.mean(attr_results["original_acc"])) if attr_results["original_acc"] else 0.0,
        "encrypted_accuracy": float(np.mean(attr_results["encrypted_acc"])) if attr_results["encrypted_acc"] else 0.0,
    }

    # Model inversion (proxy by VAE reconstruction on encrypted)
    inv_psnr, inv_lpips, inv_ssim = [], [], []
    tb = 0
    for batch in test_loader:
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        with torch.no_grad():
            enc = encrypt_to_3ch(cipher, images, args.privacy_level)
            res = evaluator.evaluate_reconstruction_attack(images, enc)
        inv_psnr.append(res['psnr'])
        inv_lpips.append(res['lpips'])
        inv_ssim.append(res['ssim'])
        tb += 1
        if tb >= args.num_batches:
            break
    inversion_summary = {
        "psnr": float(np.mean(inv_psnr)) if inv_psnr else 0.0,
        "lpips": float(np.mean(inv_lpips)) if inv_lpips else 0.0,
        "ssim": float(np.mean(inv_ssim)) if inv_ssim else 0.0,
    }

    # Membership inference by loss thresholding
    mia = {}
    if train_loader is not None:
        # members
        mem_scores_o, _ = collect_mia_scores(evaluator, train_loader, cipher, args.privacy_level, device, args.num_batches, use_encrypted=False)
        mem_scores_e, _ = collect_mia_scores(evaluator, train_loader, cipher, args.privacy_level, device, args.num_batches, use_encrypted=True)
        # non-members
        non_scores_o, _ = collect_mia_scores(evaluator, test_loader, cipher, args.privacy_level, device, args.num_batches, use_encrypted=False)
        non_scores_e, _ = collect_mia_scores(evaluator, test_loader, cipher, args.privacy_level, device, args.num_batches, use_encrypted=True)
        # labels: member=1, nonmember=0
        scores_o = np.array(mem_scores_o + non_scores_o)
        labels_o = np.array([1] * len(mem_scores_o) + [0] * len(non_scores_o))
        scores_e = np.array(mem_scores_e + non_scores_e)
        labels_e = np.array([1] * len(mem_scores_e) + [0] * len(non_scores_e))
        auc_o = _roc_auc(scores_o, labels_o)
        thr_o, acc_o = _best_threshold_accuracy(scores_o, labels_o)
        auc_e = _roc_auc(scores_e, labels_e)
        thr_e, acc_e = _best_threshold_accuracy(scores_e, labels_e)
        mia = {
            "original_auc": auc_o,
            "original_best_thr": thr_o,
            "original_best_acc": acc_o,
            "encrypted_auc": auc_e,
            "encrypted_best_thr": thr_e,
            "encrypted_best_acc": acc_e,
        }
    else:
        mia = {"note": "no train split available; skip MIA"}

    results = {
        "attribute_inference": attr_summary,
        "model_inversion": inversion_summary,
        "membership_inference": mia,
        "privacy_level": args.privacy_level,
    }

    out_dir = Path('results/attacks_ext')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir / 'report.json')

    return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)
