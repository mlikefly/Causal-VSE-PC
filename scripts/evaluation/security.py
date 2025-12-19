import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cipher.scne_cipher import SCNECipherAPI
from src.utils.datasets import get_celeba_dataloader
from src.evaluation.security_metrics import SecurityMetrics


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """img: [B,1,H,W] 范围 [0,1] -> np.uint8 [H,W]（第一个样本）"""
    x = img[0, 0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return x


def eval_single_pair(original: torch.Tensor, encrypted: torch.Tensor):
    o = to_uint8(original)
    e = to_uint8(encrypted)
    metrics = SecurityMetrics.evaluate_image(o, e)
    checks = SecurityMetrics.check_security_standards(metrics)
    return metrics, checks


def key_sensitivity_test(api: SCNECipherAPI, img: torch.Tensor, privacy: float) -> dict:
    """通过两个几乎相同的密码近似 1 比特密钥翻转。
    使用 cipher.set_password 实际重新初始化密钥系统。
    保持 salt 稳定以隔离密码差异。
    """
    p1 = "evaluation_password_2025"
    p2 = "evaluation_password_2025."
    # try to keep salt stable across the two passwords
    salt = None
    try:
        ks = getattr(api.cipher, 'key_system', None)
        if ks is not None and hasattr(ks, 'salt'):
            salt = ks.salt
    except Exception:
        salt = None
    try:
        api.cipher.set_password(p1, salt=salt)
    except Exception:
        api.cipher.set_password(p1)
    enc1, _ = api.encrypt_simple(img, privacy_level=privacy, semantic_preserving=False)
    try:
        api.cipher.set_password(p2, salt=salt)
    except Exception:
        api.cipher.set_password(p2)
    enc2, _ = api.encrypt_simple(img, privacy_level=privacy, semantic_preserving=False)
    c1 = to_uint8(enc1)
    c2 = to_uint8(enc2)
    npcr = SecurityMetrics.calculate_npcr(c1, c2)
    uaci = SecurityMetrics.calculate_uaci(c1, c2)
    return {
        'password1': p1,
        'password2': p2,
        'npcr_c1_c2': float(npcr),
        'uaci_c1_c2': float(uaci)
    }


def run_security_ext(
    dataset_root: str,
    device: str = 'cuda',
    image_size: int = 256,
    privacy_level: float = 1.0,
    use_frequency: bool = True,
    use_fft: bool = True,
    wrap_mode: str = 'q16',
    num_batches: int = 3
):
    device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # dataloader (no labels)
    dataloader = get_celeba_dataloader(
        root_dir=dataset_root,
        split='test',
        batch_size=1,
        image_size=image_size,
        return_labels=False,
        num_workers=0,
        shuffle=False
    )

    api = SCNECipherAPI(
        password="evaluation_password_2025",
        image_size=image_size,
        device=device,
        use_frequency=use_frequency,
        use_fft=use_fft,
        enable_crypto_wrap=(wrap_mode != 'off'),
        wrap_mode=wrap_mode,
    )

    results = {
        'settings': {
            'device': device,
            'image_size': image_size,
            'privacy_level': privacy_level,
            'use_frequency': use_frequency,
            'use_fft': use_fft,
            'wrap_mode': wrap_mode,
        },
        'pairs': [],
        'key_sensitivity': [],
        'summary': {}
    }

    pairs_metrics = []
    pairs_checks = []

    it = 0
    for batch in dataloader:
        if it >= num_batches:
            break
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device)
        if images.size(1) == 3:
            gray = images.mean(dim=1, keepdim=True)
        else:
            gray = images

        # encrypt/decrypt pair metrics
        enc, enc_info = api.encrypt_simple(
            gray, privacy_level=privacy_level, semantic_preserving=False
        )
        m, c = eval_single_pair(gray, enc)
        pairs_metrics.append(m)
        pairs_checks.append(c)
        results['pairs'].append({'metrics': m, 'checks': c})

        # key sensitivity on same input
        ks = key_sensitivity_test(api, gray, privacy=privacy_level)
        results['key_sensitivity'].append(ks)

        it += 1

    # aggregate summary
    def avg(arr, key):
        vals = [a[key] for a in arr]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    if len(pairs_metrics) > 0:
        summary = {
            'entropy_encrypted_mean': avg(pairs_metrics, 'entropy_encrypted'),
            'npcr_mean': avg(pairs_metrics, 'npcr'),
            'uaci_mean': avg(pairs_metrics, 'uaci'),
            'corr_enc_h_mean': avg(pairs_metrics, 'corr_encrypted_horizontal'),
            'corr_enc_v_mean': avg(pairs_metrics, 'corr_encrypted_vertical'),
            'corr_enc_d_mean': avg(pairs_metrics, 'corr_encrypted_diagonal'),
            'chi2_pass_rate': float(np.mean([1.0 if a['chi2_pass'] else 0.0 for a in pairs_metrics])),
            'nist_monobit_pass_rate': float(np.mean([1.0 if a.get('nist_monobit_pass', False) else 0.0 for a in pairs_metrics])),
            'nist_runs_pass_rate': float(np.mean([1.0 if a.get('nist_runs_pass', False) else 0.0 for a in pairs_metrics]))
        }
    else:
        summary = {}

    if len(results['key_sensitivity']) > 0:
        ks_npcr = float(np.mean([x['npcr_c1_c2'] for x in results['key_sensitivity']]))
        ks_uaci = float(np.mean([x['uaci_c1_c2'] for x in results['key_sensitivity']]))
        summary.update({'key_sensitivity_npcr_mean': ks_npcr, 'key_sensitivity_uaci_mean': ks_uaci})

    # key space estimation from key system (theoretical)
    try:
        if getattr(api, 'cipher', None) is not None and getattr(api.cipher, 'key_system', None) is not None:
            ks_desc, ks_bits = api.cipher.key_system.compute_key_space()
            summary.update({'key_space_desc': ks_desc, 'key_space_bits': int(ks_bits)})
    except Exception:
        pass

    results['summary'] = summary

    # save
    out_dir = Path('results/security_ext')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(out_dir / f'security_ext_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # print brief report
    print("\n=== Security-Ext Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {out_dir / f'security_ext_{ts}.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCNE Security Extended Evaluation')
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--privacy-level', type=float, default=1.0)
    parser.add_argument('--use-frequency', action='store_true')
    parser.add_argument('--use-fft', action='store_true')
    parser.add_argument('--wrap', choices=['off', 'q16', 'f32'], default='q16')
    parser.add_argument('--num-batches', type=int, default=3)
    args = parser.parse_args()

    run_security_ext(
        dataset_root=args.dataset_root,
        device=args.device,
        image_size=args.image_size,
        privacy_level=args.privacy_level,
        use_frequency=args.use_frequency,
        use_fft=args.use_fft,
        wrap_mode=args.wrap,
        num_batches=args.num_batches
    )
