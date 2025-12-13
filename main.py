#!/usr/bin/env python
"""
Causal-VSE-PC 主入口
====================

使用方法:
    # 运行完整评测
    python main.py evaluate
    
    # 运行加密演示
    python main.py demo --image path/to/image.png --privacy 0.7
    
    # 运行测试
    python main.py test
"""

import argparse
import sys
from pathlib import Path


def run_evaluate():
    """运行完整评测"""
    print("运行完整端到端评测...")
    from scripts.experiments.vse_pc.test_causal_e2e_full import main
    return main()


def run_demo(image_path: str = None, privacy_level: float = 0.7):
    """运行加密演示"""
    import torch
    from src.cipher.scne_cipher import SCNECipherAPI
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 初始化加密器
    cipher = SCNECipherAPI(
        password="demo_password",
        image_size=256,
        device=device
    )
    
    # 加载或生成图像
    if image_path and Path(image_path).exists():
        from PIL import Image
        import torchvision.transforms as T
        
        img = Image.open(image_path).convert('L')
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        image = transform(img).unsqueeze(0).to(device)
        print(f"加载图像: {image_path}")
    else:
        image = torch.rand(1, 1, 256, 256, device=device)
        print("使用随机测试图像")
    
    # 加密
    print(f"\n加密中 (privacy_level={privacy_level})...")
    encrypted, enc_info = cipher.encrypt_simple(image, privacy_level=privacy_level)
    
    mae = (encrypted - image).abs().mean().item()
    print(f"加密完成，MAE: {mae:.4f}")
    
    # 解密
    print("\n解密中...")
    mask = torch.ones_like(image)
    decrypted = cipher.cipher.decrypt(encrypted, enc_info, mask, password="demo_password")
    
    psnr = 10 * torch.log10(1.0 / ((image - decrypted)**2).mean()).item()
    print(f"解密完成，PSNR: {psnr:.2f}dB")
    
    return True


def run_test():
    """运行测试"""
    print("运行核心功能测试...")
    from tests.test_encryption import (
        test_layer1_encryption,
        test_layer2_encryption,
        test_full_scne_encryption,
        test_security_metrics
    )
    
    test_layer1_encryption()
    test_layer2_encryption()
    test_full_scne_encryption()
    test_security_metrics()
    
    print("\n所有测试通过!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Causal-VSE-PC: 因果推断驱动的可验证语义加密',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py evaluate          # 运行完整评测
    python main.py demo              # 运行加密演示
    python main.py demo --privacy 0.5
    python main.py test              # 运行测试
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # evaluate 命令
    subparsers.add_parser('evaluate', help='运行完整端到端评测')
    
    # demo 命令
    demo_parser = subparsers.add_parser('demo', help='运行加密演示')
    demo_parser.add_argument('--image', type=str, default=None, help='输入图像路径')
    demo_parser.add_argument('--privacy', type=float, default=0.7, help='隐私级别 [0-1]')
    
    # test 命令
    subparsers.add_parser('test', help='运行核心功能测试')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        success = run_evaluate()
    elif args.command == 'demo':
        success = run_demo(args.image, args.privacy)
    elif args.command == 'test':
        success = run_test()
    else:
        parser.print_help()
        return 0
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
