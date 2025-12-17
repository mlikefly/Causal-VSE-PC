# -*- coding: utf-8 -*-
"""
Open Images 数据集直接下载器

直接从 S3 下载 Open Images 图像，无需元数据文件。
使用 boto3 列出 bucket 中的图像并下载。

使用方法：
    python scripts/download_openimages_direct.py --split validation --count 1000 --max_size_gb 2.0
"""

import argparse
import os
import sys
from concurrent import futures

try:
    import boto3
    import botocore
except ImportError:
    print("请先安装 boto3: pip install boto3")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

BUCKET_NAME = 'open-images-dataset'
DEFAULT_MAX_SIZE_GB = 2.0
DEFAULT_MAX_IMAGES = 1000
DEFAULT_DOWNLOAD_FOLDER = 'data/OpenImages'


class OpenImagesDownloader:
    def __init__(self, max_size_gb=2.0, max_images=1000, download_folder=None):
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.max_images = max_images
        self.download_folder = download_folder or DEFAULT_DOWNLOAD_FOLDER
        self.downloaded_bytes = 0
        self.downloaded_count = 0
        self.size_limit_reached = False
        
        # 初始化 S3 客户端（无需认证）
        self.s3 = boto3.resource(
            's3',
            config=botocore.config.Config(signature_version=botocore.UNSIGNED)
        )
        self.bucket = self.s3.Bucket(BUCKET_NAME)
        
        os.makedirs(self.download_folder, exist_ok=True)
    
    def format_size(self, bytes_size):
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 ** 2:
            return f"{bytes_size / 1024:.2f} KB"
        elif bytes_size < 1024 ** 3:
            return f"{bytes_size / (1024**2):.2f} MB"
        else:
            return f"{bytes_size / (1024**3):.2f} GB"
    
    def list_images(self, split='validation', max_count=1000):
        """列出 bucket 中的图像"""
        print(f"正在列出 {split} 图像...")
        
        image_keys = []
        prefix = f"{split}/"
        
        try:
            for obj in self.bucket.objects.filter(Prefix=prefix):
                if obj.key.endswith('.jpg'):
                    image_keys.append(obj.key)
                    if len(image_keys) >= max_count:
                        break
                    if len(image_keys) % 1000 == 0:
                        print(f"  已找到 {len(image_keys)} 张...")
        except Exception as e:
            print(f"列出图像失败: {e}")
            return []
        
        print(f"找到 {len(image_keys)} 张图像")
        return image_keys
    
    def download_one(self, image_key):
        """下载单张图像"""
        if self.size_limit_reached or self.downloaded_count >= self.max_images:
            return 0
        
        filename = os.path.basename(image_key)
        output_path = os.path.join(self.download_folder, filename)
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            return 0
        
        try:
            self.bucket.download_file(image_key, output_path)
            file_size = os.path.getsize(output_path)
            self.downloaded_bytes += file_size
            self.downloaded_count += 1
            
            # 检查限制
            if self.downloaded_bytes >= self.max_size_bytes:
                self.size_limit_reached = True
                print(f"\n已达到大小限制 ({self.format_size(self.max_size_bytes)})")
            
            return file_size
        except Exception as e:
            return 0
    
    def download_all(self, split='validation', num_workers=5):
        """下载所有图像"""
        print(f"\n=== Open Images 下载器 ===")
        print(f"分割: {split}")
        print(f"数量限制: {self.max_images} 张")
        print(f"大小限制: {self.format_size(self.max_size_bytes)}")
        print(f"保存目录: {self.download_folder}")
        print()
        
        # 列出图像
        image_keys = self.list_images(split, self.max_images)
        if not image_keys:
            print("没有找到图像")
            return
        
        # 下载
        skipped = 0
        errors = 0
        
        if HAS_TQDM:
            pbar = tqdm(total=len(image_keys), desc='下载进度')
        
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_key = {
                executor.submit(self.download_one, key): key
                for key in image_keys
            }
            
            for future in futures.as_completed(future_to_key):
                if self.size_limit_reached or self.downloaded_count >= self.max_images:
                    for f in future_to_key:
                        f.cancel()
                    break
                
                try:
                    size = future.result()
                    if size == 0:
                        skipped += 1
                except:
                    errors += 1
                
                if HAS_TQDM:
                    pbar.update(1)
                    pbar.set_postfix({
                        '已下载': self.format_size(self.downloaded_bytes),
                        '数量': self.downloaded_count
                    })
                else:
                    print(f"\r下载: {self.downloaded_count}, "
                          f"大小: {self.format_size(self.downloaded_bytes)}", end='')
        
        if HAS_TQDM:
            pbar.close()
        
        print(f"\n\n=== 下载完成 ===")
        print(f"成功下载: {self.downloaded_count} 张")
        print(f"跳过: {skipped} 张")
        print(f"失败: {errors} 张")
        print(f"总大小: {self.format_size(self.downloaded_bytes)}")
        print(f"保存位置: {self.download_folder}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='数据集分割 (默认: validation)')
    parser.add_argument(
        '--count',
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f'最大下载数量 (默认: {DEFAULT_MAX_IMAGES})')
    parser.add_argument(
        '--max_size_gb',
        type=float,
        default=DEFAULT_MAX_SIZE_GB,
        help=f'最大下载大小(GB) (默认: {DEFAULT_MAX_SIZE_GB})')
    parser.add_argument(
        '--download_folder',
        type=str,
        default=DEFAULT_DOWNLOAD_FOLDER,
        help=f'下载目录 (默认: {DEFAULT_DOWNLOAD_FOLDER})')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=5,
        help='并行下载线程数 (默认: 5)')
    
    args = parser.parse_args()
    
    downloader = OpenImagesDownloader(
        max_size_gb=args.max_size_gb,
        max_images=args.count,
        download_folder=args.download_folder
    )
    
    downloader.download_all(
        split=args.split,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
