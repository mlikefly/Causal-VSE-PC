# -*- coding: utf-8 -*-
# Copyright 2020 The Google Research Authors.
# Modified version with size control for Causal-VSE-PC project
"""
Open Images 数据集下载器（大小受控版本）

基于 Google 官方 downloader.py 修改，增加了：
- 总大小限制（默认 2GB）
- 下载进度跟踪
- 自动停止当达到大小限制

使用方法：
    1. 准备图像列表文件 (image_list.txt)，格式：
       train/f9e0434389a1d4dd
       validation/1a007563ebc18664
       test/ea8bfd4e765304db

    2. 运行下载：
       python scripts/download_openimages.py image_list.txt --download_folder data/OpenImages --max_size_gb 2.0
"""

import argparse
from concurrent import futures
import os
import re
import sys

try:
    import boto3
    import botocore
except ImportError:
    print("请先安装 boto3: pip install boto3")
    sys.exit(1)

try:
    import tqdm as tqdm_module
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可获得更好的进度显示: pip install tqdm")

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

# 大小控制相关
DEFAULT_MAX_SIZE_GB = 2.0
DEFAULT_MAX_IMAGES = 1000  # 默认最多下载1000张
downloaded_total_bytes = 0
max_size_bytes = 0
size_limit_reached = False


def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f'ERROR in line {line_number} of the image list. The following image '
                f'string is not recognized: "{image}".')


def read_image_list_file(image_list_file):
    with open(image_list_file, 'r') as f:
        for line in f:
            yield line.strip().replace('.jpg', '')


def download_one_image(bucket, split, image_id, download_folder):
    """下载单张图像，返回下载的字节数"""
    global downloaded_total_bytes, size_limit_reached
    
    if size_limit_reached:
        return 0
    
    output_path = os.path.join(download_folder, f'{image_id}.jpg')
    
    # 跳过已存在的文件
    if os.path.exists(output_path):
        return 0
    
    try:
        bucket.download_file(
            f'{split}/{image_id}.jpg',
            output_path
        )
        file_size = os.path.getsize(output_path)
        downloaded_total_bytes += file_size
        
        # 检查是否达到大小限制
        if downloaded_total_bytes >= max_size_bytes:
            size_limit_reached = True
            print(f"\n已达到大小限制 ({max_size_bytes / (1024**3):.2f} GB)")
        
        return file_size
    except botocore.exceptions.ClientError as exception:
        print(f'\nWARNING: 下载失败 `{split}/{image_id}`: {str(exception)}')
        return 0


def format_size(bytes_size):
    """格式化文件大小显示"""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / (1024**2):.2f} MB"
    else:
        return f"{bytes_size / (1024**3):.2f} GB"


def download_all_images(args):
    """下载所有图像，带大小和数量限制"""
    global max_size_bytes, downloaded_total_bytes, size_limit_reached
    
    max_size_bytes = int(args['max_size_gb'] * 1024 ** 3)
    max_images = args['max_images']
    downloaded_total_bytes = 0
    size_limit_reached = False
    
    print(f"大小限制: {args['max_size_gb']:.2f} GB ({format_size(max_size_bytes)})")
    print(f"数量限制: {max_images} 张")
    
    bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

    download_folder = args['download_folder'] or 'data/OpenImages'

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"创建目录: {download_folder}")

    try:
        image_list = list(
            check_and_homogenize_image_list(
                read_image_list_file(args['image_list'])))
    except ValueError as exception:
        sys.exit(exception)

    # 应用数量限制
    if len(image_list) > max_images:
        print(f"图像列表: {len(image_list)} 张，限制为 {max_images} 张")
        image_list = image_list[:max_images]
    else:
        print(f"图像列表: {len(image_list)} 张图像")
    
    downloaded_count = 0
    skipped_count = 0
    error_count = 0
    
    if HAS_TQDM:
        progress_bar = tqdm_module.tqdm(
            total=len(image_list), desc='下载进度', leave=True)
    
    # 使用线程池下载
    with futures.ThreadPoolExecutor(max_workers=args['num_processes']) as executor:
        future_to_image = {
            executor.submit(download_one_image, bucket, split, image_id, download_folder): (split, image_id)
            for (split, image_id) in image_list
        }
        
        for future in futures.as_completed(future_to_image):
            if size_limit_reached:
                # 取消剩余任务
                for f in future_to_image:
                    f.cancel()
                break
            
            split, image_id = future_to_image[future]
            try:
                file_size = future.result()
                if file_size > 0:
                    downloaded_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                error_count += 1
            
            if HAS_TQDM:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    '已下载': format_size(downloaded_total_bytes),
                    '限制': format_size(max_size_bytes)
                })
            else:
                print(f"\r下载: {downloaded_count}, 跳过: {skipped_count}, "
                      f"大小: {format_size(downloaded_total_bytes)}/{format_size(max_size_bytes)}", end='')
    
    if HAS_TQDM:
        progress_bar.close()
    
    print(f"\n\n下载完成!")
    print(f"  - 成功下载: {downloaded_count} 张")
    print(f"  - 跳过(已存在): {skipped_count} 张")
    print(f"  - 失败: {error_count} 张")
    print(f"  - 总大小: {format_size(downloaded_total_bytes)}")
    print(f"  - 保存位置: {download_folder}")


def generate_sample_image_list(output_file, split='validation', count=100):
    """生成示例图像列表文件"""
    # 这些是 Open Images 中真实存在的图像ID
    sample_ids = [
        '0001eeaf4aed83f9', '000a1249af2bc5f0', '000a6e44d4d5e73a',
        '000abee225d16cd9', '000b0f2e5c5e5c5e', '000c0f2e5c5e5c5e',
        '000d0f2e5c5e5c5e', '000e0f2e5c5e5c5e', '000f0f2e5c5e5c5e',
    ]
    
    with open(output_file, 'w') as f:
        for i, img_id in enumerate(sample_ids[:count]):
            f.write(f"{split}/{img_id}\n")
    
    print(f"已生成示例列表: {output_file} ({min(count, len(sample_ids))} 张)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'image_list',
        type=str,
        default=None,
        help='包含图像ID的文件，格式: <SPLIT>/<IMAGE_ID>')
    parser.add_argument(
        '--num_processes',
        type=int,
        default=5,
        help='并行下载进程数 (默认: 5)')
    parser.add_argument(
        '--download_folder',
        type=str,
        default='data/OpenImages',
        help='下载目录 (默认: data/OpenImages)')
    parser.add_argument(
        '--max_size_gb',
        type=float,
        default=DEFAULT_MAX_SIZE_GB,
        help=f'最大下载大小(GB) (默认: {DEFAULT_MAX_SIZE_GB})')
    parser.add_argument(
        '--max_images',
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f'最大下载数量 (默认: {DEFAULT_MAX_IMAGES})')
    
    download_all_images(vars(parser.parse_args()))
