# -*- coding: utf-8 -*-
"""
加密基线实现。

提供 AES-CTR 和 ChaCha20-Poly1305 加密基线用于对比实验。
"""

import os
import hashlib
from typing import Tuple, Dict

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


def aes_ctr_encrypt_uint8(image_u8: np.ndarray, key: bytes, iv: bytes) -> Tuple[np.ndarray, Dict]:
    """
    使用 AES-CTR 模式加密 uint8 图像。
    
    参数:
        image_u8: 输入图像（uint8，2D）
        key: 加密密钥
        iv: 初始化向量
        
    返回:
        加密后的图像和元数据字典
    """
    assert image_u8.dtype == np.uint8 and image_u8.ndim == 2
    data = image_u8.tobytes()
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    enc = cipher.encryptor()
    ct = enc.update(data) + enc.finalize()
    out = np.frombuffer(ct, dtype=np.uint8).reshape(image_u8.shape)
    return out, {"alg": "AES-CTR", "iv_hex": iv.hex(), "key_len": len(key)}


def chacha20_poly1305_encrypt_uint8(image_u8: np.ndarray, key: bytes, nonce: bytes) -> Tuple[np.ndarray, Dict]:
    """
    使用 ChaCha20-Poly1305 加密 uint8 图像。
    
    参数:
        image_u8: 输入图像（uint8，2D）
        key: 加密密钥
        nonce: 随机数
        
    返回:
        加密后的图像和元数据字典
    """
    assert image_u8.dtype == np.uint8 and image_u8.ndim == 2
    data = image_u8.tobytes()
    aead = ChaCha20Poly1305(key)
    ct = aead.encrypt(nonce, data, b"")
    img_len = image_u8.size
    out = np.frombuffer(ct[:img_len], dtype=np.uint8).reshape(image_u8.shape)
    tag_hex = ct[img_len:].hex() if len(ct) > img_len else ""
    return out, {"alg": "ChaCha20-Poly1305", "nonce_hex": nonce.hex(), "key_len": len(key), "tag_hex": tag_hex}


def derive_key_iv(name: str, length: int) -> bytes:
    """
    从名称派生密钥/IV。
    
    参数:
        name: 用于派生的名称字符串
        length: 输出长度
        
    返回:
        派生的字节序列
    """
    return hashlib.sha256(name.encode("utf-8")).digest()[:length]
