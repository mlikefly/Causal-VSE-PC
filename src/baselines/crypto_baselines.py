import os
import hashlib
from typing import Tuple, Dict

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


def aes_ctr_encrypt_uint8(image_u8: np.ndarray, key: bytes, iv: bytes) -> Tuple[np.ndarray, Dict]:
    assert image_u8.dtype == np.uint8 and image_u8.ndim == 2
    data = image_u8.tobytes()
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    enc = cipher.encryptor()
    ct = enc.update(data) + enc.finalize()
    out = np.frombuffer(ct, dtype=np.uint8).reshape(image_u8.shape)
    return out, {"alg": "AES-CTR", "iv_hex": iv.hex(), "key_len": len(key)}


def chacha20_poly1305_encrypt_uint8(image_u8: np.ndarray, key: bytes, nonce: bytes) -> Tuple[np.ndarray, Dict]:
    assert image_u8.dtype == np.uint8 and image_u8.ndim == 2
    data = image_u8.tobytes()
    aead = ChaCha20Poly1305(key)
    ct = aead.encrypt(nonce, data, b"")
    img_len = image_u8.size
    out = np.frombuffer(ct[:img_len], dtype=np.uint8).reshape(image_u8.shape)
    tag_hex = ct[img_len:].hex() if len(ct) > img_len else ""
    return out, {"alg": "ChaCha20-Poly1305", "nonce_hex": nonce.hex(), "key_len": len(key), "tag_hex": tag_hex}


def derive_key_iv(name: str, length: int) -> bytes:
    return hashlib.sha256(name.encode("utf-8")).digest()[:length]
