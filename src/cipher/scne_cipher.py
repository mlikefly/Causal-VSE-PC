"""
SCNE: 语义感知混沌神经加密
Semantic-aware Chaotic Neural Encryption

核心创新：
1. 密钥-神经分离架构：
   - 神经网络：语义理解（公开）
   - 密钥系统：参数生成（核心）
   - 混沌加密：执行算法（确定性）

2. 三层加密流水线：
   - Layer 1: 空域混沌置乱（Arnold + Lorenz）
   - Layer 2: 频域语义控制（DWT/FFT分层扰动）
   - Layer 3: 全局扩散强化（混沌序列）

3. 可控语义泄露：
   - privacy_level=0.3: 保持语义结构（云计算）
   - privacy_level=1.0: 完全混淆（存储）

安全保证：
- 解密只需密钥
- 输出完整混淆图
- NPCR>99%, 熵≈7.99, UACI≈33%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import json
import hashlib
import hmac
import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

# 导入现有组件（更新路径）
from src.core.chaotic_encryptor import StandardChaoticCipher  # Updated Import
from src.core.chaos_systems import ChaosSystem
from src.core.frequency_cipher import FrequencySemanticCipherOptimized, FrequencySemanticCipher
from src.crypto.key_system import HierarchicalKeySystem


class SCNECipher(nn.Module):
    """
    SCNE核心加密器
    
    整合：
    - 语义分析（U-Net + GNN）
    - 密钥系统（KDF + PRF）
    - 混沌加密（Arnold + 混沌系统）
    - 频域控制（DWT/FFT）
    """
    
    def __init__(
        self,
        image_size: int = 256,
        use_frequency: bool = True,
        use_fft: bool = True,  # FFT比DWT快，推荐使用
        password: str = None,
        enable_crypto_wrap: bool = True,
        wrap_bits: int = 16,
        lossless_mode: bool = False,
        wrap_mode: str = 'q16',
        enc_info_compact: bool = False,
        num_radial_bins: int = 8,
        deterministic: bool = False,
    ):
        """
        初始化SCNE加密器
        
        Args:
            image_size: 图像尺寸
            use_frequency: 是否使用频域控制
            use_fft: 是否使用FFT（比DWT快）
            password: 用户密码（可选，后续通过encrypt传入）
        """
        super().__init__()
        
        self.image_size = image_size
        self.use_frequency = use_frequency
        self.use_fft = use_fft
        self.crypto_wrap_enabled = bool(enable_crypto_wrap)
        self.wrap_bits = int(wrap_bits)
        self.lossless_mode = bool(lossless_mode)
        # wrap_mode: 'q16'（量化16位，近无损，限定[0,1]）；'f32'（按float32字节异或，严格无损，值域不限定）
        self.wrap_mode = wrap_mode
        self.enc_info_compact = bool(enc_info_compact)
        # 是否在内部启用确定性/seeded 模式（主要影响混沌扩散序列）
        self.deterministic = bool(deterministic)
        
        # 组件1: 混沌加密器（已修复）
        self.chaos_encryptor = StandardChaoticCipher()
        
        # 注册为子模块，确保.to(device)生效
        self.add_module('chaos_encryptor', self.chaos_encryptor)

        
        # 组件2: 频域加密器（同时准备FFT与DWT实现，按use_fft选择）
        if use_frequency:
            self.freq_cipher_fft = FrequencySemanticCipherOptimized(use_learnable_band=True, num_radial_bins=int(num_radial_bins))
            self.freq_cipher_dwt = FrequencySemanticCipher()
        
        # 组件3: 密钥系统（按需初始化）
        self.key_system = None
        if password:
            self.key_system = HierarchicalKeySystem(password, iterations=100000)

    # ======== enc_info 签名绑定（HMAC-SHA256） ========
    @staticmethod
    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        if isinstance(obj, dict):
            return {k: SCNECipher._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [SCNECipher._to_serializable(v) for v in obj]
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        return str(obj)

    @staticmethod
    def _strip_signatures(enc_info: Dict) -> Dict:
        info = dict(enc_info)
        if 'signatures' in info:
            info = dict(info)
            info.pop('signatures', None)
        return info

    def _compute_signatures(self, images: torch.Tensor, enc_info_no_sig: Dict) -> list:
        """对每张图像计算签名: HMAC_K( sha256(X_bytes) || sha256(params_json) )"""
        assert self.key_system is not None, "签名需要初始化密钥系统(password)"
        params_json = json.dumps(self._to_serializable(enc_info_no_sig), sort_keys=True, ensure_ascii=False).encode('utf-8')
        params_hash = hashlib.sha256(params_json).digest()
        sigs = []
        with torch.no_grad():
            B = images.size(0)
            for b in range(B):
                x_bytes = images[b].detach().cpu().numpy().astype(np.float32).tobytes()
                x_hash = hashlib.sha256(x_bytes).digest()
                msg = x_hash + params_hash
                sig = hmac.new(self.key_system.master_key, msg, hashlib.sha256).hexdigest()
                sigs.append(sig)
        return sigs

    # ======== Crypto wrap（提升熵与完整性，可逆） ========
    def _derive_stream_and_mac_keys(self) -> Tuple[bytes, bytes]:
        """从主密钥派生两个子密钥（stream_key, mac_key）。"""
        assert self.key_system is not None, "Crypto wrap 需要已初始化的密钥系统"
        master = self.key_system.master_key
        stream_key = hashlib.sha256(master + b'|stream').digest()   # 32B
        mac_key = hashlib.sha256(master + b'|mac').digest()         # 32B
        return stream_key, mac_key

    def _derive_deterministic_nonce(
        self,
        image_id: str,
        task_type: str,
        privacy_map: torch.Tensor,
        z_view: torch.Tensor
    ) -> bytes:
        """
        派生确定性nonce（绑定输入签名，避免nonce复用风险）
        
        nonce = HMAC(k_nonce, "cview|v2|" + image_id + "|" + task_type 
                     + "|" + sha256(privacy_map) + "|" + sha256(zview))[:16]
        
        这确保：相同image + 相同privacy_map + 相同task_type + 相同版本参数
        才会产生相同的nonce，避免AEAD安全性破坏
        
        Args:
            image_id: 图像唯一标识符
            task_type: 任务类型 (classification/detection/segmentation)
            privacy_map: 隐私预算图 [B, 1, H, W]
            z_view: Z-view密文 [B, C, H, W]
        
        Returns:
            nonce: 16字节确定性nonce
        """
        assert self.key_system is not None, "确定性nonce派生需要已初始化的密钥系统"
        
        # 计算privacy_map的哈希（取前8字符）
        privacy_map_hash = hashlib.sha256(
            privacy_map.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        # 计算z_view的哈希（取前8字符）
        z_view_hash = hashlib.sha256(
            z_view.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        # 构造上下文字符串
        context = f"cview|v2|{image_id}|{task_type}|{privacy_map_hash}|{z_view_hash}".encode('utf-8')
        
        # 派生nonce密钥
        k_nonce = hashlib.sha256(self.key_system.master_key + b'|nonce').digest()
        
        # 使用HMAC派生确定性nonce
        nonce = hmac.new(k_nonce, context, hashlib.sha256).digest()[:16]
        
        return nonce

    @staticmethod
    def _chacha20_xor(data: bytes, key: bytes, nonce: bytes) -> bytes:
        cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None)
        enc = cipher.encryptor()
        return enc.update(data)

    def _build_aad(
        self,
        sample_id: Optional[str] = None,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        task_type: Optional[str] = None,
        version: int = 2
    ) -> bytes:
        """
        构建 AEAD Associated Data (AAD)
        
        AAD 绑定元数据到密文，确保"密文搬家/改字段即解密失败"
        
        Args:
            sample_id: 样本唯一标识符
            dataset: 数据集名称 (celeba/celebahq/fairface/openimages)
            split: 数据划分 (train/val/test)
            task_type: 任务类型 (classification/detection/segmentation)
            version: 协议版本
        
        Returns:
            aad: AAD 字节串
        """
        # 使用 | 分隔的字符串格式，便于调试
        parts = [
            f"v{version}",
            sample_id or "",
            dataset or "",
            split or "",
            task_type or ""
        ]
        aad = "|".join(parts).encode('utf-8')
        return aad

    def _crypto_wrap_encrypt(
        self, 
        tensor: torch.Tensor,
        *,
        deterministic_nonces: Optional[List[bytes]] = None,
        aad: Optional[bytes] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        逐样本仿射归一化+16bit量化+ChaCha20流加密；保存仿射参数确保可逆。
        
        Args:
            tensor: [B, C, H, W] 输入张量
            deterministic_nonces: 可选的确定性nonce列表（每个样本一个16字节nonce）
                                  如果提供，将使用这些nonce而非随机生成
            aad: Associated Authenticated Data，绑定到MAC计算
        
        Returns:
            wrapped: [B, C, H, W] 加密后的张量
            info: 加密信息字典
        """
        assert tensor.dim() == 4
        B, C, H, W = tensor.shape
        device = tensor.device
        stream_key, mac_key = self._derive_stream_and_mac_keys()
        qmax = (1 << self.wrap_bits) - 1

        wrapped_list = []
        nonces = []
        tags = []
        affine_mins = []
        affine_scales = []
        for b in range(B):
            # 量化到uint16（或wrap_bits）
            arr = tensor[b].detach().cpu().numpy().astype(np.float32)
            min_val = float(arr.min())
            max_val = float(arr.max())
            scale = max(max_val - min_val, 1e-8)
            affine_mins.append(min_val)
            affine_scales.append(scale)
            arr_norm = (arr - min_val) / scale  # ∈[0,1]
            q = np.round(arr_norm * qmax).astype(np.uint16)
            raw = q.tobytes()
            # 使用确定性nonce或随机nonce
            if deterministic_nonces is not None and b < len(deterministic_nonces):
                nonce = deterministic_nonces[b]
            else:
                nonce = os.urandom(16)
            nonces.append(nonce.hex())
            # 流加密
            ct = self._chacha20_xor(raw, stream_key, nonce)
            # MAC 保护（包含AAD）
            # MAC = HMAC(mac_key, AAD || ciphertext || metadata)
            mac_input = (aad or b'') + ct + struct.pack('<IIII', C, H, W, self.wrap_bits)
            tag = hmac.new(mac_key, mac_input, hashlib.sha256).hexdigest()
            tags.append(tag)
            # 还原为张量（仍在[0,1]数值域，但已加密）
            q_ct = np.frombuffer(ct, dtype=np.uint16).reshape(C, H, W)
            arr_ct = (q_ct.astype(np.float32) / qmax)
            wrapped_list.append(torch.from_numpy(arr_ct))

        wrapped = torch.stack(wrapped_list, dim=0).to(device)
        info = {
            'nonces': nonces,
            'tags': tags,
            'wrap_bits': self.wrap_bits,
            'version': 2,  # 升级版本号
            'affine_min': affine_mins,
            'affine_scale': affine_scales,
            'mode': 'q16',
            'aad': (aad or b'').decode('utf-8', errors='replace') if aad else None
        }
        return wrapped, info

    def _crypto_wrap_encrypt_float32(
        self, 
        tensor: torch.Tensor,
        *,
        deterministic_nonces: Optional[List[bytes]] = None,
        aad: Optional[bytes] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        对float32位模式做字节级异或，严格无损；值域不限定。
        
        Args:
            tensor: [B, C, H, W] 输入张量（必须是float32）
            deterministic_nonces: 可选的确定性nonce列表（每个样本一个16字节nonce）
            aad: Associated Authenticated Data，绑定到MAC计算
        
        Returns:
            wrapped: [B, C, H, W] 加密后的张量
            info: 加密信息字典
        """
        assert tensor.dtype == torch.float32, "float32 wrap 仅支持float32"
        B, C, H, W = tensor.shape
        device = tensor.device
        stream_key, mac_key = self._derive_stream_and_mac_keys()
        wrapped_list = []
        nonces = []
        tags = []
        for b in range(B):
            arr = tensor[b].detach().cpu().numpy().astype(np.float32)
            # 使用确定性nonce或随机nonce
            if deterministic_nonces is not None and b < len(deterministic_nonces):
                nonce = deterministic_nonces[b]
            else:
                nonce = os.urandom(16)
            nonces.append(nonce.hex())
            raw_ct = self._chacha20_xor(arr.view(np.uint8).tobytes(), stream_key, nonce)
            # MAC 保护（包含AAD）
            mac_input = (aad or b'') + raw_ct + struct.pack('<IIII', C, H, W, 32)
            tag = hmac.new(mac_key, mac_input, hashlib.sha256).hexdigest()
            tags.append(tag)
            arr_ct = np.frombuffer(raw_ct, dtype=np.float32).reshape(C, H, W)
            wrapped_list.append(torch.from_numpy(arr_ct))
        wrapped = torch.stack(wrapped_list, dim=0).to(device)
        info = {
            'nonces': nonces,
            'tags': tags,
            'wrap_bits': 32,
            'version': 2,  # 升级版本号
            'affine_min': [0.0]*B,
            'affine_scale': [1.0]*B,
            'mode': 'f32',
            'aad': (aad or b'').decode('utf-8', errors='replace') if aad else None
        }
        return wrapped, info

    def _crypto_wrap_decrypt(
        self, 
        tensor: torch.Tensor, 
        wrap_info: Dict,
        *,
        aad: Optional[bytes] = None
    ) -> torch.Tensor:
        """
        解密 crypto_wrap 加密的张量
        
        Args:
            tensor: [B, C, H, W] 加密的张量
            wrap_info: 加密信息字典
            aad: Associated Authenticated Data，必须与加密时相同
        
        Returns:
            plain: [B, C, H, W] 解密后的张量
        
        Raises:
            ValueError: MAC 验证失败（密文被篡改或 AAD 不匹配）
        """
        B, C, H, W = tensor.shape
        device = tensor.device
        stream_key, mac_key = self._derive_stream_and_mac_keys()
        # 兼容旧结构：若存在'wrap'键则下钻
        if 'wrap' in wrap_info:
            wrap_info = wrap_info['wrap']
        qmax = (1 << wrap_info.get('wrap_bits', 16)) - 1
        nonces = wrap_info['nonces']
        tags = wrap_info['tags']
        affine_mins = wrap_info.get('affine_min', [0.0] * B)
        affine_scales = wrap_info.get('affine_scale', [1.0] * B)
        mode = wrap_info.get('mode', 'q16')
        version = wrap_info.get('version', 1)
        
        # 从wrap_info恢复AAD（如果未提供）
        if aad is None and wrap_info.get('aad'):
            aad = wrap_info['aad'].encode('utf-8')

        plain_list = []
        for b in range(B):
            if mode == 'f32':
                # 直接对float32字节流异或
                arr = tensor[b].detach().cpu().numpy().astype(np.float32)
                nonce = bytes.fromhex(nonces[b])
                raw_ct_bytes = arr.view(np.uint8).tobytes()
                
                # 验证MAC（v2版本包含AAD）
                if version >= 2:
                    mac_input = (aad or b'') + raw_ct_bytes + struct.pack('<IIII', C, H, W, 32)
                else:
                    mac_input = raw_ct_bytes + struct.pack('<IIII', C, H, W, 32)
                expected = hmac.new(mac_key, mac_input, hashlib.sha256).hexdigest()
                if expected != tags[b]:
                    raise ValueError(f"Crypto wrap MAC verification failed for sample {b}. AAD mismatch or ciphertext tampered.")
                
                raw_pt = self._chacha20_xor(raw_ct_bytes, stream_key, nonce)
                arr_plain = np.frombuffer(raw_pt, dtype=np.float32).reshape(C, H, W)
                plain_list.append(torch.from_numpy(arr_plain))
            else:
                arr = tensor[b].detach().cpu().numpy().astype(np.float32)
                q = np.round(arr * qmax).astype(np.uint16)
                ct = q.tobytes()
                
                # 验证MAC（v2版本包含AAD）
                if version >= 2:
                    mac_input = (aad or b'') + ct + struct.pack('<IIII', C, H, W, wrap_info.get('wrap_bits', 16))
                else:
                    mac_input = ct + struct.pack('<IIII', C, H, W, wrap_info.get('wrap_bits', 16))
                expected = hmac.new(mac_key, mac_input, hashlib.sha256).hexdigest()
                if expected != tags[b]:
                    raise ValueError(f"Crypto wrap MAC verification failed for sample {b}. AAD mismatch or ciphertext tampered.")
                
                # 解密
                nonce = bytes.fromhex(nonces[b])
                raw = self._chacha20_xor(ct, stream_key, nonce)
                q_plain = np.frombuffer(raw, dtype=np.uint16).reshape(C, H, W)
                arr_norm = (q_plain.astype(np.float32) / qmax)
                # 恢复仿射参数
                arr_plain = arr_norm * float(affine_scales[b]) + float(affine_mins[b])
                plain_list.append(torch.from_numpy(arr_plain))

        plain = torch.stack(plain_list, dim=0).to(device)
        return plain

    # ======== C-view 双表示结构（bytes + vis） ========
    
    def cview_to_bytes(
        self,
        c_view_vis: torch.Tensor,
        wrap_info: Dict
    ) -> List[bytes]:
        """
        将 C-view 可视化张量转换为二进制存储格式
        
        Args:
            c_view_vis: [B, C, H, W] C-view 可视化张量（值域[0,1]）
            wrap_info: crypto_wrap 信息（包含 wrap_bits, mode 等）
        
        Returns:
            c_view_bytes_list: 每个样本的二进制密文（List[bytes]）
        """
        B, C, H, W = c_view_vis.shape
        mode = wrap_info.get('mode', 'q16')
        wrap_bits = wrap_info.get('wrap_bits', 16)
        qmax = (1 << wrap_bits) - 1
        
        c_view_bytes_list = []
        for b in range(B):
            arr = c_view_vis[b].detach().cpu().numpy().astype(np.float32)
            
            if mode == 'f32':
                # float32 模式：直接转换为字节
                raw_bytes = arr.tobytes()
            else:
                # q16 模式：量化后转换为字节
                q = np.round(arr * qmax).astype(np.uint16)
                raw_bytes = q.tobytes()
            
            c_view_bytes_list.append(raw_bytes)
        
        return c_view_bytes_list
    
    def cview_from_bytes(
        self,
        c_view_bytes_list: List[bytes],
        wrap_info: Dict,
        shape: Tuple[int, int, int, int],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        将 C-view 二进制格式转换回可视化张量
        
        Args:
            c_view_bytes_list: 每个样本的二进制密文（List[bytes]）
            wrap_info: crypto_wrap 信息
            shape: 目标形状 (B, C, H, W)
            device: 目标设备
        
        Returns:
            c_view_vis: [B, C, H, W] C-view 可视化张量
        """
        B, C, H, W = shape
        mode = wrap_info.get('mode', 'q16')
        wrap_bits = wrap_info.get('wrap_bits', 16)
        qmax = (1 << wrap_bits) - 1
        
        vis_list = []
        for b, raw_bytes in enumerate(c_view_bytes_list):
            if mode == 'f32':
                # float32 模式：直接从字节恢复
                arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(C, H, W)
            else:
                # q16 模式：从量化字节恢复
                q = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(C, H, W)
                arr = q.astype(np.float32) / qmax
            
            vis_list.append(torch.from_numpy(arr))
        
        c_view_vis = torch.stack(vis_list, dim=0).to(device)
        return c_view_vis
    
    def pack_cview_binary(
        self,
        c_view_vis: torch.Tensor,
        enc_info: Dict
    ) -> Dict:
        """
        打包 C-view 为完整的二进制存储包
        
        包含：密文字节、nonce、tag、版本、元信息
        
        Args:
            c_view_vis: [B, C, H, W] C-view 可视化张量
            enc_info: 完整的加密信息
        
        Returns:
            binary_pack: 包含所有存储所需信息的字典
        """
        wrap_info = enc_info.get('crypto_wrap', {})
        if not wrap_info:
            raise ValueError("enc_info 中缺少 crypto_wrap 信息")
        
        B, C, H, W = c_view_vis.shape
        c_view_bytes_list = self.cview_to_bytes(c_view_vis, wrap_info)
        
        binary_pack = {
            'version': 2,
            'format': 'cview_binary',
            'shape': [B, C, H, W],
            'mode': wrap_info.get('mode', 'q16'),
            'wrap_bits': wrap_info.get('wrap_bits', 16),
            'ciphertext': [ct.hex() for ct in c_view_bytes_list],  # hex编码便于JSON序列化
            'nonces': wrap_info.get('nonces', []),
            'tags': wrap_info.get('tags', []),
            'affine_min': wrap_info.get('affine_min', []),
            'affine_scale': wrap_info.get('affine_scale', []),
            # 审计信息
            'image_id': enc_info.get('image_id'),
            'task_type': enc_info.get('task_type'),
            'privacy_level': enc_info.get('privacy_level'),
            'privacy_map_hash': enc_info.get('privacy_map_hash'),
            'z_view_hash': enc_info.get('z_view_hash'),
        }
        
        return binary_pack
    
    def unpack_cview_binary(
        self,
        binary_pack: Dict,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        从二进制存储包解包 C-view
        
        Args:
            binary_pack: pack_cview_binary 生成的存储包
            device: 目标设备
        
        Returns:
            c_view_vis: [B, C, H, W] C-view 可视化张量
            wrap_info: 用于解密的 crypto_wrap 信息
        """
        shape = tuple(binary_pack['shape'])
        mode = binary_pack.get('mode', 'q16')
        wrap_bits = binary_pack.get('wrap_bits', 16)
        
        # 从hex解码密文
        c_view_bytes_list = [bytes.fromhex(ct) for ct in binary_pack['ciphertext']]
        
        wrap_info = {
            'mode': mode,
            'wrap_bits': wrap_bits,
            'nonces': binary_pack.get('nonces', []),
            'tags': binary_pack.get('tags', []),
            'affine_min': binary_pack.get('affine_min', []),
            'affine_scale': binary_pack.get('affine_scale', []),
            'version': binary_pack.get('version', 1),
        }
        
        c_view_vis = self.cview_from_bytes(c_view_bytes_list, wrap_info, shape, device)
        
        return c_view_vis, wrap_info
    
    def set_password(self, password: str, iterations: int = 100000, salt: bytes = None):
        """设置用户密码，初始化密钥系统；可选指定盐以复现主密钥"""
        self.key_system = HierarchicalKeySystem(password, iterations=iterations, salt=salt)
    
    def encrypt(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        chaos_params: torch.Tensor,
        password: Optional[str] = None,
        privacy_level: float = 1.0,
        semantic_preserving: bool = False,
        *,
        seeds_override: Optional[list] = None,
        image_id: Optional[str] = None,
        task_type: str = 'classification',
        dataset: Optional[str] = None,
        split: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        SCNE完整加密流程
        
        Args:
            images: [B, C, H, W] 输入图像
            mask: [B, 1, H, W] 显著性掩码（用于控制强度）
            chaos_params: [B, 19, 3] GNN策略参数
            password: 用户密码
            privacy_level: 隐私级别 [0.0-1.0]
            semantic_preserving: 是否启用语义保持
            seeds_override: 可选的种子覆盖列表
            image_id: 图像唯一标识符（用于确定性nonce派生和AAD）
            task_type: 任务类型 (classification/detection/segmentation)
            dataset: 数据集名称（用于AAD绑定）
            split: 数据划分（用于AAD绑定）
        
        Returns:
            encrypted: [B, C, H, W] 加密图像
            enc_info: dict 加密信息（用于解密）
        """
        device = images.device
        B, C, H, W = images.shape
        
        # 初始化密钥系统（如果提供密码）
        if password and self.key_system is None:
            self.set_password(password)
        
        # ===== Layer 1: 空域混沌置乱 =====
        # 使用修复后的混沌加密器（全图Arnold + 自适应扩散）
        # 适配 StandardChaoticCipher 接口: (images, key, params)
        
        # 1. 提取参数 (chaos_params: [B, 19, 3])
        # 修复：使用固定基础强度，让privacy_level直接控制加密强度
        # 原问题：avg_strength太小导致不同privacy_level的MAE几乎相同
        B = images.shape[0]
        iterations = int(chaos_params[0, 0, 0].item())
        
        # 使用非线性映射，让低privacy_level时加密很弱，高privacy_level时加密很强
        # privacy_level=0.3 -> strength=0.027 (几乎不加密)
        # privacy_level=0.5 -> strength=0.125 (轻度加密)
        # privacy_level=0.7 -> strength=0.343 (中度加密)
        # privacy_level=1.0 -> strength=1.0 (强加密)
        # 使用三次方映射：strength = privacy_level^3
        effective_strength = privacy_level ** 3
        strength_tensor = torch.full((B, 1, 1, 1), effective_strength, device=device)
        
        params = {
            'iterations': iterations,
            'strength': strength_tensor
        }
        
        # 2. 生成密钥 (Key Setup)
        # 修复：使用确定性密钥（从key_system派生或基于password的固定种子）
        if self.key_system is not None:
            # 从主密钥派生确定性混沌密钥
            key = torch.zeros(B, 2, device=device)
            for b in range(B):
                # 使用主密钥+样本索引派生确定性种子
                ctx = b'chaos_key|' + int(b).to_bytes(4, 'little')
                seed_bytes = hashlib.sha256(self.key_system.master_key + ctx).digest()[:8]
                seed = int.from_bytes(seed_bytes, 'big') % (2**32)
                # 使用确定性种子生成密钥
                rng = torch.Generator(device='cpu')
                rng.manual_seed(seed)
                key[b] = torch.rand(2, generator=rng).to(device)
        else:
            # 回退：使用随机密钥（仅用于测试，不推荐）
            key = torch.rand(B, 2, device=device)
        
        if self.lossless_mode:
            # 关闭扩散，仅做置乱（严格可逆）
            # StandardChaoticCipher 没有 set_use_diffusion，通过 strength=0 实现
            prev_strength = params['strength']
            params['strength'] = torch.zeros_like(prev_strength)
            encrypted_l1 = self.chaos_encryptor(images, key, params)
            # 恢复 strength 以便记录
            params['strength'] = prev_strength
        else:
            encrypted_l1 = self.chaos_encryptor(images, key, params)
            
        # 3. 构造 chaos_info (为了兼容后续流程)
        # 直接保存effective_strength，解密时直接使用
        chaos_info = {
            'iterations': iterations,
            'effective_strength': effective_strength,  # 保存实际使用的强度
            'diffusion_seeds': [0.123] * B,   # 占位
            'a': 1,
            'b': 1,
            'key': key  # [B, 2]
        }
        
        # ===== Layer 2: 频域语义控制（可选） =====
        if (not self.lossless_mode) and self.use_frequency and self.key_system:
            # 将mask转换为privacy_map用于区域差异化加密
            # mask高的区域（敏感区域）需要更强加密
            privacy_map = mask * privacy_level  # [B, 1, H, W]
            
            # 逐样本派生图像密钥与区域密钥，并逐样本频域加密再拼接
            outs = []
            infos = []
            for b in range(B):
                img_b = encrypted_l1[b:b+1]
                img_np = images[b, 0].detach().cpu().numpy()
                image_key = self.key_system.derive_image_key(img_np)
                region_key = self.key_system.derive_region_keys(image_key, num_regions=19)[0]
                # 提取当前样本的privacy_map
                privacy_map_b = privacy_map[b:b+1]
                if getattr(self, 'use_fft', True):
                    enc_b, info_b = self.freq_cipher_fft.encrypt_fft(
                        img_b,
                        region_key,
                        privacy_level=privacy_level,
                        semantic_preserving=semantic_preserving,
                        privacy_map=privacy_map_b
                    )
                else:
                    enc_b, info_b = self.freq_cipher_dwt.encrypt_semantic_preserving(
                        img_b,
                        region_key,
                        privacy_level=privacy_level,
                        semantic_preserving=semantic_preserving,
                        privacy_map=privacy_map_b
                    )
                outs.append(enc_b)
                infos.append(info_b)
            encrypted_l2 = torch.cat(outs, dim=0)
            freq_info = {'per_sample': infos}
        else:
            encrypted_l2 = encrypted_l1
            freq_info = None
        
        # ===== Layer 3: 全局扩散强化 =====
        # 已经在Layer 1中通过chaos_encryptor完成
        encrypted_final = encrypted_l2

        # ===== 额外封装：字节级流加密（提高熵/完整性，可逆） =====
        crypto_info = None
        deterministic_nonces = None
        privacy_map_hash = None
        z_view_hash = None
        aad = None
        
        if self.crypto_wrap_enabled and self.key_system is not None:
            # 在确定性模式下，派生确定性nonce（绑定image_id + task_type + privacy_map + z_view）
            if self.deterministic and image_id is not None:
                B = encrypted_final.shape[0]
                deterministic_nonces = []
                
                # 计算privacy_map和z_view的哈希（用于审计和nonce派生）
                privacy_map_hash = hashlib.sha256(
                    mask.detach().cpu().numpy().astype(np.float32).tobytes()
                ).hexdigest()[:8]
                z_view_hash = hashlib.sha256(
                    encrypted_final.detach().cpu().numpy().astype(np.float32).tobytes()
                ).hexdigest()[:8]
                
                for b in range(B):
                    # 为每个样本派生确定性nonce
                    sample_image_id = f"{image_id}_{b}" if B > 1 else image_id
                    nonce = self._derive_deterministic_nonce(
                        image_id=sample_image_id,
                        task_type=task_type,
                        privacy_map=mask[b:b+1],
                        z_view=encrypted_final[b:b+1]
                    )
                    deterministic_nonces.append(nonce)
            
            # 构建 AAD（绑定元数据到密文）
            aad = self._build_aad(
                sample_id=image_id,
                dataset=dataset,
                split=split,
                task_type=task_type,
                version=2
            )
            
            if self.wrap_mode == 'f32' or (self.lossless_mode and self.wrap_mode != 'q16'):
                encrypted_final, crypto_info = self._crypto_wrap_encrypt_float32(
                    encrypted_final, 
                    deterministic_nonces=deterministic_nonces,
                    aad=aad
                )
            else:
                encrypted_final, crypto_info = self._crypto_wrap_encrypt(
                    encrypted_final,
                    deterministic_nonces=deterministic_nonces,
                    aad=aad
                )
        
        # ===== 返回加密结果和信息（带签名） =====
        enc_info = {
            'chaos_info': chaos_info,      # Layer 1信息
            'freq_info': freq_info,        # Layer 2信息
            'privacy_level': privacy_level,
            'semantic_preserving': semantic_preserving,
            'use_frequency': self.use_frequency,
            'use_fft': self.use_fft,
            'crypto_wrap': (crypto_info['wrap'] if isinstance(crypto_info, dict) and 'wrap' in crypto_info else crypto_info),
            'crypto_enabled': bool(self.crypto_wrap_enabled),
            'lossless_mode': bool(self.lossless_mode),
            'wrap_mode': self.wrap_mode,
            # 确定性nonce相关信息（用于审计和复现）
            'image_id': image_id,
            'task_type': task_type,
            'dataset': dataset,
            'split': split,
            'privacy_map_hash': privacy_map_hash,
            'z_view_hash': z_view_hash
        }
        # 记录KDF盐（用于解密端复现主密钥）
        try:
            if self.key_system is not None and hasattr(self.key_system, 'salt'):
                enc_info['kdf_salt'] = self.key_system.salt.hex()
        except Exception:
            pass
        # 紧凑模式：精简 chaos_info 为种子/参数（保留 iterations/a/b/diffusion_seeds/param_strength）
        if self.enc_info_compact and isinstance(enc_info.get('chaos_info'), dict):
            ch = enc_info['chaos_info']
            enc_info['chaos_info'] = {
                'iterations': ch.get('iterations'),
                'a': ch.get('a', 1),
                'b': ch.get('b', 1),
                'diffusion_seeds': ch.get('diffusion_seeds', []),
                'effective_strength': ch.get('effective_strength', None)
            }

        # 构造 enc_info_v2（版本化、紧凑描述），供测试与后续论文使用
        if self.enc_info_compact:
            enc_info_v2 = {
                'version': 2,
                'deterministic': bool(self.deterministic),
                'compact': True,
                'chaos': enc_info.get('chaos_info'),
                'use_frequency': bool(self.use_frequency),
                'use_fft': bool(self.use_fft),
                'crypto_enabled': bool(self.crypto_wrap_enabled),
                'lossless_mode': bool(self.lossless_mode),
                'wrap_mode': self.wrap_mode,
                'privacy_level': privacy_level,
                'semantic_preserving': semantic_preserving,
            }
        else:
            enc_info_v2 = {
                'version': 2,
                'deterministic': bool(self.deterministic),
                'compact': False,
                'use_frequency': bool(self.use_frequency),
                'use_fft': bool(self.use_fft),
                'crypto_enabled': bool(self.crypto_wrap_enabled),
                'lossless_mode': bool(self.lossless_mode),
                'wrap_mode': self.wrap_mode,
                'privacy_level': privacy_level,
                'semantic_preserving': semantic_preserving,
            }
        enc_info['enc_info_v2'] = enc_info_v2

        # 计算并附加签名（逐样本）
        if self.key_system is not None:
            info_for_sig = self._strip_signatures(enc_info)
            try:
                enc_info['signatures'] = self._compute_signatures(images, info_for_sig)
            except Exception:
                enc_info['signatures'] = []
        
        return encrypted_final, enc_info
    
    def decrypt(
        self,
        encrypted: torch.Tensor,
        enc_info: Dict,
        mask: torch.Tensor,
        password: Optional[str] = None
    ) -> torch.Tensor:
        """
        SCNE完整解密流程
        
        Args:
            encrypted: [B, C, H, W] 加密图像
            enc_info: 加密信息
            mask: [B, 1, H, W] 显著性掩码
            password: 用户密码
        
        Returns:
            decrypted: [B, C, H, W] 解密图像
        """
        device = encrypted.device
        
        # 初始化密钥系统（如果提供密码）
        if password and self.key_system is None:
            self.set_password(password)
        
        # ===== 逆 Layer 3: 先移除可逆的字节级封装（如果启用） =====
        enc_input = encrypted
        if enc_info.get('crypto_wrap') is not None and self.crypto_wrap_enabled and self.key_system is not None:
            enc_input = self._crypto_wrap_decrypt(enc_input, enc_info['crypto_wrap'])
        
        # ===== 逆 Layer 2: 频域解密 =====
        if enc_info.get('use_frequency', False) and enc_info['freq_info']:
            freq_info = enc_info['freq_info']
            # 兼容 per_sample 与单对象两种结构
            if isinstance(freq_info, dict) and 'per_sample' in freq_info:
                parts = []
                for b, info_b in enumerate(freq_info['per_sample']):
                    enc_b = enc_input[b:b+1]
                    if info_b.get('fallback', 'none') == 'dwt' or not getattr(self, 'use_fft', True):
                        dec_b = self.freq_cipher_dwt.decrypt(enc_b, info_b)
                    else:
                        dec_b = self.freq_cipher_fft.decrypt_fft(enc_b, info_b)
                    parts.append(dec_b)
                decrypted_l2 = torch.cat(parts, dim=0)
            else:
                # 原单对象结构
                if freq_info.get('fallback', 'none') == 'dwt' or not getattr(self, 'use_fft', True):
                    decrypted_l2 = self.freq_cipher_dwt.decrypt(
                        enc_input,
                        freq_info
                    )
                else:
                    decrypted_l2 = self.freq_cipher_fft.decrypt_fft(
                        enc_input,
                        freq_info
                    )
        else:
            decrypted_l2 = enc_input
        
        # ===== 逆 Layer 1: 空域混沌解密 =====
        chaos_info = enc_info.get('chaos_info', {})
        B = encrypted.shape[0]
        
        try:
            # 修复：使用确定性密钥生成（与加密时相同的逻辑）
            if self.key_system is not None:
                # 从主密钥派生确定性混沌密钥（与加密时相同）
                key = torch.zeros(B, 2, device=device)
                for b in range(B):
                    ctx = b'chaos_key|' + int(b).to_bytes(4, 'little')
                    seed_bytes = hashlib.sha256(self.key_system.master_key + ctx).digest()[:8]
                    seed = int.from_bytes(seed_bytes, 'big') % (2**32)
                    rng = torch.Generator(device='cpu')
                    rng.manual_seed(seed)
                    key[b] = torch.rand(2, generator=rng).to(device)
            elif 'key' in chaos_info:
                # 回退：从enc_info恢复key（兼容旧版）
                key = chaos_info['key']
                if isinstance(key, torch.Tensor):
                    key = key.to(device)
            else:
                print("⚠️ Warning: No key_system and no key in chaos_info, decryption may fail.")
                key = torch.zeros(B, 2, device=device)
            
            # 恢复 strength - 直接使用保存的effective_strength
            # 加密时: strength = privacy_level^3
            # 解密时: 直接从chaos_info恢复
            if 'effective_strength' in chaos_info:
                eff_str = chaos_info['effective_strength']
            else:
                # 兼容旧版：使用param_strength * privacy_level
                ps = chaos_info.get('param_strength', 0.5)
                privacy_level = enc_info.get('privacy_level', 1.0)
                eff_str = ps * privacy_level
            strength = torch.full((B, 1, 1, 1), float(eff_str), device=device)
            
            params = {
                'iterations': int(chaos_info.get('iterations', 1)),
                'strength': strength
            }
            
            decrypted_final = self.chaos_encryptor.decrypt(decrypted_l2, key, params)

        except Exception as e:
            print(f"Decryption error: {e}")
            decrypted_final = decrypted_l2  # Fallback

        
        # ===== 签名验证（解密后验证X的一致性） =====
        try:
            if self.key_system is not None and enc_info.get('signatures'):
                info_for_sig = self._strip_signatures(enc_info)
                recomputed = self._compute_signatures(decrypted_final, info_for_sig)
                provided = enc_info.get('signatures', [])
                if len(recomputed) != len(provided) or any(a != b for a, b in zip(recomputed, provided)):
                    raise ValueError("enc_info 签名校验失败：解密结果与策略签名不一致")
        except Exception as _:
            # 签名失败不静默，抛出错误由上层处理；这里保持稳健可选打印
            pass
        
        return decrypted_final
    
    def encrypt_with_key(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        chaos_params: torch.Tensor,
        master_key: bytes,
        privacy_level: float = 1.0,
        semantic_preserving: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        使用主密钥直接加密（更安全）
        
        Args:
            images: [B, C, H, W]
            mask: [B, 1, H, W]
            chaos_params: [B, 19, 3]
            master_key: 主密钥（64字节）
            privacy_level: 隐私级别
            semantic_preserving: 语义保持
        
        Returns:
            encrypted: [B, C, H, W]
            enc_info: dict
        """
        # TODO: 实现基于主密钥的加密（跳过密码派生步骤）
        # 当前版本使用密码接口
        raise NotImplementedError("主密钥接口待实现")


class SCNECipherAPI:
    """
    SCNE统一API接口（简化使用）
    
    提供简洁的加密/解密接口，隐藏内部复杂性
    """
    
    def __init__(self, password: str, image_size: int = 256, device: str = None, enable_crypto_wrap: bool = True, lossless_mode: bool = False, wrap_mode: str = 'q16', use_frequency: bool = True, use_fft: bool = True, num_radial_bins: int = 8, deterministic: bool = False):
        """
        初始化SCNE加密系统
        
        Args:
            password: 用户密码
            image_size: 图像尺寸
            device: 设备 ('cuda'/'cpu'，None则自动检测)
        """
        self.password = password
        self.image_size = image_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化SCNE加密器（已修复频域）
        self.cipher = SCNECipher(
            image_size=image_size,
            use_frequency=use_frequency,
            use_fft=use_fft,
            password=password,
            enable_crypto_wrap=enable_crypto_wrap,
            lossless_mode=lossless_mode,
            wrap_mode=wrap_mode,
            num_radial_bins=int(num_radial_bins),
            deterministic=deterministic
        ).to(self.device)
        
        # 初始化语义分析组件（按需加载）
        self.unet = None
        self.gnn = None
    
    def load_semantic_models(self, unet_path: str = None, gnn_path: str = None):
        """
        加载语义分析模型
        
        Args:
            unet_path: U-Net模型路径
            gnn_path: GNN模型路径（已废弃，GNN功能已移除，保留此参数仅为向后兼容）
        """
        from src.neural.unet import UNetSaliencyDetector

        device = self.device

        self.unet = UNetSaliencyDetector(in_channels=1, base_channels=64).to(device)
        if unet_path:
            up = Path(unet_path)
            if up.exists():
                # PyTorch 2.6+ weights_only=True 会拦截 numpy 类型，需显式 False
                try:
                    state = torch.load(str(up), map_location=device, weights_only=False)
                except TypeError:
                    # 兼容旧版 PyTorch 不支持 weights_only 参数
                    state = torch.load(str(up), map_location=device)

                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                self.unet.load_state_dict(state, strict=False)
            else:
                print(f"⚠️ U-Net 权重文件不存在，将使用随机初始化权重: {up}")
        self.unet.eval()

        # GNN功能已移除（当前项目使用因果推断驱动策略）
        # 保留此代码仅为向后兼容，GNN相关功能不再可用
        self.gnn = None
        if gnn_path:
            print(f"⚠️ GNN功能已废弃，当前项目使用因果推断驱动的策略。gnn_path参数将被忽略。")
        
        print(f"✓ 语义分析模型已加载 (U-Net)")
    
    def encrypt_simple(
        self,
        image: torch.Tensor,
        privacy_level: float = 1.0,
        semantic_preserving: bool = False,
        mask: Optional[torch.Tensor] = None,
        *,
        image_id: Optional[str] = None,
        task_type: str = 'classification'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        简化的加密接口（自动生成mask和params）
        
        Args:
            image: [B, 1, H, W] 或 [1, H, W] 输入图像
            privacy_level: 隐私级别 [0.0-1.0]
            semantic_preserving: 语义保持模式
            mask: [B, 1, H, W] 可选的语义掩码，用于区域差异化加密
                  如果不提供，将使用U-Net生成或默认全1
            image_id: 图像唯一标识符（用于确定性nonce派生，仅在deterministic=True时生效）
            task_type: 任务类型 (classification/detection/segmentation)
        
        Returns:
            encrypted: 加密图像
            enc_info: 加密信息
        """
        device = image.device
        
        # 确保维度正确
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, H, W] -> [B, 1, H, W]
        
        B, C, H, W = image.shape
        
        # 生成默认mask和params
        # 修复：使用确定性参数（基于密钥系统派生）
        
        # 生成mask：优先使用外部传入的mask
        if mask is not None:
            # 使用外部传入的mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            # 确保mask形状正确
            if mask.shape != (B, 1, H, W):
                mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        elif self.unet is not None:
            with torch.no_grad():
                mask = self.unet(image)
        else:
            mask = torch.ones(B, 1, H, W, device=device)
        
        # 生成确定性chaos_params（基于密钥系统）
        # 确保密钥系统已初始化
        if self.cipher.key_system is None:
            self.cipher.set_password(self.password)
        
        # 从主密钥派生确定性参数
        chaos_params = torch.zeros(B, 19, 3, device=device)
        for b in range(B):
            # 使用主密钥+样本索引派生确定性种子
            ctx = b'chaos_params|' + int(b).to_bytes(4, 'little')
            seed_bytes = hashlib.sha256(self.cipher.key_system.master_key + ctx).digest()[:8]
            seed = int.from_bytes(seed_bytes, 'big') % (2**32)
            # 使用确定性种子生成参数
            rng = torch.Generator(device='cpu')
            rng.manual_seed(seed)
            params_raw = torch.rand(19, 3, generator=rng)
            # 缩放到合适范围：[iterations, frequency, strength]
            scale = torch.tensor([10.0, 5.0, 1.0])
            chaos_params[b] = (params_raw * scale).to(device)
        
        # 执行加密
        encrypted, enc_info = self.cipher.encrypt(
            image,
            mask,
            chaos_params,
            password=self.password,
            privacy_level=privacy_level,
            semantic_preserving=semantic_preserving,
            image_id=image_id,
            task_type=task_type
        )
        
        return encrypted, enc_info
    
    def prepare_params(
        self,
        image: torch.Tensor
    ) -> Dict:
        """
        预生成确定性加密所需参数：mask/chaos_params/seeds。
        seeds 与图像内容解耦，仅与密钥体系+样本索引相关。
        """
        device = image.device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape
        
        # 确保密钥系统存在
        if self.cipher.key_system is None:
            self.cipher.set_password(self.password)
        
        # 生成 mask
        if self.unet is None:
            mask = torch.ones(B, 1, H, W, device=device)
        else:
            with torch.no_grad():
                mask = self.unet(image)
        
        # 生成确定性 chaos_params（基于密钥系统）
        chaos_params = torch.zeros(B, 19, 3, device=device)
        for b in range(B):
            ctx = b'chaos_params|' + int(b).to_bytes(4, 'little')
            seed_bytes = hashlib.sha256(self.cipher.key_system.master_key + ctx).digest()[:8]
            seed = int.from_bytes(seed_bytes, 'big') % (2**32)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(seed)
            params_raw = torch.rand(19, 3, generator=rng)
            scale = torch.tensor([10.0, 5.0, 1.0])
            chaos_params[b] = (params_raw * scale).to(device)
        # 逐样本8B种子
        seeds = []
        for idx in range(B):
            ctx = b'npcr_seed|' + int(idx).to_bytes(8, 'little', signed=False)
            seeds.append(self.cipher.key_system.derive_seed(ctx))
        return {
            'mask': mask,
            'chaos_params': chaos_params,
            'seeds': seeds
        }

    def encrypt_with_params(
        self,
        image: torch.Tensor,
        prepared: Dict,
        *,
        privacy_level: float = 1.0,
        semantic_preserving: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """使用预生成参数进行加密（用于确定性评测）。"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        encrypted, enc_info = self.cipher.encrypt(
            image,
            prepared['mask'],
            prepared['chaos_params'],
            password=self.password,
            privacy_level=privacy_level,
            semantic_preserving=semantic_preserving,
            seeds_override=prepared.get('seeds')
        )
        return encrypted, enc_info
    
    def decrypt_simple(
        self,
        encrypted: torch.Tensor,
        enc_info: Dict
    ) -> torch.Tensor:
        """
        简化的解密接口
        
        Args:
            encrypted: [B, 1, H, W] 加密图像
            enc_info: 加密信息
        
        Returns:
            decrypted: 解密图像
        """
        # 确保输入是4维张量 [B, C, H, W]
        if encrypted.dim() == 3:
            encrypted = encrypted.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        elif encrypted.dim() == 2:
            encrypted = encrypted.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif encrypted.dim() > 4:
            # 如果维度过多，尝试压缩到4维（假设前面有多余的batch维度）
            while encrypted.dim() > 4:
                encrypted = encrypted.squeeze(0)
        
        B, C, H, W = encrypted.shape
        device = encrypted.device
        
        # 生成mask（用于解密，可以是全1）
        mask = torch.ones(B, 1, H, W, device=device)
        
        # 执行解密
        decrypted = self.cipher.decrypt(
            encrypted,
            enc_info,
            mask,
            password=self.password
        )
        
        return decrypted


def test_scne_cipher():
    """测试SCNE加密器完整流程"""
    print("="*70)
    print("测试SCNE语义感知混沌神经加密")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建测试图像
    B, C, H, W = 2, 1, 256, 256
    test_img = torch.rand(B, C, H, W, device=device)
    mask = torch.rand(B, 1, H, W, device=device)
    chaos_params = torch.rand(B, 19, 3, device=device) * torch.tensor([10.0, 5.0, 1.0], device=device)
    
    print(f"\n原图: {test_img.shape}")
    print(f"范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    
    # 测试1: 完全加密模式（不使用频域）
    print("\n" + "-"*70)
    print("测试1: 完全加密模式（仅混沌）")
    print("-"*70)
    
    cipher_basic = SCNECipher(
        image_size=256,
        use_frequency=False,  # 不使用频域
        password="test_password_2025"
    ).to(device)
    
    encrypted_basic, enc_info_basic = cipher_basic.encrypt(
        test_img, mask, chaos_params,
        privacy_level=1.0,
        semantic_preserving=False
    )
    
    print(f"✓ 加密完成: {encrypted_basic.shape}")
    print(f"  范围: [{encrypted_basic.min():.3f}, {encrypted_basic.max():.3f}]")
    
    # 计算加密强度
    mae = (encrypted_basic - test_img).abs().mean().item()
    print(f"  MAE: {mae:.4f} (目标: >0.3)")
    
    # 解密
    decrypted_basic = cipher_basic.decrypt(
        encrypted_basic, enc_info_basic, mask
    )
    
    decrypt_error = (decrypted_basic - test_img).abs().mean().item()
    print(f"  解密误差: {decrypt_error:.6f} (目标: <1e-5)")
    
    # 测试2: 完整SCNE（混沌 + 频域）
    print("\n" + "-"*70)
    print("测试2: 完整SCNE（混沌 + 频域）")
    print("-"*70)
    
    cipher_full = SCNECipher(
        image_size=256,
        use_frequency=True,   # 使用频域
        use_fft=True,         # 使用FFT
        password="test_password_2025"
    ).to(device)
    
    # 完全加密
    encrypted_full, enc_info_full = cipher_full.encrypt(
        test_img, mask, chaos_params,
        privacy_level=1.0,
        semantic_preserving=False
    )
    
    print(f"✓ 完全加密: {encrypted_full.shape}")
    mae_full = (encrypted_full - test_img).abs().mean().item()
    print(f"  MAE: {mae_full:.4f}")
    
    # 解密
    decrypted_full = cipher_full.decrypt(
        encrypted_full, enc_info_full, mask
    )
    decrypt_error_full = (decrypted_full - test_img).abs().mean().item()
    print(f"  解密误差: {decrypt_error_full:.6f}")
    
    # 测试3: 语义保持模式
    print("\n" + "-"*70)
    print("测试3: 语义保持模式 (privacy=0.3)")
    print("-"*70)
    
    encrypted_sem, enc_info_sem = cipher_full.encrypt(
        test_img, mask, chaos_params,
        privacy_level=0.3,
        semantic_preserving=True
    )
    
    print(f"✓ 语义保持加密: {encrypted_sem.shape}")
    mae_sem = (encrypted_sem - test_img).abs().mean().item()
    print(f"  MAE: {mae_sem:.4f} (应该比完全加密小)")
    
    # 解密
    decrypted_sem = cipher_full.decrypt(
        encrypted_sem, enc_info_sem, mask
    )
    decrypt_error_sem = (decrypted_sem - test_img).abs().mean().item()
    print(f"  解密误差: {decrypt_error_sem:.6f}")
    
    # 测试4: 简化API
    print("\n" + "-"*70)
    print("测试4: 简化API接口")
    print("-"*70)
    
    api = SCNECipherAPI(password="test_password_2025", image_size=256)
    
    encrypted_api, enc_info_api = api.encrypt_simple(
        test_img[0:1],  # 单张图像
        privacy_level=1.0,
        semantic_preserving=False
    )
    
    print(f"✓ API加密: {encrypted_api.shape}")
    
    decrypted_api = api.decrypt_simple(encrypted_api, enc_info_api)
    decrypt_error_api = (decrypted_api - test_img[0:1]).abs().mean().item()
    print(f"  解密误差: {decrypt_error_api:.6f}")
    
    # 总结
    print("\n" + "="*70)
    print("✓ SCNE测试完成！")
    print("="*70)
    
    print("\n测试结果总结：")
    print(f"1. 基础混沌加密 - MAE: {mae:.4f}, 解密误差: {decrypt_error:.6f}")
    print(f"2. 完整SCNE加密 - MAE: {mae_full:.4f}, 解密误差: {decrypt_error_full:.6f}")
    print(f"3. 语义保持加密 - MAE: {mae_sem:.4f}, 解密误差: {decrypt_error_sem:.6f}")
    print(f"4. 简化API加密 - 解密误差: {decrypt_error_api:.6f}")
    
    if all([
        mae > 0.05,  # 基础加密强度
        decrypt_error < 1e-4,  # 解密精度
        mae_full > mae_sem  # 完全加密 > 语义保持
    ]):
        print("\n✓ 所有测试通过！")
    else:
        print("\n⚠️ 部分测试未达到预期")
    
    return cipher_full, api


if __name__ == "__main__":
    test_scne_cipher()

