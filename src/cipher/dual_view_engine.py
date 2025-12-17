# -*- coding: utf-8 -*-
"""
双视图加密引擎 (Dual-View Encryption Engine)

实现 Z-view 和 C-view 双视图加密：
- Z-view: 可视化密文（Layer 1 + Layer 2），用于云端ML训练
- C-view: 完整密文（Z-view + AEAD），用于安全存储

Requirements: 1.1, 1.2, 1.3, 1.6
"""

import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict

from src.cipher.scne_cipher import SCNECipher
from src.crypto.key_system import HierarchicalKeySystem


@dataclass
class EncryptionResult:
    """加密结果数据类"""
    z_view: torch.Tensor  # [B, C, H, W] Z-view 可视化密文
    c_view: torch.Tensor  # [B, C, H, W] C-view 完整密文
    enc_info: Dict[str, Any]  # 加密信息
    
    def to_dict(self) -> Dict:
        """转换为字典（不包含张量）"""
        return {
            'z_view_shape': list(self.z_view.shape),
            'c_view_shape': list(self.c_view.shape),
            'enc_info': self.enc_info
        }


class DualViewEncryptionEngine:
    """
    双视图加密引擎
    
    整合 SCNECipher 和 HierarchicalKeySystem，提供：
    - Z-view 加密：空域混沌置乱 + 频域语义控制
    - C-view 加密：Z-view + ChaCha20-Poly1305 AEAD
    - 确定性 nonce 派生
    - 完整的加密信息审计
    """
    
    PRIVACY_LEVELS = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    def __init__(
        self,
        password: str,
        image_size: int = 256,
        device: str = None,
        use_frequency: bool = True,
        use_fft: bool = True,
        deterministic: bool = True,
        wrap_mode: str = 'q16',
        num_radial_bins: int = 8
    ):
        """
        初始化双视图加密引擎
        
        Args:
            password: 用户密码
            image_size: 图像尺寸
            device: 计算设备
            use_frequency: 是否使用频域加密
            use_fft: 是否使用FFT（比DWT快）
            deterministic: 是否使用确定性模式
            wrap_mode: AEAD封装模式 ('q16' 或 'f32')
            num_radial_bins: 频域径向分区数
        """
        self.password = password
        self.image_size = image_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.deterministic = deterministic
        self.wrap_mode = wrap_mode
        
        # 初始化密钥系统
        self.key_system = HierarchicalKeySystem(
            password, 
            iterations=100000,
            deterministic=deterministic
        )
        
        # 初始化 Z-view 加密器（不启用 crypto_wrap）
        self.z_cipher = SCNECipher(
            image_size=image_size,
            use_frequency=use_frequency,
            use_fft=use_fft,
            password=password,
            enable_crypto_wrap=False,  # Z-view 不使用 AEAD
            lossless_mode=False,
            wrap_mode=wrap_mode,
            num_radial_bins=num_radial_bins,
            deterministic=deterministic
        )
        # 移动到目标设备
        self.z_cipher = self.z_cipher.to(self.device)
        # 确保内部组件也在正确设备上
        if hasattr(self.z_cipher, 'chaos_encryptor') and hasattr(self.z_cipher.chaos_encryptor, 'device'):
            self.z_cipher.chaos_encryptor.device = self.device
        
        # 初始化 C-view 加密器（启用 crypto_wrap）
        self.c_cipher = SCNECipher(
            image_size=image_size,
            use_frequency=use_frequency,
            use_fft=use_fft,
            password=password,
            enable_crypto_wrap=True,  # C-view 使用 AEAD
            lossless_mode=False,
            wrap_mode=wrap_mode,
            num_radial_bins=num_radial_bins,
            deterministic=deterministic
        )
        # 移动到目标设备
        self.c_cipher = self.c_cipher.to(self.device)
        # 确保内部组件也在正确设备上
        if hasattr(self.c_cipher, 'chaos_encryptor') and hasattr(self.c_cipher.chaos_encryptor, 'device'):
            self.c_cipher.chaos_encryptor.device = self.device
    
    def encrypt(
        self,
        images: torch.Tensor,
        privacy_map: torch.Tensor,
        privacy_level: float = 1.0,
        image_id: str = None,
        task_type: str = 'classification',
        dataset: str = None,
        split: str = None,
        semantic_preserving: bool = False
    ) -> EncryptionResult:
        """
        双视图加密
        
        **Property 1: Dual-View Encryption Completeness**
        同时生成 Z-view 和 C-view
        
        Args:
            images: [B, C, H, W] 输入图像
            privacy_map: [B, 1, H, W] 隐私预算图
            privacy_level: 全局隐私级别 [0.0, 1.0]
            image_id: 图像唯一标识符
            task_type: 任务类型
            dataset: 数据集名称
            split: 数据划分
            semantic_preserving: 是否启用语义保持
        
        Returns:
            EncryptionResult: 包含 z_view, c_view, enc_info
        """
        B, C, H, W = images.shape
        device = images.device
        
        # 确保输入在正确的设备上
        images = images.to(self.device)
        privacy_map = privacy_map.to(self.device)
        
        # 生成默认 chaos_params
        chaos_params = self._generate_chaos_params(B, device)
        
        # ===== Step 1: 生成 Z-view (Layer 1 + Layer 2) =====
        z_view, z_info = self.z_cipher.encrypt(
            images=images,
            mask=privacy_map,
            chaos_params=chaos_params,
            password=self.password,
            privacy_level=privacy_level,
            semantic_preserving=semantic_preserving,
            image_id=image_id,
            task_type=task_type,
            dataset=dataset,
            split=split
        )
        
        # ===== Step 2: 生成 C-view (Z-view + AEAD) =====
        # 计算 Z-view 哈希（用于确定性 nonce 派生）
        z_view_hash = hashlib.sha256(
            z_view.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        # 计算 privacy_map 哈希
        privacy_map_hash = hashlib.sha256(
            privacy_map.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        # 使用 C-view 加密器对 Z-view 进行 AEAD 封装
        c_view, c_info = self.c_cipher.encrypt(
            images=images,
            mask=privacy_map,
            chaos_params=chaos_params,
            password=self.password,
            privacy_level=privacy_level,
            semantic_preserving=semantic_preserving,
            image_id=image_id,
            task_type=task_type,
            dataset=dataset,
            split=split
        )
        
        # ===== Step 3: 构建完整的加密信息 =====
        enc_info = {
            'version': 2,
            'image_id': image_id,
            'task_type': task_type,
            'dataset': dataset,
            'split': split,
            'privacy_level': privacy_level,
            'semantic_preserving': semantic_preserving,
            # Z-view 信息
            'z_view': {
                'chaos_info': z_info.get('chaos_info'),
                'freq_info': z_info.get('freq_info'),
                'hash': z_view_hash
            },
            # C-view 信息
            'c_view': {
                'crypto_wrap': c_info.get('crypto_wrap'),
                'aad': c_info.get('crypto_wrap', {}).get('aad') if c_info.get('crypto_wrap') else None
            },
            # 审计信息
            'audit': {
                'privacy_map_hash': privacy_map_hash,
                'z_view_hash': z_view_hash,
                'deterministic': self.deterministic
            },
            # KDF 信息（用于解密）
            'kdf_salt': self.key_system.salt.hex()
        }
        
        return EncryptionResult(
            z_view=z_view,
            c_view=c_view,
            enc_info=enc_info
        )
    
    def encrypt_zview_only(
        self,
        images: torch.Tensor,
        privacy_map: torch.Tensor,
        privacy_level: float = 1.0,
        image_id: str = None,
        task_type: str = 'classification',
        semantic_preserving: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        仅生成 Z-view（用于云端ML训练）
        
        **Requirements 1.2**: Z-view = Layer 1 + Layer 2
        
        Args:
            images: [B, C, H, W] 输入图像
            privacy_map: [B, 1, H, W] 隐私预算图
            privacy_level: 全局隐私级别
            image_id: 图像标识符
            task_type: 任务类型
            semantic_preserving: 是否启用语义保持
        
        Returns:
            z_view: [B, C, H, W] Z-view 密文
            z_info: Z-view 加密信息
        """
        # 确保输入在正确的设备上
        images = images.to(self.device)
        privacy_map = privacy_map.to(self.device)
        
        B = images.shape[0]
        chaos_params = self._generate_chaos_params(B, self.device)
        
        z_view, z_info = self.z_cipher.encrypt(
            images=images,
            mask=privacy_map,
            chaos_params=chaos_params,
            password=self.password,
            privacy_level=privacy_level,
            semantic_preserving=semantic_preserving,
            image_id=image_id,
            task_type=task_type
        )
        
        return z_view, z_info
    
    def encrypt_cview_from_zview(
        self,
        z_view: torch.Tensor,
        privacy_map: torch.Tensor,
        z_info: Dict,
        image_id: str = None,
        task_type: str = 'classification',
        dataset: str = None,
        split: str = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        从 Z-view 生成 C-view（添加 AEAD 封装）
        
        **Requirements 1.3**: C-view = Z-view + AEAD
        
        Args:
            z_view: [B, C, H, W] Z-view 密文
            privacy_map: [B, 1, H, W] 隐私预算图
            z_info: Z-view 加密信息
            image_id: 图像标识符
            task_type: 任务类型
            dataset: 数据集名称
            split: 数据划分
        
        Returns:
            c_view: [B, C, H, W] C-view 密文
            c_info: C-view 加密信息（包含 AEAD tag/nonce）
        """
        # 确保输入在正确的设备上
        z_view = z_view.to(self.device)
        privacy_map = privacy_map.to(self.device)
        
        B = z_view.shape[0]
        
        # 计算哈希用于确定性 nonce
        z_view_hash = hashlib.sha256(
            z_view.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        privacy_map_hash = hashlib.sha256(
            privacy_map.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()[:8]
        
        # 构建 AAD
        aad = self.c_cipher._build_aad(
            sample_id=image_id,
            dataset=dataset,
            split=split,
            task_type=task_type,
            version=2
        )
        
        # 派生确定性 nonce
        deterministic_nonces = []
        if self.deterministic and image_id is not None:
            for b in range(B):
                sample_image_id = f"{image_id}_{b}" if B > 1 else image_id
                nonce = self.c_cipher._derive_deterministic_nonce(
                    image_id=sample_image_id,
                    task_type=task_type,
                    privacy_map=privacy_map[b:b+1],
                    z_view=z_view[b:b+1]
                )
                deterministic_nonces.append(nonce)
        
        # 应用 AEAD 封装
        if self.wrap_mode == 'f32':
            c_view, crypto_info = self.c_cipher._crypto_wrap_encrypt_float32(
                z_view,
                deterministic_nonces=deterministic_nonces if deterministic_nonces else None,
                aad=aad
            )
        else:
            c_view, crypto_info = self.c_cipher._crypto_wrap_encrypt(
                z_view,
                deterministic_nonces=deterministic_nonces if deterministic_nonces else None,
                aad=aad
            )
        
        c_info = {
            'crypto_wrap': crypto_info,
            'z_info': z_info,
            'audit': {
                'privacy_map_hash': privacy_map_hash,
                'z_view_hash': z_view_hash
            }
        }
        
        return c_view, c_info
    
    def decrypt_zview(
        self,
        z_view: torch.Tensor,
        z_info: Dict,
        privacy_map: torch.Tensor
    ) -> torch.Tensor:
        """
        解密 Z-view
        
        Args:
            z_view: [B, C, H, W] Z-view 密文
            z_info: Z-view 加密信息
            privacy_map: [B, 1, H, W] 隐私预算图
        
        Returns:
            decrypted: [B, C, H, W] 解密图像
        """
        # 构建完整的 enc_info
        enc_info = {
            'chaos_info': z_info.get('chaos_info'),
            'freq_info': z_info.get('freq_info'),
            'privacy_level': z_info.get('privacy_level', 1.0),
            'use_frequency': z_info.get('use_frequency', True),
            'use_fft': z_info.get('use_fft', True),
            'crypto_wrap': None  # Z-view 没有 AEAD
        }
        
        return self.z_cipher.decrypt(
            encrypted=z_view,
            enc_info=enc_info,
            mask=privacy_map,
            password=self.password
        )
    
    def decrypt_cview(
        self,
        c_view: torch.Tensor,
        c_info: Dict,
        privacy_map: torch.Tensor,
        image_id: str = None,
        task_type: str = 'classification',
        dataset: str = None,
        split: str = None
    ) -> torch.Tensor:
        """
        解密 C-view
        
        Args:
            c_view: [B, C, H, W] C-view 密文
            c_info: C-view 加密信息
            privacy_map: [B, 1, H, W] 隐私预算图
            image_id: 图像标识符（用于 AAD 验证）
            task_type: 任务类型
            dataset: 数据集名称
            split: 数据划分
        
        Returns:
            decrypted: [B, C, H, W] 解密图像
        """
        # 构建 AAD（必须与加密时相同）
        aad = self.c_cipher._build_aad(
            sample_id=image_id,
            dataset=dataset,
            split=split,
            task_type=task_type,
            version=2
        )
        
        # 先解密 AEAD 层
        crypto_wrap = c_info.get('crypto_wrap')
        if crypto_wrap:
            z_view = self.c_cipher._crypto_wrap_decrypt(c_view, crypto_wrap, aad=aad)
        else:
            z_view = c_view
        
        # 再解密 Z-view 层
        z_info = c_info.get('z_info', {})
        enc_info = {
            'chaos_info': z_info.get('chaos_info'),
            'freq_info': z_info.get('freq_info'),
            'privacy_level': z_info.get('privacy_level', 1.0),
            'use_frequency': z_info.get('use_frequency', True),
            'use_fft': z_info.get('use_fft', True),
            'crypto_wrap': None
        }
        
        return self.z_cipher.decrypt(
            encrypted=z_view,
            enc_info=enc_info,
            mask=privacy_map,
            password=self.password
        )
    
    def _generate_chaos_params(
        self,
        batch_size: int,
        device: str
    ) -> torch.Tensor:
        """生成默认混沌参数"""
        # [B, 19, 3] - 19个区域，每个区域3个参数
        chaos_params = torch.zeros(batch_size, 19, 3, device=device)
        
        # 设置默认迭代次数
        chaos_params[:, :, 0] = 3  # iterations
        chaos_params[:, :, 1] = 0.5  # strength
        chaos_params[:, :, 2] = 1.0  # scale
        
        return chaos_params
    
    def pack_cview_for_storage(
        self,
        c_view: torch.Tensor,
        enc_info: Dict
    ) -> Dict:
        """
        打包 C-view 用于存储
        
        Args:
            c_view: [B, C, H, W] C-view 密文
            enc_info: 加密信息
        
        Returns:
            storage_pack: 存储包（可序列化为JSON）
        """
        # 从 enc_info 中提取 crypto_wrap 信息
        # DualViewEncryptionEngine 的 enc_info 结构: {'c_view': {'crypto_wrap': ...}}
        crypto_wrap = None
        if 'c_view' in enc_info and 'crypto_wrap' in enc_info['c_view']:
            crypto_wrap = enc_info['c_view']['crypto_wrap']
        elif 'crypto_wrap' in enc_info:
            crypto_wrap = enc_info['crypto_wrap']
        
        if crypto_wrap is None:
            raise ValueError("enc_info 中缺少 crypto_wrap 信息")
        
        # 构建适配 SCNECipher.pack_cview_binary 的 enc_info
        adapted_enc_info = {
            'crypto_wrap': crypto_wrap,
            'image_id': enc_info.get('image_id'),
            'task_type': enc_info.get('task_type'),
            'privacy_level': enc_info.get('privacy_level'),
            'privacy_map_hash': enc_info.get('audit', {}).get('privacy_map_hash'),
            'z_view_hash': enc_info.get('audit', {}).get('z_view_hash') or enc_info.get('z_view', {}).get('hash')
        }
        
        return self.c_cipher.pack_cview_binary(c_view, adapted_enc_info)
    
    def unpack_cview_from_storage(
        self,
        storage_pack: Dict,
        device: str = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        从存储包解包 C-view
        
        Args:
            storage_pack: 存储包
            device: 目标设备
        
        Returns:
            c_view: [B, C, H, W] C-view 密文
            wrap_info: AEAD 信息
        """
        device = device or self.device
        return self.c_cipher.unpack_cview_binary(storage_pack, device)
    
    def get_supported_privacy_levels(self) -> List[float]:
        """获取支持的隐私级别档位"""
        return self.PRIVACY_LEVELS.copy()
