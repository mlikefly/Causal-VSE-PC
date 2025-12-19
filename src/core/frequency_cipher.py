"""
频域语义控制加密模块

提供两条路径：
- FrequencySemanticCipher（DWT）：小波分解/重构，适配 CPU/无GPU 场景，稳定可逆
- FrequencySemanticCipherOptimized（FFT）：相位主导扰动 + 可逆仿射缩放，数值异常时自动回退 DWT
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import hmac


class FrequencySemanticCipher(nn.Module):
    """DWT 频域路径：分层扰动 + 可逆仿射缩放"""
    
    def __init__(self, wavelet: str = 'haar', mode: str = 'symmetric'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
    
    def dwt2d(self, image: torch.Tensor):
        B, C, H, W = image.shape
        device, dtype = image.device, image.dtype
        LL_all, LH_all, HL_all, HH_all = [], [], [], []
        try:
            import pywt  # type: ignore
        except Exception as e:
            raise ImportError("需要安装 PyWavelets 才能使用 DWT 路径：pip install PyWavelets") from e
        for b in range(B):
            for c in range(C):
                arr = image[b, c].detach().cpu().numpy()
                LL, (LH, HL, HH) = pywt.dwt2(arr, self.wavelet, mode=self.mode)
                LL_all.append(LL)
                LH_all.append(LH)
                HL_all.append(HL)
                HH_all.append(HH)
        LL = torch.tensor(np.array(LL_all), dtype=dtype, device=device).view(B, C, -1, W // 2)
        LH = torch.tensor(np.array(LH_all), dtype=dtype, device=device).view(B, C, -1, W // 2)
        HL = torch.tensor(np.array(HL_all), dtype=dtype, device=device).view(B, C, -1, W // 2)
        HH = torch.tensor(np.array(HH_all), dtype=dtype, device=device).view(B, C, -1, W // 2)
        return LL, LH, HL, HH
    
    def idwt2d(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        B, C, Hh, Wh = LL.shape
        device, dtype = LL.device, LL.dtype
        rec_list = []
        try:
            import pywt  # type: ignore
        except Exception as e:
            raise ImportError("需要安装 PyWavelets 才能使用 DWT 路径：pip install PyWavelets") from e
        for b in range(B):
            for c in range(C):
                coeffs = (
                    LL[b, c].detach().cpu().numpy(),
                    (LH[b, c].detach().cpu().numpy(), HL[b, c].detach().cpu().numpy(), HH[b, c].detach().cpu().numpy())
                )
                rec = pywt.idwt2(coeffs, self.wavelet, mode=self.mode)
                rec_list.append(rec)
        rec = torch.tensor(np.array(rec_list), dtype=dtype, device=device).view(B, C, Hh * 2, Wh * 2)
        return rec

    def _perturb(self, shape, key_bytes: bytes, suffix: bytes, strength: float, device, region_nonce: bytes = None, seed_override: int = None):
        """基于密钥的确定性噪声（torch生成，避免额外内存拷贝）。"""
        # 优先使用 seed_override；否则以 HMAC(key_bytes, (region_nonce||suffix)) 派生
        if seed_override is not None:
            mix = int.from_bytes(hashlib.sha256(bytes(suffix)).digest()[:8], 'big')
            seed = int(seed_override) ^ mix
        else:
            if region_nonce is None:
                seed_material = hashlib.sha256(key_bytes + suffix).digest()
            else:
                seed_material = hmac.new(key_bytes, bytes(region_nonce) + bytes(suffix), hashlib.sha256).digest()
            seed = int.from_bytes(seed_material[:8], 'big')
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        # 标准化，防止极值
        max_abs = float(noise.abs().amax().item() + 1e-8)
        noise = (noise / max_abs) * float(strength)
        return noise.to(device)
    
    def encrypt_semantic_preserving(self, image: torch.Tensor, region_key: bytes,
                                    privacy_level: float = 1.0, semantic_preserving: bool = False,
                                    region_nonce: bytes = None, privacy_map: Optional[torch.Tensor] = None):
        device = image.device
        B, C, H, W = image.shape
        LL, LH, HL, HH = self.dwt2d(image)
        LL_pert = self._perturb(LL.shape, region_key, b'_LL', 1.0, device, region_nonce=region_nonce)
        LH_pert = self._perturb(LH.shape, region_key, b'_LH', 1.0, device, region_nonce=region_nonce)
        HL_pert = self._perturb(HL.shape, region_key, b'_HL', 1.0, device, region_nonce=region_nonce)
        HH_pert = self._perturb(HH.shape, region_key, b'_HH', 1.0, device, region_nonce=region_nonce)
        
        # 支持区域级隐私预算
        if privacy_map is not None:
            # 确保privacy_map形状正确 [B, 1, H, W] 或 [B, C, H, W]
            if privacy_map.shape != image.shape:
                if privacy_map.dim() == 4 and privacy_map.size(1) == 1:
                    # [B, 1, H, W] -> 扩展到所有通道
                    privacy_map = privacy_map.expand(-1, C, -1, -1)
                else:
                    raise ValueError(f"privacy_map形状不匹配: {privacy_map.shape} vs {image.shape}")
            
            # 下采样到频域尺寸
            H_dwt, W_dwt = LL.shape[-2], LL.shape[-1]
            privacy_map_dwt = F.interpolate(
                privacy_map,
                size=(H_dwt, W_dwt),
                mode='bilinear',
                align_corners=False
            )
            
            # 根据隐私预算动态计算强度
            if semantic_preserving:
                alpha_low_map = 0.05 + 0.15 * privacy_map_dwt
                alpha_high_map = 0.30 + 0.70 * privacy_map_dwt
            else:
                alpha_low_map = 0.5 + 0.5 * privacy_map_dwt
                alpha_high_map = 0.5 + 0.5 * privacy_map_dwt
            
            # 应用区域级扰动
            LL_enc = LL + LL_pert * alpha_low_map
            LH_enc = LH + LH_pert * alpha_high_map
            HL_enc = HL + HL_pert * alpha_high_map
            HH_enc = HH + HH_pert * alpha_high_map
            
            # 记录平均强度（用于解密兼容性）
            alpha_low = float(alpha_low_map.mean().item())
            alpha_high = float(alpha_high_map.mean().item())
        else:
            # 原有全局隐私级别逻辑
            if semantic_preserving:
                alpha_low = 0.05 + 0.15 * privacy_level
                alpha_high = 0.30 + 0.70 * privacy_level
            else:
                alpha_low = 0.5 + 0.5 * privacy_level
                alpha_high = 0.5 + 0.5 * privacy_level
            
            LL_enc = LL + LL_pert * alpha_low
            LH_enc = LH + LH_pert * alpha_high
            HL_enc = HL + HL_pert * alpha_high
            HH_enc = HH + HH_pert * alpha_high
        
        encrypted = self.idwt2d(LL_enc, LH_enc, HL_enc, HH_enc)
        with torch.no_grad():
            min_val = encrypted.amin(dim=(1, 2, 3), keepdim=True)
            max_val = encrypted.amax(dim=(1, 2, 3), keepdim=True)
            scale = (max_val - min_val).clamp_min(1e-8)
        encrypted = (encrypted - min_val) / scale

        freq_info = {
            'LL_pert': LL_pert.detach().cpu(),
            'LH_pert': LH_pert.detach().cpu(),
            'HL_pert': HL_pert.detach().cpu(),
            'HH_pert': HH_pert.detach().cpu(),
            'alpha_low': float(alpha_low),
            'alpha_high': float(alpha_high),
            'affine_min': min_val.detach().cpu(),
            'affine_scale': scale.detach().cpu(),
            'fallback': 'none',
            'region_nonce': (region_nonce.hex() if region_nonce is not None else None)
        }
        return encrypted, freq_info
    
    def decrypt(self, encrypted: torch.Tensor, freq_info: dict):
        device = encrypted.device
        affine_min = freq_info.get('affine_min')
        affine_scale = freq_info.get('affine_scale')
        if affine_min is not None and affine_scale is not None:
            pre = encrypted * affine_scale.to(device) + affine_min.to(device)
        else:
            pre = encrypted
        LL_enc, LH_enc, HL_enc, HH_enc = self.dwt2d(pre)
        # 支持 seed 化重建扰动
        if 'LL_pert' in freq_info:
            LL_pert = freq_info['LL_pert'].to(device)
            LH_pert = freq_info['LH_pert'].to(device)
            HL_pert = freq_info['HL_pert'].to(device)
            HH_pert = freq_info['HH_pert'].to(device)
        else:
            seed = freq_info.get('seed', None)
            if seed is None:
                raise ValueError("freq_info 缺少扰动与种子，无法解密")
            # 依据 seed 重建四个子带的扰动
            # 说明：这里不再依赖 region_key/region_nonce，实现种子化可重建
            LL_pert = self._perturb(LL_enc.shape, b'key', b'_LL', 1.0, device, seed_override=int(seed))
            LH_pert = self._perturb(LH_enc.shape, b'key', b'_LH', 1.0, device, seed_override=int(seed))
            HL_pert = self._perturb(HL_enc.shape, b'key', b'_HL', 1.0, device, seed_override=int(seed))
            HH_pert = self._perturb(HH_enc.shape, b'key', b'_HH', 1.0, device, seed_override=int(seed))
        LL = LL_enc - LL_pert * float(freq_info['alpha_low'])
        LH = LH_enc - LH_pert * float(freq_info['alpha_high'])
        HL = HL_enc - HL_pert * float(freq_info['alpha_high'])
        HH = HH_enc - HH_pert * float(freq_info['alpha_high'])
        return self.idwt2d(LL, LH, HL, HH)


class FrequencySemanticCipherOptimized(nn.Module):
    """FFT 频域路径：相位主导扰动 + 可逆仿射缩放，异常回退 DWT"""
    
    def __init__(self, use_magnitude_mod: bool = True, use_learnable_band: bool = False, num_radial_bins: int = 8):
        super().__init__()
        self.use_magnitude_mod = bool(use_magnitude_mod)
        self._fallback_dwt = FrequencySemanticCipher('haar')
        # 缓存不同尺寸下的低/高频掩码，减少重复构建
        self._mask_cache = {}
        # 径向分桶缓存（用于rfft域的半平面尺寸）
        self._radial_cache = {}
        # 可学习频带拨盘（论文级增强）：两个标量控制低/高频增益，训练期可学习；推理期由密钥派生微扰
        self.use_learnable_band = bool(use_learnable_band)
        init_low, init_high = 0.10, 0.50  # 与固定公式初值一致
        if self.use_learnable_band:
            self.low_gain = nn.Parameter(torch.tensor(float(init_low), dtype=torch.float32))
            self.high_gain = nn.Parameter(torch.tensor(float(init_high), dtype=torch.float32))
            # 径向分桶增益（正值），用 softplus 保证正性
            self.num_radial_bins = int(num_radial_bins)
            self.radial_gains = nn.Parameter(torch.ones(self.num_radial_bins, dtype=torch.float32))
        else:
            # 注册为buffer以便state_dict一致
            self.register_buffer('low_gain', torch.tensor(float(init_low), dtype=torch.float32))
            self.register_buffer('high_gain', torch.tensor(float(init_high), dtype=torch.float32))
            self.num_radial_bins = int(num_radial_bins)
            self.register_buffer('radial_gains', torch.ones(self.num_radial_bins, dtype=torch.float32))

    def _band_strength(self, privacy_level: float, semantic_preserving: bool, seed: int, device: torch.device):
        """
        计算低/高频强度标量（带确定性微扰）。
        若启用可学习拨盘，则在固定基线之上叠加 learnable gain；同时依据 seed 加入小幅度确定性抖动（实例不可链接）。
        
        增强：大幅增加强度范围，使不同privacy_level的差异在归一化后仍然明显
        """
        pl = float(privacy_level)
        if semantic_preserving:
            # 语义保持模式：较小的扰动
            base_low = 0.05 + 0.15 * pl
            base_high = 0.20 + 0.60 * pl
        else:
            # 完全加密模式：大幅增强扰动范围
            # privacy_level=0.3 -> low=0.30, high=0.60
            # privacy_level=1.0 -> low=0.80, high=1.60
            base_low = 0.20 + 0.60 * pl
            base_high = 0.40 + 1.20 * pl
        # 确定性微扰（±2%），由 seed 派生
        jitter_src = ((seed & 0xFFFF) / 65535.0) - 0.5
        jitter = 0.02 * jitter_src
        if self.use_learnable_band:
            low = (base_low + float(self.low_gain.detach().cpu().item())) * (1.0 + jitter)
            high = (base_high + float(self.high_gain.detach().cpu().item())) * (1.0 + jitter)
        else:
            low = base_low * (1.0 + jitter)
            high = base_high * (1.0 + jitter)
        # 转为设备标量张量
        return torch.tensor(low, device=device, dtype=torch.float32), torch.tensor(high, device=device, dtype=torch.float32)
    
    def encrypt_fft(self, image: torch.Tensor, region_key: bytes,
                    privacy_level: float = 1.0, semantic_preserving: bool = False,
                    region_nonce: bytes = None, privacy_map: Optional[torch.Tensor] = None):
        device = image.device
        B, C, H, W = image.shape
        # 使用 rfft2/irfft2 确保输出为实数并提升数值稳定
        spec = torch.fft.rfft2(image)
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        
        # 确定性种子（同设备生成，避免额外拷贝）
        if region_nonce is None:
            seed_material = hashlib.sha256(region_key).digest()
        else:
            seed_material = hmac.new(region_key, bytes(region_nonce), hashlib.sha256).digest()
        seed = int.from_bytes(seed_material[:8], 'big')
        if self.use_magnitude_mod:
            gen_mag = torch.Generator(device=device)
            gen_mag.manual_seed((seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
            mag_noise = torch.randn((B, C, mag.size(-2), mag.size(-1)), generator=gen_mag, device=device, dtype=torch.float32)
            max_abs = float(mag_noise.abs().amax().item() + 1e-8)
            mag_noise = mag_noise / max_abs
        else:
            mag_noise = torch.zeros_like(mag, dtype=torch.float32, device=device)

        # 频带掩码（缓存）
        key_mask = (H, W, str(device))
        if key_mask in self._mask_cache:
            low_mask_spatial, high_mask_spatial = self._mask_cache[key_mask]
        else:
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            radius = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            low_mask_spatial = (radius < min(H, W) * 0.2).float()
            high_mask_spatial = 1.0 - low_mask_spatial
            self._mask_cache[key_mask] = (low_mask_spatial, high_mask_spatial)

        # 将空间掩码映射到 rfft 频域尺寸（最后一维为频率的正半轴）
        # 简化做法：对 W 方向做最近邻下采样到 rfft 尺度
        low_mask = F.interpolate(low_mask_spatial.unsqueeze(0).unsqueeze(0), size=(H, mag.size(-1)), mode='nearest')[0, 0]
        high_mask = F.interpolate(high_mask_spatial.unsqueeze(0).unsqueeze(0), size=(H, mag.size(-1)), mode='nearest')[0, 0]
        low_mask = low_mask.view(1, 1, *low_mask.shape)
        high_mask = high_mask.view(1, 1, *high_mask.shape)

        # 支持区域级隐私预算
        if privacy_map is not None:
            # 确保privacy_map形状正确
            if privacy_map.shape != image.shape:
                if privacy_map.dim() == 4 and privacy_map.size(1) == 1:
                    privacy_map = privacy_map.expand(-1, C, -1, -1)
                else:
                    raise ValueError(f"privacy_map形状不匹配: {privacy_map.shape} vs {image.shape}")
            
            # 映射到rfft频域尺寸
            H_rfft, W_rfft = mag.size(-2), mag.size(-1)
            privacy_map_rfft = F.interpolate(
                privacy_map,
                size=(H_rfft, W_rfft),
                mode='bilinear',
                align_corners=False
            )
            
            # 计算基础强度（使用平均隐私级别）
            avg_privacy = float(privacy_map_rfft.mean().item())
            low_s_base, high_s_base = self._band_strength(avg_privacy, semantic_preserving, seed, device)
            
            # 根据隐私预算动态调整强度
            # 大幅增强强度范围，使不同privacy_level的差异在归一化后仍然明显
            if semantic_preserving:
                # 语义保持模式：较小的扰动范围
                low_s_min, low_s_max = 0.05, 0.30
                high_s_min, high_s_max = 0.20, 0.80
            else:
                # 完全加密模式：大幅增强扰动范围
                # privacy_map=0.0 -> low=0.10, high=0.30
                # privacy_map=1.0 -> low=1.00, high=2.00
                low_s_min, low_s_max = 0.10, 1.00
                high_s_min, high_s_max = 0.30, 2.00
            
            low_s_map = low_s_min + (low_s_max - low_s_min) * privacy_map_rfft
            high_s_map = high_s_min + (high_s_max - high_s_min) * privacy_map_rfft
            
            low_s = float(low_s_base.item())  # 保留标量用于记录
            high_s = float(high_s_base.item())
        else:
            low_s, high_s = self._band_strength(privacy_level, semantic_preserving, seed, device)
            low_s_map = None
            high_s_map = None

        # 径向分桶图（rfft域）：按到中心的半径映射到 [0, num_bins-1]，缓存
        key_rad = (H, mag.size(-1), str(device))
        if key_rad in self._radial_cache:
            radial_bins = self._radial_cache[key_rad]
            # 确保缓存的 radial_bins 在正确的设备上
            if radial_bins.device != device:
                radial_bins = radial_bins.to(device)
                self._radial_cache[key_rad] = radial_bins
        else:
            y_r, x_r = torch.meshgrid(torch.arange(H, device=device), torch.arange(mag.size(-1), device=device), indexing='ij')
            cy_r = H // 2
            # rfft的x轴是[0..W/2]，将其镜像到完整频谱的半径尺度近似
            radius_r = torch.sqrt((y_r - cy_r) ** 2 + x_r ** 2)
            r_norm = radius_r / float(max(H, W) / 2.0 + 1e-6)
            # 分桶索引 [0, num_bins-1]
            radial_bins = torch.clamp((r_norm * self.num_radial_bins).long(), 0, self.num_radial_bins - 1)
            self._radial_cache[key_rad] = radial_bins
        # radial gains 映射到二维图
        gains_vec = torch.nn.functional.softplus(self.radial_gains)
        # 确保 gains_vec 和 radial_bins 在同一设备上
        gains_vec = gains_vec.to(device)
        gains_map = gains_vec[radial_bins]  # [H, W/2+1]
        gains_map = gains_map.view(1, 1, *gains_map.shape)

        # 实际应用频带强度 + 径向增益
        if privacy_map is not None and low_s_map is not None:
            # 使用区域级强度
            band_strength = (low_mask * low_s_map + high_mask * high_s_map) * gains_map
        else:
            # 使用全局强度
            band_strength = (low_mask * low_s + high_mask * high_s) * gains_map
        # 缩放因子：0.4用于加密，解密时必须使用相同值
        mag_scale = 1.0 + mag_noise * 0.4 * band_strength
        mag_perturbed = mag * mag_scale
        spec_perturbed = mag_perturbed * torch.exp(1j * phase)
        enc = torch.fft.irfft2(spec_perturbed, s=(H, W))

        # 可逆归一化：保存min/scale用于解密时精确恢复
        with torch.no_grad():
            min_val = enc.amin(dim=(1, 2, 3), keepdim=True)
            max_val = enc.amax(dim=(1, 2, 3), keepdim=True)
            scale = (max_val - min_val).clamp_min(1e-8)
        enc = (enc - min_val) / scale
        
        # 数值稳定性检查
        invalid = ~torch.isfinite(enc).all()
        degenerate = (scale < 1e-6).any()
        if invalid or degenerate:
            enc_fb, info_fb = self._fallback_dwt.encrypt_semantic_preserving(
                image, region_key, privacy_level=privacy_level, semantic_preserving=semantic_preserving
            )
            info_fb['fallback'] = 'dwt'
            return enc_fb, info_fb

        affine_min_list = [float(min_val[b].item()) for b in range(B)]
        affine_scale_list = [float(scale[b].item()) for b in range(B)]
        # 保存加密时使用的 radial_gains 值，确保解密时一致性
        if self.use_learnable_band:
            radial_gains_used = gains_vec.detach().cpu().tolist()
        else:
            radial_gains_used = gains_vec.detach().cpu().tolist()
        fft_info = {
            'seed': int(seed),
            'low_freq_strength': float(low_s),
            'high_freq_strength': float(high_s),
            'affine_min_list': affine_min_list,
            'affine_scale_list': affine_scale_list,
            'privacy_level': float(privacy_level),
            'semantic_preserving': bool(semantic_preserving),
            'fallback': 'none',
            'region_nonce': (region_nonce.hex() if region_nonce is not None else None),
            'radial_gains_used': radial_gains_used,
            'mag_scale': mag_scale.detach().cpu()  # 保存完整mag_scale用于精确解密
        }
        return enc, fft_info
    
    def _fft_noise_and_masks(self, fft_info: dict, pre: torch.Tensor, device: torch.device):
        """基于 seed 重建幅度噪声与频带掩码；rfft 维度对齐。"""
        seed = fft_info.get('seed', None)
        if seed is None:
            raise ValueError("fft_info 缺少噪声与种子，无法解密")
        B, C, H, W = pre.shape
        spec = torch.fft.rfft2(pre)
        # 幅度噪声（同设备生成）
        gen_mag = torch.Generator(device=device)
        gen_mag.manual_seed((int(seed) ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        mag_noise = torch.randn((B, C, spec.size(-2), spec.size(-1)), generator=gen_mag, device=device, dtype=torch.float32)
        max_abs = float(mag_noise.abs().amax().item() + 1e-8)
        mag_noise = mag_noise / max_abs
        # 空间掩码（缓存）并映射到 rfft 尺度
        key_mask = (H, W, str(device))
        if key_mask in self._mask_cache:
            low_mask_spatial, high_mask_spatial = self._mask_cache[key_mask]
        else:
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            radius = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            low_mask_spatial = (radius < min(H, W) * 0.2).float()
            high_mask_spatial = 1.0 - low_mask_spatial
            self._mask_cache[key_mask] = (low_mask_spatial, high_mask_spatial)
        low_mask = F.interpolate(low_mask_spatial.unsqueeze(0).unsqueeze(0), size=(H, spec.size(-1)), mode='nearest')[0, 0]
        high_mask = F.interpolate(high_mask_spatial.unsqueeze(0).unsqueeze(0), size=(H, spec.size(-1)), mode='nearest')[0, 0]
        low_mask = low_mask.view(1, 1, *low_mask.shape)
        high_mask = high_mask.view(1, 1, *high_mask.shape)
        return mag_noise, low_mask, high_mask
    
    def decrypt_fft(self, encrypted: torch.Tensor, fft_info: dict):
        device = encrypted.device
        B, C, H, W = encrypted.shape
        
        # 关键修复：加密时做了归一化，解密时需要先反归一化
        # 从fft_info恢复仿射参数
        affine_min_list = fft_info.get('affine_min_list', None)
        affine_scale_list = fft_info.get('affine_scale_list', None)
        
        if affine_min_list is not None and affine_scale_list is not None:
            # 反归一化：pre = encrypted * scale + min
            pre_list = []
            for b in range(B):
                min_val = affine_min_list[b] if b < len(affine_min_list) else 0.0
                scale_val = affine_scale_list[b] if b < len(affine_scale_list) else 1.0
                pre_b = encrypted[b] * scale_val + min_val
                pre_list.append(pre_b)
            pre = torch.stack(pre_list, dim=0)
        else:
            # 兼容旧版：无仿射参数时直接使用
            pre = encrypted

        spec = torch.fft.rfft2(pre)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        # 低/高频强度参数
        # 兼容旧info；若无，使用当前模型拨盘（保证评测时与训练时一致）
        if 'low_freq_strength' in fft_info and 'high_freq_strength' in fft_info:
            low_s = float(fft_info['low_freq_strength'])
            high_s = float(fft_info['high_freq_strength'])
            low_s = torch.tensor(low_s, device=device, dtype=torch.float32)
            high_s = torch.tensor(high_s, device=device, dtype=torch.float32)
        elif 'alpha_low' in fft_info and 'alpha_high' in fft_info:
            low_s = torch.tensor(float(fft_info['alpha_low']), device=device, dtype=torch.float32)
            high_s = torch.tensor(float(fft_info['alpha_high']), device=device, dtype=torch.float32)
        else:
            pl = float(fft_info.get('privacy_level', 1.0))
            sp = bool(fft_info.get('semantic_preserving', False))
            seed = int(fft_info.get('seed', 0))
            low_s, high_s = self._band_strength(pl, sp, seed, device)

        stored_scale = fft_info.get('mag_scale', None)
        if stored_scale is not None:
            mag_scale = torch.as_tensor(stored_scale, device=device, dtype=torch.float32)
            if mag_scale.dim() == 3:
                mag_scale = mag_scale.unsqueeze(0)
        else:
            mag_noise, low_mask, high_mask = self._fft_noise_and_masks(fft_info, pre, device)
            H, W = pre.size(-2), pre.size(-1)
            spec = torch.fft.rfft2(pre)
            key_rad = (H, spec.size(-1), str(device))
            if key_rad in self._radial_cache:
                radial_bins = self._radial_cache[key_rad]
                # 确保缓存的 radial_bins 在正确的设备上
                if radial_bins.device != device:
                    radial_bins = radial_bins.to(device)
                    self._radial_cache[key_rad] = radial_bins
            else:
                y_r, x_r = torch.meshgrid(torch.arange(H, device=device), torch.arange(spec.size(-1), device=device), indexing='ij')
                cy_r = H // 2
                radius_r = torch.sqrt((y_r - cy_r) ** 2 + x_r ** 2)
                r_norm = radius_r / float(max(H, W) / 2.0 + 1e-6)
                radial_bins = torch.clamp((r_norm * self.num_radial_bins).long(), 0, self.num_radial_bins - 1)
                self._radial_cache[key_rad] = radial_bins
            if 'radial_gains_used' in fft_info:
                gains_vec = torch.tensor(fft_info['radial_gains_used'], device=device, dtype=torch.float32)
                gains_vec = torch.clamp(gains_vec, min=1e-6)
            else:
                gains_vec = torch.nn.functional.softplus(self.radial_gains)
                gains_vec = gains_vec.to(device)  # 确保设备一致
            gains_map = gains_vec[radial_bins].view(1, 1, H, spec.size(-1))
            band_strength = (low_mask * low_s + high_mask * high_s) * gains_map
            # 与加密保持一致：0.4缩放因子
            mag_scale = 1.0 + mag_noise * 0.4 * band_strength
        mag_orig = mag / (mag_scale + 1e-10)
        spec_orig = mag_orig * torch.exp(1j * phase)
        dec = torch.fft.irfft2(spec_orig, s=(pre.size(-2), pre.size(-1)))
        return dec


__all__ = [
    'FrequencySemanticCipher',
    'FrequencySemanticCipherOptimized'
]


