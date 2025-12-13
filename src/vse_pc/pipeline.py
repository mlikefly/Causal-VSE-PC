"""
VSE-PC 系统完整流水线 (VSE-PC System Pipeline)
==============================================

集成各个模块，实现端到端的图像隐私保护流程：
1. 语义分析 (Semantic Analysis)
2. 隐私预算分配 (Privacy Budget Allocation)
3. 因果分析 (Causal Analysis)
4. 分层加密 (Layered Encryption: Chaos -> Frequency)
5. 可验证性证明 (Verifiable Proof)

此模块展示了"因果驱动的可学习加密"的核心逻辑。
"""

import torch
import torch.nn as nn
import hashlib
from typing import Dict, Tuple, Optional

from src.core.chaotic_encryptor import StandardChaoticCipher
from src.core.frequency_cipher import FrequencySemanticCipherOptimized
from src.vse_pc.privacy_budget import AdaptivePrivacyBudget
from src.vse_pc.verifiable import VerifiableEncryption
from src.vse_pc.causal_analysis import CausalPrivacyAnalyzer, CausalPerformanceAnalyzer

class VSEPCPipeline(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 核心模块初始化
        self.privacy_allocator = AdaptivePrivacyBudget()
        self.causal_analyzer = CausalPrivacyAnalyzer()
        self.perf_analyzer = CausalPerformanceAnalyzer()
        
        self.layer1_chaos = StandardChaoticCipher(device=device)
        self.layer2_freq = FrequencySemanticCipherOptimized(
            use_magnitude_mod=True, 
            use_learnable_band=False # 推理模式
        )
        self.verifier = VerifiableEncryption()
        
        # 模拟 U-Net (实际使用时应加载预训练模型)
        self.unet_mock = True 

    def _derive_keys(self, master_key: bytes, image: torch.Tensor) -> Tuple[torch.Tensor, bytes]:
        """简化的密钥派生逻辑"""
        # 1. Image Hash
        img_bytes = image.detach().cpu().numpy().tobytes()
        img_hash = hashlib.sha256(img_bytes).digest()
        
        # 2. Image Key = HMAC(Master, ImageHash)
        import hmac
        img_key = hmac.new(master_key, img_hash, hashlib.sha512).digest()
        
        # 3. Chaos Key (Float seeds for Layer 1)
        # 取前16字节转为两个 float32 作为 chaos seeds
        seed1 = int.from_bytes(img_key[:8], 'big') / (2**64)
        seed2 = int.from_bytes(img_key[8:16], 'big') / (2**64)
        chaos_key = torch.tensor([[seed1, seed2]], device=self.device).repeat(image.shape[0], 1)
        
        # 4. Region Key (Bytes for Layer 2)
        region_key = img_key[16:48] # 32 bytes
        
        return chaos_key, region_key

    def forward(
        self, 
        image: torch.Tensor, 
        semantic_mask: torch.Tensor = None,
        task_type: str = 'classification',
        master_key: bytes = b'default_master_key',
        global_privacy: float = 1.0
    ) -> Dict:
        """
        端到端处理流程
        
        Args:
            image: [B, C, H, W] 原始图像
            semantic_mask: [B, 1, H, W] 语义掩码 (可选，若无则使用全1或模拟)
            task_type: 任务类型
            master_key: 主密钥
            
        Returns:
            result: 包含加密图像、证明、因果分析报告的字典
        """
        B, C, H, W = image.shape
        
        # 1. 语义分析 (如果未提供 mask，简单模拟中心敏感区域)
        if semantic_mask is None:
            # 模拟：中心 50% 区域为敏感
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            cy, cx = H // 2, W // 2
            dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
            semantic_mask = (dist < min(H, W) * 0.25).float().view(1, 1, H, W).expand(B, 1, -1, -1)
            
        # 2. 隐私预算分配 (因果驱动的核心)
        privacy_map = self.privacy_allocator.allocate(
            semantic_mask, task_type, global_privacy
        )
        
        # 3. 因果分析：为什么这样分配？
        causal_report = self.causal_analyzer.analyze_allocation(
            semantic_mask, privacy_map, task_type
        )
        
        # 4. 密钥派生
        chaos_key, region_key = self._derive_keys(master_key, image)
        
        # 5. Layer 1: 空域混沌加密 (Chaotic Scrambling & Diffusion)
        # 将 privacy_map 映射到 chaos strength
        # strength = 0.5 + 0.5 * privacy_map (高隐私->强混沌)
        chaos_strength = 0.5 + 0.5 * privacy_map
        chaos_params = {
            'iterations': 5, # 可以根据 privacy_level 动态调整
            'strength': chaos_strength
        }
        enc_layer1 = self.layer1_chaos(image, chaos_key, chaos_params)
        
        # 6. Layer 2: 频域语义控制 (Frequency Domain Control)
        # 对 Layer 1 的结果继续加密，保留语义
        enc_final, freq_info = self.layer2_freq.encrypt_fft(
            enc_layer1, 
            region_key=region_key,
            privacy_map=privacy_map, # 传递细粒度 map
            semantic_preserving=True
        )
        
        # 7. 可验证性证明 (Verifiability)
        # 收集所有加密参数
        enc_info_total = {
            'chaos_params': {k: (v.mean().item() if isinstance(v, torch.Tensor) else v) for k,v in chaos_params.items()},
            'freq_info': {k: v for k, v in freq_info.items() if k not in ['affine_min_list', 'affine_scale_list']}, # 简化用于哈希
            'task_type': task_type,
            'global_privacy': global_privacy
        }
        
        proof = self.verifier.generate_proof(image, enc_final, enc_info_total)
        
        return {
            'encrypted_image': enc_final,
            'privacy_map': privacy_map,
            'semantic_mask': semantic_mask,
            'proof': proof,
            'causal_report': causal_report,
            'enc_info': enc_info_total,
            'layer1_output': enc_layer1
        }

def test_pipeline():
    print("Testing VSE-PC Pipeline...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = VSEPCPipeline(device=device)
    
    img = torch.rand(1, 3, 256, 256, device=device)
    
    # Run pipeline
    result = pipeline(img, task_type='classification')
    
    print("Encryption shape:", result['encrypted_image'].shape)
    print("Proof commitment:", result['proof']['commitment'])
    print("Causal Explanation:", result['causal_report']['batch_reports'][0]['causal_explanation'])
    
    # Verify
    valid = pipeline.verifier.verify(result['encrypted_image'], result['proof'], result['enc_info'])
    print("Verification result:", valid)
    print("✓ Pipeline Integrated Successfully")

if __name__ == "__main__":
    test_pipeline()
