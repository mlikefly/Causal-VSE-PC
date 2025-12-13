"""
VSE-PC统一接口

功能：
- 集成所有VSE-PC组件
- 提供简化的API
- 支持完整的工作流程

设计思路：
- 封装复杂性，提供简洁接口
- 支持加密、ML推理、可验证性
- 自动处理组件之间的交互
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .privacy_budget import AdaptivePrivacyBudget
from .ciphertext_ml import CiphertextMLAdapter
from .verifiable import VerifiableEncryption

# 导入现有组件
from src.cipher.scne_cipher import SCNECipher
from src.neural.unet import UNetSaliencyDetector
from src.crypto.key_system import HierarchicalKeySystem


class VSEPCInterface:
    """
    VSE-PC统一接口
    
    提供简化的API，隐藏内部复杂性。
    """
    
    def __init__(
        self,
        password: str,
        image_size: int = 256,
        device: Optional[str] = None,
        use_frequency: bool = True,
        use_fft: bool = True
    ):
        """
        初始化VSE-PC接口
        
        Args:
            password: 用户密码
            image_size: 图像尺寸
            device: 设备（'cuda'/'cpu'，None则自动检测）
            use_frequency: 是否使用频域加密
            use_fft: 是否使用FFT（比DWT快）
        """
        self.password = password
        self.image_size = image_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        # 1. 密钥系统
        self.key_system = HierarchicalKeySystem(password, iterations=100000)
        
        # 2. 加密器（使用现有SCNE加密器）
        self.cipher = SCNECipher(
            image_size=image_size,
            use_frequency=use_frequency,
            use_fft=use_fft,
            password=password
        ).to(self.device)
        
        # 3. 语义分析（U-Net）
        self.unet = UNetSaliencyDetector(in_channels=1, base_channels=64).to(self.device)
        self.unet.eval()
        
        # 4. VSE-PC新组件
        self.privacy_allocator = AdaptivePrivacyBudget()
        self.ml_adapter = CiphertextMLAdapter()
        self.verifier = VerifiableEncryption(master_key=self.key_system.master_key)
    
    def encrypt_for_ml(
        self,
        image: torch.Tensor,
        task_type: str,
        privacy_requirement: float = 1.0,
        generate_proof: bool = True
    ) -> Tuple[torch.Tensor, Dict, Optional[Dict]]:
        """
        为ML任务加密图像
        
        Args:
            image: [B, C, H, W] 或 [1, H, W] 输入图像
            task_type: 任务类型 {'classification', 'segmentation', 'detection'}
            privacy_requirement: 全局隐私需求 [0.0, 1.0]
            generate_proof: 是否生成可验证性证明
        
        Returns:
            encrypted: [B, C, H, W] 加密图像
            enc_info: 加密信息字典
            proof: 可验证性证明（如果generate_proof=True）
        """
        # 1. 确保维度正确
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
        
        image = image.to(self.device)
        B, C, H, W = image.shape
        
        # 2. 语义分析
        with torch.no_grad():
            semantic_mask = self.unet(image)  # [B, 1, H, W]
        
        # 3. 隐私预算分配
        privacy_map = self.privacy_allocator.allocate(
            semantic_mask,
            task_type,
            privacy_requirement
        )  # [B, 1, H, W]
        
        # 4. 加密（使用现有SCNE加密器，但需要扩展支持privacy_map）
        # 简化：使用全局隐私级别
        global_privacy = privacy_map.mean().item()
        
        # 生成默认mask和params
        mask = torch.ones(B, 1, H, W, device=self.device)
        chaos_params = torch.rand(B, 19, 3, device=self.device) * torch.tensor([10.0, 5.0, 1.0], device=self.device)
        
        encrypted, enc_info = self.cipher.encrypt(
            image,
            mask,
            chaos_params,
            password=self.password,
            privacy_level=global_privacy,
            semantic_preserving=(global_privacy < 0.5)
        )
        
        # 5. 生成可验证性证明
        proof = None
        if generate_proof:
            proof = self.verifier.generate_proof(image, encrypted, enc_info)
        
        return encrypted, enc_info, proof
    
    def ml_inference(
        self,
        encrypted_image: torch.Tensor,
        model: nn.Module,
        task_type: str
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        在加密图像上运行ML推理
        
        Args:
            encrypted_image: [B, C, H, W] 加密图像
            model: ML模型
            task_type: 任务类型
        
        Returns:
            predictions: 预测结果
        """
        return self.ml_adapter.forward(encrypted_image, model, task_type)
    
    def verify_encryption(
        self,
        encrypted: torch.Tensor,
        proof: Dict,
        enc_info: Dict
    ) -> bool:
        """
        验证加密正确性
        
        Args:
            encrypted: 加密图像
            proof: 可验证性证明
            enc_info: 加密信息
        
        Returns:
            valid: 是否验证通过
        """
        return self.verifier.verify(encrypted, proof, enc_info)
    
    def decrypt(
        self,
        encrypted: torch.Tensor,
        enc_info: Dict
    ) -> torch.Tensor:
        """
        解密图像
        
        Args:
            encrypted: 加密图像
            enc_info: 加密信息
        
        Returns:
            decrypted: 解密图像
        """
        # 使用现有SCNE解密功能
        mask = torch.ones_like(encrypted[:, :1])
        decrypted = self.cipher.decrypt(
            encrypted,
            enc_info,
            mask,
            password=self.password
        )
        return decrypted
    
    def full_pipeline(
        self,
        image: torch.Tensor,
        model: nn.Module,
        task_type: str,
        privacy_requirement: float = 1.0
    ) -> Dict:
        """
        完整流程：加密 + ML推理 + 可验证性
        
        Args:
            image: 输入图像
            model: ML模型
            task_type: 任务类型
            privacy_requirement: 隐私需求
        
        Returns:
            results: 结果字典
                - encrypted: 加密图像
                - predictions: ML预测结果
                - proof: 可验证性证明
                - enc_info: 加密信息
        """
        # 1. 加密
        encrypted, enc_info, proof = self.encrypt_for_ml(
            image,
            task_type,
            privacy_requirement,
            generate_proof=True
        )
        
        # 2. ML推理
        predictions = self.ml_inference(encrypted, model, task_type)
        
        # 3. 验证
        verification_result = self.verify_encryption(encrypted, proof, enc_info)
        
        return {
            'encrypted': encrypted,
            'predictions': predictions,
            'proof': proof,
            'enc_info': enc_info,
            'verification': verification_result
        }


def test_vsepc_interface():
    """测试VSE-PC接口"""
    print("测试VSE-PC统一接口")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建接口
    interface = VSEPCInterface(
        password='test_password',
        image_size=256,
        device=device
    )
    
    # 创建测试图像
    image = torch.rand(1, 1, 256, 256, device=device)
    
    # 创建模拟ML模型
    class SimpleModel(nn.Module):
        def forward(self, x):
            return torch.rand(x.size(0), 10, device=x.device)
    
    model = SimpleModel().to(device)
    
    # 测试完整流程
    results = interface.full_pipeline(
        image,
        model,
        task_type='classification',
        privacy_requirement=0.7
    )
    
    print(f"  加密图像形状: {results['encrypted'].shape}")
    print(f"  预测结果形状: {results['predictions'].shape}")
    print(f"  验证结果: {'通过' if results['verification'] else '失败'}")
    
    print("✓ 测试完成")


if __name__ == "__main__":
    test_vsepc_interface()




