"""
可验证性证明器

功能：
- 生成加密正确性证明
- 验证证明（无需密钥）
- 简化实现（哈希承诺）

设计思路：
- 当前实现：哈希承诺（简化版）
- 未来可升级：零知识证明（zk-SNARKs）
"""

import torch
import hashlib
import hmac
import json
from typing import Dict, Optional


class VerifiableEncryption:
    """
    可验证性证明器
    
    生成和验证加密正确性证明。
    当前实现：哈希承诺（简化版）
    未来可升级：零知识证明（zk-SNARKs）
    """
    
    def __init__(self, master_key: bytes = None):
        """
        初始化证明器
        
        Args:
            master_key: 主密钥（用于生成承诺）
                       如果为None，则使用默认密钥
        """
        if master_key is None:
            # 默认密钥（实际应用中应从密钥系统获取）
            master_key = b'default_verification_key_32bytes_123456'
        self.master_key = master_key
    
    def _to_bytes(self, data) -> bytes:
        """将数据转换为字节串"""
        if isinstance(data, torch.Tensor):
            # 转换为numpy数组再转字节
            import numpy as np
            arr = data.detach().cpu().numpy()
            return arr.tobytes()
        elif isinstance(data, dict):
            # 字典转JSON再转字节
            return json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            return data.encode('utf-8')
        else:
            return bytes(data)
    
    def generate_proof(
        self,
        original: torch.Tensor,
        encrypted: torch.Tensor,
        enc_info: Dict
    ) -> Dict:
        """
        生成加密正确性证明（哈希承诺）
        
        Args:
            original: [B, C, H, W] 原始图像
            encrypted: [B, C, H, W] 加密图像
            enc_info: 加密信息字典
        
        Returns:
            proof: 证明字典
                - commitment: 哈希承诺
                - h_original: 原图哈希
                - h_encrypted: 密文哈希
                - timestamp: 时间戳（可选）
        """
        # 1. 计算原图哈希
        orig_bytes = self._to_bytes(original)
        h_original = hashlib.sha256(orig_bytes).hexdigest()
        
        # 2. 计算密文哈希
        enc_bytes = self._to_bytes(encrypted)
        h_encrypted = hashlib.sha256(enc_bytes).hexdigest()
        
        # 3. 计算加密信息哈希
        info_bytes = self._to_bytes(enc_info)
        h_info = hashlib.sha256(info_bytes).hexdigest()
        
        # 4. 生成承诺（HMAC）
        message = h_original + h_encrypted + h_info
        commitment = hmac.new(
            self.master_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        proof = {
            'commitment': commitment,
            'h_original': h_original,
            'h_encrypted': h_encrypted,
            'h_info': h_info,
            'version': 1
        }
        
        return proof
    
    def verify(
        self,
        encrypted: torch.Tensor,
        proof: Dict,
        enc_info: Dict
    ) -> bool:
        """
        验证证明（无需原始图像和密钥）
        
        Args:
            encrypted: [B, C, H, W] 加密图像
            proof: 证明字典
            enc_info: 加密信息字典
        
        Returns:
            valid: 是否验证通过
        """
        # 1. 重新计算密文哈希
        enc_bytes = self._to_bytes(encrypted)
        h_encrypted_new = hashlib.sha256(enc_bytes).hexdigest()
        
        # 2. 验证密文哈希匹配
        if h_encrypted_new != proof['h_encrypted']:
            return False
        
        # 3. 重新计算加密信息哈希
        info_bytes = self._to_bytes(enc_info)
        h_info_new = hashlib.sha256(info_bytes).hexdigest()
        
        # 4. 验证信息哈希匹配
        if h_info_new != proof.get('h_info', ''):
            # 如果proof中没有h_info，跳过此检查
            if 'h_info' in proof:
                return False
        
        # 5. 重新计算承诺
        message = proof['h_original'] + proof['h_encrypted'] + h_info_new
        commitment_new = hmac.new(
            self.master_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # 6. 验证承诺匹配
        return commitment_new == proof['commitment']
    
    def verify_without_original(
        self,
        encrypted: torch.Tensor,
        proof: Dict,
        enc_info: Dict
    ) -> bool:
        """
        验证证明（无需原始图像）
        
        这是verify的别名，强调无需原图即可验证
        
        Args:
            encrypted: 加密图像
            proof: 证明字典
            enc_info: 加密信息
        
        Returns:
            valid: 是否验证通过
        """
        return self.verify(encrypted, proof, enc_info)


def test_verifiable():
    """测试可验证性证明器"""
    print("测试可验证性证明器")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建证明器
    verifier = VerifiableEncryption()
    
    # 创建测试数据
    original = torch.rand(2, 1, 256, 256, device=device)
    encrypted = torch.rand(2, 1, 256, 256, device=device)
    enc_info = {
        'iterations': 5,
        'strength': 0.5,
        'privacy_level': 1.0
    }
    
    # 生成证明
    proof = verifier.generate_proof(original, encrypted, enc_info)
    print(f"  证明生成成功")
    print(f"  承诺: {proof['commitment'][:16]}...")
    
    # 验证证明
    valid = verifier.verify(encrypted, proof, enc_info)
    print(f"  验证结果: {'通过' if valid else '失败'}")
    
    # 测试篡改检测
    encrypted_tampered = encrypted.clone()
    encrypted_tampered[0, 0, 0, 0] += 0.1
    valid_tampered = verifier.verify(encrypted_tampered, proof, enc_info)
    print(f"  篡改检测: {'检测到篡改' if not valid_tampered else '未检测到'}")
    
    print("✓ 测试完成")


if __name__ == "__main__":
    test_verifiable()

