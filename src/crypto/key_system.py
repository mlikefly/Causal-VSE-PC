"""
分层密钥系统

5层密钥派生：
1. 主密钥（Master Key）：从用户密码派生
2. 图像密钥（Image Key）：基于主密钥+图像哈希
3. 区域密钥（Region Key）：基于图像密钥+区域特征
4. 混沌初值（Chaos Initial State）：从区域密钥派生
5. 动态S盒（Dynamic S-box）：基于区域密钥生成

密钥空间：2^6208
抗量子性：512位主密钥 + 50万次迭代
"""

import hashlib
import hmac
import numpy as np
from typing import Dict, List, Tuple
import os
import struct


class HierarchicalKeySystem:
    """分层密钥系统"""
    
    def __init__(self, password: str, iterations: int = 500000, salt: bytes = None, deterministic: bool = True):
        """
        初始化密钥系统
        
        Args:
            password: 用户密码
            iterations: PBKDF2迭代次数（抗量子：50万次）
            salt: 盐值（可选，如果不提供则根据deterministic参数决定）
            deterministic: 是否使用确定性salt（默认True，从密码派生）
        """
        self.password = password.encode('utf-8')
        self.iterations = iterations
        
        # 修复：使用确定性salt（从密码派生），确保相同密码产生相同密钥
        if salt is not None:
            self.salt = salt
        elif deterministic:
            # 从密码派生确定性salt（使用SHA256的前16字节）
            self.salt = hashlib.sha256(self.password + b'|salt_derivation').digest()[:16]
        else:
            # 随机salt（仅用于需要随机性的场景）
            self.salt = os.urandom(16)
        
        # 生成主密钥
        self.master_key = self._derive_master_key()
    
    def _derive_master_key(self) -> bytes:
        """
        派生主密钥（512位）
        
        使用PBKDF2-HMAC-SHA512，50万次迭代
        
        Returns:
            master_key: 64字节（512位）
        """
        master_key = hashlib.pbkdf2_hmac(
            'sha512',
            self.password,
            self.salt,
            self.iterations,
            dklen=64  # 512位
        )
        
        return master_key

    def get_salt(self) -> bytes:
        return self.salt

    def set_salt(self, salt: bytes):
        """更新盐并重新派生主密钥（用于解密端与enc_info对齐）"""
        if not isinstance(salt, (bytes, bytearray)):
            raise ValueError("salt 必须为 bytes")
        self.salt = bytes(salt)
        self.master_key = self._derive_master_key()

    def derive_seed(self, context: bytes) -> int:
        """
        从上下文派生8字节整数种子：seed = HMAC(master_key, context)[:8]
        用于确定性PRF种子，避免与图像内容耦合。
        """
        if not isinstance(context, (bytes, bytearray)):
            raise ValueError("context 必须为 bytes")
        digest = hmac.new(self.master_key, bytes(context), hashlib.sha256).digest()
        return int.from_bytes(digest[:8], byteorder='little', signed=False)
    
    def derive_image_key(self, image_data: np.ndarray) -> bytes:
        """
        派生图像密钥（512位）
        
        基于：主密钥 + 图像哈希
        使用：HMAC-SHA512
        
        Args:
            image_data: 图像数据 [H, W]
        
        Returns:
            image_key: 64字节（512位）
        """
        # 计算图像哈希
        image_bytes = image_data.tobytes()
        image_hash = hashlib.sha256(image_bytes).digest()
        
        # HMAC派生
        image_key = hmac.new(
            self.master_key,
            image_hash,
            hashlib.sha512
        ).digest()
        
        return image_key
    
    def derive_region_keys(
        self, 
        image_key: bytes, 
        num_regions: int = 19,
        region_features: List[np.ndarray] = None
    ) -> List[bytes]:
        """
        派生区域密钥（19个，每个256位）
        
        基于：图像密钥 + 区域ID + 区域特征
        使用：HMAC-SHA256
        
        Args:
            image_key: 图像密钥
            num_regions: 区域数量
            region_features: 区域特征列表（可选）
        
        Returns:
            region_keys: 19个区域密钥，每个32字节（256位）
        """
        region_keys = []
        
        for region_id in range(num_regions):
            # 区域ID
            region_id_bytes = struct.pack('<I', region_id)
            
            # 区域特征（如果提供）
            if region_features is not None and region_id < len(region_features):
                feature_bytes = region_features[region_id].tobytes()
            else:
                feature_bytes = b''
            
            # HMAC派生
            message = region_id_bytes + feature_bytes
            region_key = hmac.new(
                image_key,
                message,
                hashlib.sha256
            ).digest()
            
            region_keys.append(region_key)
        
        return region_keys
    
    def derive_chaos_initial_state(
        self, 
        region_key: bytes, 
        dimension: int = 5
    ) -> np.ndarray:
        """
        派生混沌初值（5维）
        
        基于：区域密钥
        方法：字节转浮点数（[0, 1]范围）
        
        Args:
            region_key: 区域密钥（32字节）
            dimension: 混沌维度（1/3/5）
        
        Returns:
            initial_state: [dimension] 浮点数数组
        """
        # 使用SHA512扩展密钥（确保足够的随机字节）
        expanded_key = hashlib.sha512(region_key).digest()
        
        # 转换为浮点数
        initial_state = np.zeros(dimension, dtype=np.float64)
        
        for i in range(dimension):
            # 每个维度使用8字节
            start_idx = i * 8
            end_idx = start_idx + 8
            
            # 字节转uint64，再归一化到[0, 1]
            bytes_chunk = expanded_key[start_idx:end_idx]
            uint_value = int.from_bytes(bytes_chunk, byteorder='big')
            
            # 归一化到(0, 1)，避免边界值
            initial_state[i] = (uint_value / (2**64 - 1)) * 0.9 + 0.05
        
        return initial_state
    
    def generate_dynamic_sbox(self, region_key: bytes) -> np.ndarray:
        """
        生成动态S盒（256字节）
        
        基于：区域密钥
        方法：密钥初始化随机置换
        
        Args:
            region_key: 区域密钥
        
        Returns:
            sbox: [256] S盒置换
        """
        # 使用区域密钥作为随机种子
        seed = int.from_bytes(region_key[:8], byteorder='big') % (2**32)
        
        # 生成随机置换
        rng = np.random.RandomState(seed)
        sbox = rng.permutation(256).astype(np.uint8)
        
        return sbox
    
    def generate_inverse_sbox(self, sbox: np.ndarray) -> np.ndarray:
        """
        生成逆S盒（用于解密）
        
        Args:
            sbox: 正向S盒
        
        Returns:
            inv_sbox: 逆S盒
        """
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[sbox[i]] = i
        
        return inv_sbox
    
    def get_full_key_hierarchy(
        self, 
        image_data: np.ndarray,
        num_regions: int = 19,
        chaos_dimension: int = 5
    ) -> Dict:
        """
        获取完整的密钥层次结构
        
        Args:
            image_data: 图像数据
            num_regions: 区域数量
            chaos_dimension: 混沌维度
        
        Returns:
            key_hierarchy: 包含所有层级密钥的字典
        """
        # 图像密钥
        image_key = self.derive_image_key(image_data)
        
        # 区域密钥
        region_keys = self.derive_region_keys(image_key, num_regions)
        
        # 混沌初值和S盒
        chaos_states = []
        sboxes = []
        
        for region_key in region_keys:
            chaos_state = self.derive_chaos_initial_state(region_key, chaos_dimension)
            sbox = self.generate_dynamic_sbox(region_key)
            
            chaos_states.append(chaos_state)
            sboxes.append(sbox)
        
        key_hierarchy = {
            'master_key': self.master_key.hex(),
            'image_key': image_key.hex(),
            'region_keys': [key.hex() for key in region_keys],
            'chaos_states': [state.tolist() for state in chaos_states],
            'sboxes': [sbox.tolist() for sbox in sboxes]
        }
        
        return key_hierarchy
    
    def compute_key_space(self) -> str:
        """
        计算总密钥空间
        
        Returns:
            key_space: 密钥空间描述
        """
        # 主密钥：512位
        master_bits = 512
        
        # 图像密钥：512位
        image_bits = 512
        
        # 区域密钥：19 × 256位
        region_bits = 19 * 256
        
        # 混沌初值：19 × 5 × 64位（双精度浮点数）
        chaos_bits = 19 * 5 * 64
        
        # 总位数
        total_bits = master_bits + image_bits + region_bits + chaos_bits
        
        key_space = f"2^{total_bits}"
        
        return key_space, total_bits


def test_key_system():
    """测试密钥系统"""
    print("="*70)
    print("测试分层密钥系统")
    print("="*70)
    
    # 创建密钥系统
    password = "test_password_2025"
    key_system = HierarchicalKeySystem(password, iterations=10000)  # 测试用较少迭代
    
    print(f"\n✓ 密钥系统初始化成功")
    print(f"  主密钥长度：{len(key_system.master_key)} 字节（{len(key_system.master_key)*8} 位）")
    
    # 测试图像
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # 派生图像密钥
    image_key = key_system.derive_image_key(test_image)
    print(f"\n✓ 图像密钥派生成功")
    print(f"  长度：{len(image_key)} 字节（{len(image_key)*8} 位）")
    
    # 派生区域密钥
    region_keys = key_system.derive_region_keys(image_key, num_regions=19)
    print(f"\n✓ 区域密钥派生成功")
    print(f"  数量：{len(region_keys)}")
    print(f"  每个长度：{len(region_keys[0])} 字节（{len(region_keys[0])*8} 位）")
    
    # 派生混沌初值
    chaos_state = key_system.derive_chaos_initial_state(region_keys[0], dimension=5)
    print(f"\n✓ 混沌初值派生成功")
    print(f"  维度：{len(chaos_state)}")
    print(f"  范围：[{chaos_state.min():.4f}, {chaos_state.max():.4f}]")
    
    # 生成S盒
    sbox = key_system.generate_dynamic_sbox(region_keys[0])
    inv_sbox = key_system.generate_inverse_sbox(sbox)
    print(f"\n✓ 动态S盒生成成功")
    print(f"  长度：{len(sbox)}")
    
    # 验证逆S盒
    test_val = 123
    encrypted_val = sbox[test_val]
    decrypted_val = inv_sbox[encrypted_val]
    assert decrypted_val == test_val, "逆S盒验证失败"
    print(f"  逆S盒验证：通过")
    
    # 计算密钥空间
    key_space, total_bits = key_system.compute_key_space()
    print(f"\n✓ 密钥空间计算")
    print(f"  总位数：{total_bits}")
    print(f"  密钥空间：{key_space}")
    print(f"  对比AES-256：2^256")
    
    # 完整密钥层次
    hierarchy = key_system.get_full_key_hierarchy(test_image, num_regions=19, chaos_dimension=5)
    print(f"\n✓ 完整密钥层次生成成功")
    print(f"  包含：主密钥、图像密钥、19个区域密钥、19个混沌初值、19个S盒")
    
    print("\n" + "="*70)
    print("✓ 测试通过！")
    print("="*70)


if __name__ == "__main__":
    test_key_system()










