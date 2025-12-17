"""
属性测试：加密系统的正确性属性

**Feature: top-tier-journal-upgrade**
**Property 2: Deterministic Encryption Round-Trip**
**Property 3: C-view AEAD Integrity**
**Validates: Requirements 1.3, 1.6, 9.1, 9.2**
"""

import pytest
import torch
import numpy as np
import hashlib
import hmac
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite

from src.cipher.scne_cipher import SCNECipherAPI, SCNECipher
from src.crypto.key_system import HierarchicalKeySystem


# ============ 自定义策略 ============

@composite
def privacy_levels(draw):
    """隐私级别策略"""
    return draw(st.sampled_from([0.3, 0.5, 0.7, 1.0]))


@composite
def task_types(draw):
    """任务类型策略"""
    return draw(st.sampled_from(['classification', 'detection', 'segmentation']))


@composite
def image_ids(draw):
    """图像ID策略"""
    return draw(st.text(
        alphabet='abcdefghijklmnopqrstuvwxyz0123456789',
        min_size=8,
        max_size=16
    ))


@composite
def datasets(draw):
    """数据集策略"""
    return draw(st.sampled_from(['celeba', 'celebahq', 'fairface', 'openimages']))


@composite
def splits(draw):
    """数据划分策略"""
    return draw(st.sampled_from(['train', 'val', 'test']))


# ============ Property 2: Deterministic Encryption Round-Trip ============

class TestDeterministicEncryption:
    """
    **Feature: top-tier-journal-upgrade, Property 2: Deterministic Encryption Round-Trip**
    **Validates: Requirements 1.6, 9.1, 9.2**
    
    *For any* master_key, image_id, task_type, and privacy_map combination, 
    encrypting the same image multiple times SHALL produce identical Z-view 
    and C-view outputs (byte-for-byte equality).
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """设置测试环境"""
        self.password = "test_property_password_2025"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @settings(max_examples=20, deadline=None)
    @given(
        privacy_level=privacy_levels(),
        task_type=task_types(),
        image_id=image_ids()
    )
    def test_deterministic_encryption_produces_identical_output(
        self, privacy_level, task_type, image_id
    ):
        """
        **Property 2: Deterministic Encryption Round-Trip**
        
        相同输入应产生完全相同的输出
        """
        # 跳过空image_id
        assume(len(image_id) > 0)
        
        # 创建确定性API
        api = SCNECipherAPI(
            password=self.password,
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        # 固定随机种子生成测试图像
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=self.device)
        
        # 第一次加密
        enc1, info1 = api.encrypt_simple(
            test_img.clone(),
            privacy_level=privacy_level,
            image_id=image_id,
            task_type=task_type
        )
        
        # 第二次加密（相同参数）
        enc2, info2 = api.encrypt_simple(
            test_img.clone(),
            privacy_level=privacy_level,
            image_id=image_id,
            task_type=task_type
        )
        
        # 验证密文完全相同
        diff = (enc1 - enc2).abs().max().item()
        assert diff == 0, f"确定性加密应产生相同密文，但差异为 {diff}"
        
        # 验证nonce相同
        if info1.get('crypto_wrap') and info2.get('crypto_wrap'):
            nonce1 = info1['crypto_wrap'].get('nonces', [])
            nonce2 = info2['crypto_wrap'].get('nonces', [])
            assert nonce1 == nonce2, "确定性加密应产生相同nonce"
    
    @settings(max_examples=10, deadline=None)
    @given(
        privacy_level=privacy_levels(),
        task_type1=task_types(),
        task_type2=task_types()
    )
    def test_different_task_type_produces_different_nonce(
        self, privacy_level, task_type1, task_type2
    ):
        """
        不同task_type应产生不同nonce
        """
        assume(task_type1 != task_type2)
        
        api = SCNECipherAPI(
            password=self.password,
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=self.device)
        image_id = "test_image_001"
        
        _, info1 = api.encrypt_simple(
            test_img.clone(),
            privacy_level=privacy_level,
            image_id=image_id,
            task_type=task_type1
        )
        
        _, info2 = api.encrypt_simple(
            test_img.clone(),
            privacy_level=privacy_level,
            image_id=image_id,
            task_type=task_type2
        )
        
        if info1.get('crypto_wrap') and info2.get('crypto_wrap'):
            nonce1 = info1['crypto_wrap'].get('nonces', [])
            nonce2 = info2['crypto_wrap'].get('nonces', [])
            assert nonce1 != nonce2, "不同task_type应产生不同nonce"


# ============ Property 3: C-view AEAD Integrity ============

class TestAEADIntegrity:
    """
    **Feature: top-tier-journal-upgrade, Property 3: C-view AEAD Integrity**
    **Validates: Requirements 1.3**
    
    *For any* C-view ciphertext, decryption with the correct key SHALL succeed 
    and produce the original Z-view; decryption with an incorrect key or 
    tampered ciphertext SHALL fail with an authentication error.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """设置测试环境"""
        self.password = "test_aead_password_2025"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @settings(max_examples=20, deadline=None)
    @given(
        privacy_level=privacy_levels(),
        task_type=task_types(),
        dataset=datasets(),
        split=splits()
    )
    def test_correct_key_decryption_succeeds(
        self, privacy_level, task_type, dataset, split
    ):
        """
        **Property 3: C-view AEAD Integrity - 正确密钥解密成功**
        
        使用正确密钥解密应成功
        """
        api = SCNECipherAPI(
            password=self.password,
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=self.device)
        
        # 加密
        encrypted, enc_info = api.cipher.encrypt(
            test_img,
            mask=torch.ones(1, 1, 256, 256, device=self.device),
            chaos_params=torch.rand(1, 19, 3, device=self.device) * torch.tensor([10.0, 5.0, 1.0], device=self.device),
            password=self.password,
            privacy_level=privacy_level,
            image_id='test_aead_001',
            task_type=task_type,
            dataset=dataset,
            split=split
        )
        
        # 解密
        decrypted = api.cipher.decrypt(
            encrypted,
            enc_info,
            mask=torch.ones(1, 1, 256, 256, device=self.device),
            password=self.password
        )
        
        # 验证解密成功（误差在可接受范围内）
        error = (decrypted - test_img).abs().mean().item()
        assert error < 0.01, f"解密误差过大: {error}"
    
    @settings(max_examples=10, deadline=None)
    @given(
        privacy_level=privacy_levels(),
        task_type=task_types()
    )
    def test_tampered_aad_decryption_fails(self, privacy_level, task_type):
        """
        **Property 3: C-view AEAD Integrity - 篡改AAD解密失败**
        
        篡改AAD后解密应失败
        """
        api = SCNECipherAPI(
            password=self.password,
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=self.device)
        
        # 加密
        encrypted, enc_info = api.cipher.encrypt(
            test_img,
            mask=torch.ones(1, 1, 256, 256, device=self.device),
            chaos_params=torch.rand(1, 19, 3, device=self.device) * torch.tensor([10.0, 5.0, 1.0], device=self.device),
            password=self.password,
            privacy_level=privacy_level,
            image_id='test_tamper_001',
            task_type=task_type,
            dataset='celeba',
            split='train'
        )
        
        # 篡改AAD
        enc_info_tampered = dict(enc_info)
        if enc_info_tampered.get('crypto_wrap'):
            enc_info_tampered['crypto_wrap'] = dict(enc_info_tampered['crypto_wrap'])
            enc_info_tampered['crypto_wrap']['aad'] = 'v2|tampered|wrong|data|here'
        
        # 解密应失败
        with pytest.raises(ValueError, match="MAC verification failed"):
            api.cipher.decrypt(
                encrypted,
                enc_info_tampered,
                mask=torch.ones(1, 1, 256, 256, device=self.device),
                password=self.password
            )
    
    @settings(max_examples=10, deadline=None)
    @given(
        privacy_level=privacy_levels(),
        task_type=task_types()
    )
    def test_wrong_password_decryption_fails(self, privacy_level, task_type):
        """
        **Property 3: C-view AEAD Integrity - 错误密码解密失败**
        
        使用错误密码解密应失败或产生错误结果
        """
        api = SCNECipherAPI(
            password=self.password,
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        torch.manual_seed(42)
        test_img = torch.rand(1, 1, 256, 256, device=self.device)
        
        # 加密
        encrypted, enc_info = api.cipher.encrypt(
            test_img,
            mask=torch.ones(1, 1, 256, 256, device=self.device),
            chaos_params=torch.rand(1, 19, 3, device=self.device) * torch.tensor([10.0, 5.0, 1.0], device=self.device),
            password=self.password,
            privacy_level=privacy_level,
            image_id='test_wrong_pwd_001',
            task_type=task_type,
            dataset='celeba',
            split='train'
        )
        
        # 使用错误密码创建新API
        wrong_api = SCNECipherAPI(
            password="wrong_password_12345",
            image_size=256,
            deterministic=True,
            device=self.device
        )
        
        # 解密应失败（MAC验证失败）
        try:
            decrypted = wrong_api.cipher.decrypt(
                encrypted,
                enc_info,
                mask=torch.ones(1, 1, 256, 256, device=self.device),
                password="wrong_password_12345"
            )
            # 如果没有抛出异常，验证解密结果与原图差异很大
            error = (decrypted - test_img).abs().mean().item()
            assert error > 0.1, "错误密码解密应产生明显不同的结果"
        except ValueError as e:
            # MAC验证失败是预期行为
            assert "MAC verification failed" in str(e)


# ============ 辅助测试 ============

class TestNonceDerivation:
    """测试nonce派生逻辑"""
    
    def test_nonce_uniqueness(self):
        """测试nonce唯一性"""
        key_system = HierarchicalKeySystem('test_password', iterations=10000)
        
        def derive_nonce(image_id, task_type, pm_hash, zv_hash):
            context = f"cview|v2|{image_id}|{task_type}|{pm_hash}|{zv_hash}".encode('utf-8')
            k_nonce = hashlib.sha256(key_system.master_key + b'|nonce').digest()
            return hmac.new(k_nonce, context, hashlib.sha256).digest()[:16]
        
        nonces = set()
        for i in range(100):
            for task in ['classification', 'detection', 'segmentation']:
                pm_hash = hashlib.sha256(f'pm_{i}_{task}'.encode()).hexdigest()[:8]
                zv_hash = hashlib.sha256(f'zv_{i}_{task}'.encode()).hexdigest()[:8]
                nonce = derive_nonce(f'image_{i:04d}', task, pm_hash, zv_hash)
                nonces.add(nonce)
        
        assert len(nonces) == 300, f"应有300个唯一nonce，实际{len(nonces)}个"


class TestAADFormat:
    """测试AAD格式"""
    
    def test_aad_contains_all_fields(self):
        """测试AAD包含所有字段"""
        cipher = SCNECipher(password='test')
        
        aad = cipher._build_aad(
            sample_id='sample_001',
            dataset='celeba',
            split='train',
            task_type='classification',
            version=2
        )
        
        assert b'v2' in aad
        assert b'sample_001' in aad
        assert b'celeba' in aad
        assert b'train' in aad
        assert b'classification' in aad
    
    def test_aad_handles_missing_fields(self):
        """测试AAD处理缺失字段"""
        cipher = SCNECipher(password='test')
        
        aad = cipher._build_aad(
            sample_id='sample_002',
            task_type='detection'
        )
        
        assert b'sample_002' in aad
        assert b'detection' in aad
        # 缺失字段应为空字符串
        assert b'||' in aad


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
