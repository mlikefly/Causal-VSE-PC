"""
攻击模型集合
用于评估加密系统的安全性

攻击类型：
1. 识别攻击（ArcFace/FaceNet）- 验收：Top-1≤5%
2. 重建攻击（VAE/GAN）- 验收：PSNR≤8dB
3. 属性推断攻击
4. 分割攻击
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class SimpleArcFace(nn.Module):
    """简化版ArcFace识别模型（用于攻击测试）"""
    
    def __init__(self, embedding_dim=512, num_classes=1000):
        super().__init__()
        
        # 特征提取器（ResNet-like）
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 分类器（用于训练）
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_embedding: 是否返回嵌入向量
        
        Returns:
            embedding or logits
        """
        # 特征提取
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        
        # 嵌入
        embedding = self.embedding(feat)
        
        if return_embedding:
            return F.normalize(embedding, p=2, dim=1)
        
        # 分类
        logits = self.classifier(embedding)
        return logits
    
    def get_similarity(self, img1, img2):
        """计算两张图像的相似度"""
        emb1 = self.forward(img1, return_embedding=True)
        emb2 = self.forward(img2, return_embedding=True)
        
        # 余弦相似度
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        return similarity


class VAEReconstructor(nn.Module):
    """VAE重建攻击模型"""
    
    def __init__(self, latent_dim=128, image_channels=3):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 8
            nn.ReLU(inplace=True),
        )
        
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        
        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),  # 256
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """编码"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 8, 8)
        return self.decoder(h)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAEReconstructorStrong(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class AttributeClassifier(nn.Module):
    """属性推断攻击模型（多标签分类）"""
    
    def __init__(self, num_attributes=40, image_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_attributes)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            logits: [B, num_attributes]
        """
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class SegmentationAttacker(nn.Module):
    """分割攻击模型（简化UNet）"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈
        self.bottleneck = self._conv_block(512, 1024)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # 解码
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out_conv(dec1)
        return torch.sigmoid(out)


class AttackEvaluator:
    """攻击评估器（统一接口）"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 强识别器优先（FaceNet-VGGFace2），不可用时才退回简化模型
        try:
            from .strong_recognizers import StrongArcFaceWrapper  # type: ignore
            self.arcface = StrongArcFaceWrapper(device=device).to(device)
            self.recognizer_name = "StrongArcFaceWrapper"
        except Exception as e:
            print(f"⚠️ 强识别器加载失败，改用简化识别器 SimpleArcFace: {e}")
            self.arcface = SimpleArcFace().to(device)
            self.recognizer_name = "SimpleArcFace"
        
        self.vae = VAEReconstructor().to(device)
        self.vae_strong = VAEReconstructorStrong().to(device)
        self.attr_classifier = AttributeClassifier().to(device)
        self.segmenter = SegmentationAttacker().to(device)
        
        # 设置为评估模式
        self.arcface.eval()
        self.vae.eval()
        self.vae_strong.eval()
        self.attr_classifier.eval()
        self.segmenter.eval()

        # 全图库检索用的特征与ID（跨batch累计）
        self.gallery_embeddings: List[torch.Tensor] = []
        self.gallery_ids: List[int] = []
        self._next_id: int = 0

    @torch.no_grad()
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """统一获取人脸嵌入，兼容强识别器与SimpleArcFace。"""
        if isinstance(self.arcface, SimpleArcFace):
            return self.arcface(x, return_embedding=True)
        return self.arcface(x)

    @torch.no_grad()
    def _append_gallery(self, emb: torch.Tensor) -> List[int]:
        """将一批原图特征加入图库，并返回分配的ID列表"""
        batch = emb.detach()
        ids = list(range(self._next_id, self._next_id + batch.size(0)))
        self._next_id += batch.size(0)
        self.gallery_embeddings.append(batch)
        self.gallery_ids.extend(ids)
        return ids
    
    @torch.no_grad()
    def get_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """计算两批图像在当前识别器下的余弦相似度向量。"""
        emb1 = self._get_embeddings(img1)
        emb2 = self._get_embeddings(img2)
        return F.cosine_similarity(emb1, emb2, dim=1)
    
    def load_pretrained(self, model_dir: str):
        """加载预训练模型（如果存在）"""
        model_dir = Path(model_dir)
        
        models = {
            'arcface': self.arcface,
            'vae': self.vae,
            'attr_classifier': self.attr_classifier,
            'segmenter': self.segmenter
        }
        
        for name, model in models.items():
            model_path = model_dir / f"{name}.pth"
            if model_path.exists():
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"✓ 加载预训练模型: {name}")
                except Exception as e:
                    print(f"⚠️ 加载 {name} 失败: {e}")
    
    @torch.no_grad()
    def evaluate_recognition_attack(
        self, 
        original: torch.Tensor, 
        encrypted: torch.Tensor,
        add_to_gallery: bool = True
    ) -> Dict[str, float]:
        """
        评估识别攻击
        
        - 使用强识别器（若可用）在“密文→全图库(原图)”上做最近邻检索
        - 返回 Top-1 准确率（≤5% 为通过）与成对余弦相似度（仅作参考）
        """
        # 计算嵌入
        emb_orig = self._get_embeddings(original)
        emb_enc = self._get_embeddings(encrypted)
        
        # 配对相似度（同一索引的原/密配对），仅供参考
        pair_similarity = F.cosine_similarity(emb_orig, emb_enc, dim=1).mean().item()
        
        # 先把当前原图批加入图库，获取其ID作为真值
        if add_to_gallery:
            true_ids = self._append_gallery(emb_orig)
        else:
            # 不加入图库时，用当前批一个临时ID区间，但不写入状态
            start_id = self._next_id
            true_ids = list(range(start_id, start_id + emb_orig.size(0)))

        # 组装全图库特征矩阵
        gallery = torch.cat(self.gallery_embeddings, dim=0) if len(self.gallery_embeddings) > 0 else emb_orig
        gid = torch.tensor(self.gallery_ids, device=gallery.device) if len(self.gallery_ids) > 0 else torch.tensor(true_ids, device=gallery.device)

        # 检索：对每个密文查询，找全图库最近邻
        sim = torch.matmul(emb_enc, gallery.t())  # [B, G]
        nn_idx = torch.argmax(sim, dim=1)        # [B]
        pred_ids = gid[nn_idx]

        # 计算Top-1（与当前批对应的真值ID对比）
        true_ids_tensor = torch.tensor(true_ids, device=pred_ids.device)
        top1_acc = (pred_ids == true_ids_tensor).float().mean().item()
        
        return {
            'cosine_similarity': pair_similarity,
            'top1_accuracy': top1_acc,
            'pass_threshold': top1_acc <= 0.05,
            'recognizer': self.recognizer_name
        }
    
    @torch.no_grad()
    def evaluate_reconstruction_attack(
        self, 
        original: torch.Tensor, 
        encrypted: torch.Tensor
    ) -> Dict[str, float]:
        """
        评估重建攻击
        
        验收标准：PSNR ≤ 8 dB
        """
        recon_base, _, _ = self.vae(encrypted)
        mse_base = F.mse_loss(recon_base, original)
        psnr_base = 10 * torch.log10(1.0 / (mse_base + 1e-10))
        lpips_base = F.mse_loss(recon_base, original)
        mu_orig = original.mean()
        mu_recon_base = recon_base.mean()
        sigma_orig = original.std()
        sigma_recon_base = recon_base.std()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_base = ((2 * mu_orig * mu_recon_base + c1) * (2 * sigma_orig * sigma_recon_base + c2)) / \
                    ((mu_orig ** 2 + mu_recon_base ** 2 + c1) * (sigma_orig ** 2 + sigma_recon_base ** 2 + c2))

        recon_strong, _, _ = self.vae_strong(encrypted)
        mse_strong = F.mse_loss(recon_strong, original)
        psnr_strong = 10 * torch.log10(1.0 / (mse_strong + 1e-10))
        lpips_strong = F.mse_loss(recon_strong, original)
        mu_recon_strong = recon_strong.mean()
        sigma_recon_strong = recon_strong.std()
        ssim_strong = ((2 * mu_orig * mu_recon_strong + c1) * (2 * sigma_orig * sigma_recon_strong + c2)) / \
                      ((mu_orig ** 2 + mu_recon_strong ** 2 + c1) * (sigma_orig ** 2 + sigma_recon_strong ** 2 + c2))

        if psnr_strong > psnr_base:
            psnr_val = psnr_strong.item()
            lpips_val = lpips_strong.item()
            ssim_val = ssim_strong.item()
            variant = 'strong'
        else:
            psnr_val = psnr_base.item()
            lpips_val = lpips_base.item()
            ssim_val = ssim_base.item()
            variant = 'base'

        return {
            'psnr': psnr_val,
            'lpips': lpips_val,
            'ssim': ssim_val,
            'pass_threshold': psnr_val <= 8.0,  # 验收标准
            'variant': variant
        }
    
    @torch.no_grad()
    def evaluate_attribute_attack(
        self, 
        original: torch.Tensor, 
        encrypted: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """评估属性推断攻击"""
        # 获取属性预测
        logits_orig = self.attr_classifier(original)
        logits_enc = self.attr_classifier(encrypted)
        
        pred_orig = torch.sigmoid(logits_orig)
        pred_enc = torch.sigmoid(logits_enc)
        
        if true_labels is not None:
            # 计算准确率
            acc_orig = ((pred_orig > 0.5) == true_labels).float().mean().item()
            acc_enc = ((pred_enc > 0.5) == true_labels).float().mean().item()
        else:
            # 使用模拟数据
            acc_orig = float('nan')
            acc_enc = float('nan')
        
        return {
            'accuracy_original': acc_orig,
            'accuracy_encrypted': acc_enc,
            'pass_threshold': acc_enc <= 0.55  # 随机猜测水平
        }
    
    def evaluate_all_attacks(
        self,
        original: torch.Tensor,
        encrypted: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """评估所有攻击"""
        
        results = {
            'recognition': self.evaluate_recognition_attack(original, encrypted),
            'reconstruction': self.evaluate_reconstruction_attack(original, encrypted),
            'attribute': self.evaluate_attribute_attack(original, encrypted, true_labels)
        }
        
        return results


if __name__ == "__main__":
    # 测试攻击模型
    print("="*70)
    print("测试攻击模型")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = AttackEvaluator(device=device)
    
    # 创建测试数据
    original = torch.rand(4, 3, 256, 256).to(device)
    encrypted = torch.rand(4, 3, 256, 256).to(device)
    
    # 评估攻击
    results = evaluator.evaluate_all_attacks(original, encrypted)
    
    print("\n识别攻击:")
    print(f"  余弦相似度: {results['recognition']['cosine_similarity']:.4f}")
    print(f"  Top-1准确率: {results['recognition']['top1_accuracy']:.4f}")
    print(f"  通过验收: {results['recognition']['pass_threshold']}")
    
    print("\n重建攻击:")
    print(f"  PSNR: {results['reconstruction']['psnr']:.2f} dB")
    print(f"  LPIPS: {results['reconstruction']['lpips']:.4f}")
    print(f"  通过验收: {results['reconstruction']['pass_threshold']}")
    
    print("\n属性推断:")
    print(f"  原图准确率: {results['attribute']['accuracy_original']:.4f}")
    print(f"  密文准确率: {results['attribute']['accuracy_encrypted']:.4f}")
    print(f"  通过验收: {results['attribute']['pass_threshold']}")

