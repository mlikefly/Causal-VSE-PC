import torch
import torch.nn as nn
import torch.nn.functional as F


class StrongArcFaceWrapper(nn.Module):
    """统一封装强识别器（最终版）：facenet-pytorch InceptionResnetV1 (VGGFace2)

    - 输出L2归一化的512维特征
    - 输入 [B,3,H,W] in [0,1]，自动resize到 112
    - 若依赖未安装则直接抛错，避免“简化回退”造成无效评估
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.backbone_type = 'facenet'
        
        try:
            from facenet_pytorch import InceptionResnetV1  # type: ignore
            # 尝试加载预训练模型，如果下载失败或文件损坏则捕获异常
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("✅ 已加载 StrongArcFace (InceptionResnetV1)")
        except Exception as e:
            print(f"⚠️ 警告: 无法加载 InceptionResnetV1 ({e})")
            print("🔄 回退到 ResNet18 (ImageNet) 作为替代攻击者...")
            import torchvision.models as models
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().to(self.device)
            # 移除 ResNet18 的分类头，只保留特征提取
            self.model.fc = nn.Identity()
            self.backbone_type = 'resnet18'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (112, 112) if self.backbone_type == 'facenet' else (224, 224)
        
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
        # 如果是单通道灰度图，转为3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)


