import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
from src.core.chaotic_encryptor import StandardChaoticCipher

# ==========================================
# 1. 极简数据加载器 (支持 data 目录或随机生成)
# ==========================================
class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, image_size=128, max_samples=200, debug=False):
        self.image_size = image_size
        self.image_paths = []
        
        if debug:
             max_samples = 20 # Debug 模式只取少量数据
        
        # 扫描 data 目录
        all_images = []
        found_enough = False
        
        for root, dirs, files in os.walk(root_dir):
            if found_enough:
                break
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_images.append(os.path.join(root, f))
                    if len(all_images) >= max_samples:
                        found_enough = True
                        break
        
        self.image_paths = all_images
        
        self.use_real_data = len(self.image_paths) > 0
        if self.use_real_data:
            print(f"✅ 在 {root_dir} 快速采样了 {len(self.image_paths)} 张图片路径（最多 {max_samples} 张），内存安全。")
        else:
            print(f"⚠️ 在 {root_dir} 未找到图片，将使用【随机生成数据】进行测试。")
            self.fake_len = 100 # 假数据数量

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths) if self.use_real_data else self.fake_len

    def __getitem__(self, idx):
        if self.use_real_data:
            try:
                img_path = self.image_paths[idx]
                img = Image.open(img_path).convert('RGB')
                return self.transform(img)
            except Exception as e:
                print(f"Error loading {self.image_paths[idx]}: {e}")
                return torch.rand(3, self.image_size, self.image_size)
        else:
            # 生成随机图，带一点结构让 CNN 能提取特征
            return torch.rand(3, self.image_size, self.image_size)

# ==========================================
# 2. 策略网络 & 攻击者
# ==========================================
class PolicyNet(nn.Module):
    """输入图像，输出 diffusion_strength (0~1)"""
    def __init__(self):
        super().__init__()
        # 简单的 CNN
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),   # 64
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),  # 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # 16
            nn.AdaptiveAvgPool2d((1, 1)),           # [B, 64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class Q2System(nn.Module):
    def __init__(self, fixed_iterations=8.0):
        super().__init__()
        self.policy = PolicyNet()
        self.encryptor = StandardChaoticCipher()
        self.fixed_iterations = fixed_iterations
        
        # 攻击者：ResNet18 (冻结)
        # 用它来计算 Feature 损失，代表“隐私泄露程度”
        self.attacker = models.resnet18(pretrained=True)
        self.attacker.eval()
        for param in self.attacker.parameters():
            param.requires_grad = False

    def get_attack_feature(self, x):
        # 简单的提取 ResNet 倒数第二层特征的方法
        # 这里直接用 forward，取 fc 之前的输出不太方便，
        # 简单起见：直接用 logits 作为 "语义特征"
        return self.attacker(x)

    def forward(self, x):
        # 1. 策略输出强度
        strengths = self.policy(x) # [B]
        
        # 2. 构造加密参数
        # 我们的 StandardChaoticCipher 期望 params={'iterations': ..., 'strength': ...}
        # 这里我们只用 strengths 控制扩散强度
        b = x.shape[0]
        
        # 生成随机密钥 [B, 2]
        key = torch.rand(b, 2, device=x.device)
        
        # 构造 params 字典
        # strength 广播成 [B, 1, 1, 1] 以便和图像相乘
        params = {
            'iterations': int(self.fixed_iterations),
            'strength': strengths.view(b, 1, 1, 1)
        }
        
        # 3. 加密
        # 调用 StandardChaoticCipher.encrypt(images, key, params)
        encrypted = self.encryptor(x, key, params)
        
        return encrypted, strengths

# ==========================================
# 3. 主训练脚本
# ==========================================
def train_q2_real():
    parser = argparse.ArgumentParser(description="Q2 Real Experiment")
    parser.add_argument("--mode", type=str, default="ranked", choices=["teacher", "privacy", "ranked"], help="Training mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (less data)")
    args = parser.parse_args()
    
    print(f"🚀 启动 Q2-Real 实验：基于 ResNet 攻击者的策略自适应验证 (Mode: {args.mode})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    # 为了控制实验时间，这里限制最多加载 100 张图片
    dataset = SimpleImageDataset(data_dir, image_size=128, max_samples=100, debug=args.debug)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    if len(dataset) == 0:
        print("❌ 数据集为空且生成失败，退出。")
        return

    # 2. 模型
    model = Q2System().to(device)

    # mode 用于切换训练模式：
    mode = args.mode  # 使用命令行参数

    # 3. 训练
    target_avg_strength = 0.5 # 我们希望平均强度控制在 0.5

    # 根据模式设置学习率和训练轮数
    if mode == "teacher":
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-4)
        epochs = args.epochs
    elif mode == "ranked":
        # 排序 loss 模式：学习率略大，epoch 控制较小以节约时间
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
        epochs = args.epochs
    else:
        # 纯 privacy 模式：仅作短实验/对照
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
        epochs = args.epochs
    
    history = {'loss': [], 'privacy': [], 'budget': []}

    for epoch in range(epochs):
        model.policy.train()
        total_loss = 0
        
        for i, images in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            
            # 前向
            encrypted, strengths = model(images)
            
            # --- Loss 计算 ---
            if mode == "teacher":
                # 使用攻击者的置信度作为“易识别程度”，直接监督策略输出
                with torch.no_grad():
                    logits = model.get_attack_feature(images)
                    probs = F.softmax(logits, dim=1)
                    confidences, _ = probs.max(dim=1)  # [B], 0~1，越大越容易被识别

                    # 将置信度标准化并放大差异，再映射到 [0,1]：易识别 -> 强度更高
                    conf_mean_batch = confidences.mean()
                    conf_std_batch = confidences.std(unbiased=False)
                    normalized = (confidences - conf_mean_batch) / (conf_std_batch + 1e-6)
                    alpha = 0.5  # 放大系数，控制强弱拉开的程度
                    target_strength = torch.clamp(0.5 + alpha * normalized, 0.0, 1.0)

                loss = F.mse_loss(strengths, target_strength.detach())
                privacy_loss = torch.tensor(0.0, device=device)
                budget_loss = torch.tensor(0.0, device=device)

                if i % 10 == 0:
                    conf_mean = confidences.mean().item()
                    conf_min = confidences.min().item()
                    conf_max = confidences.max().item()
                    str_mean = strengths.mean().item()
                    str_min = strengths.min().item()
                    str_max = strengths.max().item()
                    print(f"    Teacher batch stats: conf_mean={conf_mean:.4f}, conf_min={conf_min:.4f}, conf_max={conf_max:.4f}; "
                          f"str_mean={str_mean:.4f}, str_min={str_min:.4f}, str_max={str_max:.4f}")
            elif mode == "ranked":
                # Q2-v2：在隐私损失 + 预算的基础上，引入排序 loss，鼓励“更易识别样本的强度更大”

                # A. 隐私损失 (Privacy Loss)
                # 先提取原图特征与置信度（不反传到输入）
                with torch.no_grad():
                    logits = model.get_attack_feature(images)
                    probs = F.softmax(logits, dim=1)
                    confidences, _ = probs.max(dim=1)  # [B]
                    feat_orig = logits.detach()

                feat_enc = model.get_attack_feature(encrypted)
                cos_sim = F.cosine_similarity(feat_enc, feat_orig)
                privacy_loss = cos_sim.mean()

                # B. 预算约束 (Budget Loss)
                avg_str = strengths.mean()
                budget_loss = (avg_str - target_avg_strength) ** 2

                # C. 排序约束 (Ranking Loss)
                # 随机打乱一个副本，构造 (a,b) 对；若 conf_a > conf_b，则希望 strength_a >= strength_b + margin
                idx = torch.randperm(confidences.size(0), device=device)
                conf_a = confidences
                conf_b = confidences[idx]
                str_a = strengths
                str_b = strengths[idx]

                mask = conf_a > conf_b + 1e-3
                if mask.any():
                    diff_str = str_a[mask] - str_b[mask]
                    margin = 0.02
                    rank_loss = F.relu(margin - diff_str).mean()
                else:
                    rank_loss = torch.tensor(0.0, device=device)

                # D. 总损失：隐私 + 预算 + 排序
                loss = privacy_loss + 1.0 * budget_loss + 0.5 * rank_loss
            else:
                # 原始隐私 + 预算版本（短训练，对照用）
                # A. 隐私损失 (Privacy Loss)
                # 目标：让密文的 Feature 和原图 Feature 越不象越好
                with torch.no_grad():
                    feat_orig = model.get_attack_feature(images).detach()
                feat_enc = model.get_attack_feature(encrypted)

                cos_sim = F.cosine_similarity(feat_enc, feat_orig)
                privacy_loss = cos_sim.mean()  # 最小化相似度

                # B. 预算约束 (Budget Loss)
                avg_str = strengths.mean()
                budget_loss = (avg_str - target_avg_strength) ** 2

                # C. 总损失
                loss = privacy_loss + 2.0 * budget_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss={loss.item():.4f} (Priv={privacy_loss.item():.4f}, Budg={budget_loss.item():.4f}) Str={strengths.mean().item():.4f}")

    # 4. 验证自适应性 (画图)
    print("\n=== 验证自适应性 ===")
    model.eval()
    all_diffs = [] # 难度 (Baseline Confidence / Feature Norm)
    all_strs = []  # 策略输出强度
    
    # 这里我们要定义一个“难度”。
    # 对于 ResNet，我们可以用 "分类置信度的最大值" (Max Probability) 作为“易识别程度”。
    # 容易识别的图 -> Max Prob 高 -> 我们希望 Strength 高
    # 难识别的图 -> Max Prob 低 -> 我们希望 Strength 低
    
    with torch.no_grad():
        # 抽样验证
        for i, images in enumerate(dataloader):
            if i > 10: break # 只看几批
            images = images.to(device)
            
            # 1. 计算 Baseline 难度 (Confidence)
            logits = model.get_attack_feature(images) # [B, 1000]
            probs = F.softmax(logits, dim=1)
            confidences, _ = probs.max(dim=1) # [B] 0~1
            
            # 2. 策略输出
            _, strengths = model(images)
            
            all_diffs.extend(confidences.cpu().numpy())
            all_strs.extend(strengths.cpu().numpy())

    # 画图
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_diffs, all_strs, alpha=0.6)
        plt.xlabel("Attacker Confidence (Baseline Difficulty)")
        plt.ylabel("Learned Diffusion Strength")
        plt.title("Q2 Real: Adaptivity Check")
        plt.grid(True, alpha=0.3)
        
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "q2_real_adaptivity.png")
        plt.savefig(out_path)
        print(f"✅ 结果图已保存至: {out_path}")
        
        all_diffs_arr = np.array(all_diffs)
        all_strs_arr = np.array(all_strs)

        print(f"统计: conf_mean={all_diffs_arr.mean():.4f}, conf_min={all_diffs_arr.min():.4f}, conf_max={all_diffs_arr.max():.4f}; "
              f"str_mean={all_strs_arr.mean():.4f}, str_min={all_strs_arr.min():.4f}, str_max={all_strs_arr.max():.4f}")

        # 简单统计相关性
        corr = np.corrcoef(all_diffs_arr, all_strs_arr)[0, 1]
        print(f"📊 难度(置信度) 与 强度 的相关系数: {corr:.4f}")
        
        if corr > 0.1:
            print("✅ 发现正相关：策略网络倾向于对‘易识别’(高置信度)样本施加更强加密。")
        else:
            print("⚠️ 相关性较弱：可能是训练不足，或 ResNet 对该数据特征提取不稳定。")
            
    except Exception as e:
        print(f"画图失败: {e}")

if __name__ == "__main__":
    train_q2_real()
