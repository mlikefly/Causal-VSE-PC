#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-Net显著性检测器 - 完整实现
用于识别图像中的重要区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """U-Net的双卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块（最大池化 + 双卷积）"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块（转置卷积 + 拼接 + 双卷积）"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: 来自上层的上采样特征
        # x2: 来自编码器的跳跃连接特征
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetSaliencyDetector(nn.Module):
    """
    完整的U-Net显著性检测网络
    
    Architecture:
        - 编码器: 4个下采样块
        - 瓶颈层: 1个双卷积块
        - 解码器: 4个上采样块
        - 输出: 1通道显著性图
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 编码器（下采样）
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # 解码器（上采样）
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
        
        Returns:
            显著性图 (B, 1, H, W)，值在[0, 1]范围
        """
        # 编码器
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)
        
        # 解码器（带跳跃连接）
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # 输出
        logits = self.outc(x)  # (B, 1, H, W)
        out = torch.sigmoid(logits)
        
        return out
    
    def get_num_params(self) -> int:
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters())


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = UNetSaliencyDetector(in_channels=1, base_channels=64)
    
    print("="*60)
    print("U-Net显著性检测器")
    print("="*60)
    print(f"输入通道: {model.in_channels}")
    print(f"基础通道: {model.base_channels}")
    print(f"参数数量: {model.get_num_params():,}")
    
    # 测试前向传播
    x = torch.randn(1, 1, 256, 256)
    print(f"\n测试输入形状: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"输出形状: {out.shape}")
    print(f"输出范围: [{out.min():.4f}, {out.max():.4f}]")
    print("="*60)
    print("✓ 测试通过！")




