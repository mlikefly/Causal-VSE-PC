"""
自适应隐私预算分配器

功能：
- 根据语义掩码、任务类型、全局隐私需求分配隐私预算
- 任务感知的动态分配
- 区域级细粒度控制
- 支持因果推断驱动的隐私预算建议（Causal-VSE-PC核心创新）

设计思路：
- 规则-based实现，无需训练
- 根据任务类型（分类/分割/检测）采用不同策略
- 敏感区域强加密，任务区域弱加密，背景区域不加密
- 可选：接受因果分析器的建议，动态调整隐私预算分配
"""

import torch
import torch.nn as nn
from typing import Dict, Union, Optional


class AdaptivePrivacyBudget:
    """
    自适应隐私预算分配器
    
    根据语义掩码、任务类型、全局隐私需求，动态分配每个区域的隐私预算。
    """
    
    def __init__(self):
        """初始化隐私预算分配器（规则-based，无需参数）"""
        pass
    
    def allocate(
        self,
        semantic_mask: torch.Tensor,
        task_type: str,
        global_privacy: float = 1.0,
        causal_suggestion: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        分配隐私预算
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码，值域[0, 1]
                - 1.0: 敏感区域（人脸、身份证等）
                - 0.5: 任务相关区域（物体、场景）
                - 0.0: 背景区域
            task_type: 任务类型
                - 'classification': 分类任务
                - 'segmentation': 分割任务
                - 'detection': 检测任务
            global_privacy: 全局隐私级别 [0.0, 1.0]
            causal_suggestion: 因果分析器的建议（可选）
                - 如果提供，将使用因果建议的隐私预算值
                - 格式: {'privacy_budget': {'sensitive': float, 'task': float, 'background': float}}
        
        Returns:
            privacy_map: [B, 1, H, W] 隐私预算图，值域[0, 1]
                - 1.0: 强加密
                - 0.3: 弱加密（保留可用性）
                - 0.0: 不加密
        """
        B, C, H, W = semantic_mask.shape
        device = semantic_mask.device
        
        # 如果有因果建议，使用因果建议的隐私预算值
        if causal_suggestion is not None and 'privacy_budget' in causal_suggestion:
            privacy_budget = causal_suggestion['privacy_budget']
            
            # 使用因果建议的隐私预算值
            sensitive_privacy = privacy_budget.get('sensitive', 0.9)
            task_privacy = privacy_budget.get('task', 0.3)
            background_privacy = privacy_budget.get('background', 0.0)
            
            # 根据语义掩码识别不同区域
            sensitive_regions = (semantic_mask > 0.7).float()
            task_regions = ((semantic_mask > 0.3) & (semantic_mask <= 0.7)).float()
            background_regions = (semantic_mask <= 0.3).float()
            
            # 应用因果建议的隐私预算
            privacy_map = (
                sensitive_regions * sensitive_privacy +
                task_regions * task_privacy +
                background_regions * background_privacy
            )
        else:
            # 否则，使用规则-based策略
            # 根据任务类型选择分配策略
            if task_type == 'classification':
                # 分类任务：保护身份，保留物体特征
                # 敏感区域：强加密（privacy=0.7~1.0）
                # 任务区域：弱加密（privacy=0.3），保留分类能力
                # 背景：不加密（privacy=0.0）
                privacy_map = semantic_mask * 0.7 + (1 - semantic_mask) * 0.3
                
            elif task_type == 'segmentation':
                # 分割任务：保护敏感区域，保留场景结构
                # 敏感区域：强加密（privacy=1.0）
                # 任务区域：极弱加密（privacy=0.1），保留分割能力
                # 背景：不加密（privacy=0.0）
                privacy_map = semantic_mask * 1.0 + (1 - semantic_mask) * 0.1
                
            elif task_type == 'detection':
                # 检测任务：保护身份，保留物体检测能力
                # 敏感区域：强加密（privacy=0.8）
                # 任务区域：弱加密（privacy=0.2），保留检测能力
                # 背景：不加密（privacy=0.0）
                privacy_map = semantic_mask * 0.8 + (1 - semantic_mask) * 0.2
                
            else:
                # 默认策略：中等加密强度
                privacy_map = semantic_mask * 0.5 + (1 - semantic_mask) * 0.3
        
        # 应用全局隐私需求
        privacy_map = privacy_map * global_privacy
        
        # 确保值域在[0, 1]
        privacy_map = torch.clamp(privacy_map, 0.0, 1.0)
        
        return privacy_map
    
    def allocate_with_regions(
        self,
        semantic_mask: torch.Tensor,
        region_types: Dict[str, torch.Tensor],
        task_type: str,
        global_privacy: float = 1.0
    ) -> torch.Tensor:
        """
        基于区域类型的细粒度隐私预算分配
        
        Args:
            semantic_mask: [B, 1, H, W] 语义掩码
            region_types: 区域类型字典
                - 'sensitive': 敏感区域掩码（人脸等）
                - 'task_relevant': 任务相关区域掩码（物体等）
                - 'background': 背景区域掩码
            task_type: 任务类型
            global_privacy: 全局隐私级别
        
        Returns:
            privacy_map: [B, 1, H, W] 隐私预算图
        """
        B, C, H, W = semantic_mask.shape
        device = semantic_mask.device
        
        # 初始化隐私预算图
        privacy_map = torch.zeros(B, 1, H, W, device=device)
        
        # 根据区域类型分配隐私预算
        if 'sensitive' in region_types:
            # 敏感区域：强加密
            privacy_map += region_types['sensitive'] * 1.0
        
        if 'task_relevant' in region_types:
            # 任务相关区域：根据任务类型调整
            if task_type == 'classification':
                privacy_map += region_types['task_relevant'] * 0.3
            elif task_type == 'segmentation':
                privacy_map += region_types['task_relevant'] * 0.1
            elif task_type == 'detection':
                privacy_map += region_types['task_relevant'] * 0.2
            else:
                privacy_map += region_types['task_relevant'] * 0.3
        
        if 'background' in region_types:
            # 背景区域：不加密
            privacy_map += region_types['background'] * 0.0
        
        # 应用全局隐私需求
        privacy_map = privacy_map * global_privacy
        
        # 确保值域在[0, 1]
        privacy_map = torch.clamp(privacy_map, 0.0, 1.0)
        
        return privacy_map


def test_privacy_budget():
    """测试隐私预算分配器"""
    print("测试自适应隐私预算分配器")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    allocator = AdaptivePrivacyBudget()
    
    # 创建测试语义掩码
    semantic_mask = torch.rand(2, 1, 256, 256, device=device)
    
    # 测试不同任务类型
    for task_type in ['classification', 'segmentation', 'detection']:
        privacy_map = allocator.allocate(semantic_mask, task_type, global_privacy=1.0)
        print(f"  {task_type}: 隐私预算范围 [{privacy_map.min():.2f}, {privacy_map.max():.2f}]")
    
    print("✓ 测试完成")


if __name__ == "__main__":
    test_privacy_budget()

