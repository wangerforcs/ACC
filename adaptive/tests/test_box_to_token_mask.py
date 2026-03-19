#!/usr/bin/env python3
"""
测试box_to_token_mask方法的功能
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from .adaptivepolicy import UncertaintyEngine

def visualize_mask(mask, title="Mask"):
    """可视化mask"""
    grid_size = (16, 16)
    mask_2d = mask.view(grid_size)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_2d, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"{title}.png")

def test_box_to_token_mask():
    """测试box_to_token_mask方法"""
    # 初始化不确定性引擎
    engine = UncertaintyEngine(grid_size=(16, 16), image_size=(224, 224))
    
    # 测试1: 中心位置的box
    print("测试1: 中心位置的box")
    box_center = torch.tensor([[0.35, 0.35, 0.65, 0.65]])  # [x1, y1, x2, y2]
    mask_center = engine.boxes_to_token_mask(box_center)
    print(f"中心box: {box_center}")
    print(f"Mask总和: {mask_center.sum()}")
    visualize_mask(mask_center, "中心位置的box")
    
    # 测试2: 边缘位置的box
    print("\n测试2: 边缘位置的box")
    box_edge = torch.tensor([[0.0, 0.0, 0.2, 0.2]])  # [x1, y1, x2, y2]
    mask_edge = engine.boxes_to_token_mask(box_edge)
    print(f"边缘box: {box_edge}")
    print(f"Mask总和: {mask_edge.sum()}")
    visualize_mask(mask_edge, "边缘位置的box")
    
    # 测试3: 大box
    print("\n测试3: 大box")
    box_large = torch.tensor([[0.1, 0.1, 0.9, 0.9]])  # [x1, y1, x2, y2]
    mask_large = engine.boxes_to_token_mask(box_large)
    print(f"大box: {box_large}")
    print(f"Mask总和: {mask_large.sum()}")
    visualize_mask(mask_large, "大box")
    
    # 测试4: 小box
    print("\n测试4: 小box")
    box_small = torch.tensor([[0.45, 0.45, 0.55, 0.55]])  # [x1, y1, x2, y2]
    mask_small = engine.boxes_to_token_mask(box_small)
    print(f"小box: {box_small}")
    print(f"Mask总和: {mask_small.sum()}")
    visualize_mask(mask_small, "小box")
    
    # 测试5: 超出边界的box
    print("\n测试5: 超出边界的box")
    box_out = torch.tensor([[1.0, 1.0, 1.3, 1.3]])  # [x1, y1, x2, y2]
    mask_out = engine.boxes_to_token_mask(box_out)
    print(f"超出边界的box: {box_out}")
    print(f"Mask总和: {mask_out.sum()}")
    visualize_mask(mask_out, "超出边界的box")
    
    # 测试6: 负值的box
    print("\n测试6: 负值的box")
    box_negative = torch.tensor([[-0.3, -0.3, 0.0, 0.0]])  # [x1, y1, x2, y2]
    mask_negative = engine.boxes_to_token_mask(box_negative)
    print(f"负值的box: {box_negative}")
    print(f"Mask总和: {mask_negative.sum()}")
    visualize_mask(mask_negative, "负值的box")
    
    # 测试7: 宽度或高度为负值的box
    print("\n测试7: 宽度或高度为负值的box")
    box_negative_dim = torch.tensor([[0.65, 0.35, 0.35, 0.65]])  # [x1, y1, x2, y2] (x2 < x1)
    mask_negative_dim = engine.boxes_to_token_mask(box_negative_dim)
    print(f"负值维度的box: {box_negative_dim}")
    print(f"Mask总和: {mask_negative_dim.sum()}")
    visualize_mask(mask_negative_dim, "负值维度的box")
    
    # 测试8: 空box
    print("\n测试8: 空box")
    box_empty = torch.tensor([])
    mask_empty = engine.boxes_to_token_mask(box_empty)
    print(f"空box: {box_empty}")
    print(f"Mask总和: {mask_empty.sum()}")
    visualize_mask(mask_empty, "空box")
    
    # 测试9: None box
    print("\n测试9: None box")
    mask_none = engine.boxes_to_token_mask(None)
    print(f"None box")
    print(f"Mask总和: {mask_none.sum()}")
    visualize_mask(mask_none, "None box")
    
    # 测试10: 特定的boxes
    print("\n测试10: 特定的boxes")
    # 输入已经是xyxy格式
    specific_boxes = torch.tensor([[0.0557, 0.4073, 0.2954, 0.6370], 
                                   [0.4142, 0.5889, 0.5076, 0.7211]])
    mask_specific = engine.boxes_to_token_mask(specific_boxes)
    print(f"特定的boxes (xyxy): {specific_boxes}")
    print(f"Mask总和: {mask_specific.sum()}")
    visualize_mask(mask_specific, "特定的boxes")

def test_compute_uncertainty():
    """测试compute_uncertainty方法"""
    # 初始化不确定性引擎
    engine = UncertaintyEngine(grid_size=(16, 16), image_size=(224, 224))
    
    # 创建模拟的VLA注意力图
    num_heads = 8
    seq_len = 256
    vla_attn = torch.rand(num_heads, seq_len)
    
    # 测试1: 中心位置的box
    print("\n测试compute_uncertainty: 中心位置的box")
    box_center = torch.tensor([[0.35, 0.35, 0.65, 0.65]])  # [x1, y1, x2, y2]
    metrics = engine.compute_uncertainty(vla_attn, box_center)
    print(f"不确定性指标: {metrics}")
    
    # 测试2: 边缘位置的box
    print("\n测试compute_uncertainty: 边缘位置的box")
    box_edge = torch.tensor([[0.0, 0.0, 0.2, 0.2]])  # [x1, y1, x2, y2]
    metrics = engine.compute_uncertainty(vla_attn, box_edge)
    print(f"不确定性指标: {metrics}")
    
    # 测试3: 超出边界的box
    print("\n测试compute_uncertainty: 超出边界的box")
    box_out = torch.tensor([[1.0, 1.0, 1.3, 1.3]])  # [x1, y1, x2, y2]
    metrics = engine.compute_uncertainty(vla_attn, box_out)
    print(f"不确定性指标: {metrics}")
    
    # 测试4: 空box
    print("\n测试compute_uncertainty: 空box")
    box_empty = torch.tensor([])
    metrics = engine.compute_uncertainty(vla_attn, box_empty)
    print(f"不确定性指标: {metrics}")

def test_compute_attention_coverage():
    """测试compute_attention_coverage方法"""
    # 初始化不确定性引擎
    engine = UncertaintyEngine(grid_size=(16, 16), image_size=(224, 224))
    
    # 创建模拟的VLA注意力图
    num_heads = 8
    seq_len = 256
    vla_attn = torch.rand(num_heads, seq_len)
    
    # 测试1: 中心位置的box
    print("\n测试compute_attention_coverage: 中心位置的box")
    box_center = torch.tensor([[0.35, 0.35, 0.65, 0.65]])  # [x1, y1, x2, y2]
    metrics = engine.compute_attention_coverage(vla_attn, box_center)
    print(f"覆盖率指标: {metrics}")
    
    # 测试2: 边缘位置的box
    print("\n测试compute_attention_coverage: 边缘位置的box")
    box_edge = torch.tensor([[0.0, 0.0, 0.2, 0.2]])  # [x1, y1, x2, y2]
    metrics = engine.compute_attention_coverage(vla_attn, box_edge)
    print(f"覆盖率指标: {metrics}")
    
    # 测试3: 超出边界的box
    print("\n测试compute_attention_coverage: 超出边界的box")
    box_out = torch.tensor([[1.0, 1.0, 1.3, 1.3]])  # [x1, y1, x2, y2]
    metrics = engine.compute_attention_coverage(vla_attn, box_out)
    print(f"覆盖率指标: {metrics}")
    
    # 测试4: 空box
    print("\n测试compute_attention_coverage: 空box")
    box_empty = torch.tensor([])
    metrics = engine.compute_attention_coverage(vla_attn, box_empty)
    print(f"覆盖率指标: {metrics}")

if __name__ == "__main__":
    print("开始测试box_to_token_mask方法...")
    test_box_to_token_mask()
    
    print("\n开始测试compute_uncertainty方法...")
    test_compute_uncertainty()
    
    print("\n开始测试compute_attention_coverage方法...")
    test_compute_attention_coverage()