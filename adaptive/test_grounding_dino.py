#!/usr/bin/env python3
"""
测试Grounding DINO集成
"""

import numpy as np
import cv2
from adaptivepolicy import AdaptiveActionChunker

# 加载测试图像
def load_test_image():
    # 这里可以替换为实际的测试图像路径
    # 或者使用随机生成的图像进行测试
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = 255  # 白色背景
    
    # 绘制一些简单的形状作为测试目标
    cv2.rectangle(img, (200, 150), (300, 250), (0, 0, 0), -1)  # 黑色碗
    cv2.rectangle(img, (100, 150), (180, 230), (128, 128, 128), -1)  # 灰色盘子
    cv2.rectangle(img, (320, 170), (380, 220), (255, 165, 0), -1)  # 橙色小碟子
    
    return img

# 测试函数
def test_grounding_dino():
    print("测试Grounding DINO集成...")
    print("=" * 60)
    
    # 初始化AdaptiveActionChunker
    chunker = AdaptiveActionChunker()
    print("✅ AdaptiveActionChunker初始化成功")
    
    # 加载测试图像
    image = load_test_image()
    print("✅ 测试图像加载成功")
    
    # 测试指令
    instruction = "Pick up the black bowl between the plate and the ramekin and place it on the plate."
    print(f"测试指令: {instruction}")
    
    # 测试对象提取
    from object_extractor.extractor import extract_objects
    parsed_objs = extract_objects(instruction)
    print(f"✅ 对象提取结果: {parsed_objs}")
    
    # 测试get_dino_boxes方法
    print("\n测试Grounding DINO目标检测...")
    try:
        boxes = chunker.get_dino_boxes(image, parsed_objs)
        print(f"✅ 目标检测成功，检测到 {len(boxes)} 个目标")
        print(f"边界框: {boxes.tolist()}")
    except Exception as e:
        print(f"❌ 目标检测失败: {str(e)}")
    
    # 测试完整的ValueFunc
    print("\n测试完整的ValueFunc...")
    try:
        # 生成模拟的VLA注意力图
        import torch
        vla_attn = torch.rand(12, 512, 512)  # 模拟注意力图
        
        horizon, info = chunker.ValueFunc(image, instruction, vla_attn)
        print(f"✅ ValueFunc调用成功")
        print(f"推荐的horizon: {horizon}")
        print(f"详细信息: {info}")
    except Exception as e:
        print(f"❌ ValueFunc调用失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_grounding_dino()