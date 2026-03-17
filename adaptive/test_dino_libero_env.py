#!/usr/bin/env python3
"""
测试Grounding DINO对真实Libero环境的识别结果
使用真实环境的初始图像进行识别
"""

import os
import json
import numpy as np
import cv2
import gymnasium as gym
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import close_envs
from adaptivepolicy import AdaptiveActionChunker
from object_extractor.mock_extractor import extract_objects

# Grounding DINO visualization
from groundingdino.util.inference import annotate

# 测试配置
TEST_CONFIG = {
    "env_types": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    "max_episodes": 1,  # 每个环境只测试一次
    "output_dir": "dino_libero_results"
}

# 获取环境的初始图像
def get_env_initial_image(env):
    """
    获取环境的初始图像
    """
    # 重置环境
    observation, info = env.reset()
    
    # 提取图像
    if "pixels" in observation and isinstance(observation["pixels"], dict):
        # 通常Libero环境会有多个相机视角
        if "image" in observation["pixels"]:
            # 获取主相机图像
            image = observation["pixels"]["image"]
            # 转换形状 (H, W, C)
            if len(image.shape) == 4:
                image = image[0]  # 移除batch维度
            return image
    
    # 如果没有找到图像，返回None
    return None

# 测试单个Libero环境
def test_single_libero_env(env_type):
    """
    测试单个Libero环境的Grounding DINO识别
    """
    print(f"测试环境: {env_type}")
    print("-" * 60)
    
    # 创建环境配置
    env_cfg = LiberoEnv(task=env_type)
    
    # 创建环境
    try:
        all_envs_dict = make_env(env_cfg, n_envs=1)
        print(f"✅ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {str(e)}")
        return []
    
    # 存储结果
    results = []
    
    # 遍历所有任务
    for suite_name, tasks in all_envs_dict.items():
        print(f"处理任务集: {suite_name}")
        
        for task_id, env in tasks.items():
            print(f"  处理任务ID: {task_id}")
            
            try:
                # 获取任务描述
                try:
                    task_description = env.get_attr("task_description")[0]
                    print(f"    任务描述: {task_description}")
                except Exception as e:
                    print(f"    ⚠️  无法获取任务描述: {str(e)}")
                    # 使用环境类型作为默认任务描述
                    task_description = env_type
                    print(f"    使用默认任务描述: {task_description}")
                
                # 获取初始图像
                image = get_env_initial_image(env)
                if image is None:
                    print(f"    ❌ 无法获取初始图像")
                    continue
                
                print(f"    ✅ 初始图像获取成功，形状: {image.shape}")
                
                # 使用mock extractor提取对象
                extracted_objects = extract_objects(task_description)
                print(f"    提取的对象: {extracted_objects}")
                
                # 使用Grounding DINO识别
                chunker = AdaptiveActionChunker()
                boxes_cxcywh, boxes_xyxy, logits, phrases, image_source = chunker.get_dino_boxes(image, extracted_objects)
                print(f"    识别的边界框: {boxes_cxcywh.tolist()}")
                print(f"    识别的短语: {phrases}")
                print(f"    置信度: {logits.tolist() if len(logits) > 0 else []}")
                
                # 保存原图
                image_output_dir = os.path.join(TEST_CONFIG["output_dir"], env_type, "images")
                os.makedirs(image_output_dir, exist_ok=True)
                
                # 生成文件名
                original_image_filename = f"{suite_name}_{task_id}_original.png"
                original_image_path = os.path.join(image_output_dir, original_image_filename)
                
                # 保存原图
                cv2.imwrite(original_image_path, image_source)
                print(f"    ✅ 原图保存到: {original_image_path}")
                
                # 绘制识别结果
                if len(boxes_xyxy) > 0:
                    # 绘制边界框和标签
                    annotated_image = annotate(image_source, boxes_xyxy, logits, phrases)
                    
                    # 生成带框图像文件名
                    annotated_image_filename = f"{suite_name}_{task_id}_annotated.png"
                    annotated_image_path = os.path.join(image_output_dir, annotated_image_filename)
                    
                    # 保存带框图像
                    cv2.imwrite(annotated_image_path, annotated_image)
                    print(f"    ✅ 识别结果图像保存到: {annotated_image_path}")
                
                # 存储结果
                result = {
                    "env_type": env_type,
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "extracted_objects": extracted_objects,
                    "dino_boxes": boxes_cxcywh.tolist(),
                    "box_count": len(boxes_cxcywh),
                    "phrases": phrases,
                    "logits": logits.tolist() if len(logits) > 0 else []
                }
                results.append(result)
                
                print(f"    ✅ 处理成功")
                
            except Exception as e:
                print(f"    ❌ 处理失败: {str(e)}")
                # 存储失败结果
                result = {
                    "env_type": env_type,
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "task_description": "",
                    "extracted_objects": [],
                    "dino_boxes": [],
                    "box_count": 0,
                    "error": str(e)
                }
                results.append(result)
    
    # 关闭环境
    close_envs(all_envs_dict)
    
    return results

# 主测试函数
def main():
    print("测试Grounding DINO对真实Libero环境的识别...")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)
    
    # 测试所有环境类型
    all_results = {}
    
    for env_type in TEST_CONFIG["env_types"]:
        print(f"\n测试环境类型: {env_type}")
        print("=" * 60)
        
        # 测试该环境类型
        results = test_single_libero_env(env_type)
        all_results[env_type] = results
        
        # 保存该环境类型的结果
        env_output_dir = os.path.join(TEST_CONFIG["output_dir"], env_type)
        os.makedirs(env_output_dir, exist_ok=True)
        
        with open(os.path.join(env_output_dir, "results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 环境 {env_type} 测试完成，保存结果到 {env_output_dir}")
    
    # 保存所有结果
    with open(os.path.join(TEST_CONFIG["output_dir"], "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print(f"所有结果保存到: {TEST_CONFIG['output_dir']}")
    print("=" * 80)

if __name__ == "__main__":
    main()