"""
Compute FLOPs for Grounding DINO inference (get_boxes)
"""
import torch
import numpy as np
import cv2
from deepspeed.profiling.flops_profiler import FlopsProfiler

# 保持路径一致
import sys
sys.path.append('/home/wzb/vla/GroundingDINO')
from groundingdino.util.inference import load_model, load_image, predict

def test_grounding_dino_flops():
    # 1. 配置与初始化
    config_path = '/home/wzb/vla/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint_path = '/home/wzb/vla/GroundingDINO/weights/groundingdino_swint_ogc.pth'
    
    # config_path = '/home/wzb/vla/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'
    # checkpoint_path = '/home/wzb/vla/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Grounding DINO to {device}...")
    model = load_model(config_path, checkpoint_path).to(device)
    model.eval()

    # 2. 模拟输入数据
    image = torch.randn(3, 800, 800).to("cuda")
    text_prompt = "blue plastic cup . coaster ."


    prof = FlopsProfiler(model)
    
    print("Starting DeepSpeed Profiler for Grounding DINO...")
    
    # 预热
    with torch.no_grad():
        # 注意：这里直接调用 forward，传参需匹配模型定义
        model(image[None], captions=[text_prompt])

    prof.start_profile()
    
    with torch.no_grad():
        # 使用相同逻辑执行推理
        model(image[None], captions=[text_prompt])
    
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    prof.end_profile()

    print("\n" + "="*50)
    print(f"Grounding DINO Profile Results")
    print(f"Input Resolution: {image.shape[1]}x{image.shape[2]}")
    print(f"Total Params: {params / 1e6:.2f} M")
    print(f"Total MACs:   {macs / 1e9:.2f} G")
    print(f"Total FLOPs:  {flops / 1e9:.2f} GFLOPs")
    print("="*50)

if __name__ == "__main__":
    try:
        test_grounding_dino_flops()
    except Exception as e:
        import traceback
        traceback.print_exc()