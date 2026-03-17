import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# 排成一行
def save_images(images, output_dir='visualizations', filename='combined_images.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
    
    for i, img in enumerate(images):
        # 确保 img 是 numpy 数组且在 0-255 范围
        if isinstance(img, Image.Image):
            img = np.array(img)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path)

def heatmap_videos(images, attn_maps,
                   filename='visualizations/heatmap_video.mp4',
                   img_start_indices=[0, 256],
                   fps=8):
    """
    images: list of images [img1, img2, ...]
    attn_maps: list over samples -> list over heads -> tensor [seq_len]
    filename: 输出视频路径
    img_start_indices: 每张图的 token 起始 index
    fps: 帧率
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    num_samples = len(attn_maps)
    num_images = len(images)
    
    print(num_samples, num_images)

    frames = []

    for sample_idx in range(num_samples):
        # ===== 1. 对当前 sample 的所有 head 求和 =====
        num_heads = len(attn_maps[sample_idx][0])
        seq_len = attn_maps[sample_idx][0][0].shape[0]

        sum_attn = torch.zeros(seq_len, dtype=torch.float32)
        for head_idx in range(num_heads):
            sum_attn += attn_maps[sample_idx][0][head_idx].to(torch.float32)

        full_attn = sum_attn.numpy()

        sample_images = []

        # ===== 2. 对每张图生成 heatmap overlay =====
        for i, img in enumerate(images[sample_idx]):

            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img.copy()

            if img_np.max() <= 1.00001:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            h, w = img_np.shape[:2]

            start = img_start_indices[i]
            attn_weights = full_attn[start:start+256]

            attn_matrix = attn_weights.reshape(16, 16)

            # 归一化
            attn_matrix = (attn_matrix - attn_matrix.min()) / \
                          (attn_matrix.max() - attn_matrix.min() + 1e-8)

            heatmap = cv2.resize(attn_matrix, (w, h), interpolation=cv2.INTER_CUBIC)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

            # 写 sample index
            cv2.putText(
                overlay,
                f"Sample {sample_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            sample_images.append(overlay)

        # ===== 3. 横向拼接所有图片 =====
        combined_frame = np.concatenate(sample_images, axis=1)
        frames.append(combined_frame)

    # ===== 4. 写视频 =====
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

    print(f"Video saved to {filename}")
    
def original_videos(images,
                    filename='visualizations/original_video.mp4',
                    fps=8):
    """
    images: list over samples -> list of images [img1, img2, ...]
    filename: 输出视频路径
    fps: 帧率
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    num_samples = len(images)

    frames = []

    for sample_idx in range(num_samples):

        sample_images = []

        for img in images[sample_idx]:

            # 转 numpy
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img.copy()

            # 保证 uint8
            if img_np.max() <= 1.00001:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # 写 sample index
            cv2.putText(
                img_np,
                f"Sample {sample_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            sample_images.append(img_np)

        # 横向拼接
        combined_frame = np.concatenate(sample_images, axis=1)
        frames.append(combined_frame)

    # 写视频
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

    print(f"Video saved to {filename}")


def visualize_vla_attention(images, attn_maps, layer_idx=0, head_idx=0, img_start_indices=[0, 256], output_dir='visualizations'):
    """
    images: 原始图像列表 [img1, img2], 每张图为 (256, 256, 3)
    attn_maps: get_attn_maps 返回的 list
    layer_idx: 想要查看的层索引 (在传入的 layers 列表中的相对位置)
    head_idx: 想要查看的头索引 (在传入的 heads 列表中的相对位置)
    img_start_indices: 图像 token 在序列中的起始位置 (根据你的模型实际情况调整)
    """
    # 文件保存路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 提取特定层和头的注意力向量
    # 形状为 (total_seq_len,)
    # full_attn = attn_maps[layer_idx][head_idx].numpy()
    full_attn = attn_maps[layer_idx][head_idx].to(torch.float32).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, img in enumerate(images):
        # 2. 截取对应图像的 256 个 token
        start = img_start_indices[i]
        end = start + 256
        attn_weights = full_attn[start:end]
        
        # 3. Reshape 到 16x16 (假设 256 = 16x16)
        attn_matrix = attn_weights.reshape(16, 16)
        
        # 4. 归一化处理以便显示
        attn_matrix = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min() + 1e-8)
        
        # 5. 将 16x16 插值放大到 256x256
        heatmap = cv2.resize(attn_matrix, (256, 256), interpolation=cv2.INTER_CUBIC)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 6. 叠加到原图
        # 确保 img 是 numpy 数组且在 0-255 范围
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        axes[i].imshow(overlay)
        axes[i].set_title(f"Image {i+1} - Layer {layer_idx} Head {head_idx}")
        axes[i].axis('off')

    fig_path = os.path.join(output_dir, f'vla_attention_layer{layer_idx}_head{head_idx}.png')
    plt.savefig(fig_path)

# # 使用示例:
# # 假设你的序列中，Language 占了 64 个 token
# # img_start_indices = [64, 64 + 256] 
# # visualize_vla_attention([raw_img1, raw_img2], attn_results)

def visualize_vla_attention_dino(images, attn_maps, layer_idx=0, head_idx=0, 
                                 img_start_indices=[0, 256], threshold=0.6, output_dir='visualizations'):
    """
    images: 原始图像列表 [img1, img2]
    attn_maps: 提取的注意力 tensor
    threshold: 质量阈值 (0-1)，保留前 X% 的权重质量。降低此值会让热力图更集中。
    """
    # 1. 转换精度并提取特定的 Head
    full_attn = attn_maps[layer_idx][head_idx].to(torch.float32) # [seq_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    for i, img in enumerate(images):
        # --- 图像预处理 ---
        if img.max() <= 1.00001: img = (img * 255).astype(np.uint8)
        else: img = img.astype(np.uint8)
        h, w = img.shape[:2]

        # --- 2. 核心逻辑：DINO 式质量阈值处理 ---
        start = img_start_indices[i]
        attn = full_attn[start : start + 256] # 提取 256 个图像 token

        # 排序并计算累积分布
        val, idx = torch.sort(attn)
        val /= torch.sum(val)
        cumval = torch.cumsum(val, dim=0)
        
        # 仅保留权重最大的前 (threshold * 100)% 的位置
        th_attn = cumval > (1 - threshold)
        
        # 将结果映射回原始索引顺序
        idx2 = torch.argsort(idx)
        mask = th_attn[idx2].reshape(16, 16).float().numpy()
        
        # 同时保留一个原始权重的平滑版本用于热力图
        raw_attn = attn.reshape(16, 16).numpy()

        # --- 3. 尺寸缩放与平滑 (DINO 风格插值) ---
        # 放大掩码和原始权重
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        attn_resized = cv2.resize(raw_attn, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # --- 4. 生成叠加效果 ---
        # 归一化注意力图
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        heatmap = np.uint8(255 * attn_norm)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 结合 Mask：只显示被阈值选中的区域
        # 这样可以强行去掉那些“空地”上的背景噪音
        overlay = img.copy()
        mask_3d = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
        
        # 在 Mask 选中的区域叠加颜色，未选中区域保持原图
        active_area = (mask_3d > 0.5)
        overlay[active_area] = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)[active_area]
        
        # 绘制轮廓线 (类似 DINO)
        contours, _ = cv2.findContours(np.uint8(mask_resized > 0.5), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        axes[i].imshow(overlay)
        axes[i].set_title(f"Image {i} | DINO-Threshold: {threshold}")
        axes[i].axis('off')
    
    output_path = f"{output_dir}/vla_dino_attention_layer{layer_idx}_head{head_idx}_th{int(threshold*100)}.png"
    plt.savefig(output_path)



def vla_attention_specs(images, attn_maps, layer_idx=0, head_idx=0, 
                                 img_start_indices=[0, 256], threshold=0.6, output_dir='visualizations'):
    """
    images: 原始图像列表 [img1, img2]
    attn_maps: 提取的注意力 tensor
    threshold: 质量阈值 (0-1)，保留前 X% 的权重质量。降低此值会让热力图更集中。
    """
    # 1. 转换精度并提取特定的 Head
    full_attn = attn_maps[layer_idx][head_idx].to(torch.float32) # [seq_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    lan_attn = full_attn[512:-1]  # 假设语言 token 在后面
    print(f"Layer {layer_idx} Head {head_idx} Language Attention Sum: {lan_attn.sum().item():.4f}, Max: {lan_attn.max().item():.4f}, Min: {lan_attn.min().item():.4f}")
    
    self_attn = full_attn[-1]  # 自注意力 token
    print(f"Layer {layer_idx} Head {head_idx} Self Attention Value: {self_attn.item():.4f}")

    # sum = full_attn.sum().item()
    # print(f"Layer {layer_idx} Head {head_idx} Total Attention Sum: {sum:.4f}")

    for i, img in enumerate(images):
        # --- 图像预处理 ---
        if img.max() <= 1.00001: img = (img * 255).astype(np.uint8)
        else: img = img.astype(np.uint8)
        h, w = img.shape[:2]

        # --- 2. 核心逻辑：DINO 式质量阈值处理 ---
        start = img_start_indices[i]
        attn = full_attn[start : start + 256] # 提取 256 个图像 token
        # 看一下attn的最大值和最小值
        print(f"Layer {layer_idx} Head {head_idx} Image {i} Attention Sum: {attn.sum().item():.4f}, Max: {attn.max().item():.4f}, Min: {attn.min().item():.4f}")
        
        
        
def visualize_vla_attention_sum_heads_per_layer(images, attn_maps, img_start_indices=[0, 256], output_dir='visualizations'):
    """
    新函数：对每层的所有head的注意力向量求和，然后可视化热图。
    attn_maps: list of layers, each layer is list of heads, each head is tensor [seq_len]
    对于每个层，聚合所有head的注意力（sum over heads），得到 [seq_len]，然后可视化类似于 visualize_vla_attention。
    会为每个层生成一个独立的热图文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_layers = len(attn_maps)
    
    for layer_idx in range(num_layers):
        # 假设每个层有相同的 num_heads
        num_heads = len(attn_maps[layer_idx])
        
        # 初始化 sum_attn 为零向量 [seq_len]
        seq_len = attn_maps[layer_idx][0].shape[0]
        sum_attn = torch.zeros(seq_len, dtype=torch.float32)
        
        # 对所有 head 求和
        for head_idx in range(num_heads):
            sum_attn += attn_maps[layer_idx][head_idx].to(torch.float32)
        
        # 转换为 numpy
        full_attn = sum_attn.numpy()
        
        # 创建 figure
        fig, axes = plt.subplots(1, len(images), figsize=(12, 6))
        
        for i, img in enumerate(images):
            # 截取对应图像的 256 个 token
            start = img_start_indices[i]
            end = start + 256
            attn_weights = full_attn[start:end]
            
            # Reshape 到 16x16
            attn_matrix = attn_weights.reshape(16, 16)
            
            # 归一化
            attn_matrix = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min() + 1e-8)
            
            # 放大到 256x256
            heatmap = cv2.resize(attn_matrix, (256, 256), interpolation=cv2.INTER_CUBIC)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
            # 叠加到原图
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            
            axes[i].imshow(overlay)
            axes[i].set_title(f"Image {i+1} - Layer {layer_idx} Sum of All Heads")
            axes[i].axis('off')
        
        # 保存
        fig_path = os.path.join(output_dir, f'vla_attention_layer{layer_idx}_sum_all_heads.png')
        plt.savefig(fig_path)
        plt.close(fig)  # 关闭 figure 以节省内存