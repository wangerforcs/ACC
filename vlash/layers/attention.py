#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attention Layer with KV Cache.

This module provides a minimal scaled dot-product attention implementation
with optional KV caching for efficient autoregressive inference.

The KV cache strategy:
- First call with use_cache=True: Initialize cache with K/V (prefix prefill)
- Subsequent calls: Concatenate cached prefix K/V with new suffix K/V

"""

# TODO: use flashinfer kernels

import torch
from torch import nn


class Attention(nn.Module):
    """Scaled dot-product attention with optional KV cache.
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
    
    The KV cache enables efficient inference by storing prefix K/V
    and reusing them across multiple forward passes.
    """

    def __init__(self, scale: float):
        """Initialize attention layer.
        
        Args:
            scale: Scaling factor for attention scores, typically 1/sqrt(head_dim).
        """
        super().__init__()
        self.scale = scale
        
        # KV cache buffers: [B, H, L_prefix, D]
        self.k_cache = None
        self.v_cache = None
        self.attn_weights = None
        self.entropys = {}
        self.layer_entropys = {}

    def reset_cache(self):
        """Clear KV cache. Call when starting a new sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [B, H, L_q, D].
            k: Key tensor [B, H, L_k_new, D].
            v: Value tensor [B, H, L_v_new, D].
            attention_mask: Additive mask [B, 1, L_q, L_k] or [B, H, L_q, L_k].
                           Use 0 for positions to attend, -inf for masked positions.
            use_cache: If True, use KV caching for prefix tokens.
            
        Returns:
            Output tensor [B, H, L_q, D].
        """
        # Handle KV cache
        if use_cache:
            if self.k_cache is None:
                # First call: initialize cache with prefix K/V
                self.k_cache = k.detach()
                self.v_cache = v.detach()
                k_full = k
                v_full = v
            else:
                # Subsequent calls: concatenate cached prefix with new suffix
                # Note: cache is not updated, always stores prefix only
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Compute attention scores: [B, H, L_q, L_k]
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Apply attention mask (additive)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = attention_mask
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError(f"Unsupported attention_mask ndim: {attention_mask.ndim}")
            attn_scores = attn_scores + mask

        # Softmax and weighted sum
        attn_weights = torch.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_weights #[B, head, chunksize, dim]
        
        # if attn_weights.shape[2] < 512:
        #     with torch.no_grad():
        #         eps = 1e-8
        #         for headi in range(attn_weights.shape[1]):
        #             if headi not in self.entropys:
        #                 self.entropys[headi] = []
        #             # 我觉得应该用图像的熵而不是整体的
        #             entropy = -torch.sum(attn_weights[0][headi][0] * torch.log2(attn_weights[0][headi][0] + eps), dim=-1).mean().item()
        #             self.entropys[headi].append(entropy)
        
        # if attn_weights.shape[2] < 512:   # 确保有两张图的 token
        #     with torch.no_grad():
        #         eps = 1e-8
        #         if "img1" not in self.layer_entropys:
        #             self.layer_entropys["img1"] = []
        #         if "img2" not in self.layer_entropys:
        #             self.layer_entropys["img2"] = []
        #         target_attn = attn_weights[0, :, 0, :]
        #         target_head_mean = target_attn.mean(dim=0, keepdim=False)
        #         img1_attn = target_head_mean[0:256]
        #         img2_attn = target_head_mean[256:512]
        #         img1_attn = img1_attn / (img1_attn.sum() + eps)  # 重新归一化
        #         entropy1 = -torch.sum(
        #             img1_attn * torch.log2(img1_attn + eps)
        #         ).item()
        #         img2_attn = img2_attn / (img2_attn.sum() + eps)  # 重新归一化
        #         entropy2 = -torch.sum(
        #             img2_attn * torch.log2(img2_attn + eps)
        #         ).item()
        #         self.layer_entropys["img1"].append(entropy1)
        #         self.layer_entropys["img2"].append(entropy2)
                
        #         for headi in range(attn_weights.shape[1]):
        #             if headi not in self.entropys:
        #                 self.entropys[headi] = {
        #                     "img1": [],
        #                     "img2": []
        #                 }
        #             # 取第一个 batch，第一个 query token
        #             attn_vec = attn_weights[0, headi, 0]  # [L_k]
        #             # ===== 图1 =====
        #             img1_attn = attn_vec[0:256]
        #             img1_attn = img1_attn / (img1_attn.sum() + eps)  # 重新归一化
        #             entropy1 = -torch.sum(
        #                 img1_attn * torch.log2(img1_attn + eps)
        #             ).item()
        #             # ===== 图2 =====
        #             img2_attn = attn_vec[256:512]
        #             img2_attn = img2_attn / (img2_attn.sum() + eps)
        #             entropy2 = -torch.sum(
        #                 img2_attn * torch.log2(img2_attn + eps)
        #             ).item()
        #             self.entropys[headi]["img1"].append(entropy1)
        #             self.entropys[headi]["img2"].append(entropy2)

        out = torch.matmul(attn_weights, v_full)
        
        return out
    
    
    
    
    # def forward(
    #     self,
    #     q: torch.Tensor,
    #     k: torch.Tensor,
    #     v: torch.Tensor,
    #     attention_mask: torch.Tensor | None = None,
    #     use_cache: bool = False,
    # ) -> torch.Tensor:
    #     # ... (KV cache 处理部分保持不变) ...
    #     if use_cache:
    #         if self.k_cache is None:
    #             self.k_cache, self.v_cache = k.detach(), v.detach()
    #             k_full, v_full = k, v
    #         else:
    #             k_full = torch.cat([self.k_cache, k], dim=2)
    #             v_full = torch.cat([self.v_cache, v], dim=2)
    #     else:
    #         k_full, v_full = k, v

    #     # 计算得分
    #     attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

    #     # ================== 验证逻辑开始 ==================
    #     if attention_mask is not None:
    #         # 统一 Mask 维度到 [B, 1, L_q, L_k]
    #         mask = attention_mask if attention_mask.dim() == 4 else attention_mask[:, None, :, :]
            
    #         # 打印验证信息 (仅在调试阶段使用)
    #         if not self.training and q.shape[2] > 1: # 当一次性传入多个 Query (比如 Suffix 阶段)
    #             import matplotlib.pyplot as plt
    #             plt.imshow(mask[0, 0].to(torch.float32).cpu().numpy() < -1) # 黑色代表被遮挡，白色代表可见
    #             plt.savefig(f"debug_attention_mask{mask.shape[2]}.png")
            
    #         attn_scores = attn_scores + mask
    #     # ================== 验证逻辑结束 ==================

    #     attn_weights = torch.softmax(attn_scores, dim=-1)
    #     self.attn_weights = attn_weights

    #     # 验证验证 Action 是否看到了 Prefix (图像+文本)
    #     # 如果 L_q 是 Suffix 长度，L_k 是 Full 长度
    #     if not self.training and use_cache and self.k_cache is not None:
    #         prefix_len = self.k_cache.shape[2]
    #         # 检查 Action Token 对 Prefix 的注意力权重均值
    #         # 正常情况下应该大于 0
    #         attn_to_prefix = attn_weights[0, :, :, :prefix_len].mean().item()
    #         # print(f"DEBUG: Action to Prefix attention mean: {attn_to_prefix:.4f}")

    #     out = torch.matmul(attn_weights, v_full)
    #     return out