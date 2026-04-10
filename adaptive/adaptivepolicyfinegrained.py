import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import List, Tuple, Dict, Optional
from .object_extractor.mock_extractor import extract_object_roles, extract_objects
# from object_extractor.mock_extractor import extract_objects

# Grounding DINO imports
import sys
sys.path.append('/pub/data/wangzb/GroundingDINO')
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2


def _instruction_role_info(instruction: str) -> Dict[str, List[str]]:
    role_info = extract_object_roles(instruction)
    return {
        "target_object": list(role_info.get("all_objects", [])),
        "tograsp_objects": list(role_info.get("tograsp", [])),
        "reference_objects": list(role_info.get("reference", [])),
    }

# ==============================================================================
# 2. Vision & Math Module: Uncertainty Calculation (Value Function)
# ==============================================================================
class UncertaintyEngine:
    def __init__(self, grid_size=(16, 16), image_size=(224, 224)):
        self.gh, self.gw = grid_size
        self.H, self.W = image_size
        self.num_patches = self.gh * self.gw
    
    def boxes_to_token_mask(self, boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
        """
        将多个归一化的Bounding Box转换为合并的Patch Mask
        Output: [num_patches], 1.0 inside any box, 0.0 outside all boxes
        """
        mask = torch.zeros(self.num_patches)
        patch_h = 1.0 / self.gh
        patch_w = 1.0 / self.gw
        
        if boxes_xyxy_norm is None or len(boxes_xyxy_norm) == 0:
            return mask

        # 遍历所有box
        for i in range(boxes_xyxy_norm.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy_norm[i]
            
            # 确保值在有效范围内
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            # 遍历所有patch计算是否在box内
            for j in range(self.num_patches):
                row = j // self.gw
                col = j % self.gw
                
                # Patch center
                px = (col + 0.5) * patch_w
                py = (row + 0.5) * patch_h
                
                if x1 <= px <= x2 and y1 <= py <= y2:
                    mask[j] = 1.0
        
        # 不需要归一化，按需归一化
        return mask

    def calculate_kl_divergence(self, attn_map: torch.Tensor, target_mask: torch.Tensor) -> float:
        """
        计算 VLA Attention 分布与 Grounding DINO 目标分布的 KL 散度
        Metric: KL(P || Q)
        P: VLA Attention (Model Prediction)
        Q: Ideal Distribution (Grounding DINO Mask)
        """
        # 1. 构造 Q (Ideal Distribution)
        # 加上 epsilon 防止除零，并归一化
        epsilon = 1e-9
        Q = target_mask + epsilon
        Q = Q / Q.sum()

        # 2. 处理 P (Attention Map)
        # attn_map shape 应该是 [num_patches]
        P = attn_map + epsilon
        P = P / P.sum()

        # 3. 计算 KL = sum(P * log(P / Q))
        # 如果 P 很确信地指向了 Q 之外的地方，KL 会很大
        kl_div = F.kl_div(P.log(), Q, reduction='mean').item() #batchmean是每个样本的特征和，mean是每个特征
        
        return kl_div

    def compute_uncertainty(self, 
                          vla_attn_layer: torch.Tensor, 
                          dino_boxes: torch.Tensor) -> Dict[str, float]:
        """
        Input:
            vla_attn_layer: [num_heads, seq_len] (Raw attention from VLA)
            dino_boxes: [N, 4] (Normalized xyxy from Grounding DINO)
        """
        # 1. 提取 Action Token 对 Image Tokens 的 Attention
        # 这里的索引需要根据 PI0.5 的具体 tokenizer 和 embedding 结构调整
        img_start, img_end = 0, self.num_patches
        action_token_idx = -1         
        # 平均所有 head 的注意力 (或者选取特定的 Expert Head)
        attn_avg = vla_attn_layer.mean(dim=0, keepdim=False)[img_start:img_end]
        attn_prob = attn_avg / attn_avg.sum() if attn_avg.sum() > 0 else attn_avg
        
        # 2. 生成目标掩码
        target_mask = self.boxes_to_token_mask(dino_boxes)
        
        masked = attn_prob * target_mask
        mib = masked.sum().item()
        
        max_val_global = attn_prob.max().item()
        max_val_in_target = masked.max().item()
        peak_ratio = max_val_in_target / (max_val_global + 1e-9)
        
        # 如果 DINO 没检测到东西，不确定性直接拉满
        if target_mask.sum() == 0:
            logging.info(f"target_mask sum is 0, attn_avg max is {attn_avg.max().item()}")
            return {"kl": 10.0, "coverage": 0.0, "peak_ratio": 0.0, "uncertainty": 0.0}
        target_mask = target_mask / target_mask.sum() 
        
        uniform_mask = torch.ones(self.num_patches) / self.num_patches

        # 3. 计算指标
        kl = self.calculate_kl_divergence(attn_avg, target_mask)
        
        # 小于0更好
        uncertainty = self.calculate_kl_divergence(attn_avg, uniform_mask) - kl
        
        # 辅助指标：Attention Overlap (有多少注意力落在了框内)
        overlap = (attn_avg * target_mask).sum().item() / attn_avg.sum().item()
        
        return {"kl": kl, "coverage": mib, "peak_ratio": peak_ratio, "uncertainty": uncertainty}

# ==============================================================================
# 3. Strategy Module: Adaptive Horizon Controller
# ==============================================================================
class AdaptiveActionChunker:
    def __init__(self, 
                 min_horizon=5, 
                 max_horizon=30, 
                 uncertainty_threshold=0.01,
                 coverage_threshold=0.1,
                 continuity_threshold=0.2,
                 action_history_window=5):
        
        self.uncertainty_engine = UncertaintyEngine()
        
        self.min_h = min_horizon
        self.max_h = max_horizon
        self.threshold = uncertainty_threshold
        self.coverage_threshold = coverage_threshold
        self.action_history_window = action_history_window
        self.continuity_threshold = continuity_threshold 
        # 状态记录
        self.chunk_levels = 5
        self.current_horizon = self.get_chunk_size(self.chunk_levels // 2)
        self.history = []
        # 动作历史记录，用于计算轨迹连续性
        self.action_history = []
        self.prev_continuity = None
        # 打断标志，用于指示是否需要立即打断当前chunk
        self.needs_interruption = False
        
        # Grounding DINO model initialization
        self.config_path = '/pub/data/wangzb/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'
        # self.checkpoint_path = '/pub/data/wangzb/GroundingDINO/weights/20260314_pool_8~2_best.pth'
        self.checkpoint_path = '/pub/data/wangzb/GroundingDINO/weights/20260318_train2-3_best.pth'
        # self.checkpoint_path = '/pub/data/wangzb/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model
        self.model = load_model(self.config_path, self.checkpoint_path, device=self.device)
        self.cached_boxes = None
        self.query_box_times = 0

    def apply_chunk_proposal(self, proposal: Optional[Dict[str, object]]) -> Tuple[int, Dict[str, object]]:
        if not proposal:
            return self.current_horizon, {}
        horizon = int(proposal.get("current_horizon", self.current_horizon))
        self.current_horizon = horizon
        self.needs_interruption = bool(proposal.get("needs_interruption", False))
        return self.current_horizon, dict(proposal.get("info") or {})
        
    def get_runtime_config(self):
        return {
            "policy_type": "finegrained_base",
            "min_horizon": self.min_h,
            "max_horizon": self.max_h,
            "uncertainty_threshold": self.threshold,
            "coverage_threshold": self.coverage_threshold,
            "continuity_threshold": self.continuity_threshold,
            "action_history_window": self.action_history_window,
            "chunk_levels": self.chunk_levels,
            "grid_size": [
                self.uncertainty_engine.gh,
                self.uncertainty_engine.gw,
            ],
            "image_size": [
                self.uncertainty_engine.H,
                self.uncertainty_engine.W,
            ],
            "grounding_dino": {
                "config_path": self.config_path,
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
            },
        }
        
    def reset(self):
        """重置状态"""
        self.current_horizon = self.get_chunk_size(self.chunk_levels // 2)
        self.history = []
        self.action_history = []
        self.prev_continuity = None
        self.needs_interruption = False

    def get_dino_boxes_all(self, image, text_prompt):
        # if self.cached_boxes is not None and self.query_box_times % 10 == 0:
            # return self.cached_boxes
        self.query_box_times += 1
        """
        调用 Grounding DINO 模型进行目标检测（保留所有结果）
        Input:
            image: 图像数据 (numpy array 或路径)
            text_prompt: 文本提示，可以是字符串或对象列表
        Output:
            tuple: (boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
                boxes_cxcywh: torch.Tensor 归一化的边界框坐标 [N, 4] (cxcywh)
                boxes_xyxy: torch.Tensor 原始边界框坐标 [N, 4] (xyxy)
                logits: torch.Tensor 置信度 [N]
                phrases: list 识别的短语 [N]
                image_source: numpy array 原始图像
        """
        # 处理文本提示
        if isinstance(text_prompt, list):
            # 如果是对象列表，将其转换为逗号分隔的字符串
            prompt = " . ".join(text_prompt)
        else:
            prompt = text_prompt
        
        # 处理图像输入
        if isinstance(image, np.ndarray):
            # 如果是numpy数组，保存为临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                image_path = tmp.name
        else:
            # 假设是图像路径
            image_path = image
        
        try:
            # 加载图像
            image_source, image = load_image(image_path)
            
            # 设置预测参数
            box_threshold = 0.3
            text_threshold = 0.25
            
            # 调用模型进行预测
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            # 将边界框转换为归一化的 cxcywh 格式
            if len(boxes) > 0:
                # 模型返回的是 [cx, cy, w, h] 格式
                # 需要转换为 [x1, y1, x2, y2] 格式用于可视化
                h, w = image_source.shape[:2]
                
                # 确保boxes在有效范围内
                boxes = torch.clamp(boxes, 0.0, 1.0)
                
                # 确保宽度和高度为正值
                boxes[:, 2] = torch.clamp(boxes[:, 2], min=0.01)
                boxes[:, 3] = torch.clamp(boxes[:, 3], min=0.01)
                
                # 提取中心点、宽度和高度
                cx = boxes[:, 0]
                cy = boxes[:, 1]
                width = boxes[:, 2]
                height = boxes[:, 3]
                
                # 转换为 [x1, y1, x2, y2] 格式用于可视化
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2
                
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                boxes_cxcywh = boxes
            else:
                # 如果没有检测到目标，返回空张量
                boxes_cxcywh = torch.tensor([])
                boxes_xyxy = torch.tensor([])
                
        finally:
            # 清理临时文件
            if 'image_path' in locals() and isinstance(image, np.ndarray):
                import os
                if os.path.exists(image_path):
                    os.unlink(image_path)
        # self.cached_boxes = (boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
        return boxes_cxcywh, boxes_xyxy, logits, phrases, image_source

    def get_dino_boxes(self, image, text_prompt):
        """
        调用 Grounding DINO 模型进行目标检测（去重，保留每个phrase的最高概率结果）
        Input:
            image: 图像数据 (numpy array 或路径)
            text_prompt: 文本提示，可以是字符串或对象列表
        Output:
            tuple: (boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
                boxes_cxcywh: torch.Tensor 归一化的边界框坐标 [N, 4] (cxcywh)
                boxes_xyxy: torch.Tensor 原始边界框坐标 [N, 4] (xyxy)
                logits: torch.Tensor 置信度 [N]
                phrases: list 识别的短语 [N]
                image_source: numpy array 原始图像
        """
        # 调用保留所有结果的方法
        boxes_cxcywh_all, boxes_all, logits_all, phrases_all, image_source = self.get_dino_boxes_all(image, text_prompt)
        
        # 去重：对于相同phrase的结果，只保留概率最高的那个
        if len(boxes_all) > 0:
            # 创建字典存储每个phrase的最高概率结果
            phrase_dict = {}
            
            for i, phrase in enumerate(phrases_all):
                if phrase not in phrase_dict or logits_all[i] > phrase_dict[phrase]['logit']:
                    phrase_dict[phrase] = {
                        'box': boxes_all[i],
                        'logit': logits_all[i],
                        'phrase': phrase,
                        'box_cxcywh': boxes_cxcywh_all[i]
                    }
            
            # 重新构建结果列表
            unique_boxes = []
            unique_logits = []
            unique_phrases = []
            unique_boxes_cxcywh = []
            
            for item in phrase_dict.values():
                unique_boxes.append(item['box'])
                unique_logits.append(item['logit'])
                unique_phrases.append(item['phrase'])
                unique_boxes_cxcywh.append(item['box_cxcywh'])
            
            # 转换为张量
            boxes = torch.stack(unique_boxes) if unique_boxes else torch.tensor([])
            logits = torch.tensor(unique_logits) if unique_logits else torch.tensor([])
            phrases = unique_phrases
            boxes_cxcywh = torch.stack(unique_boxes_cxcywh) if unique_boxes_cxcywh else torch.tensor([])
            
            # 计算boxes_xyxy
            if len(boxes) > 0:
                h, w = image_source.shape[:2]
                boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
            else:
                boxes_xyxy = torch.tensor([])
        else:
            # 如果没有检测到目标，返回空张量
            boxes_cxcywh = torch.tensor([])
            boxes = torch.tensor([])
            logits = torch.tensor([])
            phrases = []
            boxes_xyxy = torch.tensor([])
        
        return boxes_cxcywh, boxes, logits, phrases, image_source

    def enqueue_history(self, history_action):
        """将历史记录添加到队列中"""
        if history_action is not None:
            action = np.asarray(history_action, dtype=np.float32)
            if action.ndim == 0:
                return
            action = action.reshape(-1)
            if action.size == 0:
                return
            self.action_history.append(action.tolist())
    
    def compute_continuity(self):
        continuity_info = self.compute_continuity_info()
        return continuity_info["label"]

    def compute_continuity_info(self):
        continuity_info = {
            "label": "Unknown continuity",
            "level": "unknown",
            "score": None,
            "delta": None,
            "gripper_changed": False,
            "ready": False,
        }
        if len(self.action_history) > self.action_history_window:
            self.action_history = self.action_history[-self.action_history_window:]
        continuity = 0.0
        # 计算轨迹连续性：时间窗口内相邻动作的绝对值差之和
        if len(self.action_history) == self.action_history_window:
            gripper_changed = False
            for i in range(1, len(self.action_history)):
                prev_action = np.array(self.action_history[i-1][:-1])
                curr_action = np.array(self.action_history[i][:-1])
                continuity += np.linalg.norm(curr_action - prev_action)
            if self.action_history[-1][-1] - self.action_history[-2][-1] > 1.5: #不abs了吧就用正的减去负的
                gripper_changed = True #闭合
            continuity /= (len(self.action_history) - 1)

            prev_cont = self.prev_continuity
            delta = None if prev_cont is None else (continuity - prev_cont)
            continuity_info.update(
                {
                    "score": float(continuity),
                    "delta": None if delta is None else float(delta),
                    "gripper_changed": gripper_changed,
                    "ready": True,
                }
            )

            thr = max(float(self.continuity_threshold), 1e-6)
            if continuity <= 0.4 * thr:
                level = "very_high"
                label = f"Very high trajectory continuity: {continuity:.4f}"
            elif continuity <= 0.8 * thr:
                level = "high"
                label = f"High trajectory continuity: {continuity:.4f}"
            elif continuity <= 1.25 * thr:
                level = "medium"
                label = f"Medium trajectory continuity: {continuity:.4f}"
            elif continuity <= 1.75 * thr:
                level = "low"
                label = f"Low continuity: {continuity:.4f}, gripper_changed: {gripper_changed}, prev_continuity: {prev_cont}"
            else:
                level = "very_low"
                label = f"Very low continuity: {continuity:.4f}, gripper_changed: {gripper_changed}, prev_continuity: {prev_cont}"

            continuity_info["level"] = level #md没看清把他删了结果出问题了
            continuity_info["label"] = label
            # logging.info(
            #     "continuity: %.4f, level: %s, gripper_changed: %s, prev_continuity: %s, delta: %s",
            #     continuity,
            #     level,
            #     gripper_changed,
            #     prev_cont,
            #     delta,
            # )
            self.prev_continuity = continuity
        else:
            gripper_changed = False
            if len(self.action_history) > 1:
                if self.action_history[-1][-1] - self.action_history[-2][-1] > 1.5: #不abs了吧就用正的减去负的
                    gripper_changed = True #闭合
            logging.info(f"gripper_changed: {gripper_changed}")
            continuity_info.update(
                {
                    "label": "Continuity warmup",
                    "level": "warmup",
                    "gripper_changed": gripper_changed,
                }
            )
            self.prev_continuity = None #感觉这个很关键
        return continuity_info
    
    def set_interruption(self):
        self.needs_interruption = True
        self.action_history = []

    def _normalize_attention_map(self, vla_attention_map):
        if isinstance(vla_attention_map, list) and len(vla_attention_map) > 0:
            if all(isinstance(layer, list) for layer in vla_attention_map) and all(
                isinstance(item, torch.Tensor) for layer in vla_attention_map for item in layer
            ):
                return torch.stack([torch.stack(layer) for layer in vla_attention_map])
            if all(isinstance(item, torch.Tensor) for item in vla_attention_map):
                return torch.stack(vla_attention_map)
        return vla_attention_map

    def propose_value_horizon(
        self,
        image,
        instruction,
        vla_attention_map,
        use_uncertainty=True,
        use_coverage=True,
        *,
        parsed_objs=None,
        boxes_xyxy=None,
        phrases=None,
        continuity_info=None,
        current_horizon=None,
        box_source=None,
    ):
        """
        核心函数：根据图像、指令和 VLA 内部状态计算当前的"策略价值"
        并返回推荐的 horizon。
        Input:
            image: 图像数据 (numpy array 或路径)
            instruction: 任务指令
            vla_attention_map: VLA 注意力图 [num_layer, num_heads, seq_len] / list of list of tensors
            use_uncertainty: 是否使用不确定性计算策略
            use_coverage: 是否使用覆盖率计算策略
        Output:
            tuple: (horizon, info_dict)
                horizon: 推荐的 horizon 值
                info_dict: 包含计算信息的字典
        """
        vla_attention_map = self._normalize_attention_map(vla_attention_map)

        if current_horizon is None:
            current_horizon = self.current_horizon

        local_needs_interruption = False

        # 1. 解析指令，获取 DINO 的 Prompt
        role_info = _instruction_role_info(instruction)
        if parsed_objs is None:
            parsed_objs = role_info["target_object"]
        if not parsed_objs:
            # 如果解析失败，保守策略
            return {
                "current_horizon": self.min_h,
                "needs_interruption": False,
                "info": {"reason": "parse_fail"},
            }

        # 2. 获取 Ground Truth (DINO)
        if boxes_xyxy is None or phrases is None:
            _, boxes_xyxy, _, phrases, _ = self.get_dino_boxes(image, parsed_objs)
              
        # if len(phrases) != len(parsed_objs): #感觉不能这样判断
        #     logging.warning(f"DINO 识别的短语数量与解析的对象数量不匹配！parsed_objs: {parsed_objs}, phrases: {phrases}")
        #     return self.current_horizon, {"reason": "dino_parse_mismatch", "parsed_objs": parsed_objs, "dino_phrases": phrases}
        
        boxes = boxes_xyxy
        if boxes is None:
            boxes = torch.tensor([])
        elif not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) > 0:
            boxes = torch.clamp(boxes, 0.0, 1.0)
        
        logging.info(f"parsed_objs: {parsed_objs}, boxes: {boxes}, phrases: {phrases}")

        # 3. 计算指标
        info_dict = {
            "target_object": parsed_objs,
            "tograsp_objects": role_info["tograsp_objects"],
            "reference_objects": role_info["reference_objects"],
            "dino_boxes": boxes.tolist()
        }
        if phrases:
            info_dict["phrases"] = phrases
        if box_source is not None:
            info_dict["box_source"] = box_source
        
        confidence = 1.0
        reasons = []
        
        kl_scores = []
        overlaps = []
        peak_scores = []
        uncertainty_scores = []
        peak_high_count = 0
        peak_low_count = 0
        peak_thresh_high = 0.8
        peak_thresh_low = 0.3
        
        if continuity_info is None:
            continuity_info = self.compute_continuity_info()
        continuity_factor = continuity_info["label"]
        
        # 遍历每一层
        for layer_idx in range(vla_attention_map.shape[0]):
            layer_attn = vla_attention_map[layer_idx]
            metrics = self.uncertainty_engine.compute_uncertainty(layer_attn, boxes)
            # logging.info(f"layer_idx: {layer_idx}, metrics: {metrics}")
            kl_scores.append(metrics['kl'])
            overlaps.append(metrics.get("coverage", 0.0))
            peak_scores.append(metrics['peak_ratio'])
            uncertainty_scores.append(metrics['uncertainty'])
            if peak_scores[-1] > peak_thresh_high:
                peak_high_count += 1
            elif peak_scores[-1] < peak_thresh_low:
                peak_low_count += 1
            # logging.info(metrics)
            
        # if peak_high_count >= 2:
        #     confidence = 5
        # elif continuity_factor.startswith("High"):
        #     confidence = 3
        
        # if peak_low_count >= 2 and continuity_factor.startswith("Low"):
        #     confidence = 1
        num_layers = len(peak_scores)
        peak_high_ratio = peak_high_count / max(num_layers, 1)
        peak_low_ratio = peak_low_count / max(num_layers, 1)
        avg_coverage = float(np.mean(overlaps)) if overlaps else 0.0
        best_kl = float(np.min(kl_scores)) if kl_scores else 10.0
        avg_uncertainty = float(np.mean(uncertainty_scores)) if uncertainty_scores else -1.0

        continuity_level = continuity_info["level"]
        is_high_cont = continuity_level in {"high", "very_high"}
        is_low_cont = continuity_level in {"low", "very_low"}

        confidence = 2  # default -> chunk 15
        # 不想用kl来判断

        # if is_low_cont and (peak_low_ratio >= 0.75 or avg_coverage < 0.05 or best_kl > 0.02):
        #     confidence = 0
        #     self.needs_interruption = True
        # elif is_low_cont or peak_low_ratio >= 0.5 or avg_coverage < 0.08 or best_kl > 0.018:
        #     confidence = 1
        # elif peak_high_ratio >= 0.5 and avg_coverage >= 0.20 and best_kl <= 0.008 and is_high_cont:
        #     confidence = 5
        # elif peak_high_ratio >= 0.5 and avg_coverage >= 0.14 and best_kl <= 0.012:
        #     confidence = 4
        # elif peak_high_ratio >= 0.25 and avg_coverage >= 0.10 and best_kl <= 0.016:
        #     confidence = 3
        # elif is_high_cont:
        #     confidence = 3
        # else:
        #     confidence = 2
        if is_low_cont and (peak_low_ratio >= 0.75 or avg_coverage < 0.05):
            confidence = 0
            # local_needs_interruption = True
        elif is_low_cont or peak_low_ratio >= 0.5 or avg_coverage < 0.08:
            confidence = 1
        elif peak_high_ratio >= 0.5 and avg_coverage >= 0.20 and is_high_cont:
            confidence = 5
        elif peak_high_ratio >= 0.5 and avg_coverage >= 0.14:
            confidence = 4
        elif peak_high_ratio >= 0.25 and avg_coverage >= 0.10 :
            confidence = 3
        elif is_high_cont:
            confidence = 3
        else:
            confidence = 2

        # 现在的策略基本是被attention控制的
        # logging.info(
        #     "confidence: %s, continuity_factor: %s, continuity_level: %s",
        #     confidence,
        #     continuity_factor,
        #     continuity_level,
        # )
                
        proposed_horizon = max(self.min_h, self.get_chunk_size(confidence))
        return {
            "current_horizon": proposed_horizon,
            "needs_interruption": local_needs_interruption,
            "info": info_dict,
        }

    def ValueFunc(self, image, instruction, vla_attention_map, use_uncertainty=True, use_coverage=True):
        proposal = self.propose_value_horizon(
            image,
            instruction,
            vla_attention_map,
            use_uncertainty=use_uncertainty,
            use_coverage=use_coverage,
        )
        return self.apply_chunk_proposal(proposal)
    
    def get_chunk_size(self, confidence): # 0, 1, 2, 3, 4, 5
        return int(self.min_h + (self.max_h - self.min_h) / self.chunk_levels * confidence)


    def propose_adaptive_chunk_heuristic(
        self,
        image,
        instruction,
        robot_state=None,
        object_state=None,
        *,
        current_horizon=None,
    ):
        """
        基于启发式逻辑的自适应chunk size调整函数
        不依赖VLA attention map，可在后台持续运行
        Input:
            image: 图像数据 (numpy array 或路径)
            instruction: 任务指令
            robot_state: 机器人状态 (可选)，包含位置、速度等信息
            object_state: 对象状态 (可选)，包含位置、速度等信息
            current_action: 当前动作 (可选)，用于计算轨迹连续性
        Output:
            tuple: (horizon, info_dict)
                horizon: 推荐的 horizon 值
                info_dict: 包含计算信息的字典
        """
        # 1. 解析指令，获取 DINO 的 Prompt
        role_info = _instruction_role_info(instruction)
        parsed_objs = role_info["target_object"]
        
        if not parsed_objs:
            # 如果解析失败，保守策略
            return self.min_h, {"reason": "parse_fail"}

        # 2. 获取 Ground Truth / tracked boxes
        boxes_cxcywh, boxes_xyxy, _, phrases, _ = self.get_dino_boxes(image, parsed_objs)
        boxes = boxes_cxcywh
        
        # 3. 初始化信息字典
        info_dict = {
            "target_object": parsed_objs,
            "tograsp_objects": role_info["tograsp_objects"],
            "reference_objects": role_info["reference_objects"],
            "dino_boxes": (
                boxes_xyxy.tolist()
                if isinstance(boxes_xyxy, torch.Tensor)
                else boxes.tolist()
            ),
        }
        if phrases:
            info_dict["phrases"] = phrases
        if hasattr(self, "last_query_used_dino"):
            info_dict["box_source"] = "dino" if self.last_query_used_dino else "tracker"
        
        factors = []
        # 4.1 基于目标检测的启发式
        if len(boxes) > 0:
            # 目标数量因子：目标越多，需要更频繁的决策
            num_objects = len(boxes)
            info_dict["num_objects"] = num_objects
            if num_objects > 3:
                factors.append("Multiple objects detected")
        elif hasattr(self, "trackers"):
            info_dict["tracker_state"] = "missing"
                
        local_needs_interruption = False
        prev_h = self.current_horizon if current_horizon is None else current_horizon
        continuity_info = self.compute_continuity_info()
        continuity_factor = continuity_info["label"]
        continuity_level = continuity_info["level"]
        
        if continuity_level == "very_high":
            proposed_horizon = min(self.max_h, prev_h + 5)
        elif continuity_level == "high":
            proposed_horizon = min(self.max_h, prev_h + 3)
        elif continuity_level == "medium":
            proposed_horizon = prev_h
        elif continuity_level == "low":
            proposed_horizon = max(self.min_h, min(prev_h - 3, self.get_chunk_size(2)))
        elif continuity_level == "very_low":
            local_needs_interruption = True
            logging.info("continuity very low -> reduce to cautious chunk")
            proposed_horizon = max(self.min_h, self.get_chunk_size(1))
        else:
            proposed_horizon = prev_h

        factors.append(continuity_factor)
        info_dict["continuity_level"] = continuity_level
        info_dict["continuity_score"] = continuity_info["score"]
        info_dict["continuity_info"] = dict(continuity_info)
        info_dict["factors"] = factors
        return {
            "current_horizon": proposed_horizon,
            "needs_interruption": local_needs_interruption,
            "info": info_dict,
        }

    def adaptive_chunk_heuristic(self, image, instruction, robot_state=None, object_state=None):
        proposal = self.propose_adaptive_chunk_heuristic(
            image,
            instruction,
            robot_state=robot_state,
            object_state=object_state,
        )
        return self.apply_chunk_proposal(proposal)

# ==============================================================================
# 4. Simulation / Test Run
# ==============================================================================
if __name__ == "__main__":
    # 模拟环境
    chunker = AdaptiveActionChunker(max_horizon=30)
    
    # 模拟数据
    dummy_image = np.zeros((224, 224, 3))
    instruction = "Pick up the blue plastic cup and put it on the coaster."
    
    # 模拟 VLA Attention: [Heads, Seq, Seq] -> 假设 Image Tokens 是前 256 个
    # Case 1: Attention 很分散 (High Entropy)
    vla_attn_scattered = torch.rand(12, 512, 512) 
    
    # Case 2: Attention 聚焦在中心 (模拟聚焦物体)
    vla_attn_focused = torch.zeros(12, 512, 512)
    # 假设物体在 patch 119, 120 (中心附近)
    vla_attn_focused[:, -1, 119:121] = 5.0 
    vla_attn_focused = vla_attn_focused.softmax(dim=-1)

    print(f"Instruction: {instruction}")
    
    # Test 1: 分散注意力 -> 应该降低 Horizon
    h1, info1 = chunker.ValueFunc(dummy_image, instruction, vla_attn_scattered)
    print(f"\n[Case 1 Scattered] Horizon: {h1}, Reason: {info1['reason']}")
    
    # Test 2: 聚焦注意力 -> 应该增加 Horizon
    # 强行重置 horizon 以测试增加逻辑
    chunker.current_horizon = 10
    h2, info2 = chunker.ValueFunc(dummy_image, instruction, vla_attn_focused)
    print(f"\n[Case 2 Focused] Horizon: {h2}, Reason: {info2['reason']}")
