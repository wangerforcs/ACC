#!/usr/bin/env python3
"""
基于evalpi05.py的自适应chunk size版本
使用adaptivepolicy.py中的结构来实现运行时改变chunk size
"""

import concurrent.futures
import threading
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from collections import Counter

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange, tqdm
import cv2

# LeRobot & VLASH 核心导入
from lerobot.configs import parser
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import close_envs
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm
from lerobot.utils.random_utils import set_seed
from lerobot.utils.control_utils import prepare_observation_for_inference

# 导入 VLASH 特有组件
from vlash.policies.factory import get_policy_class, get_lerobot_policy_class
from visualutils import visualize_vla_attention, visualize_vla_attention_dino, vla_attention_specs, save_images, visualize_vla_attention_sum_heads_per_layer
from vlash.configs.train_config import VLASHTrainConfig

# 导入自适应策略
# from adaptive.adaptivepolicyfinegrained_tracker import AdaptiveActionChunker
from adaptive.adaptivepolicyfinegrained_tracker_scoregate import AdaptiveActionChunker
# from adaptive.adaptivepolicyfinegrained import AdaptiveActionChunker
from adaptive.monitor import GPUMonitor

# =============================================================================
# 1. 配置定义
# =============================================================================
from lerobot.configs.policies import PreTrainedConfig

# =============================================================================
# 2. 向量化异步管理器（带自适应chunk size）
# =============================================================================
from dataclasses import dataclass, field

@dataclass
class VLASHEvalConfig:
    """评估配置类"""
    env_type: str = "libero"
    env_task: str = "libero_object"
    max_episodes: int = 20  # 每个子任务测试的集数
    eval_batch_size: int = 5 # 并行环境数
    seed: int = 42
    output_dir: str = "outputs/eval/vlash_adaptive"

    policy: Union[PreTrainedConfig, dict[str, Any], None] = None
    
    fps: int = 30
    display_data: bool = False
    play_sounds: bool = False
    action_quant_ratio: int = 1
    inference_overlap_steps: int = 0
    # 注意：不再强制要求 single_task，评估时会从环境元数据提取
    single_task: Union[str, None] = None 
    task_ids: Union[List[int], None] = None
    train_cfg: VLASHTrainConfig = field(default_factory=VLASHTrainConfig)
    
    # 自适应策略配置
    use_adaptive_chunk: bool = True
    adaptive_min_horizon: int = 5 #这会不会是隐患，要改成5吗，和sync5对齐
    adaptive_max_horizon: int = 30
    adaptive_uncertainty_threshold: float = 0.015
    adaptive_coverage_threshold: float = 0.1
    adaptive_continuity_threshold: float = 0.2
    
    # 视频保存配置
    save_video: bool = False
    video_output_dir: str = "fuckvideos"
    
    use_monitor: bool = False
    eval_log_level: str = "INFO"
    tracker_log_level: str = "WARNING"
    scoregate_log_level: str = "WARNING"
    
    def __post_init__(self):
        if not isinstance(self.policy, PreTrainedConfig):
            policy_path = None
            cli_overrides = []
            if isinstance(self.policy, dict):
                if "path" not in self.policy and "pretrained_path" in self.policy:
                    self.policy["path"] = self.policy.pop("pretrained_path")
                policy_path = self.policy.pop("path")
                for k, v in self.policy.items():
                    val = str(v).lower() if isinstance(v, bool) else v
                    cli_overrides.append(f"--{k}={val}")
            
            cli_policy_path = parser.get_path_arg("policy")
            if cli_policy_path: policy_path = cli_policy_path
            
            cli_policy_overrides = parser.get_cli_overrides("policy")
            if cli_policy_overrides: cli_overrides.extend(cli_policy_overrides)
        
            if policy_path:
                self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
                self.policy.pretrained_path = policy_path


def _configure_component_log_levels(cfg: VLASHEvalConfig):
    root_level = getattr(logging, str(cfg.eval_log_level).upper(), logging.INFO)
    logging.getLogger().setLevel(root_level)
    logging.getLogger("adaptive.tracker").setLevel(
        getattr(logging, str(cfg.tracker_log_level).upper(), logging.WARNING)
    )
    logging.getLogger("adaptive.tracker_scoregate").setLevel(
        getattr(logging, str(cfg.scoregate_log_level).upper(), logging.WARNING)
    )

class VLASHVectorAsyncManager:
    """
    针对向量化环境的异步管理器
    支持：多线程预取 (Hungry Strategy)、Batch推理、自动 Chunk 切换、自适应chunk size
    """
    def __init__(self, policy: PreTrainedPolicy, overlap_steps: int, num_envs: int, task_name, cfg: VLASHEvalConfig):
        self.policy = policy
        self.n_action_steps = policy.config.n_action_steps
        self.overlap_steps = overlap_steps
        self.num_envs = num_envs
        self.device = get_safe_torch_device(policy.config.device)
        self.task_name = task_name
        self.cfg = cfg
        
        # 初始化自适应策略
        if cfg.use_adaptive_chunk:
            env_continuity_threshold = cfg.adaptive_continuity_threshold
            # if cfg.env_task == "libero_object":
            #     env_continuity_threshold = 0.2
            # elif cfg.env_task == "libero_goal":
            #     env_continuity_threshold = 0.25
            self.adaptive_chunker = AdaptiveActionChunker(
                min_horizon=cfg.adaptive_min_horizon,
                max_horizon=cfg.adaptive_max_horizon,
                uncertainty_threshold=cfg.adaptive_uncertainty_threshold,
                coverage_threshold=cfg.adaptive_coverage_threshold,
                continuity_threshold=env_continuity_threshold,
            )
            # 初始chunk size
            self.current_chunk_size = self.adaptive_chunker.current_horizon
        else:
            self.adaptive_chunker = None
            self.current_chunk_size = self.n_action_steps
        self.valid_chunk_size = self.current_chunk_size
        self.last_chunk_size = self.current_chunk_size
        
        # 线程池：用于后台执行推理任务和自适应任务
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # 增加线程用于后台处理
        self.next_chunk_future = None
        self.background_future = None
        
        # 状态追踪
        self.current_chunks = None  # 形状: [Batch, Horizon, ActionDim]
        self.chunk_index = 0
        
        # 校验
        assert self.n_action_steps > self.overlap_steps
        
        self.last_observation_imgs = None
        self.save_attention_step = 5
        
        self.video_images = []
        self.target_layers = [9, 10, 11, 12]
        # self.target_layers = [12]
        self.video_heatmap = []
        
        # 视频保存相关
        self.save_video = cfg.save_video
        self.video_output_dir = cfg.video_output_dir
        
        self.indexes = None
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None
        self.launch_step_intervals = []

    def reset_state(self):
        """每个 Batch 开始前调用"""
        if self.next_chunk_future:
            self.next_chunk_future.cancel()
        if self.background_future:
            self.background_future.cancel()
        self.next_chunk_future = None
        self.background_future = None
        self.current_chunks = None
        self.chunk_index = 0
        # 重置视频帧列表
        self.video_images = []
        # 添加与 rollout.py 一致的 policy.reset() 调用
        self.policy.reset()
        if self.adaptive_chunker:
            self.adaptive_chunker.reset()
            self.current_chunk_size = self.adaptive_chunker.current_horizon
        else:
            self.current_chunk_size = self.n_action_steps
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None

    def _record_launch_interval(self):
        if self.last_launch_step is not None:
            interval = self.batch_env_step - self.last_launch_step
            self.launch_step_intervals.append(interval)
        self.last_launch_step = self.batch_env_step

    def get_launch_interval_stats(self):
        if not self.launch_step_intervals:
            return {
                "count": 0,
                "intervals": [],
                "distribution": {},
                "mean": 0.0,
                "min": None,
                "max": None,
            }
        dist = Counter(self.launch_step_intervals)
        return {
            "count": len(self.launch_step_intervals),
            "intervals": list(self.launch_step_intervals),
            "distribution": dict(sorted(dist.items())),
            "mean": float(np.mean(self.launch_step_intervals)),
            "min": int(min(self.launch_step_intervals)),
            "max": int(max(self.launch_step_intervals)),
        }

    def save_video_frames(self, filename):
        """保存视频"""
        if not self.video_images:
            logging.info("No video frames to save.")
            return

        # 创建视频输出目录
        video_dir = Path(self.video_output_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        # 获取视频帧的尺寸
        height, width, _ = self.video_images[0].shape

        # 创建视频写入器
        video_path = video_dir / f"{filename}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.cfg.fps, (width, height))

        # 写入视频帧
        for img in self.video_images:
            # 确保图像是BGR格式
            if img.shape[-1] == 3:
                # 如果是RGB格式，转换为BGR
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                frame = img
            out.write(frame)

        # 释放视频写入器
        out.release()
        logging.info(f"Video saved to {video_path}")
        self.video_images = []

    def _prepare_batch_obs(self, raw_obs: Dict[str, Any], task_description: str):
        """
        将 VectorEnv 的原始观测转换为 Policy 需要的 Batch Tensor 格式
        """
        # 注意：VectorEnv 返回的 raw_obs 已经是 [B, ...] 格式了
        
        # 为了兼容 prepare_observation_for_inference，我们手动构造 Batch 数据
        batch_input = {}
        
        # 1. 处理状态 (Proprioception)
        if "agent_pos" in raw_obs:
            # raw_obs["agent_pos"] 形状是 [B, StateDim]
            batch_input["observation.state"] = torch.from_numpy(raw_obs["agent_pos"]).float().to(self.device)
        
        # 2. 处理图像
        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            for k, v in raw_obs["pixels"].items():
                '''
                image (1, 256, 256, 3)
                image2 (1, 256, 256, 3)
                '''
                # v 形状是 [B, H, W, C] -> 需要转换为 [B, C, H, W] 并归一化
                img_tensor = torch.from_numpy(v).float().permute(0, 3, 1, 2).to(self.device) / 255.0
                batch_input[f"observation.images.{k}"] = img_tensor
        
        # 3. 处理任务描述
        # 策略通常期望任务描述是一个 List[str] 或统一的字符串
        batch_input["task"] = [task_description] * self.num_envs
        
        return batch_input

    def _inference_task(self, batch_obs_prepared):
        """在后台线程执行的推理函数"""
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch_obs_prepared)
            return actions.cpu().numpy() # [B, Horizon, ActionDim]

    def _background_adaptive_task(self, image, instruction, robot_state=None):
        """在后台线程执行的自适应chunk size调整"""
        if self.adaptive_chunker:
            # 使用启发式策略调整chunk size
            _, info = self.adaptive_chunker.adaptive_chunk_heuristic(image, instruction, robot_state)
            # 注意：这里不直接更新current_chunk_size，而是在主线程中处理
            return info
        return {}

    def launch_next_inference(self, raw_obs: Dict[str, Any], task_description: str):
        """启动后台推理"""
        self.inference_times += 1
        self._record_launch_interval()
        if self.adaptive_chunker is not None:
            logging.info(f"Launching inference for task")
        self.save_attention_step -= 1
        prepared_obs = self._prepare_batch_obs(raw_obs, task_description)
        self.valid_chunk_size = self.current_chunk_size
        # 启动推理任务
        self.next_chunk_future = self.executor.submit(self._inference_task, prepared_obs)

    def reset_chunk(self, reason=False):
        self.current_chunks = self.next_chunk_future.result()
        self.last_chunk_size = self.valid_chunk_size
        self.next_chunk_future = None
        if self.adaptive_chunker is not None:
            self.adaptive_chunker.action_history = []
            if reason is False:
                self.get_gripper_change_index()
        
    def get_gripper_change_index(self):
        """获取当前 Chunk 中 gripper 动作变化的索引"""
        # 计算 gripper 动作的变化量
        gripper_changes = self.current_chunks[:, 1:, -1] - self.current_chunks[:, :-1, -1]
        # 找到第一个变化大于阈值的索引
        # 这个where返回的是元祖，第一个元素是满足条件的行索引，第二个元素是满足条件的列索引
        self.indexes = np.where(gripper_changes > 1.5)[1][0] if np.any(gripper_changes > 1.5) else None
        logging.info(f"Gripper change index: {self.indexes}")
        
    def should_interrupt_gripper(self):
        if self.adaptive_chunker and self.indexes is not None:
            if self.chunk_index + 5 == self.indexes:
            # if self.chunk_index + 5 <= self.indexes and self.chunk_index + 10 >= self.indexes: # 提前5步到后10步内预测到有夹爪变化
                logging.info(f"Interrupting current chunk due to predicted gripper change at index {self.indexes}")
                self.indexes = None
                return True
            else:
                return False
        return False

    def get_action_batch(self, raw_obs: Dict[str, Any], task_description: str) -> np.ndarray:
        """
        获取当前步的 Batch Action
        """
        if self.save_video:
            if 'image' in raw_obs["pixels"]:
                self.video_images.append(raw_obs["pixels"]['image'][0])
        should_launch = False
        gripper_change = False
        # 启动后台自适应任务（每次调用都会执行，比inference频率更高）
        if self.adaptive_chunker and "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            # 获取当前图像
            image = raw_obs["pixels"]['image'][0]  # 取第一个环境的图像
            
            # 构建机器人状态（如果可用）
            robot_state = None
            if "agent_pos" in raw_obs:
                # 取第一个环境的机器人位置, 我们 假设使用时batch大小为1
                agent_pos = raw_obs["agent_pos"][0]
                robot_state = {
                    "ee_position": agent_pos[:3],  # 假设前三个元素是末端执行器位置
                }
            
            self.background_future = self.executor.submit(self._background_adaptive_task, image, task_description, robot_state)
        
        # 第一次和后面的逻辑合一了
        if self.current_chunks is None:
            if self.next_chunk_future is None:
                self.launch_next_inference(raw_obs, task_description)
                self.chunk_index = 0
            else:
                self.chunk_index = self.overlap_steps
                
            # 阻塞等待结果（如果是预取的，这里会立即返回；如果是同步的，这里会等待）
            self.reset_chunk()
            # self.current_chunks = self.next_chunk_future.result()
            # self.next_chunk_future = None
            
            # 检查后台自适应任务是否完成
            if self.background_future:
                try:
                    background_info = self.background_future.result(timeout=1.0)  # 最多等待0.1秒
                    # 更新当前chunk size
                    self.current_chunk_size = self.adaptive_chunker.current_horizon
                    logging.info(f"Adaptive chunk size: {self.current_chunk_size}")
                except concurrent.futures.TimeoutError:
                    print(f"Adaptive chunk size background task timed out: {background_info}")
                    pass  # 超时不影响主流程
                self.background_future = None
            
            # 获取注意力图并使用它进一步调整chunk size (只有在policy真正做chunk推理时才使用ValueFunc)
            if self.adaptive_chunker:
                try:
                    attn_maps = self.policy.get_attn_maps(layers=self.target_layers, heads=range(8), force=True)
                    # 使用注意力图调整chunk size
                    if attn_maps:
                        vla_attention_map = attn_maps #[layers, num_heads, seq_len]
                        # 使用ValueFunc根据注意力图调整chunk size
                        _, info = self.adaptive_chunker.ValueFunc(
                            raw_obs["pixels"]['image'][0], 
                            task_description, 
                            vla_attention_map
                        )
                        # 更新当前chunk size
                        self.current_chunk_size = self.adaptive_chunker.current_horizon
                        # 立马根据Value func更新last_chunk_size，仅用当前指定的chunk size
                        self.last_chunk_size = min(self.current_chunk_size, self.last_chunk_size)
                        # 检查是否需要立即打断当前chunk
                        if hasattr(self.adaptive_chunker, 'needs_interruption') and self.adaptive_chunker.needs_interruption:
                            # 立即打断当前chunk，重置状态
                            logging.info(f"Interrupting current chunk due to attention-based adjustment")
                            should_launch = True
                            # self.current_chunks = None
                            # self.chunk_index = 0
                            # 立即启动新的推理
                            # self.launch_next_inference(raw_obs, task_description)
                        logging.info(f"Attention-based chunk size: {self.current_chunk_size}")
                except Exception as e:
                    import traceback
                    logging.warning(f"Failed to get attention maps: {e}")
                    traceback.print_exc()
            
            # self.video_heatmap.append(attn_maps)
            # if self.save_attention_step == 0 and self.last_observation_imgs is not None:
            #     # 可视化注意力
            #     see_layers = range(17)
            #     see_heads = range(8)
            #     attn_maps = self.policy.get_attn_maps(layers=range(17), heads=range(8))
            #     visualize_vla_attention_sum_heads_per_layer(
            #         images=self.last_observation_imgs,
            #         attn_maps=attn_maps,
            #         img_start_indices=[0, 256],
            #         output_dir='visualizations/8to14everylayer'+self.task_name
            #     )
            #     save_images(self.last_observation_imgs, filename=self.task_name)
        else:
            # 检查后台自适应任务是否完成
            if self.background_future:
                try:
                    background_info = self.background_future.result(timeout=1.0)  # 最多等待0.05秒
                    # 更新当前chunk size
                    self.current_chunk_size = self.adaptive_chunker.current_horizon
                    # 检查是否需要立即打断当前chunk
                    if hasattr(self.adaptive_chunker, 'needs_interruption') and self.adaptive_chunker.needs_interruption:
                        # 立即打断当前chunk，重置状态并启动新的预测
                        logging.info(f"Interrupting current chunk due to: {background_info.get('factors', 'N/A')}")
                        should_launch = True
                        # self.chunk_index = 0
                        # # 立即启动新的推理
                        # self.launch_next_inference(raw_obs, task_description)
                    # 每5步记录一次启发式调整
                    # if self.chunk_index % 5 == 0:
                    # logging.info(f"Heuristic-based chunk size: {self.current_chunk_size}, Reason: {background_info.get('factors', 'N/A')}")
                    if self.should_interrupt_gripper():
                        should_launch = True
                        gripper_change = True
                except concurrent.futures.TimeoutError:
                    pass  # 超时不影响主流程
                self.background_future = None

        # --- 3. 预取逻辑 (Hungry Strategy) ---
        # 当执行到当前 Chunk 的特定位置时，提前启动下一波推理
        if self.overlap_steps > 0:
            overlap_point = min(self.current_chunk_size, self.n_action_steps, self.last_chunk_size) - self.overlap_steps
            if self.chunk_index == overlap_point and self.next_chunk_future is None:
                logging.info(f"Manager: Reached overlap point ({overlap_point}), launching prefetch...")
                self.launch_next_inference(raw_obs, task_description)

        # --- 4. 提取当前步 Action ---
        # 打断开始推理
        if should_launch:
            if self.next_chunk_future is None:
                # 这个应该只需要current chunk size > overlapped就行 
                # 重置模型的KV缓存，避免注意力掩码大小不匹配
                # for layer in self.policy.model.layers:
                #     layer.self_attn.attn.reset_cache()
                logging.info(f"Manager: Interrupting current chunk with chunk size: {self.current_chunk_size}")    
                self.launch_next_inference(raw_obs, task_description)
                # 还要继续执行吗 执行的话就是index继续增加, 欸但是貌似这个没有起作用吧
                # 因为后续chunk size可能增加会来了，那还是打断吧
                
            # 阻塞等待结果 #这里如果打断了需要同步吗
            self.reset_chunk(gripper_change)
            # self.current_chunks = self.next_chunk_future.result() #md还是回出错在这里
            # self.next_chunk_future = None
            # # self.chunk_index = self.overlap_steps
            self.chunk_index = 0
        
        # 确保不超出当前chunk size
        current_step = min(self.chunk_index, min(self.current_chunk_size, self.n_action_steps, self.last_chunk_size) - 1)
        actions = self.current_chunks[:, current_step, :] # [B, ActionDim]
        # print("gripper action: ", actions[0, -1])

        if self.adaptive_chunker:
            self.adaptive_chunker.enqueue_history(actions[0].tolist())

        # logging.info(f"Manager: Current chunk index: {self.chunk_index}")

        # overlapped=4且min horizon=5时会导致一但进入5，就会持续每步预测，然后不连续导致进一步维持在5
        self.chunk_index = (self.chunk_index + 1)
        # logging.info(f"last chunk size: {self.last_chunk_size}, valid chunk size: {self.valid_chunk_size}")
        # 当达到当前chunk size或最大步骤时，重置
        if self.chunk_index >= min(self.current_chunk_size, self.n_action_steps, self.last_chunk_size):
            self.current_chunks = None
            self.chunk_index = 0
        self.batch_env_step += 1
        return actions


# =============================================================================
# 3. 评估函数逻辑
# =============================================================================

def eval_single_vec_env(
    vec_env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    cfg: VLASHEvalConfig,
    task_description: str,
    task_name: str
) -> Dict[str, Any]:
    
    effective_overlap = cfg.inference_overlap_steps * cfg.action_quant_ratio
    
    # 初始化异步 Batch 管理器
    async_mgr = VLASHVectorAsyncManager(
        policy=policy,
        overlap_steps=effective_overlap,
        num_envs=cfg.eval_batch_size,
        task_name=task_name,
        cfg=cfg
    )

    total_episodes = cfg.max_episodes
    n_batches = (total_episodes + cfg.eval_batch_size - 1) // cfg.eval_batch_size
    
    successes = []
    episode_steps = []
    max_steps = vec_env.call("_max_episode_steps")[0] #这里有max_steps
    monitor = GPUMonitor(gpu_id=0, interval=0.05)
    if cfg.use_monitor:
        monitor.start()
    for b in range(n_batches):
        # 使用全局种子 + 批次索引作为种子，确保每次运行结果一致
        batch_seed = cfg.seed + b
        observation, info = vec_env.reset(seed=batch_seed)
        async_mgr.reset_state()
        
        dones = np.array([False] * cfg.eval_batch_size)
        batch_success = np.array([False] * cfg.eval_batch_size)
        batch_steps = np.zeros(cfg.eval_batch_size, dtype=int)
        
        pbar = trange(max_steps, desc=f" Batch {b+1}/{n_batches}", leave=False, disable=inside_slurm())
        for step in pbar:
            if np.all(dones): break
            
            # 使用异步管理器直接获取 Batch Actions
            # 它内部会自动处理：
            # 1. 什么时候该在后台跑推理
            # 2. 什么时候该等待推理结果并切换 Chunk
            # 3. 后台运行Grounding DINO并调整chunk size
            # 4. 根据attention map进一步调整chunk size
            actions = async_mgr.get_action_batch(observation, task_description)
            
            observation, reward, terminated, truncated, info = vec_env.step(actions)
            
            # 增加步骤计数
            batch_steps[~dones] += 1
            
            # 成功率统计逻辑
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict) and "is_success" in final_info:
                    is_success_array = final_info["is_success"]
                    just_finished = (terminated | truncated) & (~dones)
                    if np.any(just_finished):
                        batch_success = np.where(just_finished, is_success_array, batch_success)
            
            dones = terminated | truncated | dones

        successes.extend(batch_success.tolist())
        episode_steps.extend(batch_steps.tolist())

        # 保存视频
        if cfg.save_video:
            async_mgr.save_video_frames(f"{task_name}_env0_batch{b}")
    if cfg.use_monitor:
        monitor.stop()
        stats = monitor.get_report() 
        logging.info(f"inference times: {async_mgr.inference_times}")
        launch_interval_stats = async_mgr.get_launch_interval_stats()
        logging.info(f"Launch inference interval stats: {launch_interval_stats}")
        logging.info(f"GPU Monitor Stats: {stats}")
    else:
        launch_interval_stats = async_mgr.get_launch_interval_stats()
        logging.info(f"inference times: {async_mgr.inference_times}")
        logging.info(f"Launch inference interval stats: {launch_interval_stats}")
    
    successes = successes[:total_episodes]
    
    # from visualutils import heatmap_videos
    # heatmap_videos(async_mgr.video_images, async_mgr.video_heatmap, filename='visualizations/'+task_name+".mp4")
    episode_steps = episode_steps[:total_episodes]
    return {
        "sr": np.mean(successes),
        "count": len(successes),
        "success_list": successes,
        "steps_list": episode_steps,
        "avg_steps": np.mean(episode_steps) if episode_steps else 0,
        "inference_times": async_mgr.inference_times,
        "launch_interval_stats": launch_interval_stats,
    }
    
@parser.wrap()
def run_benchmark(cfg: VLASHEvalConfig):
    init_logging()
    _configure_component_log_levels(cfg)
    set_seed(cfg.seed)
    
    # 添加与 rollout.py 相同的 torch 后端设置，确保结果一致
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    logging.info(f"max_episodes: {cfg.max_episodes}, inference_overlap_steps: {cfg.inference_overlap_steps}, ")
    logging.info(f"Adaptive chunk size enabled: {cfg.use_adaptive_chunk}")
    if cfg.use_adaptive_chunk:
        logging.info(f"Adaptive chunk size range: {cfg.adaptive_min_horizon} - {cfg.adaptive_max_horizon}")
    
    from vlash.train import make_vlash_dataset
    dataset = make_vlash_dataset(cfg.train_cfg)
    
    # 1. 加载策略
    policy_cls = get_policy_class(cfg.policy.type)
    # print(dataset.meta.stats)
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy, dataset_stats=dataset.meta.stats) 
    policy.eval()
    # 2. 初始化所有任务的环境
    if cfg.env_type == "libero":
        # 不指定 task_ids，让 factory 自动创建该 suite 下的所有任务
        env_cfg = LiberoEnv(task=cfg.env_task, task_ids=cfg.task_ids) 
    else:
        raise ValueError(f"Env type {cfg.env_type} not supported")

    logging.info(f"Initializing all tasks for {cfg.env_task}...")
    # make_env 返回 {task_group: {task_id: VectorEnv}}
    all_envs_dict = make_env(env_cfg, n_envs=cfg.eval_batch_size)

    # 3. 遍历评估
    results_per_task = {}
    overall_successes = []
    start_time = time.time()
    task_name = ""

    for suite_name, tasks in all_envs_dict.items():
        logging.info(f"Evaluating Suite: {suite_name}")
        for task_id, vec_env in tasks.items():
            # 从环境元数据中提取任务指令 (Libero 特有)
            # 如果配置中指定了 single_task 则覆盖，否则从环境取
            try:
                task_description = vec_env.get_attr("task_description")[0]
            except Exception:
                task_description = cfg.single_task
                
            logging.info(f"Starting Task [{task_id}]: {task_description}")
            
            task_name = suite_name + str(task_id)
            
            task_info = eval_single_vec_env(vec_env, policy, cfg, task_description, task_name=task_name)
            
            results_per_task[f"{suite_name}_{task_id}"] = {
                "task_description": task_description,
                "success_rate": task_info["sr"],
                "num_episodes": task_info["count"],
                "avg_steps": task_info["avg_steps"],
                "steps_list": task_info["steps_list"],
                "inference_times": task_info["inference_times"],
                "launch_interval_stats": task_info["launch_interval_stats"],
            }
            overall_successes.extend(task_info["success_list"])
            
            logging.info(f"Task Done. SR: {task_info['sr']*100:.1f}%")

    # 4. 汇总与保存
    total_time = time.time() - start_time
    
    # 收集所有episode的步骤
    all_steps = []
    for task_result in results_per_task.values():
        all_steps.extend(task_result["steps_list"])
    
    final_results = {
        "overall": {
            "avg_success_rate": np.mean(overall_successes),
            "total_episodes": len(overall_successes),
            "total_eval_time_s": total_time,
            "env_task": cfg.env_task,
            "use_adaptive_chunk": cfg.use_adaptive_chunk,
            "avg_steps_per_episode": np.mean(all_steps) if all_steps else 0,
            "total_steps": sum(all_steps) if all_steps else 0
        },
        "per_task": results_per_task
    }
    
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    logging.info("="*50)
    logging.info(f"FINAL OVERALL SR: {final_results['overall']['avg_success_rate']*100:.2f}%")
    logging.info(f"Results saved to {out_dir}")
    logging.info("="*50)

    close_envs(all_envs_dict)
    
    num_tasks = len(results_per_task)
    # 每个任务测多少
    ep_per_task = cfg.max_episodes
    # 总测试数
    total_test_count = len(overall_successes)

    logging.info("="*50)
    logging.info(f"评估统计报告:")
    logging.info(f" - 环境集合 (Suite): {cfg.env_task}")
    logging.info(f" - 任务总数 (Num Tasks): {num_tasks}")
    logging.info(f" - 每任务测试数 (Episodes per Task): {ep_per_task}")
    logging.info(f" - 总测试规模 (Total Episodes): {total_test_count}")
    logging.info(f" - 最终总成功率 (Overall SR): {np.mean(overall_successes)*100:.2f}%")
    logging.info(f" - 平均每episode步骤数 (Avg Steps per Episode): {np.mean(all_steps):.2f}")
    logging.info(f" - 总步骤数 (Total Steps): {sum(all_steps)}")
    logging.info(f" - 自适应chunk size: {'启用' if cfg.use_adaptive_chunk else '禁用'}")
    if cfg.use_adaptive_chunk:
        logging.info(f" - Chunk size范围: {cfg.adaptive_min_horizon} - {cfg.adaptive_max_horizon}")
    logging.info("="*50)
    
    # entropy_dict = policy.get_attention_entropies()
    # # 保存
    # with open(out_dir / (task_name+"entropy_dict.json"), "w") as f:
    #     json.dump(entropy_dict, f, indent=2)

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        traceback.print_exc()
