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
import importlib
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


def _unwrap_to_libero_env(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_env") and hasattr(current._env, "sim"):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
            continue
        break
    return None


def _extract_camera_projection(vec_env: gym.vector.VectorEnv, camera_name: str = "agentview"):
    if not hasattr(vec_env, "envs") or not vec_env.envs:
        return None

    libero_env = _unwrap_to_libero_env(vec_env.envs[0])
    if libero_env is None:
        return None

    try:
        from robosuite.utils.camera_utils import get_camera_transform_matrix
    except Exception as exc:
        logging.warning(f"Failed to import robosuite camera utils: {exc}")
        return None

    try:
        sim = libero_env._env.sim
        height = int(getattr(libero_env, "observation_height", 256))
        width = int(getattr(libero_env, "observation_width", 256))
        world_to_image = get_camera_transform_matrix(
            sim=sim,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        )
        return {
            "camera_name": camera_name,
            "camera_world_to_image": np.array(world_to_image, dtype=np.float32),
            "camera_resolution": (height, width),
        }
    except Exception as exc:
        logging.warning(f"Failed to extract camera projection for {camera_name}: {exc}")
        return None

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
    target_control_hz: float = 30.0
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
    # tracker_log_level: str = "WARNING"
    tracker_log_level: str = "INFO"
    scoregate_log_level: str = "WARNING"
    enable_gripper_interruption: bool = False
    adaptive_chunker_cls_path: str = ( #默认是finegrained_tracker_scoregate
        "adaptive.adaptivepolicyfinegrained_tracker_scoregate:AdaptiveActionChunker"
    )
    tracker_logger_names: List[str] = field(
        default_factory=lambda: ["adaptive.tracker", "adaptive.tracker_scoregate"]
    )
    
    def __post_init__(self):
        if self.target_control_hz is not None and self.target_control_hz <= 0:
            raise ValueError("target_control_hz must be positive when provided")
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


def _yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "\n", "\t"]):
        return json.dumps(text, ensure_ascii=False)
    return text


def _to_yaml_lines(data, indent=0):
    prefix = " " * indent
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        return lines
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(data)}"]


def _write_yaml_file(path: Path, data):
    path.write_text("\n".join(_to_yaml_lines(data)) + "\n", encoding="utf-8")


def _collect_eval_runtime_config(cfg: VLASHEvalConfig, adaptive_chunker):
    config = {
        "env_task": cfg.env_task,
        "use_adaptive_chunk": cfg.use_adaptive_chunk,
        "enable_gripper_interruption": cfg.enable_gripper_interruption,
        "adaptive_min_horizon": cfg.adaptive_min_horizon,
        "adaptive_max_horizon": cfg.adaptive_max_horizon,
        "adaptive_uncertainty_threshold": cfg.adaptive_uncertainty_threshold,
        "adaptive_coverage_threshold": cfg.adaptive_coverage_threshold,
        "adaptive_continuity_threshold": cfg.adaptive_continuity_threshold,
        "adaptive_chunker_cls_path": cfg.adaptive_chunker_cls_path,
        "target_control_hz": cfg.target_control_hz,
        "eval_batch_size": cfg.eval_batch_size,
        "max_episodes": cfg.max_episodes,
        "seed": cfg.seed,
    }
    if adaptive_chunker is not None and hasattr(adaptive_chunker, "get_runtime_config"):
        config["adaptive_chunker"] = adaptive_chunker.get_runtime_config()
    else:
        config["adaptive_chunker"] = None
    return config


def _configure_component_log_levels(cfg: VLASHEvalConfig):
    root_level = getattr(logging, str(cfg.eval_log_level).upper(), logging.INFO)
    logging.getLogger().setLevel(root_level)
    tracker_level = getattr(logging, str(cfg.tracker_log_level).upper(), logging.WARNING)
    scoregate_level = getattr(logging, str(cfg.scoregate_log_level).upper(), logging.WARNING)
    for logger_name in getattr(cfg, "tracker_logger_names", []) or []:
        level = scoregate_level if "scoregate" in logger_name else tracker_level
        logging.getLogger(logger_name).setLevel(level)


def _resolve_adaptive_chunker_cls(cfg: VLASHEvalConfig):
    cls_path = getattr(cfg, "adaptive_chunker_cls_path", None)
    if not cls_path:
        return AdaptiveActionChunker
    if ":" in cls_path:
        module_name, class_name = cls_path.split(":", 1)
    else:
        module_name, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _log_effective_eval_setup(cfg: VLASHEvalConfig):
    logging.info("=" * 50)
    logging.info("Effective Eval Setup")
    logging.info(f"Eval config class: {cfg.__class__.__module__}.{cfg.__class__.__name__}")
    logging.info(f"Env type/task: {cfg.env_type} / {cfg.env_task}")
    logging.info(f"Output dir: {cfg.output_dir}")
    logging.info(f"Max episodes: {cfg.max_episodes}")
    logging.info(f"Eval batch size: {cfg.eval_batch_size}")
    logging.info(f"Seed: {cfg.seed}")
    logging.info(f"Use adaptive chunk: {cfg.use_adaptive_chunk}")
    logging.info(f"Enable gripper interruption: {cfg.enable_gripper_interruption}")
    logging.info(f"Inference overlap steps: {cfg.inference_overlap_steps}")
    logging.info(f"Target control hz: {cfg.target_control_hz}")
    logging.info(f"Action quant ratio: {cfg.action_quant_ratio}")
    logging.info(f"Tracker logger names: {getattr(cfg, 'tracker_logger_names', [])}")
    if cfg.policy is not None:
        policy_path = getattr(cfg.policy, "pretrained_path", None)
        logging.info(f"Policy config class: {cfg.policy.__class__.__module__}.{cfg.policy.__class__.__name__}")
        logging.info(f"Policy type: {getattr(cfg.policy, 'type', None)}")
        logging.info(f"Policy path: {policy_path}")
    if cfg.train_cfg is not None:
        logging.info(f"Train config class: {cfg.train_cfg.__class__.__module__}.{cfg.train_cfg.__class__.__name__}")
        dataset = getattr(cfg.train_cfg, "dataset", None)
        if dataset is not None:
            logging.info(f"Dataset repo_id: {getattr(dataset, 'repo_id', None)}")
    if cfg.use_adaptive_chunk:
        adaptive_chunker_cls = _resolve_adaptive_chunker_cls(cfg)
        logging.info(
            f"Adaptive chunker class: {adaptive_chunker_cls.__module__}.{adaptive_chunker_cls.__name__}"
        )
        logging.info(
            f"Adaptive thresholds: min={cfg.adaptive_min_horizon}, max={cfg.adaptive_max_horizon}, "
            f"uncertainty={cfg.adaptive_uncertainty_threshold}, coverage={cfg.adaptive_coverage_threshold}, "
            f"continuity={cfg.adaptive_continuity_threshold}"
        )
    logging.info("=" * 50)

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
            adaptive_chunker_cls = _resolve_adaptive_chunker_cls(cfg)
            env_continuity_threshold = cfg.adaptive_continuity_threshold
            # if cfg.env_task == "libero_object":
            #     env_continuity_threshold = 0.2
            # elif cfg.env_task == "libero_goal":
            #     env_continuity_threshold = 0.25
            self.adaptive_chunker = adaptive_chunker_cls(
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
        
        # 推理和 background 分开，background 串行排队，保证“每步都提交”同时不并发写 chunker
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.next_chunk_future = None
        self.background_futures = []
        self.background_task_counter = 0
        self.latest_consumed_background_task_id = -1
        self.chunker_lock = threading.Lock()
        
        # 状态追踪
        self.current_chunks = None  # 形状: [Batch, Horizon, ActionDim]
        self.chunk_index = 0
        
        # 校验
        assert self.n_action_steps > self.overlap_steps
        
        self.last_observation_imgs = None
        self.save_attention_step = 5
        
        self.video_images = []
        self.target_layers = [9, 10, 11, 12]
        self.video_heatmap = []
        
        # 视频保存相关
        self.save_video = cfg.save_video
        self.video_output_dir = cfg.video_output_dir
        
        self.indexes = None
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None
        self.launch_step_intervals = []
        self.active_env_mask = np.ones(self.num_envs, dtype=bool)
        self.per_env_inference_counts = np.zeros(self.num_envs, dtype=int)
        self.camera_projection = None
        self.latest_background_info = None

    def reset_state(self):
        """每个 Batch 开始前调用"""
        if self.next_chunk_future:
            self.next_chunk_future.cancel()
        for _, future in self.background_futures:
            future.cancel()
        self.next_chunk_future = None
        self.background_futures = []
        self.background_task_counter = 0
        self.latest_consumed_background_task_id = -1
        self.current_chunks = None
        self.chunk_index = 0
        # 重置视频帧列表
        self.video_images = []
        # 添加与 rollout.py 一致的 policy.reset() 调用
        self.policy.reset()
        if self.adaptive_chunker:
            with self.chunker_lock:
                self.adaptive_chunker.reset()
                self.current_chunk_size = self.adaptive_chunker.current_horizon
        else:
            self.current_chunk_size = self.n_action_steps
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None
        self.launch_step_intervals = []
        self.active_env_mask = np.ones(self.num_envs, dtype=bool)
        self.per_env_inference_counts = np.zeros(self.num_envs, dtype=int)
        self.latest_background_info = None

    def set_active_env_mask(self, active_mask):
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape[0] != self.num_envs:
            raise ValueError(f"active_mask length {active_mask.shape[0]} != num_envs {self.num_envs}")
        self.active_env_mask = active_mask

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

        video_path = video_dir / f"{filename}.mp4"
        frames = [np.asarray(img, dtype=np.uint8) for img in self.video_images]

        writer_used = None
        try:
            import imageio.v2 as imageio

            imageio.mimwrite(str(video_path), frames, fps=self.cfg.fps)
            writer_used = "imageio"
        except Exception:
            try:
                import mediapy as media

                media.write_video(str(video_path), frames, fps=self.cfg.fps)
                writer_used = "mediapy"
            except Exception:
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(video_path), fourcc, self.cfg.fps, (width, height))
                for img in frames:
                    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
                    out.write(frame)
                out.release()
                writer_used = "cv2"

        logging.info(f"Video saved to {video_path}")
        logging.info(f"Video writer used: {writer_used}")
        self.video_images = []

    def _annotate_debug_frame(self, frame):
        annotated = frame.copy()
        info = self.latest_background_info or {}
        gripper_box = info.get("gripper_box")
        if gripper_box is None:
            return annotated

        try:
            height, width = annotated.shape[:2]
            x1 = int(float(gripper_box[0]) * width)
            y1 = int(float(gripper_box[1]) * height)
            x2 = int(float(gripper_box[2]) * width)
            y2 = int(float(gripper_box[3]) * height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            contact_phrases = info.get("gripper_contact_phrases") or []
            if contact_phrases:
                cv2.putText(
                    annotated,
                    f"gripper: {','.join(contact_phrases)}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        except Exception:
            return frame
        return annotated

    def _project_gripper_box_from_obs(self, raw_obs: Dict[str, Any]):
        if self.camera_projection is None or "agent_pos" not in raw_obs:
            return None
        try:
            from robosuite.utils.camera_utils import project_points_from_world_to_camera

            agent_pos = raw_obs["agent_pos"][0]
            point = np.array([[agent_pos[0], agent_pos[1], agent_pos[2]]], dtype=np.float32)
            transform = np.array(self.camera_projection["camera_world_to_image"], dtype=np.float32)
            height, width = self.camera_projection["camera_resolution"]
            projected = project_points_from_world_to_camera(
                points=point,
                world_to_camera_transform=transform,
                camera_height=int(height),
                camera_width=int(width),
            )
            row = int(projected[0][0])
            col = int(projected[0][1])
            col = (int(width) - 1) - col
            half = 18
            x1 = max(0, col - half)
            y1 = max(0, row - half)
            x2 = min(int(width) - 1, col + half)
            y2 = min(int(height) - 1, row + half)
            return [
                x1 / max(int(width), 1),
                y1 / max(int(height), 1),
                x2 / max(int(width), 1),
                y2 / max(int(height), 1),
            ]
        except Exception:
            return None

    def _annotate_debug_frame_from_obs(self, frame, raw_obs: Dict[str, Any]):
        frame_info = dict(self.latest_background_info or {})
        current_gripper_box = self._project_gripper_box_from_obs(raw_obs)
        if current_gripper_box is not None:
            frame_info["gripper_box"] = current_gripper_box
        previous_info = self.latest_background_info
        self.latest_background_info = frame_info
        annotated = self._annotate_debug_frame(frame)
        self.latest_background_info = previous_info
        return annotated

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

    def _log_step_state(
        self,
        current_step: int,
        should_launch: bool,
        gripper_change: bool,
        background_info: Optional[Dict[str, Any]] = None,
    ):
        background_ready_count = sum(1 for _, future in self.background_futures if future.done())
        background_pending_count = sum(1 for _, future in self.background_futures if not future.done())
        background_factors = None
        if background_info:
            background_factors = background_info.get("factors")
        elif self.latest_background_info:
            background_factors = self.latest_background_info.get("factors")
        logging.info(
            "step=%s chunk_index=%s action_step=%s current_chunk_size=%s last_chunk_size=%s "
            "should_launch=%s gripper_change=%s next_chunk_pending=%s background_pending=%s "
            "background_ready=%s background_factors=%s",
            self.batch_env_step,
            self.chunk_index,
            current_step,
            self.current_chunk_size,
            self.last_chunk_size,
            should_launch,
            gripper_change,
            self.next_chunk_future is not None,
            background_pending_count > 0,
            background_ready_count > 0,
            background_factors,
        )

    def _inference_task(self, batch_obs_prepared):
        """在后台线程执行的推理函数"""
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch_obs_prepared)
            return actions.cpu().numpy() # [B, Horizon, ActionDim]

    def _background_adaptive_task(self, task_id, image, instruction, robot_state=None):
        """在后台线程执行的自适应chunk size调整"""
        if self.adaptive_chunker:
            with self.chunker_lock:
                _, info = self.adaptive_chunker.adaptive_chunk_heuristic(image, instruction, robot_state)
                return {
                    "task_id": task_id,
                    "info": info,
                    "current_horizon": self.adaptive_chunker.current_horizon,
                }
        return {}

    def _maybe_launch_background_adaptive(self, raw_obs: Dict[str, Any], task_description: str):
        """Queue one adaptive task for every step on a serial background worker."""
        if not (self.adaptive_chunker and "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict)):
            return

        image = raw_obs["pixels"]["image"][0]
        robot_state = None
        if "agent_pos" in raw_obs:
            agent_pos = raw_obs["agent_pos"][0]
            robot_state = {
                "ee_position": agent_pos[:3],
                "gripper_state": agent_pos[-1],
            }
            if self.camera_projection is not None:
                robot_state.update(self.camera_projection)
        task_id = self.background_task_counter
        self.background_task_counter += 1
        future = self.background_executor.submit(
            self._background_adaptive_task, task_id, image, task_description, robot_state
        )
        self.background_futures.append((task_id, future))

    def _consume_background_result_nonblocking(self):
        """Apply finished adaptive-task results without blocking the control loop."""
        if not self.background_futures:
            return None
        latest_result = None
        remaining = []
        for task_id, future in self.background_futures:
            if future.done():
                try:
                    result = future.result()
                    if result:
                        latest_result = result
                except Exception as exc:
                    logging.warning(f"Background adaptive task failed: {exc}")
            else:
                remaining.append((task_id, future))
        self.background_futures = remaining
        if latest_result is None:
            return None
        task_id = latest_result.get("task_id", -1)
        if task_id < self.latest_consumed_background_task_id:
            return None
        self.latest_consumed_background_task_id = task_id
        logging.info("Consuming adaptive task result task_id=%s", task_id)
        # self.current_chunk_size = latest_result.get("current_horizon", self.current_chunk_size)
        background_info = dict(latest_result.get("info") or {})
        background_info["background_task_id"] = task_id
        background_info["current_horizon"] = latest_result.get("current_horizon", self.current_chunk_size)
        return background_info

    def launch_next_inference(self, raw_obs: Dict[str, Any], task_description: str):
        """启动后台推理"""
        self.inference_times += 1
        self.per_env_inference_counts[self.active_env_mask] += 1
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
            # self.adaptive_chunker.action_history = []
            if reason is False and self.cfg.enable_gripper_interruption:
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
                self.video_images.append(
                    self._annotate_debug_frame_from_obs(raw_obs["pixels"]['image'][0], raw_obs)
                )
        should_launch = False
        gripper_change = False
        self._maybe_launch_background_adaptive(raw_obs, task_description)
        
        # 第一次和后面的逻辑合一了
        background_info = None
        if self.current_chunks is None:
            if self.next_chunk_future is None:
                self.launch_next_inference(raw_obs, task_description)
                self.chunk_index = 0
            else:
                self.chunk_index = self.overlap_steps                
            self.reset_chunk()
            
            # 检查后台自适应任务是否完成
            final_chunk = None
            background_info = self._consume_background_result_nonblocking()
            if background_info is not None:
                self.latest_background_info = background_info
                final_chunk = background_info.get("current_horizon")
                logging.info(f"background horizon: {final_chunk}")
            
            # 获取注意力图并使用它进一步调整chunk size (只有在policy真正做chunk推理时才使用ValueFunc)
            if self.adaptive_chunker:
                try:
                    attn_maps = self.policy.get_attn_maps(layers=self.target_layers, heads=range(8), force=True)
                    if attn_maps:
                        vla_attention_map = attn_maps #[layers, num_heads, seq_len]
                        with self.chunker_lock:
                            _, info = self.adaptive_chunker.ValueFunc(
                                raw_obs["pixels"]['image'][0], 
                                task_description, 
                                vla_attention_map
                            )
                            attention_chunk_size = self.adaptive_chunker.current_horizon
                            if final_chunk is not None:
                                final_chunk = min(attention_chunk_size, final_chunk)
                            else:
                                final_chunk = attention_chunk_size
                        # 立马根据Value func更新last_chunk_size，仅用当前指定的chunk size
                        # self.last_chunk_size = min(attention_chunk_size, self.last_chunk_size)
                        logging.info(f"Attention-based chunk size: {attention_chunk_size}")
                except Exception as e:
                    import traceback
                    logging.warning(f"Failed to get attention maps: {e}")
                    traceback.print_exc()
            if final_chunk is not None:
                self.current_chunk_size = final_chunk
                # self.last_chunk_size = max(self.last_chunk_size, self.current_chunk_size)
                logging.info(f"Final chunk size: {self.current_chunk_size}")
        else:
            # 检查后台自适应任务是否完成
            background_info = self._consume_background_result_nonblocking()
            if background_info is not None:
                self.current_chunk_size = background_info.get("current_horizon")
                if self.adaptive_chunker.needs_interruption:
                    logging.info(f"Interrupting current chunk due to: {background_info.get('factors', 'N/A')}")
                    # should_launch = True
                if self.cfg.enable_gripper_interruption and self.should_interrupt_gripper():
                    should_launch = True
                    gripper_change = True

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
                logging.info(f"Manager: Interrupting current chunk with chunk size: {self.current_chunk_size}")    
                self.launch_next_inference(raw_obs, task_description)
                # 还要继续执行吗 执行的话就是index继续增加, 欸但是貌似这个没有起作用吧
                # 因为后续chunk size可能增加会来了，那还是打断吧
                
            self.reset_chunk(gripper_change)
            self.chunk_index = 0
        
        # 确保不超出当前chunk size
        current_step = min(self.chunk_index, min(self.current_chunk_size, self.n_action_steps, self.last_chunk_size) - 1)
        actions = self.current_chunks[:, current_step, :] # [B, ActionDim]
        # print("gripper action: ", actions[0, -1])

        if self.adaptive_chunker:
            with self.chunker_lock:
                self.adaptive_chunker.enqueue_history(actions[0].tolist())

        self._log_step_state(
            current_step=current_step,
            should_launch=should_launch,
            gripper_change=gripper_change,
            background_info=background_info,
        )

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
    async_mgr.camera_projection = _extract_camera_projection(vec_env)

    total_episodes = cfg.max_episodes
    n_batches = (total_episodes + cfg.eval_batch_size - 1) // cfg.eval_batch_size
    
    successes = []
    episode_steps = []
    per_episode_inference_counts = []
    total_inference_times = 0
    all_launch_intervals = []
    launch_interval_batches = []
    env_step_times_ms = []
    control_sleep_times_ms = []
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
            async_mgr.set_active_env_mask(~dones)
            
            # 使用异步管理器直接获取 Batch Actions
            # 它内部会自动处理：
            # 1. 什么时候该在后台跑推理
            # 2. 什么时候该等待推理结果并切换 Chunk
            # 3. 后台运行Grounding DINO并调整chunk size
            # 4. 根据attention map进一步调整chunk size
            actions = async_mgr.get_action_batch(observation, task_description)
            
            step_start = time.perf_counter()
            observation, reward, terminated, truncated, info = vec_env.step(actions)
            env_step_elapsed = time.perf_counter() - step_start
            env_step_times_ms.append(env_step_elapsed * 1000.0)

            if cfg.target_control_hz is not None:
                target_period_s = 1.0 / cfg.target_control_hz
                extra_sleep_s = max(0.0, target_period_s - env_step_elapsed)
                if extra_sleep_s > 0:
                    time.sleep(extra_sleep_s)
                control_sleep_times_ms.append(extra_sleep_s * 1000.0)
            
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
        batch_episode_count = min(cfg.eval_batch_size, total_episodes - len(per_episode_inference_counts))
        per_episode_inference_counts.extend(
            async_mgr.per_env_inference_counts[:batch_episode_count].tolist()
        )
        total_inference_times += async_mgr.inference_times
        all_launch_intervals.extend(async_mgr.launch_step_intervals)
        launch_interval_batches.append(list(async_mgr.launch_step_intervals))

        # 保存视频
        if cfg.save_video:
            async_mgr.save_video_frames(f"{task_name}_env0_batch{b}")

    if all_launch_intervals:
        dist = Counter(all_launch_intervals)
        launch_interval_stats = {
            "count": len(all_launch_intervals),
            "distribution": dict(sorted(dist.items())),
            "mean": float(np.mean(all_launch_intervals)),
            "min": int(min(all_launch_intervals)),
            "max": int(max(all_launch_intervals)),
            "intervals_by_batch": launch_interval_batches,
        }
    else:
        launch_interval_stats = {
            "count": 0,
            "distribution": {},
            "mean": 0.0,
            "min": None,
            "max": None,
            "intervals_by_batch": [],
        }

    if cfg.use_monitor:
        monitor.stop()
        stats = monitor.get_report() 
        logging.info(f"inference times: {total_inference_times}")
        logging.info(f"GPU Monitor Stats: {stats}")
    else:
        logging.info(f"inference times: {total_inference_times}")
    
    successes = successes[:total_episodes]
    
    # from visualutils import heatmap_videos
    # heatmap_videos(async_mgr.video_images, async_mgr.video_heatmap, filename='visualizations/'+task_name+".mp4")
    episode_steps = episode_steps[:total_episodes]
    per_episode_inference_counts = per_episode_inference_counts[:total_episodes]
    successful_steps_list = [steps for steps, success in zip(episode_steps, successes) if success]
    successful_inference_counts = [
        count for count, success in zip(per_episode_inference_counts, successes) if success
    ]
    return {
        "sr": np.mean(successes),
        "count": len(successes),
        "success_list": successes,
        "steps_list": episode_steps,
        "avg_steps": np.mean(episode_steps) if episode_steps else 0,
        "successful_steps_list": successful_steps_list,
        "avg_successful_steps": (float(np.mean(successful_steps_list)) if successful_steps_list else None),
        "inference_times": total_inference_times,
        "per_episode_inference_counts": per_episode_inference_counts,
        "successful_inference_counts": successful_inference_counts,
        "avg_successful_inference_times": (
            float(np.mean(successful_inference_counts)) if successful_inference_counts else None
        ),
        "env_step_timing": {
            "count": len(env_step_times_ms),
            "target_control_hz": cfg.target_control_hz,
            "avg_env_step_ms": (float(np.mean(env_step_times_ms)) if env_step_times_ms else None),
            "min_env_step_ms": (float(np.min(env_step_times_ms)) if env_step_times_ms else None),
            "max_env_step_ms": (float(np.max(env_step_times_ms)) if env_step_times_ms else None),
            "avg_control_sleep_ms": (float(np.mean(control_sleep_times_ms)) if control_sleep_times_ms else None),
            "total_control_sleep_ms": float(np.sum(control_sleep_times_ms)) if control_sleep_times_ms else 0.0,
        },
        "launch_interval_stats": launch_interval_stats,
        "adaptive_chunker_runtime_config": (
            async_mgr.adaptive_chunker.get_runtime_config()
            if async_mgr.adaptive_chunker is not None and hasattr(async_mgr.adaptive_chunker, "get_runtime_config")
            else None
        ),
    }
    
def run_benchmark_common(cfg: VLASHEvalConfig):
    init_logging()
    _configure_component_log_levels(cfg)
    set_seed(cfg.seed)
    _log_effective_eval_setup(cfg)
    
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
    launch_interval_results = {}
    overall_successes = []
    suite_successful_inference_counts = defaultdict(list)
    shared_chunker_runtime_config = None
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
            if shared_chunker_runtime_config is None and cfg.use_adaptive_chunk:
                shared_chunker_runtime_config = task_info.get("adaptive_chunker_runtime_config")
            
            results_per_task[f"{suite_name}_{task_id}"] = {
                "task_description": task_description,
                "success_rate": task_info["sr"],
                "num_episodes": task_info["count"],
                "avg_steps": task_info["avg_steps"],
                "avg_successful_steps": task_info["avg_successful_steps"],
                "steps_list": task_info["steps_list"],
                "successful_steps_list": task_info["successful_steps_list"],
                "inference_times": task_info["inference_times"],
                "env_step_timing": task_info["env_step_timing"],
                "per_episode_inference_counts": task_info["per_episode_inference_counts"],
                "successful_inference_counts": task_info["successful_inference_counts"],
                "avg_successful_inference_times": task_info["avg_successful_inference_times"],
            }
            launch_interval_results[f"{suite_name}_{task_id}"] = task_info["launch_interval_stats"]
            overall_successes.extend(task_info["success_list"])
            suite_successful_inference_counts[suite_name].extend(task_info["successful_inference_counts"])
            
            logging.info(f"Task Done. SR: {task_info['sr']*100:.1f}%")

    # 4. 汇总与保存
    total_time = time.time() - start_time
    
    # 收集所有episode的步骤
    all_steps = []
    all_successful_steps = []
    for task_result in results_per_task.values():
        all_steps.extend(task_result["steps_list"])
        all_successful_steps.extend(task_result["successful_steps_list"])
    
    runtime_config = _collect_eval_runtime_config(cfg, None)
    if shared_chunker_runtime_config is not None:
        runtime_config["adaptive_chunker"] = shared_chunker_runtime_config

    final_results = {
        "overall": {
            "avg_success_rate": np.mean(overall_successes),
            "total_episodes": len(overall_successes),
            "total_eval_time_s": total_time,
            "env_task": cfg.env_task,
            "use_adaptive_chunk": cfg.use_adaptive_chunk,
            "avg_steps_per_episode": np.mean(all_steps) if all_steps else 0,
            "avg_steps_per_successful_episode": (
                float(np.mean(all_successful_steps)) if all_successful_steps else None
            ),
            "total_steps": sum(all_steps) if all_steps else 0
        },
        "per_task": results_per_task,
        "per_suite": {
            suite_name: {
                "successful_episode_count": len(counts),
                "avg_successful_steps": (
                    float(np.mean(successful_steps))
                    if (successful_steps := [
                        step
                        for task_key, task_result in results_per_task.items()
                        if task_key.startswith(f"{suite_name}_")
                        for step in task_result["successful_steps_list"]
                    ])
                    else None
                ),
                "avg_successful_inference_times": (float(np.mean(counts)) if counts else None),
                "successful_inference_counts": counts,
            }
            for suite_name, counts in suite_successful_inference_counts.items()
        },
        "runtime_config": runtime_config,
    }
    
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    with open(out_dir / "launch_interval_stats.json", "w") as f:
        json.dump(launch_interval_results, f, indent=2)
    _write_yaml_file(out_dir / "runtime_config.yaml", runtime_config)
    
    logging.info("="*50)
    close_envs(all_envs_dict)
    return {
        "final_results": final_results,
        "out_dir": out_dir,
        "results_per_task": results_per_task,
        "overall_successes": overall_successes,
        "all_steps": all_steps,
        "all_successful_steps": all_successful_steps,
        "suite_successful_inference_counts": dict(suite_successful_inference_counts),
    }


@parser.wrap()
def run_benchmark(cfg: VLASHEvalConfig):
    run_benchmark_common(cfg)
    
    # entropy_dict = policy.get_attention_entropies()
    # # 保存
    # with open(out_dir / (task_name+"entropy_dict.json"), "w") as f:
    #     json.dump(entropy_dict, f, indent=2)

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        traceback.print_exc()
