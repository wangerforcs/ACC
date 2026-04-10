#!/usr/bin/env python3
"""
基于evalpi05.py的自适应chunk size版本
使用adaptivepolicy.py中的结构来实现运行时改变chunk size
"""

import concurrent.futures
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
from adaptive.motion_estimators import (
    project_world_point_to_observed_pixel,
    reconstruct_box_center_world_point,
)


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


def _unwrap_to_inner_problem_env(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if (
            hasattr(current, "sim")
            and (
                hasattr(current, "get_dynamic_object_position")
                or hasattr(current, "objects_dict")
                or hasattr(current, "moving_object_name")
            )
        ):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        if hasattr(current, "_env"):
            current = current._env
            continue
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
            continue
        break
    return None


def _canonical_object_key(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("-", "_").replace(" ", "_")
    parts = [part for part in normalized.split("_") if part]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _extract_sim_gt_positions(
    vec_env: gym.vector.VectorEnv,
    object_names: list[str] | tuple[str, ...] | None,
):
    if not hasattr(vec_env, "envs") or not vec_env.envs:
        return {}

    libero_env = _unwrap_to_libero_env(vec_env.envs[0])
    if libero_env is None:
        return {}

    try:
        base_env = _unwrap_to_inner_problem_env(libero_env)
        if base_env is None:
            logging.info("sim_gt inner env unresolved from libero wrapper")
            return {}
        sim = base_env.sim
        objects_dict = getattr(base_env, "objects_dict", {}) or {}
        positions = {}

        # Dynamic LIBERO env already knows which object is being scripted and how
        # to expose its root-body position. Prefer that over guessing from raw
        # MuJoCo body names.
        get_dynamic_object_position = getattr(base_env, "get_dynamic_object_position", None)
        moving_object_name = getattr(base_env, "moving_object_name", None)
        if callable(get_dynamic_object_position):
            try:
                dynamic_pos = get_dynamic_object_position()
            except Exception:
                dynamic_pos = None
            if dynamic_pos is not None and moving_object_name:
                dynamic_pos = np.asarray(dynamic_pos, dtype=np.float32).reshape(-1)
                if dynamic_pos.shape[0] >= 3 and np.isfinite(dynamic_pos[:3]).all():
                    positions[str(moving_object_name)] = dynamic_pos[:3]
                    canonical = _canonical_object_key(moving_object_name)
                    if canonical:
                        positions[canonical] = dynamic_pos[:3]
                    logging.debug(
                        "sim_gt dynamic object resolved via env API: moving_object_name=%s canonical=%s position=%s",
                        moving_object_name,
                        canonical,
                        dynamic_pos[:3].tolist(),
                    )

        for object_name, obj in objects_dict.items():
            root_body = getattr(obj, "root_body", None) if obj is not None else None
            if root_body is None:
                continue
            try:
                body_id = int(sim.model.body_name2id(root_body))
            except Exception:
                continue
            resolved_position = np.array(sim.data.body_xpos[body_id], dtype=np.float32)
            if resolved_position.shape[0] < 3 or not np.isfinite(resolved_position[:3]).all():
                continue
            positions[str(object_name)] = resolved_position[:3]
            object_key = _canonical_object_key(object_name)
            if object_key:
                positions[object_key] = resolved_position[:3]
            positions[str(root_body)] = resolved_position[:3]
            root_key = _canonical_object_key(root_body)
            if root_key:
                positions[root_key] = resolved_position[:3]

        for name in object_names or ():
            if not name:
                continue
            requested = str(name)
            requested_canonical = _canonical_object_key(requested)
            if requested in positions or (requested_canonical and requested_canonical in positions):
                continue

            candidate_object_names = [requested]
            if requested_canonical:
                for object_name in objects_dict.keys():
                    object_key = _canonical_object_key(object_name)
                    if object_key == requested_canonical and object_name not in candidate_object_names:
                        candidate_object_names.append(object_name)

            resolved_position = None
            resolved_name = None
            resolved_root_body = None
            for object_name in candidate_object_names:
                if object_name in positions:
                    resolved_position = np.asarray(positions[object_name], dtype=np.float32)
                    resolved_name = object_name
                    candidate_obj = objects_dict.get(object_name)
                    resolved_root_body = getattr(candidate_obj, "root_body", None) if candidate_obj is not None else None
                    break

            if resolved_position is None:
                logging.info(
                    "sim_gt object unresolved: requested=%s requested_canonical=%s objects_dict_keys_sample=%s",
                    requested,
                    requested_canonical,
                    list(objects_dict.keys())[:20],
                )
                continue

            positions[requested] = resolved_position
            if requested_canonical:
                positions[requested_canonical] = resolved_position
            if resolved_name is not None:
                positions[resolved_name] = resolved_position
                resolved_name_key = _canonical_object_key(resolved_name)
                if resolved_name_key:
                    positions[resolved_name_key] = resolved_position
            if resolved_root_body is not None:
                positions[str(resolved_root_body)] = resolved_position
                resolved_root_key = _canonical_object_key(resolved_root_body)
                if resolved_root_key:
                    positions[resolved_root_key] = resolved_position
            logging.debug(
                "sim_gt object resolved via objects_dict: requested=%s resolved_name=%s resolved_root_body=%s position=%s",
                requested,
                resolved_name,
                resolved_root_body,
                resolved_position[:3].tolist(),
            )
        return positions
    except Exception as exc:
        logging.warning(f"Failed to extract sim ground-truth positions: {exc}")
        return {}


def _extract_camera_projection(vec_env: gym.vector.VectorEnv, camera_name: str = "agentview"):
    if not hasattr(vec_env, "envs") or not vec_env.envs:
        return None

    libero_env = _unwrap_to_libero_env(vec_env.envs[0])
    if libero_env is None:
        return None

    try:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_camera_transform_matrix,
        )
    except Exception as exc:
        logging.warning(f"Failed to import robosuite camera utils: {exc}")
        return None

    try:
        base_env = _unwrap_to_inner_problem_env(libero_env)
        if base_env is None:
            return None
        sim = base_env.sim
        height = int(getattr(libero_env, "observation_height", 256))
        width = int(getattr(libero_env, "observation_width", 256))
        env_control_freq = float(getattr(base_env, "control_freq", getattr(libero_env, "control_freq", 20.0)))
        camera_intrinsics = get_camera_intrinsic_matrix(
            sim=sim,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        )
        camera_to_world = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
        world_to_image = get_camera_transform_matrix(
            sim=sim,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        )
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear * extent)
        far = float(sim.model.vis.map.zfar * extent)
        return {
            "camera_name": camera_name,
            "camera_intrinsics": np.array(camera_intrinsics, dtype=np.float32),
            "camera_to_world": np.array(camera_to_world, dtype=np.float32),
            "camera_world_to_image": np.array(world_to_image, dtype=np.float32),
            "camera_resolution": (height, width),
            "camera_depth_near": near,
            "camera_depth_far": far,
            "camera_depth_normalized": True,
            "env_control_freq": env_control_freq,
            "env_control_dt": 1.0 / max(env_control_freq, 1.0),
            # LeRobot LIBERO flips observations with image[::-1, ::-1]. Existing
            # projection utilities are already row-compatible, so we only need
            # to undo the horizontal mirror when back-projecting from observed
            # pixels into 3D.
            "camera_observation_mirror_x": True,
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
    episode_length: int = 280
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
    dino_refresh_interval: int = 400
    
    # 视频保存配置
    save_video: bool = False
    video_output_dir: str = "fuckvideos"
    video_overlay_debug: bool = False
    
    use_monitor: bool = False
    eval_log_level: str = "INFO"
    # tracker_log_level: str = "WARNING"
    tracker_log_level: str = "INFO"
    scoregate_log_level: str = "WARNING"
    enable_gripper_interruption: bool = False
    enable_object_motion_penalty: bool = False
    object_motion_metric: str = "bbox2d"
    object_motion_speed_threshold: float = 0.03
    object_motion_speed_exit_threshold: float = 0.015
    object_motion_fast_speed_threshold: float = 0.08
    object_motion_depth_inner_ratio: float = 0.6
    object_motion_depth_min_valid_ratio: float = 0.2
    enable_small_chunk_action_skip: bool = False
    small_chunk_action_skip_threshold: int = 5
    small_chunk_action_skip_offset: int = 1
    small_chunk_action_skip_max_gripper_delta: float = 0.5
    adaptive_chunker_cls_path: str = ( #默认是finegrained_tracker_scoregate
        "adaptive.adaptivepolicyfinegrained_tracker_scoregate:AdaptiveActionChunker"
    )
    tracker_logger_names: List[str] = field(
        default_factory=lambda: ["adaptive.tracker", "adaptive.tracker_scoregate"]
    )
    
    def __post_init__(self):
        if self.target_control_hz is not None and self.target_control_hz <= 0:
            raise ValueError("target_control_hz must be positive when provided")
        if self.target_control_hz is not None:
            self.fps = int(round(float(self.target_control_hz)))
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
                logging.info(f"Loading policy config from {policy_path} with CLI overrides: {cli_overrides}")
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
        "episode_length": cfg.episode_length,
        "use_adaptive_chunk": cfg.use_adaptive_chunk,
        "enable_gripper_interruption": cfg.enable_gripper_interruption,
        "enable_object_motion_penalty": cfg.enable_object_motion_penalty,
        "object_motion_metric": cfg.object_motion_metric,
        "object_motion_speed_threshold": cfg.object_motion_speed_threshold,
        "object_motion_speed_exit_threshold": cfg.object_motion_speed_exit_threshold,
        "object_motion_fast_speed_threshold": cfg.object_motion_fast_speed_threshold,
        "object_motion_depth_inner_ratio": cfg.object_motion_depth_inner_ratio,
        "object_motion_depth_min_valid_ratio": cfg.object_motion_depth_min_valid_ratio,
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
        "video_overlay_debug": cfg.video_overlay_debug,
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
    logging.info(f"Enable object motion penalty: {cfg.enable_object_motion_penalty}")
    logging.info(f"Object motion metric: {cfg.object_motion_metric}")
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
                enable_object_motion_penalty=cfg.enable_object_motion_penalty,
                dynamic_motion_object_name=getattr(cfg, "dynamic_moving_object_name", None),
                object_motion_metric=cfg.object_motion_metric,
                object_motion_speed_threshold=cfg.object_motion_speed_threshold,
                object_motion_speed_exit_threshold=cfg.object_motion_speed_exit_threshold,
                object_motion_fast_speed_threshold=cfg.object_motion_fast_speed_threshold,
                object_motion_control_dt=1.0 / max(cfg.target_control_hz, 1e-6),
                object_motion_depth_inner_ratio=cfg.object_motion_depth_inner_ratio,
                object_motion_depth_min_valid_ratio=cfg.object_motion_depth_min_valid_ratio,
                dino_refresh_interval=cfg.dino_refresh_interval,
            )
            # 初始chunk size
            self.current_chunk_size = self.adaptive_chunker.current_horizon
        else:
            self.adaptive_chunker = None
            self.current_chunk_size = self.n_action_steps
        self.active_chunk_size = self.current_chunk_size
        self.valid_chunk_size = self.current_chunk_size
        self.last_chunk_size = self.current_chunk_size
        
        # 推理和 background 分开，background 串行排队，保证“每步都提交”同时不并发写 chunker
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.value_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.next_chunk_future = None
        self.background_futures = []
        self.value_futures = []
        self.background_task_counter = 0
        self.latest_consumed_background_task_id = -1
        self.value_task_counter = 0
        self.latest_consumed_value_task_id = -1
        
        # 状态追踪
        self.current_chunks = None  # 形状: [Batch, Horizon, ActionDim]
        self.chunk_index = 0
        
        # 校验
        assert self.n_action_steps > self.overlap_steps
        
        self.last_observation_imgs = None
        self.save_attention_step = 5
        
        self.video_images = []
        self.target_layers = [9, 10, 11, 12]
        # self.target_layers = [6, 11, 12, 17]
        self.video_heatmap = []
        
        # 视频保存相关
        self.save_video = cfg.save_video
        self.video_output_dir = cfg.video_output_dir
        self.video_overlay_debug = bool(cfg.video_overlay_debug)
        
        self.indexes = None
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None
        self.launch_step_intervals = []
        self.active_env_mask = np.ones(self.num_envs, dtype=bool)
        self.per_env_inference_counts = np.zeros(self.num_envs, dtype=int)
        self.camera_projection = None
        self.image_depth = None
        self.vec_env = None
        self.current_task_description = task_name
        self.latest_background_info = None

    def _update_latest_background_info(self, background_info: Optional[Dict[str, Any]]) -> None:
        if not background_info:
            return
        merged = dict(self.latest_background_info or {})
        for key, value in background_info.items():
            if value is not None:
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        self.latest_background_info = merged

    def _drop_stale_background_futures(self) -> None:
        """Keep the single newest background task and discard stale queued ones.

        Background heuristic tasks can queue up behind expensive DINO refreshes.
        For control we care much more about the newest observation than about
        consuming every intermediate heuristic result in order.
        """
        if len(self.background_futures) <= 1:
            return

        newest_task_id, newest_future = self.background_futures[-1]
        kept: list[tuple[int, concurrent.futures.Future]] = []

        for task_id, future in self.background_futures[:-1]:
            if future.done():
                # Let the normal consume path decide whether to use/drop it.
                kept.append((task_id, future))
                continue
            if future.cancel():
                logging.info(
                    "Dropping stale background task before execution: task_id=%s (latest=%s)",
                    task_id,
                    newest_task_id,
                )
            else:
                # Already running; can't cancel, so keep it.
                kept.append((task_id, future))

        kept.append((newest_task_id, newest_future))
        self.background_futures = kept

    def reset_state(self):
        """每个 Batch 开始前调用"""
        if self.next_chunk_future:
            self.next_chunk_future.cancel()
        for _, future in self.background_futures:
            future.cancel()
        for _, future in self.value_futures:
            future.cancel()
        self.next_chunk_future = None
        self.background_futures = []
        self.value_futures = []
        self.background_task_counter = 0
        self.latest_consumed_background_task_id = -1
        self.value_task_counter = 0
        self.latest_consumed_value_task_id = -1
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
        self.active_chunk_size = self.current_chunk_size
        self.valid_chunk_size = self.current_chunk_size
        self.last_chunk_size = self.current_chunk_size
        self.inference_times = 0
        self.batch_env_step = 0
        self.last_launch_step = None
        self.launch_step_intervals = []
        self.active_env_mask = np.ones(self.num_envs, dtype=bool)
        self.per_env_inference_counts = np.zeros(self.num_envs, dtype=int)
        self.latest_background_info = None

    def shutdown(self):
        """显式关闭后台线程池，避免在解释器退出阶段留下悬挂 worker。"""
        try:
            if self.next_chunk_future:
                self.next_chunk_future.cancel()
        except Exception:
            pass
        for _, future in self.background_futures:
            try:
                future.cancel()
            except Exception:
                pass
        for _, future in self.value_futures:
            try:
                future.cancel()
            except Exception:
                pass
        for executor in (self.executor, self.background_executor, self.value_executor):
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

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
        try:
            height, width = annotated.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            small_font_scale = 0.36
            tiny_font_scale = 0.32
            text_thickness = 1
            box_thickness = 1
            phrases = info.get("phrases") or []
            boxes = info.get("dino_boxes") or []

            def _norm_to_px(box):
                x1 = int(float(box[0]) * width)
                y1 = int(float(box[1]) * height)
                x2 = int(float(box[2]) * width)
                y2 = int(float(box[3]) * height)
                return x1, y1, x2, y2

            for idx, box in enumerate(boxes):
                if box is None or len(box) < 4:
                    continue
                phrase = phrases[idx] if idx < len(phrases) else f"obj_{idx}"
                x1, y1, x2, y2 = _norm_to_px(box)
                color = (255, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
                cv2.putText(
                    annotated,
                    phrase[:48],
                    (x1, max(12, y1 - 4)),
                    font,
                    small_font_scale,
                    color,
                    text_thickness,
                    cv2.LINE_AA,
                )

            object_points_pixel_by_phrase = info.get("object_points_pixel_by_phrase") or {}
            object_points_world_by_phrase = info.get("object_points_world_by_phrase") or {}
            for phrase, pixel in object_points_pixel_by_phrase.items():
                if pixel is None or len(pixel) < 2:
                    continue
                col = int(pixel[0])
                row = int(pixel[1])
                cv2.circle(annotated, (col, row), 3, (0, 0, 255), -1)
                coords = object_points_world_by_phrase.get(phrase)
                if coords is not None and len(coords) >= 3:
                    coord_text = f"{phrase[:18]} ({coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f})"
                else:
                    coord_text = phrase[:24]
                text_y = min(max(12, row - 6), height - 6)
                text_x = min(max(0, col + 6), max(0, width - 180))
                cv2.putText(
                    annotated,
                    coord_text,
                    (text_x, text_y),
                    font,
                    tiny_font_scale,
                    (0, 0, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

            gripper_box = info.get("gripper_box")
            if gripper_box is not None:
                x1, y1, x2, y2 = _norm_to_px(gripper_box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            cv2.putText(
                annotated,
                f"step={info.get('step', self.batch_env_step)}",
                (8, 14),
                font,
                tiny_font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        except Exception:
            return frame
        return annotated

    def _compute_frame_object_points(self, raw_obs: Dict[str, Any], frame_info: Dict[str, Any]) -> None:
        if self.camera_projection is None:
            return
        if "pixels" not in raw_obs or not isinstance(raw_obs["pixels"], dict):
            return
        depth = raw_obs["pixels"].get("image_depth")
        if depth is None:
            return
        phrases = frame_info.get("phrases") or []
        boxes = frame_info.get("dino_boxes") or []
        if not phrases or not boxes:
            return

        depth_map = np.asarray(depth[0], dtype=np.float32)
        robot_state = dict(self.camera_projection)
        object_points_world_by_phrase = {}
        object_points_pixel_by_phrase = {}
        point_meta_by_phrase = {}
        for idx, phrase in enumerate(phrases):
            if idx >= len(boxes):
                continue
            box = boxes[idx]
            if box is None or len(box) < 4:
                continue
            point_world, meta = reconstruct_box_center_world_point(
                box,
                depth_map,
                robot_state,
                depth_inner_ratio=self.cfg.object_motion_depth_inner_ratio,
                depth_min_valid_ratio=self.cfg.object_motion_depth_min_valid_ratio,
            )
            if point_world is None:
                continue
            pixel = project_world_point_to_observed_pixel(point_world, robot_state)
            object_points_world_by_phrase[phrase] = np.asarray(point_world, dtype=np.float32).tolist()
            if pixel is not None:
                object_points_pixel_by_phrase[phrase] = np.asarray(pixel, dtype=np.int32).tolist()
            point_meta_by_phrase[phrase] = dict(meta)

        if object_points_world_by_phrase:
            frame_info["object_points_world_by_phrase"] = object_points_world_by_phrase
        if object_points_pixel_by_phrase:
            frame_info["object_points_pixel_by_phrase"] = object_points_pixel_by_phrase
        if point_meta_by_phrase:
            frame_info["object_points_meta_by_phrase"] = point_meta_by_phrase

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
            if self.camera_projection.get("camera_observation_mirror_x", False):
                col = (int(width) - 1) - col
            if self.camera_projection.get("camera_observation_mirror_y", False):
                row = (int(height) - 1) - row
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
        if not (self.video_overlay_debug or self.save_video):
            return frame
        frame_info = dict(self.latest_background_info or {})
        cached_boxes = getattr(self.adaptive_chunker, "cached_boxes", None) if self.adaptive_chunker is not None else None
        if cached_boxes is not None and len(cached_boxes) >= 4:
            boxes_xyxy = cached_boxes[1]
            phrases = cached_boxes[3]
            if boxes_xyxy is not None:
                try:
                    if hasattr(boxes_xyxy, "detach"):
                        boxes_xyxy = boxes_xyxy.detach().cpu().tolist()
                    else:
                        boxes_xyxy = np.asarray(boxes_xyxy).tolist()
                    frame_info["dino_boxes"] = boxes_xyxy
                    frame_info["phrases"] = list(phrases) if phrases is not None else []
                    frame_info.setdefault("box_source", "tracker_cache")
                except Exception:
                    pass
        current_gripper_box = self._project_gripper_box_from_obs(raw_obs)
        if current_gripper_box is not None:
            frame_info["gripper_box"] = current_gripper_box
        self._compute_frame_object_points(raw_obs, frame_info)
        frame_info["step"] = self.batch_env_step
        tracker_untrusted = getattr(self.adaptive_chunker, "tracker_untrusted_by_phrase", None)
        if tracker_untrusted:
            frame_info["tracker_untrusted_phrases"] = [phrase for phrase, flagged in tracker_untrusted.items() if flagged]
        previous_info = self.latest_background_info
        self.latest_background_info = frame_info
        annotated = self._annotate_debug_frame(frame)
        self.latest_background_info = previous_info
        return annotated

    def _prepare_batch_obs(self, raw_obs: Dict[str, Any], task_description: str):
        """
        将 VectorEnv 的原始观测转换为 Policy 需要的 Batch Tensor 格式
        """
        # 0.4.1是agent_pos, 0,4,4是 robot_state字典，我觉得这里似乎
        # 注意：VectorEnv 返回的 raw_obs 已经是 [B, ...] 格式了
        
        batch_input = {}
        
        if "agent_pos" in raw_obs:
            # 统一都是xyz(3) angles(3) gripper(2) 不管实际返回什么，模型需要的事这个
            batch_input["observation.state"] = torch.from_numpy(raw_obs["agent_pos"]).float().to(self.device)
        
        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            for k, v in raw_obs["pixels"].items():
                '''
                image (1, 256, 256, 3) image2 (1, 256, 256, 3)
                '''
                # v 形状是 [B, H, W, C] -> 需要转换为 [B, C, H, W] 并归一化
                if "depth" in k:
                    continue
                img_tensor = torch.from_numpy(v).float().permute(0, 3, 1, 2).to(self.device) / 255.0
                batch_input[f"observation.images.{k}"] = img_tensor
        
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
            "active_chunk_size=%s "
            "should_launch=%s gripper_change=%s next_chunk_pending=%s background_pending=%s "
            "background_ready=%s background_factors=%s",
            self.batch_env_step,
            self.chunk_index,
            current_step,
            self.current_chunk_size,
            self.last_chunk_size,
            self.active_chunk_size,
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
            # if self.cfg.use_adaptive_chunk:
            #     actions = self.policy.predict_action_chunk(batch_obs_prepared)
            # else:
            #     actions_shape = (
            #         batch_obs_prepared["observation.state"].shape[0],
            #         self.policy.config.n_action_steps,
            #         self.policy.config.max_action_dim,
            #     )
            #     noise = self.policy.model.sample_noise(actions_shape, device=batch_obs_prepared["observation.state"].device)
            #     actions = self.policy.predict_action_chunk(batch_obs_prepared, noise=noise)
            return actions.cpu().numpy() # [B, Horizon, ActionDim]

    def _background_adaptive_task(self, task_id, image, instruction, robot_state=None):
        """在后台线程执行的自适应chunk size调整"""
        if self.adaptive_chunker:
            proposal = self.adaptive_chunker.propose_adaptive_chunk_heuristic(
                image,
                instruction,
                robot_state,
                current_horizon=self.current_chunk_size,
            )
            proposal = self._normalize_chunk_proposal(proposal)
            return {
                "task_id": task_id,
                "info": dict(proposal.get("info") or {}),
                "current_horizon": proposal.get("current_horizon", self.current_chunk_size),
                "needs_interruption": bool(proposal.get("needs_interruption", False)),
            }
        return {}

    def _value_func_task(self, task_id, image, instruction, attention_map, value_context):
        if self.adaptive_chunker is None:
            return {}
        proposal = self.adaptive_chunker.propose_value_horizon(
            image,
            instruction,
            attention_map,
            parsed_objs=value_context.get("target_object"),
            boxes_xyxy=value_context.get("dino_boxes"),
            phrases=value_context.get("phrases"),
            continuity_info=value_context.get("continuity_info"),
            current_horizon=value_context.get("current_horizon", self.current_chunk_size),
            box_source=value_context.get("box_source"),
        )
        proposal = self._normalize_chunk_proposal(proposal)
        return {
            "task_id": task_id,
            "info": dict(proposal.get("info") or {}),
            "current_horizon": proposal.get("current_horizon", self.current_chunk_size),
        }

    def _normalize_chunk_proposal(self, proposal):
        if proposal is None:
            return {}
        if isinstance(proposal, dict):
            return proposal
        if isinstance(proposal, tuple) and len(proposal) == 2:
            horizon, info = proposal
            return {
                "current_horizon": horizon,
                "info": dict(info or {}),
            }
        raise TypeError(f"Unsupported chunk proposal type: {type(proposal)!r}")

    # 周期性的
    def _maybe_launch_background_adaptive(self, raw_obs: Dict[str, Any], task_description: str):
        """Queue one adaptive task for every step on a serial background worker."""
        if not (self.adaptive_chunker and "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict)):
            return

        image = raw_obs["pixels"]["image"][0]
        self.image_depth = raw_obs["pixels"].get("image_depth")
        if self.image_depth is not None:
            self.image_depth = self.image_depth[0]
        robot_state = None
        if "agent_pos" in raw_obs:
            agent_pos = raw_obs["agent_pos"][0]
            robot_state = {
                "ee_position": agent_pos[:3],
                "gripper_state": agent_pos[-1],
                "step_index": int(self.batch_env_step),
                "control_dt": 1.0 / max(self.cfg.target_control_hz, 1e-6),
            }
            if self.camera_projection is not None:
                robot_state.update(self.camera_projection)
            if self.image_depth is not None:
                robot_state["image_depth"] = self.image_depth
            sim_gt_object_names = []
            dynamic_name = getattr(self.cfg, "dynamic_moving_object_name", None)
            if dynamic_name:
                sim_gt_object_names.append(dynamic_name)
            if self.vec_env is not None and self.cfg.object_motion_metric == "sim_gt":
                sim_gt_positions = _extract_sim_gt_positions(self.vec_env, sim_gt_object_names)
                if sim_gt_positions:
                    robot_state["sim_gt_positions"] = sim_gt_positions
                    logging.debug(
                        "sim_gt robot_state prepared: requested=%s available_keys=%s",
                        sim_gt_object_names,
                        sorted(sim_gt_positions.keys()),
                    )
        task_id = self.background_task_counter
        self.background_task_counter += 1
        future = self.background_executor.submit(
            self._background_adaptive_task, task_id, image, task_description, robot_state
        )
        self.background_futures.append((task_id, future))
        self._drop_stale_background_futures()

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
        background_info["needs_interruption"] = bool(latest_result.get("needs_interruption", False))
        return background_info

    def _build_value_context(self, background_info: Optional[Dict[str, Any]] = None):
        source = background_info or self.latest_background_info or {}
        if not source:
            return None
        return {
            "target_object": source.get("target_object"),
            "dino_boxes": source.get("dino_boxes"),
            "phrases": source.get("phrases"),
            "continuity_info": source.get("continuity_info"),
            "box_source": source.get("box_source"),
            "current_horizon": source.get("current_horizon", self.current_chunk_size),
        }

    def _maybe_launch_value_func(self, raw_obs: Dict[str, Any], task_description: str, attention_map, value_context):
        if self.adaptive_chunker is None:
            return
        if not value_context or not value_context.get("dino_boxes"):
            logging.info("Skipping ValueFunc launch because no value context is available yet")
            return
        task_id = self.value_task_counter
        self.value_task_counter += 1
        future = self.value_executor.submit(
            self._value_func_task,
            task_id,
            raw_obs["pixels"]["image"][0],
            task_description,
            attention_map,
            value_context,
        )
        self.value_futures.append((task_id, future))

    def _consume_value_func_result_nonblocking(self):
        if not self.value_futures:
            return None
        latest_result = None
        remaining = []
        for task_id, future in self.value_futures:
            if future.done():
                try:
                    result = future.result()
                    if result:
                        latest_result = result
                except Exception as exc:
                    logging.warning(f"Background value func task failed: {exc}")
            else:
                remaining.append((task_id, future))
        self.value_futures = remaining
        if latest_result is None:
            return None
        task_id = latest_result.get("task_id", -1)
        if task_id < self.latest_consumed_value_task_id:
            return None
        self.latest_consumed_value_task_id = task_id
        logging.info("Consuming value func result task_id=%s", task_id)
        value_info = dict(latest_result.get("info") or {})
        value_info["value_task_id"] = task_id
        value_info["current_horizon"] = latest_result.get("current_horizon", self.current_chunk_size)
        return value_info

    def launch_next_inference(self, raw_obs: Dict[str, Any], task_description: str):
        """启动后台推理"""
        self.inference_times += 1
        self.per_env_inference_counts[self.active_env_mask] += 1
        self._record_launch_interval()
        if self.adaptive_chunker is not None:
            logging.info(f"Launching inference for task: interval={self.launch_step_intervals[-1] if len(self.launch_step_intervals) > 0 else 'N/A'}")
        self.save_attention_step -= 1
        prepared_obs = self._prepare_batch_obs(raw_obs, task_description)
        self.valid_chunk_size = self.current_chunk_size
        # 启动推理任务
        self.next_chunk_future = self.executor.submit(self._inference_task, prepared_obs)

    def reset_chunk(self, reason=False, set_chunk_size=False):
        logging.info(f"Resetting chunk: set_chunk_size={set_chunk_size}")
        self.current_chunks = self.next_chunk_future.result()
        self.active_chunk_size = self.valid_chunk_size
        self.last_chunk_size = self.active_chunk_size
        if set_chunk_size:
            self.current_chunk_size = self.valid_chunk_size
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

    def _select_action_step_index(self, current_step: int, effective_limit: int) -> int:
        """Optionally skip one micro-step for very small chunks in post-grasp phases."""
        if not self.cfg.enable_small_chunk_action_skip or self.cfg.use_adaptive_chunk == False:
            return current_step, None, None
        if effective_limit > self.cfg.small_chunk_action_skip_threshold:
            return current_step, None, None

        skip_offset = max(0, int(self.cfg.small_chunk_action_skip_offset))
        if skip_offset <= 0:
            return current_step, None, None

        candidate_step = current_step + skip_offset
        if candidate_step >= effective_limit:
            return current_step, None, None
        original_action_slice = self.current_chunks[:, current_step:candidate_step + 1, :]
        accumulated_action = original_action_slice.sum(axis=1)
        return candidate_step, accumulated_action, original_action_slice

    def _apply_chunk_update_immediately(self, raw_obs: Dict[str, Any], task_description: str) -> bool:
        """Make newly computed chunk sizes take effect immediately on the current control flow."""
        if self.current_chunks is None:
            return False

        effective_limit = min(self.active_chunk_size, self.n_action_steps, self.last_chunk_size)

        if self.overlap_steps > 0 and self.next_chunk_future is None:
            overlap_point = max(0, effective_limit - self.overlap_steps)
            if self.chunk_index >= overlap_point:
                logging.info(
                    "Manager: Updated chunk size makes current index cross overlap point (%s >= %s), launching prefetch immediately...",
                    self.chunk_index,
                    overlap_point,
                )
                self.launch_next_inference(raw_obs, task_description)

        if self.chunk_index >= effective_limit:
            logging.info(
                "Manager: Updated chunk size makes current chunk expired immediately (%s >= %s), switching now...",
                self.chunk_index,
                effective_limit,
            )
            return True

        return False

    def get_action_batch(self, raw_obs: Dict[str, Any], task_description: str) -> np.ndarray:
        """
        获取当前步的 Batch Action
        """
        self.current_task_description = task_description
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
            resumed_from_prefetch = self.next_chunk_future is not None
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
                self._update_latest_background_info(background_info)
                final_chunk = background_info.get("current_horizon")
                logging.info(f"background horizon: {final_chunk}")
            value_info = self._consume_value_func_result_nonblocking()
            if value_info is not None:
                value_horizon = value_info.get("current_horizon")
                if value_horizon is not None:
                    final_chunk = min(final_chunk, value_horizon) if final_chunk is not None else value_horizon
                    logging.info(f"ValueFunc background chunk size: {value_horizon}")
            
            # 获取注意力图并使用它进一步调整chunk size (只有在policy真正做chunk推理时才使用ValueFunc)
            if self.adaptive_chunker:
                try:
                    attn_maps = self.policy.get_attn_maps(layers=self.target_layers, heads=range(8), force=True)
                    if attn_maps:
                        vla_attention_map = attn_maps #[layers, num_heads, seq_len]
                        self._maybe_launch_value_func(
                            raw_obs,
                            task_description,
                            vla_attention_map,
                            self._build_value_context(background_info),
                        )
                        logging.info("Launched background ValueFunc task")
                except Exception as e:
                    import traceback
                    logging.warning(f"Failed to get attention maps: {e}")
                    traceback.print_exc()
            if final_chunk is not None:
                self.current_chunk_size = final_chunk
                logging.info(f"Final chunk size: {self.current_chunk_size}")

            # If we resume from a prefetched chunk and its starting index has already
            # crossed the overlap point (e.g. chunk=5, overlap=3), launch the next
            # prefetch immediately; otherwise this chunk would miss its own prefetch window.
            if resumed_from_prefetch and self.overlap_steps > 0 and self.next_chunk_future is None:
                overlap_point = min(self.active_chunk_size, self.n_action_steps, self.last_chunk_size) - self.overlap_steps
                if self.chunk_index >= overlap_point:
                    logging.info(
                        "Manager: Resumed prefetched chunk beyond overlap point (%s >= %s), launching next prefetch immediately...",
                        self.chunk_index,
                        overlap_point,
                    )
                    self.launch_next_inference(raw_obs, task_description)
        else:
            # 检查后台自适应任务是否完成
            background_info = self._consume_background_result_nonblocking()
            if background_info is not None:
                self._update_latest_background_info(background_info)
                background_horizon = background_info.get("current_horizon")
                if background_horizon is not None:
                    self.current_chunk_size = background_horizon
                    if self.current_chunks is not None and background_horizon < self.active_chunk_size:
                        self.active_chunk_size = background_horizon
                if background_info.get("needs_interruption", False):
                    logging.info(f"Interrupting current chunk due to: {background_info.get('factors', 'N/A')}")
                if self.cfg.enable_gripper_interruption and self.should_interrupt_gripper():
                    should_launch = True
                    gripper_change = True
                should_launch = self._apply_chunk_update_immediately(raw_obs, task_description) or should_launch
            value_info = self._consume_value_func_result_nonblocking()
            if value_info is not None:
                value_horizon = value_info.get("current_horizon")
                if value_horizon is not None:
                    self.current_chunk_size = min(self.current_chunk_size, value_horizon)
                    self.active_chunk_size = min(self.active_chunk_size, value_horizon)
                    logging.info(f"Applying ValueFunc chunk size immediately: {self.current_chunk_size}")
                    should_launch = self._apply_chunk_update_immediately(raw_obs, task_description) or should_launch

        # --- 3. 预取逻辑 (Hungry Strategy) ---
        # 当执行到当前 Chunk 的特定位置时，提前启动下一波推理
        if self.overlap_steps > 0:
            overlap_point = min(self.active_chunk_size, self.n_action_steps, self.last_chunk_size) - self.overlap_steps
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
                
            self.reset_chunk(gripper_change, set_chunk_size=True)
            self.chunk_index = 0
        
        effective_limit = min(self.active_chunk_size, self.n_action_steps, self.last_chunk_size)
        current_step = min(self.chunk_index, effective_limit - 1)
        candidate_step, actions, original_action_slice = self._select_action_step_index(current_step, effective_limit)
        
        if actions is None:
            actions = self.current_chunks[:, current_step, :]
            self.chunk_index += 1
        else:
            self.chunk_index = candidate_step + 1
        
        if self.adaptive_chunker:
            if original_action_slice is not None:
                for action_step in original_action_slice[0]:
                    self.adaptive_chunker.enqueue_history(action_step)
            else:
                self.adaptive_chunker.enqueue_history(actions[0])

        # overlapped=4且min horizon=5时会导致一但进入5，就会持续每步预测，然后不连续导致进一步维持在5
        # self.chunk_index = (self.chunk_index + 1)
        # logging.info(f"last chunk size: {self.last_chunk_size}, valid chunk size: {self.valid_chunk_size}")
        # 当达到当前chunk size或最大步骤时，重置
        if self.chunk_index >= min(self.active_chunk_size, self.n_action_steps, self.last_chunk_size):
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
    async_mgr.vec_env = vec_env
    async_mgr.camera_projection = _extract_camera_projection(vec_env)
    if hasattr(vec_env, "set_frame_annotator"):
        vec_env.set_frame_annotator(
            async_mgr._annotate_debug_frame_from_obs if (cfg.video_overlay_debug or cfg.save_video) else None
        )

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
    max_steps = (
        int(cfg.episode_length)
        if getattr(cfg, "episode_length", None) is not None
        else vec_env.call("_max_episode_steps")[0]
    )  # Prefer explicit eval config when provided.
    monitor = GPUMonitor(gpu_id=0, interval=0.05)
    if cfg.use_monitor:
        monitor.start()
    try:
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
    finally:
        if cfg.use_monitor:
            try:
                monitor.stop()
            except Exception:
                pass
        async_mgr.shutdown()

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
    
    if "check" in cfg.policy.path:
        cfg.policy.normalization_mapping = field(
            default_factory=lambda: {
                "VISUAL": NormalizationMode.IDENTITY,  # Images: no normalization
                "STATE": NormalizationMode.QUANTILES,   # States: quantile normalization (matching official)
                "ACTION": NormalizationMode.QUANTILES,  # Actions: quantile normalization (matching official)
            }
        )
    
    # 1. 加载策略
    policy_cls = get_policy_class(cfg.policy.type)
    # print(dataset.meta.stats)
    # 这里config有的话就直接用，没的话从path加载，所以 感觉其实可以不写，哦但是tmd compile model就一定会搞，算了还是用我的policy config写吧
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy, dataset_stats=dataset.meta.stats) 
    # policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, dataset_stats=dataset.meta.stats) 
    policy.eval()
    logging.info(f"Policy normalization mapping: {cfg.policy.normalization_mapping}")
    # 2. 初始化所有任务的环境
    if cfg.env_type == "libero":
        # 不指定 task_ids，让 factory 自动创建该 suite 下的所有任务
        env_cfg = LiberoEnv(
            task=cfg.env_task,
            task_ids=cfg.task_ids,
            obs_type="pixels_depth_agent_pos",
            episode_length=cfg.episode_length,
        )
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
    
if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        traceback.print_exc()
