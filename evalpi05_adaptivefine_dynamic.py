#!/usr/bin/env python3
"""
Evaluate the existing VLASH pipeline on a custom dynamic LIBERO environment.

This file reuses the normal adaptive eval logic, but monkey-patches the
lerobot LIBERO environment builder so that task creation loads a custom dynamic
BDDL + dynamic OffScreenRenderEnv instead of the benchmark-provided static BDDL.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tempfile
import cv2

import numpy as np
import torch
import yaml
import draccus
try:
    import imageio
except Exception:
    imageio = None
try:
    import mediapy as media
except Exception:
    media = None

from lerobot.configs import parser
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm

from libero.libero.envs import OffScreenRenderEnv

from evalpi05_adaptivefine_norm import (
    VLASHEvalConfig,
    _collect_eval_runtime_config,
    _configure_component_log_levels,
    _log_effective_eval_setup,
    _write_yaml_file,
    eval_single_vec_env,
)
from vlash.configs.train_config import VLASHTrainConfig
from vlash.policies.factory import get_policy_class


@dataclass
class VLASHDynamicLiberoEvalConfig(VLASHEvalConfig):
    """Dynamic LIBERO eval config.

    Most eval/policy/environment fields are inherited directly from
    `VLASHEvalConfig`. This subclass only keeps dynamic-environment specific
    options plus the relaxed `train_cfg` / `policy` loading used here.
    """

    output_dir: str = "outputs/eval/vlash_adaptive_dynamic"
    enable_object_motion_penalty: bool = True

    dynamic_bddl_file: str = (
        "/home/wangzb/vlash/LIBERO/libero/libero/bddl_files/libero_dynamic_examples/"
        "test_dynamic_example.bddl"
    )
    # 注意这个也要加入到extractor中，不然就会回退到提前返回导致实际chunk size并没有变
    dynamic_task_description: str = "pick up the orange juice and place it in the basket"
    # dynamic_task_description: str = "pick up the orange juice and place it in the moving basket"
    dynamic_shell_task_id: int = 9
    dynamic_use_init_states: bool = False
    dynamic_moving_object_name: str = "orange_juice_1"
    dynamic_motion_axis: str = "y"
    dynamic_motion_settle_steps: int = 10
    dynamic_motion_drive_steps: int = 40
    dynamic_motion_drive_speed: float = 0.005
    dynamic_motion_end_speed: float = 0.0
    dynamic_motion_return_enabled: bool = False
    dynamic_save_video: bool = True
    dynamic_video_dir: str = "outputs/eval/dynamic_videos"
    dynamic_video_fps: int = 20

    # Allow dynamic eval to bootstrap these from the default yaml instead of
    # requiring the full training config at parser construction time.
    # train_cfg: Any = None
    # policy: Any = None
    default_eval_yaml: str = "/home/wangzb/vlash/evalpi05.yaml"


class RecordingVecEnvWrapper:
    def __init__(self, vec_env, enabled: bool, output_path: Path | None, fps: int = 20):
        self.vec_env = vec_env
        self.enabled = enabled
        self.output_path = output_path
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def _capture_frame(self):
        if not self.enabled:
            return
        try:
            frame = self.vec_env.call("render")[0]
        except Exception:
            return
        if frame is None:
            return
        frame = np.asarray(frame)
        if frame.ndim == 3:
            self.frames.append(frame.copy())

    def reset(self, *args, **kwargs):
        out = self.vec_env.reset(*args, **kwargs)
        self._capture_frame()
        return out

    def step(self, actions):
        # print(f"Stepping env with actions: {actions}")
        out = self.vec_env.step(actions)
        self._capture_frame()
        return out

    def save_video(self):
        if not self.enabled or not self.frames or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if imageio is not None:
            with imageio.get_writer(str(self.output_path), fps=self.fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            return
        if media is not None:
            media.write_video(str(self.output_path), self.frames, fps=self.fps)
            return

        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )
        for frame in self.frames:
            # vec_env.render() returns RGB; cv2 expects BGR.
            writer.write(frame[:, :, ::-1])
        writer.release()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)


def _load_default_eval_yaml(cfg: VLASHDynamicLiberoEvalConfig) -> dict[str, Any]:
    source_yaml = Path(cfg.default_eval_yaml)
    if not source_yaml.exists():
        raise FileNotFoundError(f"default_eval_yaml does not exist: {source_yaml}")
    return yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}


def _resolve_train_cfg(cfg: VLASHDynamicLiberoEvalConfig, raw_default_cfg: dict[str, Any]) -> VLASHTrainConfig:
    if isinstance(cfg.train_cfg, VLASHTrainConfig):
        return cfg.train_cfg

    if cfg.train_cfg is None:
        train_cfg_data = raw_default_cfg.get("train_cfg")
        if train_cfg_data is None:
            raise ValueError(f"No 'train_cfg' found in {cfg.default_eval_yaml}")
    elif isinstance(cfg.train_cfg, dict):
        train_cfg_data = cfg.train_cfg
    else:
        raise TypeError(f"Unsupported train_cfg type: {type(cfg.train_cfg)}")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(train_cfg_data, f, sort_keys=False)
        tmp_path = f.name
    try:
        return draccus.parse(config_class=VLASHTrainConfig, config_path=tmp_path, args=[])
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _resolve_policy(cfg: VLASHDynamicLiberoEvalConfig, raw_default_cfg: dict[str, Any]):
    if cfg.policy is not None:
        return cfg.policy

    policy_data = raw_default_cfg.get("policy")
    if policy_data is None:
        raise ValueError(f"No 'policy' found in {cfg.default_eval_yaml}")

    policy_path = policy_data.get("path") or policy_data.get("pretrained_path")
    if not policy_path:
        raise ValueError("Default eval yaml policy block must include 'path' or 'pretrained_path'")

    cli_overrides = []
    for k, v in policy_data.items():
        if k in {"path", "pretrained_path"}:
            continue
        val = str(v).lower() if isinstance(v, bool) else v
        cli_overrides.append(f"--{k}={val}")

    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = policy_path
    return policy_cfg


@contextmanager
def patch_dynamic_libero_env(cfg: VLASHDynamicLiberoEvalConfig):
    from lerobot.envs import libero as lerobot_libero

    original_make_envs_task = lerobot_libero.LiberoEnv._make_envs_task

    def _make_dynamic_envs_task(self, task_suite: Any, task_id: int = 0):
        try:
            shell_task = task_suite.get_task(task_id)
            fallback_description = getattr(shell_task, "language", cfg.dynamic_task_description)
            self.task = f"dynamic_{getattr(shell_task, 'name', 'task')}"
        except Exception:
            fallback_description = cfg.dynamic_task_description
            self.task = f"dynamic_task_{task_id}"

        self.task_description = cfg.dynamic_task_description or fallback_description

        env_args = {
            "bddl_file_name": cfg.dynamic_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
            "moving_object_name": cfg.dynamic_moving_object_name,
            "motion_axis": cfg.dynamic_motion_axis,
            "motion_settle_steps": cfg.dynamic_motion_settle_steps,
            "motion_drive_steps": cfg.dynamic_motion_drive_steps,
            "motion_drive_speed": cfg.dynamic_motion_drive_speed,
            "motion_end_speed": cfg.dynamic_motion_end_speed,
            "motion_return_enabled": cfg.dynamic_motion_return_enabled,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    lerobot_libero.LiberoEnv._make_envs_task = _make_dynamic_envs_task
    try:
        yield
    finally:
        lerobot_libero.LiberoEnv._make_envs_task = original_make_envs_task


@parser.wrap()
def run_benchmark(cfg: VLASHDynamicLiberoEvalConfig):
    init_logging()
    _configure_component_log_levels(cfg)
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    from vlash.train import make_vlash_dataset

    raw_default_cfg = _load_default_eval_yaml(cfg)
    train_cfg = _resolve_train_cfg(cfg, raw_default_cfg)
    cfg.policy = _resolve_policy(cfg, raw_default_cfg)
    cfg.train_cfg = train_cfg

    _log_effective_eval_setup(cfg)
    logging.info("Dynamic Env Setup")
    logging.info(f"Dynamic BDDL: {cfg.dynamic_bddl_file}")
    logging.info(f"Dynamic task description: {cfg.dynamic_task_description}")
    logging.info(f"Dynamic shell task id: {cfg.dynamic_shell_task_id}")
    logging.info(f"Dynamic use init states: {cfg.dynamic_use_init_states}")
    logging.info(f"Dynamic moving object: {cfg.dynamic_moving_object_name}")
    logging.info(f"Dynamic motion axis: {cfg.dynamic_motion_axis}")
    logging.info(f"Dynamic motion settle steps: {cfg.dynamic_motion_settle_steps}")
    logging.info(f"Dynamic motion drive steps: {cfg.dynamic_motion_drive_steps}")
    logging.info(f"Dynamic motion drive speed: {cfg.dynamic_motion_drive_speed}")
    logging.info(f"Dynamic motion end speed: {cfg.dynamic_motion_end_speed}")
    logging.info(f"Dynamic motion return enabled: {cfg.dynamic_motion_return_enabled}")
    logging.info(f"Dynamic save video: {cfg.dynamic_save_video}")
    logging.info(f"Dynamic video dir: {cfg.dynamic_video_dir or (Path(cfg.output_dir) / 'dynamic_videos')}")
    logging.info("=" * 50)

    dataset = make_vlash_dataset(train_cfg)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(
        cfg.policy.pretrained_path,
        config=cfg.policy,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()

    dynamic_task_ids = cfg.task_ids if cfg.task_ids is not None else [cfg.dynamic_shell_task_id]
    env_cfg = LiberoEnv(
        task=cfg.env_task,
        task_ids=dynamic_task_ids,
        init_states=cfg.dynamic_use_init_states,
    )

    logging.info(f"Initializing all tasks for {cfg.env_task}...")
    with patch_dynamic_libero_env(cfg):
        all_envs_dict = make_env(env_cfg, n_envs=cfg.eval_batch_size)

        results_per_task = {}
        launch_interval_results = {}
        suite_successful_inference_counts = {}
        overall_successes = []
        shared_chunker_runtime_config = None
        start_time = time.time()

        for suite_name, tasks in all_envs_dict.items():
            logging.info(f"Evaluating Dynamic Suite Shell: {suite_name}")
            for task_id, vec_env in tasks.items():
                task_description = cfg.dynamic_task_description
                task_name = f"{suite_name}_dynamic_{task_id}"
                logging.info(f"Starting Dynamic Task [{task_id}]: {task_description}")

                video_dir = Path(cfg.dynamic_video_dir) if cfg.dynamic_video_dir else Path(cfg.output_dir) / "dynamic_videos"
                recording_env = RecordingVecEnvWrapper(
                    vec_env,
                    enabled=cfg.dynamic_save_video,
                    output_path=video_dir / f"{task_name}.mp4",
                    fps=cfg.dynamic_video_fps,
                )

                task_info = eval_single_vec_env(
                    recording_env,
                    policy,
                    cfg,
                    task_description,
                    task_name=task_name,
                )
                if shared_chunker_runtime_config is None and cfg.use_adaptive_chunk:
                    shared_chunker_runtime_config = task_info.get("adaptive_chunker_runtime_config")
                recording_env.save_video()

                results_per_task[f"{suite_name}_{task_id}"] = {
                    "task_description": task_description,
                    "success_rate": task_info["sr"],
                    "num_episodes": task_info["count"],
                    "avg_steps": task_info["avg_steps"],
                    "avg_successful_steps": task_info["avg_successful_steps"],
                    "steps_list": task_info["steps_list"],
                    "successful_steps_list": task_info["successful_steps_list"],
                    "inference_times": task_info["inference_times"],
                    "per_episode_inference_counts": task_info["per_episode_inference_counts"],
                    "successful_inference_counts": task_info["successful_inference_counts"],
                    "avg_successful_inference_times": task_info["avg_successful_inference_times"],
                }
                launch_interval_results[f"{suite_name}_{task_id}"] = task_info["launch_interval_stats"]
                suite_successful_inference_counts.setdefault(suite_name, [])
                suite_successful_inference_counts[suite_name].extend(task_info["successful_inference_counts"])
                overall_successes.extend(task_info["success_list"])
                logging.info(f"Task Done. SR: {task_info['sr']*100:.1f}%")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total_time = time.time() - start_time

        runtime_config = _collect_eval_runtime_config(cfg, None)
        if shared_chunker_runtime_config is not None:
            runtime_config["adaptive_chunker"] = shared_chunker_runtime_config
        runtime_config["dynamic_env"] = {
            "dynamic_bddl_file": cfg.dynamic_bddl_file,
            "dynamic_task_description": cfg.dynamic_task_description,
            "dynamic_shell_task_id": cfg.dynamic_shell_task_id,
            "dynamic_use_init_states": cfg.dynamic_use_init_states,
            "dynamic_moving_object_name": cfg.dynamic_moving_object_name,
            "dynamic_motion_axis": cfg.dynamic_motion_axis,
            "dynamic_motion_settle_steps": cfg.dynamic_motion_settle_steps,
            "dynamic_motion_drive_steps": cfg.dynamic_motion_drive_steps,
            "dynamic_motion_drive_speed": cfg.dynamic_motion_drive_speed,
            "dynamic_motion_end_speed": cfg.dynamic_motion_end_speed,
            "dynamic_motion_return_enabled": cfg.dynamic_motion_return_enabled,
            "dynamic_save_video": cfg.dynamic_save_video,
            "dynamic_video_dir": cfg.dynamic_video_dir,
            "dynamic_video_fps": cfg.dynamic_video_fps,
        }

        all_steps = [step for task_result in results_per_task.values() for step in task_result["steps_list"]]
        all_successful_steps = [
            step for task_result in results_per_task.values() for step in task_result["successful_steps_list"]
        ]

        final_results = {
            "overall": {
                "avg_success_rate": float(np.mean(overall_successes)) if overall_successes else 0.0,
                "total_episodes": len(overall_successes),
                "total_eval_time_s": total_time,
                "env_task": cfg.env_task,
                "use_adaptive_chunk": cfg.use_adaptive_chunk,
                "avg_steps": float(np.mean(all_steps)) if all_steps else 0.0,
                "avg_steps_per_successful_episode": (
                    float(np.mean(all_successful_steps)) if all_successful_steps else None
                ),
                "total_steps": int(sum(all_steps)) if all_steps else 0,
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

        (out_dir / "full_benchmark_results.json").write_text(
            __import__("json").dumps(final_results, indent=2),
            encoding="utf-8",
        )
        (out_dir / "launch_interval_stats.json").write_text(
            __import__("json").dumps(launch_interval_results, indent=2),
            encoding="utf-8",
        )
        _write_yaml_file(out_dir / "runtime_config.yaml", runtime_config)

        logging.info("=" * 50)
        logging.info(f"FINAL OVERALL SR: {final_results['overall']['avg_success_rate'] * 100:.2f}%")
        logging.info(f"Results saved to {out_dir}")
        logging.info(f"Runtime config saved to {out_dir / 'runtime_config.yaml'}")
        logging.info("=" * 50)

        num_tasks = len(results_per_task)
        ep_per_task = cfg.max_episodes
        total_test_count = len(overall_successes)

        logging.info("=" * 50)
        logging.info("评估统计报告:")
        logging.info(f" - 环境集合 (Suite): {cfg.env_task}")
        logging.info(f" - 任务总数 (Num Tasks): {num_tasks}")
        logging.info(f" - 每任务测试数 (Episodes per Task): {ep_per_task}")
        logging.info(f" - 总测试规模 (Total Episodes): {total_test_count}")
        logging.info(f" - 最终总成功率 (Overall SR): {final_results['overall']['avg_success_rate']*100:.2f}%")
        logging.info(
            f" - 平均每episode步骤数 (Avg Steps per Episode): "
            f"{(float(np.mean(all_steps)) if all_steps else float('nan')):.2f}"
        )
        logging.info(
            f" - 成功episode平均步骤数 (Avg Steps per Successful Episode): "
            f"{(float(np.mean(all_successful_steps)) if all_successful_steps else float('nan')):.2f}"
        )
        logging.info(f" - 总步骤数 (Total Steps): {sum(all_steps) if all_steps else 0}")
        logging.info(f" - 自适应chunk size: {'启用' if cfg.use_adaptive_chunk else '禁用'}")
        if cfg.use_adaptive_chunk:
            logging.info(f" - Chunk size范围: {cfg.adaptive_min_horizon} - {cfg.adaptive_max_horizon}")
        for suite_name, counts in suite_successful_inference_counts.items():
            logging.info(
                f" - Suite {suite_name} 成功episode平均推理次数 (Avg Inference Count on Success): "
                f"{(float(np.mean(counts)) if counts else float('nan')):.2f}"
            )
        logging.info("=" * 50)

        close_envs(all_envs_dict)


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        traceback.print_exc()
