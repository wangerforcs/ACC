#!/usr/bin/env python3
"""
Power-oriented record / replay evaluation for VLASH.

Two modes:
- record: run one real rollout, save per-step observations / actions / metadata
- replay: re-run inference on the saved observations, but replace env.step with
  a fixed sleep(1 / control_hz) to exclude simulator stepping cost from power tests
"""

import gzip
import json
import logging
import pickle
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.utils.utils import init_logging, inside_slurm
from lerobot.utils.random_utils import set_seed

from adaptive.monitor import GPUMonitor
from evalpi05_adaptivefine_norm import (
    VLASHEvalConfig,
    VLASHVectorAsyncManager,
    _collect_eval_runtime_config,
    _configure_component_log_levels,
    _log_effective_eval_setup,
    _write_yaml_file,
)
from vlash.train import make_vlash_dataset
from vlash.policies.factory import get_policy_class


@dataclass
class PowerReplayEvalConfig(VLASHEvalConfig):
    output_dir: str = "outputs/eval/vlash_power_replay"
    max_episodes: int = 1
    eval_batch_size: int = 1
    save_video: bool = False
    use_monitor: bool = True

    mode: str = "record"  # record | replay
    trace_path: str = "outputs/eval/vlash_power_replay/trace.pkl.gz"
    control_hz: float = 10.0
    monitor_gpu_id: int = 0
    monitor_interval: float = 0.05
    save_trace_images_preview: bool = False
    replay_quiet: bool = True
    export_readable_trace: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.eval_batch_size != 1:
            raise ValueError("Power replay script only supports eval_batch_size=1")
        if self.max_episodes != 1:
            raise ValueError("Power replay script only supports max_episodes=1")
        if self.mode not in {"record", "replay"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")


def _extract_inference_input(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the observation fields that the policy path actually consumes."""
    extracted: Dict[str, Any] = {}
    if "agent_pos" in obs:
        extracted["agent_pos"] = np.array(obs["agent_pos"], copy=True)
    if "pixels" in obs and isinstance(obs["pixels"], dict):
        extracted["pixels"] = {
            key: np.array(value, copy=True)
            for key, value in obs["pixels"].items()
        }
    return extracted


def _save_trace(trace_path: Path, trace_data: Dict[str, Any]):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace_path, "wb") as f:
        pickle.dump(trace_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_trace(trace_path: Path) -> Dict[str, Any]:
    with gzip.open(trace_path, "rb") as f:
        return pickle.load(f)


def _trace_base_path(trace_path: Path) -> Path:
    name = trace_path.name
    if name.endswith(".pkl.gz"):
        return trace_path.with_name(name[:-7])
    return trace_path.with_suffix("")


def _write_image_rgb(path: Path, image: np.ndarray):
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
        return
    except Exception:
        pass

    import cv2

    cv2.imwrite(str(path), image[:, :, ::-1])


def _save_preview_images(trace_data: Dict[str, Any], output_dir: Path):
    try:
        import imageio.v2 as imageio
    except Exception:
        logging.warning("imageio not available, skipping preview image export")
        return

    observations = trace_data.get("inference_inputs", [])
    if not observations:
        return

    preview_dir = output_dir / "trace_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, obs in enumerate(observations[: min(10, len(observations))]):
        pixels = obs.get("pixels", {})
        if "image" in pixels:
            imageio.imwrite(preview_dir / f"step_{idx:04d}.png", pixels["image"][0])


def _export_trace_readable(trace_data: Dict[str, Any], trace_path: Path):
    base_path = _trace_base_path(trace_path)
    summary_path = base_path.with_name(base_path.name + "_summary.json")
    images_dir = base_path.with_name(base_path.name + "_images")

    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    num_steps = len(trace_data.get("inference_inputs", []))
    chunk_sizes = trace_data.get("chunk_sizes", [None] * num_steps)
    box_sources = trace_data.get("box_sources", [None] * num_steps)
    actions = trace_data.get("actions", [])

    step_entries = []
    for step_idx, step_input in enumerate(trace_data.get("inference_inputs", [])):
        step_entry: Dict[str, Any] = {
            "step": step_idx,
            "chunk_size": chunk_sizes[step_idx] if step_idx < len(chunk_sizes) else None,
            "box_source": box_sources[step_idx] if step_idx < len(box_sources) else None,
            "action": np.asarray(actions[step_idx]).tolist() if step_idx < len(actions) else None,
        }

        if "agent_pos" in step_input:
            step_entry["agent_pos"] = np.asarray(step_input["agent_pos"]).tolist()

        pixels = step_input.get("pixels", {})
        image_files = {}
        for image_key, image_value in pixels.items():
            image_arr = np.asarray(image_value)
            if image_arr.ndim == 4 and image_arr.shape[0] == 1:
                image_arr = image_arr[0]
            image_name = f"step_{step_idx:04d}_{image_key}.png"
            _write_image_rgb(images_dir / image_name, image_arr)
            image_files[image_key] = image_name
        if image_files:
            step_entry["pixel_files"] = image_files

        step_entries.append(step_entry)

    readable_summary = {
        "task_key": trace_data.get("task_key"),
        "task_description": trace_data.get("task_description"),
        "seed": trace_data.get("seed"),
        "control_hz": trace_data.get("control_hz"),
        "success": trace_data.get("success"),
        "num_steps": trace_data.get("num_steps"),
        "recorded_at": trace_data.get("recorded_at"),
        "runtime_config": trace_data.get("runtime_config"),
        "monitor": trace_data.get("monitor"),
        "steps": step_entries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(readable_summary, f, indent=2)


def _build_policy(cfg: PowerReplayEvalConfig):
    dataset = make_vlash_dataset(cfg.train_cfg)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(
        cfg.policy.pretrained_path,
        config=cfg.policy,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()
    return policy


def _build_monitor(cfg: PowerReplayEvalConfig):
    return GPUMonitor(gpu_id=cfg.monitor_gpu_id, interval=cfg.monitor_interval)


def _record_single_task(
    vec_env: gym.vector.VectorEnv,
    policy,
    cfg: PowerReplayEvalConfig,
    task_description: str,
    task_key: str,
) -> Dict[str, Any]:
    async_mgr = VLASHVectorAsyncManager(
        policy=policy,
        overlap_steps=cfg.inference_overlap_steps * cfg.action_quant_ratio,
        num_envs=1,
        task_name=task_key,
        cfg=cfg,
    )

    monitor = _build_monitor(cfg)
    if cfg.use_monitor:
        monitor.start()

    observation, info = vec_env.reset(seed=cfg.seed)
    async_mgr.reset_state()

    trace_inputs: List[Dict[str, Any]] = []
    trace_actions: List[np.ndarray] = []
    trace_chunk_sizes: List[int] = []
    trace_sources: List[Optional[str]] = []

    success = False
    max_steps = vec_env.call("_max_episode_steps")[0]
    done = False

    pbar = trange(max_steps, desc=" Record", leave=False, disable=inside_slurm())
    for step in pbar:
        if done:
            break

        async_mgr.set_active_env_mask(np.array([True], dtype=bool))
        trace_inputs.append(_extract_inference_input(observation))

        actions = async_mgr.get_action_batch(observation, task_description)
        trace_actions.append(np.array(actions, copy=True))
        trace_chunk_sizes.append(int(async_mgr.current_chunk_size))
        trace_sources.append(
            None
            if async_mgr.adaptive_chunker is None
            else ("dino" if async_mgr.adaptive_chunker.last_query_used_dino else "tracker")
        )

        observation, reward, terminated, truncated, info = vec_env.step(actions)
        done = bool(np.asarray(terminated | truncated)[0])

        if "final_info" in info and isinstance(info["final_info"], dict) and "is_success" in info["final_info"]:
            success = bool(np.asarray(info["final_info"]["is_success"])[0])

    if cfg.use_monitor:
        monitor.stop()
    monitor_stats = monitor.get_report() if cfg.use_monitor else None

    trace_data = {
        "task_key": task_key,
        "task_description": task_description,
        "seed": cfg.seed,
        "control_hz": cfg.control_hz,
        "success": success,
        "num_steps": len(trace_actions),
        "inference_inputs": trace_inputs,
        "actions": trace_actions,
        "chunk_sizes": trace_chunk_sizes,
        "box_sources": trace_sources,
        "runtime_config": _collect_eval_runtime_config(cfg, async_mgr.adaptive_chunker),
        "monitor": monitor_stats,
        "recorded_at": time.time(),
    }
    return trace_data


def _replay_trace(trace_data: Dict[str, Any], policy, cfg: PowerReplayEvalConfig) -> Dict[str, Any]:
    task_description = trace_data["task_description"]
    task_key = trace_data.get("task_key", "replay_task")
    observations = trace_data["inference_inputs"]
    recorded_actions = trace_data["actions"]

    async_mgr = VLASHVectorAsyncManager(
        policy=policy,
        overlap_steps=cfg.inference_overlap_steps * cfg.action_quant_ratio,
        num_envs=1,
        task_name=task_key,
        cfg=cfg,
    )
    async_mgr.reset_state()

    monitor = _build_monitor(cfg)
    if cfg.use_monitor:
        monitor.start()

    replay_actions: List[np.ndarray] = []
    chunk_sizes: List[int] = []
    per_step_max_abs_diff: List[float] = []
    step_sleep_s = 1.0 / cfg.control_hz

    pbar = trange(len(observations), desc=" Replay", leave=False, disable=inside_slurm())
    for obs, recorded_action in zip(observations, recorded_actions):
        async_mgr.set_active_env_mask(np.array([True], dtype=bool))
        actions = async_mgr.get_action_batch(obs, task_description)
        replay_actions.append(np.array(actions, copy=True))
        chunk_sizes.append(int(async_mgr.current_chunk_size))
        diff = np.max(np.abs(np.asarray(actions) - np.asarray(recorded_action)))
        per_step_max_abs_diff.append(float(diff))
        time.sleep(step_sleep_s)

    if cfg.use_monitor:
        monitor.stop()
    monitor_stats = monitor.get_report() if cfg.use_monitor else None

    return {
        "task_key": task_key,
        "task_description": task_description,
        "recorded_num_steps": len(recorded_actions),
        "replayed_num_steps": len(replay_actions),
        "control_hz": cfg.control_hz,
        "sleep_per_step_s": step_sleep_s,
        "inference_times": int(async_mgr.inference_times),
        "per_env_inference_counts": async_mgr.per_env_inference_counts.tolist(),
        "avg_replayed_chunk_size": float(np.mean(chunk_sizes)) if chunk_sizes else 0.0,
        "chunk_sizes": chunk_sizes,
        "action_max_abs_diff_mean": float(np.mean(per_step_max_abs_diff)) if per_step_max_abs_diff else 0.0,
        "action_max_abs_diff_max": float(np.max(per_step_max_abs_diff)) if per_step_max_abs_diff else 0.0,
        "monitor": monitor_stats,
        "runtime_config": _collect_eval_runtime_config(cfg, async_mgr.adaptive_chunker),
    }


@parser.wrap()
def run_benchmark(cfg: PowerReplayEvalConfig):
    init_logging()
    _configure_component_log_levels(cfg)
    if cfg.mode == "replay" and cfg.replay_quiet:
        logging.getLogger().setLevel(logging.ERROR)
        for logger_name in getattr(cfg, "tracker_logger_names", []) or []:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    set_seed(cfg.seed)
    _log_effective_eval_setup(cfg)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("=" * 50)
    logging.info("Power Replay Setup")
    logging.info(f"Mode: {cfg.mode}")
    logging.info(f"Trace path: {cfg.trace_path}")
    logging.info(f"Control Hz: {cfg.control_hz}")
    logging.info(f"Use monitor: {cfg.use_monitor}")
    logging.info(f"Monitor GPU id: {cfg.monitor_gpu_id}")
    logging.info(f"Monitor interval: {cfg.monitor_interval}")
    logging.info("=" * 50)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(cfg.trace_path)

    policy = _build_policy(cfg)

    if cfg.mode == "record":
        if cfg.env_type != "libero":
            raise ValueError(f"Unsupported env type: {cfg.env_type}")

        env_cfg = LiberoEnv(task=cfg.env_task, task_ids=cfg.task_ids)
        all_envs_dict = make_env(env_cfg, n_envs=1)
        results = {}

        try:
            for suite_name, tasks in all_envs_dict.items():
                for task_id, vec_env in tasks.items():
                    try:
                        task_description = vec_env.get_attr("task_description")[0]
                    except Exception:
                        task_description = cfg.single_task or f"{suite_name}_{task_id}"

                    task_key = f"{suite_name}_{task_id}"
                    task_trace_path = trace_path
                    if len(tasks) > 1 or len(all_envs_dict) > 1:
                        task_trace_path = trace_path.parent / f"{task_key}_{trace_path.name}"

                    logging.info(f"Recording task {task_key}: {task_description}")
                    trace_data = _record_single_task(vec_env, policy, cfg, task_description, task_key)
                    _save_trace(task_trace_path, trace_data)
                    if cfg.export_readable_trace:
                        _export_trace_readable(trace_data, task_trace_path)
                    if cfg.save_trace_images_preview:
                        _save_preview_images(trace_data, out_dir / task_key)

                    task_result = {
                        "mode": "record",
                        "task_description": task_description,
                        "trace_path": str(task_trace_path),
                        "success": trace_data["success"],
                        "num_steps": trace_data["num_steps"],
                        "monitor": trace_data["monitor"],
                        "runtime_config": trace_data["runtime_config"],
                    }
                    results[task_key] = task_result
                    with open(out_dir / f"{task_key}_record_result.json", "w", encoding="utf-8") as f:
                        json.dump(task_result, f, indent=2)
        finally:
            close_envs(all_envs_dict)

        final_results = {
            "mode": "record",
            "env_task": cfg.env_task,
            "results": results,
        }
    else:
        trace_data = _load_trace(trace_path)
        replay_result = _replay_trace(trace_data, policy, cfg)
        final_results = {
            "mode": "replay",
            "trace_path": str(trace_path),
            "result": replay_result,
        }

    with open(out_dir / "full_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    runtime_config = final_results["result"]["runtime_config"] if cfg.mode == "replay" else None
    if runtime_config is None and final_results.get("results"):
        first_result = next(iter(final_results["results"].values()))
        runtime_config = first_result.get("runtime_config")
    if runtime_config is not None:
        _write_yaml_file(out_dir / "runtime_config.yaml", runtime_config)

    logging.info(f"Saved results to {out_dir / 'full_benchmark_results.json'}")


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        traceback.print_exc()
