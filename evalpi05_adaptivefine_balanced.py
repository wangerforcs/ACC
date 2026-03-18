#!/usr/bin/env python3
"""
基于evalpi05.py的自适应chunk size版本
使用 balanced tracker policy，在保持 chunk 大于 normal 的同时，尽量保住 goal/10 的 SR。
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

from lerobot.configs import parser
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import close_envs
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm
from lerobot.utils.random_utils import set_seed
from lerobot.utils.control_utils import prepare_observation_for_inference

from vlash.policies.factory import get_policy_class, get_lerobot_policy_class
from visualutils import visualize_vla_attention, visualize_vla_attention_dino, vla_attention_specs, save_images, visualize_vla_attention_sum_heads_per_layer
from vlash.configs.train_config import VLASHTrainConfig

from adaptive.adaptivepolicyfinegrained_tracker_balanced import AdaptiveActionChunker
from adaptive.monitor import GPUMonitor
from evalpi05_adaptivefine_norm import _write_yaml_file, _collect_eval_runtime_config

from lerobot.configs.policies import PreTrainedConfig
from dataclasses import dataclass, field


@dataclass
class VLASHEvalConfig:
    env_type: str = "libero"
    env_task: str = "libero_object"
    max_episodes: int = 20
    eval_batch_size: int = 5
    seed: int = 42
    output_dir: str = "outputs/eval/vlash_adaptive_balanced"

    policy: Union[PreTrainedConfig, dict[str, Any], None] = None

    fps: int = 30
    display_data: bool = False
    play_sounds: bool = False
    action_quant_ratio: int = 1
    inference_overlap_steps: int = 0
    single_task: Union[str, None] = None
    task_ids: Union[List[int], None] = None
    train_cfg: VLASHTrainConfig = field(default_factory=VLASHTrainConfig)

    use_adaptive_chunk: bool = True
    adaptive_min_horizon: int = 5
    adaptive_max_horizon: int = 30
    adaptive_uncertainty_threshold: float = 0.015
    adaptive_coverage_threshold: float = 0.1
    adaptive_continuity_threshold: float = 0.2

    save_video: bool = False
    video_output_dir: str = "fuckvideos"

    use_monitor: bool = False
    eval_log_level: str = "INFO"
    tracker_log_level: str = "WARNING"
    scoregate_log_level: str = "WARNING"
    enable_gripper_interruption: bool = False

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
            if cli_policy_path:
                policy_path = cli_policy_path

            cli_policy_overrides = parser.get_cli_overrides("policy")
            if cli_policy_overrides:
                cli_overrides.extend(cli_policy_overrides)

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
    logging.getLogger("adaptive.tracker_balanced").setLevel(
        getattr(logging, str(cfg.scoregate_log_level).upper(), logging.WARNING)
    )


# Reuse the existing implementation by importing and rebinding the balanced policy above.
from evalpi05_adaptivefine_norm import VLASHVectorAsyncManager, eval_single_vec_env  # noqa: E402


@parser.wrap()
def run_benchmark(cfg: VLASHEvalConfig):
    init_logging()
    _configure_component_log_levels(cfg)
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"max_episodes: {cfg.max_episodes}, inference_overlap_steps: {cfg.inference_overlap_steps}, ")
    logging.info(f"Adaptive chunk size enabled: {cfg.use_adaptive_chunk}")
    if cfg.use_adaptive_chunk:
        logging.info(f"Adaptive chunk size range: {cfg.adaptive_min_horizon} - {cfg.adaptive_max_horizon}")

    from vlash.train import make_vlash_dataset

    dataset = make_vlash_dataset(cfg.train_cfg)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy, dataset_stats=dataset.meta.stats)
    policy.eval()

    if cfg.env_type == "libero":
        env_cfg = LiberoEnv(task=cfg.env_task, task_ids=cfg.task_ids)
    else:
        raise ValueError(f"Env type {cfg.env_type} not supported")

    logging.info(f"Initializing all tasks for {cfg.env_task}...")
    all_envs_dict = make_env(env_cfg, n_envs=cfg.eval_batch_size)

    results_per_task = {}
    launch_interval_results = {}
    overall_successes = []
    suite_successful_inference_counts = defaultdict(list)
    start_time = time.time()
    task_name = ""

    for suite_name, tasks in all_envs_dict.items():
        logging.info(f"Evaluating Suite: {suite_name}")
        for task_id, vec_env in tasks.items():
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
                "avg_successful_steps": task_info["avg_successful_steps"],
                "steps_list": task_info["steps_list"],
                "successful_steps_list": task_info["successful_steps_list"],
                "inference_times": task_info["inference_times"],
                "per_episode_inference_counts": task_info["per_episode_inference_counts"],
                "successful_inference_counts": task_info["successful_inference_counts"],
                "avg_successful_inference_times": task_info["avg_successful_inference_times"],
                "adaptive_chunker_runtime_config": task_info["adaptive_chunker_runtime_config"],
            }
            launch_interval_results[f"{suite_name}_{task_id}"] = task_info["launch_interval_stats"]
            overall_successes.extend(task_info["success_list"])
            suite_successful_inference_counts[suite_name].extend(task_info["successful_inference_counts"])

            logging.info(f"Task Done. SR: {task_info['sr']*100:.1f}%")

    total_time = time.time() - start_time
    all_steps = []
    all_successful_steps = []
    for task_result in results_per_task.values():
        all_steps.extend(task_result["steps_list"])
        all_successful_steps.extend(task_result["successful_steps_list"])

    runtime_config = _collect_eval_runtime_config(cfg, None)
    for task_result in results_per_task.values():
        task_runtime_config = task_result.get("adaptive_chunker_runtime_config")
        if task_runtime_config is not None:
            runtime_config["adaptive_chunker"] = task_runtime_config
            break

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
            "total_steps": sum(all_steps) if all_steps else 0,
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

    logging.info("=" * 50)
    logging.info(f"FINAL OVERALL SR: {final_results['overall']['avg_success_rate']*100:.2f}%")
    logging.info(f"Results saved to {out_dir}")
    logging.info(f"Runtime config saved to {out_dir / 'runtime_config.yaml'}")
    logging.info("=" * 50)

    close_envs(all_envs_dict)

    num_tasks = len(results_per_task)
    ep_per_task = cfg.max_episodes
    total_test_count = len(overall_successes)

    logging.info("=" * 50)
    logging.info("评估统计报告:")
    logging.info(f" - 环境集合 (Suite): {cfg.env_task}")
    logging.info(f" - 任务总数 (Num Tasks): {num_tasks}")
    logging.info(f" - 每任务测试数 (Episodes per Task): {ep_per_task}")
    logging.info(f" - 总测试规模 (Total Episodes): {total_test_count}")
    logging.info(f" - 最终总成功率 (Overall SR): {np.mean(overall_successes)*100:.2f}%")
    logging.info(f" - 平均每episode步骤数 (Avg Steps per Episode): {np.mean(all_steps):.2f}")
    logging.info(
        f" - 成功episode平均步骤数 (Avg Steps per Successful Episode): "
        f"{(float(np.mean(all_successful_steps)) if all_successful_steps else float('nan')):.2f}"
    )
    logging.info(f" - 总步骤数 (Total Steps): {sum(all_steps)}")
    logging.info(f" - 自适应chunk size: {'启用' if cfg.use_adaptive_chunk else '禁用'}")
    if cfg.use_adaptive_chunk:
        logging.info(f" - Chunk size范围: {cfg.adaptive_min_horizon} - {cfg.adaptive_max_horizon}")
    for suite_name, counts in suite_successful_inference_counts.items():
        logging.info(
            f" - Suite {suite_name} 成功episode平均推理次数 (Avg Inference Count on Success): "
            f"{(float(np.mean(counts)) if counts else float('nan')):.2f}"
        )
    logging.info("=" * 50)


if __name__ == "__main__":
    run_benchmark()
