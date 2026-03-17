#!/usr/bin/env python3
"""
基于 evalpi05_adaptivefine.py 的随机动作评测版本。

特点：
- 不加载 policy / dataset
- 正常创建 LIBERO vector env 并 rollout
- 每步使用随机动作
- 保留 GPUMonitor 功耗统计
"""

import json
import logging
import time
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
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm

from adaptive.monitor import GPUMonitor


@dataclass
class RandomEvalConfig:
    env_type: str = "libero"
    env_task: str = "libero_object"
    max_episodes: int = 1
    eval_batch_size: int = 1
    seed: int = 42
    output_dir: str = "outputs/eval/random_monitor"
    task_ids: Optional[List[int]] = None
    single_task: Optional[str] = None
    use_monitor: bool = True
    monitor_gpu_id: int = 0
    monitor_interval: float = 0.05


def _sample_random_actions(vec_env: gym.vector.VectorEnv, batch_size: int) -> np.ndarray:
    """为每个并行环境采样一个随机动作。"""
    if hasattr(vec_env, "single_action_space"):
        single_action_space = vec_env.single_action_space
        actions = np.stack([single_action_space.sample() for _ in range(batch_size)], axis=0)
    else:
        sampled = vec_env.action_space.sample()
        sampled = np.asarray(sampled)
        if sampled.shape[:1] == (batch_size,):
            actions = sampled
        elif sampled.shape == vec_env.action_space.shape:
            actions = sampled
        else:
            actions = np.stack([vec_env.action_space.sample() for _ in range(batch_size)], axis=0)

    action_dtype = getattr(vec_env.action_space, "dtype", None)
    if action_dtype is not None:
        actions = actions.astype(action_dtype, copy=False)
    return actions


def eval_single_vec_env(
    vec_env: gym.vector.VectorEnv,
    cfg: RandomEvalConfig,
    task_description: str,
) -> Dict[str, Any]:
    total_episodes = cfg.max_episodes
    n_batches = (total_episodes + cfg.eval_batch_size - 1) // cfg.eval_batch_size
    max_steps = vec_env.call("_max_episode_steps")[0]

    successes: List[bool] = []
    episode_steps: List[int] = []

    monitor = GPUMonitor(gpu_id=cfg.monitor_gpu_id, interval=cfg.monitor_interval)
    if cfg.use_monitor:
        monitor.start()

    try:
        for batch_idx in range(n_batches):
            batch_seed = cfg.seed + batch_idx
            _, info = vec_env.reset(seed=batch_seed)

            dones = np.array([False] * cfg.eval_batch_size)
            batch_success = np.array([False] * cfg.eval_batch_size)
            batch_steps = np.zeros(cfg.eval_batch_size, dtype=int)

            pbar = trange(
                max_steps,
                desc=f" Batch {batch_idx + 1}/{n_batches}",
                leave=False,
                disable=inside_slurm(),
            )
            for _ in pbar:
                if np.all(dones):
                    break

                actions = _sample_random_actions(vec_env, cfg.eval_batch_size)
                _, _, terminated, truncated, info = vec_env.step(actions)

                batch_steps[~dones] += 1

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
    finally:
        monitor_stats = None
        if cfg.use_monitor:
            monitor.stop()
            monitor_stats = monitor.get_report()
            logging.info(f"GPU Monitor Stats: {monitor_stats}")

    successes = successes[:total_episodes]
    episode_steps = episode_steps[:total_episodes]

    return {
        "sr": float(np.mean(successes)) if successes else 0.0,
        "count": len(successes),
        "success_list": successes,
        "steps_list": episode_steps,
        "avg_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        "monitor": monitor_stats,
        "task_description": task_description,
    }


@parser.wrap()
def run_benchmark(cfg: RandomEvalConfig):
    init_logging()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.env_type != "libero":
        raise ValueError(f"Env type {cfg.env_type} not supported")

    env_cfg = LiberoEnv(task=cfg.env_task, task_ids=cfg.task_ids)

    logging.info(
        "Running random-action evaluation | env_task=%s | max_episodes=%s | batch_size=%s | monitor=%s",
        cfg.env_task,
        cfg.max_episodes,
        cfg.eval_batch_size,
        cfg.use_monitor,
    )

    all_envs_dict = make_env(env_cfg, n_envs=cfg.eval_batch_size)

    results_per_task: Dict[str, Any] = {}
    overall_successes: List[bool] = []
    all_steps: List[int] = []
    monitor_reports: Dict[str, Any] = {}
    start_time = time.time()

    try:
        for suite_name, tasks in all_envs_dict.items():
            logging.info(f"Evaluating Suite: {suite_name}")
            for task_id, vec_env in tasks.items():
                try:
                    task_description = vec_env.get_attr("task_description")[0]
                except Exception:
                    task_description = cfg.single_task

                logging.info(f"Starting Task [{task_id}]: {task_description}")
                task_key = f"{suite_name}_{task_id}"
                task_info = eval_single_vec_env(vec_env, cfg, task_description)

                results_per_task[task_key] = {
                    "task_description": task_info["task_description"],
                    "success_rate": task_info["sr"],
                    "num_episodes": task_info["count"],
                    "avg_steps": task_info["avg_steps"],
                    "steps_list": task_info["steps_list"],
                    "monitor": task_info["monitor"],
                }
                if task_info["monitor"] is not None:
                    monitor_reports[task_key] = task_info["monitor"]
                overall_successes.extend(task_info["success_list"])
                all_steps.extend(task_info["steps_list"])

                logging.info(f"Task Done. SR: {task_info['sr'] * 100:.1f}%")
    finally:
        close_envs(all_envs_dict)

    total_time = time.time() - start_time
    final_results = {
        "overall": {
            "avg_success_rate": float(np.mean(overall_successes)) if overall_successes else 0.0,
            "total_episodes": len(overall_successes),
            "total_eval_time_s": total_time,
            "env_task": cfg.env_task,
            "avg_steps_per_episode": float(np.mean(all_steps)) if all_steps else 0.0,
            "total_steps": int(sum(all_steps)) if all_steps else 0,
            "random_policy": True,
            "use_monitor": cfg.use_monitor,
        },
        "per_task": results_per_task,
        "monitor_reports": monitor_reports,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    logging.info("=" * 50)
    logging.info(f"FINAL OVERALL SR: {final_results['overall']['avg_success_rate'] * 100:.2f}%")
    logging.info(f"Results saved to {out_dir}")
    logging.info("=" * 50)
    logging.info(f"评估统计报告:")
    logging.info(f" - 环境集合 (Suite): {cfg.env_task}")
    logging.info(f" - 任务总数 (Num Tasks): {len(results_per_task)}")
    logging.info(f" - 每任务测试数 (Episodes per Task): {cfg.max_episodes}")
    logging.info(f" - 总测试规模 (Total Episodes): {len(overall_successes)}")
    logging.info(f" - 最终总成功率 (Overall SR): {final_results['overall']['avg_success_rate'] * 100:.2f}%")
    logging.info(f" - 平均每episode步骤数 (Avg Steps per Episode): {final_results['overall']['avg_steps_per_episode']:.2f}")
    logging.info(f" - 总步骤数 (Total Steps): {final_results['overall']['total_steps']}")
    logging.info(f" - Monitor: {'启用' if cfg.use_monitor else '禁用'}")
    logging.info("=" * 50)


if __name__ == "__main__":
    run_benchmark()


# python evalpi05_random_monitor.py --env_task=libero_object --task_ids=[4] --output_dir=testadapt --use_monitor=True