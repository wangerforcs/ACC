#!/usr/bin/env python3
"""Measure VLA policy inference latency for the current policy.compile_model setting."""

import copy
import gzip
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.utils.utils import init_logging

from evalpi05_adaptivefine_norm import VLASHEvalConfig
from vlash.policies.factory import get_policy_class
from vlash.train import make_vlash_dataset


@dataclass
class MeasureVLALatencyConfig(VLASHEvalConfig):
    trace_path: str = "/home/wangzb/vlash/power_traces/object9.pkl.gz"
    trace_step_idx: int = 0
    warmup_iters: int = 5
    measure_iters: int = 20
    output_json: str = "/home/wangzb/vlash/vla_latency_results.json"


def _load_trace_obs(trace_path: Path, step_idx: int) -> Dict[str, Any]:
    with gzip.open(trace_path, "rb") as f:
        trace_data = pickle.load(f)
    obs = trace_data["inference_inputs"][step_idx]
    return obs


def _ensure_batch_obs(single_obs: Dict[str, Any]) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}

    if "agent_pos" in single_obs:
        agent_pos = np.asarray(single_obs["agent_pos"])
        if agent_pos.ndim == 1:
            agent_pos = agent_pos[None, :]
        batched["agent_pos"] = agent_pos

    if "pixels" in single_obs and isinstance(single_obs["pixels"], dict):
        batched_pixels = {}
        for key, value in single_obs["pixels"].items():
            image = np.asarray(value)
            if image.ndim == 3:
                image = image[None, ...]
            batched_pixels[key] = image
        batched["pixels"] = batched_pixels

    return batched


def _prepare_batch_obs(raw_obs: Dict[str, Any], task_description: str, device: torch.device):
    batch_input = {}

    if "agent_pos" in raw_obs:
        batch_input["observation.state"] = torch.from_numpy(raw_obs["agent_pos"]).float().to(device)

    if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
        for key, value in raw_obs["pixels"].items():
            img_tensor = torch.from_numpy(value).float().permute(0, 3, 1, 2).to(device) / 255.0
            batch_input[f"observation.images.{key}"] = img_tensor

    batch_input["task"] = [task_description] * raw_obs["agent_pos"].shape[0]
    return batch_input


def _synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_one_mode(
    cfg: MeasureVLALatencyConfig,
    batched_obs: Dict[str, Any],
    task_description: str,
    dataset_stats,
    policy_cls,
) -> Dict[str, Any]:
    policy_cfg = copy.deepcopy(cfg.policy)
    mode_name = "compile=true" if bool(policy_cfg.compile_model) else "compile=false"

    logging.info("Loading policy for %s", mode_name)
    load_t0 = time.perf_counter()
    policy = policy_cls.from_pretrained(
        cfg.policy.pretrained_path,
        config=policy_cfg,
        dataset_stats=dataset_stats,
    )
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    logging.info("Policy load finished for %s in %.1f ms", mode_name, load_ms)
    policy.eval()

    device = torch.device(policy_cfg.device)
    batch = _prepare_batch_obs(batched_obs, task_description, device)

    first_call_ms = None
    warmup_ms = []
    measured_ms = []

    with torch.inference_mode():
        logging.info(
            "Starting warmup for %s (warmup_iters=%s, measure_iters=%s)",
            mode_name,
            cfg.warmup_iters,
            cfg.measure_iters,
        )
        for i in range(cfg.warmup_iters):
            if i == 0:
                logging.info("Running first warmup call for %s; this may take a while if compilation is triggered", mode_name)
            _synchronize_if_needed(device)
            t0 = time.perf_counter()
            _ = policy.predict_action_chunk(batch)
            _synchronize_if_needed(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            warmup_ms.append(dt_ms)
            if i == 0:
                first_call_ms = dt_ms

        logging.info("Warmup finished for %s; starting measured iterations", mode_name)
        for _ in range(cfg.measure_iters):
            _synchronize_if_needed(device)
            t0 = time.perf_counter()
            _ = policy.predict_action_chunk(batch)
            _synchronize_if_needed(device)
            measured_ms.append((time.perf_counter() - t0) * 1000.0)

    measured_arr = np.asarray(measured_ms, dtype=np.float64)
    warmup_arr = np.asarray(warmup_ms, dtype=np.float64)

    return {
        "compile_model": bool(policy_cfg.compile_model),
        "device": str(device),
        "policy_load_ms": float(load_ms),
        "warmup_iters": cfg.warmup_iters,
        "measure_iters": cfg.measure_iters,
        "first_call_ms": float(first_call_ms) if first_call_ms is not None else None,
        "warmup_avg_ms": float(warmup_arr.mean()) if len(warmup_arr) else None,
        "measured_avg_ms": float(measured_arr.mean()) if len(measured_arr) else None,
        "measured_min_ms": float(measured_arr.min()) if len(measured_arr) else None,
        "measured_max_ms": float(measured_arr.max()) if len(measured_arr) else None,
        "measured_p50_ms": float(np.percentile(measured_arr, 50)) if len(measured_arr) else None,
        "measured_p95_ms": float(np.percentile(measured_arr, 95)) if len(measured_arr) else None,
        "all_measured_ms": [float(x) for x in measured_ms],
    }


@parser.wrap()
def main(cfg: MeasureVLALatencyConfig):
    init_logging()

    trace_path = Path(cfg.trace_path)
    logging.info("Loading trace input from %s (step=%s)", trace_path, cfg.trace_step_idx)
    raw_obs = _load_trace_obs(trace_path, cfg.trace_step_idx)
    batched_obs = _ensure_batch_obs(raw_obs)

    task_description = "pick up the orange juice and place it in the basket"
    logging.info("Building dataset stats and resolving policy class")
    dataset = make_vlash_dataset(cfg.train_cfg)
    policy_cls = get_policy_class(cfg.policy.type)

    results = {
        "trace_path": str(trace_path),
        "trace_step_idx": cfg.trace_step_idx,
        "task_description": task_description,
        "policy_path": cfg.policy.pretrained_path,
        "n_action_steps": cfg.policy.n_action_steps,
        "compile_model": bool(cfg.policy.compile_model),
        "result": _measure_one_mode(
            cfg, batched_obs, task_description, dataset.meta.stats, policy_cls
        ),
    }

    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
