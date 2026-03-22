#!/usr/bin/env python3
"""
Measure GPU idle power with or without creating a CUDA context.

Examples:
  python /home/wangzb/vlash/measure_cuda_context_power.py --gpu-id 0 --duration 30
  python /home/wangzb/vlash/measure_cuda_context_power.py --gpu-id 0 --duration 30 --no-context
"""

import argparse
import json
import threading
import time
from pathlib import Path

import pynvml
import torch


class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        self.power_samples = []
        self.keep_running = False
        self.thread = None
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)

    def _monitor(self):
        while self.keep_running:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_samples.append(power / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def start(self):
        if not self.keep_running:
            self.keep_running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()

    def stop(self):
        if self.keep_running:
            self.keep_running = False
            if self.thread:
                self.thread.join()

    def clear(self):
        self.power_samples = []

    def get_report(self):
        if not self.power_samples:
            return None

        avg_power = sum(self.power_samples) / len(self.power_samples)
        max_power = max(self.power_samples)
        total_duration = len(self.power_samples) * self.interval
        energy_j = avg_power * total_duration

        return {
            "gpu_name": self.gpu_name,
            "avg_power_w": round(avg_power, 2),
            "max_power_w": round(max_power, 2),
            "total_energy_j": round(energy_j, 2),
            "total_time_s": round(total_duration, 2),
            "samples_count": len(self.power_samples),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure GPU power with optional CUDA context creation.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index for NVML / CUDA.")
    parser.add_argument("--interval", type=float, default=0.01, help="Power sampling interval in seconds.")
    parser.add_argument("--warmup", type=float, default=2.0, help="Warmup sleep before measurement in seconds.")
    parser.add_argument("--duration", type=float, default=30.0, help="Measurement duration in seconds.")
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Do not create a CUDA context before measuring.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cuda_context_power.json",
        help="Where to save the JSON report.",
    )
    return parser.parse_args()


def maybe_create_cuda_context(gpu_id: int):
    if not torch.cuda.is_available():
        return False
    device = torch.device(f"cuda:{gpu_id}")
    with torch.no_grad():
        x = torch.zeros((1,), device=device)
        _ = x + 1
    torch.cuda.synchronize(device)
    return True


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    monitor = GPUMonitor(gpu_id=args.gpu_id, interval=args.interval)

    context_created = False
    if not args.no_context:
        context_created = maybe_create_cuda_context(args.gpu_id)

    if args.warmup > 0:
        time.sleep(args.warmup)

    monitor.clear()
    monitor.start()
    start = time.time()
    time.sleep(args.duration)
    elapsed = time.time() - start
    monitor.stop()

    report = monitor.get_report() or {}
    result = {
        "gpu_id": args.gpu_id,
        "interval_s": args.interval,
        "warmup_s": args.warmup,
        "duration_s": args.duration,
        "measured_elapsed_s": elapsed,
        "create_cuda_context": not args.no_context,
        "context_created": context_created,
        "torch_cuda_available": torch.cuda.is_available(),
        "monitor": report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
