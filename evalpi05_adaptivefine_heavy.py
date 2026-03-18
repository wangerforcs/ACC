#!/usr/bin/env python3
"""
Heavy adaptive eval.

This is a thin wrapper around the shared norm eval implementation with a
different adaptive chunker policy and output defaults.
"""

from dataclasses import dataclass

from lerobot.configs import parser

from evalpi05_adaptivefine_norm import (
    VLASHEvalConfig as BaseEvalConfig,
    run_benchmark_common,
)


@dataclass
class VLASHEvalConfig(BaseEvalConfig):
    output_dir: str = "outputs/eval/vlash_adaptive_heavy"
    adaptive_chunker_cls_path: str = (
        "adaptive.adaptivepolicyfinegrained_tracker_heavy:AdaptiveActionChunker"
    )
    tracker_logger_names: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tracker_logger_names is None:
            self.tracker_logger_names = [
                "adaptive.tracker",
                "adaptive.tracker_scoregate",
                "adaptive.tracker_heavy",
            ]
        super().__post_init__()


@parser.wrap()
def run_benchmark(cfg: VLASHEvalConfig):
    run_benchmark_common(cfg)


if __name__ == "__main__":
    run_benchmark()
