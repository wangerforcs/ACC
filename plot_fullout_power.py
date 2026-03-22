# python /home/wangzb/vlash/export_fullout.py
# python /home/wangzb/vlash/plot_fullout_power.py




#!/usr/bin/env python3
"""Plot power curves from fullout.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


X_LABELS = ["sync", "async1", "async2", "async3"]
SERIES_ORDER = [
    ("adaptive", "Adapt (DINO=20)"),
    ("dino1", "Adapt (DINO=1x)"),
    ("fixed20", "Fixed 20"),
    ("fixed10", "Fixed 10"),
    ("fixed5", "Fixed 5"),
]


def plot_metric(
    grouped: dict[str, dict[str, dict[str, float]]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for task_key, label in SERIES_ORDER:
        y_values = []
        for x_label in X_LABELS:
            value = grouped.get(x_label, {}).get(task_key, {}).get(metric)
            y_values.append(value if value is not None else math.nan)
        ax.plot(X_LABELS, y_values, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Mode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Across Sync / Async Modes")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot power curves from fullout.json.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/wangzb/vlash/fullout.json"),
        help="Path to the exported fullout json summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wangzb/vlash/plots/fullout_power"),
        help="Directory for generated plots.",
    )
    args = parser.parse_args()

    grouped = json.loads(args.input.read_text(encoding="utf-8"))
    plot_metric(grouped, "avg_power_w", "Average Power (W)", args.output_dir / "avg_power_lines.png")
    plot_metric(grouped, "total_energy_j", "Total Energy (J)", args.output_dir / "total_energy_lines.png")
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
