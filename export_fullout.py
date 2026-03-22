#!/usr/bin/env python3
"""Export monitor summaries from power run directories into fullout.txt and fullout.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/wangzb/vlash/power-runs_nocompiled_valuefuncasync")
OUTPUT_PATH = ROOT / "fullout.txt"
OUTPUT_JSON_PATH = ROOT / "fullout.json"

SECTION_DIRS = [
    ("sync", ROOT / "power_runs_sync"),
    ("async1", ROOT / "power_runs_async1"),
    ("async2", ROOT / "power_runs_async2"),
    ("async3", ROOT / "power_runs_async3"),
]

TASKS = [
    ("adaptive", "object9_replay"),
    ("dino1", "object9_replay_dino1"),
    ("fixed20", "object9_replay_fixed20"),
    ("fixed10", "object9_replay_fixed10"),
    ("fixed5", "object9_replay_fixed5"),
]


def load_monitor(results_path: Path) -> dict:
    text = results_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("empty results file")
    data = json.loads(text)
    return data["result"]["monitor"]


def format_monitor_block(monitor: dict) -> str:
    monitor_json = json.dumps(monitor, ensure_ascii=False, indent=2)
    return f'"monitor": {monitor_json},'


def build_output() -> str:
    lines = [",".join(name for name, _ in TASKS)]
    for section_name, section_dir in SECTION_DIRS:
        lines.append(section_name)
        for task_name, task_dir_name in TASKS:
            results_path = section_dir / task_dir_name / "full_benchmark_results.json"
            if not results_path.exists():
                lines.append(f"MISSING: {results_path}")
                lines.append("")
                continue
            try:
                monitor = load_monitor(results_path)
            except Exception as exc:
                lines.append(f"INVALID: {section_name}/{task_name}: {exc}")
                lines.append("")
                continue
            lines.append(format_monitor_block(monitor))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_json_output() -> dict:
    output = {}
    for section_name, section_dir in SECTION_DIRS:
        section_data = {}
        for task_name, task_dir_name in TASKS:
            results_path = section_dir / task_dir_name / "full_benchmark_results.json"
            if not results_path.exists():
                continue
            try:
                section_data[task_name] = load_monitor(results_path)
            except Exception:
                continue
        output[section_name] = section_data
    return output


def main() -> None:
    text = build_output()
    json_data = build_json_output()
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    OUTPUT_JSON_PATH.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {OUTPUT_PATH}")
    print(f"Saved summary to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
