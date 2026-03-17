#!/usr/bin/env python3
"""
Batch export tracker and SAM2 tracking videos for the first episode of each LIBERO subtask.

This version is optimized to avoid repeated model loads:
- one tracker chunker instance for all tracker exports
- one SAM2 chunker instance for all SAM2 exports
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from adaptive.object_extractor.mock_extractor import extract_objects
from adaptive.adaptivepolicyfinegrained_sam2image import AdaptiveActionChunker as SAM2Chunker
# from adaptive.adaptivepolicyfinegrained_tracker import AdaptiveActionChunker as TrackerChunker
from adaptive.adaptivepolicyfinegrained_tracker_scoregate import AdaptiveActionChunker as TrackerChunker


SUITE_TO_EVAL_DIR = {
    "libero_object": "eval_logs_object",
    "libero_spatial": "eval_logs_spatial",
    "libero_goal": "eval_logs_goal",
    "libero_10": "eval_logs_10",
}

TASK_INSTRUCTIONS = {
    "libero_object": {
        0: "pick up the alphabet soup and place it in the basket",
        1: "pick up the cream cheese and place it in the basket",
        2: "pick up the salad dressing and place it in the basket",
        3: "pick up the bbq sauce and place it in the basket",
        4: "pick up the ketchup and place it in the basket",
        5: "pick up the tomato sauce and place it in the basket",
        6: "pick up the butter and place it in the basket",
        7: "pick up the milk and place it in the basket",
        8: "pick up the chocolate pudding and place it in the basket",
        9: "pick up the orange juice and place it in the basket",
    },
    "libero_spatial": {
        0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
        1: "pick up the black bowl next to the ramekin and place it on the plate",
        2: "pick up the black bowl from table center and place it on the plate",
        3: "pick up the black bowl on the cookie box and place it on the plate",
        4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        5: "pick up the black bowl on the ramekin and place it on the plate",
        6: "pick up the black bowl next to the cookie box and place it on the plate",
        7: "pick up the black bowl on the stove and place it on the plate",
        8: "pick up the black bowl next to the plate and place it on the plate",
        9: "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
    "libero_goal": {
        0: "open the middle drawer of the cabinet",
        1: "put the bowl on the stove",
        2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside",
        4: "put the bowl on top of the cabinet",
        5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl",
        7: "turn on the stove",
        8: "put the bowl on the plate",
        9: "put the wine bottle on the rack",
    },
    "libero_10": {
        0: "put both the alphabet soup and the tomato sauce in the basket",
        1: "put both the cream cheese box and the butter in the basket",
        2: "turn on the stove and put the moka pot on it",
        3: "put the black bowl in the bottom drawer of the cabinet and close it",
        4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
        5: "pick up the book and place it in the back compartment of the caddy",
        6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        7: "put both the alphabet soup and the cream cheese box in the basket",
        8: "put both moka pots on the stove",
        9: "put the yellow and white mug in the microwave and close it",
    },
}


def extract_episode_idx(path: Path) -> int:
    match = re.search(r"eval_episode_(\d+)\.mp4$", path.name)
    return int(match.group(1)) if match else 10**9


def get_first_video_per_subtask(eval_dir: Path) -> Dict[int, Path]:
    videos_dir = eval_dir / "videos"
    results: Dict[int, Path] = {}
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_match = re.search(r"_(\d+)$", task_dir.name)
        if not task_match:
            continue
        task_id = int(task_match.group(1))
        video_files = sorted(task_dir.glob("eval_episode_*.mp4"), key=extract_episode_idx)
        if video_files:
            results[task_id] = video_files[0]
    return results


def _format_score(scores, idx):
    if scores is None:
        return None
    try:
        value = scores[idx]
    except Exception:
        return None

    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def draw_boxes(frame, boxes_xyxy, phrases, scores=None, source=None, frame_idx=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    status_items = []
    if source is not None:
        status_items.append(f"source: {source}")
    if frame_idx is not None:
        status_items.append(f"frame: {frame_idx}")
    if status_items:
        cv2.putText(
            vis,
            " | ".join(status_items),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return vis

    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = int(np.clip(x1 * w, 0, w - 1))
        y1 = int(np.clip(y1 * h, 0, h - 1))
        x2 = int(np.clip(x2 * w, 0, w - 1))
        y2 = int(np.clip(y2 * h, 0, h - 1))
        label = phrases[idx] if idx < len(phrases) else f"obj_{idx}"
        score = _format_score(scores, idx)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = max(18, y1 - 18)
        cv2.putText(
            vis,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if score is not None:
            cv2.putText(
                vis,
                f"score: {score:.3f}",
                (x1, min(h - 6, label_y + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return vis


def write_video_with_mediapy(video_out: Path, frames: List[np.ndarray], fps: float) -> None:
    try:
        import mediapy as media
    except ImportError as exc:
        raise RuntimeError(
            "mediapy is required for video export. Install it with `pip install mediapy`."
        ) from exc

    if not frames:
        raise RuntimeError(f"No frames collected for output video: {video_out}")

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    media.write_video(str(video_out), rgb_frames, fps=fps)


def export_with_tracker(
    chunker: TrackerChunker,
    video_path: Path,
    instruction: str,
    output_dir: Path,
    max_frames: int,
    pure_tracker: bool,
    output_prefix: str,
) -> Dict[str, object]:
    chunker.reset()

    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_out = output_dir / f"{output_prefix}_tracked.mp4"
    output_frames: List[np.ndarray] = []

    total_inference_ms = 0.0
    frame_idx = 0
    frame = first_frame
    parsed_objs = extract_objects(instruction)
    if not parsed_objs:
        raise RuntimeError(f"Failed to extract objects from instruction: {instruction}")

    while True:
        if max_frames >= 0 and frame_idx >= max_frames:
            break

        start = time.perf_counter()
        if pure_tracker and frame_idx > 0:
            _, boxes_xyxy, logits, phrases, _ = chunker.get_tracked_boxes(frame, parsed_objs)
            source = "tracker"
        else:
            _, boxes_xyxy, logits, phrases, _ = chunker.get_dino_boxes(frame, parsed_objs)
            source = "dino" if chunker.last_query_used_dino else "tracker"
        total_inference_ms += (time.perf_counter() - start) * 1000.0

        output_frames.append(draw_boxes(frame, boxes_xyxy, phrases, scores=logits, source=source, frame_idx=frame_idx))
        frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            break

    cap.release()
    write_video_with_mediapy(video_out, output_frames, fps)
    return {
        "video": str(video_path),
        "instruction": instruction,
        "output_video": str(video_out),
        "frames_processed": frame_idx,
        "total_inference_ms": total_inference_ms,
        "avg_inference_ms": total_inference_ms / max(frame_idx, 1),
        "active_tracker_backend": chunker.active_tracker_backend,
    }


def export_with_sam2(
    chunker: SAM2Chunker,
    video_path: Path,
    instruction: str,
    output_dir: Path,
    max_frames: int,
    output_prefix: str,
) -> Dict[str, object]:
    chunker.reset()

    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_out = output_dir / f"{output_prefix}_sam2_image.mp4"
    output_frames: List[np.ndarray] = []

    total_inference_ms = 0.0
    frame_idx = 0
    frame = first_frame
    parsed_objs = extract_objects(instruction)
    if not parsed_objs:
        raise RuntimeError(f"Failed to extract objects from instruction: {instruction}")

    while True:
        if max_frames >= 0 and frame_idx >= max_frames:
            break

        start = time.perf_counter()
        _, boxes_xyxy, logits, phrases, _ = chunker.get_dino_boxes(frame, parsed_objs)
        total_inference_ms += (time.perf_counter() - start) * 1000.0
        source = getattr(chunker, "last_box_source", None) or ("dino" if chunker.last_query_used_dino else "sam2_image")

        output_frames.append(draw_boxes(frame, boxes_xyxy, phrases, scores=logits, source=source, frame_idx=frame_idx))
        frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            break

    cap.release()
    write_video_with_mediapy(video_out, output_frames, fps)
    return {
        "video": str(video_path),
        "instruction": instruction,
        "output_video": str(video_out),
        "frames_processed": frame_idx,
        "total_inference_ms": total_inference_ms,
        "avg_inference_ms": total_inference_ms / max(frame_idx, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Export tracker and SAM2 videos for the first episode of each LIBERO subtask.")
    parser.add_argument("--eval-root", default="/home/wangzb/vlash/lerobotevals")
    parser.add_argument("--output-root", default="/home/wangzb/vlash/lerobotevals_tracking_exports")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_object", "libero_spatial", "libero_goal", "libero_10"],
        choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"],
    )
    parser.add_argument("--tracker-backend", default="csrt")
    parser.add_argument("--tracker-dino-refresh-interval", type=int, default=20)
    parser.add_argument("--sam2-dino-refresh-interval", type=int, default=10)
    parser.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--pure-tracker", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    output_root = Path(args.output_root)
    tracker_out = output_root / "tracker"
    sam2_out = output_root / "sam2_image"
    tracker_out.mkdir(parents=True, exist_ok=True)
    sam2_out.mkdir(parents=True, exist_ok=True)

    tracker_chunker = None
    sam2_chunker = None
    if not args.skip_tracker:
        tracker_chunker = TrackerChunker(
            dino_refresh_interval=args.tracker_dino_refresh_interval,
            tracker_backend=args.tracker_backend,
        )
    if not args.skip_sam2:
        sam2_chunker = SAM2Chunker(
            dino_refresh_interval=args.sam2_dino_refresh_interval,
            sam2_model_id=args.sam2_model_id,
            sam2_config=args.sam2_config,
            sam2_ckpt=args.sam2_ckpt,
        )

    manifest = {
        "eval_root": str(eval_root),
        "tracker_output_dir": str(tracker_out),
        "sam2_output_dir": str(sam2_out),
        "items": [],
    }

    for suite in args.suites:
        eval_dir = eval_root / SUITE_TO_EVAL_DIR[suite]
        if not eval_dir.exists():
            print(f"[SKIP] Missing eval dir: {eval_dir}")
            continue

        first_videos = get_first_video_per_subtask(eval_dir)
        for task_id in sorted(first_videos):
            video_path = first_videos[task_id]
            instruction = TASK_INSTRUCTIONS[suite][task_id]
            output_prefix = f"{suite}_task{task_id}_{video_path.parent.name}_{video_path.stem}"
            item = {
                "suite": suite,
                "task_id": task_id,
                "instruction": instruction,
                "video": str(video_path),
                "output_prefix": output_prefix,
            }
            print(f"[TASK] {suite}_{task_id}: {instruction}")

            if tracker_chunker is not None:
                item["tracker"] = export_with_tracker(
                    tracker_chunker,
                    video_path,
                    instruction,
                    tracker_out,
                    args.max_frames,
                    args.pure_tracker,
                    output_prefix,
                )

            if sam2_chunker is not None:
                item["sam2"] = export_with_sam2(
                    sam2_chunker,
                    video_path,
                    instruction,
                    sam2_out,
                    args.max_frames,
                    output_prefix,
                )

            manifest["items"].append(item)

    manifest_path = output_root / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Processed subtasks: {len(manifest['items'])}")
    if tracker_chunker is not None:
        print(f"Expected tracker videos: {len(manifest['items'])}")
    if sam2_chunker is not None:
        print(f"Expected SAM2 videos: {len(manifest['items'])}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
