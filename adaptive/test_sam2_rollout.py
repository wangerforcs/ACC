#!/usr/bin/env python3
"""
Online-style SAM2 rollout test.

Reads a video frame by frame and feeds each frame into the rollout-style
AdaptiveActionChunker, which internally mixes periodic DINO refresh with
windowed SAM2 tracking.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from adaptive.adaptivepolicyfinegrained_sam2_rollout import AdaptiveActionChunker
from adaptive.object_extractor.mock_extractor import extract_objects


def draw_boxes(frame, boxes_xyxy, phrases, source=None, frame_idx=None, elapsed_ms=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    status_items = []
    if source is not None:
        status_items.append(f"source: {source}")
    if frame_idx is not None:
        status_items.append(f"frame: {frame_idx}")
    if elapsed_ms is not None:
        status_items.append(f"time: {elapsed_ms:.1f} ms")
    if status_items:
        cv2.putText(
            vis,
            " | ".join(status_items),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
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
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Frame-by-frame online SAM2 rollout test.")
    parser.add_argument("--video", required=True, help="Path to an mp4 file.")
    parser.add_argument("--instruction", required=True, help="Task instruction used to extract target objects.")
    parser.add_argument(
        "--output-dir",
        default="/home/wangzb/vlash/adaptive/sam2_rollout_results",
        help="Directory for annotated video and metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on processed frames; -1 means full video.",
    )
    parser.add_argument(
        "--dino-refresh-interval",
        type=int,
        default=10,
        help="Refresh DINO every N queries; SAM2 rollout tracks in between.",
    )
    parser.add_argument(
        "--sam2-window-size",
        type=int,
        default=24,
        help="Rolling frame window size used by online SAM2 propagation.",
    )
    parser.add_argument(
        "--sam2-model-id",
        default="facebook/sam2.1-hiera-tiny",
        help="SAM2 Hugging Face model id.",
    )
    parser.add_argument("--sam2-config", default=None, help="Optional SAM2 config yaml path.")
    parser.add_argument("--sam2-ckpt", default=None, help="Optional SAM2 checkpoint path.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    parsed_objs = extract_objects(args.instruction)
    if not parsed_objs:
        raise RuntimeError(f"Failed to extract objects from instruction: {args.instruction}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height, width = first_frame.shape[:2]
    video_out = out_dir / f"{video_path.stem}_sam2_rollout.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    chunker = AdaptiveActionChunker(
        dino_refresh_interval=args.dino_refresh_interval,
        sam2_window_size=args.sam2_window_size,
        sam2_model_id=args.sam2_model_id,
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
    )

    records = []
    frame = first_frame
    frame_idx = 0
    total_inference_ms = 0.0
    wall_start = time.perf_counter()
    while True:
        if args.max_frames >= 0 and frame_idx >= args.max_frames:
            break

        start = time.perf_counter()
        _, boxes_xyxy, logits, phrases, _ = chunker.get_dino_boxes(frame, parsed_objs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_inference_ms += elapsed_ms
        source = "dino" if chunker.last_query_used_dino else "sam2"
        vis = draw_boxes(frame, boxes_xyxy, phrases, source=source, frame_idx=frame_idx, elapsed_ms=elapsed_ms)
        writer.write(vis)

        records.append(
            {
                "frame_idx": frame_idx,
                "mode": source,
                "phrases": phrases,
                "boxes_xyxy": boxes_xyxy.tolist() if len(boxes_xyxy) > 0 else [],
                "logits": logits.tolist() if len(logits) > 0 else [],
                "elapsed_ms": elapsed_ms,
            }
        )

        frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            break

    cap.release()
    writer.release()
    wall_total_s = time.perf_counter() - wall_start
    avg_inference_ms = total_inference_ms / max(frame_idx, 1)
    avg_fps = frame_idx / max(wall_total_s, 1e-9)

    metadata = {
        "video": str(video_path),
        "instruction": args.instruction,
        "parsed_objects": parsed_objs,
        "dino_refresh_interval": args.dino_refresh_interval,
        "sam2_window_size": args.sam2_window_size,
        "sam2_model_id": args.sam2_model_id,
        "sam2_config": args.sam2_config,
        "sam2_ckpt": args.sam2_ckpt,
        "frames_processed": frame_idx,
        "total_inference_ms": total_inference_ms,
        "avg_inference_ms": avg_inference_ms,
        "wall_total_s": wall_total_s,
        "avg_fps": avg_fps,
        "output_video": str(video_out),
        "records": records,
    }
    metadata_path = out_dir / f"{video_path.stem}_sam2_rollout.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"SAM2 rollout video saved to: {video_out}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Total inference time: {total_inference_ms:.1f} ms")
    print(f"Average inference time: {avg_inference_ms:.1f} ms/frame")
    print(f"Wall time: {wall_total_s:.2f} s")
    print(f"Approx FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
