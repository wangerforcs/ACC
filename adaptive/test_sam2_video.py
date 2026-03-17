#!/usr/bin/env python3
"""
Track DINO-initialized objects in an mp4 video with SAM2 propagation.
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

from adaptive.adaptivepolicyfinegrained_sam2 import AdaptiveActionChunker


def draw_boxes(frame, boxes_xyxy, phrases, source="sam2", frame_idx=None, elapsed_ms=None):
    vis = frame.copy()
    status_items = [f"source: {source}"]
    if frame_idx is not None:
        status_items.append(f"frame: {frame_idx}")
    if elapsed_ms is not None:
        status_items.append(f"time: {elapsed_ms:.1f} ms")
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

    h, w = vis.shape[:2]
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
    parser = argparse.ArgumentParser(description="Track DINO-initialized objects with SAM2 in an mp4 video.")
    parser.add_argument("--video", required=True, help="Path to an mp4 file.")
    parser.add_argument("--instruction", required=True, help="Task instruction used to extract target objects.")
    parser.add_argument(
        "--output-dir",
        default="/home/wangzb/vlash/adaptive/sam2_video_results",
        help="Directory for annotated video and metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on propagated frames. Default tracks the full video.",
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        if args.max_frames is not None and len(frames) >= args.max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to decode any frames from {video_path}")

    chunker = AdaptiveActionChunker(
        sam2_model_id=args.sam2_model_id,
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
    )

    video_out = out_dir / f"{video_path.stem}_sam2.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    records = []
    start = time.perf_counter()
    for result in chunker.propagate_video(
        video_path=str(video_path),
        instruction=args.instruction,
        max_frames=args.max_frames,
    ):
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        frame_idx = result["frame_idx"]
        if frame_idx >= len(frames):
            break

        boxes_xyxy = result["boxes_xyxy"]
        phrases = result["phrases"]
        vis = draw_boxes(
            frames[frame_idx],
            boxes_xyxy,
            phrases,
            source="sam2",
            frame_idx=frame_idx,
            elapsed_ms=elapsed_ms,
        )
        writer.write(vis)

        records.append(
            {
                "frame_idx": frame_idx,
                "mode": "sam2",
                "phrases": phrases,
                "boxes_xyxy": boxes_xyxy.tolist() if len(boxes_xyxy) > 0 else [],
                "mask_areas": result["mask_areas"],
                "elapsed_ms_total": elapsed_ms,
            }
        )

    writer.release()

    metadata = {
        "video": str(video_path),
        "instruction": args.instruction,
        "sam2_model_id": args.sam2_model_id,
        "sam2_config": args.sam2_config,
        "sam2_ckpt": args.sam2_ckpt,
        "frames_processed": len(records),
        "output_video": str(video_out),
        "records": records,
    }
    metadata_path = out_dir / f"{video_path.stem}_sam2.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"SAM2 tracked video saved to: {video_out}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
