#!/usr/bin/env python3
"""
Frame-by-frame SAM2 image predictor test.
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

from adaptive.adaptivepolicyfinegrained_sam2image import AdaptiveActionChunker
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
    parser = argparse.ArgumentParser(description="Frame-by-frame SAM2 image predictor test.")
    parser.add_argument("--video", required=True, help="Path to an mp4 file.")
    parser.add_argument("--instruction", required=True, help="Task instruction used to extract target objects.")
    parser.add_argument(
        "--output-dir",
        default="/home/wangzb/vlash/adaptive/sam2_image_results",
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
        help="Refresh DINO every N queries; SAM2Image refines between refreshes.",
    )
    parser.add_argument(
        "--sam2-model-id",
        default="facebook/sam2.1-hiera-tiny",
        help="SAM2 Hugging Face model id.",
    )
    parser.add_argument("--sam2-config", default=None, help="Optional SAM2 config yaml path.")
    parser.add_argument("--sam2-ckpt", default=None, help="Optional SAM2 checkpoint path.")
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable annotated video writing to focus on timing.",
    )
    parser.add_argument(
        "--no-records",
        action="store_true",
        help="Do not store per-frame records in the output json.",
    )
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
    video_out = out_dir / f"{video_path.stem}_sam2_image.mp4"
    writer = None
    if not args.no_save_video:
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    chunker = AdaptiveActionChunker(
        dino_refresh_interval=args.dino_refresh_interval,
        sam2_model_id=args.sam2_model_id,
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
    )

    records = []
    frame = first_frame
    frame_idx = 0
    total_inference_ms = 0.0
    dino_inference_ms = 0.0
    sam2_inference_ms = 0.0
    dino_frame_count = 0
    sam2_frame_count = 0
    set_image_ms_total = 0.0
    predict_ms_total = 0.0
    wall_start = time.perf_counter()
    while True:
        if args.max_frames >= 0 and frame_idx >= args.max_frames:
            break

        start = time.perf_counter()
        _, boxes_xyxy, logits, phrases, _ = chunker.get_dino_boxes(frame, parsed_objs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_inference_ms += elapsed_ms
        source = getattr(chunker, "last_box_source", None) or ("dino" if chunker.last_query_used_dino else "sam2_image")
        if chunker.last_query_used_dino:
            dino_frame_count += 1
            dino_inference_ms += elapsed_ms
        else:
            sam2_frame_count += 1
            sam2_inference_ms += elapsed_ms
            set_image_ms_total += float(chunker.last_stage_timings_ms.get("set_image_ms", 0.0))
            predict_ms_total += float(chunker.last_stage_timings_ms.get("predict_ms", 0.0))

        vis = draw_boxes(frame, boxes_xyxy, phrases, source=source, frame_idx=frame_idx, elapsed_ms=elapsed_ms)
        if writer is not None:
            writer.write(vis)

        if not args.no_records:
            records.append(
                {
                    "frame_idx": frame_idx,
                    "mode": source,
                    "phrases": phrases,
                    "boxes_xyxy": boxes_xyxy.tolist() if len(boxes_xyxy) > 0 else [],
                    "logits": logits.tolist() if len(logits) > 0 else [],
                    "elapsed_ms": elapsed_ms,
                    "stage_timings_ms": dict(chunker.last_stage_timings_ms),
                }
            )

        frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            break

    cap.release()
    if writer is not None:
        writer.release()
    wall_total_s = time.perf_counter() - wall_start
    avg_inference_ms = total_inference_ms / max(frame_idx, 1)
    avg_fps = frame_idx / max(wall_total_s, 1e-9)
    inference_only_fps = frame_idx / max(total_inference_ms / 1000.0, 1e-9)
    avg_dino_ms = dino_inference_ms / max(dino_frame_count, 1)
    avg_sam2_ms = sam2_inference_ms / max(sam2_frame_count, 1)
    avg_set_image_ms = set_image_ms_total / max(sam2_frame_count, 1)
    avg_predict_ms = predict_ms_total / max(sam2_frame_count, 1)

    metadata = {
        "video": str(video_path),
        "instruction": args.instruction,
        "parsed_objects": parsed_objs,
        "dino_refresh_interval": args.dino_refresh_interval,
        "sam2_model_id": args.sam2_model_id,
        "sam2_config": args.sam2_config,
        "sam2_ckpt": args.sam2_ckpt,
        "frames_processed": frame_idx,
        "total_inference_ms": total_inference_ms,
        "avg_inference_ms": avg_inference_ms,
        "wall_total_s": wall_total_s,
        "avg_fps": avg_fps,
        "inference_only_fps": inference_only_fps,
        "dino_frame_count": dino_frame_count,
        "sam2_frame_count": sam2_frame_count,
        "dino_inference_ms": dino_inference_ms,
        "sam2_inference_ms": sam2_inference_ms,
        "avg_dino_ms": avg_dino_ms,
        "avg_sam2_ms": avg_sam2_ms,
        "sam2_set_image_ms_total": set_image_ms_total,
        "sam2_predict_ms_total": predict_ms_total,
        "avg_set_image_ms": avg_set_image_ms,
        "avg_predict_ms": avg_predict_ms,
        "output_video": None if args.no_save_video else str(video_out),
        "records": None if args.no_records else records,
    }
    metadata_path = out_dir / f"{video_path.stem}_sam2_image.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if writer is not None:
        print(f"SAM2 image video saved to: {video_out}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Total inference time: {total_inference_ms:.1f} ms")
    print(f"Average inference time: {avg_inference_ms:.1f} ms/frame")
    print(f"Inference-only FPS: {inference_only_fps:.2f}")
    print(f"DINO frames: {dino_frame_count}, total: {dino_inference_ms:.1f} ms, avg: {avg_dino_ms:.1f} ms/frame")
    print(f"SAM2 frames: {sam2_frame_count}, total: {sam2_inference_ms:.1f} ms, avg: {avg_sam2_ms:.1f} ms/frame")
    if sam2_frame_count > 0:
        print(f"SAM2 set_image avg: {avg_set_image_ms:.1f} ms/frame")
        print(f"SAM2 predict avg: {avg_predict_ms:.1f} ms/frame")
    print(f"Wall time: {wall_total_s:.2f} s")
    print(f"Approx FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
