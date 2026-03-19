#!/usr/bin/env python3
"""
Minimal OpenCV tracker benchmark.

This isolates pure tracker.update(frame) speed from any DINO/adaptive logic.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_bbox(text: str):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _get_tracker_factory(module, class_name: str):
    factory_name = f"{class_name}_create"
    factory = getattr(module, factory_name, None)
    if factory is not None:
        return factory

    cls = getattr(module, class_name, None)
    if cls is not None:
        create = getattr(cls, "create", None)
        if create is not None:
            return create
    return None


def available_backends():
    backend_to_class = {
        "csrt": "TrackerCSRT",
        "kcf": "TrackerKCF",
        "mil": "TrackerMIL",
    }
    available = []
    modules = [cv2]
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        modules.append(legacy)

    for backend, class_name in backend_to_class.items():
        for module in modules:
            if _get_tracker_factory(module, class_name) is not None:
                available.append(backend)
                break
    return available


def make_tracker(backend: str):
    backend = backend.lower()
    class_name = {
        "csrt": "TrackerCSRT",
        "kcf": "TrackerKCF",
        "mil": "TrackerMIL",
    }.get(backend)
    if class_name is None:
        raise ValueError(f"Unsupported backend: {backend}")

    modules = [cv2]
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        modules.append(legacy)

    for module in modules:
        factory = _get_tracker_factory(module, class_name)
        if factory is not None:
            return factory()

    raise RuntimeError(
        f"OpenCV tracker backend '{backend}' is not available. "
        f"Available backends: {available_backends()}"
    )


def draw_bbox(frame, bbox, ok, frame_idx, elapsed_ms):
    vis = frame.copy()
    color = (0, 255, 0) if ok else (0, 0, 255)
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        vis,
        f"frame={frame_idx} | ok={ok} | {elapsed_ms:.2f} ms",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Pure OpenCV tracker benchmark.")
    parser.add_argument("--video", required=True, help="Path to an mp4 file.")
    parser.add_argument("--bbox", required=True, help="Initial bbox in pixels: x1,y1,x2,y2")
    parser.add_argument("--backend", default="kcf", choices=["kcf", "csrt", "mil"])
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--output-dir", default="/home/wangzb/vlash/adaptive/opencv_tracker_benchmark")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    x1, y1, x2, y2 = parse_bbox(args.bbox)
    init_bbox = (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))

    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    tracker = make_tracker(args.backend)
    tracker.init(first_frame, init_bbox)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = first_frame.shape[:2]
        video_out = out_dir / f"{video_path.stem}_{args.backend}_benchmark.mp4"
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        writer.write(draw_bbox(first_frame, init_bbox, True, 0, 0.0))

    frame_idx = 1
    total_update_ms = 0.0
    successful_updates = 0
    wall_start = time.perf_counter()
    records = []
    while True:
        if args.max_frames >= 0 and frame_idx >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        start = time.perf_counter()
        track_ok, bbox = tracker.update(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_update_ms += elapsed_ms
        if track_ok:
            successful_updates += 1

        records.append(
            {
                "frame_idx": frame_idx,
                "ok": bool(track_ok),
                "bbox": [float(v) for v in bbox],
                "elapsed_ms": elapsed_ms,
            }
        )

        if writer is not None:
            writer.write(draw_bbox(frame, bbox, track_ok, frame_idx, elapsed_ms))

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    wall_total_s = time.perf_counter() - wall_start
    processed_updates = max(frame_idx - 1, 1)
    avg_update_ms = total_update_ms / processed_updates
    avg_fps = (frame_idx - 1) / max(wall_total_s, 1e-9)

    metadata = {
        "video": str(video_path),
        "backend": args.backend,
        "init_bbox": init_bbox,
        "frames_processed": frame_idx,
        "processed_updates": frame_idx - 1,
        "successful_updates": successful_updates,
        "total_update_ms": total_update_ms,
        "avg_update_ms": avg_update_ms,
        "wall_total_s": wall_total_s,
        "avg_fps": avg_fps,
        "records": records,
    }
    metadata_path = out_dir / f"{video_path.stem}_{args.backend}_benchmark.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Backend: {args.backend}")
    print(f"Processed updates: {frame_idx - 1}")
    print(f"Successful updates: {successful_updates}")
    print(f"Total update time: {total_update_ms:.1f} ms")
    print(f"Average update time: {avg_update_ms:.2f} ms/frame")
    print(f"Wall time: {wall_total_s:.2f} s")
    print(f"Approx FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
