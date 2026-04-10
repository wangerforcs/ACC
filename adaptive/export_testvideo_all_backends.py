#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from adaptive.adaptivepolicyfinegrained_tracker_scoregate import AdaptiveActionChunker as TrackerChunker
from adaptive.benchopencv import create_tracker_with_models
from adaptive.benchopencv import discover_tracker_specs
from adaptive.benchopencv import get_tracker_spec
from adaptive.object_extractor.mock_extractor import extract_objects


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


def draw_boxes(frame, boxes_xyxy, phrases, scores=None, source=None, frame_idx=None, backend_label=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    status_items = []
    if backend_label is not None:
        status_items.append(f"backend: {backend_label}")
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


def write_video_with_mediapy(video_out: Path, frames: list[np.ndarray], fps: float) -> None:
    try:
        import mediapy as media
    except ImportError as exc:
        raise RuntimeError("mediapy is required for video export. Install it with `pip install mediapy`.") from exc

    if not frames:
        raise RuntimeError(f"No frames collected for output video: {video_out}")

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    media.write_video(str(video_out), rgb_frames, fps=fps)


def _normalize_prompt_objects(instruction: str, explicit_objects: list[str] | None) -> list[str]:
    if explicit_objects:
        return explicit_objects
    return extract_objects(instruction)


def detect_initial_boxes(
    video_path: Path,
    instruction: str,
    explicit_objects: list[str] | None,
    dino_refresh_interval: int,
):
    chunker = TrackerChunker(dino_refresh_interval=dino_refresh_interval, tracker_backend="csrt")
    chunker.reset()

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    parsed_objects = _normalize_prompt_objects(instruction, explicit_objects)
    if not parsed_objects:
        raise RuntimeError(
            "Failed to resolve prompt objects. Pass --objects explicitly or use an instruction known to mock_extractor."
        )

    _, boxes_xyxy, logits, phrases, _ = chunker.get_dino_boxes(frame, parsed_objects)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        raise RuntimeError(f"No initial boxes detected for objects: {parsed_objects}")

    return frame, boxes_xyxy, logits, phrases, parsed_objects


def _build_trackers_for_backend(spec, boxes_xyxy, first_frame: np.ndarray, models_dir: Path):
    trackers: list[Any] = []
    for box_xyxy in boxes_xyxy:
        tracker = create_tracker_with_models(spec, models_dir)
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        height, width = first_frame.shape[:2]
        bbox = (
            int(round(np.clip(x1 * width, 0, max(width - 1, 0)))),
            int(round(np.clip(y1 * height, 0, max(height - 1, 0)))),
            int(round(max(1.0, np.clip(x2 * width, 0, width) - np.clip(x1 * width, 0, width)))),
            int(round(max(1.0, np.clip(y2 * height, 0, height) - np.clip(y1 * height, 0, height)))),
        )
        ok = tracker.init(first_frame, bbox)
        if ok is False:
            raise RuntimeError(f"tracker init returned False for bbox={bbox}")
        trackers.append(tracker)
    return trackers


def _tracker_bbox_to_norm_xyxy(bbox, frame_shape):
    height, width = frame_shape[:2]
    x, y, box_width, box_height = bbox
    x1 = float(np.clip(x / max(width, 1), 0.0, 1.0))
    y1 = float(np.clip(y / max(height, 1), 0.0, 1.0))
    x2 = float(np.clip((x + box_width) / max(width, 1), 0.0, 1.0))
    y2 = float(np.clip((y + box_height) / max(height, 1), 0.0, 1.0))
    return [x1, y1, x2, y2]


def export_backend_video(
    spec,
    video_path: Path,
    output_dir: Path,
    models_dir: Path,
    first_frame: np.ndarray,
    boxes_xyxy: torch.Tensor,
    logits,
    phrases: list[str],
    max_frames: int,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    backend_slug = spec.display_name.replace(":", "_").lower()
    output_path = output_dir / f"{video_path.stem}_{backend_slug}.mp4"
    trackers = _build_trackers_for_backend(spec, boxes_xyxy, first_frame, models_dir)

    frames_out: list[np.ndarray] = []
    update_latencies_ms: list[float] = []
    success_count = 0
    total_updates = 0
    frame_idx = 0
    current_boxes = boxes_xyxy.detach().cpu().tolist() if hasattr(boxes_xyxy, "detach") else list(boxes_xyxy)

    while True:
        if max_frames >= 0 and frame_idx >= max_frames:
            break

        if frame_idx == 0:
            source = "dino_init"
        else:
            next_boxes = []
            for tracker in trackers:
                start = time.perf_counter()
                ok, bbox = tracker.update(frame)
                update_latencies_ms.append((time.perf_counter() - start) * 1000.0)
                total_updates += 1
                if ok and bbox is not None:
                    success_count += 1
                    next_boxes.append(_tracker_bbox_to_norm_xyxy(bbox, frame.shape))
                else:
                    next_boxes.append(None)
            current_boxes = [box if box is not None else current_boxes[idx] for idx, box in enumerate(next_boxes)]
            source = "tracker"

        frames_out.append(
            draw_boxes(
                frame,
                current_boxes,
                phrases,
                scores=logits,
                source=source,
                frame_idx=frame_idx,
                backend_label=spec.display_name,
            )
        )
        frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            break

    cap.release()
    write_video_with_mediapy(output_path, frames_out, fps)
    avg_update_ms = float(np.mean(update_latencies_ms)) if update_latencies_ms else 0.0
    return {
        "backend": spec.display_name,
        "namespace": spec.namespace,
        "create_name": spec.create_name,
        "output_video": str(output_path),
        "frames_processed": frame_idx,
        "avg_update_ms": avg_update_ms,
        "success_rate": success_count / total_updates if total_updates else 0.0,
        "num_updates": total_updates,
    }


def main():
    parser = argparse.ArgumentParser(description="Export tracking videos for all available OpenCV backends on adaptive/testvideo.mp4.")
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parent / "testvideo.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "testvideo_tracking_exports")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument(
        "--instruction",
        default="pick up the cream cheese and place it on the plate",
        help="Language instruction used to extract target objects and initialize DINO boxes.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Optional explicit object phrases. Overrides object extraction from --instruction.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["all"],
        help="Trackers to export. Use names like CSRT or cv2.legacy:CSRT. Default: all discovered trackers.",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--dino-refresh-interval", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.trackers == ["all"]:
        tracker_specs = discover_tracker_specs()
    else:
        tracker_specs = []
        for tracker_name in args.trackers:
            spec = get_tracker_spec(tracker_name)
            if spec is None:
                raise RuntimeError(f"Tracker {tracker_name} is not available in this OpenCV build.")
            tracker_specs.append(spec)

    first_frame, boxes_xyxy, logits, phrases, parsed_objects = detect_initial_boxes(
        args.video,
        args.instruction,
        args.objects,
        args.dino_refresh_interval,
    )

    manifest: dict[str, object] = {
        "video": str(args.video),
        "instruction": args.instruction,
        "parsed_objects": parsed_objects,
        "models_dir": str(args.models_dir),
        "output_dir": str(args.output_dir),
        "items": [],
        "failures": [],
    }

    print(f"Initial phrases: {phrases}")
    print(f"Exporting trackers: {[spec.display_name for spec in tracker_specs]}")

    for spec in tracker_specs:
        print(f"[RUN ] {spec.display_name} ...")
        try:
            result = export_backend_video(
                spec,
                args.video,
                args.output_dir,
                args.models_dir,
                first_frame,
                boxes_xyxy,
                logits,
                phrases,
                args.max_frames,
            )
            manifest["items"].append(result)
            print(
                f"[OK  ] {spec.display_name}: avg_update={result['avg_update_ms']:.3f} ms, "
                f"success_rate={result['success_rate']:.3f}"
            )
        except Exception as exc:
            failure = {"backend": spec.display_name, "error": str(exc)}
            manifest["failures"].append(failure)
            print(f"[FAIL] {spec.display_name}: {exc}")

    manifest_path = args.output_dir / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Exported videos: {len(manifest['items'])}")
    print(f"Failures: {len(manifest['failures'])}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
