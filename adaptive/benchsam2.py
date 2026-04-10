import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


BBox = Tuple[int, int, int, int]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * q
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return values_sorted[lo]
    frac = idx - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


def now_ms() -> float:
    return time.perf_counter_ns() / 1e6


def generate_random_frame(
    height: int,
    width: int,
    channels: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


def create_initial_bbox(frame_w: int, frame_h: int, box_w: int, box_h: int) -> BBox:
    x = max(0, (frame_w - box_w) // 2)
    y = max(0, (frame_h - box_h) // 2)
    return x, y, box_w, box_h


def move_bbox_randomly(
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    max_shift: int,
    rng: np.random.Generator,
) -> BBox:
    x, y, w, h = bbox
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    x = max(0, min(frame_w - w, x + dx))
    y = max(0, min(frame_h - h, y + dy))
    return x, y, w, h


def bbox_to_xyxy_abs(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def make_trial_frames(
    width: int,
    height: int,
    box_w: int,
    box_h: int,
    frames_per_trial: int,
    max_shift: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[BBox]]:
    rng = np.random.default_rng(seed)
    frames: List[np.ndarray] = []
    boxes: List[BBox] = []
    bbox = create_initial_bbox(width, height, box_w, box_h)
    for _ in range(frames_per_trial):
        frames.append(generate_random_frame(height, width, rng=rng))
        boxes.append(bbox)
        bbox = move_bbox_randomly(bbox, width, height, max_shift, rng)
    return frames, boxes


def prepare_video_dir(frames: Sequence[np.ndarray], suffix: str = ".jpg") -> str:
    temp_dir = tempfile.mkdtemp(prefix="sam2_bench_")
    temp_path = Path(temp_dir)
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(temp_path / f"{idx:05d}{ext}"), frame)
    return temp_dir


def build_image_predictor(
    model_id: str,
    config: Optional[str],
    ckpt: Optional[str],
    device: str,
) -> Tuple[SAM2ImagePredictor, float]:
    t0 = now_ms()
    if config:
        model = build_sam2(config, ckpt_path=ckpt, device=device)
        predictor = SAM2ImagePredictor(model)
    else:
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    t1 = now_ms()
    return predictor, t1 - t0


def build_video_predictor(
    model_id: str,
    config: Optional[str],
    ckpt: Optional[str],
    device: str,
) -> Tuple[SAM2VideoPredictor, float]:
    t0 = now_ms()
    if config:
        predictor = build_sam2_video_predictor(config, ckpt_path=ckpt, device=device)
    else:
        predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    predictor.eval()
    t1 = now_ms()
    return predictor, t1 - t0


def benchmark_sam2_image(
    predictor: SAM2ImagePredictor,
    num_trials: int,
    warmup_trials: int,
    frames_per_trial: int,
    width: int,
    height: int,
    box_w: int,
    box_h: int,
    max_shift: int,
    seed: int,
) -> Dict[str, float]:
    set_image_latencies_ms: List[float] = []
    predict_latencies_ms: List[float] = []
    total_frames = 0

    total_trials = warmup_trials + num_trials
    for trial_idx in range(total_trials):
        frames, boxes = make_trial_frames(
            width, height, box_w, box_h, frames_per_trial, max_shift, seed + trial_idx
        )
        prev_box = bbox_to_xyxy_abs(boxes[0])
        for frame_idx, frame in enumerate(frames):
            t0 = now_ms()
            predictor.set_image(frame)
            t1 = now_ms()

            masks, scores, logits = predictor.predict(box=prev_box[None, :], multimask_output=False)
            t2 = now_ms()

            if trial_idx >= warmup_trials:
                set_image_latencies_ms.append(t1 - t0)
                predict_latencies_ms.append(t2 - t1)
                total_frames += 1

            if masks is not None and len(masks) > 0:
                mask = np.squeeze(masks[0])
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    prev_box = np.array(
                        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                        dtype=np.float32,
                    )
                else:
                    prev_box = bbox_to_xyxy_abs(boxes[min(frame_idx + 1, len(boxes) - 1)])

    if not predict_latencies_ms:
        raise RuntimeError("SAM2 image benchmark did not produce usable measurements.")

    total_mean = statistics.mean([a + b for a, b in zip(set_image_latencies_ms, predict_latencies_ms)])
    return {
        "mode": "image",
        "set_image_mean_ms": statistics.mean(set_image_latencies_ms),
        "set_image_p95_ms": percentile(set_image_latencies_ms, 0.95),
        "predict_mean_ms": statistics.mean(predict_latencies_ms),
        "predict_p50_ms": percentile(predict_latencies_ms, 0.50),
        "predict_p95_ms": percentile(predict_latencies_ms, 0.95),
        "predict_p99_ms": percentile(predict_latencies_ms, 0.99),
        "total_mean_ms": total_mean,
        "fps_equiv": 1000.0 / total_mean if total_mean > 0 else float("inf"),
        "num_updates": total_frames,
    }


def benchmark_sam2_video(
    predictor: SAM2VideoPredictor,
    num_trials: int,
    warmup_trials: int,
    frames_per_trial: int,
    width: int,
    height: int,
    box_w: int,
    box_h: int,
    max_shift: int,
    seed: int,
    frame_ext: str,
) -> Dict[str, float]:
    init_latencies_ms: List[float] = []
    prompt_latencies_ms: List[float] = []
    propagate_latencies_ms: List[float] = []
    per_frame_latencies_ms: List[float] = []

    total_trials = warmup_trials + num_trials
    for trial_idx in range(total_trials):
        frames, boxes = make_trial_frames(
            width, height, box_w, box_h, frames_per_trial, max_shift, seed + 10000 + trial_idx
        )
        video_dir = prepare_video_dir(frames, frame_ext)
        try:
            t0 = now_ms()
            state = predictor.init_state(video_path=video_dir)
            t1 = now_ms()

            prompt_box = bbox_to_xyxy_abs(boxes[0])
            t2 = now_ms()
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=prompt_box,
            )
            t3 = now_ms()

            frame_count = 0
            t4 = now_ms()
            for _ in predictor.propagate_in_video(state):
                frame_count += 1
            t5 = now_ms()

            if hasattr(predictor, "reset_state"):
                predictor.reset_state(state)

            if trial_idx >= warmup_trials:
                init_latencies_ms.append(t1 - t0)
                prompt_latencies_ms.append(t3 - t2)
                propagate_latencies_ms.append(t5 - t4)
                if frame_count > 0:
                    per_frame_latencies_ms.append((t5 - t4) / frame_count)
        finally:
            shutil.rmtree(video_dir, ignore_errors=True)

    if not propagate_latencies_ms:
        raise RuntimeError("SAM2 video benchmark did not produce usable measurements.")

    mean_per_frame = statistics.mean(per_frame_latencies_ms)
    return {
        "mode": "video",
        "init_mean_ms": statistics.mean(init_latencies_ms),
        "init_p95_ms": percentile(init_latencies_ms, 0.95),
        "prompt_mean_ms": statistics.mean(prompt_latencies_ms),
        "propagate_mean_ms": statistics.mean(propagate_latencies_ms),
        "propagate_p95_ms": percentile(propagate_latencies_ms, 0.95),
        "per_frame_mean_ms": mean_per_frame,
        "per_frame_p95_ms": percentile(per_frame_latencies_ms, 0.95),
        "fps_equiv": 1000.0 / mean_per_frame if mean_per_frame > 0 else float("inf"),
        "num_updates": len(per_frame_latencies_ms),
    }


def print_results(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results.")
        return

    headers = [
        "mode",
        "load_ms",
        "init_mean_ms",
        "predict_mean_ms",
        "per_frame_mean_ms",
        "p95_ms",
        "fps_equiv",
        "num_updates",
    ]

    rows = []
    for r in results:
        if r["mode"] == "image":
            rows.append(
                [
                    "image",
                    f"{r['load_ms']:.3f}",
                    f"{r['set_image_mean_ms']:.3f}",
                    f"{r['predict_mean_ms']:.3f}",
                    f"{r['total_mean_ms']:.3f}",
                    f"{r['predict_p95_ms']:.3f}",
                    f"{r['fps_equiv']:.1f}",
                    str(int(r["num_updates"])),
                ]
            )
        else:
            rows.append(
                [
                    "video",
                    f"{r['load_ms']:.3f}",
                    f"{r['init_mean_ms']:.3f}",
                    f"{r['propagate_mean_ms']:.3f}",
                    f"{r['per_frame_mean_ms']:.3f}",
                    f"{r['per_frame_p95_ms']:.3f}",
                    f"{r['fps_equiv']:.1f}",
                    str(int(r["num_updates"])),
                ]
            )

    col_widths = []
    for i, h in enumerate(headers):
        width = max(len(h), max(len(row[i]) for row in rows))
        col_widths.append(width)

    def fmt_row(row: List[str]) -> str:
        return " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SAM2 image and video predictors with synthetic data.")
    parser.add_argument("--modes", nargs="+", default=["image", "video"], choices=["image", "video"])
    parser.add_argument("--width", type=int, default=224, help="Frame width")
    parser.add_argument("--height", type=int, default=224, help="Frame height")
    parser.add_argument("--box-w", type=int, default=40, help="Initial bounding box width")
    parser.add_argument("--box-h", type=int, default=40, help="Initial bounding box height")
    parser.add_argument("--trials", type=int, default=10, help="Measured trials per mode")
    parser.add_argument("--frames-per-trial", type=int, default=32, help="Frames per trial")
    parser.add_argument("--warmup-trials", type=int, default=2, help="Warmup trials per mode")
    parser.add_argument("--max-shift", type=int, default=8, help="Random bbox shift per frame")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--sam2-model-id", type=str, default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--sam2-config", type=str, default=None)
    parser.add_argument("--sam2-ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-frame-ext", type=str, default="jpg", help="Frame extension for video predictor")
    args = parser.parse_args()

    results: List[Dict[str, float]] = []

    if "image" in args.modes:
        print("[RUN ] SAM2 image ...")
        try:
            predictor, load_ms = build_image_predictor(
                args.sam2_model_id,
                args.sam2_config,
                args.sam2_ckpt,
                args.device,
            )
            image_result = benchmark_sam2_image(
                predictor=predictor,
                num_trials=args.trials,
                warmup_trials=args.warmup_trials,
                frames_per_trial=args.frames_per_trial,
                width=args.width,
                height=args.height,
                box_w=args.box_w,
                box_h=args.box_h,
                max_shift=args.max_shift,
                seed=args.seed,
            )
            image_result["load_ms"] = load_ms
            results.append(image_result)
            print(
                f"       total_mean={image_result['total_mean_ms']:.3f} ms, "
                f"predict_p95={image_result['predict_p95_ms']:.3f} ms, "
                f"fps≈{image_result['fps_equiv']:.1f}"
            )
        except Exception as exc:
            print(f"[FAIL] SAM2 image: {exc}")

    if "video" in args.modes:
        print("[RUN ] SAM2 video ...")
        try:
            predictor, load_ms = build_video_predictor(
                args.sam2_model_id,
                args.sam2_config,
                args.sam2_ckpt,
                args.device,
            )
            video_result = benchmark_sam2_video(
                predictor=predictor,
                num_trials=args.trials,
                warmup_trials=args.warmup_trials,
                frames_per_trial=args.frames_per_trial,
                width=args.width,
                height=args.height,
                box_w=args.box_w,
                box_h=args.box_h,
                max_shift=args.max_shift,
                seed=args.seed,
                frame_ext=args.video_frame_ext,
            )
            video_result["load_ms"] = load_ms
            results.append(video_result)
            print(
                f"       per_frame_mean={video_result['per_frame_mean_ms']:.3f} ms, "
                f"per_frame_p95={video_result['per_frame_p95_ms']:.3f} ms, "
                f"fps≈{video_result['fps_equiv']:.1f}"
            )
        except Exception as exc:
            print(f"[FAIL] SAM2 video: {exc}")

    print("\n=== Summary ===")
    print_results(results)


if __name__ == "__main__":
    main()
