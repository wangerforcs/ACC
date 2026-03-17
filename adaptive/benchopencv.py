import argparse
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def percentile(values: List[float], q: float) -> float:
    """Return percentile in milliseconds."""
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


def get_tracker_factory(name: str) -> Optional[Callable[[], object]]:
    """
    Return a callable that creates the requested tracker.
    Supports both cv2 and cv2.legacy namespaces.
    """
    name = name.upper()

    candidates = {
        "BOOSTING": ["TrackerBoosting_create"],
        "MIL": ["TrackerMIL_create"],
        "KCF": ["TrackerKCF_create"],
        "TLD": ["TrackerTLD_create"],
        "MEDIANFLOW": ["TrackerMedianFlow_create"],
        "GOTURN": ["TrackerGOTURN_create"],
        "MOSSE": ["TrackerMOSSE_create"],
        "CSRT": ["TrackerCSRT_create"],
    }

    if name not in candidates:
        return None

    namespaces = [cv2]
    if hasattr(cv2, "legacy"):
        namespaces.append(cv2.legacy)

    for ns in namespaces:
        for attr in candidates[name]:
            if hasattr(ns, attr):
                fn = getattr(ns, attr)
                return fn

    return None


def list_available_trackers() -> List[str]:
    tracker_names = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    ]
    available = []
    for name in tracker_names:
        if get_tracker_factory(name) is not None:
            available.append(name)
    return available


def generate_random_frame(
    height: int,
    width: int,
    channels: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a random uint8 image."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


def move_bbox_randomly(
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    max_shift: int,
    rng: np.random.Generator,
) -> BBox:
    """Move bbox a little while keeping it inside the frame."""
    x, y, w, h = bbox
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))

    x = max(0, min(frame_w - w, x + dx))
    y = max(0, min(frame_h - h, y + dy))
    return x, y, w, h


def create_initial_bbox(
    frame_w: int,
    frame_h: int,
    box_w: int,
    box_h: int,
) -> BBox:
    x = max(0, (frame_w - box_w) // 2)
    y = max(0, (frame_h - box_h) // 2)
    return x, y, box_w, box_h


def benchmark_tracker(
    tracker_name: str,
    num_trials: int,
    frames_per_trial: int,
    width: int,
    height: int,
    box_w: int,
    box_h: int,
    max_shift: int,
    seed: int,
    warmup_trials: int = 2,
) -> Dict[str, float]:
    """
    Benchmark one tracker with synthetic random frames.

    For each trial:
      1) create tracker
      2) init on first random frame + bbox
      3) update across N-1 random frames
    """
    factory = get_tracker_factory(tracker_name)
    if factory is None:
        raise RuntimeError(f"Tracker {tracker_name} is not available in this OpenCV build.")

    rng = np.random.default_rng(seed)

    init_latencies_ms: List[float] = []
    update_latencies_ms: List[float] = []
    success_count = 0
    total_updates = 0

    total_trials = warmup_trials + num_trials

    for trial_idx in range(total_trials):
        frame0 = generate_random_frame(height, width, rng=rng)
        bbox = create_initial_bbox(width, height, box_w, box_h)

        tracker = factory()

        t0 = now_ms()
        ok = tracker.init(frame0, bbox)
        t1 = now_ms()

        # Some tracker implementations return None instead of bool.
        init_ok = True if ok is None else bool(ok)

        # warmup阶段不统计
        if trial_idx >= warmup_trials:
            init_latencies_ms.append(t1 - t0)

        if not init_ok:
            continue

        cur_bbox = bbox
        for _ in range(frames_per_trial - 1):
            frame = generate_random_frame(height, width, rng=rng)
            cur_bbox = move_bbox_randomly(cur_bbox, width, height, max_shift, rng)

            t2 = now_ms()
            try:
                ok, pred_bbox = tracker.update(frame)
            except Exception:
                ok, pred_bbox = False, None
            t3 = now_ms()

            if trial_idx >= warmup_trials:
                update_latencies_ms.append(t3 - t2)
                total_updates += 1
                if ok:
                    success_count += 1

    if not init_latencies_ms or not update_latencies_ms:
        raise RuntimeError(
            f"Tracker {tracker_name} did not produce usable benchmark results. "
            f"(Possible unsupported model/dependency issue, e.g. GOTURN.)"
        )

    mean_update = statistics.mean(update_latencies_ms)
    fps = 1000.0 / mean_update if mean_update > 0 else float("inf")

    return {
        "tracker": tracker_name,
        "init_mean_ms": statistics.mean(init_latencies_ms),
        "init_p50_ms": percentile(init_latencies_ms, 0.50),
        "init_p95_ms": percentile(init_latencies_ms, 0.95),
        "update_mean_ms": mean_update,
        "update_p50_ms": percentile(update_latencies_ms, 0.50),
        "update_p95_ms": percentile(update_latencies_ms, 0.95),
        "update_p99_ms": percentile(update_latencies_ms, 0.99),
        "fps_equiv": fps,
        "success_rate": success_count / total_updates if total_updates else 0.0,
        "num_inits": len(init_latencies_ms),
        "num_updates": len(update_latencies_ms),
    }


def print_results(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results.")
        return

    headers = [
        "tracker",
        "init_mean_ms",
        "init_p95_ms",
        "update_mean_ms",
        "update_p50_ms",
        "update_p95_ms",
        "update_p99_ms",
        "fps_equiv",
        "success_rate",
        "num_updates",
    ]

    rows = []
    for r in results:
        rows.append([
            str(r["tracker"]),
            f"{r['init_mean_ms']:.3f}",
            f"{r['init_p95_ms']:.3f}",
            f"{r['update_mean_ms']:.3f}",
            f"{r['update_p50_ms']:.3f}",
            f"{r['update_p95_ms']:.3f}",
            f"{r['update_p99_ms']:.3f}",
            f"{r['fps_equiv']:.1f}",
            f"{r['success_rate']:.3f}",
            str(int(r["num_updates"])),
        ])

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
    parser = argparse.ArgumentParser(description="Benchmark OpenCV trackers with random synthetic frames.")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF", "MIL", "MEDIANFLOW", "CSRT"],
        help="Trackers to test, e.g. MOSSE KCF MIL CSRT",
    )
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--box-w", type=int, default=120, help="Initial bounding box width")
    parser.add_argument("--box-h", type=int, default=120, help="Initial bounding box height")
    parser.add_argument("--trials", type=int, default=20, help="Measured trials per tracker")
    parser.add_argument("--frames-per-trial", type=int, default=100, help="Frames per trial including init frame")
    parser.add_argument("--warmup-trials", type=int, default=3, help="Warmup trials per tracker")
    parser.add_argument("--max-shift", type=int, default=10, help="Random bbox shift per frame")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    available = list_available_trackers()
    print("Available trackers in this OpenCV build:", available)
    print()

    results = []
    for tracker_name in args.trackers:
        if tracker_name.upper() not in available:
            print(f"[SKIP] {tracker_name}: not available in this OpenCV build.")
            continue

        print(f"[RUN ] {tracker_name} ...")
        try:
            result = benchmark_tracker(
                tracker_name=tracker_name,
                num_trials=args.trials,
                frames_per_trial=args.frames_per_trial,
                width=args.width,
                height=args.height,
                box_w=args.box_w,
                box_h=args.box_h,
                max_shift=args.max_shift,
                seed=args.seed,
                warmup_trials=args.warmup_trials,
            )
            results.append(result)
            print(
                f"       update_mean={result['update_mean_ms']:.3f} ms, "
                f"p95={result['update_p95_ms']:.3f} ms, "
                f"fps≈{result['fps_equiv']:.1f}"
            )
        except Exception as e:
            print(f"[FAIL] {tracker_name}: {e}")

    print("\n=== Summary ===")
    results.sort(key=lambda x: x["update_mean_ms"])
    print_results(results)


if __name__ == "__main__":
    main()