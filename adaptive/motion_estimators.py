from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = __import__("logging").getLogger("adaptive.motion_estimators")


@dataclass(frozen=True)
class MotionObservation:
    magnitude: float
    vector: np.ndarray
    source: str
    units: str
    depth_valid_ratio: float | None = None
    depth_median: float | None = None
    fallback_used: bool = False
    point_world: np.ndarray | None = None
    projected_pixel: np.ndarray | None = None


def _canonical_object_key(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("-", "_").replace(" ", "_")
    parts = [part for part in normalized.split("_") if part]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def linearize_depth_map(depth_map: np.ndarray, robot_state) -> np.ndarray | None:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim > 2:
        depth = depth[..., 0]
    if depth.ndim != 2:
        return None

    is_normalized = bool(robot_state.get("camera_depth_normalized", False))
    near = robot_state.get("camera_depth_near")
    far = robot_state.get("camera_depth_far")
    if not is_normalized:
        return depth
    if near is None or far is None:
        return None

    near = float(near)
    far = float(far)
    if not np.isfinite(near) or not np.isfinite(far) or near <= 0.0 or far <= near:
        return None

    depth = np.clip(depth, 0.0, 1.0)
    denom = 1.0 - depth * (1.0 - near / far)
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    linearized = np.full_like(depth, np.nan, dtype=np.float32)
    linearized[valid] = near / denom[valid]
    return linearized


def median_depth_in_box(
    depth_map: np.ndarray,
    box_xyxy,
    *,
    depth_inner_ratio: float = 0.6,
    depth_min_valid_ratio: float = 0.2,
) -> tuple[float | None, float]:
    if depth_map.ndim > 2:
        depth_map = np.asarray(depth_map)[..., 0]
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2:
        return None, 0.0

    depth_inner_ratio = float(np.clip(depth_inner_ratio, 0.1, 1.0))
    depth_min_valid_ratio = float(np.clip(depth_min_valid_ratio, 0.0, 1.0))
    h, w = depth_map.shape
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(x2 - x1, 1e-4) * depth_inner_ratio
    bh = max(y2 - y1, 1e-4) * depth_inner_ratio
    ix1 = int(np.clip((cx - bw / 2.0) * w, 0, max(w - 1, 0)))
    iy1 = int(np.clip((cy - bh / 2.0) * h, 0, max(h - 1, 0)))
    ix2 = int(np.clip((cx + bw / 2.0) * w, ix1 + 1, w))
    iy2 = int(np.clip((cy + bh / 2.0) * h, iy1 + 1, h))
    crop = depth_map[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None, 0.0

    valid = crop[np.isfinite(crop) & (crop > 1e-6)]
    valid_ratio = float(valid.size / crop.size)
    if valid.size == 0 or valid_ratio < depth_min_valid_ratio:
        return None, valid_ratio
    return float(np.median(valid)), valid_ratio


def project_world_point_to_observed_pixel(point_world, robot_state) -> np.ndarray | None:
    transform = robot_state.get("camera_world_to_image")
    resolution = robot_state.get("camera_resolution")
    if transform is None or resolution is None:
        return None

    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        return None
    point_world = np.asarray(point_world, dtype=np.float32).reshape(-1)
    if point_world.shape[0] != 3:
        return None

    point_h = np.concatenate([point_world, np.array([1.0], dtype=np.float32)], axis=0)
    pixel_h = transform @ point_h
    if not np.isfinite(pixel_h).all() or abs(float(pixel_h[2])) < 1e-8:
        return None
    pixel_xy = pixel_h[:2] / pixel_h[2]
    width = int(resolution[1])
    height = int(resolution[0])
    col = int(np.clip(np.round(pixel_xy[0]), 0, max(width - 1, 0)))
    row = int(np.clip(np.round(pixel_xy[1]), 0, max(height - 1, 0)))
    if robot_state.get("camera_observation_mirror_x", False):
        col = (width - 1) - col
    if robot_state.get("camera_observation_mirror_y", False):
        row = (height - 1) - row
    return np.array([col, row], dtype=np.int32)


def reconstruct_box_center_world_point(
    box_xyxy,
    depth_map: np.ndarray,
    robot_state,
    *,
    depth_inner_ratio: float = 0.6,
    depth_min_valid_ratio: float = 0.2,
) -> tuple[np.ndarray | None, dict[str, float | str | list[float]]]:
    camera_intrinsics = robot_state.get("camera_intrinsics")
    camera_to_world = robot_state.get("camera_to_world")
    if camera_intrinsics is None or camera_to_world is None:
        return None, {"missing_camera_metadata": 1.0}

    linearized_depth = linearize_depth_map(depth_map, robot_state)
    if linearized_depth is None:
        return None, {"missing_camera_metadata": 1.0}

    median_depth, valid_ratio = median_depth_in_box(
        linearized_depth,
        box_xyxy,
        depth_inner_ratio=depth_inner_ratio,
        depth_min_valid_ratio=depth_min_valid_ratio,
    )
    if median_depth is None:
        return None, {"depth_valid_ratio": valid_ratio}

    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    if camera_intrinsics.shape != (3, 3) or camera_to_world.shape != (4, 4):
        return None, {"missing_camera_metadata": 1.0}

    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None, {"missing_camera_metadata": 1.0}

    h, w = linearized_depth.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx_px = ((x1 + x2) / 2.0) * max(w - 1, 1)
    cy_px = ((y1 + y2) / 2.0) * max(h - 1, 1)
    if robot_state.get("camera_observation_mirror_x", False):
        cx_px = (w - 1) - cx_px
    if robot_state.get("camera_observation_mirror_y", False):
        cy_px = (h - 1) - cy_px

    x_cam = (cx_px - cx) / fx * median_depth
    y_cam = (cy_px - cy) / fy * median_depth
    point_cam = np.array([x_cam, y_cam, median_depth, 1.0], dtype=np.float32)
    point_world = camera_to_world @ point_cam
    point_world = point_world[:3]
    projected_pixel = project_world_point_to_observed_pixel(point_world, robot_state)
    meta: dict[str, float | str | list[float]] = {
        "depth_valid_ratio": valid_ratio,
        "depth_median": median_depth,
        "reconstruction_frame": "world",
    }
    if projected_pixel is not None:
        meta["projected_pixel"] = projected_pixel.astype(float).tolist()
    return point_world, meta


class BaseMotionEstimator:
    def __init__(self, *, lookback: int):
        self.lookback = max(1, int(lookback))
        self.last_debug_info: dict[str, object] = {}

    def reset(self) -> None:
        raise NotImplementedError

    def compute(self, *, phrases, boxes_xyxy, robot_state=None):
        raise NotImplementedError

    def _resolve_motion_timing(self, robot_state) -> tuple[int | None, float]:
        step_index = None if robot_state is None else robot_state.get("step_index")
        try:
            if step_index is not None:
                step_index = int(step_index)
        except Exception:
            step_index = None

        step_dt = None if robot_state is None else robot_state.get("env_control_dt")
        try:
            if step_dt is not None:
                step_dt = float(step_dt)
        except Exception:
            step_dt = None
        if step_dt is None or not np.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = None if robot_state is None else robot_state.get("control_dt")
            try:
                if step_dt is not None:
                    step_dt = float(step_dt)
            except Exception:
                step_dt = None
        if step_dt is None or not np.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = None
        return step_index, step_dt

    def _compute_elapsed_seconds(
        self,
        *,
        current_step_index: int | None,
        previous_step_index: int | None,
        step_dt: float | None,
        default_dt: float,
    ) -> float:
        if (
            current_step_index is not None
            and previous_step_index is not None
            and step_dt is not None
            and np.isfinite(step_dt)
            and step_dt > 0.0
        ):
            step_delta = current_step_index - previous_step_index
            if step_delta > 0:
                return float(step_delta) * float(step_dt)
        return self.lookback * max(float(default_dt), 1e-6)


class BBoxMotionEstimator(BaseMotionEstimator):
    def __init__(self, *, lookback: int):
        super().__init__(lookback=lookback)
        self.center_history: dict[str, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self.center_history = {}
        self.last_debug_info = {}

    def compute(self, *, phrases, boxes_xyxy, robot_state=None):
        observations: dict[str, MotionObservation] = {}
        active_phrases = set()
        for idx, phrase in enumerate(phrases):
            if idx >= len(boxes_xyxy):
                continue
            box = boxes_xyxy[idx]
            if len(box) != 4:
                continue
            active_phrases.add(phrase)
            x1, y1, x2, y2 = [float(v) for v in box]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            history = self.center_history.setdefault(phrase, deque(maxlen=self.lookback + 1))
            prev_center = history[0] if len(history) >= self.lookback else None
            if prev_center is not None:
                motion_vec = center - prev_center
                observations[phrase] = MotionObservation(
                    magnitude=float(np.linalg.norm(motion_vec)),
                    vector=motion_vec,
                    source="bbox2d",
                    units="norm_delta",
                )
            history.append(center)

        self.center_history = {
            phrase: history for phrase, history in self.center_history.items() if phrase in active_phrases
        }
        return observations


class DepthMotionEstimator(BaseMotionEstimator):
    def __init__(
        self,
        *,
        lookback: int,
        control_dt: float,
        depth_inner_ratio: float = 0.6,
        depth_min_valid_ratio: float = 0.2,
        fallback_estimator: BaseMotionEstimator | None = None,
    ):
        super().__init__(lookback=lookback)
        self.control_dt = max(float(control_dt), 1e-6)
        self.depth_inner_ratio = float(np.clip(depth_inner_ratio, 0.1, 1.0))
        self.depth_min_valid_ratio = float(np.clip(depth_min_valid_ratio, 0.0, 1.0))
        self.fallback_estimator = fallback_estimator or BBoxMotionEstimator(lookback=lookback)
        self.point_history: dict[str, deque[tuple[int | None, np.ndarray]]] = {}

    def reset(self) -> None:
        self.point_history = {}
        self.fallback_estimator.reset()
        self.last_debug_info = {}

    def _linearize_depth_map(self, depth_map: np.ndarray, robot_state) -> np.ndarray | None:
        return linearize_depth_map(depth_map, robot_state)

    def _median_depth_in_box(self, depth_map: np.ndarray, box_xyxy) -> tuple[float | None, float]:
        return median_depth_in_box(
            depth_map,
            box_xyxy,
            depth_inner_ratio=self.depth_inner_ratio,
            depth_min_valid_ratio=self.depth_min_valid_ratio,
        )

    def _box_center_to_world_point(
        self,
        box_xyxy,
        depth_map: np.ndarray,
        robot_state,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        point_world, meta = reconstruct_box_center_world_point(
            box_xyxy,
            depth_map,
            robot_state,
            depth_inner_ratio=self.depth_inner_ratio,
            depth_min_valid_ratio=self.depth_min_valid_ratio,
        )
        return point_world, meta

    def compute(self, *, phrases, boxes_xyxy, robot_state=None):
        fallback_observations = self.fallback_estimator.compute(
            phrases=phrases,
            boxes_xyxy=boxes_xyxy,
            robot_state=robot_state,
        )
        depth_map = None if robot_state is None else robot_state.get("image_depth")
        if depth_map is None:
            return {
                phrase: MotionObservation(
                    magnitude=obs.magnitude,
                    vector=obs.vector,
                    source="bbox2d_fallback",
                    units=obs.units,
                    fallback_used=True,
                )
                for phrase, obs in fallback_observations.items()
            }

        observations: dict[str, MotionObservation] = {}
        active_phrases = set()
        unmatched_phrases = []
        current_step_index, step_dt = self._resolve_motion_timing(robot_state)
        for idx, phrase in enumerate(phrases):
            if idx >= len(boxes_xyxy):
                continue
            box = boxes_xyxy[idx]
            if len(box) != 4:
                continue
            active_phrases.add(phrase)
            point, depth_meta = self._box_center_to_world_point(box, depth_map, robot_state)
            if point is None:
                unmatched_phrases.append(
                    {
                        "phrase": phrase,
                        "reason": "depth_reconstruction_failed",
                        "meta": dict(depth_meta or {}),
                    }
                )
                fallback = fallback_observations.get(phrase)
                if fallback is not None:
                    observations[phrase] = MotionObservation(
                        magnitude=fallback.magnitude,
                        vector=fallback.vector,
                        source="bbox2d_fallback",
                        units=fallback.units,
                        depth_valid_ratio=depth_meta.get("depth_valid_ratio"),
                        fallback_used=True,
                    )
                continue

            history = self.point_history.setdefault(phrase, deque(maxlen=self.lookback + 1))
            prev_entry = history[0] if len(history) >= self.lookback else None
            if prev_entry is not None:
                prev_step_index, prev_point = prev_entry
                elapsed = self._compute_elapsed_seconds(
                    current_step_index=current_step_index,
                    previous_step_index=prev_step_index,
                    step_dt=step_dt,
                    default_dt=self.control_dt,
                )
                motion_vec = (point - prev_point) / max(elapsed, 1e-6)
                projected_pixel = project_world_point_to_observed_pixel(point, robot_state)
                observations[phrase] = MotionObservation(
                    magnitude=float(np.linalg.norm(motion_vec)),
                    vector=motion_vec,
                    source="depth3d",
                    units="meters_per_second",
                    depth_valid_ratio=depth_meta.get("depth_valid_ratio"),
                    depth_median=depth_meta.get("depth_median"),
                    point_world=point,
                    projected_pixel=projected_pixel,
                )
            history.append((current_step_index, point))

        self.point_history = {
            phrase: history for phrase, history in self.point_history.items() if phrase in active_phrases
        }
        self.last_debug_info = {
            "metric": "depth3d",
            "num_phrases": len(phrases),
            "available_depth": depth_map is not None,
            "step_index": current_step_index,
            "step_dt": step_dt,
            "observed_phrases": sorted(observations.keys()),
            "unmatched_phrases": unmatched_phrases,
        }
        return observations


class SimGroundTruthMotionEstimator(BaseMotionEstimator):
    def __init__(
        self,
        *,
        lookback: int,
        control_dt: float,
        dynamic_object_name: str | None = None,
        fallback_estimator: BaseMotionEstimator | None = None,
    ):
        super().__init__(lookback=lookback)
        self.control_dt = max(float(control_dt), 1e-6)
        self.dynamic_object_name = dynamic_object_name
        self.dynamic_object_key = _canonical_object_key(dynamic_object_name)
        self.point_history: dict[str, deque[tuple[int | None, np.ndarray]]] = {}

    def reset(self) -> None:
        self.point_history = {}
        self.last_debug_info = {}

    def _resolve_phrase_point(self, phrase: str, robot_state):
        sim_gt_positions = None if robot_state is None else robot_state.get("sim_gt_positions")
        if not isinstance(sim_gt_positions, dict) or not sim_gt_positions:
            return None, None

        candidates = []
        phrase_key = _canonical_object_key(phrase)
        if phrase:
            candidates.append(str(phrase))
        if phrase_key:
            candidates.append(phrase_key)
        if self.dynamic_object_name and str(phrase) == str(self.dynamic_object_name):
            candidates.append(str(self.dynamic_object_name))
        if self.dynamic_object_key and phrase_key == self.dynamic_object_key:
            candidates.append(self.dynamic_object_key)

        for key in candidates:
            if key in sim_gt_positions:
                point = np.asarray(sim_gt_positions[key], dtype=np.float32).reshape(-1)
                if point.shape[0] >= 3 and np.isfinite(point[:3]).all():
                    return point[:3], key
        return None, None

    def compute(self, *, phrases, boxes_xyxy, robot_state=None):
        observations: dict[str, MotionObservation] = {}
        active_phrases = set()
        sim_gt_positions = None if robot_state is None else robot_state.get("sim_gt_positions")
        sim_gt_keys = sorted(sim_gt_positions.keys()) if isinstance(sim_gt_positions, dict) else []
        matched_phrases = []
        unmatched_phrases = []
        current_step_index, step_dt = self._resolve_motion_timing(robot_state)

        for idx, phrase in enumerate(phrases):
            if idx >= len(boxes_xyxy):
                continue
            box = boxes_xyxy[idx]
            if len(box) != 4:
                continue
            active_phrases.add(phrase)

            point, matched_key = self._resolve_phrase_point(phrase, robot_state)
            if point is None:
                unmatched_phrases.append(
                    {
                        "phrase": phrase,
                        "phrase_key": _canonical_object_key(phrase),
                        "reason": "no_matching_sim_gt_key",
                    }
                )
                continue

            history = self.point_history.setdefault(phrase, deque(maxlen=self.lookback + 1))
            matched_phrases.append(
                {
                    "phrase": phrase,
                    "phrase_key": _canonical_object_key(phrase),
                    "matched_key": matched_key,
                    "point_world": point.astype(float).tolist(),
                    "history_len_before_append": len(history),
                }
            )
            prev_entry = history[0] if len(history) >= self.lookback else None
            if prev_entry is not None:
                prev_step_index, prev_point = prev_entry
                elapsed = self._compute_elapsed_seconds(
                    current_step_index=current_step_index,
                    previous_step_index=prev_step_index,
                    step_dt=step_dt,
                    default_dt=self.control_dt,
                )
                motion_vec = (point - prev_point) / max(elapsed, 1e-6)
                observations[phrase] = MotionObservation(
                    magnitude=float(np.linalg.norm(motion_vec)),
                    vector=motion_vec,
                    source="sim_gt",
                    units="meters_per_second",
                    fallback_used=False,
                    point_world=point,
                )
            history.append((current_step_index, point))

        self.point_history = {
            phrase: history for phrase, history in self.point_history.items() if phrase in active_phrases
        }
        self.last_debug_info = {
            "metric": "sim_gt",
            "dynamic_object_name": self.dynamic_object_name,
            "dynamic_object_key": self.dynamic_object_key,
            "step_index": current_step_index,
            "step_dt": step_dt,
            "sim_gt_keys": sim_gt_keys,
            "phrases": list(phrases),
            "matched_phrases": matched_phrases,
            "unmatched_phrases": unmatched_phrases,
            "observed_phrases": sorted(observations.keys()),
        }
        return observations


def create_motion_estimator(
    *,
    metric: str,
    lookback: int,
    control_dt: float,
    depth_inner_ratio: float = 0.6,
    depth_min_valid_ratio: float = 0.2,
    dynamic_object_name: str | None = None,
):
    metric_key = str(metric or "bbox2d").lower()
    if metric_key == "bbox2d":
        return BBoxMotionEstimator(lookback=lookback)
    if metric_key == "depth3d":
        return DepthMotionEstimator(
            lookback=lookback,
            control_dt=control_dt,
            depth_inner_ratio=depth_inner_ratio,
            depth_min_valid_ratio=depth_min_valid_ratio,
        )
    if metric_key == "sim_gt":
        return SimGroundTruthMotionEstimator(
            lookback=lookback,
            control_dt=control_dt,
            dynamic_object_name=dynamic_object_name,
        )
    raise ValueError(f"Unsupported object motion metric: {metric}")
