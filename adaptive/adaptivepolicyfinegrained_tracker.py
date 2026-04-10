import logging
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .adaptivepolicyfinegrained import AdaptiveActionChunker as BaseAdaptiveActionChunker
from .motion_estimators import create_motion_estimator
from .tracker_utils import (
    bbox_to_norm_xyxy,
    box_center_inside,
    compute_box_edge_distance,
    ensure_frame,
    get_tracker_factory,
    intersection_over_box_area,
    norm_xyxy_to_bbox,
    normalize_phrase_key,
    normalize_prompt,
    single_box_center_distance,
    single_box_iou,
    xyxy_to_cxcywh,
)

logger = logging.getLogger("adaptive.tracker")


def _require_tracker_model_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing tracker model file: {path}")
    return str(path)


class AdaptiveActionChunker(BaseAdaptiveActionChunker):
    """Tracker-backed variant of the fine-grained adaptive chunker.

    DINO is still used when `ValueFunc` runs during VLA inference.
    Between VLA inferences, the heuristic path reuses tracker-updated boxes to
    avoid repeatedly querying DINO.
    """

    def __init__(
        self,
        *args,
        tracker_backend: str = "nano", #"dasiamrpn", #"csrt",
        tracker_models_dir: str = "/home/wangzb/vlash/models",
        dino_refresh_interval: int = 400, #先拉满不用这个了
        dino_accept_iou: float = 0.1,
        dino_accept_center_dist: float = 0.15,
        enable_object_motion_penalty: bool = False,
        dynamic_motion_object_name: Optional[str] = None,
        object_motion_metric: str = "bbox2d",
        
        
        object_motion_threshold: float = 0.01, #进入移动
        object_motion_exit_threshold: float = 0.006, #退出移动
        object_motion_fast_threshold: float = 0.025, #移动速度
        
        object_motion_speed_threshold: float = 0.03,
        object_motion_speed_exit_threshold: float = 0.01,
        object_motion_fast_speed_threshold: float = 0.1,
        
        object_motion_small_chunk_confidence: int = 0,
        object_motion_medium_chunk_confidence: int = 1,
        object_motion_lookback: int = 3, # motion计算的窗口 防止抖动
        object_motion_enter_persistence: int = 3, # 进入移动次数
        object_motion_exit_persistence: int = 3, # 退出移动次数
        object_motion_control_dt: float = 1.0 / 30.0,
        object_motion_depth_inner_ratio: float = 0.6,
        object_motion_depth_min_valid_ratio: float = 0.2,
        relative_progress_lookback: int = 10, # 进展窗口 加大了
        relative_progress_improve_threshold: float = 0.05, # improving这个感觉不是很会调
        motion_penalty_max_center_distance: float = 0.3,
        gripper_box_half_size_px: int = 18,
        gripper_contact_iou_threshold: float = 0.1,#这两个感觉不好设置
        gripper_contact_center_dist_threshold: float = 0.08,
        gripper_contact_persistence: int = 2, #判断接触要多少次
        gripper_closed_threshold: float = -0.039,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tracker_backend = tracker_backend.lower()
        if tracker_models_dir is None:
            tracker_models_dir = Path(__file__).resolve().parent.parent / "models"
        self.tracker_models_dir = Path(tracker_models_dir)
        self.dino_refresh_interval = max(1, int(dino_refresh_interval))
        self.dino_accept_iou = float(dino_accept_iou)
        self.dino_accept_center_dist = float(dino_accept_center_dist)
        self.enable_object_motion_penalty = bool(enable_object_motion_penalty)
        self.dynamic_motion_object_name = dynamic_motion_object_name
        self.object_motion_metric = str(object_motion_metric).lower()
        self.object_motion_threshold = float(object_motion_threshold)
        self.object_motion_fast_threshold = float(object_motion_fast_threshold)
        self.object_motion_speed_threshold = float(object_motion_speed_threshold)
        self.object_motion_speed_exit_threshold = float(object_motion_speed_exit_threshold)
        self.object_motion_fast_speed_threshold = float(object_motion_fast_speed_threshold)
        self.object_motion_small_chunk_confidence = int(object_motion_small_chunk_confidence)
        self.object_motion_medium_chunk_confidence = int(object_motion_medium_chunk_confidence)
        self.object_motion_lookback = max(1, int(object_motion_lookback))
        self.object_motion_exit_threshold = float(object_motion_exit_threshold)
        self.object_motion_enter_persistence = max(1, int(object_motion_enter_persistence))
        self.object_motion_exit_persistence = max(1, int(object_motion_exit_persistence))
        self.object_motion_control_dt = max(float(object_motion_control_dt), 1e-6)
        self.object_motion_depth_inner_ratio = float(object_motion_depth_inner_ratio)
        self.object_motion_depth_min_valid_ratio = float(object_motion_depth_min_valid_ratio)
        self.relative_progress_lookback = max(2, int(relative_progress_lookback))
        self.relative_progress_improve_threshold = float(relative_progress_improve_threshold)
        self.motion_penalty_max_center_distance = float(motion_penalty_max_center_distance)
        self.gripper_box_half_size_px = max(1, int(gripper_box_half_size_px))
        self.gripper_contact_iou_threshold = float(gripper_contact_iou_threshold)
        self.gripper_contact_center_dist_threshold = float(gripper_contact_center_dist_threshold)
        self.gripper_contact_persistence = max(1, int(gripper_contact_persistence))
        self.gripper_closed_threshold = float(gripper_closed_threshold)
        self.trackers: List[Any] = []
        self.tracker_phrases: List[str] = []
        self.tracker_prompt: Optional[Tuple[str, ...]] = None
        self.cached_boxes = None
        self.box_query_count = 0
        self.last_query_used_dino = False
        self.active_tracker_backend: Optional[str] = None
        self.object_motion_state: dict[str, bool] = {}
        self.object_motion_counts: dict[str, int] = {}
        self.motion_estimator = create_motion_estimator(
            metric=self.object_motion_metric,
            lookback=self.object_motion_lookback,
            control_dt=self.object_motion_control_dt,
            depth_inner_ratio=self.object_motion_depth_inner_ratio,
            depth_min_valid_ratio=self.object_motion_depth_min_valid_ratio,
            dynamic_object_name=self.dynamic_motion_object_name,
        )
        self.grasped_phrases: set[str] = set()
        self.gripper_contact_counts: dict[str, int] = {}
        # 抓住物体后去目标
        self.relative_progress_history: deque[float] = deque(maxlen=self.relative_progress_lookback)
        # 去抓物体
        self.pregrasp_relative_progress_history: deque[float] = deque(maxlen=self.relative_progress_lookback)

    def reset(self):
        super().reset()
        self.trackers = []
        self.tracker_phrases = []
        self.tracker_prompt = None
        self.cached_boxes = None
        self.box_query_count = 0
        self.last_query_used_dino = False
        self.active_tracker_backend = None
        self.motion_estimator.reset()
        self.object_motion_state = {}
        self.object_motion_counts = {}
        self.grasped_phrases = set()
        self.gripper_contact_counts = {}
        self.relative_progress_history = deque(maxlen=self.relative_progress_lookback)
        self.pregrasp_relative_progress_history = deque(maxlen=self.relative_progress_lookback)

    def get_runtime_config(self):
        config = super().get_runtime_config()
        config.update(
            {
                "policy_type": "finegrained_tracker",
                "tracker_backend": self.tracker_backend,
                "tracker_models_dir": str(self.tracker_models_dir),
                "dino_refresh_interval": self.dino_refresh_interval,
                "dino_accept_iou": self.dino_accept_iou,
                "dino_accept_center_dist": self.dino_accept_center_dist,
                "enable_object_motion_penalty": self.enable_object_motion_penalty,
                "dynamic_motion_object_name": self.dynamic_motion_object_name,
                "object_motion_metric": self.object_motion_metric,
                "object_motion_threshold": self.object_motion_threshold,
                "object_motion_fast_threshold": self.object_motion_fast_threshold,
                "object_motion_speed_threshold": self.object_motion_speed_threshold,
                "object_motion_speed_exit_threshold": self.object_motion_speed_exit_threshold,
                "object_motion_fast_speed_threshold": self.object_motion_fast_speed_threshold,
                "object_motion_small_chunk_confidence": self.object_motion_small_chunk_confidence,
                "object_motion_medium_chunk_confidence": self.object_motion_medium_chunk_confidence,
                "object_motion_lookback": self.object_motion_lookback,
                "object_motion_exit_threshold": self.object_motion_exit_threshold,
                "object_motion_enter_persistence": self.object_motion_enter_persistence,
                "object_motion_exit_persistence": self.object_motion_exit_persistence,
                "object_motion_control_dt": self.object_motion_control_dt,
                "object_motion_depth_inner_ratio": self.object_motion_depth_inner_ratio,
                "object_motion_depth_min_valid_ratio": self.object_motion_depth_min_valid_ratio,
                "relative_progress_lookback": self.relative_progress_lookback,
                "relative_progress_improve_threshold": self.relative_progress_improve_threshold,
                "motion_penalty_max_center_distance": self.motion_penalty_max_center_distance,
                "gripper_box_half_size_px": self.gripper_box_half_size_px,
                "gripper_contact_iou_threshold": self.gripper_contact_iou_threshold,
                "gripper_contact_center_dist_threshold": self.gripper_contact_center_dist_threshold,
                "gripper_contact_persistence": self.gripper_contact_persistence,
                "gripper_closed_threshold": self.gripper_closed_threshold,
            }
        )
        return config

    def _empty_box_result(self):
        empty = torch.tensor([])
        return empty, empty, torch.tensor([]), [], None

    def _tracker_factory_candidates(self):
        backend_classes = {
            "csrt": ["TrackerCSRT", "TrackerKCF", "TrackerMIL"],
            "kcf": ["TrackerKCF", "TrackerCSRT", "TrackerMIL"],
            "mil": ["TrackerMIL", "TrackerKCF", "TrackerCSRT"],
            "dasiamrpn": ["TrackerDaSiamRPN", "TrackerMIL"],
            "goturn": ["TrackerGOTURN", "TrackerMIL"],
            "nano": ["TrackerNano", "TrackerMIL"],
            "vit": ["TrackerVit", "TrackerMIL"],
        }
        for class_name in backend_classes.get(self.tracker_backend, backend_classes["mil"]):
            factory = get_tracker_factory(cv2, class_name)
            if factory is not None:
                yield class_name.replace("Tracker", "").lower(), factory
            legacy = getattr(cv2, "legacy", None)
            if legacy is not None:
                factory = get_tracker_factory(legacy, class_name)
                if factory is not None:
                    yield class_name.replace("Tracker", "").lower(), factory

    def _create_tracker_with_models(self, backend_name: str, factory):
        if backend_name == "dasiamrpn":
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = _require_tracker_model_file(self.tracker_models_dir / "dasiamrpn_model.onnx")
            params.kernel_cls1 = _require_tracker_model_file(self.tracker_models_dir / "dasiamrpn_kernel_cls1.onnx")
            params.kernel_r1 = _require_tracker_model_file(self.tracker_models_dir / "dasiamrpn_kernel_r1.onnx")
            return cv2.TrackerDaSiamRPN_create(params)

        if backend_name == "goturn":
            params = cv2.TrackerGOTURN_Params()
            params.modelTxt = _require_tracker_model_file(self.tracker_models_dir / "goturn.prototxt")
            params.modelBin = _require_tracker_model_file(self.tracker_models_dir / "goturn.caffemodel")
            return cv2.TrackerGOTURN_create(params)

        if backend_name == "nano":
            params = cv2.TrackerNano_Params()
            params.backbone = _require_tracker_model_file(self.tracker_models_dir / "nanotrack_backbone_sim.onnx")
            params.neckhead = _require_tracker_model_file(self.tracker_models_dir / "nanotrack_head_sim.onnx")
            return cv2.TrackerNano_create(params)

        if backend_name == "vit":
            params = cv2.TrackerVit_Params()
            params.net = _require_tracker_model_file(self.tracker_models_dir / "object_tracking_vittrack_2023sep.onnx")
            return cv2.TrackerVit_create(params)

        return factory()

    def _create_tracker(self):
        for backend_name, factory in self._tracker_factory_candidates():
            try:
                tracker = self._create_tracker_with_models(backend_name, factory)
                self.active_tracker_backend = backend_name
                return tracker
            except Exception as exc:
                logger.warning(
                    "Failed to create tracker backend '%s' using models dir '%s': %s",
                    backend_name,
                    self.tracker_models_dir,
                    exc,
                )
                continue
        return None

    def _recent_gripper_close(self) -> bool:
        if len(self.action_history) < 2:
            return False
        try:
            for action in self.action_history[-self.action_history_window:]:
                if action[-1] > 0.5:
                    return True
        except Exception:
            return False

    def _is_target_phrase(self, phrase: str, target_objects: Sequence[str]) -> bool:
        phrase_key = normalize_phrase_key(phrase)
        if not phrase_key:
            return False
        for target in target_objects or []:
            target_key = normalize_phrase_key(target)
            if not target_key:
                continue
            if target_key in phrase_key or phrase_key in target_key:
                return True
        return False

    def _motion_thresholds(self):
        if self.object_motion_metric in {"depth3d", "sim_gt"}:
            return (
                self.object_motion_speed_threshold,
                self.object_motion_speed_exit_threshold,
                self.object_motion_fast_speed_threshold,
            )
        return (
            self.object_motion_threshold,
            self.object_motion_exit_threshold,
            self.object_motion_fast_threshold,
        )

    # 已看完 实际是基于选择的estimator来计算物体的运动，可能是基于2D bbox的，也可能是基于3D点云的，甚至是sim_gt的
    def _compute_object_motion(self, info_dict, robot_state=None):
        boxes_xyxy = info_dict.get("dino_boxes", [])
        phrases = info_dict.get("phrases", [])
        if not boxes_xyxy or not phrases:
            self.motion_estimator.reset()
            self.object_motion_state = {}
            self.object_motion_counts = {}
            return [], {}, {}, {}

        motion_observations = self.motion_estimator.compute(
            phrases=phrases,
            boxes_xyxy=boxes_xyxy,
            robot_state=robot_state,
        )
        movement_by_phrase = {phrase: obs.magnitude for phrase, obs in motion_observations.items()}
        motion_vector_by_phrase = {phrase: obs.vector for phrase, obs in motion_observations.items()}
        motion_info_by_phrase = {
            phrase: {
                "source": obs.source,
                "units": obs.units,
                "depth_valid_ratio": obs.depth_valid_ratio,
                "depth_median": obs.depth_median,
                "fallback_used": obs.fallback_used,
                "point_world": obs.point_world.tolist() if obs.point_world is not None else None,
                "projected_pixel": obs.projected_pixel.tolist() if obs.projected_pixel is not None else None,
            }
            for phrase, obs in motion_observations.items()
        }
        if self.object_motion_metric == "sim_gt":
            logger.debug(
                "sim_gt motion debug: estimator=%s movement_by_phrase=%s motion_info_by_phrase=%s",
                getattr(self.motion_estimator, "last_debug_info", {}),
                movement_by_phrase,
                motion_info_by_phrase,
            )
        elif self.object_motion_metric == "depth3d":
            logger.info(
                "depth3d motion debug: movement_by_phrase=%s motion_info_by_phrase=%s",
                movement_by_phrase,
                motion_info_by_phrase,
            )

        active_phrases = set(phrases)
        self.object_motion_state = {
            phrase: state
            for phrase, state in self.object_motion_state.items()
            if phrase in active_phrases
        }
        self.object_motion_counts = {
            phrase: count
            for phrase, count in self.object_motion_counts.items()
            if phrase in active_phrases
        }
        moving_phrases = []
        enter_threshold, exit_threshold, _ = self._motion_thresholds()
        for phrase, dist in movement_by_phrase.items():
            prev_state = self.object_motion_state.get(phrase, False)
            count = self.object_motion_counts.get(phrase, 0)
            enter = dist > enter_threshold
            stay = dist > exit_threshold
            # 让moving和不moving的状态持续一段时间
            if prev_state: #在移动
                next_state = stay
                next_count = 0 if next_state else count + 1 #进入不移动
                if next_count >= self.object_motion_exit_persistence: #超过次数就该状态
                    next_state = False
                    next_count = 0
            else:
                next_count = count + 1 if enter else 0
                next_state = next_count >= self.object_motion_enter_persistence
                if next_state:
                    next_count = 0

            self.object_motion_state[phrase] = next_state
            self.object_motion_counts[phrase] = next_count
            if next_state:
                moving_phrases.append(phrase)
        return moving_phrases, movement_by_phrase, motion_vector_by_phrase, motion_info_by_phrase

    def _compute_relative_target_progress(self, info_dict, phrase_states, gripper_box=None):
        def _empty_result():
            return {
                "center_distance": None,
                "edge_distance": None,
                "overlap_ratio": None,
                "center_inside_target": False,
                "progress_improving": None,
                "pair": [],
                "source": None,
            }

        def _build_result(progress_improving, source_name):
            return {
                "center_distance": float(best_center_distance),
                "edge_distance": float(best_edge_distance),
                "overlap_ratio": float(best_overlap_ratio),
                "center_inside_target": bool(best_center_inside),
                "progress_improving": progress_improving,
                "pair": list(best_pair),
                "source": source_name,
            }

        boxes_xyxy = info_dict.get("dino_boxes", [])
        phrases = info_dict.get("phrases", [])
        if not boxes_xyxy or not phrases:
            self.relative_progress_history.clear()
            return _empty_result()

        phrase_centers = {}
        phrase_boxes = {}
        for idx, phrase in enumerate(phrases):
            box = boxes_xyxy[idx]
            phrase_box = torch.tensor([float(v) for v in box], dtype=torch.float32)
            x1, y1, x2, y2 = [float(v) for v in phrase_box]
            phrase_boxes[phrase] = phrase_box
            phrase_centers[phrase] = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        target_objects = info_dict.get("target_object") or []
        tograsp_objects = info_dict.get("tograsp_objects") or []
        reference_objects = info_dict.get("reference_objects") or []
        pregrasp_focus_objects = tograsp_objects or reference_objects or target_objects
        postgrasp_focus_objects = reference_objects or target_objects

        grasped_phrases = [
            phrase for phrase, state in phrase_states.items()
            if state == "grasped" and phrase in phrase_boxes
        ]

        using_gripper_source = False # false，抓后，true，抓前
        source_candidates = []
        
        # source的判断还是挺准的，要么griper要么抓住的物体
        if grasped_phrases:
            grasped_source_phrases = [
                phrase for phrase in grasped_phrases
                if self._is_target_phrase(phrase, tograsp_objects or target_objects)
            ] or list(grasped_phrases)
            source_candidates = [
                (phrase, phrase_boxes[phrase], phrase_centers[phrase])
                for phrase in grasped_source_phrases # source是抓住的物体
            ]
            moving_target_phrases = [
                phrase for phrase, state in phrase_states.items()
                if "free_moving" in state
                and phrase in phrase_boxes
                and self._is_target_phrase(phrase, postgrasp_focus_objects)
            ]
            progress_history = self.relative_progress_history
            self.pregrasp_relative_progress_history.clear()
        elif gripper_box is not None:
            gx1, gy1, gx2, gy2 = [float(v) for v in gripper_box]
            gripper_center = np.array([(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0], dtype=np.float32)
            source_candidates = [("gripper", gripper_box, gripper_center)] # source是抓手
            moving_target_phrases = [
                phrase for phrase, state in phrase_states.items()
                if "free_moving" in state
                and phrase in phrase_boxes
                and self._is_target_phrase(phrase, pregrasp_focus_objects)
            ]
            progress_history = self.pregrasp_relative_progress_history
            self.relative_progress_history.clear()
            using_gripper_source = True
        else:
            self.relative_progress_history.clear()
            self.pregrasp_relative_progress_history.clear()
            return _empty_result()

        if not moving_target_phrases:
            self.relative_progress_history.clear()
            self.pregrasp_relative_progress_history.clear()
            return _empty_result()

        best_pair = None
        best_score = None
        best_center_distance = None
        best_edge_distance = None
        best_overlap_ratio = None
        best_center_inside = False
        
        # source和目标的距离是否在提升
        for source_phrase, source_box, source_center in source_candidates:
            for moving_phrase in moving_target_phrases: #就是在free moving的物体，找到最接近的当成目标
                moving_center = phrase_centers[moving_phrase]
                moving_box = phrase_boxes[moving_phrase]
                center_distance = float(np.linalg.norm(source_center - moving_center))
                edge_distance = compute_box_edge_distance(source_box, moving_box)
                overlap_ratio = intersection_over_box_area(source_box, moving_box)
                center_inside = box_center_inside(source_box, moving_box)
                # Prefer pairs that are more geometrically aligned for placement:
                # center-inside first, then larger overlap, then smaller edge gap,
                # and only then smaller center distance.
                score = (
                    0 if center_inside else 1,
                    -overlap_ratio,
                    center_distance,
                    edge_distance,
                )
                # 没有重叠时edge_distance 然后用overlap_ratio指标越小越好
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (source_phrase, moving_phrase)
                    best_center_distance = center_distance
                    best_edge_distance = edge_distance
                    best_overlap_ratio = overlap_ratio
                    best_center_inside = center_inside

        if best_pair is None or best_edge_distance is None:
            progress_history.clear()
            return _empty_result()

        # progress的指标
        progress_metric = -float(best_overlap_ratio) if best_center_inside or (best_overlap_ratio or 0.0) > 0.0 else float(best_edge_distance)
        progress_history.append(progress_metric)
        source_name = "pregrasp_gripper" if using_gripper_source else "postgrasp_object"
        if len(progress_history) < self.relative_progress_lookback:
            return _build_result(None, source_name)

        oldest = float(progress_history[0])
        newest = float(progress_history[-1])
        progress_delta = oldest - newest
        improving = progress_delta > self.relative_progress_improve_threshold
        return _build_result(improving, source_name)

    def _motion_penalty_bucket(
        self,
        active_penalty_motion,
        relative_center_distance,
        grasped_estimate,
        relative_progress_source,
        relative_progress_improving,
        relative_progress_pair,
        reference_objects,
        target_objects,
    ):
        if (
            grasped_estimate
            and relative_progress_source == "postgrasp_object"
            and relative_center_distance is not None
            and relative_center_distance < 0.15
            and relative_progress_improving is False
            and isinstance(relative_progress_pair, (list, tuple))
            and len(relative_progress_pair) >= 2
            and self._is_target_phrase(relative_progress_pair[1], reference_objects or target_objects)
        ):
            return 2, "stalled"

        # if self.object_motion_metric in {"depth3d", "sim_gt"}:
        #     enter_threshold, exit_threshold, fast_threshold = self._motion_thresholds()
        #     if active_penalty_motion >= fast_threshold:
        #         return 2, "very fast"
        #     if active_penalty_motion >= enter_threshold:
        #         if relative_center_distance is not None and relative_center_distance < 0.1:
        #             return 5, "fast and close"
        #         return 5, "fast"
        #     if active_penalty_motion >= exit_threshold:
        #         if relative_center_distance is not None and relative_center_distance < 0.1:
        #             return 10, "medium fast and close"
        #         return 10, "medium"
        #     if active_penalty_motion >= 0.5 * exit_threshold:
        #         return 15, "slow"
        #     return 20, "very slow"
        if self.object_motion_metric in {"depth3d", "sim_gt"}:
            if active_penalty_motion >= 0.15:
                return 2, "very fast"
            if active_penalty_motion >= 0.1:
                if relative_center_distance is not None and relative_center_distance < 0.1:
                    return 2, "fast and close"
                return 5, "fast"
            if active_penalty_motion >= 0.05:
                if relative_center_distance is not None and relative_center_distance < 0.1:
                    return 5, "medium fast and close"
                return 10, "medium"
            if active_penalty_motion >= 0.02:
                if relative_center_distance is not None and relative_center_distance < 0.1:
                    return 10, "slow and close"
                return 15, "slow"
            return 20, "very slow"
        else:
            if active_penalty_motion >= 0.03:
                return 2, "very fast"
            if active_penalty_motion >= 0.02:
                if relative_center_distance is not None and relative_center_distance < 0.1:
                    return 2, "fast and close"
                return 5, "fast"
            if active_penalty_motion >= 0.01:
                if relative_center_distance is not None and relative_center_distance < 0.1:
                    return 5, "medium fast and close"
                return 10, "medium"
            if active_penalty_motion >= 0.005:
                return 15, "slow"
            return 20, "very slow"

    # 这个是对的不用看
    def _project_gripper_to_image_box(self, robot_state):
        if not robot_state:
            return None
        ee_position = robot_state.get("ee_position")
        world_to_image = robot_state.get("camera_world_to_image")
        camera_resolution = robot_state.get("camera_resolution")
        if ee_position is None or world_to_image is None or camera_resolution is None:
            return None

        try:
            from robosuite.utils.camera_utils import project_points_from_world_to_camera

            point = np.array([[ee_position[0], ee_position[1], ee_position[2]]], dtype=np.float32)
            transform = np.array(world_to_image, dtype=np.float32)
            pixel_h = int(camera_resolution[0])
            pixel_w = int(camera_resolution[1])
            projected = project_points_from_world_to_camera(
                points=point,
                world_to_camera_transform=transform,
                camera_height=pixel_h,
                camera_width=pixel_w,
            )
            row = int(projected[0][0])
            col = int(projected[0][1])
            if robot_state.get("camera_observation_mirror_x", False):
                col = (pixel_w - 1) - col
            if robot_state.get("camera_observation_mirror_y", False):
                row = (pixel_h - 1) - row
            half = self.gripper_box_half_size_px
            x1 = max(0, col - half)
            y1 = max(0, row - half)
            x2 = min(pixel_w - 1, col + half)
            y2 = min(pixel_h - 1, row + half)
            if x2 <= x1 or y2 <= y1:
                return None
            return torch.tensor(
                [
                    x1 / max(pixel_w, 1),
                    y1 / max(pixel_h, 1),
                    x2 / max(pixel_w, 1),
                    y2 / max(pixel_h, 1),
                ],
                dtype=torch.float32,
            )
        except Exception:
            return None

    # 看完了 gripper投影并判断接触
    def _update_projected_grasp_state(self, info_dict, robot_state, gripper_closed: bool):
        boxes_xyxy = info_dict.get("dino_boxes", [])
        phrases = info_dict.get("phrases", [])
        gripper_box = self._project_gripper_to_image_box(robot_state)
        if not gripper_closed or gripper_box is None:
            self.gripper_contact_counts = {}
            self.grasped_phrases = set()
            return gripper_box, [], False

        contact_candidates = []
        next_counts = {}
        for idx, phrase in enumerate(phrases):
            box = boxes_xyxy[idx]
            phrase_box = torch.tensor([float(v) for v in box], dtype=torch.float32)
            iou = single_box_iou(gripper_box, phrase_box)
            center_dist = single_box_center_distance(gripper_box, phrase_box)
            is_contact = (
                iou >= self.gripper_contact_iou_threshold
                or center_dist <= self.gripper_contact_center_dist_threshold
            )
            if is_contact:
                count = self.gripper_contact_counts.get(phrase, 0) + 1
                next_counts[phrase] = count
                if count >= self.gripper_contact_persistence:
                    inter_x1 = max(float(gripper_box[0]), float(phrase_box[0]))
                    inter_y1 = max(float(gripper_box[1]), float(phrase_box[1]))
                    inter_x2 = min(float(gripper_box[2]), float(phrase_box[2]))
                    inter_y2 = min(float(gripper_box[3]), float(phrase_box[3]))
                    overlap_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    contact_candidates.append((phrase, center_dist, overlap_area))
        self.gripper_contact_counts = next_counts
        if contact_candidates: #排序判断抓住的物体
            # If multiple target boxes overlap the gripper region, prefer the
            # one most likely to be actually grasped: nearest to gripper, then
            # larger overlap area with the gripper box.
            contact_candidates.sort(key=lambda item: (item[1], -item[2]))
            grasped_phrase = contact_candidates[0][0]
            contact_phrases = [grasped_phrase]
        else:
            contact_phrases = []
        self.grasped_phrases = set(contact_phrases)
        return gripper_box, contact_phrases, bool(contact_phrases)

    # 这个其实没啥用，后面我改了
    def _boxes_are_consistent(
        self,
        tracked_boxes_xyxy: torch.Tensor,
        tracked_phrases: List[str],
        dino_boxes_xyxy: torch.Tensor,
        dino_phrases: List[str],
    ) -> bool:
        if len(tracked_boxes_xyxy) == 0 or len(dino_boxes_xyxy) == 0:
            return False

        tracked_by_phrase = {
            phrase: tracked_boxes_xyxy[idx]
            for idx, phrase in enumerate(tracked_phrases)
            if idx < len(tracked_boxes_xyxy)
        }
        dino_by_phrase = {
            phrase: dino_boxes_xyxy[idx]
            for idx, phrase in enumerate(dino_phrases)
            if idx < len(dino_boxes_xyxy)
        }

        shared_phrases = [phrase for phrase in dino_by_phrase if phrase in tracked_by_phrase]
        if not shared_phrases:
            return False

        accept_iou = self.dino_accept_iou
        accept_center_dist = self.dino_accept_center_dist
        if self._recent_gripper_close():
            # Carry phase: be stricter before allowing DINO to override tracker.
            accept_iou = max(accept_iou, 0.3)
            accept_center_dist = min(accept_center_dist, 0.08)

        for phrase in shared_phrases:
            tracked_box = tracked_by_phrase[phrase]
            dino_box = dino_by_phrase[phrase]
            iou = single_box_iou(tracked_box, dino_box)
            center_dist = single_box_center_distance(tracked_box, dino_box)
            if iou < accept_iou and center_dist > accept_center_dist:
                logger.info(
                    "Rejecting DINO refresh for phrase '%s': iou=%.3f, center_dist=%.3f",
                    phrase,
                    iou,
                    center_dist,
                )
                return False
        return True

    def _set_cached_boxes(
        self,
        boxes_cxcywh: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        logits: torch.Tensor,
        phrases: List[str],
        image_source,
    ):
        self.cached_boxes = (
            boxes_cxcywh.detach().cpu() if isinstance(boxes_cxcywh, torch.Tensor) else boxes_cxcywh,
            boxes_xyxy.detach().cpu() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy,
            logits.detach().cpu() if isinstance(logits, torch.Tensor) else logits,
            list(phrases),
            image_source,
        )

    def _initialize_trackers(self, image, boxes_xyxy: torch.Tensor, phrases: List[str], prompt_key: Tuple[str, ...]):
        frame = ensure_frame(image)
        self.trackers = []
        self.tracker_phrases = []
        self.tracker_prompt = prompt_key
        if frame is None or len(boxes_xyxy) == 0:
            return

        for box_xyxy, phrase in zip(boxes_xyxy, phrases):
            tracker = self._create_tracker()
            if tracker is None:
                logger.warning("No OpenCV tracker backend available; falling back to cached boxes.")
                self.trackers = []
                self.tracker_phrases = []
                return

            bbox = norm_xyxy_to_bbox(box_xyxy, frame.shape)
            try:
                ok = tracker.init(frame, bbox)
            except Exception as exc:
                logger.warning(f"Tracker init failed for phrase '{phrase}': {exc}")
                continue

            if ok is False:
                continue

            self.trackers.append(tracker)
            self.tracker_phrases.append(phrase)

    def _tracker_result_from_cached(self):
        if self.cached_boxes is None:
            return self._empty_box_result()
        return self.cached_boxes

    def get_tracked_boxes(self, image, text_prompt):
        prompt_key = normalize_prompt(text_prompt)
        if not self.trackers or self.tracker_prompt != prompt_key:
            return self._tracker_result_from_cached()

        frame = ensure_frame(image)
        if frame is None:
            return self._tracker_result_from_cached()

        tracked_boxes = []
        tracked_phrases = []
        cached_logits_by_phrase = {}
        if self.cached_boxes is not None:
            cached_logits = self.cached_boxes[2]
            cached_phrases = self.cached_boxes[3]
            for idx, phrase in enumerate(cached_phrases):
                if cached_logits is None:
                    break
                try:
                    value = cached_logits[idx]
                except Exception:
                    continue
                cached_logits_by_phrase[phrase] = value
        for tracker, phrase in zip(self.trackers, self.tracker_phrases):
            try:
                ok, bbox = tracker.update(frame)
            except Exception as exc:
                logger.warning(f"Tracker update failed for phrase '{phrase}': {exc}")
                ok = False
                bbox = None

            if not ok or bbox is None:
                continue

            tracked_boxes.append(bbox_to_norm_xyxy(bbox, frame.shape))
            tracked_phrases.append(phrase)

        if not tracked_boxes:
            self.trackers = []
            self.tracker_phrases = []
            return self._tracker_result_from_cached()

        boxes_xyxy = torch.stack(tracked_boxes)
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        logits = []
        for phrase in tracked_phrases:
            value = cached_logits_by_phrase.get(phrase, 1.0)
            try:
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "item"):
                    value = value.item()
                value = float(value)
            except Exception:
                value = 1.0
            logits.append(value)
        logits = torch.tensor(logits, dtype=torch.float32)
        self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, tracked_phrases, frame)
        return self.cached_boxes

    def _run_dino_refresh(self, image, text_prompt):
        boxes_cxcywh, boxes_xyxy, logits, phrases, image_source = super().get_dino_boxes(image, text_prompt)
        self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
        return boxes_cxcywh, boxes_xyxy, logits, phrases, image_source

    def _should_refresh_dino(self, prompt_key: Tuple[str, ...]) -> bool:
        if self.cached_boxes is None:
            return True
        if self.tracker_prompt != prompt_key:
            return True
        if not self.trackers:
            return True
        return self.box_query_count % self.dino_refresh_interval == 0

    def get_dino_boxes(self, image, text_prompt):
        """Keep the original interface while internally mixing DINO and tracker.

        Every `dino_refresh_interval` calls we refresh detections with DINO.
        Between refreshes we reuse tracker-updated boxes.
        """
        prompt_key = normalize_prompt(text_prompt)
        self.box_query_count += 1

        if self._should_refresh_dino(prompt_key):
            tracked = self.get_tracked_boxes(image, text_prompt)
            dino_result = self._run_dino_refresh(image, text_prompt)

            tracked_boxes_xyxy = tracked[1] if tracked else torch.tensor([])
            tracked_phrases = tracked[3] if tracked else []
            dino_boxes_xyxy = dino_result[1]
            dino_phrases = dino_result[3]

            if len(tracked_boxes_xyxy) > 0 and not self._boxes_are_consistent(
                tracked_boxes_xyxy, tracked_phrases, dino_boxes_xyxy, dino_phrases
            ):
                # Rebuild trackers from the trusted tracked result instead of the disagreeing DINO refresh.
                tracked_boxes_cxcywh = tracked[0]
                tracked_logits = tracked[2]
                tracked_image = ensure_frame(image)
                self._set_cached_boxes(
                    tracked_boxes_cxcywh,
                    tracked_boxes_xyxy,
                    tracked_logits,
                    tracked_phrases,
                    tracked_image,
                )
                self._initialize_trackers(image, tracked_boxes_xyxy, tracked_phrases, prompt_key)
                self.last_query_used_dino = False
                return self.cached_boxes

            self._initialize_trackers(image, dino_boxes_xyxy, dino_phrases, prompt_key)
            self.last_query_used_dino = True
            return dino_result

        tracked = self.get_tracked_boxes(image, text_prompt)
        tracked_boxes = tracked[0] if tracked else torch.tensor([])
        if len(tracked_boxes) == 0:
            dino_result = self._run_dino_refresh(image, text_prompt)
            self._initialize_trackers(image, dino_result[1], dino_result[3], prompt_key)
            self.last_query_used_dino = True
            return dino_result

        self.last_query_used_dino = False
        return tracked

    def propose_adaptive_chunk_heuristic(
        self,
        image,
        instruction,
        robot_state=None,
        object_state=None,
        *,
        current_horizon=None,
    ):
        proposal = super().propose_adaptive_chunk_heuristic(
            image,
            instruction,
            robot_state=robot_state,
            object_state=object_state,
            current_horizon=current_horizon,
        )
        info_dict = proposal.get("info")
        proposed_horizon = proposal.get("current_horizon", self.current_horizon)
        needs_interruption = proposal.get("needs_interruption", False)
        logging.info("Base proposed horizon: %d", proposed_horizon)
        moving_phrases, movement_by_phrase, _, motion_info_by_phrase = self._compute_object_motion(
            info_dict,
            robot_state=robot_state,
        )
        gripper_value = robot_state.get("gripper_state") if robot_state else None

        gripper_closed = self._recent_gripper_close()

        gripper_box, gripper_contact_phrases, grasped_estimate = self._update_projected_grasp_state(
            info_dict, robot_state, gripper_closed
        )
        recent_gripper_close = self._recent_gripper_close()
        contact_phrases = set(gripper_contact_phrases)
        grasped_phrase_set = set(self.grasped_phrases)
        target_objects = info_dict.get("target_object") or []
        tograsp_objects = info_dict.get("tograsp_objects") or []
        reference_objects = info_dict.get("reference_objects") or []
        pregrasp_focus_objects = tograsp_objects or reference_objects or target_objects
        postgrasp_focus_objects = reference_objects or target_objects

        phrase_states = {}
        free_moving_phrases = []
        free_moving_motion = 0.0
        for phrase, motion in movement_by_phrase.items():
            moving_now = phrase in moving_phrases
            in_contact = phrase in contact_phrases
            grasp_latched = phrase in grasped_phrase_set
            if grasped_estimate:
                is_focus_phrase = self._is_target_phrase(phrase, postgrasp_focus_objects)
            else:
                is_focus_phrase = self._is_target_phrase(phrase, pregrasp_focus_objects)

            if grasp_latched:
                state = "grasped"
            elif in_contact:
                state = "contact"
            elif moving_now and is_focus_phrase: # 在移动且在当前阶段目标中
                state = "focus free_moving"
            elif moving_now: #在移动但不在当前阶段目标中
                state = "not focus free_moving"
            else:
                state = "stable"

            phrase_states[phrase] = state
            if "free_moving" in state:
                free_moving_phrases.append(phrase)
                free_moving_motion = max(free_moving_motion, float(motion))

        grasp_protected = any(
            state in {"contact", "grasped"} for state in phrase_states.values()
        )

        info_dict["object_motion_metric"] = self.object_motion_metric
        info_dict["object_motion_units"] = (
            "meters_per_second" if self.object_motion_metric in {"depth3d", "sim_gt"} else "norm_delta"
        )
        info_dict["moving_phrases"] = moving_phrases
        info_dict["movement_by_phrase"] = movement_by_phrase
        info_dict["moving_phrase_motion_by_phrase"] = {
            phrase: float(movement_by_phrase[phrase])
            for phrase in moving_phrases
            if phrase in movement_by_phrase
        }
        info_dict["motion_info_by_phrase"] = motion_info_by_phrase
        info_dict["gripper_box"] = gripper_box.tolist() if gripper_box is not None else None
        info_dict["gripper_contact_phrases"] = gripper_contact_phrases
        info_dict["grasped_phrases"] = sorted(self.grasped_phrases)
        info_dict["grasped_estimate"] = grasped_estimate
        info_dict["phrase_states"] = phrase_states
        info_dict["grasp_protected"] = grasp_protected
        info_dict["free_moving_phrases"] = free_moving_phrases
        info_dict["free_moving_max"] = free_moving_motion
        relative_progress = self._compute_relative_target_progress(info_dict, phrase_states, gripper_box=gripper_box)
        info_dict["relative_target_center_distance"] = relative_progress["center_distance"]
        info_dict["relative_target_edge_distance"] = relative_progress["edge_distance"]
        info_dict["relative_target_overlap_ratio"] = relative_progress["overlap_ratio"]
        info_dict["relative_target_center_inside"] = relative_progress["center_inside_target"]
        info_dict["relative_target_progress_improving"] = relative_progress["progress_improving"]
        info_dict["relative_target_progress_pair"] = relative_progress["pair"]
        info_dict["relative_target_progress_source"] = relative_progress["source"]
        relative_center_distance = info_dict["relative_target_center_distance"]
        relative_edge_distance = info_dict["relative_target_edge_distance"]
        relative_overlap_ratio = info_dict["relative_target_overlap_ratio"]
        relative_center_inside = info_dict["relative_target_center_inside"]
        relative_progress_improving = info_dict["relative_target_progress_improving"]
        relative_progress_pair = info_dict["relative_target_progress_pair"]
        relative_progress_source = info_dict["relative_target_progress_source"]
        info_dict["relative_target_available"] = bool(relative_progress_pair)

        # 感觉和前面有点重复了
        active_penalty_targets = postgrasp_focus_objects if grasped_estimate else pregrasp_focus_objects
        active_penalty_phrases = []
        active_penalty_motion = 0.0
        for phrase in free_moving_phrases:
            should_penalize = self._is_target_phrase(phrase, active_penalty_targets)
            if should_penalize:
                active_penalty_phrases.append(phrase)
                active_penalty_motion = max(active_penalty_motion, float(movement_by_phrase.get(phrase, 0.0)))

        info_dict["motion_penalty_phrases"] = active_penalty_phrases
        info_dict["motion_penalty_max"] = active_penalty_motion
        info_dict["motion_penalty_by_phrase"] = {
            phrase: float(movement_by_phrase.get(phrase, 0.0))
            for phrase in active_penalty_phrases
        }

        skip_motion_penalty_far_target = (
            relative_center_distance is not None
            and float(relative_center_distance) > self.motion_penalty_max_center_distance
        )
        if skip_motion_penalty_far_target:
            info_dict.setdefault("factors", []).append(
                f"Skip motion penalty: relative center distance {float(relative_center_distance):.4f} > {self.motion_penalty_max_center_distance:.4f}"
            )

        logger.info(
            f"active_penalty_phrases: {active_penalty_phrases}, skip_motion_penalty_far_target: {skip_motion_penalty_far_target}"
        )

        if self.enable_object_motion_penalty and active_penalty_phrases and not skip_motion_penalty_far_target:
            pre_penalty_horizon = proposed_horizon
            target_horizon, motion_bucket = self._motion_penalty_bucket(
                active_penalty_motion,
                relative_center_distance,
                grasped_estimate,
                relative_progress_source,
                relative_progress_improving,
                relative_progress_pair,
                reference_objects,
                target_objects,
            )
            proposed_horizon = min(proposed_horizon, target_horizon)
            needs_interruption = needs_interruption or (proposed_horizon < pre_penalty_horizon)
            info_dict.setdefault("factors", []).append(
                f"Object moving without grasp ({','.join(active_penalty_phrases)}): {motion_bucket}"
            )
            logger.info(
                "Reducing horizon due to object motion without grasp: penalty_phrases=%s, penalty_phrase_motion=%s, motion_bucket=%s, target_horizon=%s, relative_progress_improving=%s, relative_target_edge_distance=%s, relative_target_center_distance=%s, relative_target_overlap_ratio=%s, relative_target_center_inside=%s, relative_target_pair=%s",
                active_penalty_phrases,
                info_dict["motion_penalty_by_phrase"],
                motion_bucket,
                target_horizon,
                relative_progress_improving,
                relative_edge_distance,
                relative_center_distance,
                relative_overlap_ratio,
                relative_center_inside,
                relative_progress_pair,
            )

        logger.info(
            "horizon factors: phrase_states=%s, free_moving=%s, penalty_phrases=%s, penalty_motion=%s, grasped_estimate=%s, relative_center_distance=%s, relative_edge_distance=%s, relative_overlap_ratio=%s, relative_center_inside=%s, relative_progress_improving=%s, relative_target_pair=%s",
            phrase_states,
            free_moving_phrases,
            active_penalty_phrases,
            info_dict["motion_penalty_by_phrase"],
            grasped_estimate,
            relative_center_distance,
            relative_edge_distance,
            relative_overlap_ratio,
            relative_center_inside,
            relative_progress_improving,
            relative_progress_pair,
        )
        logger.info(f"current_horizon in heuristic: {proposed_horizon}")
        return {
            "current_horizon": proposed_horizon,
            "needs_interruption": needs_interruption,
            "info": info_dict,
        }
