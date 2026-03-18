import logging
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .adaptivepolicyfinegrained import AdaptiveActionChunker as BaseAdaptiveActionChunker
from .object_extractor.mock_extractor import extract_objects

logger = logging.getLogger("adaptive.tracker")


class AdaptiveActionChunker(BaseAdaptiveActionChunker):
    """Tracker-backed variant of the fine-grained adaptive chunker.

    DINO is still used when `ValueFunc` runs during VLA inference.
    Between VLA inferences, the heuristic path reuses tracker-updated boxes to
    avoid repeatedly querying DINO.
    """

    def __init__(
        self,
        *args,
        tracker_backend: str = "csrt",
        dino_refresh_interval: int = 20,
        dino_accept_iou: float = 0.1,
        dino_accept_center_dist: float = 0.15,
        object_motion_threshold: float = 0.03,
        grasp_distance_threshold: float = 0.08,
        gripper_closed_threshold: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tracker_backend = tracker_backend.lower()
        self.dino_refresh_interval = max(1, int(dino_refresh_interval))
        self.dino_accept_iou = float(dino_accept_iou)
        self.dino_accept_center_dist = float(dino_accept_center_dist)
        self.object_motion_threshold = float(object_motion_threshold)
        self.grasp_distance_threshold = float(grasp_distance_threshold)
        self.gripper_closed_threshold = float(gripper_closed_threshold)
        self.trackers: List[Any] = []
        self.tracker_phrases: List[str] = []
        self.tracker_prompt: Optional[Tuple[str, ...]] = None
        self.cached_boxes = None
        self.box_query_count = 0
        self.last_query_used_dino = False
        self.active_tracker_backend: Optional[str] = None
        self.prev_object_centers: dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.trackers = []
        self.tracker_phrases = []
        self.tracker_prompt = None
        self.cached_boxes = None
        self.box_query_count = 0
        self.last_query_used_dino = False
        self.active_tracker_backend = None
        self.prev_object_centers = {}

    def get_runtime_config(self):
        config = super().get_runtime_config()
        config.update(
            {
                "policy_type": "finegrained_tracker",
                "tracker_backend": self.tracker_backend,
                "dino_refresh_interval": self.dino_refresh_interval,
                "dino_accept_iou": self.dino_accept_iou,
                "dino_accept_center_dist": self.dino_accept_center_dist,
                "object_motion_threshold": self.object_motion_threshold,
                "grasp_distance_threshold": self.grasp_distance_threshold,
                "gripper_closed_threshold": self.gripper_closed_threshold,
            }
        )
        return config

    def _empty_box_result(self):
        empty = torch.tensor([])
        return empty, empty, torch.tensor([]), [], None

    def _normalize_prompt(self, text_prompt: Sequence[str] | str) -> Tuple[str, ...]:
        if isinstance(text_prompt, str):
            return (text_prompt,)
        return tuple(text_prompt)

    def _ensure_frame(self, image) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            frame = image.copy()
        else:
            frame = cv2.imread(image)

        if frame is None:
            return None

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

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
            factory = self._get_tracker_factory(cv2, class_name)
            if factory is not None:
                yield class_name.replace("Tracker", "").lower(), factory
            legacy = getattr(cv2, "legacy", None)
            if legacy is not None:
                factory = self._get_tracker_factory(legacy, class_name)
                if factory is not None:
                    yield class_name.replace("Tracker", "").lower(), factory

    def _get_tracker_factory(self, module, class_name: str):
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

    def _create_tracker(self):
        for backend_name, factory in self._tracker_factory_candidates():
            try:
                tracker = factory()
                self.active_tracker_backend = backend_name
                return tracker
            except Exception:
                continue
        return None

    def _norm_xyxy_to_bbox(self, box_xyxy_norm: torch.Tensor, image_shape: Tuple[int, int, int]):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
        x1 = np.clip(x1 * w, 0, max(w - 1, 0))
        y1 = np.clip(y1 * h, 0, max(h - 1, 0))
        x2 = np.clip(x2 * w, x1 + 1, w)
        y2 = np.clip(y2 * h, y1 + 1, h)
        # OpenCV tracker init is picky about bbox types; use plain Python ints.
        x = int(round(float(x1)))
        y = int(round(float(y1)))
        bw = int(round(float(max(1.0, x2 - x1))))
        bh = int(round(float(max(1.0, y2 - y1))))
        return (x, y, bw, bh)

    def _bbox_to_norm_xyxy(self, bbox, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        h, w = image_shape[:2]
        x, y, bw, bh = bbox
        x1 = float(np.clip(x / max(w, 1), 0.0, 1.0))
        y1 = float(np.clip(y / max(h, 1), 0.0, 1.0))
        x2 = float(np.clip((x + bw) / max(w, 1), 0.0, 1.0))
        y2 = float(np.clip((y + bh) / max(h, 1), 0.0, 1.0))
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _xyxy_to_cxcywh(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if len(boxes_xyxy) == 0:
            return torch.tensor([])
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1e-4)
        h = (y2 - y1).clamp(min=1e-4)
        return torch.stack([cx, cy, w, h], dim=1)

    def _single_box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
        area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
        union = area1 + area2 - inter
        if union <= 1e-9:
            return 0.0
        return inter / union

    def _single_box_center_distance(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        c1x = (float(box1[0]) + float(box1[2])) / 2.0
        c1y = (float(box1[1]) + float(box1[3])) / 2.0
        c2x = (float(box2[0]) + float(box2[2])) / 2.0
        c2y = (float(box2[1]) + float(box2[3])) / 2.0
        return float(np.linalg.norm([c1x - c2x, c1y - c2y]))

    def _recent_gripper_close(self) -> bool:
        if len(self.action_history) < 2:
            return False
        try:
            return (self.action_history[-1][-1] - self.action_history[-2][-1]) > 1.5
        except Exception:
            return False

    def _compute_object_motion(self, info_dict):
        boxes_xyxy = info_dict.get("dino_boxes", [])
        phrases = info_dict.get("phrases", [])
        if not boxes_xyxy or not phrases:
            self.prev_object_centers = {}
            return 0.0, [], {}

        centers_now = {}
        movement_by_phrase = {}
        for idx, phrase in enumerate(phrases):
            if idx >= len(boxes_xyxy):
                continue
            box = boxes_xyxy[idx]
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            centers_now[phrase] = center
            prev_center = self.prev_object_centers.get(phrase)
            if prev_center is not None:
                movement_by_phrase[phrase] = float(np.linalg.norm(center - prev_center))

        self.prev_object_centers = centers_now
        max_motion = max(movement_by_phrase.values()) if movement_by_phrase else 0.0
        moving_phrases = [
            phrase for phrase, dist in movement_by_phrase.items() if dist > self.object_motion_threshold
        ]
        return max_motion, moving_phrases, movement_by_phrase

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
            accept_iou = max(accept_iou, 0.35)
            accept_center_dist = min(accept_center_dist, 0.08)

        for phrase in shared_phrases:
            tracked_box = tracked_by_phrase[phrase]
            dino_box = dino_by_phrase[phrase]
            iou = self._single_box_iou(tracked_box, dino_box)
            center_dist = self._single_box_center_distance(tracked_box, dino_box)
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
        frame = self._ensure_frame(image)
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
            # logging.debug(
            #     "Initializing tracker for phrase '%s' with requested backend '%s', actual backend '%s'",
            #     phrase,
            #     self.tracker_backend,
            #     self.active_tracker_backend,
            # )

            bbox = self._norm_xyxy_to_bbox(box_xyxy, frame.shape)
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
        prompt_key = self._normalize_prompt(text_prompt)
        if not self.trackers or self.tracker_prompt != prompt_key:
            return self._tracker_result_from_cached()

        frame = self._ensure_frame(image)
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

            tracked_boxes.append(self._bbox_to_norm_xyxy(bbox, frame.shape))
            tracked_phrases.append(phrase)

        if not tracked_boxes:
            self.trackers = []
            self.tracker_phrases = []
            return self._tracker_result_from_cached()

        boxes_xyxy = torch.stack(tracked_boxes)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
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
        prompt_key = self._normalize_prompt(text_prompt)
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
                tracked_image = self._ensure_frame(image)
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

    def adaptive_chunk_heuristic(self, image, instruction, robot_state=None, object_state=None):
        horizon, info_dict = super().adaptive_chunk_heuristic(
            image, instruction, robot_state=robot_state, object_state=object_state
        )

        max_motion, moving_phrases, movement_by_phrase = self._compute_object_motion(info_dict)
        min_object_distance = info_dict.get("min_object_distance")
        gripper_value = robot_state.get("gripper_state") if robot_state else None

        gripper_closed = False
        if gripper_value is not None:
            try:
                gripper_closed = float(gripper_value) <= self.gripper_closed_threshold
            except Exception:
                gripper_closed = False

        grasped_estimate = bool(
            min_object_distance is not None
            and min_object_distance < self.grasp_distance_threshold
            and (gripper_closed or self._recent_gripper_close())
        )

        info_dict["object_motion_max"] = max_motion
        info_dict["moving_phrases"] = moving_phrases
        info_dict["movement_by_phrase"] = movement_by_phrase
        info_dict["grasped_estimate"] = grasped_estimate

        if moving_phrases and not grasped_estimate:
            self.current_horizon = min(self.current_horizon, self.get_chunk_size(1))
            self.needs_interruption = True
            info_dict.setdefault("factors", []).append(
                f"Object moving without grasp ({','.join(moving_phrases)})"
            )

        logger.info(f"current_horizon in heuristic: {self.current_horizon}")
        return self.current_horizon, info_dict
