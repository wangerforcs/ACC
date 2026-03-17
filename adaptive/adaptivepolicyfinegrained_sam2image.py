import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .adaptivepolicyfinegrained import AdaptiveActionChunker as BaseAdaptiveActionChunker
from .object_extractor.mock_extractor import extract_objects


class AdaptiveActionChunker(BaseAdaptiveActionChunker):
    """Online image-predictor SAM2 variant.

    DINO is called infrequently to refresh semantic grounding. In between refreshes,
    the previous frame's boxes are used as box prompts for SAM2ImagePredictor on the
    current frame.
    """

    def __init__(
        self,
        *args,
        sam2_model_id: str = "facebook/sam2.1-hiera-tiny",
        sam2_config: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        sam2_device: Optional[str] = None,
        dino_refresh_interval: int = 10,
        mask_threshold: float = 0.0,
        dino_accept_iou: float = 0.2,
        dino_accept_center_dist: float = 0.12,
        sam2_prompt_mode: str = "box",
        sam2_box_shrink: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam2_model_id = sam2_model_id
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        self.sam2_device = sam2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_refresh_interval = max(1, int(dino_refresh_interval))
        self.mask_threshold = float(mask_threshold)
        self.dino_accept_iou = float(dino_accept_iou)
        self.dino_accept_center_dist = float(dino_accept_center_dist)
        self.sam2_prompt_mode = str(sam2_prompt_mode).lower()
        self.sam2_box_shrink = float(sam2_box_shrink)

        self.sam2_predictor: Optional[SAM2ImagePredictor] = None
        self.cached_boxes = None
        self.cached_prompt: Optional[Tuple[str, ...]] = None
        self.last_query_used_dino = False
        self.last_box_source: str = "none"
        self.last_stage_timings_ms = {}
        self.box_query_count = 0

    def reset(self):
        super().reset()
        self.cached_boxes = None
        self.cached_prompt = None
        self.last_query_used_dino = False
        self.last_box_source = "none"
        self.last_stage_timings_ms = {}
        self.box_query_count = 0

    def _ensure_sam2_predictor(self) -> SAM2ImagePredictor:
        if self.sam2_predictor is not None:
            return self.sam2_predictor

        if self.sam2_config:
            model = build_sam2(
                self.sam2_config,
                ckpt_path=self.sam2_ckpt,
                device=self.sam2_device,
            )
            self.sam2_predictor = SAM2ImagePredictor(model)
        else:
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                self.sam2_model_id,
                device=self.sam2_device,
            )
        self.sam2_predictor.model.eval()
        return self.sam2_predictor

    def _ensure_frame(self, image) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            frame = image
        else:
            frame = cv2.imread(image)
        if frame is None:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def _normalize_prompt(self, text_prompt: Sequence[str] | str) -> Tuple[str, ...]:
        if isinstance(text_prompt, str):
            return (text_prompt,)
        return tuple(text_prompt)

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

    def _empty_box_result(self):
        empty = torch.tensor([])
        return empty, empty, torch.tensor([]), [], None

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

    def _boxes_are_consistent(
        self,
        ref_boxes_xyxy: torch.Tensor,
        ref_phrases: List[str],
        dino_boxes_xyxy: torch.Tensor,
        dino_phrases: List[str],
    ) -> bool:
        if len(ref_boxes_xyxy) == 0 or len(dino_boxes_xyxy) == 0:
            return False

        ref_by_phrase = {
            phrase: ref_boxes_xyxy[idx]
            for idx, phrase in enumerate(ref_phrases)
            if idx < len(ref_boxes_xyxy)
        }
        dino_by_phrase = {
            phrase: dino_boxes_xyxy[idx]
            for idx, phrase in enumerate(dino_phrases)
            if idx < len(dino_boxes_xyxy)
        }

        shared_phrases = [phrase for phrase in dino_by_phrase if phrase in ref_by_phrase]
        if not shared_phrases:
            return False

        for phrase in shared_phrases:
            ref_box = ref_by_phrase[phrase]
            dino_box = dino_by_phrase[phrase]
            iou = self._single_box_iou(ref_box, dino_box)
            center_dist = self._single_box_center_distance(ref_box, dino_box)
            if iou < self.dino_accept_iou and center_dist > self.dino_accept_center_dist:
                logging.info(
                    "Rejecting DINO refresh for phrase '%s': iou=%.3f, center_dist=%.3f",
                    phrase,
                    iou,
                    center_dist,
                )
                return False
        return True

    def _mask_to_box_xyxy_norm(self, mask: np.ndarray) -> Optional[torch.Tensor]:
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None
        ys, xs = np.where(mask > self.mask_threshold)
        if len(xs) == 0 or len(ys) == 0:
            return None
        h, w = mask.shape
        x1 = float(xs.min()) / max(w, 1)
        y1 = float(ys.min()) / max(h, 1)
        x2 = float(xs.max() + 1) / max(w, 1)
        y2 = float(ys.max() + 1) / max(h, 1)
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _norm_box_to_abs(self, box_xyxy_norm: torch.Tensor, image_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
        return np.array(
            [
                x1 * w,
                y1 * h,
                x2 * w,
                y2 * h,
            ],
            dtype=np.float32,
        )

    def _shrink_abs_box(self, abs_box: np.ndarray) -> np.ndarray:
        shrink = min(max(self.sam2_box_shrink, 0.1), 1.0)
        if abs(shrink - 1.0) < 1e-6:
            return abs_box.astype(np.float32)
        x1, y1, x2, y2 = [float(v) for v in abs_box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        hw = 0.5 * (x2 - x1) * shrink
        hh = 0.5 * (y2 - y1) * shrink
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)

    def _center_point_from_abs_box(self, abs_box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in abs_box]
        return np.array([[0.5 * (x1 + x2), 0.5 * (y1 + y2)]], dtype=np.float32)

    def _run_dino_detection(self, image, text_prompt):
        return super().get_dino_boxes(image, text_prompt)

    def _refine_with_sam2image(self, image):
        if self.cached_boxes is None:
            return None

        boxes_xyxy_prev = self.cached_boxes[1]
        phrases = self.cached_boxes[3]
        if not isinstance(boxes_xyxy_prev, torch.Tensor) or len(boxes_xyxy_prev) == 0:
            return None

        predictor = self._ensure_sam2_predictor()
        frame = self._ensure_frame(image)
        if frame is None:
            return None
        abs_boxes = np.stack([self._shrink_abs_box(self._norm_box_to_abs(prev_box, frame.shape)) for prev_box in boxes_xyxy_prev], axis=0)

        set_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and self.sam2_device.startswith("cuda") else None
        set_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and self.sam2_device.startswith("cuda") else None
        pred_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and self.sam2_device.startswith("cuda") else None
        pred_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and self.sam2_device.startswith("cuda") else None

        if set_start is not None:
            set_start.record()
        predictor.set_image(frame)
        if set_end is not None:
            set_end.record()

        try:
            if pred_start is not None:
                pred_start.record()
            with torch.inference_mode():
                if self.sam2_prompt_mode == "point":
                    batched_masks = []
                    batched_scores = []
                    for abs_box in abs_boxes:
                        point_coords = self._center_point_from_abs_box(abs_box)
                        point_labels = np.array([1], dtype=np.int32)
                        obj_masks, obj_scores, _ = predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False,
                        )
                        batched_masks.append(np.asarray(obj_masks))
                        batched_scores.append(np.asarray(obj_scores) if obj_scores is not None else np.array([1.0], dtype=np.float32))
                    masks = np.stack(batched_masks, axis=0)
                    scores = np.stack(batched_scores, axis=0)
                elif self.sam2_prompt_mode in {"box+point", "box_point"}:
                    batched_masks = []
                    batched_scores = []
                    for abs_box in abs_boxes:
                        point_coords = self._center_point_from_abs_box(abs_box)
                        point_labels = np.array([1], dtype=np.int32)
                        obj_masks, obj_scores, _ = predictor.predict(
                            box=abs_box[None, :],
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False,
                        )
                        batched_masks.append(np.asarray(obj_masks))
                        batched_scores.append(np.asarray(obj_scores) if obj_scores is not None else np.array([1.0], dtype=np.float32))
                    masks = np.stack(batched_masks, axis=0)
                    scores = np.stack(batched_scores, axis=0)
                else:
                    masks, scores, _ = predictor.predict(box=abs_boxes, multimask_output=False)
            if pred_end is not None:
                pred_end.record()
        except Exception as exc:
            logging.warning(f"SAM2 image batched refine failed: {exc}")
            return None

        set_image_ms = None
        predict_ms = None
        if set_start is not None and set_end is not None and pred_start is not None and pred_end is not None:
            torch.cuda.synchronize()
            set_image_ms = float(set_start.elapsed_time(set_end))
            predict_ms = float(pred_start.elapsed_time(pred_end))

        masks = np.asarray(masks)
        scores = np.asarray(scores) if scores is not None else None

        if masks.ndim == 2:
            masks = masks[None, None, ...]
        elif masks.ndim == 3:
            masks = masks[None, ...] if len(abs_boxes) == 1 else masks[:, None, ...]
        if scores is not None:
            if scores.ndim == 0:
                scores = scores.reshape(1, 1)
            elif scores.ndim == 1:
                scores = scores[None, :] if len(abs_boxes) == 1 else scores[:, None]

        refined_boxes = []
        valid_phrases = []
        valid_scores = []
        for idx in range(min(len(boxes_xyxy_prev), masks.shape[0])):
            obj_masks = masks[idx]
            obj_scores = scores[idx] if scores is not None and idx < len(scores) else None
            if obj_masks is None or len(obj_masks) == 0:
                continue

            best_idx = int(np.argmax(obj_scores)) if obj_scores is not None and len(obj_scores) > 0 else 0
            best_mask = obj_masks[best_idx]
            best_score = float(obj_scores[best_idx]) if obj_scores is not None and len(obj_scores) > 0 else 1.0
            refined_box = self._mask_to_box_xyxy_norm(best_mask)
            if refined_box is None:
                continue
            refined_boxes.append(refined_box)
            valid_phrases.append(phrases[idx] if idx < len(phrases) else f"obj_{idx}")
            valid_scores.append(best_score)

        if not refined_boxes:
            return None

        boxes_xyxy = torch.stack(refined_boxes)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        logits = torch.tensor(valid_scores, dtype=torch.float32)
        self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, valid_phrases, None)
        self.last_query_used_dino = False
        self.last_box_source = "sam2_image"
        if set_image_ms is not None and predict_ms is not None:
            self.last_stage_timings_ms = {
                "set_image_ms": set_image_ms,
                "predict_ms": predict_ms,
                "refine_objects": len(valid_phrases),
                "prompt_mode": self.sam2_prompt_mode,
            }
        return self.cached_boxes

    def _accept_dino_result(self, dino_result):
        boxes_cxcywh, boxes_xyxy, logits, phrases, image_source = dino_result
        self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
        self.last_query_used_dino = True
        self.last_box_source = "dino"
        self.last_stage_timings_ms = {}
        return self.cached_boxes

    def _keep_reference_result(self, reference_result, source: str):
        if reference_result is None:
            self.last_query_used_dino = False
            self.last_box_source = source
            return self._empty_box_result()

        self.cached_boxes = reference_result
        self.last_query_used_dino = False
        self.last_box_source = source
        return self.cached_boxes

    def get_dino_boxes(self, image, text_prompt):
        """Interface-compatible getter.

        DINO refreshes periodically; SAM2ImagePredictor uses the previous frame's
        boxes as prompts to refine the current frame in between refreshes.
        """
        prompt_key = self._normalize_prompt(text_prompt)
        prompt_changed = self.cached_prompt != prompt_key
        if prompt_changed:
            self.reset()
            self.cached_prompt = prompt_key

        self.box_query_count += 1
        need_refresh = self.cached_boxes is None or self.box_query_count % self.dino_refresh_interval == 1
        previous_cached = self.cached_boxes

        if not need_refresh:
            refined = self._refine_with_sam2image(image)
            if refined is not None:
                return refined

            dino_result = self._run_dino_detection(image, text_prompt)
            dino_boxes_xyxy = dino_result[1]
            dino_phrases = dino_result[3]
            if previous_cached is not None and len(dino_boxes_xyxy) > 0:
                ref_boxes_xyxy = previous_cached[1]
                ref_phrases = previous_cached[3]
                if len(ref_boxes_xyxy) > 0 and not self._boxes_are_consistent(
                    ref_boxes_xyxy, ref_phrases, dino_boxes_xyxy, dino_phrases
                ):
                    self.last_stage_timings_ms = {}
                    return self._keep_reference_result(previous_cached, "cached_reject_dino")
            return self._accept_dino_result(dino_result)

        # Refresh path:
        # 1. try to refine current frame from the previous state
        # 2. run DINO as a candidate
        # 3. only accept DINO if it agrees with the current tracked reference
        reference_result = None
        reference_source = "none"
        if previous_cached is not None:
            refined = self._refine_with_sam2image(image)
            if refined is not None:
                reference_result = refined
                reference_source = "sam2_image"
            else:
                reference_result = previous_cached
                reference_source = "cached"

        dino_result = self._run_dino_detection(image, text_prompt)
        dino_boxes_xyxy = dino_result[1]
        dino_phrases = dino_result[3]

        if reference_result is None:
            return self._accept_dino_result(dino_result)

        ref_boxes_xyxy = reference_result[1]
        ref_phrases = reference_result[3]
        if len(dino_boxes_xyxy) == 0:
            self.last_stage_timings_ms = {} if reference_source == "cached" else self.last_stage_timings_ms
            return self._keep_reference_result(reference_result, f"{reference_source}_reject_empty_dino")

        if len(ref_boxes_xyxy) > 0 and not self._boxes_are_consistent(
            ref_boxes_xyxy, ref_phrases, dino_boxes_xyxy, dino_phrases
        ):
            self.last_stage_timings_ms = {} if reference_source == "cached" else self.last_stage_timings_ms
            return self._keep_reference_result(reference_result, f"{reference_source}_reject_dino")

        return self._accept_dino_result(dino_result)

    def adaptive_chunk_heuristic(self, image, instruction, robot_state=None, object_state=None):
        parsed_objs = extract_objects(instruction)
        if not parsed_objs:
            return self.min_h, {"reason": "parse_fail"}

        boxes_cxcywh, boxes_xyxy, _, phrases, _ = self.get_dino_boxes(image, parsed_objs)
        boxes = boxes_cxcywh

        info_dict = {
            "target_object": parsed_objs,
            "dino_boxes": boxes_xyxy.tolist() if isinstance(boxes_xyxy, torch.Tensor) else [],
            "phrases": phrases,
            "box_source": "dino" if self.last_query_used_dino else "sam2_image",
        }

        factors = []
        if len(boxes) > 0:
            num_objects = len(boxes)
            info_dict["num_objects"] = num_objects
            if num_objects > 3:
                factors.append("Multiple objects detected")

        self.needs_interruption = False
        if robot_state and "ee_position" in robot_state and len(boxes) > 0:
            ee_pos = np.array(robot_state["ee_position"])
            min_distance = float("inf")
            for box in boxes:
                box_center = np.array([box[0], box[1]])
                distance = np.linalg.norm(ee_pos[:2] - box_center)
                if distance < min_distance:
                    min_distance = distance
            info_dict["min_object_distance"] = min_distance
            if min_distance < 0.1:
                factors.append("Close to object (grasping phase)")

        prev_h = self.current_horizon
        continuity_factor = self.compute_continuity()
        if continuity_factor:
            if continuity_factor.startswith("High"):
                self.current_horizon = min(self.max_h, prev_h + 5)
            else:
                self.set_interruption()
                self.current_horizon = max(self.min_h, self.get_chunk_size(2))

        factors.append(continuity_factor)
        info_dict["factors"] = factors
        return self.current_horizon, info_dict
