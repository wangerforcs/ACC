import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from .adaptivepolicyfinegrained import AdaptiveActionChunker as BaseAdaptiveActionChunker
from .object_extractor.mock_extractor import extract_objects


class AdaptiveActionChunker(BaseAdaptiveActionChunker):
    """Online rollout SAM2 variant.

    External interface stays close to the tracker version:
    - `get_dino_boxes(image, text_prompt)` returns boxes for the current frame
    - periodically refreshes with DINO
    - in between refreshes, tracks from the last DINO-initialized frame to the
      current frame using SAM2 over a rolling window of buffered frames
    """

    def __init__(
        self,
        *args,
        sam2_model_id: str = "facebook/sam2.1-hiera-tiny",
        sam2_config: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        sam2_device: Optional[str] = None,
        dino_refresh_interval: int = 10,
        sam2_window_size: int = 24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam2_model_id = sam2_model_id
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        self.sam2_device = sam2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_refresh_interval = max(1, int(dino_refresh_interval))
        self.sam2_window_size = max(2, int(sam2_window_size))

        self.sam2_predictor: Optional[SAM2VideoPredictor] = None
        self.sam2_obj_id_to_phrase: Dict[int, str] = {}
        self.box_query_count = 0
        self.last_query_used_dino = False

        self.cached_boxes = None
        self.cached_prompt: Optional[Tuple[str, ...]] = None
        self.frame_buffer: List[np.ndarray] = []
        self.window_prompt_frame_idx = 0
        self.window_prompt_boxes_xyxy: Optional[torch.Tensor] = None
        self.window_prompt_phrases: List[str] = []

    def reset(self):
        super().reset()
        self.sam2_obj_id_to_phrase = {}
        self.box_query_count = 0
        self.last_query_used_dino = False
        self.cached_boxes = None
        self.cached_prompt = None
        self.frame_buffer = []
        self.window_prompt_frame_idx = 0
        self.window_prompt_boxes_xyxy = None
        self.window_prompt_phrases = []

    def _ensure_sam2_predictor(self) -> SAM2VideoPredictor:
        if self.sam2_predictor is not None:
            return self.sam2_predictor

        if self.sam2_config:
            self.sam2_predictor = build_sam2_video_predictor(
                self.sam2_config,
                ckpt_path=self.sam2_ckpt,
                device=self.sam2_device,
            )
        else:
            self.sam2_predictor = SAM2VideoPredictor.from_pretrained(
                self.sam2_model_id,
                device=self.sam2_device,
            )
        self.sam2_predictor.eval()
        return self.sam2_predictor

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

    def _norm_xyxy_to_abs_xyxy(self, box_xyxy_norm: torch.Tensor, image_shape: Tuple[int, int, int]) -> List[float]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
        x1 = float(np.clip(x1 * w, 0, max(w - 1, 0)))
        y1 = float(np.clip(y1 * h, 0, max(h - 1, 0)))
        x2 = float(np.clip(x2 * w, x1 + 1.0, w))
        y2 = float(np.clip(y2 * h, y1 + 1.0, h))
        return [x1, y1, x2, y2]

    def _mask_to_box_xyxy_norm(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        mask_np = mask.detach().float().cpu().numpy()
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim != 2:
            return None
        ys, xs = np.where(mask_np > 0.0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        h, w = mask_np.shape
        x1 = float(xs.min()) / max(w, 1)
        y1 = float(ys.min()) / max(h, 1)
        x2 = float(xs.max() + 1) / max(w, 1)
        y2 = float(ys.max() + 1) / max(h, 1)
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

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

    def _append_frame(self, image):
        frame = image.copy() if isinstance(image, np.ndarray) else cv2.imread(image)
        if frame is None:
            return
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.sam2_window_size:
            overflow = len(self.frame_buffer) - self.sam2_window_size
            self.frame_buffer = self.frame_buffer[overflow:]
            self.window_prompt_frame_idx = max(0, self.window_prompt_frame_idx - overflow)

    def _advance_prompt_to_latest(self, boxes_xyxy: torch.Tensor, phrases: List[str]):
        """Roll the SAM2 prompt forward to the newest buffered frame.

        Without this, once the original prompt frame slides out of the rolling
        window, we may accidentally reuse stale boxes on a different frame.
        """
        self.window_prompt_frame_idx = max(0, len(self.frame_buffer) - 1)
        self.window_prompt_boxes_xyxy = boxes_xyxy.detach().cpu() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
        self.window_prompt_phrases = list(phrases)
        self.sam2_obj_id_to_phrase = {
            idx: phrase for idx, phrase in enumerate(self.window_prompt_phrases, start=1)
        }

    def _dump_window_as_jpg_dir(self) -> str:
        temp_dir = tempfile.mkdtemp(prefix="sam2_rollout_", dir="/tmp")
        for idx, frame in enumerate(self.frame_buffer):
            cv2.imwrite(str(Path(temp_dir) / f"{idx:05d}.jpg"), frame)
        return temp_dir

    def _boxes_from_masks(self, video_res_masks: torch.Tensor, obj_ids: Iterable[int]):
        boxes_xyxy = []
        phrases = []
        mask_areas = []
        for idx, obj_id in enumerate(obj_ids):
            box = self._mask_to_box_xyxy_norm(video_res_masks[idx])
            if box is None:
                continue
            boxes_xyxy.append(box)
            phrases.append(self.sam2_obj_id_to_phrase.get(int(obj_id), str(obj_id)))
            mask_areas.append(float((video_res_masks[idx].detach().float().cpu().numpy() > 0.0).sum()))

        if not boxes_xyxy:
            return torch.tensor([]), torch.tensor([]), [], []

        boxes_xyxy = torch.stack(boxes_xyxy)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        return boxes_cxcywh, boxes_xyxy, phrases, mask_areas

    def _refresh_with_dino(self, image, text_prompt):
        boxes_cxcywh, boxes_xyxy, logits, phrases, image_source = super().get_dino_boxes(image, text_prompt)
        self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, phrases, image_source)
        self._advance_prompt_to_latest(boxes_xyxy, phrases)
        self.last_query_used_dino = True
        return self.cached_boxes

    def _track_with_sam2(self):
        if len(self.frame_buffer) < 2 or self.window_prompt_boxes_xyxy is None or len(self.window_prompt_boxes_xyxy) == 0:
            return self.cached_boxes

        predictor = self._ensure_sam2_predictor()
        temp_dir = self._dump_window_as_jpg_dir()
        try:
            inference_state = predictor.init_state(temp_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
            for idx, box_xyxy in enumerate(self.window_prompt_boxes_xyxy, start=1):
                abs_box = self._norm_xyxy_to_abs_xyxy(box_xyxy, self.frame_buffer[self.window_prompt_frame_idx].shape)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.window_prompt_frame_idx,
                    obj_id=idx,
                    box=abs_box,
                )

            latest = None
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=self.window_prompt_frame_idx,
                max_frame_num_to_track=len(self.frame_buffer) - self.window_prompt_frame_idx,
            ):
                latest = (frame_idx, obj_ids, video_res_masks)

            if latest is None:
                return self.cached_boxes

            frame_idx, obj_ids, video_res_masks = latest
            boxes_cxcywh, boxes_xyxy, phrases, _ = self._boxes_from_masks(video_res_masks, obj_ids)
            if len(boxes_xyxy) == 0:
                return self.cached_boxes
            logits = torch.ones(len(phrases), dtype=torch.float32)
            self._set_cached_boxes(boxes_cxcywh, boxes_xyxy, logits, phrases, self.frame_buffer[-1])
            # Use the newest tracked boxes as the next prompt anchor so the
            # rolling window always starts from a frame whose boxes match it.
            self._advance_prompt_to_latest(boxes_xyxy, phrases)
            self.last_query_used_dino = False
            return self.cached_boxes
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_dino_boxes(self, image, text_prompt):
        """Interface-compatible online getter.

        DINO refreshes every `dino_refresh_interval` queries (or when prompt changes).
        Between refreshes, SAM2 tracks from the last DINO prompt frame to the current frame
        over the rolling frame window.
        """
        prompt_key = self._normalize_prompt(text_prompt)
        prompt_changed = self.cached_prompt != prompt_key
        if prompt_changed:
            self.reset()
            self.cached_prompt = prompt_key

        self._append_frame(image)
        self.box_query_count += 1

        need_refresh = (
            self.cached_boxes is None
            or self.window_prompt_boxes_xyxy is None
            or self.box_query_count % self.dino_refresh_interval == 1
        )
        if need_refresh:
            return self._refresh_with_dino(image, text_prompt)

        tracked = self._track_with_sam2()
        if tracked is None or (isinstance(tracked[0], torch.Tensor) and len(tracked[0]) == 0):
            return self._refresh_with_dino(image, text_prompt)
        return tracked

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
            "box_source": "dino" if self.last_query_used_dino else "sam2",
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
