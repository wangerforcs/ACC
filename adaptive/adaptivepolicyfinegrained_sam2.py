import logging
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
    """SAM2-backed variant for offline video tracking.

    It keeps the DINO initialization logic from the fine-grained policy, but uses
    SAM2 video propagation to track objects across an mp4 or frame directory.
    """

    def __init__(
        self,
        *args,
        sam2_model_id: str = "facebook/sam2.1-hiera-tiny",
        sam2_config: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        sam2_device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam2_model_id = sam2_model_id
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        self.sam2_device = sam2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_predictor: Optional[SAM2VideoPredictor] = None
        self.sam2_inference_state = None
        self.sam2_obj_id_to_phrase: Dict[int, str] = {}

    def reset(self):
        super().reset()
        self.sam2_inference_state = None
        self.sam2_obj_id_to_phrase = {}

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

    def _read_first_frame(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame from {video_path}")
        return frame

    def _normalize_prompt(self, text_prompt: Sequence[str] | str) -> Tuple[str, ...]:
        if isinstance(text_prompt, str):
            return (text_prompt,)
        return tuple(text_prompt)

    def _norm_xyxy_to_abs_xyxy(self, box_xyxy_norm: torch.Tensor, image_shape: Tuple[int, int, int]) -> List[float]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
        x1 = float(np.clip(x1 * w, 0, max(w - 1, 0)))
        y1 = float(np.clip(y1 * h, 0, max(h - 1, 0)))
        x2 = float(np.clip(x2 * w, x1 + 1.0, w))
        y2 = float(np.clip(y2 * h, y1 + 1.0, h))
        return [x1, y1, x2, y2]

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
            mask_area = float((video_res_masks[idx].detach().float().cpu().numpy() > 0.0).sum())
            mask_areas.append(mask_area)

        if not boxes_xyxy:
            return torch.tensor([]), torch.tensor([]), [], []

        boxes_xyxy = torch.stack(boxes_xyxy)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        return boxes_cxcywh, boxes_xyxy, phrases, mask_areas

    def initialize_sam2_video(self, video_path: str, instruction: str, frame_idx: int = 0):
        predictor = self._ensure_sam2_predictor()
        parsed_objs = extract_objects(instruction)
        if not parsed_objs:
            raise RuntimeError(f"Failed to extract objects from instruction: {instruction}")

        first_frame = self._read_first_frame(video_path)
        _, boxes_xyxy, _, phrases, _ = super().get_dino_boxes(first_frame, parsed_objs)
        if len(boxes_xyxy) == 0:
            raise RuntimeError("Grounding DINO failed to initialize any boxes for SAM2.")

        inference_state = predictor.init_state(str(video_path))
        self.sam2_obj_id_to_phrase = {}

        for idx, (box_xyxy, phrase) in enumerate(zip(boxes_xyxy, phrases), start=1):
            abs_box = self._norm_xyxy_to_abs_xyxy(box_xyxy, first_frame.shape)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=idx,
                box=abs_box,
            )
            self.sam2_obj_id_to_phrase[idx] = phrase

        self.sam2_inference_state = inference_state
        return {
            "parsed_objects": parsed_objs,
            "phrases": phrases,
            "init_boxes_xyxy": boxes_xyxy,
            "frame_shape": first_frame.shape,
        }

    def propagate_video(
        self,
        video_path: str,
        instruction: str,
        max_frames: Optional[int] = None,
        start_frame_idx: int = 0,
    ):
        init_info = self.initialize_sam2_video(video_path, instruction, frame_idx=start_frame_idx)
        predictor = self._ensure_sam2_predictor()
        inference_state = self.sam2_inference_state

        frame_counter = 0
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frames,
        ):
            boxes_cxcywh, boxes_xyxy, phrases, mask_areas = self._boxes_from_masks(video_res_masks, obj_ids)
            logits = torch.ones(len(phrases), dtype=torch.float32) if len(phrases) > 0 else torch.tensor([])
            yield {
                "frame_idx": int(frame_idx),
                "boxes_cxcywh": boxes_cxcywh,
                "boxes_xyxy": boxes_xyxy,
                "phrases": phrases,
                "logits": logits,
                "mask_areas": mask_areas,
                "init_info": init_info if frame_counter == 0 else None,
            }
            frame_counter += 1

    def get_dino_boxes(self, image, text_prompt):
        # Preserve the original interface on still images.
        return super().get_dino_boxes(image, text_prompt)
