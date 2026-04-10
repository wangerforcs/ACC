import re
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def normalize_prompt(text_prompt: Sequence[str] | str) -> Tuple[str, ...]:
    if isinstance(text_prompt, str):
        return (text_prompt,)
    return tuple(text_prompt)


def ensure_frame(image) -> Optional[np.ndarray]:
    if isinstance(image, np.ndarray):
        frame = image.copy()
    else:
        frame = cv2.imread(image)

    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def get_tracker_factory(module, class_name: str):
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


def norm_xyxy_to_bbox(box_xyxy_norm: torch.Tensor, image_shape: Tuple[int, int, int]):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
    x1 = np.clip(x1 * w, 0, max(w - 1, 0))
    y1 = np.clip(y1 * h, 0, max(h - 1, 0))
    x2 = np.clip(x2 * w, x1 + 1, w)
    y2 = np.clip(y2 * h, y1 + 1, h)
    x = int(round(float(x1)))
    y = int(round(float(y1)))
    bw = int(round(float(max(1.0, x2 - x1))))
    bh = int(round(float(max(1.0, y2 - y1))))
    return (x, y, bw, bh)


def bbox_to_norm_xyxy(bbox, image_shape: Tuple[int, int, int]) -> torch.Tensor:
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    x1 = float(np.clip(x / max(w, 1), 0.0, 1.0))
    y1 = float(np.clip(y / max(h, 1), 0.0, 1.0))
    x2 = float(np.clip((x + bw) / max(w, 1), 0.0, 1.0))
    y2 = float(np.clip((y + bh) / max(h, 1), 0.0, 1.0))
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def xyxy_to_cxcywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
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


def single_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
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


def single_box_center_distance(box1: torch.Tensor, box2: torch.Tensor) -> float:
    c1x = (float(box1[0]) + float(box1[2])) / 2.0
    c1y = (float(box1[1]) + float(box1[3])) / 2.0
    c2x = (float(box2[0]) + float(box2[2])) / 2.0
    c2y = (float(box2[1]) + float(box2[3])) / 2.0
    return float(np.linalg.norm([c1x - c2x, c1y - c2y]))


def normalize_phrase_key(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text).lower().replace("_", " ").strip()
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_box_edge_distance(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def box_center_inside(inner_box: torch.Tensor, outer_box: torch.Tensor) -> bool:
    ix1, iy1, ix2, iy2 = [float(v) for v in inner_box]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer_box]
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def intersection_over_box_area(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    if area_a <= 1e-9:
        return 0.0
    return inter / area_a
