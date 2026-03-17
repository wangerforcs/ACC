import logging

import numpy as np
import torch

from .adaptivepolicyfinegrained_tracker_scoregate import AdaptiveActionChunker as TrackerAdaptiveActionChunker
from .object_extractor.mock_extractor import extract_objects

logger = logging.getLogger("adaptive.tracker_heavy")


class AdaptiveActionChunker(TrackerAdaptiveActionChunker):
    """Heavier tracker variant that prefers larger chunks.

    This class intentionally reuses the tracker-backed implementation and only
    overrides the chunk-sizing logic so tracker/DINO behavior stays in one place.
    """

    def ValueFunc(self, image, instruction, vla_attention_map, use_uncertainty=True, use_coverage=True):
        """
        Heavy version:
        - 默认更倾向大 chunk
        - 仅在连续性和 attention 指标同时明显恶化时才显著降档
        """
        if isinstance(vla_attention_map, list) and len(vla_attention_map) > 0:
            if all(isinstance(layer, list) for layer in vla_attention_map) and all(
                isinstance(item, torch.Tensor) for layer in vla_attention_map for item in layer
            ):
                vla_attention_map = torch.stack([torch.stack(layer) for layer in vla_attention_map])
            elif all(isinstance(item, torch.Tensor) for item in vla_attention_map):
                vla_attention_map = torch.stack(vla_attention_map)

        self.needs_interruption = False

        parsed_objs = extract_objects(instruction)
        if not parsed_objs:
            return self.min_h, {"reason": "parse_fail"}

        _, boxes_xyxy, _, phrases, _ = self.get_dino_boxes(image, parsed_objs)
        boxes = boxes_xyxy
        if len(boxes) > 0:
            boxes = torch.clamp(boxes, 0.0, 1.0)

        info_dict = {
            "target_object": parsed_objs,
            "dino_boxes": boxes.tolist() if isinstance(boxes, torch.Tensor) else [],
            "phrases": phrases,
            "box_source": "dino" if self.last_query_used_dino else "tracker",
        }

        kl_scores = []
        overlaps = []
        peak_scores = []
        uncertainty_scores = []
        peak_high_count = 0
        peak_low_count = 0
        peak_thresh_high = 0.8
        peak_thresh_low = 0.4

        continuity_factor = self.compute_continuity()

        for layer_idx in range(vla_attention_map.shape[0]):
            layer_attn = vla_attention_map[layer_idx]
            metrics = self.uncertainty_engine.compute_uncertainty(layer_attn, boxes)
            logger.info(f"layer_idx: {layer_idx}, metrics: {metrics}")
            kl_scores.append(metrics["kl"])
            overlaps.append(metrics.get("coverage", 0.0))
            peak_scores.append(metrics["peak_ratio"])
            uncertainty_scores.append(metrics["uncertainty"])
            if peak_scores[-1] > peak_thresh_high:
                peak_high_count += 1
            elif peak_scores[-1] < peak_thresh_low:
                peak_low_count += 1

        num_layers = len(peak_scores)
        peak_high_ratio = peak_high_count / max(num_layers, 1)
        peak_low_ratio = peak_low_count / max(num_layers, 1)
        avg_coverage = float(np.mean(overlaps)) if overlaps else 0.0
        best_kl = float(np.min(kl_scores)) if kl_scores else 10.0
        avg_uncertainty = float(np.mean(uncertainty_scores)) if uncertainty_scores else -1.0

        is_high_cont = bool(continuity_factor and continuity_factor.startswith("High"))
        is_low_cont = bool(continuity_factor and continuity_factor.startswith("Low"))

        # Default to chunk 25, then only downgrade when evidence is strong.
        confidence = 4

        # severe_risk = (
        #     is_low_cont
        #     and (peak_low_ratio >= 0.75 or avg_coverage < 0.04 or best_kl > 0.025)
        # )
        # medium_risk = (
        #     (is_low_cont and (peak_low_ratio >= 0.5 or avg_coverage < 0.08 or best_kl > 0.018))
        #     or (peak_low_ratio >= 0.75 and avg_coverage < 0.08)
        # )
        # mild_risk = peak_low_ratio >= 0.5 or avg_coverage < 0.10 or best_kl > 0.016

        # highly_confident = (
        #     peak_high_ratio >= 0.75
        #     and avg_coverage >= 0.22
        #     and best_kl <= 0.008
        #     and is_high_cont
        # )
        # very_stable = (
        #     peak_high_ratio >= 0.5
        #     and avg_coverage >= 0.16
        #     and best_kl <= 0.012
        #     and (is_high_cont or not is_low_cont)
        # )
        severe_risk = (
            is_low_cont
            and (peak_low_ratio >= 0.75 or avg_coverage < 0.04)
        )
        medium_risk = (
            (is_low_cont and (peak_low_ratio >= 0.5 or avg_coverage < 0.08))
            or (peak_low_ratio >= 0.75 and avg_coverage < 0.08)
        )
        mild_risk = peak_low_ratio >= 0.5 or avg_coverage < 0.10

        highly_confident = (
            peak_high_ratio >= 0.75
            and avg_coverage >= 0.22
            and is_high_cont
        )
        very_stable = (
            peak_high_ratio >= 0.5
            and avg_coverage >= 0.16
            and (is_high_cont or not is_low_cont)
        )

        if severe_risk:
            confidence = 0
            # self.needs_interruption = True
        elif medium_risk:
            confidence = 1
        elif mild_risk and is_low_cont:
            confidence = 2
        elif highly_confident or very_stable:
            confidence = 5
        elif is_high_cont or peak_high_ratio >= 0.25 or avg_coverage >= 0.12:
            confidence = 4
        else:
            confidence = 3

        info_dict.update(
            {
                "confidence": confidence,
                "continuity_factor": continuity_factor,
                "peak_high_ratio": peak_high_ratio,
                "peak_low_ratio": peak_low_ratio,
                "avg_coverage": avg_coverage,
                "best_kl": best_kl,
                "avg_uncertainty": avg_uncertainty,
            }
        )

        logger.info(
            "heavy confidence: %s, continuity_factor: %s, peak_high_ratio: %.3f, "
            "peak_low_ratio: %.3f, avg_coverage: %.3f, best_kl: %.4f",
            confidence,
            continuity_factor,
            peak_high_ratio,
            peak_low_ratio,
            avg_coverage,
            best_kl,
        )

        self.current_horizon = max(self.min_h, self.get_chunk_size(confidence))
        return self.current_horizon, info_dict

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
            "box_source": "dino" if self.last_query_used_dino else "tracker",
        }

        factors = []
        if len(boxes) > 0:
            num_objects = len(boxes)
            info_dict["num_objects"] = num_objects
            if num_objects > 3:
                factors.append("Multiple objects detected")
        else:
            info_dict["tracker_state"] = "missing"

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

        self.needs_interruption = False
        prev_h = self.current_horizon
        continuity_factor = self.compute_continuity()
        if continuity_factor:
            if continuity_factor.startswith("High"):
                self.current_horizon = min(self.max_h, prev_h + 5)
            else:
                # Heavy variant: low continuity first falls back to a medium-large chunk.
                # Only let ValueFunc handle the truly risky cases with hard interruption.
                self.current_horizon = max(self.min_h, self.get_chunk_size(2))

        factors.append(continuity_factor)
        info_dict["factors"] = factors
        return self.current_horizon, info_dict
