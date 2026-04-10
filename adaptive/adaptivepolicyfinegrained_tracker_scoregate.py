import logging

import torch

from .adaptivepolicyfinegrained import AdaptiveActionChunker as BaseAdaptiveActionChunker
from .adaptivepolicyfinegrained_tracker import AdaptiveActionChunker as TrackerAdaptiveActionChunker
from .tracker_utils import ensure_frame, normalize_prompt, single_box_center_distance, single_box_iou, xyxy_to_cxcywh

logger = logging.getLogger("adaptive.tracker_scoregate")


class AdaptiveActionChunker(TrackerAdaptiveActionChunker):
    """
    Experimental score-gated tracker policy.

    Differences from the default tracker policy:
    - DINO is treated as a candidate during refresh and is not committed first.
    - When tracker and DINO disagree, prefer DINO only if its score is higher
      than the tracked score by `dino_score_margin`.
    """

    def __init__(
        self,
        *args,
        dino_score_margin: float = 0.1,
        tracker_jump_iou: float = 0.4,
        tracker_jump_center_dist: float = 0.1,
        untrusted_horizon: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dino_score_margin = float(dino_score_margin)
        self.tracker_jump_iou = float(tracker_jump_iou)
        self.tracker_jump_center_dist = float(tracker_jump_center_dist)
        self.untrusted_horizon = int(untrusted_horizon)
        self.prev_tracked_boxes_by_phrase = {}
        self.tracker_untrusted_by_phrase = {}

    def get_runtime_config(self):
        config = super().get_runtime_config()
        config.update(
            {
                "policy_type": "finegrained_tracker_scoregate",
                "dino_score_margin": self.dino_score_margin,
                "tracker_jump_iou": self.tracker_jump_iou,
                "tracker_jump_center_dist": self.tracker_jump_center_dist,
                "untrusted_horizon": self.untrusted_horizon,
            }
        )
        return config

    def _result_has_untrusted_phrase(self, result) -> bool:
        phrases = result[3] if len(result) > 3 else []
        return any(self.tracker_untrusted_by_phrase.get(phrase, False) for phrase in phrases)

    def _normalize_chunk_proposal(self, proposal):
        if proposal is None:
            return {}
        if isinstance(proposal, dict):
            return proposal
        if isinstance(proposal, tuple) and len(proposal) == 2:
            horizon, info = proposal
            return {
                "current_horizon": horizon,
                "needs_interruption": False,
                "info": dict(info or {}),
            }
        raise TypeError(f"Unsupported chunk proposal type: {type(proposal)!r}")

    def _score_value(self, logits) -> float:
        if logits is None:
            return float("-inf")
        try:
            if hasattr(logits, "detach"):
                logits = logits.detach()
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
            if hasattr(logits, "numpy"):
                logits = logits.numpy()
            if hasattr(logits, "tolist"):
                logits = logits.tolist()
            if isinstance(logits, (list, tuple)):
                values = []
                for item in logits:
                    try:
                        if hasattr(item, "item"):
                            item = item.item()
                        values.append(float(item))
                    except Exception:
                        continue
                return max(values) if values else float("-inf")
            if hasattr(logits, "item"):
                logits = logits.item()
            return float(logits)
        except Exception:
            return float("-inf")

    def _score_map(self, phrases, logits):
        score_by_phrase = {}
        if phrases is None or logits is None:
            return score_by_phrase
        for idx, phrase in enumerate(phrases):
            try:
                value = logits[idx]
            except Exception:
                continue
            try:
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "item"):
                    value = value.item()
                score_by_phrase[phrase] = float(value)
            except Exception:
                continue
        return score_by_phrase

    def _has_boxes(self, result) -> bool:
        try:
            boxes_xyxy = result[1]
            return boxes_xyxy is not None and len(boxes_xyxy) > 0
        except Exception:
            return False

    def _commit_result(self, image, result, prompt_key, used_dino: bool):
        boxes_cxcywh, boxes_xyxy, logits, phrases, _ = result
        cached_image = ensure_frame(image)
        self._set_cached_boxes(
            boxes_cxcywh,
            boxes_xyxy,
            logits,
            phrases,
            cached_image,
        )
        # DINO refresh starts a new trust window. Any "untrusted" latch only
        # applies within the previous refresh interval.
        if used_dino:
            self.tracker_untrusted_by_phrase = {}
            self.prev_tracked_boxes_by_phrase = {}
        self._initialize_trackers(image, boxes_xyxy, phrases, prompt_key)
        self.last_query_used_dino = used_dino
        return self.cached_boxes

    def _raw_dino_candidate(self, image, text_prompt):
        return BaseAdaptiveActionChunker.get_dino_boxes(self, image, text_prompt)

    def _phrase_maps(self, result):
        boxes_xyxy = result[1]
        logits = result[2]
        phrases = result[3]
        boxes_by_phrase = {}
        scores_by_phrase = self._score_map(phrases, logits)
        for idx, phrase in enumerate(phrases):
            if idx < len(boxes_xyxy):
                boxes_by_phrase[phrase] = boxes_xyxy[idx]
        return boxes_by_phrase, scores_by_phrase

    def _update_tracker_drift_state(self, tracked):
        tracked_boxes_by_phrase, _ = self._phrase_maps(tracked)
        drift_state = {}
        for phrase, tracked_box in tracked_boxes_by_phrase.items():
            prev_box = self.prev_tracked_boxes_by_phrase.get(phrase)
            if prev_box is None:
                drift_state[phrase] = False
                continue
            iou = single_box_iou(prev_box, tracked_box)
            center_dist = single_box_center_distance(prev_box, tracked_box)
            drifted = (iou < self.tracker_jump_iou) or (center_dist > self.tracker_jump_center_dist)
            if drifted:
                self.tracker_untrusted_by_phrase[phrase] = True #只要有一次跳动就标记为untrusted
        self.prev_tracked_boxes_by_phrase = tracked_boxes_by_phrase

    def _merge_phrase_results(self, tracked, dino):
        tracked_boxes_by_phrase, tracked_scores_by_phrase = self._phrase_maps(tracked)
        dino_boxes_by_phrase, dino_scores_by_phrase = self._phrase_maps(dino)
        consistent = self._boxes_are_consistent(tracked[1], tracked[3], dino[1], dino[3])
        self._update_tracker_drift_state(tracked)

        tracked_phrase_set = set(tracked_boxes_by_phrase)
        dino_phrase_set = set(dino_boxes_by_phrase)
        if len(dino_phrase_set) > len(tracked_phrase_set):
            logger.debug(
                "Rejecting DINO merge because it introduced extra phrases: tracked=%s dino=%s",
                sorted(tracked_phrase_set),
                sorted(dino_phrase_set),
            )
            return tracked, False

        merged_phrases = []
        merged_boxes = []
        merged_scores = []
        used_dino_any = False

        phrase_order = []
        for phrase in tracked[3]:
            if phrase not in phrase_order:
                phrase_order.append(phrase)
        for phrase in dino[3]:
            if phrase not in phrase_order:
                phrase_order.append(phrase)

        for phrase in phrase_order:
            has_tracked = phrase in tracked_boxes_by_phrase
            has_dino = phrase in dino_boxes_by_phrase

            if has_tracked and has_dino:
                old_score = tracked_scores_by_phrase.get(phrase, float("-inf"))
                new_score = dino_scores_by_phrase.get(phrase, float("-inf"))
                tracked_box = tracked_boxes_by_phrase[phrase]
                dino_box = dino_boxes_by_phrase[phrase]
                iou = single_box_iou(tracked_box, dino_box)
                center_dist = single_box_center_distance(tracked_box, dino_box)
                consistent_phrase = not (iou < self.dino_accept_iou and center_dist > self.dino_accept_center_dist)
                merged_score = max(old_score, new_score)
                tracker_untrusted = self.tracker_untrusted_by_phrase.get(phrase, False)
                
                if tracker_untrusted:
                    merged_phrases.append(phrase)
                    merged_boxes.append(dino_box)
                    merged_scores.append(new_score)
                    used_dino_any = True
                    continue

                if consistent_phrase:
                    merged_phrases.append(phrase)
                    merged_boxes.append(dino_box)
                    merged_scores.append(merged_score)
                    logger.debug(
                        "Phrase gate phrase='%s': consistent use DINO box, merged_score=max(%.4f, %.4f)=%.4f",
                        phrase,
                        old_score,
                        new_score,
                        merged_score,
                    )
                    if new_score > old_score:
                        used_dino_any = True
                    continue

                accept_dino = new_score > old_score + self.dino_score_margin
                if accept_dino:
                    merged_boxes.append(dino_box)
                    merged_scores.append(new_score)
                    used_dino_any = True
                else:
                    merged_boxes.append(tracked_box)
                    merged_scores.append(old_score)
                merged_phrases.append(phrase)
                continue

            if has_tracked:
                merged_phrases.append(phrase)
                merged_boxes.append(tracked_boxes_by_phrase[phrase])
                merged_scores.append(tracked_scores_by_phrase.get(phrase, 1.0))
                logger.debug("Phrase gate phrase='%s': tracker-only keep tracker", phrase)
                continue

            if has_dino:
                merged_phrases.append(phrase)
                merged_boxes.append(dino_boxes_by_phrase[phrase])
                merged_scores.append(dino_scores_by_phrase.get(phrase, 0.0))
                used_dino_any = True
                logger.debug("Phrase gate phrase='%s': dino-only accept dino", phrase)

        if not merged_boxes:
            return tracked, False

        boxes_xyxy = torch.stack(
            [box.detach().cpu() if hasattr(box, "detach") else torch.as_tensor(box, dtype=torch.float32) for box in merged_boxes]
        )
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        logits = torch.tensor(merged_scores, dtype=torch.float32)
        merged_result = (
            boxes_cxcywh,
            boxes_xyxy,
            logits,
            merged_phrases,
            ensure_frame(tracked[4] if len(tracked) > 4 else None),
        )
        return merged_result, used_dino_any

    def get_dino_boxes(self, image, text_prompt):
        prompt_key = normalize_prompt(text_prompt)
        self.box_query_count += 1

        need_refresh = self._should_refresh_dino(prompt_key)

        tracked = None
        if self.cached_boxes is not None and self.tracker_prompt == prompt_key and self.trackers:
            try:
                tracked = self.get_tracked_boxes(image, text_prompt)
                if self._has_boxes(tracked) and self._result_has_untrusted_phrase(tracked):
                    logger.debug("Skipping tracker-only return because at least one phrase is marked untrusted")
                    need_refresh = True
                if self._has_boxes(tracked) and not need_refresh:
                    self.last_query_used_dino = False
                    return tracked
                if not self._has_boxes(tracked):
                    tracked = None
            except Exception:
                tracked = None

        dino = self._raw_dino_candidate(image, text_prompt)

        if not self._has_boxes(dino):
            if tracked is not None and self._has_boxes(tracked):
                self.last_query_used_dino = False
                return tracked
            if self.cached_boxes is not None:
                self.last_query_used_dino = False
                return self.cached_boxes
            self.last_query_used_dino = True
            return dino

        if tracked is None or not self._has_boxes(tracked):
            self.prev_tracked_boxes_by_phrase = {}
            self.tracker_untrusted_by_phrase = {}
            return self._commit_result(image, dino, prompt_key, used_dino=True)

        merged_result, used_dino_any = self._merge_phrase_results(tracked, dino)
        return self._commit_result(image, merged_result, prompt_key, used_dino=used_dino_any)

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
        proposal = self._normalize_chunk_proposal(proposal)
        info_dict = dict(proposal.get("info") or {})
        phrases = info_dict.get("phrases") or []
        untrusted_phrases = [
            phrase for phrase in phrases
            if self.tracker_untrusted_by_phrase.get(phrase, False)
        ]
        info_dict["tracker_untrusted_phrases"] = untrusted_phrases
        proposal["info"] = info_dict

        if not untrusted_phrases:
            return proposal

        previous_horizon = proposal.get("current_horizon", self.current_horizon)
        fallback_horizon = max(self.min_h, min(self.max_h, self.untrusted_horizon))
        proposal["current_horizon"] = fallback_horizon
        proposal["needs_interruption"] = proposal.get("needs_interruption", False) or (fallback_horizon < previous_horizon)
        info_dict.setdefault("factors", []).append(
            f"Untrusted tracker boxes ({','.join(untrusted_phrases)}): fallback->{fallback_horizon}"
        )
        logger.info(
            "Forcing fallback horizon due to untrusted boxes: phrases=%s previous_horizon=%s fallback_horizon=%s",
            untrusted_phrases,
            previous_horizon,
            fallback_horizon,
        )
        return proposal
