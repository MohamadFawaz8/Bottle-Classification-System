"""
counter.py - Bottle counting utilities built from the local desktop adaptation
of the Streamlit bottle tracking workflow.
"""

from datetime import datetime

import numpy as np


class ImageCounter:
    """Simple counter for image detection - counts grouped bottles."""

    def __init__(self):
        self.count = 0

    def update(self, detections):
        self.count = len(detections) if detections else 0
        return self.count

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0


class _BottleTrackingBase:
    """Shared bottle tracking/counting logic for video and webcam."""

    PROXIMITY_THRESHOLD = 100
    MAX_TRACK_AGE = 40

    def __init__(self, fps=30):
        self.default_fps = fps
        self.reset()

    def reset(self):
        self.tracked_bottles = {}
        self.counted_bottles = {}
        self.attribute_votes = {}
        self.bottle_id_counter = 1
        self.counting_line_x = None
        self.results = []
        self.frame_count = 0
        self.fps = float(self.default_fps or 30)
        self._display_order = {}
        self._next_display_id = 1

    def update(self, grouped_bottles, frame_width=None, fps=None):
        """
        Update tracked/counting state using grouped bottle detections for the
        current frame.

        `grouped_bottles` items must contain:
            bbox, center, bottle_info, confidence
        """
        self.frame_count += 1
        if frame_width and self.counting_line_x is None:
            self.counting_line_x = frame_width // 2
        if fps:
            self.fps = float(fps)

        previous_tracked_bottles = self.tracked_bottles.copy()
        current_frame_bottles = {}
        visible_detections = []

        for bottle in grouped_bottles or []:
            bbox = bottle["bbox"]
            cx, cy = bottle["center"]
            bottle_info = dict(bottle["bottle_info"])
            avg_conf = float(bottle.get("confidence", 0.0))

            matched_id, min_dist = None, float("inf")
            for bottle_id, tracked in previous_tracked_bottles.items():
                prev_cx, prev_cy = tracked["center"]
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist < self.PROXIMITY_THRESHOLD and dist < min_dist:
                    matched_id = bottle_id
                    min_dist = dist

            if matched_id is None:
                matched_id = f"Bottle{self.bottle_id_counter}"
                self.bottle_id_counter += 1

            if matched_id not in self._display_order:
                self._display_order[matched_id] = self._next_display_id
                self._next_display_id += 1

            self._record_attribute_vote(matched_id, bottle_info)

            current_frame_bottles[matched_id] = {
                "center": (cx, cy),
                "attributes": bottle_info,
                "confidence": avg_conf,
                "bbox": bbox,
                "last_seen": self.frame_count,
            }

            self._store_result(matched_id, bottle_info, avg_conf)

            visible_detections.append(
                self._build_visible_detection(
                    matched_id=matched_id,
                    bbox=bbox,
                    bottle_info=bottle_info,
                    confidence=avg_conf,
                    counted=matched_id in self.counted_bottles,
                )
            )

        self._update_bottle_counting(current_frame_bottles, previous_tracked_bottles)

        for bottle_id, tracked in current_frame_bottles.items():
            self.tracked_bottles[bottle_id] = tracked

        self._cleanup_old_bottles()

        return visible_detections

    def get_count(self):
        return len(self.counted_bottles)

    def count(self):
        return self.get_count()

    def finalize(self):
        finalized = []
        ordered = sorted(
            self.counted_bottles.items(),
            key=lambda item: item[1].get("frame", 999999),
        )
        for display_id, (bottle_id, info) in enumerate(ordered, 1):
            bottle_info = self._get_stable_attributes(
                bottle_id, fallback=info.get("attributes", {})
            )
            bbox = list(info.get("bbox") or [0, 0, 0, 0])
            finalized.append(
                {
                    "display_id": display_id,
                    "track_id": bottle_id,
                    "box": bbox,
                    "deformity": bottle_info.get("defect", ""),
                    "deformity_conf": float(info.get("confidence", 0.0)),
                    "capacity": bottle_info.get("size", ""),
                    "capacity_conf": float(info.get("confidence", 0.0)),
                    "fullness": bottle_info.get("full", ""),
                    "fullness_conf": float(info.get("confidence", 0.0)),
                    "size": bottle_info.get("size", ""),
                    "full": bottle_info.get("full", ""),
                    "defect": bottle_info.get("defect", ""),
                    "count_time": float(info.get("count_time", 0.0)),
                    "count_frame": int(info.get("frame", 0)),
                }
            )
        return finalized

    def get_results(self):
        return list(self.results)

    def get_results_df_rows(self):
        return list(self.results)

    def get_bottle_summary_rows(self):
        bottle_type_counts = {}
        for _, info in self.counted_bottles.items():
            bottle_type = (
                f"{info['attributes'].get('size', '')}_"
                f"{info['attributes'].get('full', '')}_"
                f"{info['attributes'].get('defect', '')}"
            )
            bottle_type_counts[bottle_type] = bottle_type_counts.get(bottle_type, 0) + 1

        summary_rows = []
        for bottle_type, count in sorted(bottle_type_counts.items()):
            summary_rows.append({"Bottle Type": bottle_type, "Count": count})
        return summary_rows

    def _build_visible_detection(self, matched_id, bbox, bottle_info, confidence, counted):
        x1, y1, x2, y2 = bbox
        return {
            "track_id": matched_id,
            "display_id": self._display_order.get(matched_id),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "confidence": float(confidence),
            "conf": float(confidence),
            "deformity": bottle_info.get("defect", ""),
            "capacity": bottle_info.get("size", ""),
            "fullness": bottle_info.get("full", ""),
            "size": bottle_info.get("size", ""),
            "full": bottle_info.get("full", ""),
            "defect": bottle_info.get("defect", ""),
            "counted": counted,
        }

    def _update_bottle_counting(self, current_bottles, previous_bottles):
        for bottle_id, current in current_bottles.items():
            if bottle_id not in previous_bottles:
                continue

            prev_cx, prev_cy = previous_bottles[bottle_id]["center"]
            cx, cy = current["center"]

            if (
                bottle_id not in self.counted_bottles
                and self.counting_line_x is not None
                and (
                    (prev_cx < self.counting_line_x <= cx)
                    or (prev_cx > self.counting_line_x >= cx)
                )
            ):
                self.counted_bottles[bottle_id] = {
                    "count_time": self.frame_count / max(self.fps, 1.0),
                    "attributes": self._get_stable_attributes(
                        bottle_id, fallback=current["attributes"]
                    ),
                    "frame": self.frame_count,
                    "bbox": list(current["bbox"]),
                    "confidence": float(current.get("confidence", 0.0)),
                }

    def _cleanup_old_bottles(self):
        bottles_to_remove = []
        for bottle_id, tracked in self.tracked_bottles.items():
            if (
                self.frame_count - tracked["last_seen"] > self.MAX_TRACK_AGE
                and bottle_id not in self.counted_bottles
            ):
                bottles_to_remove.append(bottle_id)

        for bottle_id in bottles_to_remove:
            del self.tracked_bottles[bottle_id]

    def _store_result(self, bottle_id, bottle_info, confidence):
        self.results.append(
            {
                "Bottle ID": bottle_id,
                "Size": bottle_info.get("size", ""),
                "Fullness": bottle_info.get("full", ""),
                "Deformation": bottle_info.get("defect", ""),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Frame": self.frame_count,
                "Counted": bottle_id in self.counted_bottles,
                "Confidence": float(confidence),
            }
        )

    def _record_attribute_vote(self, bottle_id, bottle_info):
        votes = self.attribute_votes.setdefault(
            bottle_id,
            {
                "defect": {},
                "size": {},
                "full": {},
            },
        )
        for key in ("defect", "size", "full"):
            value = (bottle_info.get(key) or "").strip()
            if not value:
                continue
            votes[key][value] = votes[key].get(value, 0) + 1

    def _get_stable_attributes(self, bottle_id, fallback=None):
        fallback = dict(fallback or {})
        votes = self.attribute_votes.get(bottle_id)
        if not votes:
            return fallback

        stable = dict(fallback)
        # Keep deformity robust to brief false "non_deformed" spikes.
        stable["defect"] = self._pick_stable_defect(
            votes.get("defect", {}), fallback.get("defect", "")
        )
        return stable

    def _pick_stable_defect(self, defect_votes, fallback):
        if not defect_votes:
            return fallback

        deformed_votes = {
            label: count
            for label, count in defect_votes.items()
            if not self._is_non_deformed_label(label)
        }
        if deformed_votes:
            return max(deformed_votes.items(), key=lambda item: item[1])[0]

        return max(defect_votes.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _is_non_deformed_label(label):
        normalized = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
        return normalized in {"non_deformed", "not_deformed", "normal", "ok"}


class VideoCounter(_BottleTrackingBase):
    """Video counter using the Streamlit tracking and counting-line algorithm."""

    pass


class WebcamCounter(_BottleTrackingBase):
    """Webcam counter using the same tracking/counting-line algorithm."""

    def __init__(self, frame_width=640, fps=30):
        super().__init__(fps=fps)
        self.counting_line_x = frame_width // 2 if frame_width else None

    def draw_on_frame(self, frame, detections):
        if frame is None:
            return

        if self.counting_line_x is None:
            self.counting_line_x = frame.shape[1] // 2

        height = frame.shape[0]
        # Draw counting line
        import cv2

        cv2.line(
            frame,
            (self.counting_line_x, 0),
            (self.counting_line_x, height),
            (0, 255, 0),
            3,
        )

        count_text = f"COUNT: {len(self.counted_bottles)}"
        (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (255, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), 2)
        cv2.putText(
            frame,
            count_text,
            (15, 15 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(
            frame,
            frame_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


class BottleStore(VideoCounter):
    """
    Compatibility alias so older imports keep working.

    The active video/webcam logic now comes from the same counting-line tracker
    used by the reference Streamlit implementation.
    """

    pass
