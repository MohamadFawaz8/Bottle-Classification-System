"""
pipeline.py - Local desktop adaptation of the Streamlit bottle pipeline.

The video and webcam algorithms match the provided reference workflow:
- run all three YOLO models on the frame
- group detections by proximity so one physical bottle is represented once
- track bottles frame-to-frame through the counter classes
- count bottles only when they cross the vertical counting line
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO

from counter import ImageCounter, VideoCounter, WebcamCounter


_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_DIR, ".yolo_config"), exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_DIR, ".yolo_config"))


def _resolve_model_path(*candidates):
    for candidate in candidates:
        path = os.path.join(_DIR, candidate)
        if os.path.exists(path):
            return path
    return os.path.join(_DIR, candidates[0])


DEFORMATION_MODEL = _resolve_model_path("models/deformation.pt", "models/deformity.pt", "deformity.pt")
FULLNESS_MODEL = _resolve_model_path("models/fullness.pt", "fullness.pt")
SIZE_MODEL = _resolve_model_path("models/size.pt", "models/capacity.pt", "capacity.pt")

CONF_THRESHOLD = 0.50
PROXIMITY_THRESHOLD = 100
CROP_CONF_THRESHOLD = 0.10
CROP_MARGIN_RATIO = 0.08

print("[pipeline] Loading models ...")
_deformation_model = YOLO(DEFORMATION_MODEL)
_fullness_model = YOLO(FULLNESS_MODEL)
_size_model = YOLO(SIZE_MODEL)
print("[pipeline] Models ready.")

# Thread pool shared across all run_pipeline calls.
# 3 workers = one per model for parallel full-frame inference.
# max_workers capped at 4 to avoid over-subscribing CPU/GPU.
_INFER_POOL = ThreadPoolExecutor(max_workers=4)


def reset_tracking_state():
    """Compatibility stub for the existing desktop app."""
    return None


def _collect_detections(results, model_name, conf_threshold=CONF_THRESHOLD):
    detections = []
    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        confidence = float(box.conf)
        if confidence < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append(
            {
                "model": model_name,
                "cls_name": results.names[int(box.cls)],
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": confidence,
            }
        )
    return detections


def _expand_box(box, frame_width, frame_height, margin_ratio=CROP_MARGIN_RATIO):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mx = int(width * margin_ratio)
    my = int(height * margin_ratio)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(frame_width, x2 + mx),
        min(frame_height, y2 + my),
    )


def _predict_on_crop(model, frame, box, conf_thresh=CROP_CONF_THRESHOLD):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = _expand_box(box, frame_width, frame_height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0

    results = model(crop, conf=conf_thresh, verbose=False)
    detections = _collect_detections(results[0], "crop", conf_thresh) if results else []
    if not detections:
        return "", 0.0

    best = max(detections, key=lambda item: item["confidence"])
    return best["cls_name"], float(best["confidence"])


def _group_detections(detections):
    detections = sorted(detections, key=lambda item: item["center"][0])
    groups = []

    for det in detections:
        matched = None
        for group in groups:
            gx = sum(item["center"][0] for item in group) / len(group)
            gy = sum(item["center"][1] for item in group) / len(group)
            if (
                abs(det["center"][0] - gx) < PROXIMITY_THRESHOLD
                and abs(det["center"][1] - gy) < PROXIMITY_THRESHOLD
            ):
                matched = group
                break
        if matched is not None:
            matched.append(det)
        else:
            groups.append([det])

    grouped_bottles = []
    for group in groups:
        bottle_info = {"size": "", "full": "", "defect": ""}
        xs, ys, confidences = [], [], []

        for detection in group:
            bottle_info[detection["model"]] = detection["cls_name"]
            xs.extend([detection["box"][0], detection["box"][2]])
            ys.extend([detection["box"][1], detection["box"][3]])
            confidences.append(detection["confidence"])

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        avg_conf = sum(confidences) / max(1, len(confidences))

        grouped_bottles.append(
            {
                "bbox": (min_x, min_y, max_x, max_y),
                "center": (cx, cy),
                "confidence": avg_conf,
                "bottle_info": bottle_info,
            }
        )

    return grouped_bottles


def _refresh_single_bottle(frame, bottle):
    """Run crop inference for one bottle. Called concurrently by the pool."""
    bbox = bottle["bbox"]
    bottle_info = dict(bottle["bottle_info"])
    size_label, size_conf = _predict_on_crop(_size_model, frame, bbox)
    fullness_label, fullness_conf = _predict_on_crop(_fullness_model, frame, bbox)
    if size_label:
        bottle_info["size"] = size_label
    if fullness_label:
        bottle_info["full"] = fullness_label
    return {
        "bbox": bottle["bbox"],
        "center": bottle["center"],
        "confidence": max(
            float(bottle.get("confidence", 0.0)),
            float(size_conf),
            float(fullness_conf),
        ),
        "bottle_info": bottle_info,
    }


def _refresh_group_labels_from_crops(frame, grouped_bottles):
    """Run per-bottle crop inference in parallel across all detected bottles."""
    if not grouped_bottles:
        return []
    if len(grouped_bottles) == 1:
        # Skip pool overhead for the single-bottle case
        return [_refresh_single_bottle(frame, grouped_bottles[0])]
    futures = {
        _INFER_POOL.submit(_refresh_single_bottle, frame, bottle): i
        for i, bottle in enumerate(grouped_bottles)
    }
    results = [None] * len(grouped_bottles)
    for future in as_completed(futures):
        idx = futures[future]
        results[idx] = future.result()
    return results


def _draw_bottle_annotation(frame, bottle_id, bbox, bottle_info, center):
    min_x, min_y, max_x, max_y = bbox

    if "deform" in str(bottle_info.get("defect", "")).lower():
        box_color = (0, 0, 255)
    elif "normal" in str(bottle_info.get("defect", "")).lower():
        box_color = (0, 255, 0)
    else:
        box_color = (255, 255, 0)

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)

    lines = [
        f"ID: {bottle_id}",
        f"Condition : {bottle_info.get('defect') or '—'}",
        f"Capacity  : {bottle_info.get('size') or '—'}",
        f"Fullness  : {bottle_info.get('full') or '—'}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_height = 18
    padding = 4

    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    block_height = line_height * len(lines) + padding * 2
    block_top = min_y - block_height - 2
    if block_top < 0:
        block_top = max_y + 2

    cv2.rectangle(
        frame,
        (min_x - 2, block_top - 2),
        (min_x + max_width + padding * 2 + 2, block_top + block_height + 2),
        box_color,
        1,
    )
    cv2.rectangle(
        frame,
        (min_x, block_top),
        (min_x + max_width + padding * 2, block_top + block_height),
        (30, 30, 30),
        -1,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (min_x + padding, block_top + padding + (index + 1) * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _draw_counting_info(frame, counter):
    if counter is None or counter.counting_line_x is None:
        return

    height = frame.shape[0]
    cv2.line(
        frame,
        (counter.counting_line_x, 0),
        (counter.counting_line_x, height),
        (0, 255, 0),
        3,
    )

    count_text = f"COUNT: {counter.get_count()}"
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), 2)
    cv2.putText(frame, count_text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    frame_text = f"Frame: {counter.frame_count}"
    cv2.putText(
        frame,
        frame_text,
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def _build_result_text(detections, counter=None, use_webcam=False):
    if not detections:
        lines = ["No bottles detected."]
    else:
        lines = []
        for idx, det in enumerate(detections, 1):
            bottle_number = det.get("display_id", idx)
            lines.append(
                f"Bottle {bottle_number}:  "
                f"Condition={det.get('deformity') or '—'} ({float(det.get('conf', det.get('confidence', 0.0))):.2f})  |  "
                f"Capacity={det.get('capacity') or '—'}  |  "
                f"Fullness={det.get('fullness') or '—'}"
            )

    if use_webcam and counter is not None:
        lines.insert(0, f"Webcam Count: {counter.get_count()}")

    if counter is not None and not isinstance(counter, ImageCounter):
        lines.insert(0, f"Total bottles detected: {counter.get_count()}")

    return "\n".join(lines)


def run_pipeline(frame, counter=None, use_webcam=False):
    """
    Run the three YOLO models on `frame` and return:
        annotated_frame  – BGR image with bounding boxes + labels drawn
        result_text      – human-readable string of model predictions
        detections       – list of grouped bottle detections
    """
    annotated = frame.copy()

    # Run all three full-frame models concurrently
    def _infer_size():
        return _collect_detections(_size_model(frame, verbose=False)[0], "size")
    def _infer_fullness():
        return _collect_detections(_fullness_model(frame, verbose=False)[0], "full")
    def _infer_deformation():
        return _collect_detections(_deformation_model(frame, verbose=False)[0], "defect")

    futures_map = {
        _INFER_POOL.submit(_infer_size):        "size",
        _INFER_POOL.submit(_infer_fullness):    "full",
        _INFER_POOL.submit(_infer_deformation): "defect",
    }
    detections = []
    for fut in as_completed(futures_map):
        detections.extend(fut.result())

    grouped_bottles = _group_detections(detections)
    grouped_bottles = _refresh_group_labels_from_crops(frame, grouped_bottles)

    if isinstance(counter, (VideoCounter, WebcamCounter)):
        visible_detections = counter.update(
            grouped_bottles,
            frame_width=frame.shape[1],
        )
    else:
        visible_detections = []
        for index, bottle in enumerate(grouped_bottles, 1):
            bbox = bottle["bbox"]
            bottle_info = bottle["bottle_info"]
            avg_conf = float(bottle["confidence"])
            visible_detections.append(
                {
                    "display_id": index,
                    "track_id": f"Bottle{index}",
                    "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "center": bottle["center"],
                    "confidence": avg_conf,
                    "conf": avg_conf,
                    "deformity": bottle_info.get("defect", ""),
                    "capacity": bottle_info.get("size", ""),
                    "fullness": bottle_info.get("full", ""),
                    "size": bottle_info.get("size", ""),
                    "full": bottle_info.get("full", ""),
                    "defect": bottle_info.get("defect", ""),
                    "counted": False,
                }
            )

        if isinstance(counter, ImageCounter):
            counter.update(visible_detections)

    for det in visible_detections:
        _draw_bottle_annotation(
            annotated,
            det.get("track_id") or f"Bottle{det.get('display_id', 0)}",
            tuple(det["box"]),
            {
                "size": det.get("size", det.get("capacity", "")),
                "full": det.get("full", det.get("fullness", "")),
                "defect": det.get("defect", det.get("deformity", "")),
            },
            det.get("center") or (
                (det["box"][0] + det["box"][2]) // 2,
                (det["box"][1] + det["box"][3]) // 2,
            ),
        )

    if isinstance(counter, (VideoCounter, WebcamCounter)):
        _draw_counting_info(annotated, counter)

    result_text = _build_result_text(visible_detections, counter=counter, use_webcam=use_webcam)
    return annotated, result_text, visible_detections
