"""Detect people, measure their depth with the chosen model, and place them on the map."""
import time

import cv2

from person_geolocator.geometry import build_capture, place_people
from person_geolocator.metadata import read_pose
from run_webcam_depth import sample_torso

TOLERANCE = 1e-9


def capture_for(image_path, pose, width, height):
    """Use the exact Skydio capture (with lens distortion) while the pose is unedited, else a pinhole build."""
    meta = read_pose(image_path)
    unchanged = meta["capture"] is not None and all(
        (pose.get(k) is None) == (v is None) and (v is None or abs(pose[k] - v) < TOLERANCE)
        for k, v in meta["fields"].items())
    return meta["capture"] if unchanged else build_capture(pose, width, height)


def annotate(frame, people, max_width=1600):
    image = frame.copy()
    scale = max(1, image.shape[1] // 1000)
    for p in people:
        x1, y1, x2, y2 = (round(v) for v in p["bbox"])
        label = f"#{p['index']} " + ("no depth" if p["depth_m"] is None else f"{p['depth_m']:.1f} m")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2 * scale)
        cv2.putText(image, label, (x1, max(30 * scale, y1 - 8 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * scale, (0, 255, 0), 2 * scale, cv2.LINE_AA)
    if image.shape[1] > max_width:
        image = cv2.resize(image, (max_width, round(image.shape[0] * max_width / image.shape[1])))
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def locate(image_path, model, pose, models, max_depth_m=100.):
    started = time.perf_counter()
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if frame is None:
        raise ValueError("Cannot decode the image")
    height, width = frame.shape[:2]
    detections = models.detect(frame)
    depth = models.depth(model, frame) if detections else None
    people = []
    for index, (bbox, confidence) in enumerate(detections, 1):
        sample = sample_torso(depth, bbox)
        roi = sample["roi_xyxy"]
        people.append({"index": index, "bbox": bbox, "confidence": confidence,
                       "depth_m": sample["depth_m"], "valid_fraction": sample["valid_fraction"],
                       "sample_pixel": None if roi is None else [(roi[0] + roi[2]) / 2 - .5, (roi[1] + roi[3]) / 2 - .5]})
    capture = capture_for(image_path, pose, width, height)
    return {"model": model, "image_size": [width, height], "geometry_status": capture["geometry_status"],
            "geometry_errors": capture.get("geometry_errors", []),
            "people": place_people(people, capture, (height, width), max_depth_m, depth_map=depth),
            "annotated_jpeg": annotate(frame, people), "seconds": time.perf_counter() - started}
