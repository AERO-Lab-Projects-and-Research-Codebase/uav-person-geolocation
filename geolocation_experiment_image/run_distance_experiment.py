"""Run every depth model on the tape-measured experiment images; compare with the distance in the file name.

Usage: python run_distance_experiment.py detect   -> detections.json + crops (pick the right person)
       python run_distance_experiment.py depth    -> results.csv (uses chosen.json: {file: person_index});
                                                   *_m samples the floor strip below the box, *_torso_m the torso patch
       python run_distance_experiment.py ground   -> adds flat ground-plane ranges (metadata pose and image pose) to results.csv
       python run_distance_experiment.py scaled   -> adds the ground-scaled UniDepth reading (Method E, image pose) to results.csv
"""
import csv, json, re, sys
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "depth_results"
IMAGES = sorted((HERE / "Images").glob("*.JPG"), key=lambda p: float(re.search(r"([\d.]+)m", p.stem).group(1)))
DETECTOR = ROOT / "skydio live data" / "yolo26x.pt"


def truth(path):
    return float(re.search(r"([\d.]+)m", path.stem).group(1))


def detect():
    from ultralytics import YOLO
    model = YOLO(str(DETECTOR), task="detect")
    found = {}
    for path in IMAGES:
        frame = cv2.imread(str(path))
        r = model.predict(frame, imgsz=2048, classes=[0], conf=0.1, verbose=False, device="cuda:0")[0]
        boxes = sorted(((b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes), key=lambda t: t[0][0])
        found[path.name] = [{"index": i + 1, "bbox": b, "conf": c} for i, (b, c) in enumerate(boxes)]
        vis = frame.copy()
        for d in found[path.name]:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(vis, str(d["index"]), (x1, max(40, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imwrite(str(OUT / f"det_{path.stem}.jpg"), cv2.resize(vis, (1600, 900)))
        print(path.name, [(d["index"], [round(v) for v in d["bbox"]], round(d["conf"], 2)) for d in found[path.name]])
    (OUT / "detections.json").write_text(json.dumps(found, indent=1))


def depth():
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "webcam_depth"))
    import torch
    from run_webcam_depth import DepthEngine, parser, sample_torso

    def sample_floor(depth, bbox):
        """Median depth of the floor strip 1-3% of box height below the box, over 35-65% of its width.

        A distant person is narrower than one UniDepth backbone patch, so the torso patch reads the wall
        behind them; the floor at the feet is at the person's distance and is resolved. The torso reading
        is kept alongside for comparison with the aerial evaluation."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        patch = depth[int(y2 + .01 * h):min(depth.shape[0], int(y2 + .03 * h)), int(x1 + .35 * w):int(x1 + .65 * w)]
        valid = np.isfinite(patch) & (patch > 0)
        return {"depth_m": float(np.median(patch[valid])) if valid.any() else None,
                "torso_m": sample_torso(depth, bbox)["depth_m"]}
    from reference_validation.depth_backends.depth_pro import DepthProBackend
    args = parser().parse_args(["--depth-imgsz", "1280", "--resolution-level", "9"])
    engine = DepthEngine(args)
    pro = DepthProBackend(device="cuda:0", precision="float16").load()
    detections = json.loads((OUT / "detections.json").read_text())
    chosen = json.loads((HERE / "chosen.json").read_text())
    rows = []
    for path in IMAGES:
        frame = cv2.imread(str(path))
        person = next(d for d in detections[path.name] if d["index"] == chosen[path.name])
        people = [{"index": 1, "bbox": person["bbox"], "estimates": {}}]
        # YOLO depth heads run at 1280 and are resized back to the full frame for the same torso sampling
        for name, model in (("yolo_original", engine.yolo_original), ("yolo_finetuned", engine.yolo)):
            res = engine.yolo_predict(model, frame, imgsz=1280)
            d = res.depth.data.detach().float().cpu().numpy().squeeze()
            d = cv2.resize(d, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) if d.shape != frame.shape[:2] else d
            people[0]["estimates"][name] = sample_floor(d, person["bbox"])
        # UniDepth + LingBot refinement (LingBot is seeded with UniDepth depth, not an independent measurement)
        engine.unidepth.device = engine.device
        engine.unidepth.model.to(engine.device)
        try:
            uni = engine.unidepth.predict(frame)
        finally:
            engine.unidepth.model.to("cpu"); engine.clear_cuda()
        people[0]["estimates"]["unidepth"] = sample_floor(uni["depth"], person["bbox"])
        ling = engine.lingbot.predict(frame, uni["depth"], intrinsics=np.asarray(uni["metadata"]["predicted_intrinsics"], dtype=np.float32))
        people[0]["estimates"]["lingbot"] = sample_floor(ling["depth"], person["bbox"])
        dp = pro.predict(frame)
        people[0]["estimates"]["depth_pro"] = sample_floor(dp["depth"], person["bbox"])
        row = {"image": path.name, "actual_m": truth(path), "person_index": person["index"]}
        for k, v in people[0]["estimates"].items():
            row[k + "_m"] = None if v["depth_m"] is None else round(v["depth_m"], 2)
            row[k + "_torso_m"] = None if v["torso_m"] is None else round(v["torso_m"], 2)
        rows.append(row); print(row, flush=True)
        torch.cuda.empty_cache()
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ponytail: assumed top-of-rail height (36 in, middle of the 34-38 in handrail range); range scales with it, so a tape-measured rail height replaces this.
RAIL_TOP_M = 0.91


def vanishing_point(gray, seed=0):
    """Corridor vanishing point from LSD segments: length-weighted RANSAC, then least squares.

    Returns (vp_xy, inlier_segments). Vertical and near-horizontal segments are ignored.
    """
    seg = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(*d.T)
    ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    keep = (length > 60) & (ang > 8) & (ang < 82)
    seg, length = seg[keep], length[keep]
    mid, u = (seg[:, :2] + seg[:, 2:]) / 2, (seg[:, 2:] - seg[:, :2]) / length[:, None]
    hom = np.cross(np.c_[seg[:, :2], np.ones(len(seg))], np.c_[seg[:, 2:], np.ones(len(seg))])

    def inliers(p, tol_deg=0.5):
        v = p - mid
        cos = np.abs((v * u).sum(1)) / np.hypot(*v.T)
        return np.degrees(np.arccos(np.clip(cos, -1, 1))) < tol_deg

    rng, best, best_score = np.random.default_rng(seed), None, -1
    for _ in range(3000):
        i, j = rng.choice(len(seg), 2, replace=False, p=length / length.sum())
        p = np.cross(hom[i], hom[j])
        if abs(p[2]) > 1e-9 and (score := length[inliers(p[:2] / p[2])].sum()) > best_score:
            best, best_score = p[:2] / p[2], score
    m = inliers(best)
    n, wt = np.c_[-u[m, 1], u[m, 0]], length[m]
    vp = np.linalg.solve((n * wt[:, None]).T @ n, (n * wt[:, None]).T @ (n * mid[m]).sum(1))
    return vp, seg[m]


def image_pose(gray, fx):
    """Pitch (deg, up positive) from the vanishing-point row; camera height from the right-wall rail.

    The rail top (longest right-wall line) and the floor/baseboard junction (steepest one) both run
    to the vanishing point, so camera height = RAIL_TOP_M * s_floor / (s_floor - s_rail) for slopes s.
    Assumes zero roll and a level floor.
    """
    h, w = gray.shape
    vp, seg = vanishing_point(gray)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    length = np.hypot(*(seg[:, 2:] - seg[:, :2]).T)
    right = (mid[:, 0] > vp[0] + 200) & (mid[:, 1] > vp[1])
    slope = (mid[right, 1] - vp[1]) / (mid[right, 0] - vp[0])
    s_rail, s_floor = slope[np.argmax(length[right])], slope.max()
    assert 0 < s_rail < s_floor, (s_rail, s_floor)
    pitch = np.degrees(np.arctan((vp[1] - (h - 1) / 2) / fx))  # horizon below centre = camera tilted up
    lines = {"rail": (vp, vp + [w, w * s_rail]), "floor": (vp, vp + [w, w * s_floor])}
    return float(pitch), float(RAIL_TOP_M * s_floor / (s_floor - s_rail)), vp, seg, lines


def ground():
    """Add flat ground-plane ranges (no depth model) to results.csv, with the camera pose taken
    (a) from the DJI metadata and (b) from the image itself. Saves line overlays for checking."""
    import math
    from PIL import Image
    sys.path[:0] = [str(ROOT), str(ROOT / "reference_validation"), str(ROOT / "skydio live data")]
    from person_geolocator.metadata import FULL_FRAME_DIAGONAL_MM, read_pose
    from human_geolocation import estimate_ground_plane_position
    detections = json.loads((OUT / "detections.json").read_text())
    chosen = json.loads((HERE / "chosen.json").read_text())
    with (OUT / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = HERE / "Images" / row["image"]
        pose = read_pose(path)["fields"]
        frame = cv2.imread(str(path))
        h, w = frame.shape[:2]
        # FC7203 (Mini 2) 16:9 photos crop a 4:3 sensor; the 35 mm equivalent describes the 4:3 frame, width = 0.8 x diagonal.
        f35 = float(Image.open(path).getexif().get_ifd(0x8769)[0xA405])
        hfov = math.degrees(2 * math.atan(0.8 * FULL_FRAME_DIAGONAL_MM / 2 / f35))
        fx = w / 2 / math.tan(math.radians(hfov) / 2)
        box = next(d["bbox"] for d in detections[row["image"]] if d["index"] == chosen[row["image"]])

        def ground_range(height, pitch, roll):
            # Indoors: no GPS, so a (0,0) origin and the floor at height 0.
            telemetry = {"latitude": 0, "longitude": 0, "altitude_hae": height, "sensor_azimuth": 0,
                         "sensor_elevation": pitch, "sensor_roll": roll, "sensor_hfov": hfov,
                         "sensor_vfov": math.degrees(2 * math.atan(h / 2 / fx))}
            g = estimate_ground_plane_position(box, telemetry, (h, w), ground_hae=0.0)
            assert g["status"] == "estimated", (row["image"], g["reason"])
            return g, round(math.hypot(g["north_m"], g["east_m"]), 2)  # horizontal, like the tape

        g, meta_range = ground_range(pose["height_above_ground_m"], pose["pitch_deg"], pose["roll_deg"])
        pitch, height, vp, seg, lines = image_pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fx)
        g_img, img_range = ground_range(height, pitch, 0.0)
        row.update(hfov_deg=round(hfov, 2),
                   camera_height_m=pose["height_above_ground_m"], gimbal_pitch_deg=pose["pitch_deg"],
                   depression_deg=round(g["depression_deg"], 2), ground_plane_m=meta_range,
                   image_height_m=round(height, 2), image_pitch_deg=round(pitch, 2),
                   image_depression_deg=round(g_img["depression_deg"], 2), ground_plane_image_m=img_range)
        for x1, y1, x2, y2 in seg.astype(int):
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        for (a, b), color in zip(lines.values(), ((255, 0, 255), (255, 255, 0))):
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 3)
        cv2.line(frame, (0, int(vp[1])), (w, int(vp[1])), (0, 255, 0), 3)
        cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 255, 0), 4)
        cv2.imwrite(str(OUT / f"ground_{path.stem}.jpg"), cv2.resize(frame, (1600, 900)))
        print(row, flush=True)
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def scaled():
    """Add the ground-scaled UniDepth reading (Method E) to results.csv: UniDepth is re-run at level 9 and its
    map rescaled by the ground it shows, with the image-read pose of ground() and every detection box excluded."""
    import math
    sys.path[:0] = [str(ROOT), str(ROOT / "webcam_depth"), str(ROOT / "reference_validation")]
    from run_webcam_depth import DepthEngine, parser
    from metadata_geometry import euler_rdf_to_ned, ground_scale
    engine = DepthEngine(parser().parse_args(["--depth-imgsz", "1280", "--resolution-level", "9"]))
    detections = json.loads((OUT / "detections.json").read_text())
    with (OUT / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        frame = cv2.imread(str(HERE / "Images" / row["image"]))
        h, w = frame.shape[:2]
        fx = w / 2 / math.tan(math.radians(float(row["hfov_deg"])) / 2)
        capture = {"width": w, "height": h, "intrinsics": dict(fx=fx, fy=fx, cx=(w - 1) / 2, cy=(h - 1) / 2),
                   "distortion_k1_k2_k3": (0.0, 0.0, 0.0),
                   "camera_rdf_to_ned": euler_rdf_to_ned(0.0, float(row["image_pitch_deg"]), 0.0).tolist(),
                   "telemetry": {"altitude_hae": float(row["image_height_m"])}, "assumed_launch_ground_hae_m": 0.0,
                   "geometry_status": "provisional"}
        engine.unidepth.device = engine.device
        engine.unidepth.model.to(engine.device)
        try:
            depth = engine.unidepth.predict(frame)["depth"]
        finally:
            engine.unidepth.model.to("cpu"); engine.clear_cuda()
        scale = ground_scale(depth, capture, frame.shape, [d["bbox"] for d in detections[row["image"]]])
        assert scale is not None and 0.5 < scale < 2, (row["image"], scale)
        row.update(ground_scale=round(scale, 3), ground_scaled_m=round(float(row["unidepth_m"]) * scale, 2))
        print(row["image"], row["unidepth_m"], "x", row["ground_scale"], "=", row["ground_scaled_m"], "true", row["actual_m"], flush=True)
    with (OUT / "results.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    {"detect": detect, "depth": depth, "ground": ground, "scaled": scaled}[sys.argv[1]]()
