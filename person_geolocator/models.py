"""Lazy-loaded detector and depth models. Every depth map is HxW axial metres at the image size."""
import os

os.environ.setdefault("XFORMERS_DISABLED", "1")

import cv2
import numpy as np

from person_geolocator import ROOT

MODELS = {"yolo_original": "YOLO original", "yolo_finetuned": "YOLO fine-tuned (AerialMetric)",
          "unidepth": "UniDepth V2", "lingbot": "LingBot (UniDepth-seeded)", "depth_pro": "Depth Pro"}
DETECTOR = ROOT / "skydio live data" / "yolo26x.pt"
YOLO_DEPTH = {"yolo_original": ROOT / "skydio live data" / "yolo26x-depth.pt",
              "yolo_finetuned": ROOT / "aerialmetric_yolo26x_training" / "weights" / "best.pt"}


class DepthModels:
    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cache = {}

    def _clear(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.empty_cache()

    def _yolo(self, path, task):
        if path not in self.cache:
            from ultralytics import YOLO
            self.cache[path] = YOLO(str(path), task=task)
        return self.cache[path]

    def _predict(self, model, frame, **kwargs):
        try:
            return model.predict(source=frame, device=self.device, quantize=32, verbose=False,
                                 save=False, **kwargs)[0]
        finally:
            model.to("cpu")  # keep only the running model on the GPU
            self._clear()

    def detect(self, frame, conf=0.4):
        result = self._predict(self._yolo(DETECTOR, "detect"), frame, imgsz=2048, classes=[0], conf=conf)
        boxes = [(b.tolist(), float(c)) for b, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        return sorted(boxes, key=lambda pair: pair[0][0])

    def _unidepth(self, frame):
        if "unidepth" not in self.cache:
            from reference_validation.depth_backends.unidepth_v2 import UniDepthV2Backend
            self.cache["unidepth"] = UniDepthV2Backend(
                device="cpu", model_id=str(ROOT / "models" / "unidepth_v2"), resolution_level=5,
                source_dir=ROOT / "external" / "UniDepth", local_files_only=True).load()
        backend = self.cache["unidepth"]
        backend.device = self.device
        backend.model.to(self.device)
        try:
            return backend.predict(frame)
        finally:
            backend.model.to("cpu")
            self._clear()

    def _lingbot(self, frame, unidepth):
        if "lingbot" not in self.cache:
            from lingbot_backend import LingBotDepthBackend
            self.cache["lingbot"] = LingBotDepthBackend(
                device=self.device, model_id=str(ROOT / "models" / "lingbot_depth" / "model.pt"),
                source_dir=ROOT / "external" / "LingBotDepth", offload=True).load()
        k = np.asarray(unidepth["metadata"]["predicted_intrinsics"], dtype=np.float32)
        return self.cache["lingbot"].predict(frame, unidepth["depth"], intrinsics=k)["depth"]

    def depth(self, name, frame):
        h, w = frame.shape[:2]
        if name in YOLO_DEPTH:
            result = self._predict(self._yolo(YOLO_DEPTH[name], "depth"), frame, imgsz=1280)
            if getattr(result, "depth", None) is None:
                raise ValueError(f"{name} checkpoint returned no depth map")
            depth = result.depth.data.detach().float().cpu().numpy().squeeze()
            return depth if depth.shape == (h, w) else cv2.resize(depth, (w, h))
        if name == "unidepth":
            return self._unidepth(frame)["depth"]
        if name == "lingbot":
            return self._lingbot(frame, self._unidepth(frame))
        if name == "depth_pro":
            if "depth_pro" not in self.cache:
                from reference_validation.depth_backends.depth_pro import DepthProBackend
                self.cache["depth_pro"] = DepthProBackend(device=self.device, precision="float16").load()
            return self.cache["depth_pro"].predict(frame)["depth"]
        raise ValueError(f"Unknown model: {name}")
