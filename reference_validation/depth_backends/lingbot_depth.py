"""LingBot-Depth refinement of a UniDepth V2 estimate, for saved photographs.

Official LingBot-Depth needs RGB plus a registered metric depth input. A
photograph carries no depth sensor, so UniDepth V2 supplies the seed on the same
rectified image and intrinsics, and LingBot refines it. The result is therefore
not an independent depth measurement. The seed model and its provenance are
recorded in the metadata. No weights are downloaded at runtime.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webcam_depth.lingbot_backend import LingBotDepthBackend  # noqa: E402
from .unidepth_v2 import UniDepthV2Backend  # noqa: E402


class LingBotUniDepthBackend:
    def __init__(self, device="cuda:0", unidepth_model_id=None, unidepth_resolution=5,
                 lingbot_model_id=None, lingbot_resolution=0, seed_backend=None, refiner=None):
        self.device = str(device)
        self.seed_backend = seed_backend or UniDepthV2Backend(
            device=device, model_id=str(unidepth_model_id), resolution_level=unidepth_resolution)
        self.refiner = refiner or LingBotDepthBackend(
            device=device, model_id=lingbot_model_id, resolution_level=lingbot_resolution)

    def load(self):
        self.seed_backend.load()
        self.refiner.load()
        return self

    def predict(self, frame_bgr, intrinsics=None):
        """Return the LingBot-refined axial-Z depth and combined provenance.

        ``intrinsics`` is the pixel K of ``frame_bgr``; both models receive it.
        """
        seed = self.seed_backend.predict(frame_bgr, intrinsics=intrinsics)
        refined = self.refiner.predict(frame_bgr, seed["depth"], intrinsics=intrinsics)
        seed_ms = seed["metadata"].get("runtime_ms")
        refine_ms = refined["metadata"].get("runtime_ms")
        metadata = dict(refined["metadata"])
        metadata.update(
            backend="lingbot_depth",
            depth_seed_source="unidepth_v2_same_image_and_intrinsics",
            independent_depth_estimate=False,
            seed=seed["metadata"],
            seed_runtime_ms=seed_ms,
            refinement_runtime_ms=refine_ms,
            runtime_ms=None if seed_ms is None or refine_ms is None else seed_ms + refine_ms,
            runtime_scope="unidepth_seed_plus_lingbot_refinement_excluding_load",
        )
        return {"depth": refined["depth"], "metadata": metadata}
