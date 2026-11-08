"""Upload an image, pick a depth model, get depth and GPS for every person."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Existing project code imports its siblings by bare name (metadata_geometry, run_webcam_depth, ...).
for sub in ("webcam_depth", "reference_validation", ""):
    sys.path.insert(0, str(ROOT / sub))
