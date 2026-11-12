"""Command line: python -m person_geolocator.cli IMAGE --model unidepth [pose overrides]."""
import argparse
import json
from pathlib import Path

from person_geolocator.metadata import read_pose
from person_geolocator.models import MODELS

OVERRIDES = {"lat": "lat", "lon": "lon", "altitude": "altitude_hae", "heading": "heading_deg", "pitch": "pitch_deg",
             "roll": "roll_deg", "hfov": "hfov_deg", "height_agl": "height_above_ground_m"}


def _where(m):
    return f"{m['lat']:.6f}, {m['lon']:.6f}" if m["status"] == "estimated" else f"n/a ({m['reason']})"


def main(argv=None, locate_fn=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", choices=list(MODELS), default="unidepth")
    for flag in OVERRIDES:
        ap.add_argument("--" + flag.replace("_", "-"), type=float)
    ap.add_argument("--json", type=Path, help="write the full result as JSON")
    ap.add_argument("--annotated", type=Path, help="write the annotated image")
    args = ap.parse_args(argv)
    meta = read_pose(args.image)
    pose = dict(meta["fields"])
    for flag, field in OVERRIDES.items():
        if getattr(args, flag) is not None:
            pose[field] = getattr(args, flag)
    for warning in meta["warnings"]:
        print("note:", warning)
    if locate_fn is None:
        from person_geolocator.engine import locate
        from person_geolocator.models import DepthModels
        models = DepthModels()
        locate_fn = lambda path, model, pose: locate(path, model, pose, models)
    result = locate_fn(args.image, args.model, pose)
    print(f"{len(result['people'])} person(s) | model {args.model} | geometry {result['geometry_status']} {result['geometry_errors']}")
    for p in result["people"]:
        m = p["methods"]
        depth = "n/a" if p["depth_m"] is None else f"{p['depth_m']:.2f} m"
        print(f"#{p['index']} depth {depth}\n  depth-ray    {_where(m['depth_ray'])}\n  ground-plane {_where(m['ground_plane'])}"
              + (f"\n  terrain      {_where(m['terrain'])}" if "terrain" in m else "")
              + (f"\n  ground-scaled {_where(m['ground_scaled'])}" if "ground_scaled" in m else "")
              + f"\n  fusion       {_where(m['fusion'])} ({m['fusion']['source'] or 'no source'})"
              + (f"\n  scaled fusion {_where(m['scaled_fusion'])} ({m['scaled_fusion']['source'] or 'no source'})" if "scaled_fusion" in m else ""))
    jpeg = result.pop("annotated_jpeg", None)
    if args.annotated and jpeg:
        args.annotated.write_bytes(jpeg)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
