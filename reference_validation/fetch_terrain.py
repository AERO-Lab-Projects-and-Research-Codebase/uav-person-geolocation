"""Download a USGS 3DEP bare-earth grid around the saved photo positions (one time).

Sends the site bounding box to USGS and one site point to NOAA NGS (GEOID18).
Run with --print-only first to see the exact requests. Output: an .npz grid
read by terrain.TerrainModel. Needs tifffile + imagecodecs (download only).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from terrain import TerrainModel

ROOT = Path(__file__).resolve().parent
EXPORT_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
GEOID_URL = "https://geodesy.noaa.gov/api/geoid/ght"
GEOID18_MODEL = 14
METRES_PER_DEGREE = 111_320.  # only used to size the request
MAX_PIXELS = 8000  # service maxImageWidth/Height


def camera_points(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    points = []
    for entry in manifest["records"]:
        saved = json.loads((run_dir/entry["result_json"]).read_text(encoding="utf-8"))
        t = saved.get("capture_metadata", {}).get("telemetry", {})
        if t.get("latitude") is not None and t.get("longitude") is not None:
            points.append((float(t["latitude"]), float(t["longitude"])))
    return points


def site_bounds(points, margin_m):
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    mid = (min(lats)+max(lats))/2
    dlat = margin_m/METRES_PER_DEGREE
    dlon = margin_m/(METRES_PER_DEGREE*math.cos(math.radians(mid)))
    return min(lons)-dlon, min(lats)-dlat, max(lons)+dlon, max(lats)+dlat


def grid_size(bounds, resolution_m):
    west, south, east, north = bounds
    mid = (south+north)/2
    width = math.ceil((east-west)*METRES_PER_DEGREE*math.cos(math.radians(mid))/resolution_m)
    height = math.ceil((north-south)*METRES_PER_DEGREE/resolution_m)
    if max(width, height) > MAX_PIXELS:
        raise ValueError("terrain_grid_too_large")
    return width, height


def export_params(bounds, size):
    return {"bbox": ",".join(f"{v:.7f}" for v in bounds), "bboxSR": 4326, "imageSR": 4326,
            "size": f"{size[0]},{size[1]}", "format": "tiff", "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation", "noData": -9999, "f": "json"}


def geoid_params(lat, lon):
    return {"lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "model": GEOID18_MODEL}


def _get(url, params=None):
    with urlopen(url+("?"+urlencode(params) if params else ""), timeout=120) as response:
        return response.read()


def download(bounds, size, output):
    import tifffile  # download-only dependency

    params = export_params(bounds, size)
    info = json.loads(_get(EXPORT_URL, params))
    if "href" not in info or "extent" not in info:
        raise ValueError(f"Unexpected exportImage response: {info}")
    grid = tifffile.imread(io.BytesIO(_get(info["href"])))
    if grid.ndim == 3:
        grid = grid[..., 0] if grid.shape[-1] == 1 else grid[0]
    if list(grid.shape) != [info["height"], info["width"]]:
        raise ValueError(f"Grid shape {grid.shape} does not match response {info['width']}x{info['height']}")
    e = info["extent"]
    lat, lon = (e["ymin"]+e["ymax"])/2, (e["xmin"]+e["xmax"])/2
    geoid = json.loads(_get(GEOID_URL, geoid_params(lat, lon)))
    n = float(geoid["geoidHeight"])
    if not (math.isfinite(n) and abs(n) < 120):
        raise ValueError(f"Implausible geoid height: {geoid}")
    metadata = {"source": "USGS 3DEP dynamic elevation service (bare earth, NAVD88 metres)",
                "export_url": EXPORT_URL, "export_params": params, "response_extent": e,
                "geoid": {"service": GEOID_URL, "model": geoid.get("geoidModel"), "point": [lat, lon],
                          "geoid_height_m": n, "error_m": geoid.get("error")},
                "resampling": "best-available 3DEP source, resampled bilinearly by the service",
                "downloaded_at": datetime.now(timezone.utc).isoformat()}
    model = TerrainModel(grid, e["xmin"], e["ymin"], e["xmax"], e["ymax"], n, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="photos_det2048_native", help="Saved detection run under outputs/")
    parser.add_argument("--margin-m", type=float, default=600.)
    parser.add_argument("--resolution-m", type=float, default=1.)
    parser.add_argument("--output", type=Path, default=ROOT/"terrain"/"site_dem.npz")
    parser.add_argument("--print-only", action="store_true", help="Show the requests without sending them")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.print_only and not args.force:
        raise SystemExit(f"{args.output} already exists; use --force to download again and overwrite it.")
    points = camera_points(ROOT/"outputs"/args.run)
    if not points:
        raise SystemExit("No camera positions in the saved run.")
    bounds = site_bounds(points, args.margin_m)
    size = grid_size(bounds, args.resolution_m)
    west, south, east, north = bounds
    print(f"{len(points)} camera positions -> {size[0]} x {size[1]} grid")
    print("USGS request:", EXPORT_URL+"?"+urlencode(export_params(bounds, size)))
    print("NOAA request:", GEOID_URL+"?"+urlencode(geoid_params((south+north)/2, (west+east)/2)))
    if args.print_only:
        return
    model = download(bounds, size, args.output)
    valid = (~np.isnan(model.elevation)).mean()
    print(f"Saved {args.output} · geoid {model.geoid_height_m:.3f} m · valid cells {valid:.1%}")


if __name__ == "__main__":
    main()
