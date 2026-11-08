# Skydio live human detection and position estimates

`main.py` reads the existing RTSP stream and CoT telemetry. A separate worker runs
`yolo26x.pt` for **person detection** and `yolo26x-depth.pt` for **dense depth** on
the same image. The dashboard displays boxes and a table of estimated person
depth, latitude/longitude, and North/East offsets. It has no aircraft control API.

## Run

Keep the controller's working RTSP and telemetry settings. In PowerShell, from
this folder:

```powershell
.\.venv\Scripts\python.exe main.py
```

Then open <http://127.0.0.1:5000>. Restart the existing Python process to load code
changes. The detector/depth models load in a background thread; raw video remains
available while loading or if inference fails. The **Show raw video** checkbox
changes the displayed stream.

The default RTSP URL is preserved from the working setup. If it changes, use:

```powershell
$env:RTSP_URL = 'rtsp://CONTROLLER_IP:8554/skydio'
.\.venv\Scripts\python.exe main.py
```

CoT still arrives separately on laptop **TCP 4242**. The telemetry destination
must be a laptop address reachable from the controller. Flask port 5000 is the
browser endpoint. No takeoff is required for this application.

## What the estimates mean

- When telemetry explicitly reports latitude **0** and longitude **0**, the
  calculation uses **(0,0)** as requested. Results are marked
  `assumed_zero_origin`: they are offsets around a hypothetical origin, not the
  laboratory's real GPS coordinates. A missing, null or nonfinite value is not
  converted to zero.
- A small central torso region inside each person box supplies the median depth.
  The estimate is the position of that visible torso region, not a surveyed foot
  point or a tracked identity. Person numbers reset each processed frame.
- The official depth checkpoint supplies a pretrained metric scale. That scale
  has not been calibrated or validated for this Skydio camera. The supplied
  detector and depth model are RGB models; thermal accuracy is unvalidated.
- Projection assumes a pinhole camera whose current CoT horizontal/vertical FOV
  and sensor attitude describe the streamed camera. Azimuth is clockwise from
  true north, elevation is positive upward, and roll is clockwise looking along
  the optical axis. These conventions and boresight need physical validation.
- The default interprets depth as optical-axis Z. With pixel ray
  `r = [(u-cx)/fx, (v-cy)/fy, 1]`, the camera point is `depth * r`.
  `HUMAN_DEPTH_MODE=range` instead uses `depth * r / norm(r)`. Camera attitude
  rotates that point into North/East/Down. A WGS84 local tangent approximation
  converts the horizontal offsets to latitude/longitude.
- HAE is the aircraft's ellipsoidal altitude, **not height above the floor**.
  `sensor_range` is not used as a measured distance to a person.
- Every person also carries a **ground-plane cross-check** (`ground_plane`). The
  ray through the bottom centre of the box is intersected with a flat plane at
  `GROUND_HAE_M`, using only aircraft HAE, CoT attitude and CoT FOV. It never
  uses the depth network, so it is immune to depth-model scale and to the depth
  shrinkage a monocular model shows when the camera zooms. It still depends on
  the CoT FOV being correct for the streamed camera, on the person's feet being
  visible and on the ground being flat at the surveyed height. When
  `GROUND_HAE_M` is unset, the first aircraft HAE received **with a real GPS fix**
  is used as launch-point ground and labelled `launch_hae_assumed_ground`; the
  indoor `0,0` placeholder never becomes ground height, so in the lab set
  `GROUND_HAE_M` explicitly. A level or
  upward-looking camera reports `ray_does_not_reach_ground`.
  `estimate_disagreement_m` is the horizontal distance between the depth and
  ground-plane fixes; both are written to the JSONL log for post-flight
  comparison against surveyed markers.
- Geolocation selects a complete recent aircraft CoT event received before the
  frame was decoded. It never merges fields from different UIDs. This is local
  arrival-time matching, not hardware timestamp synchronization. Missing/stale
  telemetry withholds GPS estimates while person detection/depth can continue.
- Only the matched inference frame gets boxes. If it becomes stale, the annotated
  route falls back to raw video and the dashboard marks old estimates historical.

## Configuration

Environment variables are optional. Defaults:

| Variable | Default | Meaning |
|---|---|---|
| `HUMAN_ANALYSIS_ENABLED` | `true` | Set `false` for the original raw dashboard workflow |
| `HUMAN_DETECTOR_WEIGHTS` | `yolo26x.pt` | Existing local person detector |
| `HUMAN_DEPTH_WEIGHTS` | `yolo26x-depth.pt` | Local depth checkpoint |
| `HUMAN_DEVICE` | `auto` | CUDA if available, otherwise CPU |
| `HUMAN_DETECTOR_IMGSZ` | `640` | Detection input size |
| `HUMAN_DEPTH_IMGSZ` | `768` | Depth input size |
| `HUMAN_CONFIDENCE` | `0.4` | Person confidence cutoff |
| `HUMAN_MAX_FPS` | `3` | Maximum analysis rate; capture runs independently |
| `HUMAN_DEPTH_MODE` | `axial` | `axial` or `range` depth convention |
| `HUMAN_DEPTH_SCALE` | `1` | Optional measured multiplicative correction |
| `HUMAN_MAX_DEPTH_M` | `100` | Maximum accepted sample depth and slant range |
| `HUMAN_FRAME_MAX_AGE_S` | `3` | Maximum displayed inference-frame age |
| `HUMAN_TELEMETRY_MAX_AGE_S` | `3` | Maximum local CoT receive age at frame decode |
| `GROUND_HAE_M` | unset | Surveyed ground ellipsoid height (m) for the ground-plane cross-check; unset uses the first aircraft HAE with a GPS fix as launch ground |
| `GEO_AIRCRAFT_UID` | empty | Pin a verified aircraft UID; otherwise select air-platform CoT types |
| `HUMAN_LOG_DIR` | `output/person_estimates` | Per-inference JSONL records |

The worker requires local weights and reports a useful error if either is absent;
it does not download models in the live loop. CPU is supported but both X models
can be slow, so use the GPU environment for live analysis.

## Installation and tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional NVIDIA CUDA environment:
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic frames, analytic geometry and local Flask test clients; they
do not connect to the controller. CUDA wheels contain very long filenames. If
Windows reports `WinError 206`, use a short path to the environment for installation
instead of changing system-wide registry settings.

Endpoints: `/api/people` returns strict JSON estimates and status; `/human_video_feed`
serves annotated frames with raw fallback; `/video_feed` preserves raw video;
`/api/telemetry` preserves the telemetry dashboard. JSONL logs are written once per
inference, independent of how many browser clients poll. Existing telemetry CSVs
remain in `telemetry_logs`.

## Model and hardware references

- [Ultralytics YOLO26 depth task and metric calibration](https://docs.ultralytics.com/tasks/depth/)
- [Skydio RTSP and telemetry streaming](https://support.skydio.com/hc/en-us/articles/51920129647387-How-to-set-up-RTSP-and-Telemetry-Streaming-on-the-Skydio-X10-X10D)
- [Skydio CoT/ATAK publishing](https://support.skydio.com/hc/en-us/articles/22838103434907-How-to-set-up-ATAK-with-the-Skydio-X10-Controller)

Receiving video and telemetry, passing software tests, and displaying coordinates
do not establish target-location accuracy. Compare depth and offsets against
stationary targets at measured distances before drawing accuracy conclusions.
