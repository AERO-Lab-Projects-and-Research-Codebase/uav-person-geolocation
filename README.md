# UAV person geolocation

Code and results for the paper *Placing Detected People on the Map: Geolocation from Small Commercial UAV Imagery*.

A person detected in one UAV photograph is placed in geographic coordinates from the photo's own metadata (GPS,
altitude, gimbal pose, lens) and one range along the detection's camera ray. The range comes from one of three
sources, with two corrections:

| Method | Range source | Code |
|---|---|---|
| A | Monocular metric depth at the torso (UniDepth V2, Depth Pro, LingBot, YOLO26-depth) | `person_geolocator/engine.py`, `reference_validation/depth_backends/` |
| B | Box-bottom ray intersected with a USGS 3DEP terrain grid | `reference_validation/terrain.py`, `fetch_terrain.py` |
| C | YOLO26x depth head fine-tuned on AerialMetric | `aerialmetric_yolo26x_training/` |
| D | Fusion of a depth ray with its flat-ground intersection, gated by depression angle and agreement | `skydio live data/human_geolocation.py` (`fuse_estimates`) |
| E | Depth map rescaled so that its ground lies at the camera's known height | `reference_validation/metadata_geometry.py` (`apply_capture_geometry`) |

Camera geometry (undistortion, pixel ray to north-east-down, offsets to latitude and longitude) is in
`reference_validation/metadata_geometry.py`; `person_geolocator/` wraps detector, depth models and geometry into one
`locate()` call with a command line, a photo-folder runner and a small web page.

## Layout

| Folder | Contents |
|---|---|
| `person_geolocator/` | Offline pipeline: `cli.py` (one photo), `photos.py` (a folder, pose from a DJI flight log), `server.py` (web page), tests |
| `reference_validation/` | Camera geometry, terrain model, depth backends, run and scoring scripts, environment setup, tests; `outputs/minise_20261101_flight4*/` holds the second-UAV flight's per-photo results |
| `skydio live data/` | Live dashboard for the Skydio RTSP stream and the estimator rules (`human_geolocation.py`) the offline code imports |
| `Skydio_Flight_Experiment/` | Three-person flight test: `run_experiment.py`, chart scripts, `outputs/` with per-photo results, `people.csv`, `truth.csv`, `summary.csv` |
| `geolocation_experiment_image/` | Ground-level lab test at 7.5–52.5 m: detections, chosen boxes and `depth_results/results.csv` |
| `webcam_depth/` | Indoor bench test with a laptop webcam; `outputs/*/people.csv` |
| `aerialmetric_yolo26x_training/` | Fine-tuning recipe for the YOLO26x depth head on AerialMetric, with the dataset licence |
| `unidepth_v2_finetune/` | Fine-tuning recipe for UniDepth V2 on AerialMetric and SkyScenes (not reported in the paper) |
| `Paper/` | The paper's figures and the scripts that draw them from the result files |

## Setup

Windows with an NVIDIA GPU is what the results were produced on; the scripts are plain Python and run on CPU, slowly.

1. Create the base environment with CUDA PyTorch and Ultralytics:
   ```powershell
   python -m venv "skydio live data\.venv"
   "skydio live data\.venv\Scripts\python.exe" -m pip install -r "skydio live data\requirements-cuda.txt"
   ```
2. Create the depth environment on top of it and clone the pinned UniDepth and Depth Pro sources into `external/`:
   ```powershell
   .\reference_validation\setup_depth_environment.ps1
   ```
3. Add LingBot-Depth (pinned source and checkpoint):
   ```powershell
   .\webcam_depth\setup_webcam_depth.ps1
   ```
4. Download the UniDepth V2 and Depth Pro weights into `models/`:
   ```powershell
   .venvs\depth-comparison\Scripts\python.exe reference_validation\download_depth_models.py
   ```
5. Place the Ultralytics checkpoints `yolo26x.pt` (detector) and `yolo26x-depth.pt` (depth) in `skydio live data/`.
   The fine-tuned checkpoint `aerialmetric_yolo26x_training/weights/best.pt` is produced by the recipe in that folder;
   it is not distributed because the AerialMetric data licence restricts derived work to academic use.

Every command below uses `.venvs\depth-comparison\Scripts\python.exe` from the repository root.

## Run the pipeline

```powershell
python -m person_geolocator.cli PHOTO.JPG --model unidepth
python -m person_geolocator.photos PHOTO_DIR --log flight.csv --out outputs/run   # DJI photos, pose from the flight log
python -m person_geolocator.server                                                # web page on http://127.0.0.1:8000
```

## Reproduce the paper

The photographs are not distributed, so the detection and depth stage cannot be rerun from this repository alone. Every number and figure in the paper comes from the saved per-detection results,
which are here, and the commands below rebuild them.

| Paper item | Source data | Command |
|---|---|---|
| Table: AerialMetric fine-tune | `aerialmetric_yolo26x_training/evaluation/*/depth_evaluation.json` | — |
| Table: indoor bench | `webcam_depth/outputs/*/people.csv` | `python webcam_depth/run_webcam_depth.py` (live webcam) |
| Tables and figure: lab test | `geolocation_experiment_image/depth_results/results.csv` | `python geolocation_experiment_image/run_distance_experiment.py depth` then `ground`; `python Paper/build_corridor_figure.py` |
| Table and figure: second UAV | `reference_validation/outputs/minise_20261101_flight4*/` | `cd reference_validation; python score_minise_flight.py --sheet 8`; `python Paper/build_uavb_figure.py` |
| Table and figures: three people | `Skydio_Flight_Experiment/outputs/` | `python Skydio_Flight_Experiment/build_error_charts.py`; `python Skydio_Flight_Experiment/build_topdown.py`; `python Paper/build_three_person_figure.py` |
| Heading check from shadows | the flight photographs | `python Skydio_Flight_Experiment/check_heading_from_shadows.py` |
| Table: correction from known ground points | `Skydio_Flight_Experiment/outputs/`, `outputs_11_9/` | `python Skydio_Flight_Experiment/ground_point_correction.py outputs outputs_11_9` |
| Pipeline figure | — | `python Paper/build_pipeline_figure.py` |

`Skydio_Flight_Experiment/run_experiment.py` reruns the three-person test end to end when the photographs are placed in
`Skydio_Flight_Experiment/Skydio/`. The second-UAV photographs were run with `person_geolocator.photos` and scored with
`reference_validation/score_minise_flight.py`, whose flight record is `reference_validation/minise_flights.json`.

The manuscript itself is distributed by the conference, not from this repository.

## Tests

```powershell
python -m unittest discover -s reference_validation -p "test_*.py"
python -m unittest discover -s "skydio live data/tests"
python -m unittest discover -s webcam_depth -p "test_*.py"
python -m unittest discover -s aerialmetric_yolo26x_training/tests
python -m unittest discover -s unidepth_v2_finetune/tests
python -m unittest discover -s person_geolocator/tests -t .
```

## Data notes

- Result files hold the aircraft's recorded position, pose and lens for each photograph, each detection's box and depth
  reading, and each method's estimate. People appear only as box coordinates.
- No photograph is distributed. The lab test's seven photographs are needed only to rerun its detection and depth
  stage and the detections figure; the tests that read one of them skip when it is absent.
- The flight test's reference positions (`truth.csv`, `minise_flights.json`) are the UAV's own GPS fixes, as described
  in the paper.

## Licence

MIT, see `LICENSE`. UniDepth V2, Depth Pro, LingBot-Depth, Ultralytics and the AerialMetric and SkyScenes datasets
keep their own licences; `aerialmetric_yolo26x_training/licenses/` reproduces the AerialMetric terms.
