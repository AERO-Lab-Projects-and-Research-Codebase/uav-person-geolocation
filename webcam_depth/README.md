# Webcam person depth comparison

From the project folder, run:

```powershell
.\run_webcam_depth.ps1
```

The window opens webcam 0. Stand in view of the camera. The left panel is the
live feed; the right panel is the captured frame used by all four models.
Each detected person gets four torso depth estimates in metres below the
images. The smaller rectangle marks the sampled torso region. Press **Q** or
**Esc**, or close the window, to quit. The script releases the camera on exit.

The comparison includes the original pretrained checkpoint
`skydio live data/yolo26x-depth.pt`, the fine-tuned
`aerialmetric_yolo26x_training/weights/best.pt`, UniDepth V2, and LingBot.
Person boxes come from the separate `skydio live data/yolo26x.pt` detector.
Each YOLO depth checkpoint's saved calibration is used directly and logged
separately; no extra scaling is fitted. Use `--yolo-original-weights PATH` to
choose another original checkpoint, or `--yolo-weights PATH` for fine-tuned weights.

**LingBot refines the UniDepth prediction.** Official LingBot-Depth needs RGB
and a registered depth input. An ordinary webcam supplies RGB, so this demo
feeds UniDepth V2's depth map and estimated camera intrinsics into LingBot.
Its column is labelled **LingBot refined UniDepth**; it is dependent on
UniDepth and is not an independent sensor measurement. The adapter uses
official LingBot v0.5 with native PyTorch attention and local, pinned weights.
See the [official LingBot interface](https://github.com/Robbyant/lingbot-depth).

The numbers are **axial depth**: distance forward along the camera's optical
axis at the torso, rather than Euclidean range to an off-centre person. They
are model estimates. AerialMetric fine-tuning has not established indoor
webcam accuracy. Compare readings against a tape-measured distance before
drawing conclusions about accuracy; model agreement alone does not establish it.

All four maps use the same captured RGB frame and the same boxes. The torso
patch spans 35–65% of box width and 25–60% of box height. Its median uses finite
positive pixels and requires at least half the patch to be valid; otherwise
the display says `unavailable`. Person numbers restart left-to-right in each
frame and are not tracked identities.

## Setup and other cameras

The current project already has the required Python environment, UniDepth
weights, YOLO checkpoints, and downloaded LingBot weights. Setup is repeatable:

```powershell
.\webcam_depth\setup_webcam_depth.ps1
.\run_webcam_depth.ps1 --check
.\run_webcam_depth.ps1 --camera 1
```

The launcher uses `.venvs/depth-comparison/Scripts/python.exe`. Setup adds the
official source at `external/LingBotDepth` and verifies the downloaded
`models/lingbot_depth/model.pt` against its pinned SHA256. It retains the
existing CUDA PyTorch runtime. The other depth models are set up by
`reference_validation/setup_depth_environment.ps1` and `download_depth_models.py`.

The viewer stays responsive using separate capture and inference threads.
Only the latest camera frame is queued; models run serially and move back to
CPU after each stage to limit GPU memory use. The analyzed-frame age is shown
above the right panel. Estimates update more slowly than the live preview.
Requested camera size defaults to 640x480; the actual camera output can differ.

Every run creates a new directory under `webcam_depth/outputs/` containing
`manifest.json`, `estimates.jsonl`, and `people.csv`. Logs include all detected
people, per-model torso statistics, processing times, calibration, and the
LingBot seed relationship. Webcam video/images are not automatically recorded.
The CSV's `yolo_original_m` column holds original YOLO depths; `yolo_m` retains
the fine-tuned values for compatibility with earlier runs. JSON uses
`yolo_original` and `yolo`, respectively.

## Saved-image and bounded checks

```powershell
.\run_webcam_depth.ps1 --image PHOTO.jpg --headless
.\run_webcam_depth.ps1 --headless --max-results 2 --timeout 180
```

Saved-image mode also saves `comparison.jpg`. `--check` opens no camera and
does not run inference. `--timeout` bounds the webcam run, including background
model loading. Camera/inference errors exit with a nonzero code and a message;
they do not silently replace any model with another one.

If calibrated pixel intrinsics are available, use `--intrinsics camera.json`
with `{"width":640,"height":480,"K":[[fx,0,cx],[0,fy,cy],[0,0,1]]}` (replace
the symbolic values with numbers). The dimensions must match the actual
processed frame. UniDepth otherwise estimates K; LingBot uses that same K.

Additional options are listed by `.\run_webcam_depth.ps1 --help`.

## Checks completed on this machine

The four-model path completed on the saved indoor `source_lab.jpg` with two
people in 3.01 seconds, including GPU transfers and offload. Original YOLO
returned 4.63 and 4.62 m; fine-tuned YOLO returned 65.97 and 82.24 m; UniDepth
returned 4.17 and 5.15 m; LingBot refinement returned 3.76 and 5.12 m. These are
model outputs, not measured ground-truth distances. The annotated comparison
and logs are in `outputs/indoor_four_models_20261005/`.

Webcam 0 was also checked at 640x480 with all four depth models loaded. Its
two processed frames had no person detections, so depth inference was skipped
and the logs correctly recorded empty results. Earlier three-model webcam
checks had person detections. These are short smoke checks, not a sustained
performance benchmark or a distance-accuracy test.

30 offline tests cover frame association, distinct original/fine-tuned outputs,
separate checkpoint calibrations and CSV columns, sampling and invalid depth,
intrinsics, logging, and LingBot's native attention adapter. The adapter was
compared with independently constructed block-diagonal attention, checked for
no cross-example mixing, and tested for GPU-model offload after failures.

```powershell
& '.\.venvs\depth-comparison\Scripts\python.exe' -m unittest discover -s webcam_depth -p 'test*.py'
```
