# AerialMetric + YOLO26x depth: portable training folder

Copy this entire folder to the training computer. Create a new Python environment
there; do not copy a virtual environment between computers. This package uses
standard 64-bit CPython **3.12 or 3.14**, Ultralytics **8.4.146**, PyTorch **2.14.0** and torchvision
**0.29.0**. NVIDIA CUDA wheels are installed separately from application packages.
Python **3.14.1 is excluded by torchvision 0.29.0**; use a different 3.14 patch
(3.14.2 or newer is recommended). The free-threaded Python variant is not covered
by this bundle's dependency set. Existing Python 3.14 does not require downgrading.
The original complete runtime/model preflight used Python 3.12; Python 3.14 package
wheel availability and setup/diagnostic behavior have been checked separately.
Run the target-PC runtime check and training memory smoke run before full training.

## Included data and model

The included dataset is the **complete dense Oblique training subset** of
[AerialMetric](https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026), from the
two `TrainingData_Final_MoGe-All.tar.part-*` files. Their combined original size is
55.69 GB. This folder does not include the additional sparse LiDAR, unusual-FOV,
synthetic or official benchmark test archives. All Oblique training assets together
are about 162 GB; the complete upstream hub is about 204 GB.

The data are aerial **RGB**, with metric depth obtained by projecting reconstructed
meshes. This training adapts aerial scene interpretation.

`weights/yolo26x-depth.pt` is the existing project's unchanged pretrained checkpoint.
Its SHA256 and provenance are in `weights/provenance.json`. It is the depth model,
not the similarly named person detection model. No training is performed on this computer.

Look at `dataset/preparation_status.json` and `dataset/verification.json` for the
actual sample counts and readiness. `complete: true` and a passing `all_pairs`
verification are required before calling this a fully prepared dataset.

## Run on Windows

Open PowerShell inside the copied folder. To inspect the PC before installing
anything, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CheckOnly
```

This reports the NVIDIA GPU, driver and memory even when Python is missing. If
Python is available, it checks PyTorch and CUDA too. A failed check on a new PC is
expected when dependencies have not been installed; read its specific remedy.
Use **standard 64-bit Python 3.12 or 3.14** and an NVIDIA driver **580 or newer** for the pinned CUDA
13.0 wheels. The scripts provide download links when prerequisites are missing.
Driver installation/updates remain a manual system step; reboot after updating.

Then run setup to install the Python packages and verify them:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\.venv\Scripts\python.exe scripts\verify_dataset.py --sample-only
.\.venv\Scripts\python.exe scripts\train_depth.py --check-only
```

The setup script checks prerequisites, creates `.venv`, installs NVIDIA-enabled
PyTorch from its official CUDA 13.0 index, replaces CPU-only or mismatched PyTorch
builds, and installs `requirements.txt`. It checks package compatibility, a CUDA
matrix calculation, torchvision's CUDA operations and the copied depth checkpoint.
**PyTorch does not need to be installed beforehand.** Internet access is required
to download the packages; the dataset and model are already included in this folder.
No separate CUDA toolkit or `nvcc` installation is normally needed: the PyTorch
wheels supply the runtime. The CUDA version shown by `nvidia-smi` describes driver
support and is not proof that PyTorch is installed or that its CUDA build works.
Neither setup nor `-CheckOnly` starts training.
Setup prefers an existing bundle environment. For a new environment it automatically
selects installed standard 64-bit Python 3.14, falling back to Python 3.12.
If Python is not registered with `py`, supply its executable:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -PythonExecutable C:\Python314\python.exe
```

To repeat the check after setup and save a report:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --json-report checks\environment_check.json
```

Reports include the Python executable, installed torch/torchvision versions,
PyTorch's CUDA runtime, GPU name and total/free VRAM. Missing packages, CPU-only
PyTorch, incompatible drivers and failed CUDA calculations produce a nonzero exit
status and an explanation. `--runtime-only` checks PyTorch/CUDA without loading
Ultralytics or the model. `--device cpu` checks software only and cannot confirm
GPU readiness. Passing these checks does not prove training will fit in GPU memory.

First verify training memory with a small one-epoch run:

```powershell
.\.venv\Scripts\python.exe scripts\train_depth.py --epochs 1 --fraction 0.002 --name smoke_20gb
```

Then start the full fine-tune:

```powershell
.\.venv\Scripts\python.exe scripts\train_depth.py
```

Defaults are `epochs=30`, `imgsz=640`, `batch=2`, AMP enabled, AdamW,
`lr0=1e-4`, `warmup_bias_lr=1e-4`, no RAM dataset cache, and Windows `workers=0`.
They are conservative starting settings for a 20 GB NVIDIA GPU, not a measured
memory guarantee. Reduce `--batch 1` if memory is insufficient. Once the smoke run
fits, optionally test `--imgsz 768` with a new run name. Each run name must be new.
Ultralytics may download a small YOLO26n reference weight for its first AMP check;
it is not the model being trained.

To resume interrupted training:

```powershell
.\.venv\Scripts\python.exe scripts\train_depth.py --resume runs\depth\aerialmetric_yolo26x\weights\last.pt
```

Resume restores optimizer, scheduler and epoch progress. It retains the original
epoch target and learning rate. Copied dataset paths are regenerated automatically.
Batch/image-size/device/worker overrides remain available for memory recovery.

## Run on Linux

Use standard 64-bit Python 3.12 or 3.14 plus venv support and NVIDIA driver 580 or newer.
The optional first command checks prerequisites and installed packages without
installing packages or starting training:

```bash
bash setup_linux.sh --check-only
bash setup_linux.sh
.venv/bin/python scripts/verify_dataset.py --sample-only
.venv/bin/python scripts/train_depth.py --check-only
.venv/bin/python scripts/train_depth.py --epochs 1 --fraction 0.002 --name smoke_20gb
.venv/bin/python scripts/train_depth.py
```

Linux defaults to two loader workers. For a custom Python executable, use
`PYTHON_EXECUTABLE=/path/to/python3.14 bash setup_linux.sh`. For a new environment,
setup tries `python3.14`, then `python3.12`, then a compatible `python3`.

## Evaluate actual metric depth

The standard Ultralytics training metrics use **per-image ground-truth median
alignment**. They measure depth shape, and select/early-stop the checkpoint using
that protocol. They are not direct evidence of correct distance in meters.

After training, Ultralytics fits one global depth-scale calibration using the
validation split and writes it into `best.pt` and `last.pt`. Fresh training resets
the inherited calibration on both the actual training model and EMA. Resuming
preserves the saved state. The native raw head clamps log depth at 5 (about 148.4 m);
scale-invariant `dlam=1.0` training plus global calibration can produce larger
metric outputs. No model architecture or installed Ultralytics source is patched.

Evaluate baseline and trained weights on the same reserved internal test split:

```powershell
.\.venv\Scripts\python.exe scripts\validate_depth.py --model weights\yolo26x-depth.pt --name pretrained --median-aligned
.\.venv\Scripts\python.exe scripts\validate_depth.py --model runs\depth\aerialmetric_yolo26x\weights\best.pt --name finetuned --median-aligned
```

The primary `raw_metric` report uses the saved model scale with **no per-image GT
rescaling** and no test-set calibration. The optional aligned report is labelled
as a diagnostic. RMSE, absolute relative error and delta thresholds are written to
`runs/evaluation/<name>_raw/depth_evaluation.json`. Evaluation is a mean of
per-image scores; invalid pixels and GT >= the selected cap are excluded, and the
underlying metric implementation clamps predictions to the evaluation bounds.
These are internal held-out-location results, not official AerialMetric benchmark
scores or surveyed Skydio person-geolocation accuracy.

## Prepared data format and splits

The converter verifies upstream SHA256 hashes, indexes the split tar as one virtual
file, and reads each pair directly. It avoids a duplicate concatenated tar and a
second extraction of the original numeric arrays. Four workers encode independent
pairs with a bounded queue; archive reads remain sequential. `--workers 1` is
available for a slower computer and produces the same files.

Images are resized, preserving aspect ratio, to a maximum side of **768 pixels**
and saved as JPEG at quality 95. Aligned depths use nearest-neighbor resizing and
source pairs differing by at most one pixel per dimension are first resampled to
the image dimensions, assuming the published pair shares its field of view. This
adjustment is recorded in each receipt; larger discrepancies stop conversion.
Depths are stored as
**16-bit grayscale PNG in centimeters**, with YAML `depth_scale: 100`. Quantization
error is at most 0.005 m for retained depths; these PNGs are numeric labels, not
colored previews. Nonfinite, nonpositive and >= **500 m** source depths become zero.
Values rounding onto the 500 m boundary also become zero. YAML `max_depth: 500`
therefore matches the prepared training range; changing YAML alone does not change
which target pixels the training loss sees.

```
dataset/
  data.yaml                  # relative paths, safe to move
  images/{train,val,test}/   # hash-prefixed flat filenames
  depth/{train,val,test}/    # paired numeric centimeter PNGs
  metadata/{train,val,test}/ # source camera metadata and conversion receipts
  train.txt, val.txt, test.txt
  pairs.csv
  scene_splits.json
  preparation_status.json
  verification.json
```

All pairs come from the authors' `Train-Split` dense archive. Of its 16,561 pairs,
16,365 are marked `train` in the CSV and 196 from `L1` are marked `NAN`; the latter
are retained with that limitation recorded in the audit and per-pair metadata.
Explicit CSV holdouts are refused. Entire scenes are
reserved for internal validation/testing; numbered parts of a named location are
kept together conservatively. Count-weighted defaults target 15% of images for
validation and 15% for test, using seed 2026, while keeping the largest location
group in training. Achieved fractions are approximate because groups are indivisible.
Scene names are a grouping proxy, not a surveyed proof of spatial separation.
Hash-first filenames interleave scenes when Ultralytics sorts files, reducing bias
in its first-200-image global calibration sample. The smoke `--fraction` selects a
sorted training prefix; it is only a pipeline/memory check.

## Re-download or rebuild

Use the Python executable from the created environment:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --remove-archives
.\.venv\Scripts\python.exe scripts\verify_dataset.py
```

Downloads resume after interruption and verify exact SHA256 hashes at pinned
Hugging Face revision `8ad32c3b83f48c89fc8eb11eb5cc7f9c024ca788`. At least 56 GB
plus conversion output/headroom is needed while rebuilding. `--remove-archives`
removes only this bundle's two verified source tar parts after complete conversion;
prepared data and source checksum records remain. Reusing different preparation
settings in the same dataset is refused to avoid mixed labels/splits.

`download_dataset.py --all-oblique` also fetches the other Oblique source archives
if you have enough disk space. The included converter handles the dense archive;
the extra sparse/FOV sources need separate matching format/mask preparation before
they can be mixed into this dataset. The full prepared dense subset is sufficient
to run the commands above.

## Research terms and references

AerialMetric Oblique data are for academic research/education under the upstream
and original source-provider terms. Read `licenses/AerialMetric_DATA_LICENSE.md`
and `licenses/AerialMetric_THIRD_PARTY_LICENSES.md`; website/code licenses do not
relicense the images. Cite the AerialMetric paper and applicable source datasets.
Ultralytics licensing also applies to the training software/model.

- [AerialMetric paper](https://arxiv.org/abs/2606.29716)
- [AerialMetric data organization](https://github.com/kuieless/AerialMetric/blob/main/DATA_ORGANIZATION.md)
- [Ultralytics depth training](https://docs.ultralytics.com/tasks/depth)
- [Ultralytics depth format](https://docs.ultralytics.com/datasets/depth)
- [PyTorch installation and driver/runtime guidance](https://pytorch.org/get-started/locally/)
