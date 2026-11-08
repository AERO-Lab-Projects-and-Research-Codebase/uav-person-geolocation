# UniDepth V2 fine-tune on aerial depth (AerialMetric + SkyScenes)

Fine-tunes the depth head of UniDepth V2 (ViT-L, the model the paper evaluates) on two aerial sets at once, so the
model sees close, middle and long range together:

| Set | Kind | Depth it covers (share of labelled pixels) |
|---|---|---|
| AerialMetric Oblique | real | 20-50 m 37%, 150-500 m 55%; under 20 m 0.5%, 50-150 m 7% (training split sample) |
| SkyScenes, clear noon | synthetic (CARLA) | under 20 m 27%, 20-50 m 41%, 50-150 m 29% (one town) |

## What is trained

Only `pixel_decoder.depth_layer` (38 M of 354 M parameters). The encoder and the camera branch are frozen, so the
camera the model predicts when none is given is unchanged. The true camera rays are fed in during training. The
loss is the model's own scale-aware depth loss (SILog, settings from its `config.json`). Every batch takes the same
number of images from each set. The authors' released training script supports V1 only, so `scripts/train.py` is a
small loop around `UniDepthV2.encode_decode`.

## Steps

All commands run from this folder with the depth environment (`..\.venvs\depth-comparison\Scripts\python.exe` here).

```
python scripts/download_skyscenes.py      # about 65 GB: 3 towns x 12 height/pitch settings, images + depth + metadata
python scripts/prepare_skyscenes.py       # writes dataset/skyscenes in the AerialMetric layout; Town10HD is validation
python -m unittest discover -s tests
python scripts/train.py --check-only      # two steps, validation, save and reload
python scripts/train.py --iters 0         # scores the untouched model on both validation sets
python scripts/train.py --name aerial_mix # the real run; best checkpoint in runs/aerial_mix/best
```

To score a checkpoint with the paper's pipeline, pass its folder to the existing comparison:
`python reference_validation/run_depth_comparison.py --models unidepth_v2 --unidepth-model unidepth_v2_finetune/runs/aerial_mix/best --output <new folder>`.
Judge it on the lab test and the second-UAV flight (references good to about 1-2 m), not on the field set.

## Training on the bigger GPU machine

Copy these to the same relative places: `unidepth_v2_finetune/` (with `dataset/skyscenes`, without `downloads/`),
`aerialmetric_yolo26x_training/dataset/`, `external/UniDepth/` and `models/unidepth_v2/`. Other locations can be
given with `--aerialmetric`, `--skyscenes`, `--source` and `--weights`. Packages: see `requirements.txt`.

Measured on this laptop (RTX 5060, 8 GB): the check run peaks at 4.8 GB with 504 x 756 images and two images per
set. On a larger GPU raise `--batch`; the defaults (20,000 steps, learning rate 1e-5 with warm-up and cosine decay)
are a starting point, not a tuned recipe.

## What is known before training

- Untouched UniDepth V2, given the true camera, already reads 0.90 of the label on AerialMetric's validation
  images (300 images: absolute relative error 0.135, 81.5% within 1.25x; 0.95 at 20-50 m, 0.89 at 50-150 m).
  On the paper's field photos it reads about 0.67 of the reference. So AerialMetric alone can only explain a small
  part of the field shortfall, and a fine-tune may not remove it.
- On SkyScenes' validation town (400 images) the untouched model reads 0.83 of the label (absolute relative
  error 0.181, 59.5% within 1.25x; 0.90 under 20 m, 0.82 at 20-150 m), so there is more to correct there.
- The three downloaded towns give 2,484 frames with depth (69 per town and setting): 1,656 for training
  (Town03, Town07) and 828 for validation (Town10HD), against 13,578 AerialMetric training images. With equal
  shares per batch each SkyScenes frame is drawn about eight times as often. The other five towns would add
  4,140 clear-noon frames.
- SkyScenes depth is an 8-bit grey image in CARLA's logarithmic encoding: one grey level is about 2.2% of the depth.
  The decoding and the 110 degree field of view were checked against flat ground in the data (fitted focal length
  725-760 px against 756 px expected).
- SkyScenes scenes contain pedestrians and vehicles (about 360 walkers per frame in the metadata).

## Limits

- The confidence output is not trained and may no longer match the depth it accompanies.
- Depth nearer than 3.4 m and beyond 500 m has no label in SkyScenes as prepared here.
- No ground-level images are mixed in; the lab test is the only guard on ground-level accuracy.
- UniDepth's code is CC BY-NC 4.0 and SkyScenes is MIT; check AerialMetric's terms before sharing weights.
