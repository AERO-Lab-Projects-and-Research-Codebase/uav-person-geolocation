"""Fine-tunes the depth head of UniDepth V2 on AerialMetric and SkyScenes together.

Only pixel_decoder.depth_layer is trained. The encoder and the camera branch stay frozen, so the camera the
model predicts when none is given is unchanged, and the true camera rays are fed in during training, as they are
when the pipeline runs on photos with calibrated intrinsics. The loss is the model's own depth loss (scale-aware
SILog) with the settings stored in its config. Every batch takes the same number of images from each set.

The best checkpoint (lowest mean absolute relative error over the two validation sets, no per-image rescaling)
is saved with save_pretrained, so the existing UniDepth V2 backend loads it by passing the folder as model_id.

Usage (from unidepth_v2_finetune):
    python scripts/train.py --check-only          # two steps, a validation pass and a save/reload, then exit
    python scripts/train.py --name aerial_mix     # the real run
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ftdata import PairDataset, mixed_batches, unit_rays  # noqa: E402


def depth_metrics(pred, target):
    """Per image: absolute relative error, share within 1.25x, median estimate/label ratio, median label depth (m)."""
    out = []
    for p, t in zip(pred, target):
        valid = t > 0
        if valid.any():
            ratio = p[valid] / t[valid]
            out.append(((ratio - 1).abs().mean().item(), (torch.maximum(ratio, 1 / ratio) < 1.25).float().mean().item(),
                        ratio.median().item(), t[valid].median().item()))
    return out


def predict(model, batch, amp):
    image = batch["image"].to(model.device, non_blocking=True)
    rays = unit_rays(batch["params"].to(model.device), *image.shape[-2:])
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        _, outputs = model.encode_decode({"image": image, "rays": rays}, [])
    return outputs["depth"].float()


@torch.no_grad()
def validate(model, loaders, amp):
    scores = {}
    for name, loader in loaders.items():
        rows = [m for batch in loader for m in depth_metrics(predict(model, batch, amp), batch["depth"].to(model.device))]
        median = lambda values: sorted(values)[len(values) // 2] if values else None  # noqa: E731
        scores[name] = {"abs_rel": sum(r[0] for r in rows) / len(rows), "delta1": sum(r[1] for r in rows) / len(rows),
                        "median_ratio": median([r[2] for r in rows]), "images": len(rows)}
        for label, lo, hi in (("under_20m", 0, 20), ("20_to_50m", 20, 50), ("50_to_150m", 50, 150), ("over_150m", 150, 1e9)):
            band = [r[2] for r in rows if lo <= r[3] < hi]      # images grouped by their median label depth
            scores[name][f"ratio_{label}"] = median(band)
            scores[name][f"images_{label}"] = len(band)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aerialmetric", default=HERE.parent / "aerialmetric_yolo26x_training" / "dataset", type=Path)
    ap.add_argument("--skyscenes", default=HERE / "dataset" / "skyscenes", type=Path)
    ap.add_argument("--weights", default=str(HERE.parent / "models" / "unidepth_v2"), help="folder or Hugging Face id")
    ap.add_argument("--source", default=HERE.parent / "external" / "UniDepth", type=Path, help="official UniDepth clone")
    ap.add_argument("--name", default="aerial_mix")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2, help="images per set in each batch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--shape", type=int, nargs=2, default=(504, 756), help="height width, multiples of 14")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-images", type=int, default=200, help="per set")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    assert args.shape[0] % 14 == 0 and args.shape[1] % 14 == 0, "shape must be multiples of 14"
    if args.check_only:
        args.iters, args.val_every, args.val_images, args.name = 2, 2, 4, "check"

    sys.path.insert(0, str(args.source))
    from unidepth.models import UniDepthV2

    roots = {name: root for name, root in (("aerialmetric", args.aerialmetric), ("skyscenes", args.skyscenes))
             if (root / "train.txt").exists()}
    if len(roots) < 2 and args.iters > 0 and not args.check_only:   # --iters 0 only scores the pretrained model
        sys.exit(f"Both training sets are needed; found only {list(roots)}. Run scripts/prepare_skyscenes.py first.")
    print("training sets:", {k: str(v) for k, v in roots.items()})
    make = lambda ds, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=args.batch, shuffle=shuffle, drop_last=shuffle, num_workers=args.workers, pin_memory=True)
    train = [make(PairDataset(root, "train", args.shape, train=True, limit=8 if args.check_only else None), True)
             for root in roots.values()]
    val = {name: make(PairDataset(root, "val", args.shape, train=False, limit=args.val_images), False)
           for name, root in roots.items()}

    amp = not args.no_amp
    model = UniDepthV2.from_pretrained(args.weights).to("cuda").eval()   # eval throughout: the head has no dropout to switch on
    for p in model.parameters():
        p.requires_grad_(False)
    head = list(model.pixel_decoder.depth_layer.parameters())
    for p in head:
        p.requires_grad_(True)
    print(f"trainable parameters: {sum(p.numel() for p in head) / 1e6:.1f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    optimizer = torch.optim.AdamW(head, lr=args.lr, weight_decay=0.1)
    warmup = min(500, args.iters // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: (i + 1) / warmup if i < warmup else
                                                 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warmup) / max(1, args.iters - warmup))))
    scaler = torch.amp.GradScaler(enabled=amp)
    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1), encoding="utf-8")

    def record(step, extra):
        scores = validate(model, val, amp)
        mean_abs_rel = sum(s["abs_rel"] for s in scores.values()) / len(scores)
        with open(out / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "mean_abs_rel": mean_abs_rel, **extra, "val": scores}) + "\n")
        print(step, f"mean abs rel {mean_abs_rel:.4f}", {k: {m: v if v is None else round(v, 3) for m, v in s.items()} for k, s in scores.items()}, flush=True)
        return mean_abs_rel

    best = record(0, {"note": "pretrained"})
    before = [p.detach().clone() for p in head] if args.check_only else []
    skipped = 0
    started, running = time.time(), 0.0
    for step, batch in zip(range(1, args.iters + 1), mixed_batches(train)):
        target = batch["depth"].to("cuda", non_blocking=True)
        pred = predict(model, batch, amp)
        no_scale_invariance = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)   # keep the metric scale term
        mask = target > 0
        # unlabelled pixels get a dummy depth of 1 m: the loss takes log(target) before masking, and log(0) poisons the gradient
        loss = model.losses["depth"](pred, target=torch.where(mask, target, torch.ones_like(target)), mask=mask,
                                     si=no_scale_invariance).mean()
        assert torch.isfinite(loss), f"loss is {loss.item()} at step {step}"
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # a non-finite gradient makes the scaler skip the step and lower its scale; normal for the first step or two
        skipped += not torch.isfinite(torch.nn.utils.clip_grad_norm_(head, 1.0))
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        running += loss.item()
        if step % 50 == 0:
            print(f"step {step}  loss {running / 50:.4f}  {(time.time() - started) / step:.2f} s/step  skipped steps so far {skipped}", flush=True)
            running = 0.0
        if step % args.val_every == 0 or step == args.iters:
            score = record(step, {"lr": schedule.get_last_lr()[0]})
            model.save_pretrained(out / "last")
            if score < best:
                best = score
                model.save_pretrained(out / "best")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; best mean abs rel {best:.4f}")
    if args.check_only:
        reloaded = UniDepthV2.from_pretrained(str(out / "last")).to("cuda").eval()
        probe = next(iter(next(iter(val.values()))))
        with torch.no_grad():
            diff = (predict(reloaded, probe, amp) - predict(model, probe, amp)).abs().max().item()
        assert diff < 1e-3, f"reloaded checkpoint differs by {diff} m"
        assert any(not torch.equal(b, p) for b, p in zip(before, head)), "the depth head did not change: every step was skipped"
        print(f"steps skipped for a non-finite gradient: {skipped} of {args.iters}")
        print("check passed: trained, validated, saved and reloaded")


if __name__ == "__main__":
    main()
