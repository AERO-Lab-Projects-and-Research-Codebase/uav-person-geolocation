"""Download pinned official inference weights without changing live models."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
UNIDEPTH_REPO = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REVISION = "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"
DEPTH_PRO_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_unidepth():
    from huggingface_hub import snapshot_download
    folder = ROOT / "models" / "unidepth_v2"
    snapshot_download(UNIDEPTH_REPO, revision=UNIDEPTH_REVISION,
                      allow_patterns=["config.json", "model.safetensors"], local_dir=str(folder))
    return {"model": "unidepth_v2", "repository": UNIDEPTH_REPO,
            "revision": UNIDEPTH_REVISION,
            "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
                      for path in (folder / "config.json", folder / "model.safetensors")]}


def download_depthpro():
    import requests
    path = ROOT / "models" / "depth_pro.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".pt.part")
    if not path.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(DEPTH_PRO_URL, stream=True, headers=headers, timeout=(30, 120)) as response:
            response.raise_for_status()
            resumed = offset > 0 and response.status_code == 206
            expected = int(response.headers["Content-Length"]) + (offset if resumed else 0)
            with partial.open("ab" if resumed else "wb") as handle:
                last = time.monotonic()
                for chunk in response.iter_content(8 * 1024 * 1024):
                    handle.write(chunk)
                    if time.monotonic() - last > 20:
                        print(json.dumps({"event": "download", "model": "depth_pro",
                                          "bytes": handle.tell(), "expected_bytes": expected}), flush=True)
                        last = time.monotonic()
            if partial.stat().st_size != expected:
                raise OSError("Incomplete Depth Pro checkpoint download")
        partial.replace(path)
    return {"model": "depth_pro", "url": DEPTH_PRO_URL,
            "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}]}


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download_unidepth), pool.submit(download_depthpro)]
        records = [future.result() for future in futures]
    destination = ROOT / "models" / "depth_model_manifest.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"event": "weights_ready", "manifest": str(destination), "models": records}), flush=True)


if __name__ == "__main__":
    main()
