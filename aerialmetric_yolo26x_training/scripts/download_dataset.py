#!/usr/bin/env python3
"""Download pinned AerialMetric assets with resume and SHA256 verification.

This script does not extract archives, install packages, or start training.
Its default selection is the complete dense Oblique training archive plus CSV.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
STATUS_PATH = DOWNLOADS / "download_status.json"
VERIFIED_PATH = DOWNLOADS / "verified_files.json"
REPOSITORY = "Kuiee/AerialMetric-ECCV2026"
REVISION = "8ad32c3b83f48c89fc8eb11eb5cc7f9c024ca788"
PREFIX = "Train-Split/AerialMetric-Oblique/"
CHUNK_BYTES = 8 * 1024 * 1024
HEADROOM_BYTES = 8 * 1024**3

DEFAULT_ASSETS = [
    {
        "name": "TrainingData_Final_MoGe-All.tar.part-aa",
        "size": 48318382080,
        "sha256": "f36eee2e0400ad5208be4a46b2e86f327973a2a9cdef60240ebcb59a4c020eb3",
    },
    {
        "name": "TrainingData_Final_MoGe-All.tar.part-ab",
        "size": 7369646080,
        "sha256": "4b94470507fb246f2ff8a5238cd4cd147712b502dce1cac1c1795756dce19f62",
    },
    {
        "name": "final_merged.csv",
        "size": 21421131,
        "sha256": "4bbcbe0a200cc1b52aced8c3b85dff813df32d444f3c85064e42ec951afc565b",
    },
]
EXTRA_NAMES = {"B.tar.part-aa", "B.tar.part-ab", "B-2.tar", "FOV.tar"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(value: int | float) -> str:
    return f"{value / 1024**3:.2f} GiB"


def save_status(status: dict) -> None:
    status["updated_at_utc"] = utc_now()
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def load_status() -> dict:
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if value.get("repository") == REPOSITORY and value.get("revision") == REVISION:
            return value
    except (OSError, ValueError, AttributeError):
        pass
    return {"repository": REPOSITORY, "revision": REVISION, "files": {}}


def record_verified_file(path: Path, asset: dict) -> None:
    """Publish the simple integrity ledger shared with dataset preparation."""
    try:
        ledger = json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            ledger = {}
    except (OSError, ValueError):
        ledger = {}
    info = path.stat()
    ledger[asset["name"]] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": asset["sha256"]}
    temporary = VERIFIED_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(VERIFIED_PATH)


def checksum(path: Path, announce: bool = True) -> str:
    digest = hashlib.sha256()
    last_update = time.monotonic()
    consumed = 0
    total = path.stat().st_size
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
            consumed += len(block)
            if announce and time.monotonic() - last_update >= 10:
                print(f"Verifying {path.name}: {consumed / max(total, 1):.1%}", flush=True)
                last_update = time.monotonic()
    return digest.hexdigest()


def asset_url(asset: dict) -> str:
    remote_path = quote(PREFIX + asset["name"], safe="/")
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote_path}?download=true"


def extra_assets(session: requests.Session, retries: int) -> list[dict]:
    """Read extra filenames, lengths, and LFS SHA256 IDs at the pinned revision."""
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{PREFIX.rstrip('/')}"
    payload = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params={"recursive": "false", "expand": "true"}, timeout=(30, 90))
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(f"Cannot fetch pinned extra-asset metadata: {error}") from error
            time.sleep(min(2**attempt, 30))
    found = []
    for item in payload:
        name = item.get("path", "").rsplit("/", 1)[-1]
        if name not in EXTRA_NAMES:
            continue
        lfs = item.get("lfs", {})
        digest = lfs.get("oid", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Pinned metadata has no SHA256 for {name}")
        found.append({"name": name, "size": int(item["size"]), "sha256": digest})
    if {item["name"] for item in found} != EXTRA_NAMES:
        raise RuntimeError("Pinned HF metadata is missing one or more additional Oblique archives")
    return sorted(found, key=lambda item: item["name"])


def mark_file(status: dict, asset: dict, state: str, **extra) -> None:
    status["files"][asset["name"]] = {
        "remote_path": PREFIX + asset["name"],
        "expected_bytes": asset["size"],
        "expected_sha256": asset["sha256"],
        "state": state,
        **extra,
    }
    save_status(status)


def cached_verification(path: Path, asset: dict, status: dict) -> bool:
    """Trust a prior completed hash only while the file size and mtime match."""
    entry = status["files"].get(asset["name"], {})
    info = path.stat()
    return (
        entry.get("state") == "verified"
        and entry.get("expected_sha256") == asset["sha256"]
        and entry.get("actual_sha256") == asset["sha256"]
        and info.st_size == asset["size"]
        and entry.get("file_mtime_ns") == info.st_mtime_ns
    )


def verify_file(path: Path, asset: dict, status: dict) -> None:
    if not path.is_file():
        mark_file(status, asset, "missing", downloaded_bytes=0)
        raise RuntimeError(f"Missing file: {path}")
    actual_size = path.stat().st_size
    if actual_size != asset["size"]:
        mark_file(status, asset, "invalid_size", downloaded_bytes=actual_size)
        raise RuntimeError(f"Incorrect size for {path.name}: {actual_size}, expected {asset['size']}")
    mark_file(status, asset, "verifying", downloaded_bytes=actual_size)
    digest = checksum(path)
    if digest != asset["sha256"]:
        mark_file(status, asset, "invalid_sha256", downloaded_bytes=actual_size, actual_sha256=digest)
        raise RuntimeError(f"SHA256 mismatch for {path.name}: {digest}, expected {asset['sha256']}")
    mark_file(
        status, asset, "verified", downloaded_bytes=actual_size,
        actual_sha256=digest, file_mtime_ns=path.stat().st_mtime_ns,
    )
    if path.name == asset["name"]:
        record_verified_file(path, asset)
    print(f"Verified {path.name} ({human_bytes(actual_size)})", flush=True)


def check_disk_space(assets: list[dict]) -> None:
    remaining = 0
    for asset in assets:
        final = DOWNLOADS / asset["name"]
        partial = DOWNLOADS / (asset["name"] + ".partial")
        present = final.stat().st_size if final.is_file() else partial.stat().st_size if partial.is_file() else 0
        if present > asset["size"]:
            raise RuntimeError(f"Oversized download file for {asset['name']}; inspect or remove it before retrying")
        remaining += max(0, asset["size"] - present)
    free = shutil.disk_usage(DOWNLOADS).free
    needed = remaining + HEADROOM_BYTES
    if free < needed:
        raise RuntimeError(
            f"Not enough free space: {human_bytes(free)} available; require {human_bytes(needed)} "
            f"({human_bytes(remaining)} remaining downloads + 8 GiB headroom). "
            "Extraction/preparation require additional space separately."
        )
    print(f"Download space check: {human_bytes(free)} free, {human_bytes(remaining)} remaining + 8 GiB headroom", flush=True)


def transfer_once(session: requests.Session, asset: dict, partial: Path, status: dict) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == asset["size"]:
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with session.get(asset_url(asset), headers=headers, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise RuntimeError(f"Missing/invalid Content-Range for {asset['name']}")
            start, end, total = map(int, match.groups())
            if start != offset or total != asset["size"] or end < start or end >= total:
                raise RuntimeError(f"Unexpected Content-Range for {asset['name']}: {response.headers.get('Content-Range')}")
            if "Content-Length" in response.headers and int(response.headers["Content-Length"]) != end - start + 1:
                raise RuntimeError(f"Content-Length disagrees with Content-Range for {asset['name']}")
            mode = "ab" if offset else "wb"
        elif response.status_code == 200:
            if offset:
                print(f"WARNING: server ignored Range for {asset['name']}; restarting only its .partial file", flush=True)
                offset = 0
            if "Content-Length" in response.headers and int(response.headers["Content-Length"]) != asset["size"]:
                raise RuntimeError(f"Unexpected Content-Length for {asset['name']}")
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP {response.status_code} for {asset['name']}")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise RuntimeError(f"Server returned encoded archive bytes for {asset['name']}")
        started = time.monotonic()
        last_update = started
        last_bytes = offset
        current = offset
        mark_file(status, asset, "downloading", downloaded_bytes=current)
        print(f"Downloading {asset['name']} from {human_bytes(offset)} / {human_bytes(asset['size'])}", flush=True)
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                if current + len(chunk) > asset["size"]:
                    raise RuntimeError(f"Server returned excess bytes for {asset['name']}")
                handle.write(chunk)
                current += len(chunk)
                now = time.monotonic()
                if now - last_update >= 10:
                    handle.flush()
                    speed = (current - last_bytes) / (now - last_update)
                    mark_file(status, asset, "downloading", downloaded_bytes=current, bytes_per_second=round(speed))
                    print(f"{asset['name']}: {current / asset['size']:.1%}, {human_bytes(current)}, {speed / 1024**2:.1f} MiB/s", flush=True)
                    last_update, last_bytes = now, current
        mark_file(status, asset, "downloaded_partial", downloaded_bytes=current)
    if current != asset["size"]:
        raise RuntimeError(f"Download stopped at {current} / {asset['size']} bytes for {asset['name']}")


def download_file(session: requests.Session, asset: dict, status: dict, retries: int) -> None:
    final = DOWNLOADS / asset["name"]
    partial = DOWNLOADS / (asset["name"] + ".partial")
    if final.exists():
        if cached_verification(final, asset, status):
            record_verified_file(final, asset)
            print(f"Already verified: {asset['name']}", flush=True)
            return
        try:
            verify_file(final, asset, status)
        except RuntimeError as error:
            raise RuntimeError(f"Existing final file failed verification. Inspect/remove {final} and rerun. {error}") from error
        return
    for attempt in range(retries + 1):
        try:
            transfer_once(session, asset, partial, status)
            verify_file(partial, asset, status)
            partial.replace(final)
            # The rename preserves contents; publish the final filename's mtime.
            mark_file(status, asset, "verified", downloaded_bytes=asset["size"], actual_sha256=asset["sha256"], file_mtime_ns=final.stat().st_mtime_ns)
            record_verified_file(final, asset)
            return
        except (requests.RequestException, OSError, RuntimeError, ValueError) as error:
            current = partial.stat().st_size if partial.exists() else 0
            mark_file(status, asset, "retrying" if attempt < retries else "failed", downloaded_bytes=current, error=str(error))
            if attempt == retries:
                raise RuntimeError(f"Failed {asset['name']} after {retries + 1} attempts: {error}") from error
            # A complete partial with a bad checksum cannot be resumed. Delete only
            # this script's specifically named partial and fetch clean bytes again.
            if current == asset["size"] and "SHA256 mismatch" in str(error):
                print(f"WARNING: removing corrupt .partial for {asset['name']} and restarting", flush=True)
                partial.unlink()
            delay = min(2**attempt, 30)
            print(f"Retrying {asset['name']} in {delay}s after: {error}", flush=True)
            time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="Hash all selected completed files; do not download them")
    parser.add_argument("--metadata-only", action="store_true", help="Select only final_merged.csv (no image/depth archives)")
    parser.add_argument("--all-oblique", action="store_true", help="Also select all sparse LiDAR and FOV archives; about 161.845 GB total")
    parser.add_argument("--retries", type=int, default=8, help="Retries per file after connection/hash errors (default: 8)")
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    if args.metadata_only and args.all_oblique:
        parser.error("--metadata-only and --all-oblique are mutually exclusive")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    status = load_status()
    with requests.Session() as session:
        session.headers["User-Agent"] = "AerialMetric-YOLO26x-training-bundle/1.0"
        assets = [dict(asset) for asset in DEFAULT_ASSETS if not args.metadata_only or asset["name"].endswith(".csv")]
        if args.all_oblique:
            assets.extend(extra_assets(session, args.retries))
        status["selected_files"] = [asset["name"] for asset in assets]
        status["selected_bytes"] = sum(asset["size"] for asset in assets)
        status["selection"] = "metadata_only" if args.metadata_only else "all_oblique" if args.all_oblique else "dense_oblique"
        status["state"] = "verifying" if args.verify_only else "downloading"
        save_status(status)
        print(f"Pinned dataset revision: {REVISION}\nSelected: {human_bytes(status['selected_bytes'])}", flush=True)
        try:
            if not args.verify_only:
                check_disk_space(assets)
            # Metadata first provides a useful completed asset while larger downloads run.
            for asset in sorted(assets, key=lambda item: (not item["name"].endswith(".csv"), item["name"])):
                if args.verify_only:
                    verify_file(DOWNLOADS / asset["name"], asset, status)
                else:
                    download_file(session, asset, status, args.retries)
            status["state"] = "complete"
            save_status(status)
            print(f"All selected files verified. Saved in {DOWNLOADS}", flush=True)
            return 0
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            save_status(status)
            print("Interrupted. Rerun the same command to resume .partial downloads.", file=sys.stderr)
            return 130
        except Exception as error:
            status["state"] = "failed"
            status["error"] = str(error)
            save_status(status)
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
