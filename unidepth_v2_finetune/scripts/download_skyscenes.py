"""Downloads the SkyScenes clear-noon archives (Images, Depth, metaData) for the chosen towns.

SkyScenes (hoffman-lab/SkyScenes on Hugging Face, MIT licence) has depth only for the ClearNoon weather.
Each town is one tar.gz per height/pitch setting; an interrupted file resumes from its .partial.

Usage (from unidepth_v2_finetune):  python scripts/download_skyscenes.py [--towns Town03 Town07 Town10HD]
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/hoffman-lab/SkyScenes/resolve/main"
COMBOS = [f"H_{h}_P_{p}" for h in (15, 35, 60) for p in (0, 45, 60, 90)]
KINDS = ("metaData", "Depth", "Images")  # small ones first, so a broken link shows up early
OUT = Path(__file__).resolve().parent.parent / "downloads" / "skyscenes"


def fetch(url, dest):
    part = dest.with_name(dest.name + ".partial")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request) as response:
        resumed = have and response.status == 206
        expected = (have if resumed else 0) + int(response.headers["Content-Length"])
        with open(part, "ab" if resumed else "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
    if part.stat().st_size != expected:
        raise IOError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}")
    part.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", nargs="+", default=["Town03", "Town07", "Town10HD"])
    args = ap.parse_args()
    for kind in KINDS:
        for town in args.towns:
            for combo in COMBOS:
                dest = OUT / kind / combo / f"{town}.tar.gz"
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                fetch(f"{BASE}/{kind}/{combo}/ClearNoon/{town}/{town}.tar.gz", dest)
                print(f"{kind}/{combo}/{town}  {dest.stat().st_size / 1e6:.0f} MB", flush=True)
    print("done")


if __name__ == "__main__":
    main()
