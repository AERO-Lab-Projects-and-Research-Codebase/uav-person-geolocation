"""Geolocate people in DJI photos, taking each photo's pose from the DJI Fly flight log instead of
the photo's own metadata (for photos whose GPS/gimbal metadata is missing, stripped or suspect).

Convert the log first:  dji-log --api-key KEY --csv flight.csv DJIFlightRecord_....txt
Then:  python -m person_geolocator.photos PHOTO_DIR --log flight.csv --out outputs/photos_run
"""
import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from person_geolocator.metadata import read_pose
from person_geolocator.models import MODELS
from person_geolocator.video import DEFAULT_HFOV, ROW_FIELDS, default_locate, load_log, people_rows, pose_at, recordings


def photo_time(path, utc_offset=None):
    """EXIF DateTimeOriginal as unix seconds, or None. The camera writes local time; it is read in
    utc_offset hours, or this computer's time zone when None."""
    with Image.open(path) as image:
        exif = image.getexif().get_ifd(0x8769)
    stamp = exif.get(0x9003)
    if not stamp:
        return None
    when = datetime.strptime(stamp.strip(), "%Y:%m:%d %H:%M:%S")
    sub = str(exif.get(0x9291, "")).strip()  # SubSecTimeOriginal
    if sub.isdigit():
        when += timedelta(seconds=float("0." + sub))
    if utc_offset is not None:
        when = when.replace(tzinfo=timezone(timedelta(hours=utc_offset)))
    return when.timestamp()


def match_times(times, shutters, snap):
    """(log time, source) per photo: its EXIF time snapped to the nearest shutter event within `snap` s.
    If any photo has no EXIF time, all photos take the shutter events in order, provided the counts match."""
    if any(t is None for t in times):
        if len(times) != len(shutters):
            raise ValueError(f"{sum(t is None for t in times)} photo(s) have no capture time, and the log has "
                             f"{len(shutters)} shutter events for {len(times)} photos, so they cannot be paired")
        return [(s, "log order") for s in shutters]
    matched = []
    for t in times:
        nearest = min(shutters, key=lambda s: abs(s - t), default=None)
        matched.append((nearest, "shutter") if nearest is not None and abs(nearest - t) <= snap else (t, "exif"))
    return matched


def main(argv=None, locate_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path, help="JPEG files, or folders of them")
    ap.add_argument("--log", type=Path, required=True, help="CSV written by `dji-log --csv`")
    ap.add_argument("--model", choices=list(MODELS), default="unidepth")
    ap.add_argument("--hfov", type=float, help=f"horizontal field of view, degrees; default: from EXIF, else {DEFAULT_HFOV}")
    ap.add_argument("--utc-offset", type=float,
                    help="hours the camera clock is ahead of UTC (e.g. -4 for EDT); default: this computer's zone")
    ap.add_argument("--snap", type=float, default=0.0,
                    help="max seconds to move an EXIF time onto a log shutter event; off by default because the "
                         "Mini SE log misses shots and lags EXIF by ~2 s")
    ap.add_argument("--out", type=Path, default=Path("person_geolocator/outputs/photos"))
    args = ap.parse_args(argv)

    paths = sorted(p for x in args.photos for p in (x.glob("*.[jJ][pP][gG]") if x.is_dir() else [x]))
    if not paths:
        raise SystemExit("No photos found")
    log = load_log(args.log)
    shutters = [start for start, _ in recordings(log, "photo")]
    matched = match_times([photo_time(p, args.utc_offset) for p in paths], shutters, args.snap)
    print(f"{len(paths)} photo(s), {len(shutters)} shutter event(s) in the log")

    locate_fn = locate_fn or default_locate()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "people.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["photo", "time_source"] + ROW_FIELDS, restval="")
        writer.writeheader()
        for path, (t, source) in zip(paths, matched):
            if not log["t"][0] - 1 <= t <= log["t"][-1] + 1:
                print(f"{path.name}: taken outside the log ({datetime.fromtimestamp(t, timezone.utc):%H:%M:%S} UTC); "
                      "skipped, check --utc-offset")
                continue
            hfov = args.hfov or read_pose(path)["fields"]["hfov_deg"] or DEFAULT_HFOV
            pose = pose_at(log, t, hfov)
            result = locate_fn(path, args.model, pose)
            jpeg = result.pop("annotated_jpeg", None)
            if jpeg:
                (args.out / f"{path.stem}_annotated.jpg").write_bytes(jpeg)
            (args.out / f"{path.stem}.json").write_text(json.dumps(
                dict(result, photo=path.name, utc=datetime.fromtimestamp(t, timezone.utc).isoformat(),
                     time_source=source, pose=pose), indent=1))
            print(f"{path.name}: {len(result['people'])} person(s), time from {source}")
            for row in people_rows(t, pose, result):
                writer.writerow(dict(row, photo=path.name, time_source=source))
    print(f"wrote {args.out / 'people.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
