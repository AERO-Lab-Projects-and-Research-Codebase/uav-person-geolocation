"""Independent check of the Skydio camera heading: project each person's feet and shadow tip onto the launch
plane with the photo's own pose, and compare the shadow bearing with the solar shadow azimuth at capture time.
A positive result means the metadata heading reads that many degrees clockwise of true.
The mean bias of a photo is then applied as a yaw correction (every estimate turned about the drone by -bias) and the
position error against the nadir truth is printed before and after, from outputs/people.csv and truth.csv.

Pixel picks (full-resolution x, y) were read by eye from crops; expect +-2 deg at 10 m and +-3 deg at 50 m.
Usage: python Skydio_Flight_Experiment/check_heading_from_shadows.py
"""
import csv
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from compare_to_ground_truth import local_enu  # noqa: E402
from metadata_geometry import calibrated_ray, read_capture  # noqa: E402

# photo -> (capture UTC, {person: (feet pixel, shadow tip pixel)})
PICKS = {
    "10_m_Alt_Flight.JPG": {1: ((1060, 4570), (2120, 4060)), 2: ((3740, 3660), (4400, 3340)), 3: ((5880, 4050), (6490, 3650))},
    "30_m_Alt_Flight.JPG": {1: ((2834, 4330), (3180, 4167)), 2: ((3779, 3968), (4057, 3800)), 3: ((4587, 4162), (4750, 4005))},
    "50_m_Alt_Flight.JPG": {1: ((3290, 4167), (3462, 4057)), 2: ((3829, 3946), (3988, 3855)), 3: ((4304, 4063), (4440, 3966))},
}


def sun_azimuth(lat, lon, when):
    """NOAA solar position (degrees clockwise from true north) and elevation."""
    jd = when.timestamp() / 86400 + 2440587.5
    t = (jd - 2451545) / 36525
    L = (280.46646 + 36000.76983 * t) % 360
    M = math.radians(357.52911 + 35999.05029 * t)
    C = (1.914602 - .004817 * t) * math.sin(M) + .019993 * math.sin(2 * M) + .000289 * math.sin(3 * M)
    lam = math.radians(L + C - .00569 - .00478 * math.sin(math.radians(125.04 - 1934.136 * t)))
    eps = math.radians(23.439291 - .0130042 * t)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    y, e = math.tan(eps / 2) ** 2, .016708
    eot = 4 * math.degrees(y * math.sin(2 * math.radians(L)) - 2 * e * math.sin(M) + 4 * e * y * math.sin(M) * math.cos(2 * math.radians(L))
                           - .5 * y * y * math.sin(4 * math.radians(L)) - 1.25 * e * e * math.sin(2 * M))
    minutes = (when.hour * 60 + when.minute + when.second / 60 + eot + 4 * lon) % 1440
    ha = minutes / 4 - 180
    phi = math.radians(lat)
    zen = math.acos(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(math.radians(ha)))
    from_south = math.degrees(math.acos(max(-1., min(1., (math.sin(phi) * math.cos(zen) - math.sin(dec)) / (math.cos(phi) * math.sin(zen))))))
    return (from_south + 180) % 360 if ha > 0 else (540 - from_south) % 360, 90 - math.degrees(zen)


def corrected_errors(name, lat, lon, bias_deg, model="unidepth", methods=("ground_plane", "ground_scaled", "fusion")):
    """{method: (mean error, mean error after turning each estimate about the drone by -bias_deg)} for one photo."""
    truth = {r["person"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader((HERE / "outputs" / "truth.csv").open())}
    rows = [r for r in csv.DictReader((HERE / "outputs" / "people.csv").open()) if r["image"] == name and r["model"] == model]
    c, s = math.cos(math.radians(bias_deg)), math.sin(math.radians(bias_deg))
    out = {}
    for m in methods:
        before, after = [], []
        for r in rows:
            n, e = local_enu(lat, lon, float(r[f"{m}_lat"]), float(r[f"{m}_lon"]))
            tn, te = local_enu(lat, lon, *truth[r["person"]])
            before.append(math.hypot(n - tn, e - te))
            after.append(math.hypot(n * c + e * s - tn, e * c - n * s - te))
        out[m] = statistics.fmean(before), statistics.fmean(after)
    return out


def main():
    for name, picks in PICKS.items():
        cap = read_capture(HERE / "Skydio" / name)
        R, t = np.array(cap["camera_rdf_to_ned"]), cap["telemetry"]
        height = t["altitude_hae"] - cap["assumed_launch_ground_hae_m"]
        when = datetime.strptime(cap["capture_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        az, el = sun_azimuth(t["latitude"], t["longitude"], when)
        shadow_az = (az + 180) % 360
        print(f"{name}: {when:%H:%M:%S} UTC, sun azimuth {az:.1f} elevation {el:.1f} -> shadows point {shadow_az:.1f} deg; "
              f"metadata heading {t['sensor_azimuth']:.1f}")

        def ground(px):
            d = R @ calibrated_ray(px, cap, (cap["height"], cap["width"]))
            return d[:2] * height / d[2]  # north, east on the launch plane
        bias = []
        for person, (feet, tip) in picks.items():
            v = ground(tip) - ground(feet)
            b = math.degrees(math.atan2(v[1], v[0])) % 360
            bias.append((b - shadow_az + 180) % 360 - 180)
            print(f"   person {person}: shadow bearing {b:.1f} deg, length {np.hypot(*v):.2f} m -> heading bias {bias[-1]:+.1f} deg")
        mean = sum(bias) / len(bias)
        print(f"   mean heading bias {mean:+.1f} deg")
        for m, (before, after) in corrected_errors(name, t["latitude"], t["longitude"], mean).items():
            print(f"   {m}: position error {before:.2f} m -> {after:.2f} m with the yaw correction")


if __name__ == "__main__":
    main()
