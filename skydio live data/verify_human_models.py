"""Offline real-weight verification. Never opens RTSP, a camera, or CoT sockets."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import torch

from human_pipeline import HumanPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/human_smoke'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('--image must be an existing local file')
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error('image could not be decoded')
    args.output.mkdir(parents=True, exist_ok=True)
    worker = HumanPipeline(lambda: None, lambda _: {},
                           {'device': args.device, 'log_dir': None})
    worker._load_models()
    # Synthetic pose, only to test zero-origin arithmetic. It is deliberately
    # not presented as telemetry matched to this saved image.
    telemetry = {
        'uid': 'OFFLINE_FIXTURE', 'event_type': 'a-f-A-M-H-Q',
        'latitude': 0.0, 'longitude': 0.0, 'altitude_hae': 17.162,
        'sensor_hfov': 78.1517279927704, 'sensor_vfov': 62.7019903543633,
        'sensor_azimuth': 90.0, 'sensor_elevation': 0.0, 'sensor_roll': 0.0,
    }
    records = []
    for index in range(3):
        now = time.time()
        telemetry['_received_at'] = now
        result, annotated = worker._process_frame(frame, index, now, dict(telemetry), 0.0,
            'OFFLINE TEST: synthetic pose at 0,0; no matched live telemetry or accuracy validation.')
        result['validation_mode'] = 'offline_image_synthetic_telemetry'
        records.append(result)
    label = 'OFFLINE TEST | synthetic pose and zero origin | no accuracy validation'
    cv2.rectangle(annotated, (0, max(0, frame.shape[0] - 30)),
                  (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
    cv2.putText(annotated, label, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (30, 215, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / 'annotated.jpg'), annotated)
    for index, result in enumerate(records):
        (args.output / f'result_{index}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    metadata = {
        'validation_mode': 'offline_image_synthetic_telemetry',
        'source': str(args.image.resolve()),
        'source_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'image_shape': list(frame.shape),
        'torch': torch.__version__, 'ultralytics': version('ultralytics'),
        'device': worker._device,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
        'inference_ms': [r['inference_ms'] for r in records],
        'people_count': [len(r['people']) for r in records],
        'model_files': {Path(worker.config[k]).name: hashlib.sha256(Path(worker.config[k]).read_bytes()).hexdigest()
                        for k in ('detector_weights', 'depth_weights')},
        'depth_calibration': worker._calibration,
        'limitations': ['Single saved image repeated; not a video benchmark',
                       'Synthetic pose was not recorded with the image',
                       'No ground-truth person depth or GPS; accuracy unmeasured'],
    }
    (args.output / 'verification.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
