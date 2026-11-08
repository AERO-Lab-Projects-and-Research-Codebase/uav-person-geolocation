"""Render a saved offline verification result. No RTSP, CoT or flight connection."""
import argparse
import copy
import json
import time
from pathlib import Path

import cv2

import main as dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=Path('output/human_smoke/lab'))
    parser.add_argument('--port', type=int, default=5051)
    args = parser.parse_args()
    result = json.loads((args.result_dir / 'result_2.json').read_text(encoding='utf-8'))
    frame = cv2.imread(str(args.result_dir / 'annotated.jpg'))
    if frame is None:
        parser.error('saved annotated image is missing')

    class SavedVideo:
        def get_frame(self):
            return frame.copy()

        def get_status(self):
            return {'connected': False, 'fps': 0, 'resolution': f'{frame.shape[1]}x{frame.shape[0]}', 'frames': 1}

    class SavedInference:
        def get_result(self):
            data = copy.deepcopy(result)
            data.update(status='offline_fixture', stale=True,
                        result_age_s=max(0, time.time()-data['captured_at']))
            return data

        def get_frame(self):
            return frame.copy()

    dashboard.video_capture = SavedVideo()
    dashboard.human_pipeline = SavedInference()
    dashboard.cot_receiver = None
    dashboard.csv_logger = None
    dashboard.DASHBOARD_HTML = dashboard.DASHBOARD_HTML.replace(
        'SKYDIO X10', 'OFFLINE PREVIEW').replace(
        'GROUND STATION', 'SAVED IMAGE / SYNTHETIC POSE')
    dashboard.app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
