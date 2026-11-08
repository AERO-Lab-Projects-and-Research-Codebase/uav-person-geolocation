"""Flask front end: python -m person_geolocator.server [--port 8000]."""
import argparse
import base64
import threading
import uuid

from flask import Flask, jsonify, request, send_file

from person_geolocator import ROOT
from person_geolocator.metadata import FIELDS, read_pose
from person_geolocator.models import MODELS

PAGE = ROOT / "person_geolocator" / "index.html"
UPLOADS = ROOT / "person_geolocator" / "outputs" / "uploads"


def _save_upload():
    file = request.files.get("file")
    if file is None or not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise ValueError("Upload a .jpg or .jpeg image")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{uuid.uuid4().hex}.jpg"
    file.save(path)
    return path


def _pose_from_form():
    pose = {}
    for key in FIELDS:
        text = request.form.get(key, "").strip()
        try:
            pose[key] = float(text) if text else None
        except ValueError:
            raise ValueError(f"{key} must be a number")
    return pose


def create_app(locate_fn=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    lock = threading.Lock()
    state = {}

    def default_locate(path, model, pose):
        from person_geolocator.engine import locate
        from person_geolocator.models import DepthModels
        state.setdefault("models", DepthModels())
        return locate(path, model, pose, state["models"])

    run = locate_fn or default_locate

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(Exception)
    def failure(error):
        if hasattr(error, "code"):  # werkzeug HTTP errors keep their own status
            return jsonify(error=str(error)), error.code
        return jsonify(error=f"{type(error).__name__}: {error}"), 500

    @app.get("/")
    def page():
        return send_file(PAGE)

    @app.get("/models")
    def models():
        return jsonify(MODELS)

    @app.post("/metadata")
    def metadata():
        pose = read_pose(_save_upload())
        pose.pop("capture")
        return jsonify(pose)

    @app.post("/locate")
    def locate_route():
        model = request.form.get("model")
        if model not in MODELS:
            raise ValueError("Choose one of: " + ", ".join(MODELS))
        pose = _pose_from_form()
        path = _save_upload()
        with lock:  # one GPU job at a time
            result = run(path, model, pose)
        jpeg = result.pop("annotated_jpeg", None)
        if jpeg is not None:
            result["annotated_jpeg_base64"] = base64.b64encode(jpeg).decode()
        return jsonify(result)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Open http://localhost:{args.port}  (first run per model is slow while weights load)")
    create_app().run(host="127.0.0.1", port=args.port, threaded=True)
