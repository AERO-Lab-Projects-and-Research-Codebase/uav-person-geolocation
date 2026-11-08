import unittest

from person_geolocator import ROOT
from person_geolocator.metadata import FIELDS
from person_geolocator.server import create_app

DJI = ROOT / "geolocation_experiment_image" / "Images" / "Human_15m.JPG"


def stub(path, model, pose):
    return {"model": model, "pose_seen": pose, "people": [], "annotated_jpeg": b"abc"}


@unittest.skipUnless(DJI.exists(), "lab photograph not distributed")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(stub).test_client()

    def post(self, url, data):
        with DJI.open("rb") as f:
            return self.client.post(url, data={"file": (f, "x.jpg"), **data}, content_type="multipart/form-data")

    def test_page_and_models(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("unidepth", self.client.get("/models").get_json())

    def test_metadata_route(self):
        body = self.post("/metadata", {}).get_json()
        self.assertEqual(body["source"], "dji")
        self.assertEqual(set(body["fields"]), set(FIELDS))
        self.assertNotIn("capture", body)

    def test_locate_passes_pose_and_encodes_image(self):
        response = self.post("/locate", {"model": "unidepth", "lat": "34.5", "lon": "", "heading_deg": "10"})
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["pose_seen"]["lat"], 34.5)
        self.assertIsNone(body["pose_seen"]["lon"])
        self.assertEqual(body["annotated_jpeg_base64"], "YWJj")
        self.assertNotIn("annotated_jpeg", body)

    def test_rejects_bad_input(self):
        self.assertEqual(self.post("/locate", {"model": "nope"}).status_code, 400)
        self.assertEqual(self.post("/locate", {"model": "unidepth", "lat": "abc"}).status_code, 400)
        self.assertEqual(self.client.post("/locate", data={"model": "unidepth"}).status_code, 400)
        with DJI.open("rb") as f:
            r = self.client.post("/metadata", data={"file": (f, "x.png")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
