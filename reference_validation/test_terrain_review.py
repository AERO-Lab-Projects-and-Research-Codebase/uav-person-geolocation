"""Terrain results join only when source, detector run and boxes all match."""
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from review_depth_models import attach_terrain_ground


class TerrainJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"terrain_photos"
        (self.run/"results").mkdir(parents=True)
        person = {"index": 1, "bbox": [1, 2, 3, 4], "bbox_original": [4, 8, 12, 16]}
        self.record = {"run_signature": "detector",
                       "source": {"relative_path": "photo.JPG", "sha256": "original", "processed_shape": [100, 200, 3]},
                       "people": [dict(person)]}
        self.saved = {"run_signature": "terrain", "detection_run_signature": "detector",
                      "source": copy.deepcopy(self.record["source"]),
                      "people": [dict(person, status="estimated", latitude=34., longitude=-84.)]}
        self.manifest = {"status": "complete", "failures": [], "run_signature": "terrain",
                         "detector_run_signature": "detector", "vertical_mode": "absolute",
                         "datum_check": {"median_m": 0.}, "terrain": {"metadata": {"source": "USGS"}},
                         "records": [{"source": "photo.JPG", "result_json": "results/photo.json"}]}

    def join(self):
        (self.run/"manifest.json").write_text(json.dumps(self.manifest))
        (self.run/"results"/"photo.json").write_text(json.dumps(self.saved))
        return attach_terrain_ground([self.record], self.run, self.root)

    def test_verified_join(self):
        info = self.join()
        self.assertEqual(info["status"], "complete")
        self.assertEqual(info["summary"], {"images": 1, "detections": 1})
        self.assertEqual(self.record["people"][0]["terrain_ground"]["latitude"], 34.)

    def test_missing_run_is_unavailable(self):
        info = attach_terrain_ground([self.record], self.root/"absent", self.root)
        self.assertEqual(info["status"], "unavailable")
        self.assertNotIn("terrain_ground", self.record["people"][0])

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.join()
        self.assertNotIn("terrain_ground", self.record["people"][0])

    def test_source_hash_mismatch_rejected(self):
        self.saved["source"]["sha256"] = "changed"
        self.assert_rejected("Source SHA mismatch")

    def test_detector_signature_mismatch_rejected(self):
        self.saved["detection_run_signature"] = "other"
        self.assert_rejected("detector run signature mismatch")

    def test_box_mismatch_rejected(self):
        self.saved["people"][0]["bbox"] = [9, 9, 9, 9]
        self.assert_rejected("box alignment mismatch")

    def test_incomplete_run_rejected(self):
        self.manifest["status"] = "failed"
        self.assert_rejected("not complete")

    def test_result_for_unknown_photo_rejected(self):
        self.manifest["records"][0]["source"] = "other.JPG"
        self.assert_rejected("no matching reviewed photo")


import build_review


class TerrainPageTemplateTests(unittest.TestCase):
    def test_terrain_method_is_added_without_touching_ground_plane(self):
        html = build_review.HTML
        self.assertIn("{key:'ground',label:'Ground plane',cls:'ground',color:'#d9700f'}", html)
        self.assertIn("key:'terrain',label:'Ground (terrain map)'", html)
        self.assertIn("m.key==='terrain'?p.terrain_ground", html)
        self.assertIn('data-opt="terrain"', html)
        self.assertIn("'terrain_position_error_m'", html)
        self.assertIn("D.terrain?.status==='complete'", html)  # hidden when no terrain run

    def test_terrain_column_in_detection_table(self):
        html = build_review.HTML
        self.assertIn("function terrainCell(", html)
        self.assertIn("m.key==='terrain'?terrainCell(p)", html)
        self.assertIn("grazing", html[html.index("function terrainCell("):])
        self.assertIn('colspan="${METHODS.length+4}"', html)  # header count follows METHODS
        self.assertNotIn('colspan="10"', html)
        self.assertIn("insertAdjacentHTML('beforeend','<th>Ground (terrain map)", html)


class GrazingMarkingTests(unittest.TestCase):
    def test_grazing_hits_are_marked_in_result_rows_and_on_the_map(self):
        html = build_review.HTML
        self.assertIn("m.key==='terrain'&&p.terrain_ground?.grazing===true?m.label+' (grazing)':m.label", html)
        self.assertIn("grazing:m.key==='terrain'&&v.grazing===true", html)  # plot points carry the flag
        marker = html[html.index("const tip="):html.index("svg+=`<g data-method=")]
        self.assertIn("grazing — range unreliable", marker)  # tooltip note
        self.assertIn("q.grazing?`fill=\"#fff\"", marker)  # hollow marker, coloured stroke
        self.assertIn("fill=\"${q.grazing?PC[q.kind]:'#fff'}\"", html)  # number stays readable on white

    def test_ground_plane_marker_and_label_are_unchanged(self):
        html = build_review.HTML
        self.assertIn("{key:'ground',label:'Ground plane',cls:'ground',color:'#d9700f'}", html)
        self.assertIn("ground:(x,y,s)=>`<path d=\"M${x},${y-s*1.35}L${x+s*1.3}", html)  # triangle in SHAPES


class TerrainRunSelectionTests(unittest.TestCase):
    def test_none_skips_terrain_without_reading_any_run(self):
        def must_not_attach(*args, **kwargs):
            raise AssertionError("attach_terrain_ground must not be called")

        for value in ("none", "NONE", "None"):
            with self.subTest(value=value), mock.patch.object(build_review, "attach_terrain_ground", must_not_attach):
                info = build_review.terrain_payload([], value, "photos", Path("."))
            self.assertEqual(info, {"status": "unavailable", "reason": "Terrain disabled by --terrain-run none."})

    def test_default_and_explicit_names_resolve_under_outputs(self):
        calls = []
        fake = lambda records, folder, root: calls.append(Path(folder)) or {"status": "complete"}
        with mock.patch.object(build_review, "attach_terrain_ground", fake):
            build_review.terrain_payload([], None, "photos", Path("root"))
            build_review.terrain_payload([], "terrain_other", "photos", Path("root"))
        self.assertEqual(calls, [Path("root")/"outputs"/"terrain_photos", Path("root")/"outputs"/"terrain_other"])


if __name__ == "__main__":
    unittest.main()
