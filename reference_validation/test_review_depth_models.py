"""Saved model joins must never pair ground truth with different detections."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from review_depth_models import attach_depth_models, attach_finetuned_model


class SavedModelJoinTests(unittest.TestCase):
    backend = 'unidepth_v2'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'models' / self.backend
        self.directory.mkdir(parents=True)
        self.record = {'run_signature': 'detector', 'source': {'relative_path': 'photo.JPG', 'sha256': 'original', 'processed_shape': [100, 200, 3]},
                       'people': [{'index': 1, 'bbox': [1, 2, 3, 4], 'bbox_original': [4, 8, 12, 16]}]}
        self.saved = copy.deepcopy(self.record)
        self.saved['run_signature'] = 'model'
        self.saved['detection_provenance'] = {'run_signature': 'detector'}
        self.manifest = {'backend': self.backend, 'detector_run_signature': 'detector',
                         'input_hashes': {'photo.JPG': 'original'}, 'status': 'complete', 'run_signature': 'model',
                         'records': [{'source': 'photo.JPG', 'result_json': 'result.json',
                                      'annotated_image': 'annotation.jpg', 'depth_preview': 'preview.jpg'}]}
        for name in ('annotation.jpg', 'preview.jpg'):
            (self.directory / name).write_bytes(b'image')

    def join(self):
        (self.directory / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.directory / 'result.json').write_text(json.dumps(self.saved))
        return attach_depth_models([self.record], self.root / 'models', self.root)

    def test_verified_join_and_missing_optional_backend(self):
        metadata = self.join()
        self.assertEqual(self.record['people'][0]['depth_models'][self.backend]['index'], 1)
        self.assertEqual(self.record['review_depth_models'][self.backend]['result_json'], f'models/{self.backend}/result.json')
        absent = 'depth_pro' if self.backend != 'depth_pro' else 'unidepth_v2'
        self.assertEqual(metadata[absent]['status'], 'unavailable')

    def test_source_hash_mismatch_rejected(self):
        self.saved['source']['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Source SHA mismatch'):
            self.join()

    def test_detector_signature_mismatch_rejected(self):
        self.saved['detection_provenance']['run_signature'] = 'different'
        with self.assertRaisesRegex(ValueError, 'detector run signature mismatch'):
            self.join()

    def test_detection_index_mismatch_rejected(self):
        self.saved['people'][0]['index'] = 2
        with self.assertRaisesRegex(ValueError, 'index alignment mismatch'):
            self.join()

    def test_detection_box_mismatch_rejected(self):
        self.saved['people'][0]['bbox'][0] = 99
        with self.assertRaisesRegex(ValueError, 'box alignment mismatch'):
            self.join()

    def test_incomplete_model_manifest_rejected(self):
        self.manifest['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'not a complete signed run'):
            self.join()

    def test_model_run_signature_mismatch_rejected(self):
        self.saved['run_signature'] = 'other-model-run'
        with self.assertRaisesRegex(ValueError, 'Model run signature mismatch'):
            self.join()

    def test_processed_shape_mismatch_rejected(self):
        self.saved['source']['processed_shape'] = [200, 400, 3]
        with self.assertRaisesRegex(ValueError, 'Processed image shape mismatch'):
            self.join()


class LingBotJoinTests(SavedModelJoinTests):
    """The UniDepth refinement joins and is verified exactly like the other saved models."""
    backend = 'lingbot_depth'


class FinetunedJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.directory = self.run / 'finetuned'
        self.directory.mkdir(parents=True)
        person = {'index': 1, 'bbox': [1, 2, 3, 4], 'bbox_original': [4, 8, 12, 16]}
        self.record = {'run_signature': 'detector', 'source': {'relative_path': 'photo.JPG', 'sha256': 'original', 'processed_shape': [100, 200, 3]},
                       'people': [dict(person)]}
        self.other = {'run_signature': 'detector', 'source': {'relative_path': 'other.JPG', 'sha256': 'other', 'processed_shape': [100, 200, 3]},
                      'people': [dict(person)]}
        self.saved = copy.deepcopy(self.record)
        self.saved['run_signature'] = 'model'
        self.saved['detection_provenance'] = {'run_signature': 'detector'}
        self.run_manifest = {'status': 'complete', 'failures': [], 'detector_run_signature': 'detector',
                             'input_hashes': {'photo.JPG': 'original'}, 'selection_policy': 'subset'}
        self.manifest = {'status': 'complete', 'failures': [], 'run_signature': 'model', 'label': 'Fine-tuned',
                         'records': [{'source': 'photo.JPG', 'result_json': 'result.json',
                                      'annotated_image': 'annotation.jpg', 'depth_preview': 'preview.jpg'}]}
        for name in ('annotation.jpg', 'preview.jpg'):
            (self.directory / name).write_bytes(b'image')

    def join(self):
        (self.run / 'manifest.json').write_text(json.dumps(self.run_manifest))
        (self.directory / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.directory / 'result.json').write_text(json.dumps(self.saved))
        return attach_finetuned_model([self.record, self.other], self.run, self.root)

    def test_subset_join_leaves_other_photos_untouched(self):
        metadata = self.join()
        self.assertEqual(metadata['summary']['images'], 1)
        self.assertEqual(self.record['review_depth_models']['finetuned']['result_json'], 'run/finetuned/result.json')
        self.assertEqual(self.record['people'][0]['depth_models']['finetuned']['index'], 1)
        self.assertNotIn('review_depth_models', self.other)
        self.assertNotIn('depth_models', self.other['people'][0])

    def test_missing_run_is_unavailable(self):
        self.assertEqual(attach_finetuned_model([self.record], self.root / 'absent', self.root)['status'], 'unavailable')

    def test_source_hash_mismatch_rejected(self):
        self.run_manifest['input_hashes']['photo.JPG'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Source SHA mismatch'):
            self.join()

    def test_detector_signature_mismatch_rejected(self):
        self.run_manifest['detector_run_signature'] = 'different'
        with self.assertRaisesRegex(ValueError, 'detector run signature mismatch'):
            self.join()

    def test_detection_box_mismatch_rejected(self):
        self.saved['people'][0]['bbox'][0] = 99
        with self.assertRaisesRegex(ValueError, 'box alignment mismatch'):
            self.join()

    def test_result_for_unreviewed_photo_rejected(self):
        self.manifest['records'][0]['source'] = 'unknown.JPG'
        with self.assertRaisesRegex(ValueError, 'no matching reviewed photo'):
            self.join()

    def test_incomplete_run_rejected(self):
        self.manifest['failures'] = [{'source': 'photo.JPG'}]
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.join()


if __name__ == '__main__':
    unittest.main()
