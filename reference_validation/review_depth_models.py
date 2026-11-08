"""Join saved depth estimates to the original detector records for review."""
import json
import os
from pathlib import Path


def attach_depth_models(records, comparison_root, review_root):
    """Attach verified model people/links; return backend availability/provenance."""
    comparison_root, review_root = Path(comparison_root), Path(review_root)
    metadata = {}
    for backend in ('unidepth_v2', 'depth_pro', 'lingbot_depth'):
        directory = comparison_root / backend
        if not directory.exists():
            metadata[backend] = {'status': 'unavailable', 'reason': 'Saved model directory is missing.'}
            continue
        def fail(message):
            raise ValueError(f'{backend}: {message}')
        def read(path):
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                fail(f'Cannot read saved JSON {path}: {exc}')
        def link(name):
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():
                fail(f'Missing or invalid saved artifact: {name}')
            return Path(os.path.relpath(path, review_root)).as_posix()
        manifest = read(directory / 'manifest.json')
        if manifest.get('backend') != backend:
            fail('Manifest backend does not match its directory.')
        if manifest.get('status') != 'complete' or not manifest.get('run_signature'):
            fail('Saved model manifest is not a complete signed run.')
        entries = {}
        for entry in manifest.get('records', []):
            source = entry['source']
            if source in entries:
                fail(f'Duplicate source {source}.')
            entries[source] = entry
        staged = []
        for record in records:
            source = record['source']['relative_path']
            if source not in entries:
                fail(f'Missing result for source {source}.')
            entry = entries[source]
            saved = read(directory / entry['result_json'])
            if saved.get('run_signature') != manifest['run_signature']:
                fail(f'Model run signature mismatch for {source}.')
            if saved.get('source', {}).get('relative_path') != source:
                fail(f'Source name mismatch for {source}.')
            sha = record['source'].get('sha256')
            if not sha or saved['source'].get('sha256') != sha or manifest.get('input_hashes', {}).get(source) != sha:
                fail(f'Source SHA mismatch for {source}.')
            shape = record['source'].get('processed_shape')
            if not shape or saved['source'].get('processed_shape') != shape:
                fail(f'Processed image shape mismatch for {source}.')
            signature = record.get('run_signature')
            if not signature or manifest.get('detector_run_signature') != signature or saved.get('detection_provenance', {}).get('run_signature') != signature:
                fail(f'Original detector run signature mismatch for {source}.')
            people = {p['index']: p for p in saved['people']}
            expected = {p['index'] for p in record['people']}
            if len(people) != len(saved['people']) or set(people) != expected or len(expected) != len(record['people']):
                fail(f'Detection index alignment mismatch for {source}.')
            for person in record['people']:
                match = people[person['index']]
                if not person.get('bbox') or person['bbox'] != match.get('bbox') or person.get('bbox_original') != match.get('bbox_original'):
                    fail(f'Detection box alignment mismatch for {source} #{person["index"]}.')
            links = {name: link(entry[name]) for name in ('annotated_image', 'depth_preview', 'result_json')}
            staged.append((record, people, links))
        for record, people, links in staged:
            record.setdefault('review_depth_models', {})[backend] = links
            for person in record['people']:
                person.setdefault('depth_models', {})[backend] = people[person['index']]
        metadata[backend] = {'status': manifest.get('status'), 'run_signature': manifest.get('run_signature'),
                             'summary': manifest.get('summary'), 'manifest': link('manifest.json')}
    return metadata


def attach_finetuned_model(records, run_directory, review_root, model='finetuned'):
    """Attach the fine-tuned YOLO-depth results to the photos they were run on.

    The fine-tuned run covers only a subset of the reviewed photos. Photos outside
    that subset are left without a result; photos inside it must match exactly.
    """
    run_directory, review_root = Path(run_directory), Path(review_root)
    directory = run_directory / model
    if not directory.exists():
        return {'status': 'unavailable', 'reason': 'Saved fine-tuned run directory is missing.'}

    def fail(message):
        raise ValueError(f'{model}: {message}')

    def read(path):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            fail(f'Cannot read saved JSON {path}: {exc}')

    def link(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            fail(f'Missing or invalid saved artifact: {name}')
        return Path(os.path.relpath(path, review_root)).as_posix()

    run = read(run_directory / 'manifest.json')
    manifest = read(directory / 'manifest.json')
    if run.get('status') != 'complete' or run.get('failures') or manifest.get('status') != 'complete' or manifest.get('failures'):
        fail('Saved fine-tuned run is not complete or reports failures.')
    if not manifest.get('run_signature') or not run.get('detector_run_signature'):
        fail('Saved fine-tuned run is not signed.')
    by_source = {record['source']['relative_path']: record for record in records}
    entries = {}
    for entry in manifest.get('records', []):
        source = entry['source']
        if source in entries:
            fail(f'Duplicate source {source}.')
        if source not in by_source:
            fail(f'Saved result for {source} has no matching reviewed photo.')
        entries[source] = entry
    staged = []
    for source, entry in entries.items():
        record = by_source[source]
        saved = read(directory / entry['result_json'])
        if saved.get('run_signature') != manifest['run_signature']:
            fail(f'Model run signature mismatch for {source}.')
        if saved.get('source', {}).get('relative_path') != source:
            fail(f'Source name mismatch for {source}.')
        sha = record['source'].get('sha256')
        if not sha or saved['source'].get('sha256') != sha or run.get('input_hashes', {}).get(source) != sha:
            fail(f'Source SHA mismatch for {source}.')
        shape = record['source'].get('processed_shape')
        if not shape or saved['source'].get('processed_shape') != shape:
            fail(f'Processed image shape mismatch for {source}.')
        signature = record.get('run_signature')
        if not signature or run['detector_run_signature'] != signature or saved.get('detection_provenance', {}).get('run_signature') != signature:
            fail(f'Original detector run signature mismatch for {source}.')
        people = {p['index']: p for p in saved['people']}
        expected = {p['index'] for p in record['people']}
        if len(people) != len(saved['people']) or set(people) != expected or len(expected) != len(record['people']):
            fail(f'Detection index alignment mismatch for {source}.')
        for person in record['people']:
            match = people[person['index']]
            if not person.get('bbox') or person['bbox'] != match.get('bbox') or person.get('bbox_original') != match.get('bbox_original'):
                fail(f'Detection box alignment mismatch for {source} #{person["index"]}.')
        links = {name: link(entry[name]) for name in ('annotated_image', 'depth_preview', 'result_json')}
        staged.append((record, people, links))
    for record, people, links in staged:
        record.setdefault('review_depth_models', {})[model] = links
        for person in record['people']:
            person.setdefault('depth_models', {})[model] = people[person['index']]
    calibration = manifest.get('calibration') or {}
    return {'status': manifest['status'], 'run_signature': manifest['run_signature'], 'label': manifest.get('label'),
            'weights_sha256': manifest.get('sha256'), 'calibration': calibration.get('status'),
            'summary': {'images': len(staged), 'detections': sum(len(r['people']) for r, _, _ in staged),
                        'selection_policy': run.get('selection_policy')},
            'manifest': link('manifest.json')}


def attach_terrain_ground(records, run_directory, review_root):
    """Attach terrain-map ground estimates from a separate saved run, if one exists.

    The flat-plane ``ground_plane`` estimate is never changed; terrain results go
    to ``person['terrain_ground']`` only after every photo in the run verifies.
    """
    run_directory = Path(run_directory)
    if not (run_directory / 'manifest.json').is_file():
        return {'status': 'unavailable', 'reason': 'Saved terrain run is missing.'}

    def fail(message):
        raise ValueError(f'terrain: {message}')

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(run_directory.resolve()):
            fail(f'Saved file outside the terrain run: {path}')
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            fail(f'Cannot read saved JSON {path}: {exc}')

    manifest = read(run_directory / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('failures'):
        fail('Saved terrain run is not complete or reports failures.')
    if not manifest.get('run_signature') or not manifest.get('detector_run_signature'):
        fail('Saved terrain run is not signed.')
    by_source = {record['source']['relative_path']: record for record in records}
    staged, seen = [], set()
    for entry in manifest.get('records', []):
        source = entry['source']
        if source in seen:
            fail(f'Duplicate source {source}.')
        seen.add(source)
        record = by_source.get(source)
        if record is None:
            fail(f'Saved terrain result for {source} has no matching reviewed photo.')
        saved = read(run_directory / entry['result_json'])
        if saved.get('run_signature') != manifest['run_signature']:
            fail(f'Terrain run signature mismatch for {source}.')
        if saved.get('source', {}).get('relative_path') != source:
            fail(f'Source name mismatch for {source}.')
        sha = record['source'].get('sha256')
        if not sha or saved['source'].get('sha256') != sha:
            fail(f'Source SHA mismatch for {source}.')
        shape = record['source'].get('processed_shape')
        if not shape or saved['source'].get('processed_shape') != shape:
            fail(f'Processed image shape mismatch for {source}.')
        signature = record.get('run_signature')
        if not signature or manifest['detector_run_signature'] != signature or saved.get('detection_run_signature') != signature:
            fail(f'Original detector run signature mismatch for {source}.')
        people = {p['index']: p for p in saved['people']}
        expected = {p['index'] for p in record['people']}
        if len(people) != len(saved['people']) or set(people) != expected or len(expected) != len(record['people']):
            fail(f'Detection index alignment mismatch for {source}.')
        for person in record['people']:
            match = people[person['index']]
            if not person.get('bbox') or person['bbox'] != match.get('bbox') or person.get('bbox_original') != match.get('bbox_original'):
                fail(f'Detection box alignment mismatch for {source} #{person["index"]}.')
        staged.append((record, people))
    for record, people in staged:
        for person in record['people']:
            person['terrain_ground'] = people[person['index']]
    return {'status': 'complete', 'run_signature': manifest['run_signature'],
            'vertical_mode': manifest.get('vertical_mode'), 'datum_check': manifest.get('datum_check'),
            'terrain_source': (manifest.get('terrain') or {}).get('metadata', {}).get('source'),
            'summary': {'images': len(staged), 'detections': sum(len(r['people']) for r, _ in staged)}}
