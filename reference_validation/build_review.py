"""Build a local, dependency-free review page from completed batch outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
from review_depth_models import attach_depth_models, attach_finetuned_model, attach_terrain_ground


def terrain_payload(records, terrain_run, run_name, root):
    """Attach terrain-map ground results, or report them as disabled with --terrain-run none."""
    if terrain_run is not None and terrain_run.lower() == 'none':
        return {'status': 'unavailable', 'reason': 'Terrain disabled by --terrain-run none.'}
    return attach_terrain_ground(records, root / 'outputs' / (terrain_run or f'terrain_{run_name}'), root)


# Extra Mini SE runs, saved as outputs/<run>_<model>, and the page backend each fills (None: YOLO-depth, on the person).
MINISE_MODELS = {'yolo_original': None, 'yolo_finetuned': 'finetuned', 'lingbot': 'lingbot_depth', 'depth_pro': 'depth_pro'}


def minise_estimate(person):
    """One model's depth-ray result for a person in a person_geolocator result."""
    ray = person['methods']['depth_ray']
    return {'status': ray['status'], 'reason': ray['reason'], 'latitude': ray['lat'], 'longitude': ray['lon'],
            'depth_m': person['depth_m'], 'slant_range_m': ray['range_m']}


def minise_records(root):
    """DJI Mini SE flight-test photos (person_geolocator.photos runs listed in minise_flights.json) as extra
    review records, each flight shown as one more sheet. outputs/<run> holds the UniDepth V2 run; the other depth
    models are matched to its detections from outputs/<run>_<model> when those runs exist."""
    flights = json.loads((root / 'minise_flights.json').read_text(encoding='utf-8'))['flights']
    records, sheets, truth = [], [], []
    for f in flights:
        run, rel = root / 'outputs' / f['run'], f"outputs/{f['run']}/"
        if not run.is_dir():
            continue
        sheet = {'sheet_id': f['sheet_id'], 'title': f['title'], 'handwritten_position_label': f['label'], 'note': f['note'],
                 'drone_latitude': f['drone_latitude'], 'drone_longitude': f['drone_longitude'],
                 'source_image': rel + f['source_image'] if f['source_image'] else None}
        sheets.append(sheet)
        truth += [{'sheet_id': str(f['sheet_id']), 'person_name': t['person_name'], 'normalized_latitude': t['latitude'],
                   'normalized_longitude': t['longitude'], 'normalized_latitude_dms': t['latitude_dms'],
                   'normalized_longitude_dms': t['longitude_dms'], 'status': t['status'],
                   'auto_fill': t.get('auto_fill', False)} for t in f['truth']]
        for path in sorted(run.glob('*.json')):
            r = json.loads(path.read_text())
            pose, (width, height) = r['pose'], r['image_size']
            extra = {m: json.loads(q.read_text()) for m in MINISE_MODELS
                     if (q := root / 'outputs' / f"{f['run']}_{m}" / path.name).is_file()}
            people = []
            for p in r['people']:
                ground = p['methods']['ground_plane']
                rng, h = ground['range_m'], pose['height_above_ground_m']
                person = {'index': p['index'], 'confidence': p['confidence'], 'bbox': p['bbox'],
                          'reason': 'YOLO-depth was not run on this photo',
                          'ground_plane': {'status': ground['status'], 'reason': ground['reason'],
                                           'latitude': ground['lat'], 'longitude': ground['lon'], 'ground_range_m': rng,
                                           'height_above_ground_m': h,
                                           'depression_deg': math.degrees(math.asin(min(1, h / rng))) if rng else None},
                          'depth_models': {'unidepth_v2': minise_estimate(p)}}
                for model, result in extra.items():
                    # Same detector and frame, so detections line up; refuse a pairing whose box moved.
                    q = next((q for q in result['people'] if q['index'] == p['index']
                              and max(abs(a - b) for a, b in zip(q['bbox'], p['bbox'])) < 1), None)
                    if q is None:
                        continue
                    if MINISE_MODELS[model] is None:
                        person.update(minise_estimate(q))
                    else:
                        person['depth_models'][MINISE_MODELS[model]] = minise_estimate(q)
                people.append(person)
            models = ', '.join(['UniDepth V2', *extra])
            records.append({
                'source': {'relative_path': f"{f['title']}/{r['photo']}", 'original_shape': [height, width]},
                'capture_metadata': {'camera_source': 'DJI Mini SE', 'photo_kind': 'visible', 'review_kind': 'DJI Mini SE / visible',
                                     'capture_time': r['utc'],
                                     'telemetry': {'latitude': pose['lat'], 'longitude': pose['lon'], 'altitude_hae': pose['altitude_hae'],
                                                   'sensor_azimuth': pose['heading_deg'], 'sensor_hfov': pose['hfov_deg']},
                                     'assumed_launch_ground_hae_m': pose['altitude_hae'] - pose['height_above_ground_m'],
                                     'limitations': ['Mini SE pose from the DJI Fly flight log (aircraft yaw, gimbal pitch, barometric '
                                                     f'height above takeoff); depth models run: {models}.']},
                'sheet': sheet, 'people': people, 'warnings': r.get('geometry_errors', []),
                'review_image': rel + path.stem + '_annotated.jpg', 'review_json': rel + path.name,
                'review_depth_models': {'unidepth_v2': {'annotated_image': rel + path.stem + '_annotated.jpg'},
                                        **{MINISE_MODELS[m]: {'annotated_image': f"outputs/{f['run']}_{m}/{path.stem}_annotated.jpg"}
                                           for m in extra if MINISE_MODELS[m]}},
                'review_links': [['Result JSON (UniDepth V2)', rel + path.name], ['People CSV', rel + 'people.csv']]
                                + [[f'Result JSON ({m})', f"outputs/{f['run']}_{m}/{path.name}"] for m in extra]})
    return records, sheets, truth

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='photos')
    parser.add_argument('--depth-run', default='unidepth_depthpro_latest')
    parser.add_argument('--finetuned-run', default='yolo26x_aerialmetric_four_person_20261005',
                        help='Output folder holding a saved fine-tuned YOLO-depth run (a subset of the photos)')
    parser.add_argument('--terrain-run', default=None,
                        help='Saved terrain-map ground run (default: terrain_<run>), or none to skip terrain; the page works without it')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = root / 'outputs' / args.run
    manifest = json.loads((run / 'manifest.json').read_text())
    gt = json.loads((root / 'ground_truth.json').read_text())
    with (root / 'ground_truth.csv').open(encoding='utf-8-sig', newline='') as f:
        truth = list(csv.DictReader(f))
    records = []
    for entry in manifest['records']:
        r = json.loads((run / entry['result_json']).read_text())
        r['capture_metadata']['review_kind'] = r['capture_metadata']['camera_source'] + ' / ' + r['capture_metadata']['photo_kind']
        r['review_image'] = f'outputs/{args.run}/' + entry['annotated_image']
        r['review_json'] = f'outputs/{args.run}/' + entry['result_json']
        r['sheet'] = next(s for s in gt['sheets']
                          if s['all_jpeg_first'] <= entry['source'] <= s['all_jpeg_last'])
        records.append(r)
    depth_models = attach_depth_models(records, root / 'outputs' / args.depth_run, root)
    depth_models['finetuned'] = attach_finetuned_model(records, root / 'outputs' / args.finetuned_run, root)
    terrain = terrain_payload(records, args.terrain_run, args.run, root)
    extra_records, extra_sheets, extra_truth = minise_records(root)
    records += extra_records
    gt['sheets'] += extra_sheets
    truth += extra_truth
    video_path = root / 'outputs' / 'video_analysis' / 'manifest.json'
    videos = json.loads(video_path.read_text()) if video_path.exists() else None
    payload = json.dumps({'manifest': manifest, 'truth': truth, 'ground_truth': gt,
                          'records': records, 'videos': videos,
                          'depth_models': depth_models,
                          'terrain': terrain}, allow_nan=False).replace('<', '\\u003c')
    page = HTML.replace('outputs/photos/photo_inventory.csv', f'outputs/{args.run}/photo_inventory.csv')
    page = page.replace('__PAYLOAD__', payload)
    dest = root / f'review_{args.run}.html'
    dest.write_text(page, encoding='utf-8')
    print(dest)


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skydio reference data — model review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#182c3d;background:#f3f6f8}*{box-sizing:border-box}body{margin:0}header{background:#17354a;color:white;padding:26px 4vw}h1{font-size:26px;margin:0 0 7px}h2{font-size:19px;margin:0 0 12px}p{line-height:1.55;margin:8px 0}main{max-width:1600px;margin:auto;padding:22px}a{color:#006790}header a{color:#c5eaff}.cards,.controls,.refs{display:flex;gap:12px;flex-wrap:wrap}.card,.panel{background:white;border:1px solid #d8e1e8;border-radius:9px;padding:18px;margin-bottom:16px}.card{flex:1;min-width:160px}.card strong{display:block;font-size:28px}.muted{color:#526779;font-size:13px}.notice{background:#fff4d9;border-left:4px solid #ca8a0a;padding:12px 16px;margin-bottom:18px}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}select,button,input{font:inherit;padding:8px;border:1px solid #abbcc9;border-radius:5px;background:white}button{cursor:pointer}label{font-size:13px;display:grid;gap:5px}.controls{align-items:end;margin-bottom:16px}#photo{width:100%;display:block}#plot{width:100%;background:#fff;border:1px solid #dce4ea;border-radius:4px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid #dfe6eb;padding:9px;vertical-align:top}th{color:#345065}td select{max-width:250px;padding:5px;font-size:12px}.scroll{overflow:auto}.refs figure{margin:0;text-align:center;width:110px}.refs img{height:125px;max-width:110px;object-fit:contain}.refs figcaption{font-size:13px;margin:6px}.legend{display:flex;gap:6px 18px;flex-wrap:wrap}.legend .lg{display:flex;align-items:center;gap:6px;font-size:12.5px;color:#345065;white-space:nowrap}.legend input{margin:0}.dot{font-weight:bold}.blue{color:#1a5fd0}.orange{color:#d9700f}.green{color:#187449}details{margin-top:14px}summary{cursor:pointer;font-weight:600}.small{font-size:12px}@media(max-width:1000px){.grid{grid-template-columns:1fr}main{padding:12px}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:14px 0}.tile{border:1px solid #d8e1e8;border-radius:9px;padding:14px;border-top:5px solid #999}.tile.ref{border-top-color:#187449}.tile.depth{border-top-color:#1a5fd0}.tile.ground{border-top-color:#d9700f}.tile .big{font-size:30px;font-weight:700;display:block}.tile .lbl{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#526779}.tile .verdict{margin-top:8px;font-size:13px;line-height:1.45}.headline{font-size:17px;line-height:1.6}.headline b{white-space:nowrap}
.gbar{display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:center;margin:10px 0}.gbar .name{font-size:13px;font-weight:600}.track{position:relative;height:9px;background:#eef2f5;border-radius:5px;flex:1}.track i{position:absolute;top:0;left:0;height:100%;border-radius:5px}.track .band{background:#bfe3cf;border-radius:0}.track .tick{width:3px;background:#187449;border-radius:0}.rr{display:flex;align-items:center;gap:6px;font-size:12px;margin:3px 0;min-width:280px}.rr .rl{width:84px;color:#526779}.rr .rv{width:120px;text-align:right;white-space:nowrap}.c-ref{background:#187449}.c-depth{background:#1a5fd0}.c-ground{background:#d9700f}.c-terrain{background:#7a5c00}.tile.terrain{border-top-color:#7a5c00}.brown{color:#7a5c00}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600}.chip.ok{background:#d6f0e1;color:#0f5a35}.chip.near{background:#fdebc4;color:#7a4b00}.chip.bad{background:#fbd8d4;color:#8a1f13}.chip.na{background:#e8edf1;color:#526779}.key{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:8px 0}#groupTable th,#groupTable td{white-space:nowrap}#groupTable tr.sub td{background:#f5f8fa;font-weight:600;color:#345065}ol.how{line-height:1.6;padding-left:22px}ol.how li{margin:6px 0}
.cellreal{min-width:240px;display:grid;gap:5px}.coord{width:100%;padding:6px 8px;font-size:13px}.coord.bad{border-color:#d0302b;background:#fff5f4}.echo{font-size:12px;color:#526779;min-height:16px}.echo.err{color:#a3271b}.echo.warn{color:#8a5a00}.tolbox{display:flex;align-items:center;gap:8px;font-size:14px}.tolbox input{width:74px}.emptybox{background:#eef5fa;border:1px dashed #9db8cc;border-radius:8px;padding:16px;margin:10px 0}
.purple{color:#8a4fe0}.teal{color:#0e9478}.red{color:#d0302b}.c-finetuned{background:#d0302b}.tile.finetuned{border-top-color:#d0302b}.magenta{color:#e85bb5}.c-lingbot{background:#e85bb5}.tile.lingbot{border-top-color:#e85bb5}.c-unidepth{background:#8a4fe0}.c-depthpro{background:#0e9478}.tile.unidepth{border-top-color:#8a4fe0}.tile.depthpro{border-top-color:#0e9478}.modelcell{min-width:220px}.modelcell .gps{border-top:1px solid #e1e7ec;padding-top:9px;margin-top:9px}.modelviews{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.modelview{border:1px solid #d8e1e8;border-radius:7px;padding:12px}.modelview img{width:100%;max-height:300px;object-fit:contain;background:#121212}.modelview h3{margin:0 0 8px;font-size:16px}#filterNote{margin-bottom:0}
</style>
<header><h1>Skydio reference data · model review</h1><p>Saved photos · shared YOLO person detections · YOLO-depth, fine-tuned YOLO-depth, UniDepth V2, LingBot refined UniDepth, Depth Pro, ground plane and terrain-map ground · provisional geolocation</p></header>
<main><div class="cards" id="cards"></div>
<section class="panel" id="glance"><h2>Result at a glance: how far off are the position estimates?</h2><p id="progress" class="muted" style="margin-top:0"></p>
<div class="controls" style="margin:6px 0"><label class="tolbox" for="tol">Accuracy circle for real coordinates (± m) <input type="number" id="tol" min="0" max="500" step="1"></label><span class="muted">Applies to every real coordinate, typed or filled from a written sheet. Sheet coordinates are whole arc-seconds, good to about ±20 m.</span></div>
<div id="glanceEmpty" class="emptybox"><strong>No real coordinates entered yet.</strong><br>Scroll to the detection table, pick a photo, and type the real GPS coordinate of each person the model found. This summary, the map and the distance bars all update as you type.</div>
<div id="glanceBody"><p class="headline" id="headline"></p><div class="tiles" id="tiles"></div><div id="bars"></div>
<div class="key" id="methodKey"></div>
<div class="scroll"><table id="groupTable"><thead><tr id="groupHead"></tr></thead><tbody id="groupBody"></tbody></table></div>
<p class="muted">Map and bar distances are horizontal GPS offsets from the camera; the detection table reports axial depth and straight-line slant range separately. <b>Position error</b> is horizontal distance between an estimate and the real coordinate you entered (median). <b>Within accuracy</b> is the share whose position error is no larger than the coordinate accuracy.</p></div>
<details><summary>How was this computed? (plain English)</summary><ol class="how"><li>For every detection you enter the real GPS coordinate of that person.</li><li>The drone's GPS position is the starting point: the one embedded in the photo, or the one you type in the drone GPS box. From your coordinate we work out the horizontal distance and compass direction from the drone to the person.</li><li>Three pretrained models give torso depth and GPS estimates: <b>YOLO-depth</b>, <b>UniDepth V2</b> and <b>Depth Pro</b>. The <b>fine-tuned YOLO-depth</b> model (AerialMetric training) is shown beside YOLO-depth for the photos it was run on. <b>LingBot refined UniDepth</b> takes UniDepth V2's depth map as its input and refines it, so it is not an independent measurement. <b>Ground-plane geometry</b> uses the box-bottom ray and an assumed flat plane at launch height.</li><li>We compare each estimated position with the real coordinate you entered, using the same boxes and reference for every method.</li><li>Nothing is matched or guessed for you: only the detections you enter are counted.</li></ol></details></section>
<div class="notice"><strong>Read these as estimates, not proven accuracy.</strong> Results depend on the coordinates you enter, the drone's embedded GPS and camera pose, and flat-ground and depth assumptions. Coordinates copied from the handwritten sheets are only good to about ±20 m.</div>
<div class="panel"><div class="controls"><label>Camera files<select id="sensor"><option value="all">All photos</option></select></label><label>Sheet / position<select id="sheet"><option value="all">All sheets and flights</option></select></label><label>Photo<select id="file"></select></label><button id="prev">← Previous</button><button id="next">Next →</button></div><div id="meta"></div><p id="filterNote" class="muted"></p></div>
<section class="panel" id="dronebox"><h2>Drone GPS position for this photo</h2><p class="muted" style="margin-top:0" id="droneEmbedded"></p>
<div class="controls"><label>Drone latitude<input class="coord" id="droneLat" style="min-width:230px" placeholder="e.g. 34.38743 or 34°23′15″ N" autocomplete="off"></label><label>Drone longitude<input class="coord" id="droneLon" style="min-width:230px" placeholder="e.g. -84.45471" autocomplete="off"></label><button id="droneSheet">Fill from the sheet’s recorded drone location</button><button id="droneReset">Use the photo’s embedded GPS</button><label style="display:flex;flex-direction:row;align-items:center;gap:6px"><input type="checkbox" id="droneAll"><span id="droneAllLabel">Apply to all photos of this sheet</span></label></div>
<div class="echo" id="droneEcho"></div>
<p class="muted">Leave blank to use the GPS embedded in the photo. The model measures each estimate as an offset from the drone, so entering a different drone position moves the estimates with it. Real distances, bearings and position errors on this page are measured from the drone position you enter.</p></section>
<div class="grid"><section class="panel"><h2 id="imageTitle">Photo detections</h2><a id="imageLink" target="_blank"><img id="photo" alt="Model person detections in the selected source photo"></a><p class="muted">Open the image for a larger view. Each box is labelled with its detection number. The detector uses a 640-pixel input; distant people may be missed.</p></section>
<section class="panel"><h2>Where the estimates landed, and where the people really were</h2><p class="muted" style="margin-top:0">Top-down view, camera at the cross. Dotted rings show distance from the camera. Hover any marker for details.</p><svg id="plot" viewBox="0 0 640 520" font-family="system-ui,-apple-system,Segoe UI,sans-serif" role="img" aria-label="Top-down map of detection estimates, real positions you entered, and the camera"></svg>
<div class="legend" id="legend" style="margin-top:10px"></div>
<p class="small" style="margin:8px 0 0"><button id="saveSvg">Save figure as SVG</button> <button id="savePng">Save figure as PNG (3×)</button> <span class="muted">Legend, caption and camera view cone are drawn inside the figure.</span></p>
<p class="muted" id="plotHint"></p><p class="muted"><b>How to read it:</b> the number inside each marker is the detection number in the table and photo. A line joins each estimate to the real position you entered for that detection, so a short line means an accurate estimate. The dashed circle around a real position is its coordinate accuracy, not measured error. Untick a marker type to hide it.</p></section></div>
<section class="panel"><h2>Detections and their real coordinates</h2><p class="muted">For each detection, type the <b>real GPS coordinate of that person</b>. Decimal degrees (<code>34.38722</code> and <code>-84.45472</code>) or degrees-minutes-seconds (<code>34°23′14″ N</code>) both work; West longitude may be typed without a minus sign. You can also paste “lat, lon” into the latitude box. Leave a detection blank if you can't tell who it is or it isn't a person. Entries stay in this browser and can be exported. Each method column shows its saved depth/range and GPS from embedded capture geometry.</p><div class="scroll"><table><thead><tr><th>Detection</th><th>Real coordinate of this person</th><th>Result<br><span class="muted">bars: horizontal range · tag: horizontal position error</span></th><th>Detector confidence</th><th>YOLO-depth model<br><span class="muted">torso depth, slant and GPS</span></th><th>Fine-tuned YOLO-depth<br><span class="muted">torso depth, slant and GPS · selected photos only</span></th><th>UniDepth V2<br><span class="muted">torso depth, slant and GPS</span></th><th>LingBot refined UniDepth<br><span class="muted">UniDepth seed, torso depth, slant and GPS</span></th><th>Depth Pro<br><span class="muted">torso depth, slant and GPS</span></th><th>Ground plane<br><span class="muted">box-bottom range and GPS</span></th></tr></thead><tbody id="detections"></tbody></table></div>
<div class="key"><span><span class="chip ok">ok</span> position error within the coordinate accuracy</span><span><span class="chip near">near</span> within 30 m (or twice the accuracy)</span><span><span class="chip bad">far</span> beyond that</span><span class="muted">Green tick on the reference bar is the real distance; the light band is the coordinate accuracy.</span></div>
<p><button id="clearPhoto">Clear coordinates for this photo</button> <button id="export">Export entered coordinates and results (CSV)</button> <span class="muted" id="saved"></span></p></section>
<section class="panel"><h2>Fine-tuned YOLO-depth, UniDepth V2, LingBot and Depth Pro images</h2><p class="muted">The same saved detections are shown for each model; the fine-tuned model exists only for the photos it was run on. Expand a depth map to review its full-frame axial-depth prediction.</p><div id="modelViews" class="modelviews"></div></section>
<section class="panel"><h2>Handwritten sheet for this photo group</h2><p class="muted" style="margin-top:0">For reference only. Use “fill from a written sheet coordinate” in the table above if a detection is one of these people; sheet coordinates are whole arc-seconds, about ±20 m.</p><p id="sheetInfo"></p><div class="scroll"><table><thead><tr><th>Person</th><th>Recorded / normalized latitude</th><th>Recorded / normalized longitude</th><th>Transcription status</th></tr></thead><tbody id="truth"></tbody></table></div><details><summary>Person reference images supplied by the user</summary><div id="refs" class="refs"></div></details></section>
<section class="panel"><h2>Video analysis</h2><div id="videos"></div></section>
<section class="panel"><h2>Files and reproducibility</h2><p id="links"></p><p><a href="outputs/photos/photo_inventory.csv">Drone GPS and photo inventory</a> · <a href="README.md">Run report</a> · <a href="ground_truth.csv">Ground-truth CSV</a> · <a href="ground_truth.json">Ground-truth JSON</a> · <a href="ground_truth_notes.md">Transcription notes</a> · <a href="video_inventory.json">Video inspection</a> · <a href="outputs/video_analysis/manifest.json">Video analysis manifest</a></p><details><summary>Assumptions for the selected photo</summary><ul id="limitations"></ul></details><p class="muted">The two thermal files in each capture set are related outputs, not independent field trials. No coordinates were used to fit model depth, tune geometry, or assign names.</p></section>
</main><script>
const D=__PAYLOAD__;
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v)), num=(v,n=1)=>finite(v)?Number(v).toFixed(n):'—';
const METHODS=[{key:'depth',label:'YOLO-depth',cls:'depth',color:'#1a5fd0'},
{key:'finetuned',label:'Fine-tuned YOLO-depth',cls:'finetuned',color:'#d0302b',backend:'finetuned',note:'AerialMetric-calibrated scale'},
{key:'unidepth',label:'UniDepth V2',cls:'unidepth',color:'#8a4fe0',backend:'unidepth_v2'},
{key:'lingbot',label:'LingBot refined UniDepth',cls:'lingbot',color:'#e85bb5',backend:'lingbot_depth',note:'refines the UniDepth V2 depth map'},
{key:'depthpro',label:'Depth Pro',cls:'depthpro',color:'#0e9478',backend:'depth_pro'},
{key:'ground',label:'Ground plane',cls:'ground',color:'#d9700f'},
{key:'terrain',label:'Ground (terrain map)',cls:'terrain',color:'#7a5c00',note:'box bottom on USGS bare-earth terrain'}].filter(m=>m.key!=='terrain'||D.terrain?.status==='complete');
if(D.terrain?.status!=='complete')document.querySelector('[data-opt="terrain"]')?.closest('label')?.remove();
else document.getElementById('detections').closest('table').querySelector('thead tr').insertAdjacentHTML('beforeend','<th>Ground (terrain map)<br><span class="muted">box-bottom horizontal range and GPS · terrain model</span></th>');
const methodValue=(p,m)=>m.key==='depth'?p:m.key==='ground'?p.ground_plane:m.key==='terrain'?p.terrain_ground:p.depth_models?.[m.backend];
function dms(value,axis){if(!finite(value)||typeof value==='boolean')return '—';const n=Number(value),limit=axis==='lat'?90:180;if(Math.abs(n)>limit)return '—';const total=Math.round(Math.abs(n)*360000),degrees=Math.floor(total/360000),minutes=Math.floor((total%360000)/6000),seconds=((total%6000)/100).toFixed(2),negative=n<0||Object.is(n,-0),hemisphere=axis==='lat'?(negative?'S':'N'):(negative?'W':'E');return `${degrees}° ${String(minutes).padStart(2,'0')}′ ${seconds.padStart(5,'0')}″ ${hemisphere}`}
function coordinate(value,axis){return `<span style="white-space:nowrap">${esc(dms(value,axis))}</span><br><span class="muted">${num(value,7)}°</span>`}
const storageKey='skydio-review-'+D.manifest.run_signature;
let real={},tolM=10,subset=[],current=null;
try{real=JSON.parse(localStorage.getItem(storageKey+'-real')||'{}')}catch(e){}
try{tolM=Math.max(0,+JSON.parse(localStorage.getItem(storageKey+'-tol')||'10'))}catch(e){}
if(!Number.isFinite(tolM))tolM=10;
let droneReal={};try{droneReal=JSON.parse(localStorage.getItem(storageKey+'-drone')||'{}')}catch(e){}
function persist(){try{localStorage.setItem(storageKey+'-real',JSON.stringify(real));localStorage.setItem(storageKey+'-tol',JSON.stringify(tolM));localStorage.setItem(storageKey+'-drone',JSON.stringify(droneReal));$('saved').textContent='Saved in this browser.'}catch(e){$('saved').textContent='Export the CSV to keep your entries.'}}
const sum=D.manifest.summary;
$('cards').innerHTML=`<div class="card"><strong>${sum.successful_images} / ${sum.expected_images}</strong>photos processed</div><div class="card"><strong>${sum.detected_people_total}</strong>detections across photos</div><div class="card"><strong>40</strong>capture sets</div><div class="card"><strong>${D.manifest.runtime.cuda_available?"NVIDIA GPU":"CPU"}</strong>compute device · ${esc(D.manifest.status)}</div>`;
for(const kind of [...new Set(D.records.map(r=>r.capture_metadata.review_kind))].sort())$('sensor').add(new Option(kind,kind));
const sheetName=s=>s.title||'Sheet '+s.sheet_id;
for(const s of D.ground_truth.sheets)$('sheet').add(new Option(sheetName(s),s.sheet_id));
for(const p of D.ground_truth.people)$('refs').innerHTML+=`<figure><img src="${esc(p.reference_image)}" alt="${esc(p.name)} user-supplied reference"><figcaption>${esc(p.name)}</figcaption></figure>`;
if(D.videos){$('videos').innerHTML=`<p>${esc(D.videos.status)} · ${num(D.videos.sampling_hz,0)} frame per second analysed · ${D.videos.sampled_frames??'in progress'} samples. Video GPS is unavailable without matched pose telemetry.</p>`+(D.videos.clips||[]).map(c=>`<details><summary>${esc(c.source)} · ${c.sampled_frames} samples · ${c.detections} detections</summary>${c.browser_video?`<video controls preload="metadata" style="width:100%;max-width:900px" src="outputs/video_analysis/${esc(c.browser_video)}"></video>`:`<a href="outputs/video_analysis/${esc(c.annotated_video)}">Annotated video</a>`}<p class="muted">${c.decoded_frames} source frames decoded. <a href="outputs/video_analysis/${esc(c.frames_jsonl)}">Per-frame JSONL</a></p></details>`).join('')}else{$('videos').textContent='See the video analysis manifest for status.'}
/* ---------- geometry and coordinate parsing ---------- */
function offset(lat,lon,olat,olon){const a=6378137,e2=.0066943799901413165,phi=olat*Math.PI/180,q=1-e2*Math.sin(phi)**2;return [(lon-olon)*Math.PI/180*a/Math.sqrt(q)*Math.cos(phi),(lat-olat)*Math.PI/180*a*(1-e2)/q**1.5]}
const bearingOf=(e,n)=>(Math.atan2(e,n)*180/Math.PI+360)%360,angDiff=(a,b)=>Math.abs(((a-b+540)%360)-180);
function parseCoord(text,axis){text=String(text??'').trim();if(!text)return null;const hemi=((text.match(/[NSEWnsew]/g)||[])[0]||'').toUpperCase();if((text.match(/[NSEWnsew]/g)||[]).length>1)return{error:'Use a single N/S/E/W letter'};const name=axis==='lat'?'latitude':'longitude';if(hemi&&!(axis==='lat'?'NS':'EW').includes(hemi))return{error:`${hemi} cannot be a ${name}`};
if(/[^0-9.\s,°'′’"″”\-−–NSEWnsew]/.test(text))return{error:'Not a valid '+name};const nums=text.match(/\d+(?:\.\d+)?/g)||[];if(!nums.length||nums.length>3)return{error:'Not a valid '+name};
const [d,m=0,s=0]=nums.map(Number);if(m>=60||s>=60)return{error:'Minutes and seconds must be under 60'};let v=d+m/60+s/3600;const neg=/^\s*[-−–]/.test(text)||hemi==='S'||hemi==='W';if(neg)v=-v;
let assumed=false;if(axis==='lon'&&!neg&&!hemi&&v>0){v=-v;assumed=true}if(Math.abs(v)>(axis==='lat'?90:180))return{error:name[0].toUpperCase()+name.slice(1)+' out of range'};return{value:v,assumed}}
function getEntry(k){const e=real[k];if(!e)return null;const la=parseCoord(e.lat,'lat'),lo=parseCoord(e.lon,'lon');if(!la&&!lo)return null;if(!la||!lo||la.error||lo.error)return{invalid:true,la,lo};return{lat:la.value,lon:lo.value,src:e.src||'typed',assumedW:lo.assumed}}
const tolOf=e=>tolM,key=(r,p)=>r.source.relative_path+':'+p.index;
// A truth row with auto_fill marks every detection of its sheet as that person (flight 4: the person stood at takeoff).
for(const r of D.records)for(const t of D.truth.filter(t=>t.auto_fill&&+t.sheet_id===r.sheet.sheet_id))for(const p of r.people){const k=key(r,p);if(!real[k])real[k]={lat:(+t.normalized_latitude).toFixed(6),lon:(+t.normalized_longitude).toFixed(6),src:'sheet'}}
function embOrigin(r){const c=r.capture_metadata,t=c.telemetry||{},lat=t.latitude??c.latitude,lon=t.longitude??c.longitude;return finite(lat)&&finite(lon)?[+lat,+lon]:null}
function droneRaw(r){return droneReal[r.source.relative_path]||droneReal['sheet:'+r.sheet.sheet_id]||null}
function droneEntry(r){const raw=droneRaw(r);if(!raw)return null;const la=parseCoord(raw.lat,'lat'),lo=parseCoord(raw.lon,'lon');if(!la||!lo||la.error||lo.error)return null;return{lat:la.value,lon:lo.value,src:raw.src||'typed'}}
function origin(r){const d=droneEntry(r);return d?[d.lat,d.lon]:embOrigin(r)}
function analyze(r,p,e){const o=origin(r),emb=embOrigin(r);if(!o||!emb)return null;const [te,tn]=offset(e.lat,e.lon,o[0],o[1]),ref=Math.hypot(te,tn),brg=bearingOf(te,tn),out={ref,brg,te,tn};
for(const m of METHODS){const v=methodValue(p,m);if(v?.status==='unavailable'||!finite(v?.latitude)||!finite(v?.longitude)){out[m.key]=null;continue}const [ee,nn]=offset(+v.latitude,+v.longitude,emb[0],emb[1]),range=Math.hypot(ee,nn);out[m.key]={e:ee,n:nn,range,rangeErr:range-ref,pos:Math.hypot(ee-te,nn-tn),brgErr:angDiff(bearingOf(ee,nn),brg)}}return out}
/* ---------- summary ---------- */
const isNum=v=>typeof v==='number'&&Number.isFinite(v),median=a=>{a=a.filter(isNum).sort((x,y)=>x-y);if(!a.length)return null;const h=a.length>>1;return a.length%2?a[h]:(a[h-1]+a[h])/2};
const m1=v=>isNum(v)?v.toFixed(1):'—',pct=v=>isNum(v)?Math.round(v*100)+'%':'—';
const KIND={'MAIN_VISIBLE / NORMAL':'Visible camera','INFRARED / NORMAL':'Thermal (normal JPEG)','INFRARED / RJPEG':'Thermal (RJPEG image)'},kindName=k=>KIND[k]||k;
function allRows(){const rows=[];for(const r of D.records)for(const p of r.people){const e=getEntry(key(r,p));if(!e||e.invalid)continue;const a=analyze(r,p,e);if(a)rows.push({...a,tol:tolOf(e),kind:r.capture_metadata.review_kind,sheet:r.sheet.sheet_id})}return rows}
function stats(rows){const methods={};for(const m of METHODS){const valid=rows.filter(r=>r[m.key]);methods[m.key]={n:valid.length,range:median(valid.map(r=>r[m.key].range)),pos:median(valid.map(r=>r[m.key].pos)),within:valid.length?valid.filter(r=>r[m.key].pos<=r.tol).length/valid.length:null,signed:median(valid.map(r=>r[m.key].rangeErr))}}return{n:rows.length,ref:median(rows.map(r=>r.ref)),methods}}
function verdict(pos,signed,within){if(!isNum(pos))return 'No estimate.';const dir=signed<0?'closer to the drone than the real person':'farther from the drone than the real person';return `Lands a median of ${m1(pos)} m from the real position, usually ${dir}. ${pct(within)} of estimates fall within the coordinate accuracy.`}
function glance(){const rows=allRows(),total=D.records.reduce((n,r)=>n+r.people.length,0);$('progress').textContent=`${rows.length} of ${total} detections have a real coordinate entered.`;
$('methodKey').innerHTML='<span><i class="c-ref" style="display:inline-block;width:14px;height:8px"></i> Real horizontal distance</span>'+METHODS.map(m=>`<span><i class="c-${m.cls}" style="display:inline-block;width:14px;height:8px"></i> ${m.label}</span>`).join('');
$('groupHead').innerHTML='<th>Group</th><th>Entered detections</th><th>Real horizontal distance</th>'+METHODS.map(m=>`<th>${m.label}: estimated</th><th>${m.label}: position error</th><th>${m.label}: within accuracy</th>`).join('');
$('glanceEmpty').style.display=rows.length?'none':'block';$('glanceBody').style.display=rows.length?'block':'none';if(!rows.length)return;
const all=stats(rows);$('headline').innerHTML=`Across <b>${rows.length}</b> detection${rows.length>1?'s':''} with a real coordinate, the median horizontal distance from the drone was <b>${m1(all.ref)} m</b>. Median horizontal position errors: `+METHODS.map(m=>`<b>${m.label}</b> ${m1(all.methods[m.key].pos)} m`).join('; ')+'.';
$('tiles').innerHTML=`<div class="tile ref"><span class="lbl">Real horizontal distance from camera</span><span class="big">${m1(all.ref)} m</span><div class="verdict">Median over the ${rows.length} detections you entered.</div></div>`+METHODS.map(m=>{const s=all.methods[m.key];return `<div class="tile ${m.cls}"><span class="lbl">${m.label} · horizontal position error</span><span class="big">${m1(s.pos)} m</span><div class="verdict">Median estimated horizontal distance ${m1(s.range)} m. ${verdict(s.pos,s.signed,s.within)}<br><span class="muted">${s.n} available estimates out of ${rows.length} entered detections.</span></div></div>`}).join('');
const kinds=[...new Set(rows.map(r=>r.kind))].sort(),groups=[['All entered',rows,0],...kinds.map(k=>[kindName(k),rows.filter(r=>r.kind===k),0]),...D.ground_truth.sheets.map(s=>[sheetName(s),rows.filter(r=>r.sheet===s.sheet_id),1]).filter(g=>g[1].length)];
const top=Math.max(...groups.filter(g=>!g[2]).flatMap(([,g])=>{const s=stats(g);return[s.ref,...METHODS.map(m=>s.methods[m.key].range)].filter(isNum)}),1)*1.08;
const bar=(cls,l,v)=>isNum(v)?`<div class="rr"><span class="rl">${l}</span><div class="track"><i class="${cls}" style="width:${Math.min(100,v/top*100)}%"></i></div><span class="rv">${m1(v)} m</span></div>`:'';
$('bars').innerHTML=groups.filter(g=>!g[2]).map(([name,g])=>{const s=stats(g);return `<div class="gbar"><div class="name">${esc(name)}<br><span class="muted">${s.n} detection${s.n>1?'s':''}</span></div><div>${bar('c-ref','Real',s.ref)}${METHODS.map(m=>bar('c-'+m.cls,m.label,s.methods[m.key].range)).join('')}</div></div>`}).join('');
$('groupBody').innerHTML=groups.map(([name,g,sub],i)=>{const s=stats(g);return `<tr${sub&&groups[i-1][2]===0?' class="sub"':''}><td>${esc(name)}</td><td>${s.n}</td><td>${m1(s.ref)} m</td>${METHODS.map(m=>{const v=s.methods[m.key];return `<td>${m1(v.range)} m</td><td>${m1(v.pos)} m</td><td>${pct(v.within)} <span class="muted">(${v.n}/${s.n} available)</span></td>`}).join('')}</tr>`}).join('')}
/* ---------- photo table ---------- */
function filter(){const old=current?.source.relative_path;subset=D.records.filter(r=>($('sensor').value==='all'||r.capture_metadata.review_kind===$('sensor').value)&&($('sheet').value==='all'||r.sheet.sheet_id===$('sheet').value*1));$('file').innerHTML='';for(const r of subset)$('file').add(new Option(`${r.source.relative_path} · ${r.people.length} detection${r.people.length===1?'':'s'}`,r.source.relative_path));if(subset.some(r=>r.source.relative_path===old))$('file').value=old;for(const id of ['file','prev','next'])$(id).disabled=!subset.length;$('filterNote').textContent=`Showing ${subset.length} of ${D.records.length} photos.`;render()}
function depthCell(p,m){return finite(p?.depth_m)?`<div><strong>${num(p.depth_m)} m</strong> depth along the camera axis</div><div class="muted">${num(p.slant_range_m)} m straight-line range</div><div class="muted">Torso point · ${m?.note||'pretrained scale'}${finite(p.depth_confidence)?' · depth confidence '+num(p.depth_confidence,3):''}</div>`:esc(p?.reason||'Saved model result unavailable')}
function terrainCell(p){const t=p.terrain_ground;if(t?.status!=='estimated'||!finite(t.north_m)||!finite(t.east_m))return esc(t?.reason||'No terrain result for this detection');return `<div><strong>${num(Math.hypot(t.north_m,t.east_m))} m</strong> horizontal range to the ground point</div><div class="muted">terrain height ${num(t.ground_hae_m)} m (ellipsoid)</div><div class="muted">incidence angle ${num(t.incidence_deg)}°</div>${t.grazing?'<div class="red"><strong>grazing — range unreliable</strong></div>':''}`}
function groundCell(p){const g=p.ground_plane||{};return finite(g.ground_range_m)?`<div><strong>${num(g.ground_range_m)} m</strong> straight-line range to the ground point</div><div class="muted">camera ${num(g.height_above_ground_m)} m above assumed ground</div><div class="muted">looking down ${num(g.depression_deg)}° · box bottom may not be feet</div>`:esc(g.reason||'Unavailable')}
function coords(p){return finite(p?.latitude)&&finite(p?.longitude)?`<div><strong>Lat</strong> ${coordinate(p.latitude,'lat')}</div><div style="margin-top:7px"><strong>Lon</strong> ${coordinate(p.longitude,'lon')}</div>`:esc(p?.reason||'Unavailable')}
const chipFor=(pos,tv)=>pos<=tv?'ok':pos<=Math.max(30,2*tv)?'near':'bad',chipWord={ok:'ok',near:'near',bad:'far'};
function resultCell(r,p){const e=getEntry(key(r,p));if(!e)return '<span class="muted">Enter this person’s real coordinate to see how far off the estimates are.</span>';if(e.invalid)return '<span class="muted">Fix the coordinate to see the result.</span>';const a=analyze(r,p,e);if(!a)return '<span class="muted">Camera GPS unavailable for this photo.</span>';const tv=tolOf(e),top=Math.max(a.ref+tv,...METHODS.map(m=>a[m.key]?.range||0),1)*1.05,w=v=>Math.max(0,Math.min(100,v/top*100)),lo=Math.max(0,a.ref-tv);
const est=(label,cls,x)=>x?`<div class="rr" title="Range error ${x.rangeErr>0?'+':''}${m1(x.rangeErr)} m · direction error ${m1(x.brgErr)}°"><span class="rl">${label}</span><div class="track"><i class="${cls}" style="width:${w(x.range)}%"></i></div><span class="rv">${m1(x.range)} <span class="chip ${chipFor(x.pos,tv)}">${m1(x.pos)} m ${chipWord[chipFor(x.pos,tv)]}</span></span></div>`:`<div class="rr"><span class="rl">${label}</span><span class="muted">unavailable</span></div>`;
return `<div class="rr"><span class="rl">Real</span><div class="track"><i class="band" style="left:${w(lo)}%;width:${w(a.ref+tv)-w(lo)}%"></i><i class="tick" style="left:${w(a.ref)}%"></i></div><span class="rv">${m1(a.ref)} <span class="muted">${Math.round(a.brg)}°</span></span></div>`+METHODS.map(m=>est(m.key==='terrain'&&p.terrain_ground?.grazing===true?m.label+' (grazing)':m.label,'c-'+m.cls,a[m.key])).join('')}
function echoHTML(r,p){const k=key(r,p),raw=real[k];if(!raw||(!raw.lat&&!raw.lon))return {html:'',cls:''};const e=getEntry(k);if(!e)return {html:'',cls:''};if(e.invalid){const msg=[e.la,e.lo].map((x,i)=>!x?(i?'Longitude is empty':'Latitude is empty'):x.error||'').filter(Boolean)[0]||'Check the coordinate';return {html:esc(msg),cls:'err'}}
const a=analyze(r,p,e);let html=`= ${e.lat.toFixed(6)}°, ${e.lon.toFixed(6)}°`+(e.assumedW?' (longitude taken as West)':'')+(e.src==='sheet'?' · from written sheet':'');let cls='';if(a&&a.ref>1000){html+=` · ⚠ ${(a.ref/1000).toFixed(1)} km from the camera — check latitude/longitude`;cls='warn'}return {html:esc(html).replace('⚠','⚠'),cls}}
function refreshRow(r,p){const k=key(r,p),rs=$('res-'+p.index),ec=$('echo-'+p.index);if(rs)rs.innerHTML=resultCell(r,p);if(ec){const h=echoHTML(r,p);ec.className='echo '+h.cls;ec.innerHTML=h.html}const raw=real[k]||{},la=parseCoord(raw.lat,'lat'),lo=parseCoord(raw.lon,'lon'),row=document.querySelector(`tr[data-idx="${p.index}"]`);if(row){row.querySelector('[data-axis="lat"]').classList.toggle('bad',!!la?.error);row.querySelector('[data-axis="lon"]').classList.toggle('bad',!!lo?.error)}}
function refreshAll(r,p){refreshRow(r,p);glance();plot(r)}
function render(){current=subset.find(r=>r.source.relative_path===$('file').value);$('dronebox').style.display=current?'':'none';if(!current){lastPlot=null;$('photo').removeAttribute('src');$('imageLink').removeAttribute('href');$('imageTitle').textContent='No matching photo';$('meta').textContent='';$('detections').innerHTML=`<tr><td colspan="${METHODS.length+4}">No photos match these filters.</td></tr>`;for(const id of ['modelViews','truth','sheetInfo','links','limitations','plotHint'])$(id).innerHTML='';$('plot').innerHTML='';return}const r=current,c=r.capture_metadata,s=r.sheet;
$('imageTitle').textContent=r.source.relative_path;$('photo').src=r.review_image;$('imageLink').href=r.review_image;
$('meta').innerHTML=`<strong>${esc(c.review_kind)} · ${esc(sheetName(s))} · ${esc(s.handwritten_position_label)}</strong> <span class="muted">${esc(c.capture_time||'')} · ${r.source.original_shape[1]} × ${r.source.original_shape[0]} original pixels</span>`;
const ts=D.truth.filter(t=>+t.sheet_id===s.sheet_id),fillable=ts.filter(t=>finite(t.normalized_latitude)&&finite(t.normalized_longitude));
$('detections').innerHTML=r.people.map(p=>{const k=key(r,p),raw=real[k]||{},sel='<option value="">or fill from a written sheet coordinate…</option>'+fillable.map((t,i)=>`<option value="${i}">${esc(t.person_name)} · ${num(t.normalized_latitude,5)}, ${num(t.normalized_longitude,5)}</option>`).join('');return `<tr data-idx="${p.index}"><td><strong>#${p.index}</strong><br><span class="muted">see photo</span></td><td><div class="cellreal"><input class="coord" data-axis="lat" aria-label="Real latitude for detection ${p.index}" placeholder="Latitude  e.g. 34.38722" value="${esc(raw.lat||'')}" inputmode="text" autocomplete="off"><input class="coord" data-axis="lon" aria-label="Real longitude for detection ${p.index}" placeholder="Longitude  e.g. -84.45472" value="${esc(raw.lon||'')}" inputmode="text" autocomplete="off"><div class="echo" id="echo-${p.index}"></div><select class="fill" aria-label="Fill detection ${p.index} from a written sheet coordinate">${sel}</select><button class="clr" style="justify-self:start;font-size:12px;padding:3px 8px">Clear</button></div></td><td id="res-${p.index}"></td><td>${num(p.confidence,3)}</td>${METHODS.map(m=>{const v=methodValue(p,m);return `<td class="modelcell" data-model="${m.key}">${m.key==='ground'?groundCell(p):m.key==='terrain'?terrainCell(p):depthCell(v,m)}<div class="gps">${coords(v)}</div></td>`}).join('')}</tr>`}).join('')||`<tr><td colspan="${METHODS.length+4}">No person detections above the configured confidence threshold.</td></tr>`;
for(const tr of $('detections').querySelectorAll('tr[data-idx]')){const p=r.people.find(x=>x.index==tr.dataset.idx),k=key(r,p),lat=tr.querySelector('[data-axis="lat"]'),lon=tr.querySelector('[data-axis="lon"]');
const onType=ev=>{const e=real[k]=real[k]||{lat:'',lon:''};if(ev.target===lat&&/[,;]/.test(lat.value)){const[a,b]=lat.value.split(/[,;]/);lat.value=a.trim();lon.value=(b||'').trim()}e.lat=lat.value;e.lon=lon.value;e.src='typed';if(!e.lat&&!e.lon)delete real[k];persist();refreshAll(r,p)};lat.oninput=onType;lon.oninput=onType;
tr.querySelector('select.fill').onchange=ev=>{const t=fillable[+ev.target.value];ev.target.value='';if(!t)return;real[k]={lat:(+t.normalized_latitude).toFixed(6),lon:(+t.normalized_longitude).toFixed(6),src:'sheet'};lat.value=real[k].lat;lon.value=real[k].lon;persist();refreshAll(r,p)};
tr.querySelector('.clr').onclick=()=>{delete real[k];lat.value='';lon.value='';persist();refreshAll(r,p)};refreshRow(r,p)}
$('truth').innerHTML=ts.map(t=>`<tr><td>${esc(t.person_name)}</td><td>${esc(t.normalized_latitude_dms||'Ambiguous')}<br>${num(t.normalized_latitude,7)}</td><td>${esc(t.normalized_longitude_dms||'Ambiguous')}<br>${num(t.normalized_longitude,7)}</td><td>${esc(t.status.replaceAll('_',' '))}</td></tr>`).join('');
$('sheetInfo').innerHTML=`${s.source_image?`<a href="${s.source_image}" target="_blank">Open ${esc(sheetName(s))}</a> · `:''}Recorded drone location:<br>Latitude: ${coordinate(s.drone_latitude,'lat')}<br>Longitude: ${coordinate(s.drone_longitude,'lon')}<br>${esc(s.note||'Group association is inferred from embedded GPS and capture sequence.')}`;
const prefix=r.review_json.split('/results/')[0];$('links').innerHTML=r.review_links?r.review_links.map(([t,h])=>`<a href="${esc(h)}">${esc(t)}</a>`).join(' · '):`<a href="${r.review_json}">YOLO selected result JSON</a> · <a href="${prefix}/detections.csv">YOLO detections CSV</a> · <a href="${prefix}/manifest.json">YOLO run manifest</a>`+METHODS.filter(m=>m.backend&&r.review_depth_models?.[m.backend]).map(m=>` · <a href="${esc(r.review_depth_models[m.backend].result_json)}">${m.label} selected result</a> · <a href="${esc(D.depth_models[m.backend].manifest)}">${m.label} run manifest</a>`).join('');
$('modelViews').innerHTML=METHODS.filter(m=>m.backend).map(m=>{const v=r.review_depth_models?.[m.backend];return v?`<div class="modelview"><h3 style="color:${m.color}">${m.label}</h3><a href="${esc(v.annotated_image)}" target="_blank"><img src="${esc(v.annotated_image)}" alt="${m.label} person estimates"></a>${v.depth_preview?`<details><summary>Full-frame depth map</summary><a href="${esc(v.depth_preview)}" target="_blank"><img src="${esc(v.depth_preview)}" alt="${m.label} axial-depth map"></a></details>`:''}</div>`:`<div class="modelview"><h3>${m.label}</h3><p class="muted">Saved results unavailable for this photo.</p></div>`}).join('');
const warnings=[...new Set([...(D.manifest.limitations||[]),...(c.limitations||[]),...(r.warnings||[])])];$('limitations').innerHTML=warnings.map(x=>`<li>${esc(x)}</li>`).join('');refreshDrone(r);plot(r);
}
/* ---------- map ---------- */
const plotOpts={depth:true,finetuned:true,unidepth:true,lingbot:true,depthpro:true,ground:true,terrain:true,links:true,boxes:true,ranges:true,errors:true};let lastPlot=null;
const PC={...Object.fromEntries(METHODS.map(m=>[m.key,m.color])),truth:'#35a04a'};
const SHAPES={depth:(x,y,s)=>`<circle cx="${x}" cy="${y}" r="${s}"`,
finetuned:(x,y,s)=>{const a=s*.42,b=s-a;return`<path d="M${x-a},${y-s}h${2*a}v${b}h${b}v${2*a}h${-b}v${b}h${-2*a}v${-b}h${-b}v${-2*a}h${b}Z"`},
unidepth:(x,y,s)=>`<path d="M${x},${y-s*1.3}L${x+s*1.3},${y}L${x},${y+s*1.3}L${x-s*1.3},${y}Z"`,
lingbot:(x,y,s)=>`<path d="M${x-s*.55},${y-s}h${s*1.1}l${s*.55},${s}l${-s*.55},${s}h${-s*1.1}l${-s*.55},${-s}Z"`,
depthpro:(x,y,s)=>`<rect x="${x-s*.9}" y="${y-s*.9}" width="${s*1.8}" height="${s*1.8}" rx="1.5"`,
ground:(x,y,s)=>`<path d="M${x},${y-s*1.35}L${x+s*1.3},${y+s*.95}L${x-s*1.3},${y+s*.95}Z"`,
terrain:(x,y,s)=>`<path d="M${x},${y+s*1.35}L${x-s*1.3},${y-s*.95}L${x+s*1.3},${y-s*.95}Z"`,
truth:(x,y,s)=>`<rect x="${x-s*.85}" y="${y-s*.85}" width="${s*1.7}" height="${s*1.7}"`};
const marker=(kind,x,y,s,style)=>SHAPES[kind](x,y,s)+' '+style+'/>',swatch=kind=>`<svg width="16" height="16" viewBox="-9 -9 18 18" aria-hidden="true">${marker(kind,0,0,6.5,`fill="${PC[kind]}" stroke="#fff" stroke-width="1"`)}</svg>`;
$('legend').innerHTML=[...METHODS.map(m=>[m.key,m.label+' estimate']),['links','Lines to real position'],['boxes','Accuracy circles'],['ranges','Range labels'],['errors','Error labels']].map(([k,t])=>`<label class="lg"><input type="checkbox" data-opt="${k}" checked>${SHAPES[k]?swatch(k):''}<span>${esc(t)}</span></label>`).join('')+`<span class="lg">${swatch('truth')}<span>Real position you entered</span></span>`;
function plot(r){lastPlot=[r];const svgEl=$('plot'),o=origin(r),emb=embOrigin(r);
if(!o||!emb){svgEl.setAttribute('viewBox','0 0 640 100');svgEl.innerHTML='<text x="20" y="55" font-size="14">Camera GPS origin unavailable for this photo</text>';return}
const [lat,lon]=o,est=[],tru=[];
for(const p of r.people)for(const m of METHODS){const v=methodValue(p,m);if(plotOpts[m.key]&&v?.status!=='unavailable'&&finite(v?.latitude)&&finite(v?.longitude)){const[e,n]=offset(+v.latitude,+v.longitude,emb[0],emb[1]);est.push({kind:m.key,idx:p.index,e,n,grazing:m.key==='terrain'&&v.grazing===true})}}
for(const p of r.people){const en=getEntry(key(r,p));if(en&&!en.invalid){const[e,n]=offset(en.lat,en.lon,lat,lon);tru.push({idx:p.index,e,n,tol:tolOf(en),src:en.src})}}
$('plotHint').textContent=tru.length?'':'No real coordinates entered for this photo yet — type them in the table below and they will appear here.';
const xs=[0,...est.map(p=>p.e),...tru.flatMap(p=>plotOpts.boxes?[p.e-p.tol,p.e+p.tol]:[p.e])],ys=[0,...est.map(p=>p.n),...tru.flatMap(p=>plotOpts.boxes?[p.n-p.tol,p.n+p.tol]:[p.n])];
const xmn=Math.min(...xs),xmx=Math.max(...xs),ymn=Math.min(...ys),ymx=Math.max(...ys),px=Math.max((xmx-xmn)*.15,10),py=Math.max((ymx-ymn)*.15,10),xmin=xmn-px,xmax=xmx+px,ymin=ymn-py,ymax=ymx+py,xr=xmax-xmin,yr=ymax-ymin;
const ML=46,MR=14,MT=30,MB=42,W=640,iw=W-ML-MR,ih=Math.round(Math.min(540,Math.max(360,iw*yr/xr))),H=MT+ih+MB,scale=Math.min(iw/xr,ih/yr),cx=(xmin+xmax)/2,cy=(ymin+ymax)/2,X=e=>ML+iw/2+(e-cx)*scale,Y=n=>MT+ih/2-(n-cy)*scale;
svgEl.setAttribute('viewBox',`0 0 ${W} ${H}`);
const nice=(span,maxN)=>[1,2,5,10,20,25,50,100,200,500].find(s=>span/s<=maxN)||1000,inside=(x,y,m=4)=>x>ML+m&&x<ML+iw-m&&y>MT+m&&y<MT+ih-m;
const ink='#345065',faint='#7f93a3',gs=nice(Math.max(xr,yr),8),maxR=Math.max(...[[xmin,ymin],[xmin,ymax],[xmax,ymin],[xmax,ymax]].map(([a,b])=>Math.hypot(a,b))),rs=nice(maxR,5),halo='style="paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round"';
const pts=[...est,...tru],da=pts.length?Math.atan2(pts.reduce((a,p)=>a+p.n,0),pts.reduce((a,p)=>a+p.e,0))*180/Math.PI:-90,ras=[45,135,225,315].sort((a,b)=>angDiff(b,da)-angDiff(a,da)).map(d=>d*Math.PI/180);
let svg=`<defs><clipPath id="pc"><rect x="${ML}" y="${MT}" width="${iw}" height="${ih}"/></clipPath></defs><rect x="${ML}" y="${MT}" width="${iw}" height="${ih}" fill="#fff"/><g clip-path="url(#pc)">`;
let ticks='';
for(let e=Math.ceil(xmin/gs)*gs;e<=xmax;e+=gs){svg+=`<line x1="${X(e)}" y1="${MT}" x2="${X(e)}" y2="${MT+ih}" stroke="${e===0?'#b9c6d1':'#eaeff3'}"/>`;ticks+=`<text x="${X(e)}" y="${MT+ih+15}" font-size="11" text-anchor="middle" fill="${faint}">${Math.round(e)}</text>`}
for(let n=Math.ceil(ymin/gs)*gs;n<=ymax;n+=gs){svg+=`<line x1="${ML}" y1="${Y(n)}" x2="${ML+iw}" y2="${Y(n)}" stroke="${n===0?'#b9c6d1':'#eaeff3'}"/>`;ticks+=`<text x="${ML-7}" y="${Y(n)+4}" font-size="11" text-anchor="end" fill="${faint}">${Math.round(n)}</text>`}
const t=r.capture_metadata.telemetry||{},RB=maxR*scale,ray=(b,R)=>[X(0)+Math.sin(b*Math.PI/180)*R,Y(0)-Math.cos(b*Math.PI/180)*R];
if(finite(t.sensor_azimuth)&&finite(t.sensor_hfov)){const [a1,b1]=ray(+t.sensor_azimuth-t.sensor_hfov/2,RB),[a2,b2]=ray(+t.sensor_azimuth+t.sensor_hfov/2,RB);svg+=`<path d="M${X(0)},${Y(0)}L${a1},${b1}A${RB},${RB},0,${t.sensor_hfov>180?1:0},1,${a2},${b2}Z" fill="#345065" fill-opacity=".045" stroke="#a9b8c4" stroke-dasharray="5 4"/>`}
for(const idx of new Set(est.map(q=>q.idx))){const g=est.filter(q=>q.idx===idx),u=q=>Math.hypot(q.e,q.n)||1,[ex,ey]=ray(bearingOf(g.reduce((a,q)=>a+q.e/u(q),0),g.reduce((a,q)=>a+q.n/u(q),0)),RB);svg+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${ex}" y2="${ey}" stroke="#c9d4dd" stroke-dasharray="2 4"/>`}
for(let k=1;k*rs<=maxR;k++){const R=k*rs*scale,[lx,ly]=ras.map(a=>[X(0)+Math.cos(a)*R,Y(0)-Math.sin(a)*R]).find(([x,y])=>inside(x,y,22))||[];svg+=`<circle cx="${X(0)}" cy="${Y(0)}" r="${R}" fill="none" stroke="#c9d4dd" stroke-dasharray="3 4"/>`;if(lx!==undefined)svg+=`<text x="${lx}" y="${ly+3.5}" font-size="10" fill="${faint}" text-anchor="middle" ${halo}>${k*rs} m</text>`}
if(plotOpts.boxes)for(const p of tru)svg+=`<circle cx="${X(p.e)}" cy="${Y(p.n)}" r="${Math.max(3,p.tol*scale)}" fill="${PC.truth}" fill-opacity=".07" stroke="${PC.truth}" stroke-opacity=".55" stroke-dasharray="4 3"/>`;
const S=8,occ=[],lab=[],mk=(x,y)=>occ.push({x:x-S-3,y:y-S-3,w:2*S+6,h:2*S+6}),hit=(a,b)=>a.x<b.x+b.w&&b.x<a.x+a.w&&a.y<b.y+b.h&&b.y<a.y+a.h;
if(plotOpts.links)for(const q of est){const tp=tru.find(p=>p.idx===q.idx);if(!tp)continue;svg+=`<line x1="${X(q.e)}" y1="${Y(q.n)}" x2="${X(tp.e)}" y2="${Y(tp.n)}" stroke="${PC[q.kind]}" stroke-width="1.2" stroke-opacity=".7"/>`;if(plotOpts.errors)lab.push({x:(X(q.e)+X(tp.e))/2,y:(Y(q.n)+Y(tp.n))/2,text:`Δ ${m1(Math.hypot(q.e-tp.e,q.n-tp.n))} m`,color:ink,size:9.5})}
for(const p of tru){const x=X(p.e),y=Y(p.n);mk(x,y);const rg=Math.hypot(p.e,p.n),br=bearingOf(p.e,p.n);svg+=`<g><title>Real position for detection #${p.idx}\n${m1(rg)} m from the camera, bearing ${Math.round(br)}°\n${p.src==='sheet'?'From written sheet':'Typed by you'} (accuracy circle ±${tolM} m)</title>${marker('truth',x,y,S,`fill="${PC.truth}" stroke="#fff" stroke-width="1.5"`)}</g>`;lab.push({x,y,text:`#${p.idx} real`,color:ink,bold:1})}
for(const q of est){const x=X(q.e),y=Y(q.n);mk(x,y);const tp=tru.find(p=>p.idx===q.idx),rg=Math.hypot(q.e,q.n),off=tp?`\n${m1(Math.hypot(q.e-tp.e,q.n-tp.n))} m from the real position you entered`:'\nNo real position entered',method=METHODS.find(m=>m.key===q.kind);
const tip=`<title>Detection #${q.idx} — ${method.label} estimate${q.grazing?'\ngrazing — range unreliable':''}\n${m1(rg)} m horizontal distance from camera${off}</title>`,style=q.grazing?`fill="#fff" stroke="${PC[q.kind]}" stroke-width="2"`:`fill="${PC[q.kind]}" stroke="#fff" stroke-width="1.5"`;
svg+=`<g data-method="${q.kind}">${tip}${marker(q.kind,x,y,S,style)}<text x="${x}" y="${y+(q.kind==='ground'?5.5:q.kind==='terrain'?1.5:3.3)}" font-size="9" font-weight="700" fill="${q.grazing?PC[q.kind]:'#fff'}" text-anchor="middle">${q.idx}</text></g>`;if(plotOpts.ranges)lab.push({x,y,text:`${m1(rg)} m`,color:faint,size:9.5})}
mk(X(0),Y(0));lab.push({x:X(0),y:Y(0),text:droneEntry(r)?'Drone (your GPS)':'Camera',color:'#182c3d',bold:1});
svg+=`<g><title>Camera position</title><path d="M${X(0)-11},${Y(0)}h22M${X(0)},${Y(0)-11}v22" stroke="#182c3d" stroke-width="1.5"/><circle cx="${X(0)}" cy="${Y(0)}" r="4.5" fill="#182c3d" stroke="#fff" stroke-width="1.5"/></g>`;
const errs=k=>est.filter(q=>q.kind===k).map(q=>{const tp=tru.find(p=>p.idx===q.idx);return tp&&Math.hypot(q.e-tp.e,q.n-tp.n)}).filter(finite);
const rows=[...METHODS.filter(m=>est.some(q=>q.kind===m.key)).map(m=>{const e=errs(m.key);return{kind:m.key,text:m.label+(e.length?` · mean error ${m1(e.reduce((a,b)=>a+b,0)/e.length)} m`:'')}}),...(tru.length?[{kind:'truth',text:'Real position'}]:[]),{kind:'camera',text:droneEntry(r)?'Drone (your GPS)':'Camera'}];
const lw=Math.max(...rows.map(x=>x.text.length))*5.6+34,lh=rows.length*15+9,corners=[[ML+8,MT+8],[ML+iw-lw-8,MT+8],[ML+8,MT+ih-lh-8],[ML+iw-lw-8,MT+ih-lh-8]],[lx0,ly0]=corners.find(([x,y])=>!occ.some(q=>hit(q,{x,y,w:lw,h:lh})))||corners[0];occ.push({x:lx0,y:ly0,w:lw,h:lh});
for(const l of lab){const fs=l.size||11.5,w=l.text.length*fs*.56+4,h=fs+2,o2=[[13,4,'start',0],[-13,4,'end',-w],[0,-15,'middle',-w/2],[0,25,'middle',-w/2],[13,-11,'start',0],[13,19,'start',0],[-13,-11,'end',-w],[-13,19,'end',-w]];let pick=null;for(const [dx,dy,anchor,bx] of o2){const b={x:l.x+dx+(anchor==='start'?0:bx),y:l.y+dy-10,w,h};if(b.x>ML&&b.x+b.w<ML+iw&&b.y>MT&&b.y+b.h<MT+ih&&!occ.some(q=>hit(q,b))){pick={dx,dy,anchor,b};break}}pick=pick||{dx:o2[0][0],dy:o2[0][1],anchor:'start',b:{x:l.x+13,y:l.y-6,w,h}};occ.push(pick.b);svg+=`<text x="${l.x+pick.dx}" y="${l.y+pick.dy}" font-size="${fs}" font-weight="${l.bold?600:400}" fill="${l.color}" text-anchor="${pick.anchor}" ${halo}>${esc(l.text)}</text>`}
svg+=`<g><rect x="${lx0}" y="${ly0}" width="${lw}" height="${lh}" rx="4" fill="#fff" fill-opacity=".92" stroke="#d3dde5"/>`+rows.map((x,i)=>{const y=ly0+13+i*15;return(x.kind==='camera'?`<path d="M${lx0+8},${y-3.5}h12M${lx0+14},${y-9.5}v12" stroke="#182c3d" stroke-width="1.3"/><circle cx="${lx0+14}" cy="${y-3.5}" r="2.8" fill="#182c3d"/>`:marker(x.kind,lx0+14,y-3.5,5.5,`fill="${PC[x.kind]}" stroke="#fff" stroke-width="1"`))+`<text x="${lx0+26}" y="${y}" font-size="10.5" fill="${ink}">${esc(x.text)}</text>`}).join('')+'</g>';
const agl=finite(t.altitude_hae)&&finite(r.capture_metadata.assumed_launch_ground_hae_m)?` · camera ${m1(t.altitude_hae-r.capture_metadata.assumed_launch_ground_hae_m)} m above launch ground`:'',cap=`${r.source.relative_path} · ${r.capture_metadata.camera_source} · ${r.people.length} detection${r.people.length===1?'':'s'}`+agl+(finite(t.sensor_azimuth)&&finite(t.sensor_hfov)?` · azimuth ${Math.round(t.sensor_azimuth)}°, HFOV ${Math.round(t.sensor_hfov)}°`:'');
ticks+=`<text x="${ML}" y="${MT-11}" font-size="10.5" fill="${ink}">${esc(cap)}</text>`;
svg+='</g>'+ticks+`<text x="${ML+iw/2}" y="${H-7}" font-size="11.5" fill="${ink}" text-anchor="middle">East offset from camera (m)</text><text x="13" y="${MT+ih/2}" font-size="11.5" fill="${ink}" text-anchor="middle" transform="rotate(-90 13 ${MT+ih/2})">North offset (m)</text><path d="M${ML+iw-22},${MT+30}v-20m-5,7l5,-8l5,8" fill="none" stroke="${ink}" stroke-width="1.5"/><text x="${ML+iw-22}" y="${MT+43}" font-size="10.5" font-weight="700" fill="${ink}" text-anchor="middle">N</text>`;
svgEl.innerHTML=svg;
}
/* ---------- wiring ---------- */
function droneEchoHTML(r){const raw=droneRaw(r),emb=embOrigin(r);if(!raw||(!raw.lat&&!raw.lon))return{html:emb?`Using the GPS embedded in the photo:<br>Latitude: <b>${esc(dms(emb[0],'lat'))}</b> (${emb[0].toFixed(6)}°)<br>Longitude: <b>${esc(dms(emb[1],'lon'))}</b> (${emb[1].toFixed(6)}°)`:'This photo has no embedded drone GPS.',cls:''};const la=parseCoord(raw.lat,'lat'),lo=parseCoord(raw.lon,'lon');if(!la||!lo||la.error||lo.error){const msg=!la?'Latitude is empty':!lo?'Longitude is empty':(la.error||lo.error);return{html:esc(msg)+' — using the embedded GPS until this is fixed.',cls:'err'}}
let html=`Using your drone position ${la.value.toFixed(6)}°, ${lo.value.toFixed(6)}°`+(lo.assumed?' (longitude taken as West)':'')+(raw.src==='sheet'?' · from the written sheet':'');let cls='';if(emb){const [e,n]=offset(la.value,lo.value,emb[0],emb[1]),d=Math.hypot(e,n);html+=` · ${m1(d)} m ${d>0.05?Math.round(bearingOf(e,n))+'° ':''}from the embedded GPS`;if(d>500){html+=' — ⚠ far from the photo GPS, check latitude/longitude';cls='warn'}}return{html,cls}}
function refreshDrone(r){const emb=embOrigin(r),raw=droneRaw(r)||{};$('droneEmbedded').innerHTML=emb?`Embedded in the photo: <b>${esc(dms(emb[0],'lat'))}, ${esc(dms(emb[1],'lon'))}</b> (${emb[0].toFixed(6)}°, ${emb[1].toFixed(6)}°)`:'This photo has no embedded drone GPS.';$('droneLat').value=raw.lat||'';$('droneLon').value=raw.lon||'';$('droneAll').checked=!!droneReal['sheetflag:'+r.sheet.sheet_id];$('droneAllLabel').textContent='Apply to all photos of '+(r.sheet.title||'sheet '+r.sheet.sheet_id);const h=droneEchoHTML(r);$('droneEcho').className='echo '+h.cls;$('droneEcho').innerHTML=h.html;const la=parseCoord(raw.lat,'lat'),lo=parseCoord(raw.lon,'lon');$('droneLat').classList.toggle('bad',!!la?.error);$('droneLon').classList.toggle('bad',!!lo?.error)}
function droneChanged(){if(!current)return;refreshDrone(current);for(const p of current.people)refreshRow(current,p);glance();plot(current)}
function setDrone(r,raw){const sid=r.sheet.sheet_id,all=$('droneAll').checked,k=all?'sheet:'+sid:r.source.relative_path;if(all){for(const rr of D.records)if(rr.sheet.sheet_id===sid)delete droneReal[rr.source.relative_path]}else delete droneReal['sheet:'+sid];if(raw&&(raw.lat||raw.lon))droneReal[k]=raw;else delete droneReal[k];persist();droneChanged()}
const onDrone=()=>{if(!current)return;let lat=$('droneLat').value,lon=$('droneLon').value;if(/[,;]/.test(lat)){const[a,b]=lat.split(/[,;]/);lat=a.trim();lon=(b||'').trim()}setDrone(current,{lat,lon,src:'typed'});$('droneLat').value=lat;$('droneLon').value=lon};
$('droneLat').oninput=onDrone;$('droneLon').oninput=onDrone;
$('droneSheet').onclick=()=>{if(!current)return;const s=current.sheet;if(finite(s.drone_latitude)&&finite(s.drone_longitude)){setDrone(current,{lat:String(s.drone_latitude),lon:String(s.drone_longitude),src:'sheet'})}};
$('droneReset').onclick=()=>{if(!current)return;const sid=current.sheet.sheet_id;delete droneReal[current.source.relative_path];delete droneReal['sheet:'+sid];delete droneReal['sheetflag:'+sid];persist();droneChanged()};
$('droneAll').onchange=()=>{if(!current)return;const r=current,sid=r.sheet.sheet_id,raw=droneRaw(r);if($('droneAll').checked){droneReal['sheetflag:'+sid]=true;for(const rr of D.records)if(rr.sheet.sheet_id===sid)delete droneReal[rr.source.relative_path];if(raw)droneReal['sheet:'+sid]=raw}else{delete droneReal['sheetflag:'+sid];delete droneReal['sheet:'+sid];if(raw)droneReal[r.source.relative_path]=raw}persist();droneChanged()};
$('tol').value=tolM;$('tol').oninput=()=>{const v=+$('tol').value;if(Number.isFinite(v)&&v>=0){tolM=v;persist();glance();if(current){for(const p of current.people)refreshRow(current,p);plot(current)}}};
const figName=ext=>`plot_${(current?.source.relative_path||'figure').replace(/\.[^.]+$/,'')}.${ext}`,dl=(href,name)=>{const a=document.createElement('a');a.href=href;a.download=name;a.click()},figSvg=()=>{const el=$('plot'),[,,w,h]=el.getAttribute('viewBox').split(' ').map(Number);return[new XMLSerializer().serializeToString(el).replace(/<svg /,`<svg width="${w}" height="${h}" `).replace(/>/,'><rect width="100%" height="100%" fill="#fff"/>'),w,h]};
$('saveSvg').onclick=()=>{const u=URL.createObjectURL(new Blob([figSvg()[0]],{type:'image/svg+xml'}));dl(u,figName('svg'));URL.revokeObjectURL(u)};
$('savePng').onclick=()=>{const [txt,w,h]=figSvg(),img=new Image();img.onload=()=>{const c=document.createElement('canvas');c.width=w*3;c.height=h*3;c.getContext('2d').drawImage(img,0,0,c.width,c.height);dl(c.toDataURL('image/png'),figName('png'))};img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(txt)};
for(const el of document.querySelectorAll('[data-opt]'))el.onchange=()=>{plotOpts[el.dataset.opt]=el.checked;if(lastPlot)plot(...lastPlot)};
$('sensor').onchange=filter;$('sheet').onchange=filter;$('file').onchange=render;for(const [id,delta]of[['prev',-1],['next',1]])$(id).onclick=()=>{if(!subset.length)return;$('file').selectedIndex=($('file').selectedIndex+delta+subset.length)%subset.length;render()};
$('clearPhoto').onclick=()=>{if(!current)return;for(const p of current.people)delete real[key(current,p)];persist();render();glance()};
function comparisonCSVRows(){const head=['source','sheet_id','camera','detection_index','real_latitude','real_longitude','coordinate_source','coordinate_accuracy_m','real_range_m','real_bearing_deg','yolo_depth_latitude','yolo_depth_longitude','yolo_depth_range_m','yolo_depth_range_error_m','yolo_depth_position_error_m','ground_plane_latitude','ground_plane_longitude','ground_plane_range_m','ground_plane_range_error_m','ground_plane_position_error_m','terrain_latitude','terrain_longitude','terrain_range_m','terrain_range_error_m','terrain_position_error_m','terrain_ground_hae_m','terrain_incidence_deg','terrain_grazing','terrain_status','terrain_reason','drone_latitude_used','drone_longitude_used','drone_gps_source'];
for(const m of METHODS.filter(m=>m.backend))for(const suffix of ['latitude','longitude','depth_z_m','slant_range_m','range_m','range_error_m','position_error_m','status','reason'])head.push(m.backend+'_'+suffix);const rows=[head];
for(const r of D.records)for(const p of r.people){const e=getEntry(key(r,p));if(!e||e.invalid)continue;const a=analyze(r,p,e);if(!a)continue;const g=p.ground_plane||{},d=a.depth||{},q=a.ground||{},tg=p.terrain_ground||{},ta=a.terrain||{};const row=[r.source.relative_path,r.sheet.sheet_id,r.capture_metadata.review_kind,p.index,e.lat,e.lon,e.src==='sheet'?'written_sheet':'typed_by_user',tolOf(e),a.ref,a.brg,p.latitude,p.longitude,d.range,d.rangeErr,d.pos,g.latitude,g.longitude,q.range,q.rangeErr,q.pos,tg.latitude,tg.longitude,ta.range,ta.rangeErr,ta.pos,tg.ground_hae_m,tg.incidence_deg,tg.grazing,tg.status||'unavailable',tg.reason||'',...origin(r),droneEntry(r)?(droneEntry(r).src==='sheet'?'written_sheet':'typed_by_user'):'embedded_in_photo'];for(const m of METHODS.filter(m=>m.backend)){const v=methodValue(p,m)||{},s=a[m.key]||{};row.push(v.latitude,v.longitude,v.depth_m,v.slant_range_m,s.range,s.rangeErr,s.pos,v.status||'unavailable',v.reason||'')}rows.push(row)}return rows}
function comparisonCSV(){const rows=comparisonCSVRows(),head=rows[0];return rows.map((row,i)=>row.map((v,j)=>'"'+String(i&&isNum(v)?+v.toFixed(/latitude|longitude/.test(head[j])?7:3):(v??'')).replaceAll('"','""')+'"').join(',')).join('\r\n')}
$('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+comparisonCSV()],{type:'text/csv;charset=utf-8'}));a.download='real_coordinate_comparison.csv';a.click();URL.revokeObjectURL(a.href)};
glance();filter();
</script></html>'''


if __name__ == '__main__':
    main()
