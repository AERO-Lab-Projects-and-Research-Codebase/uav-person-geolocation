param(
    [string]$Environment = '.venvs\depth-comparison',
    [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskEnvironment = if ([System.IO.Path]::IsPathRooted($Environment)) {
    [System.IO.Path]::GetFullPath($Environment)
} else { [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Environment)) }
$taskPython = Join-Path $taskEnvironment 'Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython)) {
    throw "Depth Python environment missing at $taskPython. Run reference_validation/setup_depth_environment.ps1 first."
}
# This uses the existing working CUDA runtime. Installing upstream mdm normally
# would replace torch with its old pin and request incompatible xformers wheels.
& $taskPython -c 'import torch, torchvision, numpy, cv2; print("CUDA:", torch.cuda.is_available(), "torch:", torch.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'The existing depth environment is missing its core runtime.' }
& $taskPython -c 'import huggingface_hub'
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install --no-deps 'huggingface-hub==2.1.1'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install huggingface-hub.' }
}
$taskSource = Join-Path $taskRoot 'external\LingBotDepth'
$taskRevision = 'f3a237e434ae987bc38281476d6cfb5df3e4d739'
if (!(Test-Path -LiteralPath $taskSource)) {
    & git clone 'https://github.com/Robbyant/lingbot-depth.git' $taskSource
    if ($LASTEXITCODE -ne 0) { throw 'Cannot clone official LingBot source.' }
    & git -C $taskSource checkout --detach $taskRevision
    if ($LASTEXITCODE -ne 0) { throw 'Cannot select pinned LingBot source.' }
}
$taskActualRevision = (& git -C $taskSource rev-parse HEAD)
if ($LASTEXITCODE -ne 0 -or $taskActualRevision.Trim() -ne $taskRevision) {
    throw "LingBot source at $taskSource differs from pinned revision $taskRevision; inspect before rerunning."
}
$taskImportProbe = @'
import os, sys
os.environ['XFORMERS_DISABLED'] = '1'
sys.path.insert(0, sys.argv[1])
from mdm.model.v2 import MDMModel
print('LingBot inference imports ready; native PyTorch SDPA will be used.')
'@
& $taskPython -c $taskImportProbe $taskSource
if ($LASTEXITCODE -ne 0) { throw 'LingBot import verification failed.' }
if (!$SkipDownload) {
    $taskDownload = @'
import hashlib, json, pathlib, sys, time
from huggingface_hub import hf_hub_download
root = pathlib.Path(sys.argv[1])
folder = root / 'models' / 'lingbot_depth'
folder.mkdir(parents=True, exist_ok=True)
repository = 'robbyant/lingbot-depth-pretrain-vitl-14-v0.5'
revision = '79204ed6b837f4fdd192cf563e59481fecfa0295'
expected_sha256 = 'b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23'
print('Downloading official LingBot v0.5 checkpoint (1.28 GB); interrupted downloads can resume.', flush=True)
started = time.perf_counter()
path = pathlib.Path(hf_hub_download(repo_id=repository, filename='model.pt', revision=revision, local_dir=str(folder)))
digest = hashlib.sha256()
with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256:
    raise RuntimeError('LingBot checkpoint SHA256 differs from the pinned official LFS object')
manifest = {'repository': repository, 'weights_revision': revision,
            'source_url': 'https://github.com/Robbyant/lingbot-depth',
            'source_revision': 'f3a237e434ae987bc38281476d6cfb5df3e4d739',
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual_sha256,
            'download_and_hash_seconds': time.perf_counter() - started}
(folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest), flush=True)
'@
    & $taskPython -c $taskDownload $taskRoot
    if ($LASTEXITCODE -ne 0) { throw 'Official LingBot checkpoint download or hash verification failed.' }
}
Write-Host 'Webcam depth LingBot setup complete. The webcam script uses LingBot to refine UniDepth estimates.'
