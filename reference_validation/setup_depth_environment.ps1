param([string]$Environment = '.venvs\depth-comparison')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBasePython = Join-Path $taskRoot 'skydio live data\.venv\Scripts\python.exe'
$taskEnvironment = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Environment))
$taskPython = Join-Path $taskEnvironment 'Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskBasePython)) { throw 'The existing project Python environment is required.' }
if (!(Test-Path -LiteralPath $taskPython)) {
    & $taskBasePython -m venv $taskEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create isolated depth environment.' }
}
# Reuse the existing CUDA PyTorch runtime read-only. New packages install into
# this environment only; the live project's site-packages are never modified.
$taskInheritedSite = Join-Path $taskRoot 'skydio live data\.venv\Lib\site-packages'
Set-Content -LiteralPath (Join-Path $taskEnvironment 'Lib\site-packages\project_runtime.pth') -Value $taskInheritedSite -Encoding ascii
& $taskPython -m pip install 'einops==0.8.2' 'huggingface-hub==2.1.1' 'timm==1.0.22' 'pillow-heif==1.8.0' 'scipy==1.18.1' 'wandb==0.30.0' 'termcolor==3.3.0' 'h5py==3.16.0' 'tqdm==4.70.1' 'safetensors==0.8.0'
if ($LASTEXITCODE -ne 0) { throw 'Cannot install inference dependencies.' }
$taskSources = @(
    @{Name='UniDepth'; Url='https://github.com/lpiccinelli-eth/UniDepth.git'; Commit='8d8cfe4c7ee15297099983607febf0d4f32eb3d6'},
    @{Name='DepthPro'; Url='https://github.com/apple-aiml-research/ml-depth-pro.git'; Commit='9e65e4dbe9568d23c546fcec53302b10445e109e'}
)
foreach ($taskSource in $taskSources) {
    $taskCheckout = Join-Path $taskRoot ('external\' + $taskSource.Name)
    if (!(Test-Path -LiteralPath $taskCheckout)) {
        & git clone $taskSource.Url $taskCheckout
        if ($LASTEXITCODE -ne 0) { throw 'Cannot clone model source.' }
        & git -C $taskCheckout checkout --detach $taskSource.Commit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot select pinned model source.' }
    }
    $taskRevision = (& git -C $taskCheckout rev-parse HEAD).Trim()
    if ($taskRevision -ne $taskSource.Commit) { throw "Model source revision differs at $taskCheckout; inspect it before rerunning." }
    & $taskPython -m pip install --no-deps -e $taskCheckout
    if ($LASTEXITCODE -ne 0) { throw 'Cannot register model package.' }
}
# Upstream requirements include incompatible NumPy bounds and Linux training
# ops. Install only the inference subset actually tested on this Windows host.
& $taskPython -c 'from unidepth.models import UniDepthV2; import depth_pro, torch; print("Depth backends ready; CUDA:", torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Model import verification failed.' }
