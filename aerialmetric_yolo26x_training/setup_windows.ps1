param(
    [string]$PythonExecutable = "py",
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
trap {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    exit 2
}
# Preserve caller-relative explicit paths before entering the copied bundle.
if ($PSBoundParameters.ContainsKey('PythonExecutable') -and (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    $PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
}
Set-Location -LiteralPath $PSScriptRoot
$bundlePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pythonHelp = 'Use standard 64-bit CPython 3.12 or 3.14 (except 3.14.1), from https://www.python.org/downloads/windows/, or provide -PythonExecutable C:\Path\To\Python\python.exe. Free-threaded builds are not supported by this bundle.'
$driverHelp = 'CUDA 13.0 requires a compatible NVIDIA driver (580 or newer). Update it from https://www.nvidia.com/Download/index.aspx. No separate CUDA toolkit is required for these PyTorch wheels.'

function Find-NvidiaSmi {
    $command = Get-Command 'nvidia-smi.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    $candidates = @()
    if ($env:WINDIR) { $candidates += Join-Path $env:WINDIR 'System32\nvidia-smi.exe' }
    if ($env:ProgramFiles) { $candidates += Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe' }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw "NVIDIA driver utility nvidia-smi was not found. $driverHelp"
}

function Confirm-NvidiaDriver {
    $smiPath = Find-NvidiaSmi
    try { $lines = @(& $smiPath '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader,nounits' 2>&1) }
    catch { throw "NVIDIA driver/GPU check failed: $($_.Exception.Message). $driverHelp" }
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) {
        throw "NVIDIA driver/GPU check failed: $($lines -join ' '). $driverHelp"
    }
    $devices = @($lines | ConvertFrom-Csv -Header 'GPU', 'Driver', 'MemoryMiB')
    foreach ($device in $devices) {
        $driver = $device.Driver.Trim()
        $driverMajor = 0
        if (-not [int]::TryParse(($driver -split '\.')[0], [ref]$driverMajor)) {
            throw "Could not read the NVIDIA driver version from: $($lines -join ' '). $driverHelp"
        }
        Write-Host "NVIDIA GPU: $($device.GPU.Trim()); driver: $driver; total VRAM: $($device.MemoryMiB.Trim()) MiB"
        if ($driverMajor -lt 580) { throw "NVIDIA driver $driver is too old for this CUDA 13.0 bundle. $driverHelp" }
    }
}

function Resolve-PythonCompatibility {
    param([string]$RequestedExecutable)
    if ($RequestedExecutable -in @('py', 'py.exe')) {
        $launcher = Get-Command 'py.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $launcher) { throw "The Python launcher 'py' is missing. $pythonHelp" }
        # Listing registered interpreters avoids Python install-manager auto-downloads.
        try { $registered = @(& $launcher.Source '-0p' 2>&1) }
        catch { throw "Could not list installed Python interpreters: $($_.Exception.Message). $pythonHelp" }
        if ($LASTEXITCODE -ne 0) { throw "Could not list installed Python interpreters. $pythonHelp" }
        $compatible = @()
        $rejected = @()
        foreach ($line in $registered) {
            # Use the registered executable directly rather than asking py to launch/install a version.
            if ([string]$line -match '^\s*-\S+\s+\*?\s*((?:[A-Za-z]:\\|\\\\).+?\.exe)\s*$') {
                $candidate = $Matches[1]
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    try {
                        $info = Confirm-PythonCompatibility -Executable $candidate -Quiet
                        $compatible += [PSCustomObject]@{
                            Executable = $info.executable
                            Minor = [int]$info.version[1]
                            Patch = [int]$info.version[2]
                        }
                    } catch { $rejected += $_.Exception.Message }
                }
            }
        }
        if ($compatible.Count) {
            # Prefer compatible 3.14, then 3.12; within each version prefer the newest patch.
            $selected = $compatible | Sort-Object -Property Minor, Patch -Descending | Select-Object -First 1
            return $selected.Executable
        }
        $detail = if ($rejected.Count) { " Rejected registered interpreters: $($rejected -join ' ')" } else { '' }
        throw "No compatible standard 64-bit CPython 3.12/3.14 is registered with 'py'.$detail $pythonHelp"
    }
    if (Test-Path -LiteralPath $RequestedExecutable -PathType Leaf) {
        return (Resolve-Path -LiteralPath $RequestedExecutable).Path
    }
    $command = Get-Command $RequestedExecutable -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { throw "Python executable '$RequestedExecutable' was not found. $pythonHelp" }
    return $command.Source
}

function Confirm-PythonCompatibility {
    param([string]$Executable, [switch]$Quiet)
    $probeCode = "import json,struct,sys,sysconfig; print(json.dumps({'version':list(sys.version_info[:3]),'bits':struct.calcsize('P')*8,'implementation':sys.implementation.name,'free_threaded':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),'executable':sys.executable}))"
    try { $probe = @(& $Executable '-c' $probeCode 2>&1) }
    catch { throw "Python could not start: $($_.Exception.Message). $pythonHelp" }
    if ($LASTEXITCODE -ne 0) { throw "Python could not start: $($probe -join ' '). $pythonHelp" }
    try { $info = ($probe -join "`n") | ConvertFrom-Json }
    catch { throw "Could not read Python version/architecture: $($probe -join ' '). $pythonHelp" }
    if ($info.implementation -ne 'cpython' -or $info.bits -ne 64 -or $info.version[0] -ne 3 -or $info.version[1] -notin @(12, 14)) {
        throw "Found $($info.implementation) $($info.version -join '.') / $($info.bits)-bit at $($info.executable). $pythonHelp"
    }
    if ($info.version[1] -eq 14 -and $info.version[2] -eq 1) {
        throw "Python 3.14.1 is explicitly excluded by torchvision 0.29.0's Requires-Python metadata: https://pypi.org/project/torchvision/0.29.0/. Choose another supported patch version. $pythonHelp"
    }
    if ($info.free_threaded) {
        throw "The interpreter at $($info.executable) is a free-threaded CPython build (Py_GIL_DISABLED=1). $pythonHelp"
    }
    if (-not $Quiet) { Write-Host "Standard CPython $($info.version -join '.') / 64-bit: $($info.executable)" }
    return $info
}

Confirm-NvidiaDriver
$explicitPython = $PSBoundParameters.ContainsKey('PythonExecutable')
if ((Test-Path -LiteralPath $bundlePython -PathType Leaf) -and -not $explicitPython) {
    # In check mode, an existing environment needs no globally registered Python.
    $selectedPython = $bundlePython
} else {
    $selectedPython = Resolve-PythonCompatibility $PythonExecutable
}
$selectedPythonInfo = Confirm-PythonCompatibility $selectedPython

if ($CheckOnly) {
    Write-Host 'Check-only mode: no environment creation or package installation.'
    & $selectedPython 'scripts/check_environment.py' '--runtime-only' '--json-report' 'checks/environment_preinstall.json'
    if ($LASTEXITCODE -ne 0) {
        throw 'Runtime check failed. Read the diagnosis above or checks/environment_preinstall.json, then run setup_windows.ps1 to install the required packages.'
    }
    Write-Host 'Runtime check passed. No training was started.'
    return
}

if (Test-Path -LiteralPath $bundlePython -PathType Leaf) {
    $existingPythonInfo = Confirm-PythonCompatibility $bundlePython
    if ($explicitPython -and $existingPythonInfo.version[1] -ne $selectedPythonInfo.version[1]) {
        throw "The existing .venv uses CPython $($existingPythonInfo.version -join '.'), but -PythonExecutable selected CPython $($selectedPythonInfo.version -join '.'). Rename the existing .venv, then rerun setup with the selected interpreter. No packages were installed."
    }
} else {
    & $selectedPython '-m' 'venv' '.venv'
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. $pythonHelp" }
}
$null = Confirm-PythonCompatibility $bundlePython
& $bundlePython '-m' 'pip' 'install' '--upgrade' 'pip'
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }

$torchProbe = @'
import json, sys
try:
    import torch, torchvision
    compatible = torch.__version__.split('+')[0] == '2.14.0' and torchvision.__version__.split('+')[0] == '0.29.0' and torch.version.cuda == '13.0' and '+cu130' in torch.__version__ and '+cu130' in torchvision.__version__
    print(json.dumps({'torch':torch.__version__, 'torchvision':torchvision.__version__, 'cuda_runtime':torch.version.cuda, 'matches_bundle':compatible}))
    sys.exit(0 if compatible else 2)
except Exception as error:
    print(json.dumps({'matches_bundle':False, 'reason':str(error)}))
    sys.exit(2)
'@
& $bundlePython '-c' $torchProbe
$torchMatches = $LASTEXITCODE -eq 0
if (-not $torchMatches) {
    Write-Host 'Installing the pinned CUDA 13.0 PyTorch/torchvision pair, replacing any CPU or mismatched builds.'
    & $bundlePython '-m' 'pip' 'install' '--force-reinstall' '--no-cache-dir' '-r' 'requirements-cuda.txt' '--index-url' 'https://download.pytorch.org/whl/cu130'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed. Check network access and the pip error above.' }
} else {
    Write-Host 'The pinned CUDA 13.0 PyTorch/torchvision pair is already installed.'
}
& $bundlePython 'scripts/check_environment.py' '--runtime-only' '--json-report' 'checks/environment_runtime.json'
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime check failed before application-package installation. Read checks/environment_runtime.json and the diagnosis above.' }
& $bundlePython '-m' 'pip' 'install' '-r' 'requirements.txt'
if ($LASTEXITCODE -ne 0) { throw 'Application requirements installation failed.' }
& $bundlePython '-m' 'pip' 'check'
if ($LASTEXITCODE -ne 0) { throw 'pip check found incompatible installed packages.' }
& $bundlePython 'scripts/check_environment.py' '--json-report' 'checks/environment_full.json'
if ($LASTEXITCODE -ne 0) { throw 'Full environment/model check failed. Read checks/environment_full.json and the diagnosis above.' }
Write-Host 'Setup complete. Use .venv\Scripts\python.exe for all commands in README.md. No training was started.'
