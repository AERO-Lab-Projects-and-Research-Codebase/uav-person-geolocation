# Launch from any directory. Pass Python flags directly, e.g. --camera 1.
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venvs\depth-comparison\Scripts\python.exe'
$taskScript = Join-Path $PSScriptRoot 'webcam_depth\run_webcam_depth.py'
if (!(Test-Path -LiteralPath $taskPython)) {
    throw 'Depth environment missing. Run .\webcam_depth\setup_webcam_depth.ps1 first.'
}
& $taskPython $taskScript @args
exit $LASTEXITCODE
