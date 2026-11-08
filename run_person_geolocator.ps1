# Launch the web interface. Pass --port N to change the port. CLI: python -m person_geolocator.cli --help
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venvs\depth-comparison\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython)) { throw 'Depth environment missing. See reference_validation\README_DEPTH_MODELS.md.' }
Set-Location $PSScriptRoot
& $taskPython -m person_geolocator.server @args
exit $LASTEXITCODE
