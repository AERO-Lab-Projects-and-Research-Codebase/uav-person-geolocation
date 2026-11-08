#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
    printf '%s\n' \
        'Usage: bash setup_linux.sh [--check-only]' \
        '' \
        'Default: verify prerequisites, create/reuse .venv, install pinned CUDA' \
        '13.0 PyTorch and application packages, then run the environment checks.' \
        '--check-only: inspect prerequisites and an existing .venv; never install' \
        'packages, create a virtual environment, or start training.' \
        'If .venv is absent, report installed runtime packages using base Python.' \
        '' \
        'Use ordinary 64-bit CPython 3.12 or 3.14 (3.14.1 is excluded).' \
        'Automatic selection: existing .venv, python3.14, python3.12, python3.' \
        'Override selection with PYTHON_EXECUTABLE=/path/to/python3.14.'
}

fail() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 2
}

check_only=false
for argument in "$@"; do
    case "$argument" in
        --check-only) check_only=true ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "Unknown argument: $argument" ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail 'nvidia-smi was not found. Install an NVIDIA Linux driver version 580 or newer, reboot, then rerun. Official driver downloads: https://www.nvidia.com/Download/index.aspx . For WSL2, update the NVIDIA Windows host driver; do not install a Linux display driver inside WSL.'
fi
if ! gpu_information="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>&1)"; then
    fail "nvidia-smi could not communicate with the GPU. Install/update the NVIDIA driver (580 or newer), reboot, and retry. Official downloads: https://www.nvidia.com/Download/index.aspx . Output: $gpu_information"
fi
if [[ -z "$gpu_information" ]]; then
    fail 'nvidia-smi reported no NVIDIA GPU. Check GPU visibility before setup.'
fi
printf 'NVIDIA GPU(s), driver version, memory MiB:\n%s\n' "$gpu_information"
while IFS=, read -r gpu_name driver_version memory_mib; do
    driver_version="${driver_version//[[:space:]]/}"
    driver_major="${driver_version%%.*}"
    if [[ ! "$driver_major" =~ ^[0-9]+$ ]]; then
        fail "Cannot read NVIDIA driver version for $gpu_name: $driver_version"
    fi
    if (( 10#$driver_major < 580 )); then
        fail "NVIDIA driver $driver_version is too old for this bundle's CUDA 13.0 wheels. Install driver 580 or newer and reboot before rerunning: https://www.nvidia.com/Download/index.aspx . For WSL2, update the Windows host driver."
    fi
done <<< "$gpu_information"

venv_python=.venv/bin/python
python_help='Use ordinary 64-bit CPython 3.12 or 3.14 with venv support. Python 3.14.1 is excluded by the pinned torchvision requirement; use Python 3.14.2 or newer within 3.14 instead. Free-threaded builds, 32-bit Python, PyPy, and other Python versions are unsupported. Install from your distribution package manager or https://www.python.org/downloads/; set PYTHON_EXECUTABLE to its executable path.'
python_probe='import struct, sys, sysconfig
free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
bits = struct.calcsize("P") * 8
ordinary_cpython = sys.implementation.name == "cpython" and not free_threaded
supported = sys.version_info[:2] in ((3, 12), (3, 14)) and sys.version_info[:3] != (3, 14, 1)
print("Python:", sys.version.split()[0], str(bits) + "-bit", sys.implementation.name, "free-threaded=" + str(free_threaded), "executable=" + sys.executable)
sys.exit(0 if supported and bits == 64 and ordinary_cpython else 2)'

confirm_python() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fail "Python executable '$1' was not found. $python_help"
    fi
    if ! "$1" -c "$python_probe"; then
        fail "The selected Python executable '$1' is incompatible. $python_help"
    fi
}

explicit_python=false
if [[ ${PYTHON_EXECUTABLE+x} ]]; then
    explicit_python=true
    python_command="$PYTHON_EXECUTABLE"
    # An explicit executable is never replaced by an automatic fallback.
    confirm_python "$python_command"
elif [[ -x "$venv_python" ]]; then
    python_command="$venv_python"
    confirm_python "$python_command"
    printf '%s\n' 'Using the existing compatible Linux .venv.'
else
    python_command=''
    for candidate in python3.14 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$python_probe" >/dev/null 2>&1; then
            python_command="$candidate"
            break
        fi
    done
    if [[ -z "$python_command" ]]; then
        fail "No supported Python interpreter was found. $python_help"
    fi
    confirm_python "$python_command"
fi

if "$check_only"; then
    check_status=0
    if [[ -x "$venv_python" && "$python_command" == "$venv_python" ]]; then
        "$python_command" scripts/check_environment.py --json-report checks/environment.json || check_status=$?
    else
        printf '%s\n' 'No Linux .venv found; checking the base Python runtime without installing packages.'
        "$python_command" scripts/check_environment.py --runtime-only --json-report checks/environment_preinstall.json || check_status=$?
    fi
    "$python_command" -m pip check || check_status=$?
    if (( check_status == 0 )); then
        printf '%s\n' 'Environment checks passed. Training memory fit still requires the separate smoke run.'
    else
        printf '%s\n' 'Environment checks found issues. Follow the reported guidance, then run bash setup_linux.sh to install or repair this environment.' >&2
    fi
    printf '%s\n' 'No packages were installed, no virtual environment was created, and no training was started. Diagnostic JSON/settings may have been written.'
    exit "$check_status"
fi

if [[ -d .venv ]]; then
    if [[ ! -x "$venv_python" ]]; then
        fail 'The existing .venv is incomplete or belongs to another operating system. Rename it, then rerun setup to create a Linux virtual environment.'
    fi
    confirm_python "$venv_python"
    if "$explicit_python"; then
        selected_minor="$("$python_command" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        existing_minor="$("$venv_python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        if [[ "$selected_minor" != "$existing_minor" ]]; then
            fail "PYTHON_EXECUTABLE selects Python $selected_minor, but .venv uses $existing_minor. Rename the existing .venv, then rerun to create an environment using the explicit interpreter. Dataset/model files are unaffected."
        fi
    fi
else
    if ! "$python_command" -m venv .venv; then
        fail "Virtual-environment creation failed. Install venv support matching your selected Python (for example python3.14-venv or python3.12-venv on Ubuntu), inspect/rename any incomplete .venv, and rerun. $python_help"
    fi
fi

"$venv_python" -m pip install --upgrade pip
if "$venv_python" -c 'import sys, torch, torchvision; sys.exit(0 if torch.__version__ == "2.14.0+cu130" and torchvision.__version__ == "0.29.0+cu130" and torch.version.cuda == "13.0" else 2)' >/dev/null 2>&1; then
    printf '%s\n' 'The pinned CUDA 13.0 PyTorch/torchvision builds are already installed.'
else
    printf '%s\n' 'Installing/replacing PyTorch and torchvision with the pinned CUDA 13.0 builds.'
    "$venv_python" -m pip install --upgrade --force-reinstall -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu130
fi

# Verify the CUDA runtime before installing application packages. This check
# performs only a small GPU computation; it does not load or train the model.
"$venv_python" scripts/check_environment.py --runtime-only
"$venv_python" -m pip install -r requirements.txt
"$venv_python" -m pip check
"$venv_python" scripts/check_environment.py
printf '%s\n' 'Setup complete. Use .venv/bin/python for all commands in README.md. Training has not been started.'
