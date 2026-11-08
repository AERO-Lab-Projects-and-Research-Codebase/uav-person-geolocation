"""Python interpreter policy for the pinned training packages (stdlib only)."""
from __future__ import annotations

import platform
import struct
import sys
import sysconfig


def compatibility_errors() -> list[str]:
    errors = []
    if sys.version_info[:2] not in {(3, 12), (3, 14)}:
        errors.append("Use standard 64-bit CPython 3.12 or 3.14 for this bundle. Python 3.14.2 or newer is recommended on the 3.14 branch.")
    if sys.version_info[:3] == (3, 14, 1):
        errors.append("torchvision 0.29.0 explicitly excludes Python 3.14.1. Update Python to 3.14.2 or newer, or use Python 3.12.")
    if struct.calcsize("P") != 8:
        errors.append("This bundle requires 64-bit Python. Install a 64-bit interpreter from https://www.python.org/downloads/.")
    if platform.python_implementation() != "CPython":
        errors.append("This bundle uses CPython wheels; select a standard CPython installation.")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        errors.append("The free-threaded Python build is not supported by this bundle's wheel set. Select standard CPython 3.14 or 3.12.")
    return errors
