"""
Package-level constants.

This module defines the pytrunc root directory and the pytrunc version,
automatically read from the pyproject toml file.
"""

import tomllib
from pathlib import Path

DIR_ROOT: Path = Path(__file__).resolve().parent.parent

# Read version from pyproject.toml
try:
    pyproject_path = DIR_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    VERSION: str = pyproject_data["project"]["version"]
except (FileNotFoundError, KeyError):
    VERSION = "0.0.0"  # Fallback
