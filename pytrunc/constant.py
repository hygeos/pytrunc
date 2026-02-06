
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


DIR_ROOT = Path(__file__).resolve().parent.parent

# Read version from pyproject.toml
if tomllib is not None:
    try:
        pyproject_path = DIR_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        VERSION = pyproject_data["project"]["version"]
    except (FileNotFoundError, KeyError):
        VERSION = '0.0.0'  # Fallback
else:
    VERSION = '0.0.0'  # Fallback

