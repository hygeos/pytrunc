
from pathlib import Path
import tomllib



DIR_ROOT = Path(__file__).resolve().parent.parent

# Read version from pyproject.toml
try:
    pyproject_path = DIR_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    VERSION = pyproject_data["project"]["version"]
except (FileNotFoundError, KeyError):
    VERSION = '0.0.0'  # Fallback


