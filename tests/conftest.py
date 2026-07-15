import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def load_script(name: str):
    """Import a scripts/*.py module even when the filename starts with a digit."""
    path = SCRIPTS / name
    mod_name = "script_" + name.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO
