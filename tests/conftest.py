import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import model_manager, pkg_installer


def _noop(*args, **kwargs):
    return None


def _fake_ensure_model(name: str, category: str, *, parent=None, auto_download=False):
    path = Path("models") / category / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


pkg_installer.ensure_package = _noop
model_manager.ensure_model = _fake_ensure_model
