import importlib
import sys
import types


def _install_qt_stubs(monkeypatch):
    class QtModule:
        def __init__(self, **attrs):
            for key, value in attrs.items():
                setattr(self, key, value)

        def __getattr__(self, name):
            dummy = type(name, (), {"__init__": lambda self, *args, **kwargs: None})
            setattr(self, name, dummy)
            return dummy

    class QApplication:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def instance():
            return None

        def exec(self):  # pragma: no cover - UI path not exercised in tests
            return 0

        def processEvents(self):  # pragma: no cover - UI path not exercised in tests
            return None

    class QMessageBox:
        @staticmethod
        def critical(*args, **kwargs):  # pragma: no cover - error path not hit in test
            return None

    class QKeySequence:
        def __init__(self, *args, **kwargs):
            pass

    qtwidgets = QtModule(
        QApplication=QApplication,
        QMessageBox=QMessageBox,
        QMainWindow=type("QMainWindow", (), {}),
        QWidget=type("QWidget", (), {}),
        QTableWidgetItem=type("QTableWidgetItem", (), {}),
        QFileDialog=type("QFileDialog", (), {}),
    )
    qtgui = QtModule(
        QAction=type("QAction", (), {"__init__": lambda self, *args, **kwargs: None}),
        QShortcut=type("QShortcut", (), {"__init__": lambda self, *args, **kwargs: None}),
        QKeySequence=QKeySequence,
    )
    qtcore = QtModule(Qt=types.SimpleNamespace(Horizontal=0, Vertical=1))

    pyside = types.ModuleType("PySide6")
    pyside.QtWidgets = qtwidgets
    pyside.QtGui = qtgui
    pyside.QtCore = qtcore

    monkeypatch.setitem(sys.modules, "PySide6", pyside)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qtwidgets)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qtgui)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qtcore)


def test_preflight_invokes_installers(monkeypatch):
    _install_qt_stubs(monkeypatch)
    sys.modules.pop("ui.main", None)
    ui_main = importlib.import_module("ui.main")

    called = {}

    def fake_find_spec(name):
        if name in {"faster_whisper", "torch", "omegaconf"}:
            return None
        return types.SimpleNamespace()

    monkeypatch.setattr(ui_main.importlib.util, "find_spec", fake_find_spec)

    def fake_ensure_package(pkg, message, ask_to_pin):
        called["pkg"] = (pkg, message, ask_to_pin)

    monkeypatch.setattr(ui_main, "ensure_package", fake_ensure_package)

    def fake_ensure_tts(engine):
        called["tts"] = engine

    monkeypatch.setattr(ui_main, "ensure_tts_dependencies", fake_ensure_tts)

    assert ui_main.preflight() is True
    assert called["pkg"][0] == "faster-whisper"
    assert called["pkg"][2] is False
    assert called["tts"] == "silero"
