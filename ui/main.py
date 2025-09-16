# PySide6 UI — Revoice alpha3
# Voice auto-detection removed
# Logs: BASE_DIR / logs / log_version_alpha3.txt

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

from core.presets import load_presets

from .ai_edit_dialog import AiEditDialog, AiEditResult
from .config import DEFAULT_WHISPER_MODEL, load_config, save_config
from .settings import SettingsDialog

# Version and log file
APP_VER = "alpha3"
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"log_version_{APP_VER}.txt"

# Default input/output directories
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("ui")


def handle_exception(exc_type, exc, tb):
    """Log uncaught exceptions before the application exits."""
    log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))


# Register global exception hook
sys.excepthook = handle_exception

# Импорт пайплайна
from core.model_manager import list_models  # noqa: E402
from core.pipeline import (  # noqa: E402
    ensure_ffmpeg,
    merge_into_phrases,
    phrases_to_marked_text,
    revoice_video,
    transcribe_whisper,
)
from core.tts_adapters import SILERO_VOICES  # noqa: E402

try:
    from core.qwen_editor import QwenEditor  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    QwenEditor = None  # type: ignore[assignment]
    _QWEN_IMPORT_ERROR = e
else:
    _QWEN_IMPORT_ERROR = None

try:
    from core.vibe_editor import VibeEditor  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    VibeEditor = None  # type: ignore[assignment]
    _VIBE_IMPORT_ERROR = e
else:
    _VIBE_IMPORT_ERROR = None

YANDEX_VOICES = ["ermil", "filipp", "alena", "jane", "oksana", "zahar", "omazh", "madirus"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Revoice — {APP_VER}")
        self.resize(1024, 680)

        # Состояние
        self.video_path = ""
        self.out_dir = str(OUTPUT_DIR)
        self.tts_engine = "silero"
        self.voice_id = "baya"
        self.language = "ru"
        self.whisper_model = DEFAULT_WHISPER_MODEL

        self.speed_pct = 100
        self.min_gap_ms = 350
        self.read_numbers = False
        self.spell_latin = False

        self.preset_rate: float | None = None
        self.preset_pitch: float | None = None
        self.preset_style: str | None = None
        self.preset_name: str = "None"
        self.presets: dict[str, dict] = {}

        self.music_path = ""
        self.music_db = -18.0
        self.duck_ratio = 8.0
        self.duck_thresh = 0.05

        self.last_phrases = []
        self.edited_text = None
        self.use_markers = True

        self.project_path = None
        self.segments = []
        self.qwen_editor = None
        if QwenEditor is not None:
            try:
                self.qwen_editor = QwenEditor()
            except Exception as err:
                log.warning("QwenEditor init failed: %s", err)

        self.vibe_editor = None
        if VibeEditor is not None:
            try:
                self.vibe_editor = VibeEditor()
            except Exception as err:
                log.warning("VibeEditor init failed: %s", err)

        # API keys for external services
        self.yandex_key = ""
        self.chatgpt_key = ""
        self.allow_beep_fallback = False
        self.auto_download_models = True
        (
            self.yandex_key,
            self.chatgpt_key,
            self.allow_beep_fallback,
            self.auto_download_models,
            self.out_dir,
            self.language,
            self.preset_name,
            self.whisper_model,
            self.speed_pct,
            self.min_gap_ms,
            self.read_numbers,
            self.spell_latin,
        ) = load_config()
        self.presets = load_presets(BASE_DIR / "presets")
        self._on_preset_change(self.preset_name)

        log.info("UI start. Version=%s", APP_VER)
        self._build_ui()

        if self.qwen_editor is None and self.vibe_editor is None:
            self.ai_edit_btn.setEnabled(False)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        file_menu = self.menuBar().addMenu("&Файл")
        self.action_open = QtGui.QAction("Открыть видео (Ctrl+O)", self)
        self.action_save = QtGui.QAction("Сохранить проект (Ctrl+S)", self)
        self.action_export = QtGui.QAction("Экспорт (Ctrl+E)", self)
        self.action_quit = QtGui.QAction("Выход", self)

        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_export)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        self.action_open.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.action_save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.action_export.setShortcut(QtGui.QKeySequence("Ctrl+E"))

        self.action_open.triggered.connect(self.open_video)
        self.action_save.triggered.connect(self.save_project)
        self.action_export.triggered.connect(self.export_media)
        self.action_quit.triggered.connect(self.close)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+A"), self, activated=self.select_all)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.preview_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.play_pause)

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.inp_video = QtWidgets.QLineEdit()
        btn_vid = QtWidgets.QPushButton("Обзор…")
        btn_vid.clicked.connect(self.pick_video)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.inp_video)
        h.addWidget(btn_vid)
        form.addRow("Видео (MP4):", h)

        self.cmb_engine = QtWidgets.QComboBox()
        self.cmb_engine.addItems(["silero", "yandex", "coqui_xtts", "gtts", "vibevoice"])
        self.cmb_engine.currentTextChanged.connect(self._on_engine_change)
        form.addRow("Движок TTS:", self.cmb_engine)

        self.cmb_voice = QtWidgets.QComboBox()
        form.addRow("Голос:", self.cmb_voice)

        hb = QtWidgets.QHBoxLayout()
        self.btn_rec = QtWidgets.QPushButton("1) Распознать")
        self.btn_edit = QtWidgets.QPushButton("2) Править (txt)")
        self.btn_run = QtWidgets.QPushButton("3) Озвучить")
        self.btn_settings = QtWidgets.QPushButton("Настройки")
        hb.addWidget(self.btn_rec)
        hb.addWidget(self.btn_edit)
        hb.addWidget(self.btn_run)
        hb.addStretch(1)
        hb.addWidget(self.btn_settings)
        layout.addLayout(hb)

        splitter = QtWidgets.QSplitter(self)
        layout.addWidget(splitter)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["start", "end", "original_text", "edited_text"])
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setPlaceholderText(
            "Полноэкранный редактор — редактируйте выделенный сегмент..."
        )
        right_layout.addWidget(self.editor)

        self.lang_list = QtWidgets.QListWidget(self)
        self.lang_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.lang_list.addItems(["ru", "en", "de", "fr"])
        right_layout.addWidget(self.lang_list)

        self.ai_edit_btn = QtWidgets.QPushButton("AI Edit", self)
        right_layout.addWidget(self.ai_edit_btn)

        splitter.addWidget(right_panel)

        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_table_menu)
        self.ai_edit_btn.clicked.connect(self.ai_edit_current_segment)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        # Привязка действий
        self.btn_rec.clicked.connect(self.recognize_only)
        self.btn_edit.clicked.connect(self.open_editor)
        self.btn_run.clicked.connect(self.start_render)
        self.btn_settings.clicked.connect(self.open_settings)

        self._on_engine_change(self.cmb_engine.currentText())
        self.log_print(f"Лог пишется в: {LOG_FILE}")

    # --- Логика переключения движков и загрузки голосов ---
    def _on_engine_change(self, engine: str):
        self._refresh_voices(engine)

    def _refresh_voices(self, engine: str):
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        self.cmb_voice.setVisible(True)
        if engine == "silero":
            self.cmb_voice.setEditable(False)
            voices = SILERO_VOICES.get(self.language, [])
            self.cmb_voice.addItems(voices)
        elif engine == "yandex":
            self.cmb_voice.setEditable(False)
            self.cmb_voice.addItems(YANDEX_VOICES)
        elif engine == "coqui_xtts":
            self.cmb_voice.setEditable(True)
        elif engine == "vibevoice":
            self.cmb_voice.setEditable(False)
            try:
                result = subprocess.run(
                    ["vibe-voice", "--list-speakers"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except FileNotFoundError:
                QMessageBox.warning(
                    self, "Missing binary", "vibe-voice executable not found. Install VibeVoice."
                )
            except subprocess.CalledProcessError as e:
                QMessageBox.warning(self, "vibe-voice error", str(e))
            else:
                voices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                self.cmb_voice.addItems(voices)
        elif engine == "gtts":
            # gTTS has no preset voices
            self.cmb_voice.setVisible(False)
        else:
            self.cmb_voice.setEditable(True)
        self.cmb_voice.blockSignals(False)

    def _on_preset_change(self, name: str) -> None:
        data = self.presets.get(name, {})
        self.preset_rate = data.get("rate")
        self.preset_pitch = data.get("pitch")
        self.preset_style = data.get("style")
        self.preset_name = name

    def _on_preset_change(self, name: str) -> None:
        data = self.presets.get(name, {})
        self.preset_rate = data.get("rate")
        self.preset_pitch = data.get("pitch")
        self.preset_style = data.get("style")
        self.preset_name = data.get("preset")

    # ---------- Хелперы ----------
    def log_print(self, *args):
        msg = " ".join(str(a) for a in args)
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QtWidgets.QApplication.processEvents()
        log.info(msg)

    def pick_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выбрать видео", str(INPUT_DIR), "Video (*.mp4 *.mkv *.mov)"
        )
        if p:
            self.video_path = p
            self.inp_video.setText(p)
            self.reset_state()
            self.log_print(f"Выбрано видео: {p}")


    def reset_state(self):
        self.last_phrases = []
        self.edited_text = None
        self.use_markers = True
        self.log_print("Состояние сброшено. Готово к новому видео.")

    # ---------- Действия ----------
    def open_settings(self):
        """Open dialog to configure advanced options and persist them."""
        models = list_models("stt")
        dlg = SettingsDialog(
            self,
            yandex_key=self.yandex_key,
            chatgpt_key=self.chatgpt_key,
            allow_beep_fallback=self.allow_beep_fallback,
            auto_download_models=self.auto_download_models,
            out_dir=self.out_dir,
            language=self.language,
            languages=sorted(SILERO_VOICES.keys()),
            preset=self.preset_name,
            presets=sorted(self.presets.keys()),
            whisper_model=self.whisper_model,
            whisper_models=list(models.keys()),
            speed_pct=self.speed_pct,
            min_gap_ms=self.min_gap_ms,
            read_numbers=self.read_numbers,
            spell_latin=self.spell_latin,
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            (
                self.yandex_key,
                self.chatgpt_key,
                self.allow_beep_fallback,
                self.auto_download_models,
                self.out_dir,
                self.language,
                self.preset_name,
                self.whisper_model,
                self.speed_pct,
                self.min_gap_ms,
                self.read_numbers,
                self.spell_latin,
            ) = dlg.get_settings()
            self._on_preset_change(self.preset_name)
            save_config(
                self.yandex_key,
                self.chatgpt_key,
                self.allow_beep_fallback,
                self.auto_download_models,
                self.out_dir,
                self.language,
                self.preset_name,
                self.whisper_model,
                self.speed_pct,
                self.min_gap_ms,
                self.read_numbers,
                self.spell_latin,
            )
            self._refresh_voices(self.cmb_engine.currentText())

    def show_help(self):
        """Show instructions for connecting Yandex API."""
        # Build HTML text with clickable links
        text = (
            "1. Зарегистрируйтесь в Yandex Cloud: "
            '<a href="https://cloud.yandex.ru">https://cloud.yandex.ru</a><br>'
            "2. В консоли создайте каталог и включите сервис SpeechKit.<br>"
            "3. Откройте меню 'Сервисные аккаунты' и создайте аккаунт.<br>"
            "4. На вкладке 'Ключи доступа' сформируйте API-ключ.<br>"
            "5. Вставьте ключ через кнопку 'Настройки' на главном экране.<br>"
            "Документация: "
            '<a href="https://cloud.yandex.ru/docs/speechkit/tts/quickstart">'
            "https://cloud.yandex.ru/docs/speechkit/tts/quickstart</a>"
        )

        # Configure message box to render HTML and allow external links
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Помощь")
        msg.setText(text)
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        label = msg.findChild(QtWidgets.QLabel, "qt_msgbox_label")
        if label:
            label.setOpenExternalLinks(True)  # enable clickable links
        msg.exec()

    def recognize_only(self):
        try:
            if not self.inp_video.text().strip():
                QtWidgets.QMessageBox.warning(self, "Нет видео", "Укажи путь к видео.")
                return
            ffmpeg = ensure_ffmpeg()
            self.log_print(f"FFmpeg: {ffmpeg}")
            self.log_print("Распознаю речь…")
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / "orig.wav"
                cmd = [
                    ffmpeg,
                    "-y",
                    "-i",
                    self.inp_video.text(),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-acodec",
                    "pcm_s16le",
                    str(wav),
                ]
                log.debug("Extract WAV cmd: %s", " ".join(map(str, cmd)))
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                segs = transcribe_whisper(
                    wav, language=self.language, model_size=self.whisper_model, device="cuda"
                )
                self.last_phrases = merge_into_phrases(segs)
            txt = " ".join(t for _, _, t in self.last_phrases)
            self.log_print("Текст:", txt[:700] + "..." if len(txt) > 700 else txt)
            self.log_print("ГОТОВО. Правка (txt).")
        except Exception as e:
            self.log_print(f"Ошибка распознавания: {e}")
            log.error("Traceback:\n%s", traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    def open_editor(self):
        if not self.last_phrases:
            self.recognize_only()
        if not self.last_phrases:
            return
        src = (
            phrases_to_marked_text(self.last_phrases)
            if self.use_markers
            else " ".join(t for _, _, t in self.last_phrases)
        )
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Правка текста")
        dlg.resize(1000, 640)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Редактируй. [[#i]] сохраняют тайминги. [[PAUSE=300]] пауза."))
        txt = QtWidgets.QPlainTextEdit()
        txt.setPlainText(self.edited_text if self.edited_text else src)
        v.addWidget(txt, 1)
        hb = QtWidgets.QHBoxLayout()
        btn_toggle = QtWidgets.QPushButton("Маркеры: ВКЛ")

        def toggle():
            self.use_markers = not self.use_markers
            btn_toggle.setText("Маркеры: ВКЛ" if self.use_markers else "Маркеры: ВЫКЛ")
            txt.setPlainText(
                phrases_to_marked_text(self.last_phrases)
                if self.use_markers
                else " ".join(t for _, _, t in self.last_phrases)
            )

        btn_toggle.clicked.connect(toggle)
        btn_save = QtWidgets.QPushButton("Сохранить")
        btn_save.clicked.connect(
            lambda: (setattr(self, "edited_text", txt.toPlainText().strip()), dlg.accept())
        )
        hb.addWidget(btn_toggle)
        hb.addStretch(1)
        hb.addWidget(btn_save)
        v.addLayout(hb)
        dlg.exec()
        self.log_print("Правки сохранены.")

    def start_render(self):
        try:
            if not self.inp_video.text().strip():
                QtWidgets.QMessageBox.warning(self, "Нет видео", "Укажи путь к видео.")
                return
            engine = self.cmb_engine.currentText()
            voice = self.cmb_voice.currentText()
            self.log_print(f"Озвучиваю… (engine={engine}, voice={voice})")
            out, fb_reason = revoice_video(
                self.inp_video.text(),
                self.out_dir,
                speaker=voice,
                whisper_size=self.whisper_model,
                device="cuda",
                sr=48000,
                min_gap_ms=self.min_gap_ms,
                speed_pct=self.speed_pct,
                edited_text=self.edited_text,
                phrases_cache=self.last_phrases if self.last_phrases else None,
                use_markers=self.use_markers,
                read_numbers=self.read_numbers,
                spell_latin=self.spell_latin,
                music_path=(self.music_path or None),
                music_db=self.music_db,
                duck_ratio=self.duck_ratio,
                duck_thresh=self.duck_thresh,
                tts_engine=engine,
                language=self.language,
                yandex_key=self.yandex_key,
                yandex_voice=voice,
                allow_beep_fallback=self.allow_beep_fallback,
                auto_download_models=self.auto_download_models,
                tts_rate=self.preset_rate,
                tts_pitch=self.preset_pitch,
                tts_style=self.preset_style,
                tts_preset=self.preset_name if self.preset_name != "None" else None,
            )
            if fb_reason:
                warn = f"Used beep fallback due to: {fb_reason}"
                self.log_print(warn)
                QtWidgets.QMessageBox.warning(self, "Предупреждение", warn)
            self.log_print("Готово:", out)
            QtWidgets.QMessageBox.information(self, "Готово", f"Сохранено:\n{out}")
        except Exception as e:
            self.log_print(f"Ошибка озвучивания: {e}")
            log.error("Traceback:\n%s", traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    # --- Segment editor methods ---
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", str(INPUT_DIR), "Video (*.mp4 *.mov *.mkv)"
        )
        if not path:
            return
        self.status.showMessage(f"Видео выбрано: {path}")
        # TODO: extract audio and run STT
        self.segments = [
            {"start": 0.0, "end": 2.0, "original_text": "Пример", "edited_text": "Пример"}
        ]
        self.reload_table()

    def save_project(self):
        if not self.project_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить проект", "projects/untitled.rvproj", "Revoice Project (*.rvproj)"
            )
            if not path:
                return
            self.project_path = path
        data = {"segments": self.segments}
        try:
            with open(self.project_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.status.showMessage(f"Сохранено: {self.project_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def export_media(self):
        """Export one file per selected language."""
        out_dir = QFileDialog.getExistingDirectory(self, "Экспорт", str(OUTPUT_DIR))
        if not out_dir:
            return
        langs = [item.text() for item in self.lang_list.selectedItems()] or ["ru"]
        for lang in langs:
            target = Path(out_dir) / f"output_{lang}.mp4"
            QMessageBox.information(self, "Экспорт", f"Видео будет сохранено: {target}")

    def undo(self):
        pass

    def redo(self):
        pass

    def select_all(self):
        self.editor.selectAll()

    def preview_segment(self):
        QMessageBox.information(self, "Предпрослушка", "Предпрослушка сегмента (будет добавлено)")

    def play_pause(self):
        pass

    def ai_edit_current_segment(self):
        row = self.table.currentRow()
        if row < 0:
            return
        text = self.editor.toPlainText().strip()
        if not text:
            return
        available_langs = [self.lang_list.item(i).text() for i in range(self.lang_list.count())]
        selected = {item.text() for item in self.lang_list.selectedItems()}

        editors = []
        if self.qwen_editor is not None:
            editors.append("qwen")
        if self.vibe_editor is not None:
            editors.append("vibevoice")
        if not editors:
            return

        async def run_ai(res: AiEditResult) -> None:
            editor_obj = self.qwen_editor if res.editor == "qwen" else self.vibe_editor
            if editor_obj is None:
                raise RuntimeError(f"Editor '{res.editor}' unavailable")
            data = await asyncio.to_thread(editor_obj.edit_text, text, res.languages)
            new_text = data.get("source", "")
            self.table.setItem(row, 3, QTableWidgetItem(new_text))
            self.editor.setPlainText(new_text)
            self.segments[row]["edited_text"] = new_text
            for lang in res.languages:
                self.segments[row][lang] = data.get(lang, "")

        dlg = AiEditDialog(self, languages=available_langs, editors=editors, runner=run_ai)
        for i in range(dlg.lang_list.count()):
            item = dlg.lang_list.item(i)
            item.setSelected(item.text() in selected)
        dlg.exec()

    def show_table_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        copy_orig = menu.addAction("Скопировать «оригинал»")
        reset_edit = menu.addAction("Сбросить правки")
        act = menu.exec(self.table.viewport().mapToGlobal(pos))
        row = self.table.currentRow()
        if row < 0:
            return
        if act == copy_orig:
            orig = self.table.item(row, 2).text() if self.table.item(row, 2) else ""
            self.table.setItem(row, 3, QTableWidgetItem(orig))
        elif act == reset_edit:
            self.table.setItem(
                row,
                3,
                QTableWidgetItem(self.table.item(row, 2).text() if self.table.item(row, 2) else ""),
            )

    def reload_table(self):
        self.table.setRowCount(0)
        for seg in self.segments:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, key in enumerate(["start", "end", "original_text", "edited_text"]):
                self.table.setItem(row, col, QTableWidgetItem(str(seg.get(key, ""))))


def main():
    """Run application or synthesize text via CLI."""

    parser = argparse.ArgumentParser(description="Revoice UI")
    parser.add_argument("--say", help="Text to synthesize and exit")
    args = parser.parse_args()

    if args.say:
        import numpy as np
        import soundfile as sf
        import torch

        from core.tts.registry import get_engine, registry

        engine = get_engine()
        engine_name = next((k for k, v in registry.items() if isinstance(engine, v)), "unknown")
        speaker = os.getenv("SILERO_SPEAKER") or "aidar"
        model_path = ""
        if engine_name == "silero":
            model_path = str(Path(torch.hub.get_dir()) / "snakers4_silero-models_master")
        wav = engine.synthesize(args.say, speaker, 48000)
        out_path = OUTPUT_DIR / "tts_test.wav"
        out_path.parent.mkdir(exist_ok=True)
        if isinstance(wav, np.ndarray):
            sf.write(out_path, wav, 48000)
        else:
            with open(out_path, "wb") as fh:
                fh.write(wav)
        print(
            f"TTS: engine={engine_name} model={model_path} speaker={speaker} -> {out_path.relative_to(BASE_DIR)}"
        )
        return 0

    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    exit_code = app.exec()
    logging.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
