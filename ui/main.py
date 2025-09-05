# -*- coding: utf-8 -*-
# PySide6 UI — RevoicePortable alpha3
# Voice auto-detection removed
# Logs: BASE_DIR / logs / log_version_alpha3.txt

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from pathlib import Path
import argparse
import json
import os, subprocess, tempfile, traceback, logging, sys
from datetime import datetime

from .settings import SettingsDialog
from .config import load_config, save_config

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
    encoding="utf-8"
)
log = logging.getLogger("ui")

def handle_exception(exc_type, exc, tb):
    """Log uncaught exceptions before the application exits."""
    log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

# Register global exception hook
sys.excepthook = handle_exception

# Импорт пайплайна
from core.pipeline import (
    revoice_video, phrases_to_marked_text,
    transcribe_whisper, merge_into_phrases, ensure_ffmpeg
)
from core.model_manager import list_models
from core.tts_adapters import SILERO_VOICES
from core.qwen_editor import QwenEditor

YANDEX_VOICES = ["ermil","filipp","alena","jane","oksana","zahar","omazh","madirus"]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RevoicePortable — {APP_VER}")
        self.resize(1024, 680)

        # Состояние
        self.video_path = ""
        self.out_dir = str(OUTPUT_DIR)
        self.tts_engine = "silero"
        self.voice_id = "baya"
        self.language = "ru"

        self.speed_pct = 100
        self.min_gap_ms = 350
        self.speed_jitter = 0.03
        self.read_numbers = False
        self.spell_latin = False

        self.music_path = ""
        self.music_db = -18.0
        self.duck_ratio = 8.0
        self.duck_thresh = 0.05

        self.last_phrases = []
        self.edited_text = None
        self.use_markers = True

        self.project_path = None
        self.segments = []
        self.qwen_editor = QwenEditor()

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
        ) = load_config()

        log.info("UI start. Version=%s", APP_VER)
        self._build_ui()

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

        # Видео и папка вывода
        self.inp_video = QtWidgets.QLineEdit()
        btn_vid = QtWidgets.QPushButton("Обзор…"); btn_vid.clicked.connect(self.pick_video)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.inp_video); h.addWidget(btn_vid)
        form.addRow("Видео (MP4):", h)

        self.inp_out = QtWidgets.QLineEdit(self.out_dir)
        btn_out = QtWidgets.QPushButton("Обзор…"); btn_out.clicked.connect(self.pick_outdir)
        h2 = QtWidgets.QHBoxLayout(); h2.addWidget(self.inp_out); h2.addWidget(btn_out)
        form.addRow("Папка вывода:", h2)

        # Блок TTS и Whisper
        grid = QtWidgets.QGridLayout()
        row = 0

        self.cmb_engine = QtWidgets.QComboBox()
        # Include gTTS among available engines
        self.cmb_engine.addItems(["silero", "yandex", "coqui_xtts", "gtts"])
        self.cmb_engine.currentTextChanged.connect(self._on_engine_change)
        grid.addWidget(QtWidgets.QLabel("Движок TTS:"), row, 0); grid.addWidget(self.cmb_engine, row, 1)

        self.lbl_voice = QtWidgets.QLabel("Голос:")
        self.cmb_voice = QtWidgets.QComboBox()
        grid.addWidget(self.lbl_voice, row, 2); grid.addWidget(self.cmb_voice, row, 3)

        row += 1
        self.lbl_language = QtWidgets.QLabel("Язык:")
        self.cmb_language = QtWidgets.QComboBox()
        self.cmb_language.addItems(sorted(SILERO_VOICES))
        self.cmb_language.setCurrentText(self.language)
        self.cmb_language.currentTextChanged.connect(self._on_language_change)
        grid.addWidget(self.lbl_language, row, 0); grid.addWidget(self.cmb_language, row, 1)

        row += 1
        self.cmb_whisper = QtWidgets.QComboBox()
        self._populate_whisper_models()
        grid.addWidget(QtWidgets.QLabel("Whisper:"), row, 0); grid.addWidget(self.cmb_whisper, row, 1)
        self.ed_speed = QtWidgets.QLineEdit(str(self.speed_pct))
        grid.addWidget(QtWidgets.QLabel("Скорость, %:"), row, 2); grid.addWidget(self.ed_speed, row, 3)

        row += 1
        self.ed_mingap = QtWidgets.QLineEdit(str(self.min_gap_ms))
        grid.addWidget(QtWidgets.QLabel("Мин. пауза, мс:"), row, 0); grid.addWidget(self.ed_mingap, row, 1)
        self.ed_jitter = QtWidgets.QLineEdit(str(self.speed_jitter))
        grid.addWidget(QtWidgets.QLabel("Speed jitter:"), row, 2); grid.addWidget(self.ed_jitter, row, 3)

        row += 1
        self.chk_numbers = QtWidgets.QCheckBox("Числа словами"); self.chk_numbers.setChecked(self.read_numbers)
        self.chk_latin = QtWidgets.QCheckBox("Латиница по буквам"); self.chk_latin.setChecked(self.spell_latin)
        grid.addWidget(self.chk_numbers, row, 0); grid.addWidget(self.chk_latin, row, 1)

        layout.addLayout(grid)

        # Музыка
        group = QtWidgets.QGroupBox("Музыка (опционально)")
        g = QtWidgets.QGridLayout(group)
        self.ed_music = QtWidgets.QLineEdit()
        btn_ms = QtWidgets.QPushButton("Выбрать…"); btn_ms.clicked.connect(self.pick_music)
        g.addWidget(QtWidgets.QLabel("Файл:"), 0, 0); g.addWidget(self.ed_music, 0, 1); g.addWidget(btn_ms, 0, 2)
        self.ed_music_db = QtWidgets.QLineEdit(str(self.music_db))
        self.ed_duck_ratio = QtWidgets.QLineEdit(str(self.duck_ratio))
        self.ed_duck_thresh = QtWidgets.QLineEdit(str(self.duck_thresh))
        g.addWidget(QtWidgets.QLabel("Громкость, dB:"), 1, 0); g.addWidget(self.ed_music_db, 1, 1)
        g.addWidget(QtWidgets.QLabel("Duck ratio:"), 1, 2); g.addWidget(self.ed_duck_ratio, 1, 3)
        g.addWidget(QtWidgets.QLabel("Thresh:"), 1, 4); g.addWidget(self.ed_duck_thresh, 1, 5)
        layout.addWidget(group)

        # Кнопки
        hb = QtWidgets.QHBoxLayout()
        self.btn_rec = QtWidgets.QPushButton("1) Распознать")
        self.btn_edit = QtWidgets.QPushButton("2) Править (txt)")
        self.btn_run = QtWidgets.QPushButton("3) Озвучить")
        self.btn_reset = QtWidgets.QPushButton("Сброс")
        self.btn_open = QtWidgets.QPushButton("Открыть выход…")
        self.btn_help = QtWidgets.QPushButton("Помощь")
        self.btn_settings = QtWidgets.QPushButton("Настройки")
        hb.addWidget(self.btn_rec); hb.addWidget(self.btn_edit); hb.addWidget(self.btn_run)
        hb.addStretch(1); hb.addWidget(self.btn_help); hb.addWidget(self.btn_settings); hb.addWidget(self.btn_reset); hb.addWidget(self.btn_open)
        layout.addLayout(hb)

        # Segments editor
        splitter = QtWidgets.QSplitter(self)
        layout.addWidget(splitter)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["start", "end", "original_text", "edited_text"])
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setPlaceholderText("Полноэкранный редактор — редактируйте выделенный сегмент...")
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
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        # Привязка действий
        self.btn_rec.clicked.connect(self.recognize_only)
        self.btn_edit.clicked.connect(self.open_editor)
        self.btn_run.clicked.connect(self.start_render)
        self.btn_reset.clicked.connect(self.reset_state)
        self.btn_open.clicked.connect(self.open_outdir)
        self.btn_help.clicked.connect(self.show_help)
        self.btn_settings.clicked.connect(self.open_settings)

        self._on_engine_change(self.cmb_engine.currentText())
        self.log_print(f"Лог пишется в: {LOG_FILE}")

    # --- Логика переключения движков и загрузки голосов ---
    def _populate_whisper_models(self) -> None:
        models = list_models("stt")
        self.cmb_whisper.clear()
        for name, info in models.items():
            size = info.get("size_mb")
            desc = info.get("description", "")
            label = f"{name} ({size} MB - {desc})" if size else f"{name} - {desc}"
            self.cmb_whisper.addItem(label, userData=name)

    def _on_engine_change(self, engine: str):
        self._refresh_voices(engine)

    def _refresh_voices(self, engine: str):
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        self.lbl_voice.show()
        self.cmb_voice.show()
        if engine == "silero":
            self.lbl_voice.setText("Silero голос:")
            self.cmb_voice.setEditable(False)
            self.lbl_language.show()
            self.cmb_language.show()
            self._on_language_change(self.cmb_language.currentText())
        elif engine == "yandex":
            self.lbl_voice.setText("Yandex голос:")
            self.cmb_voice.setEditable(False)
            self.cmb_voice.addItems(YANDEX_VOICES)
            self.lbl_language.hide()
            self.cmb_language.hide()
        elif engine == "coqui_xtts":
            self.lbl_voice.setText("Coqui speaker:")
            # Allow manual entry of speaker reference folder
            self.cmb_voice.setEditable(True)
            self.lbl_language.hide()
            self.cmb_language.hide()
        elif engine == "gtts":
            # gTTS has no preset voices, so hide the selector
            self.lbl_voice.hide()
            self.cmb_voice.hide()
            self.lbl_language.hide()
            self.cmb_language.hide()
        else:
            self.lbl_voice.setText("Голос:")
            self.cmb_voice.setEditable(True)
            self.lbl_language.hide()
            self.cmb_language.hide()
        self.cmb_voice.blockSignals(False)

    def _on_language_change(self, lang: str):
        self.language = lang
        voices = SILERO_VOICES.get(lang, [])
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        self.cmb_voice.addItems(voices)
        self.cmb_voice.blockSignals(False)

    # ---------- Хелперы ----------
    def log_print(self, *args):
        msg = " ".join(str(a) for a in args)
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QtWidgets.QApplication.processEvents()
        log.info(msg)

    def pick_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать видео", str(INPUT_DIR), "Video (*.mp4 *.mkv *.mov)")
        if p: self.video_path = p; self.inp_video.setText(p); self.reset_state(); self.log_print(f"Выбрано видео: {p}")

    def pick_outdir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Папка вывода", self.inp_out.text() or str(OUTPUT_DIR))
        if p: self.out_dir = p; self.inp_out.setText(p); self.log_print(f"Папка вывода: {p}")

    def pick_music(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Файл музыки", "", "Audio (*.mp3 *.wav *.flac *.m4a *.ogg)")
        if p: self.music_path = p; self.ed_music.setText(p); self.log_print(f"Музыка: {p}")

    def open_outdir(self):
        p = Path(self.inp_out.text() or ".").resolve()
        if p.exists(): os.startfile(str(p))

    def reset_state(self):
        self.last_phrases = []; self.edited_text = None; self.use_markers = True
        self.log_print("Состояние сброшено. Готово к новому видео.")

    # ---------- Действия ----------
    def open_settings(self):
        """Open dialog to configure API keys and persist them."""
        dlg = SettingsDialog(
            self,
            self.yandex_key,
            self.chatgpt_key,
            self.allow_beep_fallback,
            self.auto_download_models,
        )
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            (
                self.yandex_key,
                self.chatgpt_key,
                self.allow_beep_fallback,
                self.auto_download_models,
            ) = dlg.get_keys()
            save_config(
                self.yandex_key,
                self.chatgpt_key,
                self.allow_beep_fallback,
                self.auto_download_models,
            )

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
                QtWidgets.QMessageBox.warning(self, "Нет видео", "Укажи путь к видео."); return
            ffmpeg = ensure_ffmpeg(); self.log_print(f"FFmpeg: {ffmpeg}"); self.log_print("Распознаю речь…")
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td)/"orig.wav"
                cmd = [ffmpeg,"-y","-i",self.inp_video.text(),"-vn","-ac","1","-ar","48000","-acodec","pcm_s16le",str(wav)]
                log.debug("Extract WAV cmd: %s"," ".join(map(str,cmd)))
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                segs = transcribe_whisper(
                    wav, language="ru", model_size=self.cmb_whisper.currentData(), device="cuda"
                )
                self.last_phrases = merge_into_phrases(segs)
            txt = " ".join(t for _,_,t in self.last_phrases)
            self.log_print("Текст:", txt[:700]+"..." if len(txt)>700 else txt); self.log_print("ГОТОВО. Правка (txt).")
        except Exception as e:
            self.log_print(f"Ошибка распознавания: {e}"); log.error("Traceback:\n%s",traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    def open_editor(self):
        if not self.last_phrases: self.recognize_only()
        if not self.last_phrases: return
        src = phrases_to_marked_text(self.last_phrases) if self.use_markers else " ".join(t for _,_,t in self.last_phrases)
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Правка текста"); dlg.resize(1000,640)
        v = QtWidgets.QVBoxLayout(dlg); v.addWidget(QtWidgets.QLabel("Редактируй. [[#i]] сохраняют тайминги. [[PAUSE=300]] пауза."))
        txt = QtWidgets.QPlainTextEdit(); txt.setPlainText(self.edited_text if self.edited_text else src); v.addWidget(txt,1)
        hb = QtWidgets.QHBoxLayout(); btn_toggle = QtWidgets.QPushButton("Маркеры: ВКЛ")
        def toggle():
            self.use_markers = not self.use_markers
            btn_toggle.setText("Маркеры: ВКЛ" if self.use_markers else "Маркеры: ВЫКЛ")
            txt.setPlainText(phrases_to_marked_text(self.last_phrases) if self.use_markers else " ".join(t for _,_,t in self.last_phrases))
        btn_toggle.clicked.connect(toggle)
        btn_save = QtWidgets.QPushButton("Сохранить"); btn_save.clicked.connect(lambda: (setattr(self,"edited_text",txt.toPlainText().strip()),dlg.accept()))
        hb.addWidget(btn_toggle); hb.addStretch(1); hb.addWidget(btn_save); v.addLayout(hb)
        dlg.exec(); self.log_print("Правки сохранены.")

    def start_render(self):
        try:
            if not self.inp_video.text().strip():
                QtWidgets.QMessageBox.warning(self,"Нет видео","Укажи путь к видео."); return
            engine = self.cmb_engine.currentText(); voice = self.cmb_voice.currentText(); language = self.cmb_language.currentText()
            self.log_print(f"Озвучиваю… (engine={engine}, voice={voice})")
            # Forward engine choice (incl. coqui_xtts) to the synthesis pipeline
            out, fb_reason = revoice_video(
                self.inp_video.text(), self.inp_out.text(),
                speaker=voice, whisper_size=self.cmb_whisper.currentData(), device="cuda",
                sr=48000, min_gap_ms=int(self.ed_mingap.text() or "350"),
                speed_pct=max(50,min(200,int(self.ed_speed.text() or "100"))),
                edited_text=self.edited_text, phrases_cache=self.last_phrases if self.last_phrases else None,
                use_markers=self.use_markers, read_numbers=self.chk_numbers.isChecked(),
                spell_latin=self.chk_latin.isChecked(), music_path=(self.ed_music.text().strip() or None),
                music_db=float(self.ed_music_db.text() or "-18"), duck_ratio=float(self.ed_duck_ratio.text() or "8.0"),
                duck_thresh=float(self.ed_duck_thresh.text() or "0.05"), tts_engine=engine,
                language=language,
                yandex_key=self.yandex_key, yandex_voice=voice,
                speed_jitter=float(self.ed_jitter.text() or "0.03"),
                allow_beep_fallback=self.allow_beep_fallback,
                auto_download_models=self.auto_download_models,
            )
            if fb_reason:
                warn = f"Used beep fallback due to: {fb_reason}"
                self.log_print(warn)
                QtWidgets.QMessageBox.warning(self, "Предупреждение", warn)
            self.log_print("Готово:", out)
            QtWidgets.QMessageBox.information(self,"Готово",f"Сохранено:\n{out}")
        except Exception as e:
            self.log_print(f"Ошибка озвучивания: {e}"); log.error("Traceback:\n%s",traceback.format_exc())
            QtWidgets.QMessageBox.critical(self,"Ошибка",str(e))

    # --- Segment editor methods ---
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", str(INPUT_DIR), "Video (*.mp4 *.mov *.mkv)")
        if not path:
            return
        self.status.showMessage(f"Видео выбрано: {path}")
        # TODO: extract audio and run STT
        self.segments = [{"start": 0.0, "end": 2.0, "original_text": "Пример", "edited_text": "Пример"}]
        self.reload_table()

    def save_project(self):
        if not self.project_path:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "projects/untitled.rvproj", "Revoice Project (*.rvproj)")
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
        langs = [item.text() for item in self.lang_list.selectedItems()] or ["ru"]
        result = self.qwen_editor.edit_text(text, langs)
        new_text = result.get("source", "")
        self.table.setItem(row, 3, QTableWidgetItem(new_text))
        self.editor.setPlainText(new_text)
        self.segments[row]["edited_text"] = new_text
        for lang in langs:
            self.segments[row][lang] = result.get(lang, "")

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
            self.table.setItem(row, 3, QTableWidgetItem(self.table.item(row, 2).text() if self.table.item(row, 2) else ""))

    def reload_table(self):
        self.table.setRowCount(0)
        for seg in self.segments:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, key in enumerate(["start", "end", "original_text", "edited_text"]):
                self.table.setItem(row, col, QTableWidgetItem(str(seg.get(key, ""))))

def main():
    """Run application or synthesize text via CLI."""

    parser = argparse.ArgumentParser(description="RevoicePortable UI")
    parser.add_argument("--say", help="Text to synthesize and exit")
    args = parser.parse_args()

    if args.say:
        from core.tts_registry import get_engine, registry
        import torch

        engine_fn = get_engine()
        engine_name = next((k for k, v in registry.items() if v is engine_fn), "unknown")
        speaker = os.getenv("SILERO_SPEAKER") or "aidar"
        model_path = ""
        if engine_name == "silero":
            model_path = str(Path(torch.hub.get_dir()) / "snakers4_silero-models_master")
        wav = engine_fn(args.say, speaker, 48000)
        out_path = OUTPUT_DIR / "tts_test.wav"
        out_path.parent.mkdir(exist_ok=True)
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
