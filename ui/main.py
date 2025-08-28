# -*- coding: utf-8 -*-
# PySide6 UI — RevoicePortable alpha3
# Voice auto-detection removed
# Logs: BASE_DIR / logs / log_version_alpha3.txt

from PySide6 import QtWidgets, QtGui, QtCore
from pathlib import Path
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

# Предустановленные голоса
SILERO_PRESETS = ["baya", "kseniya", "aidar", "eugene", "random"]
YANDEX_VOICES = ["ermil","filipp","alena","jane","oksana","zahar","omazh","madirus"]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RevoicePortable — {APP_VER}")
        self.resize(1024, 680)

        # Состояние
        self.video_path = ""
        self.out_dir = str(OUTPUT_DIR)
        self.whisper_model = "large-v3"
        self.tts_engine = "silero"
        self.voice_id = "baya"

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

        # API keys for external services
        self.yandex_key = ""
        self.chatgpt_key = ""
        self.yandex_key, self.chatgpt_key = load_config()  # load stored keys

        log.info("UI start. Version=%s", APP_VER)
        self._build_ui()

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
        self.cmb_whisper = QtWidgets.QComboBox(); self.cmb_whisper.addItems(["large-v3","medium","small"])
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

        # Лог
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

        # Горячие клавиши
        for key in ("Ctrl+O","Ctrl+S","Ctrl+E","Ctrl+Z","Ctrl+Y","Space","Ctrl+Enter"):
            act = QtGui.QAction(self); act.setShortcut(QtGui.QKeySequence(key)); self.addAction(act)

        self._on_engine_change(self.cmb_engine.currentText())
        self.log_print(f"Лог пишется в: {LOG_FILE}")

    # --- Логика переключения движков и загрузки голосов ---
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
            self.cmb_voice.addItems(SILERO_PRESETS)
        elif engine == "yandex":
            self.lbl_voice.setText("Yandex голос:")
            self.cmb_voice.setEditable(False)
            self.cmb_voice.addItems(YANDEX_VOICES)
        elif engine == "coqui_xtts":
            self.lbl_voice.setText("Coqui speaker:")
            # Allow manual entry of speaker reference folder
            self.cmb_voice.setEditable(True)
        elif engine == "gtts":
            # gTTS has no preset voices, so hide the selector
            self.lbl_voice.hide()
            self.cmb_voice.hide()
        else:
            self.lbl_voice.setText("Голос:")
            self.cmb_voice.setEditable(True)
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
        dlg = SettingsDialog(self, self.yandex_key, self.chatgpt_key)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.yandex_key, self.chatgpt_key = dlg.get_keys()
            save_config(self.yandex_key, self.chatgpt_key)

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
                segs = transcribe_whisper(wav, language="ru", model_size=self.cmb_whisper.currentText(), device="cuda")
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
            engine = self.cmb_engine.currentText(); voice = self.cmb_voice.currentText()
            self.log_print(f"Озвучиваю… (engine={engine}, voice={voice})")
            # Forward engine choice (incl. coqui_xtts) to the synthesis pipeline
            out = revoice_video(
                self.inp_video.text(), self.inp_out.text(),
                speaker=voice, whisper_size=self.cmb_whisper.currentText(), device="cuda",
                sr=48000, min_gap_ms=int(self.ed_mingap.text() or "350"),
                speed_pct=max(50,min(200,int(self.ed_speed.text() or "100"))),
                edited_text=self.edited_text, phrases_cache=self.last_phrases if self.last_phrases else None,
                use_markers=self.use_markers, read_numbers=self.chk_numbers.isChecked(),
                spell_latin=self.chk_latin.isChecked(), music_path=(self.ed_music.text().strip() or None),
                music_db=float(self.ed_music_db.text() or "-18"), duck_ratio=float(self.ed_duck_ratio.text() or "8.0"),
                duck_thresh=float(self.ed_duck_thresh.text() or "0.05"), tts_engine=engine,
                yandex_key=self.yandex_key, yandex_voice=voice,
                speed_jitter=float(self.ed_jitter.text() or "0.03")
            )
            self.log_print("Готово:", out); QtWidgets.QMessageBox.information(self,"Готово",f"Сохранено:\n{out}")
        except Exception as e:
            self.log_print(f"Ошибка озвучивания: {e}"); log.error("Traceback:\n%s",traceback.format_exc())
            QtWidgets.QMessageBox.critical(self,"Ошибка",str(e))

def main():
    """Run application and ensure logs are flushed on exit."""
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    exit_code = app.exec()
    logging.shutdown()
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
