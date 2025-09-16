from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from .config import DEFAULT_WHISPER_MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


class SettingsDialog(QtWidgets.QDialog):
    """Dialog to configure advanced options and API keys."""

    def __init__(
        self,
        parent=None,
        yandex_key: str = "",
        chatgpt_key: str = "",
        allow_beep_fallback: bool = False,
        auto_download_models: bool = True,
        out_dir: str = str(OUTPUT_DIR),
        language: str = "ru",
        languages: list[str] | None = None,
        preset: str = "None",
        presets: list[str] | None = None,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        whisper_models: list[str] | None = None,
        speed_pct: int = 100,
        min_gap_ms: int = 350,
        read_numbers: bool = False,
        spell_latin: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Настройки")

        form = QtWidgets.QFormLayout(self)

        # Output directory
        self.ed_out = QtWidgets.QLineEdit(out_dir)
        btn_out = QtWidgets.QPushButton("Обзор…")
        btn_out.clicked.connect(self._pick_outdir)
        h_out = QtWidgets.QHBoxLayout()
        h_out.addWidget(self.ed_out)
        h_out.addWidget(btn_out)
        form.addRow("Папка вывода:", h_out)

        # Language
        self.cmb_language = QtWidgets.QComboBox()
        for lang in languages or []:
            self.cmb_language.addItem(lang)
        self.cmb_language.setCurrentText(language)
        form.addRow("Язык:", self.cmb_language)

        # Preset
        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.addItem("None")
        for name in presets or []:
            self.cmb_preset.addItem(name)
        self.cmb_preset.setCurrentText(preset)
        form.addRow("Preset:", self.cmb_preset)

        # Whisper model
        self.cmb_whisper = QtWidgets.QComboBox()
        for name in whisper_models or []:
            self.cmb_whisper.addItem(name)
        self.cmb_whisper.setCurrentText(whisper_model)
        form.addRow("Whisper:", self.cmb_whisper)

        # Advanced numeric options
        self.ed_speed = QtWidgets.QLineEdit(str(speed_pct))
        form.addRow("Скорость, %:", self.ed_speed)
        self.ed_mingap = QtWidgets.QLineEdit(str(min_gap_ms))
        form.addRow("Мин. пауза, мс:", self.ed_mingap)

        self.chk_numbers = QtWidgets.QCheckBox("Числа словами")
        self.chk_numbers.setChecked(read_numbers)
        form.addRow(self.chk_numbers)

        self.chk_latin = QtWidgets.QCheckBox("Латиница по буквам")
        self.chk_latin.setChecked(spell_latin)
        form.addRow(self.chk_latin)

        # API keys
        form.addRow(QtWidgets.QLabel("API"))
        self.ed_yandex = QtWidgets.QLineEdit(yandex_key)
        form.addRow("Yandex key:", self.ed_yandex)
        self.ed_chatgpt = QtWidgets.QLineEdit(chatgpt_key)
        form.addRow("ChatGPT key:", self.ed_chatgpt)
        self.chk_beep = QtWidgets.QCheckBox("Allow beep fallback")
        self.chk_beep.setChecked(allow_beep_fallback)
        form.addRow(self.chk_beep)
        self.chk_auto = QtWidgets.QCheckBox("Auto-download models")
        self.chk_auto.setChecked(auto_download_models)
        form.addRow(self.chk_auto)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def _pick_outdir(self) -> None:
        p = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Папка вывода", self.ed_out.text() or str(OUTPUT_DIR)
        )
        if p:
            self.ed_out.setText(p)

    def get_settings(self) -> tuple[
        str,
        str,
        bool,
        bool,
        str,
        str,
        str,
        str,
        int,
        int,
        bool,
        bool,
    ]:
        """Return updated configuration values."""
        return (
            self.ed_yandex.text().strip(),
            self.ed_chatgpt.text().strip(),
            self.chk_beep.isChecked(),
            self.chk_auto.isChecked(),
            self.ed_out.text().strip(),
            self.cmb_language.currentText(),
            self.cmb_preset.currentText(),
            self.cmb_whisper.currentText(),
            int(self.ed_speed.text() or "100"),
            int(self.ed_mingap.text() or "350"),
            self.chk_numbers.isChecked(),
            self.chk_latin.isChecked(),
        )

