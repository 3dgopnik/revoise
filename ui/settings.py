from PySide6 import QtWidgets

class SettingsDialog(QtWidgets.QDialog):
    """Dialog to configure API keys and options."""

    def __init__(
        self,
        parent=None,
        yandex_key: str = "",
        chatgpt_key: str = "",
        allow_beep_fallback: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Настройки API")

        form = QtWidgets.QFormLayout(self)
        self.ed_yandex = QtWidgets.QLineEdit(yandex_key)
        form.addRow("Yandex key:", self.ed_yandex)
        self.ed_chatgpt = QtWidgets.QLineEdit(chatgpt_key)
        form.addRow("ChatGPT key:", self.ed_chatgpt)
        self.chk_beep = QtWidgets.QCheckBox("Allow beep fallback")
        self.chk_beep.setChecked(allow_beep_fallback)
        form.addRow(self.chk_beep)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def get_keys(self) -> tuple[str, str, bool]:
        """Return the API keys and options entered by the user."""
        return (
            self.ed_yandex.text().strip(),
            self.ed_chatgpt.text().strip(),
            self.chk_beep.isChecked(),
        )
