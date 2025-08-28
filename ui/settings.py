from PySide6 import QtWidgets

class SettingsDialog(QtWidgets.QDialog):
    """Dialog to configure API keys for external services."""

    def __init__(self, parent=None, yandex_key: str = "", chatgpt_key: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Настройки API")

        form = QtWidgets.QFormLayout(self)
        # Yandex API key input
        self.ed_yandex = QtWidgets.QLineEdit(yandex_key)
        form.addRow("Yandex key:", self.ed_yandex)
        # ChatGPT API key input placeholder
        self.ed_chatgpt = QtWidgets.QLineEdit(chatgpt_key)
        form.addRow("ChatGPT key:", self.ed_chatgpt)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def get_keys(self) -> tuple[str, str]:
        """Return the API keys entered by the user."""
        return self.ed_yandex.text().strip(), self.ed_chatgpt.text().strip()
