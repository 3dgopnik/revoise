from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from PySide6 import QtWidgets


@dataclass
class AiEditResult:
    """Result returned by :class:`AiEditDialog`."""

    editor: str
    instruction: str
    languages: list[str]
    fix_grammar: bool
    stress_marks: bool
    insert_pause: bool


class AiEditDialog(QtWidgets.QDialog):
    """Dialog to craft instructions for AI text editing."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        languages: list[str] | None = None,
        editors: list[str] | None = None,
        runner: Callable[[AiEditResult], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI правка")
        self._runner = runner
        self._task: asyncio.Task[None] | None = None
        self._result: AiEditResult | None = None

        main = QtWidgets.QVBoxLayout(self)

        editor_row = QtWidgets.QHBoxLayout()
        editor_row.addWidget(QtWidgets.QLabel("Editor:"))
        self.editor_combo = QtWidgets.QComboBox()
        for name in editors or ["qwen"]:
            self.editor_combo.addItem(name)
        editor_row.addWidget(self.editor_combo)
        editor_row.addStretch(1)
        main.addLayout(editor_row)

        main.addWidget(QtWidgets.QLabel("Инструкция:"))
        self.instruction_edit = QtWidgets.QTextEdit()
        main.addWidget(self.instruction_edit)

        opts = QtWidgets.QHBoxLayout()
        self.chk_grammar = QtWidgets.QCheckBox("Fix grammar")
        self.chk_stress = QtWidgets.QCheckBox("Stress marks (+)")
        self.chk_pause = QtWidgets.QCheckBox("Insert [[PAUSE=ms]]")
        opts.addWidget(self.chk_grammar)
        opts.addWidget(self.chk_stress)
        opts.addWidget(self.chk_pause)
        opts.addStretch(1)
        main.addLayout(opts)

        self.lang_list = QtWidgets.QListWidget()
        self.lang_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for lang in languages or ["RU", "EN", "DE", "FR", "ES"]:
            self.lang_list.addItem(lang)
        main.addWidget(self.lang_list)

        presets = QtWidgets.QHBoxLayout()
        btn_stress = QtWidgets.QPushButton("Расставь ударения")
        btn_translate = QtWidgets.QPushButton("Переведи на EN")
        btn_grammar = QtWidgets.QPushButton("Только исправь грамматику")
        presets.addWidget(btn_stress)
        presets.addWidget(btn_translate)
        presets.addWidget(btn_grammar)
        presets.addStretch(1)
        main.addLayout(presets)

        btn_stress.clicked.connect(self._preset_stress)
        btn_translate.clicked.connect(self._preset_translate)
        btn_grammar.clicked.connect(self._preset_grammar)

        self.status_label = QtWidgets.QLabel()
        main.addWidget(self.status_label)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.run_button = QtWidgets.QPushButton("Run")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        bottom.addWidget(self.run_button)
        bottom.addWidget(self.cancel_button)
        main.addLayout(bottom)

        self.run_button.clicked.connect(self._on_run)
        self.cancel_button.clicked.connect(self.reject)

    # Presets -----------------------------------------------------------------
    def _select_language(self, lang: str) -> None:
        for i in range(self.lang_list.count()):
            item = self.lang_list.item(i)
            item.setSelected(item.text().upper() == lang.upper())

    def _preset_stress(self) -> None:
        self.instruction_edit.setPlainText("Расставь ударения")
        self.chk_stress.setChecked(True)
        self.chk_grammar.setChecked(False)
        self.chk_pause.setChecked(False)
        self._select_language("RU")

    def _preset_translate(self) -> None:
        self.instruction_edit.setPlainText("Переведи на EN")
        self.chk_grammar.setChecked(True)
        self.chk_stress.setChecked(False)
        self.chk_pause.setChecked(False)
        self._select_language("EN")

    def _preset_grammar(self) -> None:
        self.instruction_edit.setPlainText("Только исправь грамматику")
        self.chk_grammar.setChecked(True)
        self.chk_stress.setChecked(False)
        self.chk_pause.setChecked(False)

    # Run ---------------------------------------------------------------------
    def _collect(self) -> AiEditResult:
        langs = [i.text() for i in self.lang_list.selectedItems()]
        return AiEditResult(
            editor=self.editor_combo.currentText(),
            instruction=self.instruction_edit.toPlainText().strip(),
            languages=langs,
            fix_grammar=self.chk_grammar.isChecked(),
            stress_marks=self.chk_stress.isChecked(),
            insert_pause=self.chk_pause.isChecked(),
        )

    def _on_run(self) -> None:
        if self._task and not self._task.done():
            return
        self.run_button.setEnabled(False)
        self.status_label.setText("Работаю…")
        data = self._collect()

        async def runner() -> None:
            if self._runner:
                await self._runner(data)
            self._result = data

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(runner())
        self._task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task[None]) -> None:
        if task.exception():
            self.status_label.setText(f"Ошибка: {task.exception()}")
        else:
            self.status_label.setText("Готово")
            self.accept()
        self.run_button.setEnabled(True)

    # API ---------------------------------------------------------------------
    def result(self) -> AiEditResult | None:
        """Return the dialog result after processing."""
        return self._result
