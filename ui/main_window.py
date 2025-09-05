import json
import pathlib
import sys

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTableWidgetItem

from core.qwen_editor import QwenEditor

# Default project directories
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RevoicePortable — MVP")
        self.resize(1100, 700)

        # Central splitter
        splitter = QtWidgets.QSplitter(self)
        self.setCentralWidget(splitter)

        # Left: segments table
        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["start", "end", "original_text", "edited_text"])
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        # Right panel with editor and language selector
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

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Готово")

        # Menu + actions
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

        # Shortcuts
        self.action_open.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.action_save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.action_export.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+A"), self, activated=self.select_all)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self, activated=self.preview_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.play_pause)

        # Connections
        self.action_open.triggered.connect(self.open_video)
        self.action_save.triggered.connect(self.save_project)
        self.action_export.triggered.connect(self.export_media)
        self.action_quit.triggered.connect(self.close)

        # Context menu for table
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_table_menu)

        # basic data
        self.project_path = None
        self.segments = []
        self.qwen_editor = QwenEditor()

        self.ai_edit_btn.clicked.connect(self.ai_edit_current_segment)

    # --- basic ops ---
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", str(INPUT_DIR), "Video (*.mp4 *.mov *.mkv)")
        if not path:
            return
        self.status.showMessage(f"Видео выбрано: {path}")
        # TODO: extract audio and run STT
        self.segments = [{"start":0.0,"end":2.0,"original_text":"Пример","edited_text":"Пример"}]
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
            target = pathlib.Path(out_dir) / f"output_{lang}.mp4"
            QMessageBox.information(self, "Экспорт", f"Видео будет сохранено: {target}")

    def undo(self): pass
    def redo(self): pass
    def select_all(self):
        self.editor.selectAll()

    def preview_segment(self):
        # TODO: simple TTS preview of selected segment
        QMessageBox.information(self, "Предпрослушка", "Предпрослушка сегмента (будет добавлено)")

    def play_pause(self):
        # TODO: play/pause preview
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
            orig = self.table.item(row, 2).text() if self.table.item(row,2) else ""
            self.table.setItem(row, 3, QTableWidgetItem(orig))
        elif act == reset_edit:
            self.table.setItem(row, 3, QTableWidgetItem(self.table.item(row,2).text() if self.table.item(row,2) else ""))

    def reload_table(self):
        self.table.setRowCount(0)
        for seg in self.segments:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, key in enumerate(["start","end","original_text","edited_text"]):
                self.table.setItem(row, col, QTableWidgetItem(str(seg.get(key, ""))))

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
