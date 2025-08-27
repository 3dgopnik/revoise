from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
import json, sys, pathlib

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

        # Right: editor
        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setPlaceholderText("Полноэкранный редактор — редактируйте выделенный сегмент...")
        splitter.addWidget(self.editor)

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

    # --- basic ops ---
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video (*.mp4 *.mov *.mkv)")
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
        # TODO: call core.pipeline to render with ffmpeg mux
        QMessageBox.information(self, "Экспорт", "Экспорт будет реализован в core.pipeline.render()")

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

    def show_table_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        copy_orig = menu.addAction("Скопировать «оригинал»")
        reset_edit = menu.addAction("Сбросить правки")
        act = menu.exec(self.table.viewport().mapToGlobal(pos))
        row = self.table.currentRow()
        if row < 0: return
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
