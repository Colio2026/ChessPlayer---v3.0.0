import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def run_app(config: dict) -> None:
    app = QApplication(sys.argv)
    win = MainWindow(config=config)
    win.show()
    sys.exit(app.exec())