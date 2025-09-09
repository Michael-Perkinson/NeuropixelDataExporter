import qtpy
import warnings
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from src.gui.view import MainWindow
from src.gui.gui_controller import GUIController

os.environ["QT_API"] = "pyside6"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.debug=false"
warnings.filterwarnings(                  
    "ignore",
    category=qtpy.PythonQtWarning,
    message=r"Selected binding .* could not be found",
)


if __name__ == "__main__":
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10)) 
    controller = GUIController()
    window = MainWindow(controller)
    window.show()
    app.exec()
