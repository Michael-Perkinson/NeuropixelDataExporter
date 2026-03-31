from __future__ import annotations

import os
import sys
import warnings


def _configure_qt_environment() -> None:
    # Must be set before importing qtpy / any Qt bindings
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.debug=false")


def _suppress_qtpy_binding_warning() -> None:
    """
    qtpy may emit a warning like:
    'Selected binding ... could not be found'
    when it probes bindings. We can safely ignore that in our app.
    """
    warnings.filterwarnings(
        "ignore",
        category=Warning,  # avoid importing qtpy just for PythonQtWarning
        message=r"Selected binding .* could not be found",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_qt_environment()
    _suppress_qtpy_binding_warning()

    # Import Qt after env + warnings are configured
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from src.gui.gui_controller import GUIController
    from src.gui.view import MainWindow

    app = QApplication(argv if argv is not None else [])
    app.setFont(QFont("Segoe UI", 10))

    controller = GUIController()
    window = MainWindow(controller)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
