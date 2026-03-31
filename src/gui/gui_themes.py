# gui_themes.py

from typing import Any
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import qdarkstyle


def _toggle_theme(main_window: QMainWindow) -> None:
    if main_window.dark_mode:
        main_window.setStyleSheet(_light_theme())
    else:
        main_window.setStyleSheet(_dark_theme())

    main_window.dark_mode = not main_window.dark_mode

    # Optional: refresh help icon style
    if hasattr(main_window, "help_icon"):
        main_window.help_icon.setStyleSheet(
            help_icon_style(main_window.dark_mode))


def _light_theme() -> str:
    base_bg = "#f0f2f5"
    panel_bg = "#ffffff"
    panel_border = "#c4c8d0"
    text_fg = "#1a1a2e"
    hint_fg = "#4a4a6a"
    accent = "#0077b6"
    accent_hover = "#005f8e"
    header_bg = "#e2e6ed"
    header_fg = "#1a1a2e"
    selection_bg = "#b8d8f0"
    tooltip_bg = "#eaf4fb"
    tab_active_bg = "#ffffff"
    tab_inactive_bg = "#dde2ea"
    cb_border = "#7a8599"

    return f"""
    QWidget {{
        font-size: 9pt;
        background-color: {base_bg};
        color: {text_fg};
    }}

    /* ---------- GroupBox ------------------------------------------- */
    QGroupBox {{
        font-weight: bold;
        font-size: 9pt;
        border: 1px solid {panel_border};
        border-radius: 6px;
        margin-top: 12px;
        background-color: transparent;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {hint_fg};
    }}

    /* ---------- PushButton ----------------------------------------- */
    QPushButton {{
        background-color: {accent};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 6px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {accent_hover};
    }}
    QPushButton:disabled {{
        background-color: #b0b8c8;
        color: #f0f0f0;
    }}

    /* ---------- LineEdit / TextEdit -------------------------------- */
    QLineEdit, QTextEdit {{
        background-color: {panel_bg};
        border: 1px solid {panel_border};
        border-radius: 5px;
        padding: 4px 6px;
        color: {text_fg};
    }}
    QLineEdit:focus, QTextEdit:focus {{
        border: 1.5px solid {accent};
    }}

    /* ---------- CheckBox ------------------------------------------- */
    QCheckBox {{
        color: {text_fg};
        spacing: 6px;
        font-size: 9pt;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 2px solid {cb_border};
        border-radius: 3px;
        background-color: {panel_bg};
    }}
    QCheckBox::indicator:checked {{
        background-color: {accent};
        border-color: {accent};
        image: url(:/qt-project.org/styles/commonstyle/images/checkbox_checked-16.png);
    }}
    QCheckBox::indicator:hover {{
        border-color: {accent};
    }}

    /* ---------- TabWidget / TabBar --------------------------------- */
    QTabWidget::pane {{
        border: 1px solid {panel_border};
        border-radius: 4px;
        background-color: {tab_active_bg};
    }}
    QTabBar::tab {{
        background-color: {tab_inactive_bg};
        color: {hint_fg};
        border: 1px solid {panel_border};
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        padding: 5px 14px;
        margin-right: 2px;
        font-size: 9pt;
    }}
    QTabBar::tab:selected {{
        background-color: {tab_active_bg};
        color: {accent};
        font-weight: bold;
        border-bottom: 1px solid {tab_active_bg};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: #cdd4df;
    }}

    /* ---------- TableWidget ---------------------------------------- */
    QTableWidget {{
        background-color: {panel_bg};
        color: {text_fg};
        gridline-color: {panel_border};
        selection-background-color: {selection_bg};
        selection-color: {text_fg};
        border: 1px solid {panel_border};
        alternate-background-color: #f5f7fa;
    }}
    QHeaderView::section {{
        background-color: {header_bg};
        color: {header_fg};
        padding: 5px;
        border: 1px solid {panel_border};
        font-weight: 600;
    }}

    /* ---------- ComboBox ------------------------------------------- */
    QComboBox {{
        border: 1px solid {panel_border};
        border-radius: 4px;
        padding: 2px 20px 2px 6px;
        background-color: {panel_bg};
        color: {text_fg};
        min-height: 22px;
    }}
    QComboBox:focus {{
        border: 1.5px solid {accent};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid {panel_border};
    }}
    QComboBox QAbstractItemView {{
        background-color: {panel_bg};
        border: 1px solid {panel_border};
        selection-background-color: {selection_bg};
        selection-color: {text_fg};
    }}

    /* ---------- Menu ------------------------------------------------ */
    QMenu {{
        background-color: {panel_bg};
        border: 1px solid {panel_border};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 5px 10px;
        color: {text_fg};
    }}
    QMenu::item:selected {{
        background-color: {selection_bg};
        color: {text_fg};
    }}

    /* ---------- ToolTip -------------------------------------------- */
    QToolTip {{
        background-color: {tooltip_bg};
        color: {text_fg};
        border: 1px solid {accent};
        padding: 6px;
        border-radius: 4px;
        font-size: 9pt;
    }}
    """


def _dark_theme() -> Any:
    return qdarkstyle.load_stylesheet()


def make_help_icon(text: str) -> QWidget:
    icon = QLabel("?")
    icon.setStyleSheet(help_icon_style())
    icon.setToolTip(text)
    icon.setFixedWidth(12)
    icon.setAlignment(Qt.AlignCenter)

    wrapper = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addWidget(icon)
    wrapper.setLayout(layout)

    return wrapper


def help_icon_style(dark_mode: bool = False) -> str:
    if dark_mode:
        return """
        QLabel {
            color: #80cbc4;
            font-weight: bold;
            font-size: 10pt;
            border: 1px solid #80cbc4;
            border-radius: 8px;
            background-color: #263238;
            min-width: 16px;
            min-height: 16px;
            max-width: 16px;
            max-height: 16px;
            qproperty-alignment: AlignCenter;
        }
        QLabel:hover {
            background-color: #37474f;
            color: #a7ffeb;
            border-color: #a7ffeb;
        }
        """
    else:
        return """
        QLabel {
            color: teal;
            font-weight: bold;
            font-size: 10pt;
            border: 1px solid teal;
            border-radius: 8px;
            background-color: #f0fdfa;
            min-width: 16px;
            min-height: 16px;
            max-width: 16px;
            max-height: 16px;
            qproperty-alignment: AlignCenter;
        }
        QLabel:hover {
            background-color: #d0f0eb;
            color: darkcyan;
            border-color: darkcyan;
        }
        """
