# gui_themes.py
from PySide6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import qdarkstyle


def _toggle_theme(main_window: QMainWindow):
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
    """Material-inspired light palette that mirrors the dark‐theme spacing."""
    base_bg = "#fafafa"   # window background
    panel_bg = "#ffffff"   # group-box / text-field background
    panel_border = "#d7d7d7"
    text_fg = "#212121"
    hint_fg = "#555555"
    accent = "#00acc1"   # teal 600
    accent_hover = "#008e9b"
    header_bg = "#f0f0f0"
    header_fg = "#333333"
    selection_bg = "#c0f1ff"
    tooltip_bg = "#e0f7f5"

    return f"""
    QWidget {{
        font-size: 8pt;
        background-color: {base_bg};
        color: {text_fg};
    }}

    /* ---------- GroupBox ------------------------------------------- */
    QGroupBox {{
        font-weight: bold;
        border: 1px solid {panel_border};
        border-radius: 8px;
        margin-top: 10px;
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
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {accent_hover};
    }}
    QPushButton:disabled {{
        background-color: #bdbdbd;
        color: #f5f5f5;
    }}

    /* ---------- LineEdit / TextEdit -------------------------------- */
    QLineEdit, QTextEdit {{
        background-color: {panel_bg};
        border: 1px solid {panel_border};
        border-radius: 6px;
        padding: 4px 6px;
        color: {text_fg};
    }}
    QTextEdit {{
        background-color: #fcfcfc;
    }}

    /* ---------- TableWidget ---------------------------------------- */
    QTableWidget {{
        background-color: {panel_bg};
        color: {text_fg};
        gridline-color: {panel_border};
        selection-background-color: {selection_bg};
        selection-color: {text_fg};
        border: 1px solid {panel_border};
        alternate-background-color: #f9f9f9;
    }}
    QHeaderView::section {{
        background-color: {header_bg};
        color: {header_fg};
        padding: 6px;
        border: 1px solid {panel_border};
        font-weight: 500;
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
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid {panel_border};
    }}
    QComboBox::down-arrow {{
        image: url(:/qt-project.org/styles/commonstyle/images/arrowdown-16.png);
        width: 12px;
        height: 12px;
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
        color: {accent};
        border: 1px solid {accent};
        padding: 6px;
        border-radius: 4px;
        font-size: 9pt;
    }}
    """


def _dark_theme():
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


def help_icon_style(dark_mode=False):
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
