from __future__ import annotations

import math
from typing import TypedDict, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,

    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.file_manager import KS_LABEL_FILES, KS_REQUIRED, find_specific_files_in_folder
from src.gui.file_chooser import file_chooser
from src.gui.gui_themes import _toggle_theme, make_help_icon


class DrugEvent(TypedDict):
    name: str
    start: float
    end: float | None
    pre_time: float | None
    post_time: float | None


class MainWindow(QMainWindow):  # type: ignore[misc]
    MIN_W, MIN_H = 1080, 720

    def __init__(self, controller: Any) -> None:
        super().__init__()
        self.dark_mode: bool = False
        self.controller = controller
        self.setWindowTitle("Neuropixel Data Exporter")
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.setMaximumSize(self.MIN_W, self.MIN_H)

        self.controller.set_view(self)
        self._build_ui()
        self.controller.load_temp_settings()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._create_menu_bar()

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._create_folder_group())
        root_layout.addLayout(self._create_main_columns(), stretch=1)
        root_layout.addWidget(self._create_log_group())

        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

    def _create_menu_bar(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")

        import_action = QAction("Import Settings", self)
        import_action.triggered.connect(
            lambda: self.controller.import_user_settings(self))
        settings_menu.addAction(import_action)

        export_action = QAction("Export Settings", self)
        export_action.triggered.connect(
            lambda: self.controller.export_user_settings(self))
        settings_menu.addAction(export_action)

        toggle_action = QAction("Toggle Light/Dark Mode", self)
        toggle_action.triggered.connect(lambda: _toggle_theme(self))
        settings_menu.addAction(toggle_action)

    def _create_folder_group(self) -> QGroupBox:
        self.folder_input = QLineEdit(placeholderText="Select data folder…")
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._select_folder)

        row = QHBoxLayout()
        row.addWidget(browse_btn)
        row.addWidget(self.folder_input)

        box = QGroupBox("Data Folder")
        box.setMaximumHeight(60)
        box.setLayout(row)
        return box

    def _create_main_columns(self) -> QHBoxLayout:
        cols = QHBoxLayout()
        cols.setSpacing(10)
        cols.addLayout(self._create_left_column(), stretch=4)
        cols.addLayout(self._create_right_column(), stretch=6)
        return cols

    # ── Left column ──────────────────────────────────────────────────────────

    def _create_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)
        col.addWidget(self._create_analysis_group())
        col.addWidget(self._create_cell_typing_group())
        col.addWidget(self._create_options_group())
        col.addStretch()
        return col

    def _create_analysis_group(self) -> QGroupBox:
        self.cluster_input = QLineEdit(placeholderText="e.g. 1, good, 5")
        self.cluster_dropdown = QComboBox()
        self.cluster_dropdown.addItem("Select label…")
        self.cluster_dropdown.setFixedWidth(110)
        self.cluster_dropdown.activated.connect(self._on_label_selected)

        cluster_row = QHBoxLayout()
        cluster_row.setSpacing(4)
        cluster_row.addWidget(self.cluster_input)
        cluster_row.addWidget(self.cluster_dropdown)

        self.start_time_input = QLineEdit(placeholderText="0 s")
        self.end_time_input = QLineEdit(placeholderText="max")
        self.bin_size_input = QLineEdit(placeholderText="600 s")
        self.start_time_input.setFixedWidth(70)
        self.end_time_input.setFixedWidth(70)
        self.bin_size_input.setFixedWidth(70)

        time_row = QHBoxLayout()
        time_row.setSpacing(4)
        time_row.addWidget(QLabel("Start:"))
        time_row.addWidget(self.start_time_input)
        time_row.addWidget(QLabel("End:"))
        time_row.addWidget(self.end_time_input)
        time_row.addWidget(QLabel("Bin:"))
        time_row.addWidget(self.bin_size_input)
        time_row.addStretch()

        # ── Firing rate Baseline (inline) ────────────────────────────────────
        self.use_baseline_checkbox = QCheckBox("FR Baseline")
        self.use_baseline_checkbox.setToolTip(
            "Compute a mean firing rate over this window and use it\n"
            "as a baseline for delta-from-baseline output."
        )
        self.baseline_start_input = QLineEdit(placeholderText="Start s")
        self.baseline_end_input = QLineEdit(placeholderText="End s")
        self.baseline_start_input.setFixedWidth(70)
        self.baseline_end_input.setFixedWidth(70)

        baseline_row = QHBoxLayout()
        baseline_row.setSpacing(4)
        baseline_row.addWidget(self.use_baseline_checkbox)
        baseline_row.addWidget(QLabel("From:"))
        baseline_row.addWidget(self.baseline_start_input)
        baseline_row.addWidget(QLabel("To:"))
        baseline_row.addWidget(self.baseline_end_input)
        baseline_row.addStretch()

        # ── ISI / Hazard baseline window ──────────────────────────────────────
        isi_label = QLabel("ISI Hazard Window:")
        isi_label.setToolTip(
            "Time window used for the 'early recording' ISI histogram\n"
            "and hazard function — separate from the firing rate baseline.\n\n"
            "Default is the first 10 minutes (0–600 s), which is typically\n"
            "pre-drug and gives a stable reference hazard shape.\n\n"
            "Only used when 'Binned ISI & Hazard' is checked."
        )
        self.isi_hazard_start_input = QLineEdit(placeholderText="0 s")
        self.isi_hazard_end_input = QLineEdit(placeholderText="600 s")
        self.isi_hazard_start_input.setFixedWidth(70)
        self.isi_hazard_end_input.setFixedWidth(70)
        self.isi_hazard_start_input.setToolTip(
            "Start of the ISI/hazard reference window (seconds).\nDefault: 0"
        )
        self.isi_hazard_end_input.setToolTip(
            "End of the ISI/hazard reference window (seconds).\nDefault: 600 (10 min)"
        )

        isi_row = QHBoxLayout()
        isi_row.setSpacing(4)
        isi_row.addWidget(isi_label)
        isi_row.addWidget(QLabel("From:"))
        isi_row.addWidget(self.isi_hazard_start_input)
        isi_row.addWidget(QLabel("To:"))
        isi_row.addWidget(self.isi_hazard_end_input)
        isi_row.addStretch()

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #d0d0d0; margin: 2px 0;")

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.addWidget(QLabel("Clusters / Labels:"))
        layout.addLayout(cluster_row)
        layout.addSpacing(2)
        layout.addLayout(time_row)
        layout.addWidget(sep)
        layout.addLayout(baseline_row)
        layout.addLayout(isi_row)

        box = QGroupBox("Analysis")
        box.setLayout(layout)
        return box

    def _create_cell_typing_group(self) -> QGroupBox:
        tabs = QTabWidget()
        tabs.addTab(self._create_cck_tab(), "CCK (IV)")
        tabs.addTab(self._create_pe_tab(), "PE (IV)")

        box = QGroupBox("Cell Typing")
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(tabs)
        box.setLayout(layout)
        return box

    def _create_cck_tab(self) -> QWidget:
        self.cck_time_input = QLineEdit(placeholderText="Optional")
        self.cck_time_input.setFixedWidth(100)

        row = QHBoxLayout()
        row.addWidget(QLabel("CCK Time (s):"))
        row.addWidget(self.cck_time_input)
        row.addWidget(make_help_icon(
            "IV CCK injection time (seconds from recording start).\n\n"
            "Classifies cells using firing rate 5 min before\n"
            "vs. 5 min after injection (1 min bins):\n\n"
            "  delta ≥ +0.5 Hz → Putative Oxytocin\n"
            "  delta  < +0.5 Hz → Putative Vasopressin\n\n"
            "Pre-window slope is checked for baseline stability.\n"
            "Results exported to CCK_Cell_Typing sheet."
        ))
        row.addStretch()

        w = QWidget()
        w.setLayout(row)
        return w

    def _create_pe_tab(self) -> QWidget:
        self.pe_time_input = QLineEdit(placeholderText="Optional")
        self.pe_time_input.setFixedWidth(100)

        row = QHBoxLayout()
        row.addWidget(QLabel("PE Time (s):"))
        row.addWidget(self.pe_time_input)
        row.addWidget(make_help_icon(
            "IV Phenylephrine (PE) injection time (seconds from recording start).\n\n"
            "Classifies cells using firing rate 1 min before\n"
            "vs. 1 min after injection (10 s bins):\n\n"
            "  delta ≤ -0.5 Hz → Putative Vasopressin\n"
            "  delta  > -0.5 Hz → Putative Oxytocin\n\n"
            "Pre-window slope is checked for baseline stability.\n"
            "Results exported to PE_Cell_Typing sheet."
        ))
        row.addStretch()

        w = QWidget()
        w.setLayout(row)
        return w

    def _create_options_group(self) -> QGroupBox:
        self.binned_hazard_checkbox = QCheckBox("Binned ISI && Hazard")
        self.binned_hazard_checkbox.setChecked(True)
        self.binned_hazard_checkbox.setToolTip(
            "Compute ISI histograms and hazard functions for:\n"
            "  • Full recording\n"
            "  • Early window (set by ISI Hazard Window below)\n\n"
            "Output: isi_and_hazard_analysis.xlsx"
        )
        self.peri_hazard_checkbox = QCheckBox("Peri-drug Hazard")
        self.peri_hazard_checkbox.setChecked(True)
        self.peri_hazard_checkbox.setToolTip(
            "For each drug event with a peri window, compute ISI\n"
            "and hazard separately for the pre and post epochs.\n\n"
            "Useful for detecting ADP/AHP changes after drug.\n"
            "Adds Peri_<Drug>_ISI and Peri_<Drug>_Hazard sheets\n"
            "to isi_and_hazard_analysis.xlsx."
        )
        self.txt_export_checkbox = QCheckBox("Export TXT (Clampfit)")
        self.txt_export_checkbox.setChecked(True)
        self.all_graphs_checkbox = QCheckBox("Export All Graphs")
        self.all_graphs_checkbox.setChecked(True)
        self.group_label_data_checkbox = QCheckBox("Mean by Label")
        self.group_label_data_checkbox.setChecked(True)
        self.peri_drug_checkbox = QCheckBox("Peri-drug Sheets")
        self.peri_drug_checkbox.setChecked(True)
        self.peri_drug_checkbox.setToolTip(
            "For each drug event that has a peri-drug window set,\n"
            "export a firing rate sheet covering that window\n"
            "(e.g. 30 s pre → 90 s post) using the current bin size.\n\n"
            "Set the peri window in the 'Add Drug Event' panel (Peri column)."
        )

        # 2 columns × 3 rows grid
        grid_widget = QWidget()
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        checkboxes = [
            self.binned_hazard_checkbox,
            self.peri_hazard_checkbox,
            self.txt_export_checkbox,
            self.all_graphs_checkbox,
            self.group_label_data_checkbox,
            self.peri_drug_checkbox,
        ]
        for i, cb in enumerate(checkboxes):
            grid.addWidget(cb, i // 2, i % 2)
        grid_widget.setLayout(grid)

        layout = QVBoxLayout()
        layout.addWidget(grid_widget)

        box = QGroupBox("Output Options")
        box.setLayout(layout)
        return box

    # ── Right column ─────────────────────────────────────────────────────────

    def _create_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)
        col.addWidget(self._create_drug_input_group())
        col.addWidget(self._create_drug_table_group(), stretch=1)
        col.addLayout(self._create_run_row())
        return col

    def _create_drug_input_group(self) -> QGroupBox:
        self.drug_name_input = QLineEdit(placeholderText="e.g. Alpha-MSH")
        self.drug_route_combo = QComboBox()
        self.drug_route_combo.addItems(["Microdialysis", "IV"])
        self.drug_route_combo.setFixedWidth(120)
        self.drug_start_input = QLineEdit(placeholderText="s")
        self.drug_end_input = QLineEdit(placeholderText="s  (optional)")
        self.drug_end_input.setToolTip(
            "Leave blank for an acute injection (single time point).\n"
            "Enter a time in seconds for a continuous event, or 'max' to use the recording end."
        )
        self.peri_drug_input = QLineEdit(placeholderText="e.g. 600 or 300/900")

        self.drug_start_input.setFixedWidth(70)
        self.drug_end_input.setFixedWidth(110)

        peri_row = QHBoxLayout()
        peri_row.setContentsMargins(0, 0, 0, 0)
        peri_row.setSpacing(4)
        peri_row.addWidget(self.peri_drug_input)
        peri_row.addWidget(make_help_icon(
            "Pre/post window in seconds around drug onset.\n"
            "Used for plot shading AND for exporting a\n"
            "dedicated peri-drug firing rate sheet\n"
            "(if 'Peri-drug Sheets' is checked in Output Options).\n\n"
            "  600    → 600 s before and after\n"
            "  300/900 → 300 s pre, 900 s post"
        ))
        peri_widget = QWidget()
        peri_widget.setLayout(peri_row)

        add_btn = QPushButton("Add Drug Event")
        add_btn.clicked.connect(self._add_drug_event)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Name:"))
        row1.addWidget(self.drug_name_input)
        row1.addWidget(QLabel("Route:"))
        row1.addWidget(self.drug_route_combo)
        row1.addStretch()

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Start (s):"))
        row2.addWidget(self.drug_start_input)
        row2.addWidget(QLabel("End (s):"))
        row2.addWidget(self.drug_end_input)
        row2.addWidget(QLabel("Peri (s):"))
        row2.addWidget(peri_widget)
        row2.addStretch()

        hint = QLabel("Right-click a row to remove it.")
        hint.setStyleSheet("color: gray; font-size: 8pt;")

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(hint)
        bottom_row.addStretch()
        bottom_row.addWidget(add_btn)

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(bottom_row)

        box = QGroupBox("Add Drug Event")
        box.setLayout(layout)
        return box

    def _create_drug_table_group(self) -> QGroupBox:
        self.drug_table = QTableWidget(0, 6)
        self.drug_table.setHorizontalHeaderLabels(
            ["Name", "Route", "Start (s)", "End (s)",
             "Pre Offset", "Post Offset"]
        )
        self.drug_table.verticalHeader().setVisible(False)
        self.drug_table.horizontalHeader().setFixedHeight(24)
        self.drug_table.setAlternatingRowColors(True)
        self.drug_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.drug_table.customContextMenuRequested.connect(
            self._remove_selected_drug)

        self.drug_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5):
            self.drug_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)

        box = QGroupBox("Drug Events")
        layout = QVBoxLayout()
        layout.addWidget(self.drug_table)
        box.setLayout(layout)
        return box

    def _create_run_row(self) -> QHBoxLayout:
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setFixedHeight(40)
        self.run_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.run_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.run_button.clicked.connect(self._run_analysis)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.run_button, stretch=1)
        return row

    def _create_log_group(self) -> QGroupBox:
        self.log_output = QTextEdit(readOnly=True)
        self.log_output.setMinimumHeight(120)
        self.log_output.setMaximumHeight(180)

        layout = QVBoxLayout()
        layout.addWidget(self.log_output)

        box = QGroupBox("Output Log")
        box.setLayout(layout)
        return box

    # ── Event handlers ───────────────────────────────────────────────────────

    def _select_folder(self) -> None:
        path = file_chooser(self, start_dir=self.controller.last_browse_dir)
        self.folder_input.setText(str(path) if path else "")
        if not path:
            return
        self.controller.last_browse_dir = path

        found_files = find_specific_files_in_folder(
            path, KS_REQUIRED, KS_LABEL_FILES)
        if found_files:
            self.log_output.append(
                f"Found {len(found_files)} required file(s):")
            for name in found_files:
                self.log_output.append(f"  • {name}")
        else:
            self.log_output.append(
                "No required files found in selected folder.")
            return

        self.controller.try_populate_label_dropdown(
            found_files, self.cluster_dropdown, self.log_output)

    def _on_label_selected(self, index: int) -> None:
        if index == 0:
            return
        label = self.cluster_dropdown.currentText().strip()
        current = [s.strip()
                   for s in self.cluster_input.text().split(",") if s.strip()]
        if label and label not in current:
            current.append(label)
            self.cluster_input.setText(", ".join(current))
        self.cluster_dropdown.setCurrentIndex(0)

    def _add_drug_event(self) -> None:
        try:
            parsed = self.controller.add_drug_event(
                name=self.drug_name_input.text().strip(),
                peri_drug=self.peri_drug_input.text().strip(),
                start_text=self.drug_start_input.text().strip(),
                end_text=self.drug_end_input.text().strip(),
            )
        except ValueError as err:
            self.log_output.append(f"Drug event error: {err}")
            return

        row = self.drug_table.rowCount()
        self.drug_table.insertRow(row)

        def _cell(text: str, centre: bool = True) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            if centre:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return item

        def _fmt(v: float | None) -> str:
            if v is None:
                return "-"
            return "max" if math.isinf(v) else str(v)

        self.drug_table.setItem(row, 0, _cell(parsed["name"], centre=False))
        self.drug_table.setItem(row, 1, _cell(
            self.drug_route_combo.currentText()))
        self.drug_table.setItem(row, 2, _cell(str(parsed["start"])))
        self.drug_table.setItem(row, 3, _cell(_fmt(parsed.get("end"))))
        self.drug_table.setItem(row, 4, _cell(
            _fmt(parsed.get("start_offset"))))
        self.drug_table.setItem(row, 5, _cell(_fmt(parsed.get("end_offset"))))

        for w in (self.drug_name_input, self.peri_drug_input,
                  self.drug_start_input, self.drug_end_input):
            w.clear()
        self.drug_route_combo.setCurrentIndex(0)

    def _remove_selected_drug(self) -> None:
        row = self.drug_table.currentRow()
        if row != -1:
            self.drug_table.removeRow(row)

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        s = text.strip()
        if s in ("", "-"):
            return None
        if s.lower() in ("max", "inf"):
            return float("inf")
        return float(s)

    def _collect_drug_events(self) -> list[DrugEvent]:
        events: list[DrugEvent] = []
        for row in range(self.drug_table.rowCount()):
            def _text(col: int) -> str:
                item = self.drug_table.item(row, col)
                return item.text() if item else ""

            name = _text(0)
            if not name:
                continue
            events.append(DrugEvent(
                name=name,
                start=float(_text(2)),
                end=self._parse_optional_float(_text(3)),
                pre_time=self._parse_optional_float(_text(4)),
                post_time=self._parse_optional_float(_text(5)),
            ))
        return events

    def _run_analysis(self) -> None:
        try:
            drug_events = self._collect_drug_events()
        except ValueError as e:
            self.log_output.append(f"Invalid drug table value: {e}")
            return

        self.controller.run_analysis(
            folder=self.folder_input.text(),
            clusters=self.cluster_input.text(),
            start=self.start_time_input.text(),
            end=self.end_time_input.text(),
            bin_size=self.bin_size_input.text(),
            baseline_start=self.baseline_start_input.text(),
            baseline_end=self.baseline_end_input.text(),
            log=self.log_output,
            use_baseline=self.use_baseline_checkbox.isChecked(),
            run_hazard=self.binned_hazard_checkbox.isChecked(),
            peri_hazard=self.peri_hazard_checkbox.isChecked(),
            early_hazard_start=self._parse_optional_float(
                self.isi_hazard_start_input.text()) or 0.0,
            early_hazard_end=self._parse_optional_float(
                self.isi_hazard_end_input.text()) or 600.0,
            mean_label_data=self.group_label_data_checkbox.isChecked(),
            export_all_graphs=self.all_graphs_checkbox.isChecked(),
            export_txt=self.txt_export_checkbox.isChecked(),
            export_peri_drug=self.peri_drug_checkbox.isChecked(),
            cck_time=self._parse_optional_float(self.cck_time_input.text()),
            pe_time=self._parse_optional_float(self.pe_time_input.text()),
            drug_events=drug_events,
        )
        self.controller.save_temp_settings()
