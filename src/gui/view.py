# main_window.py
from __future__ import annotations
from src.core.file_manager import find_specific_files_in_folder, KS_REQUIRED, KS_LABEL_FILES
from src.gui.gui_themes import _toggle_theme, make_help_icon
from src.gui.file_chooser import file_chooser
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLineEdit, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QGridLayout, QHeaderView, QComboBox,
    QTabWidget, QSizePolicy,
)
from PySide6.QtGui import QAction, QFont, QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
import logging


class MainWindow(QMainWindow):
    """Neuropixel Data Exporter GUI (fixed log, no binding warning)."""
    MIN_W, MIN_H = 1080, 720

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Neuropixel Data Exporter")
        self.setMinimumSize(self.MIN_W, self.MIN_H)

        self.controller.set_view(self)
        self._build_ui()
        self.controller.load_temp_settings()

    def _build_ui(self) -> None:
        self._create_menu_bar()

        inputs_widget = QWidget()
        inputs_widget.setLayout(self._create_inputs_layout())

        log_group_box = self._create_log_group_box()

        root_layout = QVBoxLayout()
        root_layout.addWidget(inputs_widget)
        root_layout.addWidget(log_group_box)

        root_container = QWidget()
        root_container.setLayout(root_layout)
        self.setCentralWidget(root_container)

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

        toggle_theme_action = QAction("Toggle Light/Dark Mode", self)
        toggle_theme_action.triggered.connect(lambda: _toggle_theme(self))
        settings_menu.addAction(toggle_theme_action)

    def _create_inputs_layout(self) -> QVBoxLayout:
        inputs_layout = QVBoxLayout()
        inputs_layout.setContentsMargins(10, 10, 10, 10)
        inputs_layout.setSpacing(10)

        # ----- Data-folder selector ----------------------------------- #
        folder_group_box = QGroupBox("Data Folder")
        folder_layout = QHBoxLayout()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._select_folder)
        browse_button.setFixedWidth(80)
        self.folder_input = QLineEdit(placeholderText="Select data folder")
        folder_layout.addWidget(browse_button)
        folder_layout.addWidget(self.folder_input)
        folder_group_box.setLayout(folder_layout)
        folder_group_box.setMaximumHeight(60)
        inputs_layout.addWidget(folder_group_box)

        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(10)

        # Left column
        left_column_layout = QVBoxLayout()
        left_column_layout.addWidget(self._create_analysis_tabs())
        left_column_layout.addWidget(self._create_outputs_checkbox_group())
        columns_layout.addLayout(left_column_layout, stretch=2)

        # Right column
        right_column_layout = QVBoxLayout()
        right_column_layout.addWidget(self._create_drug_input_group())
        right_column_layout.addWidget(self._create_drug_table_group())
        columns_layout.addLayout(right_column_layout, stretch=6)

        inputs_layout.addLayout(columns_layout)

        return inputs_layout

    def _create_analysis_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # ------- Analysis tab ----------------------------------------- #
        self.cluster_input = QLineEdit(placeholderText="e.g. 1, good, 5")

        self.cluster_dropdown = QComboBox(placeholderText="Select Label...")
        
        self.cluster_dropdown.setCurrentIndex(0)
        self.cluster_dropdown.activated.connect(self._on_cluster_selected)
        self.cluster_dropdown.setFixedWidth(120)

        analysis_form = QFormLayout(labelAlignment=Qt.AlignmentFlag.AlignLeft)
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(self.cluster_input)
        cluster_row.addWidget(self.cluster_dropdown)
        analysis_form.addRow("Clusters / Labels:", cluster_row)


        self.start_time_input = QLineEdit(
            placeholderText="Default = 0 s")
        self.end_time_input = QLineEdit(
            placeholderText="Default = max")
        self.bin_size_input = QLineEdit(
            placeholderText="Default = 60 s")
        analysis_form.addRow("Start Time (s):", self.start_time_input)
        analysis_form.addRow("End Time (s):",   self.end_time_input)
        analysis_form.addRow("Bin Size (s):",   self.bin_size_input)

        analysis_box = QGroupBox()
        analysis_box.setLayout(analysis_form)
        tabs.addTab(analysis_box, "Analysis")

        additional_form = QFormLayout(
            labelAlignment=Qt.AlignmentFlag.AlignLeft)
        cck_label = QLabel("CCK (IV)")
        cck_label.setFont(QFont("", 12, QFont.Weight.Bold))
        additional_form.addRow(cck_label)

        self.cck_time_input = QLineEdit(
            placeholderText="Optional")
        additional_form.addRow("CCK Time (s):", self.cck_time_input)

        # Spacer (optional)
        additional_form.addRow(QLabel(""))

        # --- Baseline Section ---
        baseline_label = QLabel("Baseline Settings")
        baseline_label.setFont(QFont("", 12, QFont.Weight.Bold))
        additional_form.addRow(baseline_label)

        self.baseline_start_input = QLineEdit(
            placeholderText="Optional")
        self.baseline_end_input = QLineEdit(
            placeholderText="Optional")
        additional_form.addRow("Baseline Start (s):",
                               self.baseline_start_input)
        additional_form.addRow("Baseline End (s):", self.baseline_end_input)

        # Group box and tab
        additional_box = QGroupBox()
        additional_box.setLayout(additional_form)
        tabs.addTab(additional_box, "CCK and Baseline")

        return tabs

    def _create_outputs_checkbox_group(self) -> QGroupBox:
        self.use_baseline_checkbox = QCheckBox(
            "Delta Firing Rate",         checked=True)
        self.baseline_hazard_checkbox = QCheckBox(
            "Baseline ISI and Hazard",     checked=True)
        self.binned_hazard_checkbox = QCheckBox(
            "Binned ISI and Hazard",       checked=True)
        self.group_label_data_checkbox = QCheckBox(
            "Export Mean by Label",      checked=True)
        self.txt_export_checkbox = QCheckBox(
            "Export TXT for Clampfit",   checked=True)
        self.all_graphs_checkbox = QCheckBox(
            "Export All Graphs",         checked=True)

        analysis_box = QGroupBox("Analysis Metrics")
        analysis_col = QVBoxLayout()
        for cb in (self.use_baseline_checkbox,
                   self.baseline_hazard_checkbox,
                   self.binned_hazard_checkbox):
            analysis_col.addWidget(cb)
        analysis_box.setLayout(analysis_col)

        export_box = QGroupBox("Extra Export Options")
        export_col = QVBoxLayout()

        label_row = QHBoxLayout()
        label_row.addWidget(self.group_label_data_checkbox)
        label_row.addWidget(make_help_icon(
            "If multiple clusters share a label (e.g. 'good'),\n"
            "their mean ± SEM is exported in a sheet."
        ))
        label_row.addStretch()

        export_col.addLayout(label_row)
        export_col.addWidget(self.all_graphs_checkbox)
        export_col.addWidget(self.txt_export_checkbox)
        export_box.setLayout(export_col)

        wrapper_box = QGroupBox()
        wrapper_layout = QVBoxLayout()              # stack rows vertically

        top_row = QHBoxLayout()
        top_row.addWidget(analysis_box)
        top_row.addWidget(export_box)
        wrapper_layout.addLayout(top_row)

        self.run_button = QPushButton(
            "Run Analysis", clicked=self._run_analysis)
        self.run_button.setStyleSheet("padding:10px;font-weight:bold;")
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        wrapper_layout.addWidget(self.run_button)    # full width by default
        wrapper_box.setLayout(wrapper_layout)

        return wrapper_box

    def _create_drug_input_group(self) -> QGroupBox:
        self.drug_name_input = QLineEdit(
            placeholderText="e.g. Hypertonic Saline")
        self.peri_drug_input = QLineEdit(
            placeholderText="Optional – 60 or 30/90")
        self.drug_start_input = QLineEdit()
        self.drug_end_input = QLineEdit()
        add_button = QPushButton("Add", clicked=self._add_drug_event)

        # peri-drug field + help icon
        peri_layout = QHBoxLayout()
        peri_layout.setContentsMargins(0, 0, 0, 0)
        peri_layout.setSpacing(4)
        peri_layout.addWidget(self.peri_drug_input)
        peri_layout.addWidget(make_help_icon(
            "Set pre/post window in seconds:\n"
            "  60    → 60 s both sides\n"
            " 30/90 → 30 s pre, 90 s post"))
        peri_widget = QWidget()
        peri_widget.setLayout(peri_layout)

        grid = QGridLayout()
        grid.addWidget(QLabel("Drug Name:"),    0, 0)
        grid.addWidget(self.drug_name_input,  0, 1)
        grid.addWidget(QLabel("Peri-drug (s):"), 0, 2)
        grid.addWidget(peri_widget,           0, 3)
        grid.addWidget(QLabel("Start Time (s):"), 1, 0)
        grid.addWidget(self.drug_start_input, 1, 1)
        grid.addWidget(QLabel("End Time (s):"), 1, 2)
        grid.addWidget(self.drug_end_input,   1, 3)

        hint_label = QLabel("Right-click a table row to delete.")
        hint_label.setStyleSheet("font-size:9pt;color:gray;")

        group_layout = QVBoxLayout()
        group_layout.addLayout(grid)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(hint_label)
        bottom_row.addStretch()
        bottom_row.addWidget(add_button)
        group_layout.addLayout(bottom_row)

        drug_group_box = QGroupBox("Add Drug Event")
        drug_group_box.setLayout(group_layout)
        return drug_group_box

    def _create_drug_table_group(self) -> QGroupBox:
        self.drug_table = QTableWidget(0, 5)
        self.drug_table.setHorizontalHeaderLabels(
            ["Name", "Start", "End", "Start Offset", "End Offset"])
        self.drug_table.horizontalHeader().setFixedHeight(24)
        self.drug_table.verticalHeader().setVisible(False)
        self.drug_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.drug_table.customContextMenuRequested.connect(
            self._remove_selected_drug)
        self.drug_table.setMinimumHeight(200)

        self.drug_table.setColumnWidth(1, 60)
        self.drug_table.setColumnWidth(2, 60)
        self.drug_table.setColumnWidth(3, 80)
        self.drug_table.setColumnWidth(4, 80)
        self.drug_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        for col in [1, 2, 3, 4]:
            self.drug_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Fixed)

        table_group_box = QGroupBox("Drug Events")
        table_layout = QVBoxLayout()
        table_layout.addWidget(self.drug_table)
        table_group_box.setLayout(table_layout)
        return table_group_box

    def _create_log_group_box(self) -> QGroupBox:
        self.log_output = QTextEdit(readOnly=True)
        self.log_output.setMinimumHeight(150)
        log_group_box = QGroupBox("Output Log")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_output)
        log_group_box.setLayout(log_layout)
        return log_group_box

    def _select_folder(self) -> None:
        path = file_chooser(self)
        self.folder_input.setText(str(path) if path else "")
        if not path:
            return

        found_files = find_specific_files_in_folder(path, KS_REQUIRED, KS_LABEL_FILES)
        if found_files:
            self.log_output.append(
                f"Found {len(found_files)} required file(s):")
            for name in found_files:
                self.log_output.append(f"  • {name}")
        else:
            self.log_output.append("No required files found.")
            return

        self.controller.try_populate_label_dropdown(found_files,
                                                    self.cluster_dropdown,
                                                    self.log_output)

    def _add_drug_event(self) -> None:
        try:
            parsed = self.controller.add_drug_event(
                name=self.drug_name_input.text().strip(),
                peri_drug=self.peri_drug_input.text().strip(),
                start_text=self.drug_start_input.text().strip(),
                end_text=self.drug_end_input.text().strip())
        except ValueError as err:
            self.log_output.append(str(err))
            return

        row = self.drug_table.rowCount()
        self.drug_table.insertRow(row)

        # Column 0: Name (left-aligned)
        self.drug_table.setItem(row, 0, QTableWidgetItem(parsed["name"]))

        # Column 1: Start
        item = QTableWidgetItem(str(parsed["start"]))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drug_table.setItem(row, 1, item)

        # Column 2: End
        end_val = "-" if parsed.get("end") is None else str(parsed["end"])
        item = QTableWidgetItem(end_val)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drug_table.setItem(row, 2, item)

        # Column 3: Start-Offset
        pre_val = str(parsed.get("start-offset")
                    ) if parsed.get("start-offset") is not None else "-"
        item = QTableWidgetItem(pre_val)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drug_table.setItem(row, 3, item)

        # Column 4: End-Offset
        post_val = str(parsed.get("end-offset")
                       ) if parsed.get("end-offset") is not None else "-"
        item = QTableWidgetItem(post_val)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drug_table.setItem(row, 4, item)

    
        # clear input fields
        for widget in (self.drug_name_input, self.peri_drug_input,
                       self.drug_start_input, self.drug_end_input):
            widget.clear()

    def _remove_selected_drug(self) -> None:
        selected_row = self.drug_table.currentRow()
        if selected_row != -1:
            self.drug_table.removeRow(selected_row)

    def _on_cluster_selected(self, index: int) -> None:
        if index == 0:
            return  # ignore the “Select label…” placeholder
        
        label = self.cluster_dropdown.currentText().strip()
        current = [s.strip()
                   for s in self.cluster_input.text().split(",") if s.strip()]
        
        if label not in current:
            current.append(label)
            self.cluster_input.setText(", ".join(current))
        
        self.cluster_dropdown.setCurrentIndex(0)

    def _run_analysis(self) -> None:
        drug_events = []
        for row in range(self.drug_table.rowCount()):
            name = self.drug_table.item(row, 0).text()
            start = float(self.drug_table.item(row, 1).text())
            end_text = self.drug_table.item(row, 2).text()
            end = None if end_text == "" else float(end_text)
            pre = self.drug_table.item(row, 3).text()
            post = self.drug_table.item(row, 4).text()
            drug_events.append(dict(name=name, start=start, end=end,
                                    pre_time=pre, post_time=post))

        cck_time = float(self.cck_time_input.text()
                         ) if self.cck_time_input.text().strip() else None

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
            baseline_hazard=self.baseline_hazard_checkbox.isChecked(),
            mean_label_data=self.group_label_data_checkbox.isChecked(),
            export_all_graphs=self.all_graphs_checkbox.isChecked(),
            export_txt=self.txt_export_checkbox.isChecked(),
            cck_time=cck_time,
            drug_events=drug_events
        )
        self.controller.save_temp_settings()
