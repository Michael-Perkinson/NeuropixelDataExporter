import json
import sys
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import QTextEdit, QFileDialog, QMessageBox

from src.gui.view import MainWindow
from src.gui.gui_themes import _dark_theme, _light_theme

from src.core.file_manager import validate_ks_folder, create_label_lookup, KS_REQUIRED, KS_LABEL_FILES
from src.core.spike_filter import prepare_filtered_data
from src.core.firing_rate import process_cluster_data
from src.core.isi_hazard import calculate_isi_histogram, calculate_hazard_function
from src.core.results_writer import export_data, export_hazard_excel
from src.core.interactive_plot import export_firing_rate_html
from src.core.input_parser import parse_channels_or_labels, validate_and_parse_drug_event

# Determine runtime path (support both script and PyInstaller .exe)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

TEMP_SETTINGS_PATH = BASE_DIR / ".neuropixel_gui_last_session.json"


class GUIController:
    def __init__(self):
        pass

    def set_view(self, main_window):
        self.view = main_window

    def export_user_settings(self, parent):
        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Export Settings", "", "JSON Files (*.json)")
        if not file_path:
            return

        settings = {
            "cck_time": self.view.cck_time_input.text(),
            "optional_outputs": {
                "export_txt": self.view.txt_export_checkbox.isChecked(),
                "export_all_graphs": self.view.all_graphs_checkbox.isChecked(),
                "baseline_hazard": self.view.baseline_hazard_checkbox.isChecked(),
                "binned_hazard": self.view.binned_hazard_checkbox.isChecked()
            },
            "theme": "dark" if self.view.dark_mode else "light"
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            QMessageBox.information(
                parent, "Export Successful", "Settings saved.")
        except Exception as e:
            QMessageBox.critical(parent, "Export Failed", str(e))

    def import_user_settings(self, parent):
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Import Settings", "", "JSON Files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
        except Exception as e:
            QMessageBox.critical(parent, "Import Failed",
                                 f"Error reading settings file:\n{str(e)}")
            return

        try:
            self.view.cck_time_input.setText(settings.get("cck_time", ""))
            opts = settings.get("optional_outputs", {})
            self.view.txt_export_checkbox.setChecked(
                opts.get("export_txt", True))
            self.view.all_graphs_checkbox.setChecked(
                opts.get("export_all_graphs", True))
            self.view.binned_hazard_checkbox.setChecked(
                opts.get("baseline_hazard", True))
            self.view.binned_hazard_checkbox.setChecked(
                opts.get("binned_hazard", True))

            theme = settings.get("theme", "light")
            if theme == "dark":
                self.view.setStyleSheet(_dark_theme())
                self.view.dark_mode = True
            else:
                self.view.setStyleSheet(_light_theme())
                self.view.dark_mode = False

            QMessageBox.information(
                parent, "Import Successful", "Settings loaded.")
        except Exception as e:
            QMessageBox.critical(parent, "Import Failed",
                                 f"Error applying settings:\n{str(e)}")

    def load_temp_settings(self):
        try:
            if not TEMP_SETTINGS_PATH.exists():
                with open(TEMP_SETTINGS_PATH, 'w') as f:
                    json.dump({}, f)

                # First-time launch → apply default light theme
                self.view.setStyleSheet(_light_theme())
                self.view.dark_mode = False
                return

            with open(TEMP_SETTINGS_PATH, 'r') as f:
                settings = json.load(f)

            self.view.cck_time_input.setText(settings.get("cck_time", ""))
            opts = settings.get("optional_outputs", {})
            self.view.txt_export_checkbox.setChecked(
                opts.get("export_txt", True))
            self.view.all_graphs_checkbox.setChecked(
                opts.get("export_all_graphs", True))
            self.view.binned_hazard_checkbox.setChecked(
                opts.get("baseline_hazard", True))
            self.view.binned_hazard_checkbox.setChecked(
                opts.get("binned_hazard", True))

            # Apply theme if saved
            theme = settings.get("theme", "light")
            if theme == "dark":
                self.view.setStyleSheet(_dark_theme())
                self.view.dark_mode = True
            else:
                self.view.setStyleSheet(_light_theme())
                self.view.dark_mode = False

        except Exception as e:
            print(f"[Warning] Could not load temp settings: {e}")

    def save_temp_settings(self):
        settings = {
            "cck_time": self.view.cck_time_input.text(),
            "optional_outputs": {
                "export_txt": self.view.txt_export_checkbox.isChecked(),
                "export_all_graphs": self.view.all_graphs_checkbox.isChecked(),
                "baseline_hazard": self.view.baseline_hazard_checkbox.isChecked(),
                "binned_hazard": self.view.binned_hazard_checkbox.isChecked()
            },
            "theme": "dark" if self.view.dark_mode else "light"
        }
        try:
            with open(TEMP_SETTINGS_PATH, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"[Warning] Could not save temp settings: {e}")

    def add_drug_event(self, name: str, peri_drug: str, start_text: str, end_text: str) -> dict:
        print('est')
        return validate_and_parse_drug_event(name, peri_drug, start_text, end_text)

    def try_populate_label_dropdown(self, found_files, dropdown, log_widget):
        for fpath in found_files.values():
            print(fpath)
            if Path(fpath).name.lower() in {"cluster_group.tsv"}:
                try:
                    labels_array = create_label_lookup(str(fpath))
                    unique_labels = sorted(set(labels_array), key=str.lower)
                    dropdown.clear()
                    dropdown.addItem("Select label...")
                    dropdown.addItems([
                        label for label in unique_labels if label != "unknown"
                    ])

                    log_widget.append("Loaded labels into dropdown.")
                except Exception as e:
                    log_widget.append(f"Error loading cluster labels: {e}")
                break  # Stop after the first valid file

    def run_analysis(
        self,
        folder: str,
        clusters: str,
        drug_time: str,
        start: str,
        end: str,
        bin_size: str,
        baseline_start: str,
        baseline_end: str,
        log: QTextEdit
    ):
        log.append("Starting analysis...")

        folder_path = Path(folder)
        if not folder_path.exists():
            log.append("Invalid folder path.")
            return

        # Validate required files
        try:
            file_paths = validate_ks_folder(
                folder_path, KS_REQUIRED, KS_LABEL_FILES)
        except FileNotFoundError as e:
            log.append(f"Missing required file: {e}")
            return

        data_folder_path = Path(file_paths["spike_times.npy"]).parent
        log.append("Loading spike data...")

        # Prepare filtered dataframe
        recording_dataframe, max_time = prepare_filtered_data(file_paths)

        try:
            drug_time_val = float(drug_time) if drug_time else None
            start_time = float(start) if start else 0
            end_time = float(end) if end else max_time
            bin_size_val = float(bin_size) if bin_size else 1.0
            baseline_start_val = float(
                baseline_start) if baseline_start else None
            baseline_end_val = float(baseline_end) if baseline_end else None
        except ValueError as e:
            log.append(f"Invalid input: {e}")
            return

        log.append("Parsing cluster/channel inputs...")
        parsed_input = parse_channels_or_labels(clusters)

        if "error" in parsed_input:
            log.append(f"Input error: {parsed_input['error']}")
            return

        cluster_ids = parsed_input["channels"]
        label_names = parsed_input["labels"]

        if not cluster_ids and not label_names:
            log.append("No valid cluster IDs or labels provided.")
            return

        log.append("Computing firing rates...")
        raw_fr_dict, baseline_fr_dict = process_cluster_data(
            recording_dataframe,
            cluster_ids,
            start_time,
            end_time,
            drug_time_val,
            baseline_start_val,
            baseline_end_val
        )

        log.append("Exporting spike times and firing rate CSVs...")
        export_dir, images_dir, firing_rate_df = export_data(
            raw_fr_dict,
            baseline_fr_dict,
            data_folder_path,
            bin_size_val,
            start_time,
            end_time,
            baseline_start_val,
            baseline_end_val,
        )

        log.append("Calculating ISI histograms and hazard functions...")
        isi_df, baseline_isi_df = calculate_isi_histogram(
            raw_fr_dict,
            baseline_start_val,
            baseline_end_val
        )

        hazard_df, hazard_summary_df, baseline_hazard_df, baseline_hazard_summary_df = (
            calculate_hazard_function(isi_df, baseline_isi_df)
        )

        log.append("Exporting hazard summary Excel files...")
        export_hazard_excel(
            export_dir,
            hazard_df,
            hazard_summary_df,
            isi_df,
            baseline_isi_df,
            baseline_hazard_df,
            baseline_hazard_summary_df,
        )

        log.append("Exporting interactive firing rate plots...")
        export_firing_rate_html(
            firing_rate_df,
            images_dir,
            bin_size_val,
            drug_time_val,
        )

        log.append("Analysis complete. All files have been exported.")

        # Save session settings silently
        self.save_temp_settings()
