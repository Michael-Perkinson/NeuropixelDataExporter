import json
import math
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTextEdit, QWidget

from src.core.cck_analysis import analyse_cck_response, analyse_pe_response
from src.core.file_manager import KS_LABEL_FILES, KS_REQUIRED, create_label_lookup, validate_ks_folder
from src.core.firing_rate import process_cluster_data
from src.core.input_parser import parse_channels_or_labels, validate_and_parse_drug_event
from src.core.interactive_plot import export_firing_rate_html
from src.core.isi_hazard import calculate_hazard_function, calculate_isi_histogram, calculate_windowed_isi
from src.core.results_writer import export_data, export_hazard_excel
from src.gui.gui_themes import _dark_theme, _light_theme
from src.gui.view import MainWindow


class AnalysisWorker(QThread):
    """Runs the full analysis pipeline on a background thread."""

    log_message = Signal(str)
    finished = Signal()

    def __init__(
        self,
        file_paths: dict[str, Path],
        cluster_ids: list[int],
        start_time: float,
        end_time: float,
        bin_size: float,
        use_baseline: bool,
        baseline_start: float | None,
        baseline_end: float | None,
        run_hazard: bool,
        peri_hazard: bool,
        early_hazard_start: float,
        early_hazard_end: float,
        export_all_graphs: bool,
        export_txt: bool,
        export_peri_drug: bool,
        cck_time: float | None,
        pe_time: float | None,
        mean_label_data: bool,
        plot_events: list[dict[str, Any]],
        active_drug_events: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.cluster_ids = cluster_ids
        self.start_time = start_time
        self.end_time = end_time
        self.bin_size = bin_size
        self.use_baseline = use_baseline
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.run_hazard = run_hazard
        self.peri_hazard = peri_hazard
        self.early_hazard_start = early_hazard_start
        self.early_hazard_end = early_hazard_end
        self.export_all_graphs = export_all_graphs
        self.export_txt = export_txt
        self.export_peri_drug = export_peri_drug
        self.cck_time = cck_time
        self.pe_time = pe_time
        self.mean_label_data = mean_label_data
        self.plot_events = plot_events
        self.active_drug_events = active_drug_events

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
        finally:
            self.finished.emit()

    def _run(self) -> None:
        log = self.log_message.emit
        data_folder_path = self.file_paths["spike_times.npy"].parent

        log("Loading spike data...")
        from src.core.spike_filter import prepare_filtered_data
        recording_dataframe, max_time = prepare_filtered_data(self.file_paths)

        # Resolve label-based clusters from the loaded dataframe
        cluster_ids = self.cluster_ids

        # Build cluster → group label map for mean-by-label export
        cluster_group_map: dict[int, str] = {}
        if self.mean_label_data:
            for cid in cluster_ids:
                rows = recording_dataframe[recording_dataframe["spike_clusters"] == cid]["group"]
                if not rows.empty:
                    cluster_group_map[cid] = str(rows.iloc[0])

        log(f"Processing {len(cluster_ids)} cluster(s)...")
        raw_fr_dict, baseline_fr_dict = process_cluster_data(
            recording_dataframe,
            cluster_ids,
            self.start_time,
            self.end_time,
            drug_time=None,
            baseline_start=self.baseline_start if self.use_baseline else None,
            baseline_end=self.baseline_end if self.use_baseline else None,
        )

        cck_df = None
        if self.cck_time is not None:
            cck_pre_start = self.cck_time - 300.0
            cck_post_end = self.cck_time + 300.0
            if self.start_time > cck_pre_start:
                log(f"⚠ Warning: analysis start ({self.start_time:.1f}s) is after CCK pre-window start ({cck_pre_start:.1f}s). Bins may be clipped.")
            if self.end_time < cck_post_end:
                log(f"⚠ Warning: analysis end ({self.end_time:.1f}s) is before CCK post-window end ({cck_post_end:.1f}s). Bins may be clipped.")
            log("Running CCK cell-type classification...")
            cck_df = analyse_cck_response(raw_fr_dict, self.cck_time)

        pe_df = None
        if self.pe_time is not None:
            pe_pre_start = self.pe_time - 60.0
            pe_post_end = self.pe_time + 60.0
            if self.start_time > pe_pre_start:
                log(f"⚠ Warning: analysis start ({self.start_time:.1f}s) is after PE pre-window start ({pe_pre_start:.1f}s). Bins may be clipped.")
            if self.end_time < pe_post_end:
                log(f"⚠ Warning: analysis end ({self.end_time:.1f}s) is before PE post-window end ({pe_post_end:.1f}s). Bins may be clipped.")
            log("Running PE cell-type classification...")
            pe_df = analyse_pe_response(raw_fr_dict, self.pe_time)

        log("Exporting spike times and firing rate outputs...")
        export_dir, images_dir, firing_rate_df = export_data(
            raw_fr_dict,
            baseline_fr_dict if self.use_baseline else None,
            data_folder_path,
            self.bin_size,
            self.start_time,
            self.end_time,
            self.baseline_start if self.use_baseline else None,
            self.baseline_end if self.use_baseline else None,
            export_txt=self.export_txt,
            export_delta_from_baseline=self.use_baseline,
            export_baseline_stats=self.use_baseline,
            cck_df=cck_df,
            pe_df=pe_df,
            drug_events=self.active_drug_events,
            cck_time=self.cck_time,
            pe_time=self.pe_time,
            cluster_group_map=cluster_group_map if self.mean_label_data else {},
        )

        if self.run_hazard:
            log("Calculating ISI histograms and hazard functions...")

            # Full-recording ISI + hazard
            isi_df, _ = calculate_isi_histogram(raw_fr_dict)
            hazard_df, hazard_summary_df, _, _ = calculate_hazard_function(isi_df)

            # Early-recording window ISI + hazard
            early_end = min(self.early_hazard_end, self.end_time)
            early_isi_df = calculate_windowed_isi(
                raw_fr_dict, self.early_hazard_start, early_end)
            early_hazard_df, early_hazard_summary_df, _, _ = calculate_hazard_function(
                early_isi_df)
            early_label = f"{self.early_hazard_start:.0f}–{early_end:.0f}s"

            # Per-drug pre/post hazard epochs
            peri_epochs: list[dict] = []
            if self.peri_hazard:
                for ev in self.active_drug_events:
                    onset = float(ev["start"])
                    pre_start = ev.get("pre_time")
                    post_end_raw = ev.get("post_time")
                    if pre_start is None and post_end_raw is None:
                        continue
                    pre_start_f = float(pre_start) if pre_start is not None else onset
                    post_end_f = min(
                        float(post_end_raw) if post_end_raw is not None else onset,
                        self.end_time,
                    )
                    pre_isi = calculate_windowed_isi(
                        raw_fr_dict, pre_start_f, onset, col_suffix="_Pre")
                    post_isi = calculate_windowed_isi(
                        raw_fr_dict, onset, post_end_f, col_suffix="_Post")
                    pre_haz, _, _, _ = calculate_hazard_function(pre_isi)
                    post_haz, _, _, _ = calculate_hazard_function(post_isi)
                    peri_epochs.append({
                        "name": ev["name"],
                        "pre_isi_df": pre_isi,
                        "pre_hazard_df": pre_haz,
                        "post_isi_df": post_isi,
                        "post_hazard_df": post_haz,
                    })

            log("Exporting hazard Excel output...")
            export_hazard_excel(
                export_dir,
                hazard_df,
                hazard_summary_df,
                isi_df,
                early_isi_df=early_isi_df,
                early_hazard_df=early_hazard_df,
                early_hazard_summary_df=early_hazard_summary_df,
                early_hazard_label=early_label,
                peri_epochs=peri_epochs,
            )
        else:
            log("Skipping hazard export (disabled).")

        if firing_rate_df is not None and self.export_all_graphs:
            log("Exporting interactive firing rate plots...")
            # Resolve "max" (inf) drug end times to actual recording end_time
            resolved_events = [
                {**ev, "end": self.end_time} if (ev.get("end") is not None and math.isinf(ev["end"])) else ev
                for ev in self.plot_events
            ]
            export_firing_rate_html(
                firing_rate_df,
                images_dir,
                self.bin_size,
                resolved_events,
            )
        else:
            log("Skipping interactive plots (disabled or no data).")

        log(f"Analysis complete. Files saved to: {export_dir}")


def _get_base_dir() -> Path:
    """Return directory that should hold runtime config (script dir or exe dir)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _parse_float(value: str, default: float | None) -> float | None:
    s = value.strip()
    if not s:
        return default
    return float(s)


TEMP_SETTINGS_PATH: Path = _get_base_dir() / ".neuropixel_gui_last_session.json"


class GUIController:
    def __init__(self) -> None:
        self.view: MainWindow | None = None

    def set_view(self, main_window: MainWindow) -> None:
        self.view = main_window

    def _collect_settings(self) -> dict[str, Any]:
        view = self.view
        assert view is not None
        return {
            "optional_outputs": {
                "export_txt": view.txt_export_checkbox.isChecked(),
                "export_all_graphs": view.all_graphs_checkbox.isChecked(),
                "binned_hazard": view.binned_hazard_checkbox.isChecked(),
                "peri_hazard": view.peri_hazard_checkbox.isChecked(),
                "export_peri_drug": view.peri_drug_checkbox.isChecked(),
            },
            "theme": "dark" if view.dark_mode else "light",
        }

    def export_user_settings(self, parent: QWidget | None) -> None:
        if self.view is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Export Settings", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        settings = self._collect_settings()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
            QMessageBox.information(
                parent, "Export Successful", "Settings saved.")
        except Exception as e:
            QMessageBox.critical(parent, "Export Failed", str(e))

    def import_user_settings(self, parent: QWidget | None) -> None:
        if self.view is None:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Import Settings", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception as e:
            QMessageBox.critical(parent, "Import Failed",
                                 f"Error reading settings file:\n{e}")
            return

        view = self.view
        assert view is not None

        try:
            opts = settings.get("optional_outputs", {})

            view.txt_export_checkbox.setChecked(opts.get("export_txt", True))
            view.all_graphs_checkbox.setChecked(opts.get("export_all_graphs", True))
            view.binned_hazard_checkbox.setChecked(opts.get("binned_hazard", True))
            view.peri_hazard_checkbox.setChecked(opts.get("peri_hazard", True))
            view.peri_drug_checkbox.setChecked(opts.get("export_peri_drug", True))

            theme = settings.get("theme", "light")
            if theme == "dark":
                view.setStyleSheet(_dark_theme())
                view.dark_mode = True
            else:
                view.setStyleSheet(_light_theme())
                view.dark_mode = False

            QMessageBox.information(
                parent, "Import Successful", "Settings loaded.")
        except Exception as e:
            QMessageBox.critical(parent, "Import Failed",
                                 f"Error applying settings:\n{e}")

    def load_temp_settings(self) -> None:
        view = self.view
        if view is None:
            return

        try:
            if not TEMP_SETTINGS_PATH.exists():
                with open(TEMP_SETTINGS_PATH, "w", encoding="utf-8") as f:
                    json.dump({}, f)

                view.setStyleSheet(_light_theme())
                view.dark_mode = False
                return

            with open(TEMP_SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)

            opts = settings.get("optional_outputs", {})

            view.txt_export_checkbox.setChecked(opts.get("export_txt", True))
            view.all_graphs_checkbox.setChecked(opts.get("export_all_graphs", True))
            view.binned_hazard_checkbox.setChecked(opts.get("binned_hazard", True))
            view.peri_hazard_checkbox.setChecked(opts.get("peri_hazard", True))
            view.peri_drug_checkbox.setChecked(opts.get("export_peri_drug", True))

            theme = settings.get("theme", "light")
            if theme == "dark":
                view.setStyleSheet(_dark_theme())
                view.dark_mode = True
            else:
                view.setStyleSheet(_light_theme())
                view.dark_mode = False

        except Exception as e:
            print(f"[Warning] Could not load temp settings: {e}")

    def save_temp_settings(self) -> None:
        if self.view is None:
            return

        settings = self._collect_settings()
        try:
            with open(TEMP_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"[Warning] Could not save temp settings: {e}")

    def add_drug_event(
        self,
        name: str,
        peri_drug: str,
        start_text: str,
        end_text: str,
    ) -> Any:
        return validate_and_parse_drug_event(name, peri_drug, start_text, end_text)

    def try_populate_label_dropdown(
        self,
        found_files: dict[str, Path],
        dropdown: Any,
        log_widget: QTextEdit,
    ) -> None:
        label_path: Path | None = None
        for fname in KS_LABEL_FILES:
            if fname in found_files:
                label_path = found_files[fname]
                break

        if label_path is None:
            log_widget.append("No label file found to populate dropdown.")
            return

        try:
            labels_array = create_label_lookup(label_path)
            unique_labels = sorted(set(labels_array), key=str.lower)

            dropdown.clear()
            dropdown.addItem("Select label...")
            dropdown.addItems(
                [label for label in unique_labels if label != "unknown"])
            log_widget.append("Loaded labels into dropdown.")
        except Exception as e:
            log_widget.append(f"Error loading cluster labels: {e}")

    def run_analysis(
        self,
        folder: str,
        clusters: str,
        start: str,
        end: str,
        bin_size: str,
        baseline_start: str,
        baseline_end: str,
        log: QTextEdit,
        use_baseline: bool,
        run_hazard: bool,
        peri_hazard: bool,
        early_hazard_start: float,
        early_hazard_end: float,
        mean_label_data: bool,
        export_all_graphs: bool,
        export_txt: bool,
        export_peri_drug: bool,
        cck_time: float | None,
        pe_time: float | None,
        drug_events: list[dict[str, Any]],
    ) -> None:
        # ── Input validation (on main thread — fast) ──────────────────────────
        log.append("Validating inputs...")

        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            log.append("Invalid folder path.")
            return

        try:
            file_paths = validate_ks_folder(folder_path, KS_REQUIRED, KS_LABEL_FILES)
        except FileNotFoundError as e:
            log.append(str(e))
            return

        try:
            start_time = _parse_float(start, default=0.0) or 0.0
            bin_size_val = _parse_float(bin_size, default=60.0) or 60.0
            baseline_start_val = _parse_float(baseline_start, default=None)
            baseline_end_val = _parse_float(baseline_end, default=None)
        except ValueError as e:
            log.append(f"Invalid input: {e}")
            return

        # end_time resolved after data load in the worker; pass raw string
        parsed = parse_channels_or_labels(clusters)
        if "error" in parsed:
            log.append(f"Input error: {parsed['error']}")
            return

        cluster_ids: list[int] = parsed["channels"]
        labels: list[str] = parsed["labels"]

        if not cluster_ids and not labels:
            log.append("No valid cluster IDs or labels provided.")
            return

        # Resolve labels → cluster IDs now (needs a quick file read)
        if labels:
            log.append(f"Resolving labels {labels}...")
            from src.core.spike_filter import prepare_filtered_data as _pfd
            tmp_df, max_time_tmp = _pfd(file_paths)
            label_ids = (
                tmp_df[tmp_df["group"].isin(labels)]["spike_clusters"]
                .unique().tolist()
            )
            cluster_ids = sorted(set(cluster_ids) | {int(c) for c in label_ids})
            end_time_val = _parse_float(end, default=float(max_time_tmp)) or float(max_time_tmp)
            log.append(f"  → cluster IDs: {cluster_ids}")
        else:
            # Still need max_time for end_time default — load it lightly
            import numpy as np
            raw_st = np.load(str(file_paths["spike_times.npy"])).ravel()
            max_time_tmp = float(raw_st[-1] / 30000.0) if raw_st.size else 0.0
            end_time_val = _parse_float(end, default=max_time_tmp) or max_time_tmp

        plot_events: list[dict[str, Any]] = list(drug_events)
        if cck_time is not None:
            plot_events.append({"name": "CCK", "start": float(cck_time), "end": None})
        if pe_time is not None:
            plot_events.append({"name": "PE", "start": float(pe_time), "end": None})

        active_drug_events = [
            ev for ev in drug_events
            if ev.get("pre_time") is not None or ev.get("post_time") is not None
        ] if export_peri_drug else []

        # ── Launch worker thread ──────────────────────────────────────────────
        self._worker = AnalysisWorker(
            file_paths=file_paths,
            cluster_ids=cluster_ids,
            start_time=start_time,
            end_time=end_time_val,
            bin_size=bin_size_val,
            use_baseline=use_baseline,
            baseline_start=baseline_start_val,
            baseline_end=baseline_end_val,
            run_hazard=run_hazard,
            peri_hazard=peri_hazard,
            early_hazard_start=early_hazard_start,
            early_hazard_end=early_hazard_end,
            export_all_graphs=export_all_graphs,
            export_txt=export_txt,
            export_peri_drug=export_peri_drug,
            cck_time=cck_time,
            pe_time=pe_time,
            mean_label_data=mean_label_data,
            plot_events=plot_events,
            active_drug_events=active_drug_events,
        )
        self._worker.log_message.connect(log.append)

        if self.view is not None:
            self.view.run_button.setEnabled(False)
            self.view.run_button.setText("Running…")

        def _on_finished() -> None:
            if self.view is not None:
                self.view.run_button.setEnabled(True)
                self.view.run_button.setText("Run Analysis")
            self.save_temp_settings()

        self._worker.finished.connect(_on_finished)
        self._worker.start()
