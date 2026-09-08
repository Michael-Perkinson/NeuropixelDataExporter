import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTextEdit, QWidget

from src.core.cck_analysis import CCK_WINDOW_S, PE_WINDOW_S, analyse_cck_response, analyse_pe_response
from src.core.file_manager import KS_LABEL_FILES, KS_REQUIRED, create_label_lookup, get_recording_duration, validate_ks_folder
from src.core.firing_rate import process_cluster_data
from src.core.input_parser import ParseError, parse_channels_or_labels, validate_and_parse_drug_event
from src.core.interactive_plot import export_firing_rate_html
from src.core.isi_hazard import calculate_hazard_function, calculate_isi_histogram, calculate_windowed_isi
from src.core.results_writer import _cluster_label_map, export_data, export_hazard_excel
from src.gui.gui_themes import _dark_theme, _light_theme
from src.gui.view import MainWindow

logger = logging.getLogger(__name__)


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
        hazard_drug_events: list[dict[str, Any]] | None = None,
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
        self.hazard_drug_events = hazard_drug_events if hazard_drug_events is not None else active_drug_events

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
        recording_dataframe, recording_end, label_log = prepare_filtered_data(
            self.file_paths)
        for msg in label_log:
            log(msg)

        for protocol, onset, window in (("CCK", self.cck_time, CCK_WINDOW_S),
                                        ("PE", self.pe_time, PE_WINDOW_S)):
            if onset is not None and (
                not math.isfinite(onset)
                or onset - window < max(0.0, self.start_time)
                or onset + window > min(self.end_time, recording_end)
            ):
                raise ValueError(f"{protocol} requires a full {window:g}s pre/post window "
                                 "within the recording and analysis window.")

        # Resolve label-based clusters from the loaded dataframe
        cluster_ids = self.cluster_ids

        # Build cluster → group label map for mean-by-label export
        cluster_group_map: dict[int, str] = {}
        if self.mean_label_data:
            for cid in cluster_ids:
                rows = recording_dataframe[recording_dataframe["spike_clusters"]
                                           == cid]["group"]
                if not rows.empty:
                    cluster_group_map[cid] = str(rows.iloc[0])

        log(f"Processing {len(cluster_ids)} cluster(s)...")
        raw_fr_dict, baseline_stats_dict = process_cluster_data(
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
            log("Running CCK cell-type classification...")
            cck_df = analyse_cck_response(raw_fr_dict, self.cck_time)

        pe_df = None
        if self.pe_time is not None:
            log("Running PE cell-type classification...")
            pe_df = analyse_pe_response(raw_fr_dict, self.pe_time)

        log("Exporting spike times and firing rate outputs...")
        export_dir, images_dir, firing_rate_df = export_data(
            raw_fr_dict,
            baseline_stats_dict if self.use_baseline else None,
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

            hazard_spikes, _ = process_cluster_data(
                recording_dataframe, cluster_ids, 0.0, recording_end,
                drug_time=None, baseline_start=None, baseline_end=None,
            )
            # Full-recording ISI + hazard
            isi_df, _ = calculate_isi_histogram(hazard_spikes)
            hazard_df, hazard_summary_df, _, _ = calculate_hazard_function(
                isi_df)

            # Early-recording window ISI + hazard
            early_end = min(self.early_hazard_end, recording_end)
            early_isi_df = calculate_windowed_isi(
                hazard_spikes, self.early_hazard_start, early_end)
            early_hazard_df, early_hazard_summary_df, _, _ = calculate_hazard_function(
                early_isi_df)
            early_label = f"{self.early_hazard_start:.0f}–{early_end:.0f}s"

            # Per-drug pre/post hazard epochs (1 bin before onset; 1 bin at end of drug)
            peri_epochs: list[dict[str, Any]] = []
            if self.peri_hazard:
                for ev in self.hazard_drug_events:
                    onset = float(ev["start"])
                    drug_end_raw = ev.get("end")

                    # 1 bin immediately before onset
                    pre_win_start = max(0.0, onset - self.bin_size)
                    pre_win_end = onset

                    pre_isi = calculate_windowed_isi(
                        hazard_spikes, pre_win_start, pre_win_end, col_suffix="_PreDrug")
                    pre_haz, pre_haz_summary, _, _ = calculate_hazard_function(
                        pre_isi)

                    epoch: dict[str, Any] = {
                        "name": ev["name"],
                        "pre_win_start": pre_win_start,
                        "pre_win_end": pre_win_end,
                        "pre_isi_df": pre_isi,
                        "pre_hazard_df": pre_haz,
                        "pre_hazard_summary_df": pre_haz_summary,
                    }

                    # 1 bin at the end of drug application (only if drug has an end time)
                    if drug_end_raw is not None:
                        import math as _math
                        drug_end_f = recording_end if _math.isinf(
                            float(drug_end_raw)) else float(drug_end_raw)
                        drug_end_f = min(drug_end_f, recording_end)
                        end_win_start = max(0.0, drug_end_f - self.bin_size)
                        end_win_end = drug_end_f

                        end_isi = calculate_windowed_isi(
                            hazard_spikes, end_win_start, end_win_end, col_suffix="_EndDrug")
                        end_haz, end_haz_summary, _, _ = calculate_hazard_function(
                            end_isi)

                        epoch["end_win_start"] = end_win_start
                        epoch["end_win_end"] = end_win_end
                        epoch["end_isi_df"] = end_isi
                        epoch["end_hazard_df"] = end_haz
                        epoch["end_hazard_summary_df"] = end_haz_summary

                    peri_epochs.append(epoch)

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
                label_map=_cluster_label_map(cck_df, pe_df),
            )
        else:
            log("Skipping hazard export (disabled).")

        if firing_rate_df is not None and self.export_all_graphs:
            log("Exporting interactive firing rate plots...")
            # Resolve "max" (inf) drug end times to actual recording end_time
            resolved_events = [
                {**ev, "end": self.end_time} if (ev.get("end")
                                                 is not None and math.isinf(ev["end"])) else ev
                for ev in self.plot_events
            ]
            export_firing_rate_html(
                firing_rate_df,
                images_dir,
                self.bin_size,
                [{"name": ev["name"], "start": ev["start"], "end": ev.get("end")} for ev in resolved_events],
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
    if s.lower() in ("max", "inf"):
        return default
    # strip trailing "(max)" hint added by the UI
    s = s.split("(")[0].strip()
    if not s:
        return default
    return float(s)


TEMP_SETTINGS_PATH: Path = _get_base_dir() / ".neuropixel_gui_last_session.json"


class GUIController:
    def __init__(self) -> None:
        self.view: MainWindow | None = None
        self.last_browse_dir: Path | None = None

    def set_view(self, main_window: MainWindow) -> None:
        self.view = main_window

    def _collect_settings(self) -> dict[str, Any]:
        view = self.view
        assert view is not None
        settings: dict[str, Any] = {
            "optional_outputs": {
                "export_txt": view.txt_export_checkbox.isChecked(),
                "export_all_graphs": view.all_graphs_checkbox.isChecked(),
                "binned_hazard": view.binned_hazard_checkbox.isChecked(),
                "peri_hazard": view.peri_hazard_checkbox.isChecked(),
                "export_peri_drug": view.peri_drug_checkbox.isChecked(),
            },
            "theme": "dark" if view.dark_mode else "light",
        }
        if self.last_browse_dir is not None:
            settings["last_browse_dir"] = str(self.last_browse_dir)
        return settings

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
            if parent is not None:
                QMessageBox.information(
                    parent, "Export Successful", "Settings saved.")
        except Exception as e:
            if parent is not None:
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
            if parent is not None:
                QMessageBox.critical(parent, "Import Failed",
                                     f"Error reading settings file:\n{e}")
            return

        view = self.view
        assert view is not None

        try:
            opts = settings.get("optional_outputs", {})

            view.txt_export_checkbox.setChecked(opts.get("export_txt", True))
            view.all_graphs_checkbox.setChecked(
                opts.get("export_all_graphs", True))
            view.binned_hazard_checkbox.setChecked(
                opts.get("binned_hazard", True))
            view.peri_hazard_checkbox.setChecked(opts.get("peri_hazard", True))
            view.peri_drug_checkbox.setChecked(
                opts.get("export_peri_drug", True))

            theme = settings.get("theme", "light")
            if theme == "dark":
                view.setStyleSheet(_dark_theme())
                view.dark_mode = True
            else:
                view.setStyleSheet(_light_theme())
                view.dark_mode = False

            if parent is not None:
                QMessageBox.information(
                    parent, "Import Successful", "Settings loaded.")
        except Exception as e:
            if parent is not None:
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
            view.all_graphs_checkbox.setChecked(
                opts.get("export_all_graphs", True))
            view.binned_hazard_checkbox.setChecked(
                opts.get("binned_hazard", True))
            view.peri_hazard_checkbox.setChecked(opts.get("peri_hazard", True))
            view.peri_drug_checkbox.setChecked(
                opts.get("export_peri_drug", True))

            theme = settings.get("theme", "light")
            if theme == "dark":
                view.setStyleSheet(_dark_theme())
                view.dark_mode = True
            else:
                view.setStyleSheet(_light_theme())
                view.dark_mode = False

            raw_dir = settings.get("last_browse_dir")
            if raw_dir:
                p = Path(raw_dir)
                if p.exists():
                    self.last_browse_dir = p

        except Exception as e:
            logger.warning("Could not load temp settings: %s", e)

    def save_temp_settings(self) -> None:
        if self.view is None:
            return

        settings = self._collect_settings()
        try:
            with open(TEMP_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            logger.warning("Could not save temp settings: %s", e)

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
    ) -> float | None:
        label_path: Path | None = None
        for fname in KS_LABEL_FILES:
            if fname in found_files:
                label_path = found_files[fname]
                break

        if label_path is None:
            log_widget.append("No label file found to populate dropdown.")
            return None

        try:
            labels_array, _ = create_label_lookup(label_path)
            unique_labels = sorted(set(labels_array), key=str.lower)

            dropdown.clear()
            dropdown.addItem("Select label...")
            dropdown.addItems(
                [label for label in unique_labels if label != "unknown"])
            log_widget.append("Loaded labels into dropdown.")
        except Exception as e:
            log_widget.append(f"Error loading cluster labels: {e}")

        max_time: float | None = None
        if "spike_times.npy" in found_files:
            max_time = get_recording_duration(found_files["spike_times.npy"])
        return max_time

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
            file_paths = validate_ks_folder(
                folder_path, KS_REQUIRED, KS_LABEL_FILES)
        except FileNotFoundError as e:
            log.append(str(e))
            return

        try:
            start_time = _parse_float(start, default=0.0) or 0.0
            bin_size_val = _parse_float(bin_size, default=600.0) or 600.0
            baseline_start_val = _parse_float(baseline_start, default=None)
            baseline_end_val = _parse_float(baseline_end, default=None)
        except ValueError as e:
            log.append(f"Invalid input: {e}")
            return

        # CCK/PE window feasibility checks (before launching the worker)
        if cck_time is not None and (cck_time - 300.0) < start_time:
            needed = start_time + 300.0
            log.append(
                f"⚠ CCK time ({cck_time:.1f}s) is less than 5 minutes after the "
                f"analysis start ({start_time:.1f}s). The CCK protocol requires a full "
                f"5-minute pre-window. Please enter a CCK time ≥ {needed:.1f}s."
            )
            return

        if pe_time is not None and (pe_time - 60.0) < start_time:
            needed = start_time + 60.0
            log.append(
                f"⚠ PE time ({pe_time:.1f}s) is less than 1 minute after the "
                f"analysis start ({start_time:.1f}s). The PE protocol requires a full "
                f"1-minute pre-window. Please enter a PE time ≥ {needed:.1f}s."
            )
            return

        # end_time resolved after data load in the worker; pass raw string
        parsed = parse_channels_or_labels(clusters)
        if "error" in parsed:
            log.append(f"Input error: {cast(ParseError, parsed)['error']}")
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
            tmp_df, max_time_tmp, tmp_label_log = _pfd(file_paths)
            for msg in tmp_label_log:
                log.append(msg)

            # Case-insensitive label matching
            labels_lower = [lbl.lower() for lbl in labels]
            standard_labels = {"good", "mua", "noise"}
            has_cluster_info = any(
                "cluster_info.tsv" in m for m in tmp_label_log)
            for lbl in labels_lower:
                if lbl not in standard_labels and not has_cluster_info:
                    log.append(
                        f"  ⚠ Label '{lbl}' is not a standard Phy label (good/mua/noise) "
                        "and cluster_info.tsv was not found — it may not match any clusters."
                    )

            label_ids = (
                tmp_df[tmp_df["group"].str.lower().isin(labels_lower)
                       ]["spike_clusters"]
                .unique().tolist()
            )
            cluster_ids = sorted(set(cluster_ids) | {
                                 int(c) for c in label_ids})
            end_time_val = _parse_float(end, default=float(
                max_time_tmp)) or float(max_time_tmp)
            log.append(f"  → cluster IDs: {cluster_ids}")
        else:
            # Still need max_time for end_time default — load it lightly
            import numpy as np
            raw_st = np.load(str(file_paths["spike_times.npy"])).ravel()
            max_time_tmp = float(raw_st[-1] / 30000.0) if raw_st.size else 0.0
            end_time_val = _parse_float(
                end, default=max_time_tmp) or max_time_tmp

        plot_events: list[dict[str, Any]] = list(drug_events)
        if cck_time is not None:
            plot_events.append(
                {"name": "CCK", "start": float(cck_time), "end": None})
        if pe_time is not None:
            plot_events.append(
                {"name": "PE", "start": float(pe_time), "end": None})

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
            hazard_drug_events=drug_events,
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
