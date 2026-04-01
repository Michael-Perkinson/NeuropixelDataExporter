from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import ExcelWriter

from src.core.firing_rate import (
    calculate_firing_rate,
    create_baselined_df,
    create_firing_rate_dataframes,
)
from src.core.file_manager import make_output_folders


def _cluster_label_map(
    cck_df: pd.DataFrame | None,
    pe_df: pd.DataFrame | None,
) -> dict[str, str]:
    """
    Build a mapping of "Cluster_N" → classification string from
    the first available cell-typing result (CCK preferred over PE).
    Returns empty dict if neither is available.
    """
    source = cck_df if (cck_df is not None and not cck_df.empty) else pe_df
    if source is None or source.empty:
        return {}
    return dict(zip(source["Cluster"], source["Classification"]))


def _rename_cluster_columns(
    df: pd.DataFrame,
    label_map: dict[str, str],
) -> pd.DataFrame:
    """
    Rename columns like "Cluster_1" → "Cluster_1 (Putative Oxytocin)".
    Non-cluster columns (e.g. "Time Intervals (s)") are left unchanged.
    """
    if not label_map:
        return df
    rename = {
        col: f"{col} ({label_map[col]})"
        for col in df.columns
        if col in label_map
    }
    return df.rename(columns=rename)


def _trim_cell_typing_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape cell-typing output to:
      Cluster | Pre_Mean_FR_Hz | Post_Mean_FR_Hz | Delta_FR_Hz | Classification | Baseline_Stability
    Baseline_Stability categorises the pre-window slope as Stable / Small / Medium / Large drift.
    """
    df = df.copy()

    if "Baseline_Slope_Hz_per_min" in df.columns:
        def _stability(slope: float) -> str:
            if math.isnan(slope):
                return "Unknown"
            abs_s = abs(slope)
            direction = "rising" if slope > 0 else "falling"
            if abs_s <= 0.1:
                return "Stable"
            elif abs_s <= 0.3:
                return f"Small {direction} drift ({slope:+.2f} Hz/min)"
            elif abs_s <= 0.6:
                return f"Medium {direction} drift ({slope:+.2f} Hz/min)"
            else:
                return f"Large {direction} drift ({slope:+.2f} Hz/min) ⚠"

        df["Baseline_Stability"] = df["Baseline_Slope_Hz_per_min"].apply(
            _stability)

    keep = ["Cluster", "Pre_Mean_FR_Hz", "Post_Mean_FR_Hz",
            "Delta_FR_Hz", "Classification", "Baseline_Stability"]
    return df[[c for c in keep if c in df.columns]]


def _build_summary_sheet(
    start_time: float,
    end_time: float,
    bin_size: float,
    baseline_start: float | None,
    baseline_end: float | None,
    cck_time: float | None,
    pe_time: float | None,
    drug_events: list[dict[str, Any]],
    peri_warnings: list[str],
    cck_df: pd.DataFrame | None = None,
    pe_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a tidy key-value summary DataFrame describing the analysis parameters.
    """
    rows: list[dict[str, str]] = []

    def _add(section: str, key: str, value: str) -> None:
        rows.append({"Section": section, "Parameter": key, "Value": value})

    _add("Analysis Window", "Start (s)", str(start_time))
    _add("Analysis Window", "End (s)", str(end_time))
    _add("Analysis Window", "Bin Size (s)", str(bin_size))

    if baseline_start is not None and baseline_end is not None:
        _add("Baseline", "Start (s)", str(baseline_start))
        _add("Baseline", "End (s)", str(baseline_end))
    else:
        _add("Baseline", "Status", "Not used")

    if cck_time is not None:
        _add("CCK (IV)", "Injection Time (s)", str(cck_time))
        _add("CCK (IV)", "Protocol", "5 min pre vs 5 min post (1 min bins)")
        _add("CCK (IV)", "OT Threshold", "delta >= +0.5 Hz → Putative Oxytocin")
        _add("CCK (IV)", "VP Classification",
             "delta < +0.5 Hz → Putative Vasopressin")
        if cck_df is not None and not cck_df.empty and "Classification" in cck_df.columns:
            total_n = len(cck_df)
            ot_n = int((cck_df["Classification"] == "Putative Oxytocin").sum())
            vp_n = int((cck_df["Classification"] ==
                       "Putative Vasopressin").sum())
            _add("CCK (IV)", "Total Neurons", str(total_n))
            _add("CCK (IV)", "Putative Oxytocin", str(ot_n))
            _add("CCK (IV)", "Putative Vasopressin", str(vp_n))
    else:
        _add("CCK (IV)", "Status", "Not used")

    if pe_time is not None:
        _add("PE (IV)", "Injection Time (s)", str(pe_time))
        _add("PE (IV)", "Protocol", "1 min pre vs 1 min post (10 s bins)")
        _add("PE (IV)", "VP Threshold", "delta <= -0.5 Hz → Putative Vasopressin")
        _add("PE (IV)", "OT Classification",
             "delta > -0.5 Hz → Putative Oxytocin")
        if pe_df is not None and not pe_df.empty and "Classification" in pe_df.columns:
            total_n = len(pe_df)
            ot_n = int((pe_df["Classification"] == "Putative Oxytocin").sum())
            vp_n = int((pe_df["Classification"] ==
                       "Putative Vasopressin").sum())
            _add("PE (IV)", "Total Neurons", str(total_n))
            _add("PE (IV)", "Putative Oxytocin", str(ot_n))
            _add("PE (IV)", "Putative Vasopressin", str(vp_n))
    else:
        _add("PE (IV)", "Status", "Not used")

    for ev in drug_events:
        name = ev.get("name", "Unknown")
        pre = ev.get("pre_time")
        post = ev.get("post_time")
        drug_start = ev.get("start", "?")
        drug_end = ev.get("end")
        drug_end_str = (
            "max (recording end)" if (drug_end is not None and math.isinf(drug_end))
            else (f"{drug_end}s" if drug_end is not None else "N/A (acute injection)")
        )
        if pre is not None or post is not None:
            pre_str = f"{pre:.1f}s" if pre is not None else "?"
            post_str = ("max" if (post is not None and math.isinf(post))
                        else (f"{post:.1f}s" if post is not None else "?"))
            peri_size = f"{pre_str} – {post_str}"
        else:
            peri_size = "No peri window"
        _add(f"Drug: {name}", "Onset (s)", str(drug_start))
        _add(f"Drug: {name}", "Offset (s)", drug_end_str)
        _add(f"Drug: {name}", "Peri Window", peri_size)

    for warning in peri_warnings:
        _add("Warnings", "⚠", warning)

    df = pd.DataFrame(rows)
    # Show each section name only on its first row — cleaner to read
    df["Section"] = df["Section"].where(
        df["Section"] != df["Section"].shift(), "")
    return df


def _build_peri_sheet(
    raw_fr_dict: dict[int, NDArray[np.float64]],
    bin_size: float,
    drug_onset: float,
    pre_abs: float,
    post_abs: float,
    recording_end: float,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, str | None]:
    """
    Build a peri-drug firing rate DataFrame with time reset to 0 at drug onset.
    Returns (df, warning_message_or_None).
    """
    warning: str | None = None
    clipped_end = min(post_abs, recording_end)
    # Only warn if a specific finite time was requested but exceeds the recording
    if not math.isinf(post_abs) and post_abs > recording_end:
        warning = (
            f"Peri-drug post-window ({post_abs:.1f}s) exceeds recording end "
            f"({recording_end:.1f}s). Sheet clipped at {clipped_end:.1f}s."
        )

    peri_raw, _ = calculate_firing_rate(
        raw_fr_dict, bin_size, pre_abs, clipped_end
    )
    peri_df, _ = create_firing_rate_dataframes(peri_raw, None)

    # Shift time axis so t=0 = drug onset
    peri_df["Time Intervals (s)"] = (
        peri_df["Time Intervals (s)"] - drug_onset
    ).round(4)

    # Rename cluster columns with cell-type labels
    peri_df = _rename_cluster_columns(peri_df, label_map)

    return peri_df, warning


def export_spike_times_csv(
    data_export: dict[int, NDArray[np.float64]],
    csv_path: Path,
) -> None:
    max_length = max((arr.size for arr in data_export.values()), default=0)
    if max_length == 0:
        return

    df_spikes = pd.DataFrame(
        {
            f"Cluster_{cid}": np.pad(
                arr.astype(np.float64),
                (0, max_length - arr.size),
                constant_values=np.nan,
            )
            for cid, arr in data_export.items()
        }
    )
    df_spikes.to_csv(csv_path, index=False)


def export_spike_times_txt(
    data_export: dict[int, NDArray[np.float64]],
    txt_export_dir: Path,
) -> None:
    for cid, arr in data_export.items():
        out_file = txt_export_dir / f"spike_times_Cluster_{cid}_time_ms.txt"
        np.savetxt(out_file, arr.astype(np.float64), fmt="%.4f")


def export_data(
    data_export: dict[int, NDArray[np.float64]],
    baseline_fr_dict: dict[int, float | None] | None,
    data_folder_path: Path,
    bin_size: float,
    start_time: float,
    end_time: float,
    baseline_start: float | None,
    baseline_end: float | None,
    *,
    export_spike_csv: bool = True,
    export_txt: bool = True,
    export_firing_rate_xlsx: bool = True,
    export_delta_from_baseline: bool = True,
    export_baseline_stats: bool = True,
    cck_df: pd.DataFrame | None = None,
    pe_df: pd.DataFrame | None = None,
    drug_events: list[dict[str, Any]] | None = None,
    cck_time: float | None = None,
    pe_time: float | None = None,
    cluster_group_map: dict[int, str] | None = None,
) -> tuple[Path, Path, pd.DataFrame | None]:
    """
    Export spike times and firing rate data.

    Sheet order in the xlsx:
      1. Summary
      2. CCK_Cell_Typing (if CCK used)
      3. PE_Cell_Typing (if PE used)
      4. Baseline_Mean_and_SD (if baseline used)
      5. Peri_<Drug> (one per drug event with a peri window)
      5b. Peri_<Drug>_Delta (if baseline used)
      6. Mean_by_Label_Peri (all drugs combined, if mean_label_data)
      7. Binned_Firing_Rates  ← always last

    Returns:
      (export_dir, images_dir, raw_firing_rate_df_or_None)
    """
    export_dir, images_dir, txt_export_dir = make_output_folders(
        data_folder_path)

    filtered_export: dict[int, NDArray[np.float64]] = {
        cid: arr for cid, arr in data_export.items() if arr.size > 0
    }
    if not filtered_export:
        return export_dir, images_dir, None

    if export_spike_csv:
        export_spike_times_csv(
            filtered_export,
            export_dir / "spike_times_by_cluster_time_ms.csv",
        )

    if export_txt:
        export_spike_times_txt(filtered_export, txt_export_dir)

    if not export_firing_rate_xlsx:
        return export_dir, images_dir, None

    raw_data, _ = calculate_firing_rate(
        filtered_export,
        bin_size,
        start_time,
        end_time,
        None,
    )
    df_raw, _ = create_firing_rate_dataframes(raw_data, None)

    if df_raw.empty:
        return export_dir, images_dir, None

    label_map = _cluster_label_map(cck_df, pe_df)
    has_baseline = (
        export_delta_from_baseline
        and baseline_fr_dict is not None
        and baseline_start is not None
        and baseline_end is not None
    )
    peri_warnings: list[str] = []

    def _bin_truncation_warning(label: str, win_start: float, win_end: float) -> str | None:
        """Return a warning string if the window is not a whole multiple of bin_size."""
        duration = win_end - win_start
        remainder = duration % bin_size
        if remainder < 1e-6:
            return None
        n_full = int(duration // bin_size)
        last_bin_end = win_start + n_full * bin_size
        return (
            f"{label}: window {win_start:.1f}s–{win_end:.1f}s ({duration:.1f}s) "
            f"is not a whole multiple of bin size ({bin_size:.1f}s). "
            f"{n_full} full bin(s) used; last {remainder:.1f}s ({last_bin_end:.1f}s–{win_end:.1f}s) not included."
        )

    main_trunc_warn = _bin_truncation_warning("Main recording", start_time, end_time)
    if main_trunc_warn:
        peri_warnings.append(main_trunc_warn)

    xlsx_path = export_dir / "firing_rates_by_cluster.xlsx"
    with ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:

        # CCK / PE cell typing
        if cck_df is not None and not cck_df.empty:
            _trim_cell_typing_df(cck_df).to_excel(
                writer, sheet_name="CCK_Cell_Typing", index=False)

        if pe_df is not None and not pe_df.empty:
            _trim_cell_typing_df(pe_df).to_excel(
                writer, sheet_name="PE_Cell_Typing", index=False)

        # Baseline mean and SD
        if export_baseline_stats and baseline_start is not None and baseline_end is not None:
            baselined_df = create_baselined_df(
                baseline_start, baseline_end, bin_size, filtered_export)
            baseline_sheet = f"Baseline_Mean_and_SD ({baseline_start:.0f}s-{baseline_end:.0f}s)"
            baselined_df.to_excel(
                writer, sheet_name=baseline_sheet[:31], index=False)

        # Peri-drug sheets (raw + optional delta)
        # Accumulated per-drug means for the combined sheet
        all_drug_means: list[tuple[str, pd.DataFrame]] = []

        for ev in (drug_events or []):
            pre_abs = ev.get("pre_time")
            post_abs = ev.get("post_time")
            if pre_abs is None and post_abs is None:
                continue

            drug_onset = float(ev["start"])
            pre_abs_f = float(pre_abs) if pre_abs is not None else drug_onset
            post_abs_f = float(
                post_abs) if post_abs is not None else drug_onset

            peri_df, warning = _build_peri_sheet(
                filtered_export, bin_size,
                drug_onset, pre_abs_f, post_abs_f,
                end_time, label_map,
            )
            if warning:
                peri_warnings.append(f"{ev['name']}: {warning}")

            trunc = _bin_truncation_warning(ev["name"], pre_abs_f, min(post_abs_f, end_time))
            if trunc:
                peri_warnings.append(trunc)

            safe_name = ev["name"].replace(" ", "_")[:20]
            peri_df.to_excel(writer, sheet_name=f"Peri_{safe_name}"[
                             :31], index=False)

            # Delta-from-baseline peri sheet
            if has_baseline and baseline_fr_dict is not None:
                clipped_end = min(post_abs_f, end_time)
                peri_raw_data, peri_delta_data = calculate_firing_rate(
                    filtered_export, bin_size, pre_abs_f, clipped_end,
                    baseline_fr_dict,
                )
                _, peri_delta_df = create_firing_rate_dataframes(
                    peri_raw_data, peri_delta_data)
                if peri_delta_df is not None and len(peri_delta_df.columns) > 1:
                    peri_delta_df["Time Intervals (s)"] = (
                        peri_delta_df["Time Intervals (s)"] - drug_onset
                    ).round(4)
                    peri_delta_df = _rename_cluster_columns(
                        peri_delta_df, label_map)
                    peri_delta_df.to_excel(
                        writer, sheet_name=f"Peri_{safe_name}_Delta"[:31], index=False)

            # Accumulate per-drug mean-by-label rows for combined sheet
            if cluster_group_map:
                clipped_end = min(post_abs_f, end_time)
                lbl_cid_groups: dict[str, list[int]] = {}
                for cid, lbl in cluster_group_map.items():
                    if cid in filtered_export:
                        lbl_cid_groups.setdefault(lbl, []).append(cid)
                if lbl_cid_groups:
                    peri_raw_all, _ = calculate_firing_rate(
                        {cid: filtered_export[cid]
                         for cid in cluster_group_map if cid in filtered_export},
                        bin_size, pre_abs_f, clipped_end,
                    )
                    peri_raw_df, _ = create_firing_rate_dataframes(
                        peri_raw_all, None)
                    peri_raw_df["Time Intervals (s)"] = (
                        peri_raw_df["Time Intervals (s)"] - drug_onset
                    ).round(4)
                    mean_peri = pd.DataFrame(
                        {"Time (s, 0=onset)": peri_raw_df["Time Intervals (s)"]})
                    for lbl, cids in sorted(lbl_cid_groups.items()):
                        cols = [
                            f"Cluster_{cid}" for cid in cids
                            if f"Cluster_{cid}" in peri_raw_df.columns]
                        if cols:
                            mean_peri[f"Mean_{lbl}_Hz"] = peri_raw_df[cols].mean(
                                axis=1)
                    all_drug_means.append((ev["name"], mean_peri))

        # Combined mean-by-label peri sheet (all drugs side by side)
        if cluster_group_map and all_drug_means:
            combined_parts: list[pd.DataFrame] = []
            for drug_name, mean_df in all_drug_means:
                header = pd.DataFrame(
                    [[f"--- {drug_name} ---"] + [""]
                        * (len(mean_df.columns) - 1)],
                    columns=mean_df.columns,
                )
                combined_parts.extend(
                    [header, mean_df, pd.DataFrame(columns=mean_df.columns)])
            combined_means = pd.concat(combined_parts, ignore_index=True)
            combined_means.to_excel(
                writer, sheet_name="Mean_by_Label_Peri", index=False)

        guide_df = _build_fr_guide_sheet(
            start_time, end_time,
            baseline_start, baseline_end,
            cck_df is not None and not cck_df.empty,
            pe_df is not None and not pe_df.empty,
            has_baseline,
            drug_events or [],
            cluster_group_map is not None and bool(all_drug_means),
        )
        guide_df.to_excel(writer, sheet_name="Sheet_Guide", index=False)

        # Summary written last so warnings are included, then moved to position 0
        summary_df = _build_summary_sheet(
            start_time, end_time, bin_size,
            baseline_start, baseline_end,
            cck_time, pe_time,
            drug_events or [],
            peri_warnings,
            cck_df=cck_df,
            pe_df=pe_df,
        )
        wb = writer.book
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        wb.worksheets_objs.sort(
            key=lambda ws: (0 if ws.name == "Sheet_Guide" else 1 if ws.name == "Summary" else 2)
        )

        # Binned_Firing_Rates is always the last sheet
        df_raw_renamed = _rename_cluster_columns(df_raw, label_map)
        df_raw_renamed.to_excel(
            writer, sheet_name="Binned_Firing_Rates", index=False)

    return export_dir, images_dir, df_raw


def _build_fr_guide_sheet(
    start_time: float,
    end_time: float,
    baseline_start: float | None,
    baseline_end: float | None,
    has_cck: bool,
    has_pe: bool,
    has_baseline: bool,
    drug_events: list[dict[str, Any]],
    has_mean_label: bool,
) -> pd.DataFrame:
    """Build a Sheet_Guide tab describing every sheet in the firing-rate workbook."""
    rows: list[dict[str, str]] = []

    def _add(section: str, sheet: str, description: str) -> None:
        rows.append({"Section": section, "Sheet": sheet,
                    "Description": description})

    _add("Overview", "Sheet_Guide",
         "This sheet — plain-English description of every tab and what it contains (always first)")
    _add("Overview", "Summary",
         "Recording parameters, protocol details, neuron counts, drug event windows, and any clipping warnings")

    if has_cck:
        _add("Cell Typing", "CCK_Cell_Typing",
             "CCK protocol: Pre/Post mean FR, delta, classification (Putative OT / VP), and baseline stability")
    if has_pe:
        _add("Cell Typing", "PE_Cell_Typing",
             "PE protocol: Pre/Post mean FR, delta, classification (Putative OT / VP), and baseline stability")

    if has_baseline and baseline_start is not None and baseline_end is not None:
        lbl = f"{baseline_start:.0f}s–{baseline_end:.0f}s"
        _add("Baseline", f"Baseline_Mean_and_SD ({lbl})",
             f"Per-cluster mean and SD firing rate across the baseline window ({lbl}); mean used to compute delta sheets")

    for ev in drug_events:
        pre_abs = ev.get("pre_time")
        post_abs = ev.get("post_time")
        if pre_abs is None and post_abs is None:
            continue
        safe = ev["name"].replace(" ", "_")[:20]
        onset = float(ev["start"])
        pre_f = float(pre_abs) if pre_abs is not None else onset
        post_f = float(post_abs) if post_abs is not None else onset
        window = f"{pre_f:.0f}s–{min(post_f, end_time):.0f}s (t=0 at {onset:.0f}s)"
        _add(f"Peri-Drug: {ev['name']}", f"Peri_{safe}",
             f"Binned firing rates over the peri-drug window; {window}")
        if has_baseline:
            _add(f"Peri-Drug: {ev['name']}", f"Peri_{safe}_Delta",
                 f"Change in firing rate from baseline over the same peri-drug window; {window}")

    if has_mean_label:
        _add("Mean by Label", "Mean_by_Label_Peri",
             "All peri-drug windows combined; firing rates averaged by Phy group label; each drug block separated by a header row")
        _add("Mean by Label", "Mean_by_Label",
             "Full-recording firing rates averaged per Phy group label")

    _add("Raw Data", "Binned_Firing_Rates",
         f"Per-cluster binned firing rates across the full recording window ({start_time:.0f}s–{end_time:.0f}s); always the last sheet")

    df = pd.DataFrame(rows, columns=["Section", "Sheet", "Description"])
    df["Section"] = df["Section"].where(
        df["Section"] != df["Section"].shift(), "")
    return df


def _build_hazard_summary_sheet(
    early_hazard_label: str,
    early_hazard_summary_df: pd.DataFrame | None,
    peri_epochs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a guide sheet describing every tab in the hazard workbook."""
    rows: list[dict[str, str]] = []

    def _add(section: str, sheet: str, description: str) -> None:
        rows.append({"Section": section, "Sheet": sheet,
                    "Description": description})

    _add("Full Recording", "Full_ISI",
         "ISI histogram counts across the entire recording")
    _add("Full Recording", "Full_Hazard",
         "Hazard function values across the entire recording")
    _add("Full Recording", "Full_Hazard_Summary",
         "Peak early hazard, mean late hazard, and hazard ratio per cluster — full recording")

    lbl = early_hazard_label
    _add("Early Window", f"Early_ISI ({lbl[:12]})",
         f"ISI histogram for the early reference window ({lbl})")
    _add("Early Window", f"Early_Hazard ({lbl[:12]})",
         f"Hazard function for the early reference window ({lbl})")
    _add("Early Window", "Early_Hazard_Summary",
         f"Hazard summary metrics for the early reference window ({lbl})")

    if early_hazard_summary_df is not None:
        pass  # already described above

    for epoch in peri_epochs:
        name = epoch["name"]
        safe = name.replace(" ", "_")[:14]
        pre_s = epoch.get("pre_win_start", "?")
        pre_e = epoch.get("pre_win_end", "?")
        _add(
            f"Drug: {name}",
            f"{safe}_PreDrug_ISI",
            f"ISI histogram — 1 bin immediately before {name} onset ({pre_s:.0f}–{pre_e:.0f}s)",
        )
        _add(
            f"Drug: {name}",
            f"{safe}_PreDrug_Hazard",
            f"Hazard function — 1 bin before {name} onset ({pre_s:.0f}–{pre_e:.0f}s)",
        )
        _add(
            f"Drug: {name}",
            f"{safe}_PreDrug_HazSumm",
            f"Hazard summary — pre-drug window ({pre_s:.0f}–{pre_e:.0f}s)",
        )
        if "end_isi_df" in epoch:
            end_s = epoch.get("end_win_start", "?")
            end_e = epoch.get("end_win_end", "?")
            _add(
                f"Drug: {name}",
                f"{safe}_EndDrug_ISI",
                f"ISI histogram — 1 bin at end of {name} application ({end_s:.0f}–{end_e:.0f}s)",
            )
            _add(
                f"Drug: {name}",
                f"{safe}_EndDrug_Hazard",
                f"Hazard function — end-of-drug window ({end_s:.0f}–{end_e:.0f}s)",
            )
            _add(
                f"Drug: {name}",
                f"{safe}_EndDrug_HazSumm",
                f"Hazard summary — end-of-drug window ({end_s:.0f}–{end_e:.0f}s)",
            )

    df = pd.DataFrame(rows)
    df["Section"] = df["Section"].where(
        df["Section"] != df["Section"].shift(), "")
    return df


def export_hazard_excel(
    export_dir: Path,
    hazard_df: pd.DataFrame,
    hazard_summary_df: pd.DataFrame,
    isi_df: pd.DataFrame,
    *,
    early_isi_df: pd.DataFrame | None = None,
    early_hazard_df: pd.DataFrame | None = None,
    early_hazard_summary_df: pd.DataFrame | None = None,
    early_hazard_label: str = "Early",
    peri_epochs: list[dict[str, Any]] | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    """
    Write ISI and hazard data to Excel.

    Sheets written:
      Summary  — guide to all tabs
      Full_ISI / Full_Hazard / Full_Hazard_Summary  — whole recording
      Early_ISI (<label>) / Early_Hazard / Early_Hazard_Summary  — early window
      <Drug>_PreDrug_ISI / _Hazard / _HazSumm  — 1 bin before drug onset
      <Drug>_EndDrug_ISI / _Hazard / _HazSumm  — 1 bin at end of drug (if applicable)
    """
    lm = label_map or {}
    epochs = peri_epochs or []

    def _rename_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Remap values in the 'Cluster' column using label_map."""
        if not lm or "Cluster" not in df.columns:
            return df
        df = df.copy()
        df["Cluster"] = df["Cluster"].map(
            lambda c: f"{c} ({lm[c]})" if c in lm else c
        )
        return df

    excel_path = export_dir / "isi_and_hazard_analysis.xlsx"
    with ExcelWriter(excel_path, engine="xlsxwriter") as writer:

        # Summary guide sheet (written first, stays first)
        _build_hazard_summary_sheet(
            early_hazard_label, early_hazard_summary_df, epochs
        ).to_excel(writer, sheet_name="Summary", index=False)

        _rename_cluster_columns(isi_df, lm).to_excel(
            writer, sheet_name="Full_ISI", index=False)
        _rename_cluster_columns(hazard_df, lm).to_excel(
            writer, sheet_name="Full_Hazard", index=False)
        _rename_summary(hazard_summary_df).to_excel(
            writer, sheet_name="Full_Hazard_Summary", index=False)

        if early_isi_df is not None:
            lbl = early_hazard_label[:12]
            _rename_cluster_columns(early_isi_df, lm).to_excel(
                writer, sheet_name=f"Early_ISI ({lbl})"[:31], index=False)
        if early_hazard_df is not None:
            lbl = early_hazard_label[:12]
            _rename_cluster_columns(early_hazard_df, lm).to_excel(
                writer, sheet_name=f"Early_Hazard ({lbl})"[:31], index=False)
        if early_hazard_summary_df is not None:
            _rename_summary(early_hazard_summary_df).to_excel(
                writer, sheet_name="Early_Hazard_Summary", index=False)

        for epoch in epochs:
            safe = epoch["name"].replace(" ", "_")[:14]

            # Pre-drug (1 bin before onset)
            pre_isi = _rename_cluster_columns(epoch["pre_isi_df"], lm)
            pre_isi.to_excel(
                writer, sheet_name=f"{safe}_PreDrug_ISI"[:31], index=False)
            _rename_cluster_columns(epoch["pre_hazard_df"], lm).to_excel(
                writer, sheet_name=f"{safe}_PreDrug_Hazard"[:31], index=False)
            _rename_summary(epoch["pre_hazard_summary_df"]).to_excel(
                writer, sheet_name=f"{safe}_PreDrug_HazSumm"[:31], index=False)

            # End-of-drug (1 bin at end of drug application)
            if "end_isi_df" in epoch:
                _rename_cluster_columns(epoch["end_isi_df"], lm).to_excel(
                    writer, sheet_name=f"{safe}_EndDrug_ISI"[:31], index=False)
                _rename_cluster_columns(epoch["end_hazard_df"], lm).to_excel(
                    writer, sheet_name=f"{safe}_EndDrug_Hazard"[:31], index=False)
                _rename_summary(epoch["end_hazard_summary_df"]).to_excel(
                    writer, sheet_name=f"{safe}_EndDrug_HazSumm"[:31], index=False)
