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

    return pd.DataFrame(rows)


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
      2. Delta_from_Baseline (if used)
      3. Baseline_Stats (if used)
      4. CCK_Cell_Typing (if CCK used)
      5. PE_Cell_Typing (if PE used)
      6. Peri_<DrugName> (one per drug event with a peri window)
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

    raw_data, delta_data = calculate_firing_rate(
        filtered_export,
        bin_size,
        start_time,
        end_time,
        baseline_fr_dict if export_delta_from_baseline else None,
    )
    df_raw, df_delta = create_firing_rate_dataframes(raw_data, delta_data)

    if df_raw.empty:
        return export_dir, images_dir, None

    label_map = _cluster_label_map(cck_df, pe_df)
    peri_warnings: list[str] = []

    xlsx_path = export_dir / "firing_rates_by_cluster.xlsx"
    with ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:

        # Summary is deferred so peri-drug warnings can be collected first

        if export_delta_from_baseline and df_delta is not None and len(df_delta.columns) > 1:
            df_delta_renamed = _rename_cluster_columns(df_delta, label_map)
            df_delta_renamed.to_excel(
                writer, sheet_name="Delta_from_Baseline", index=False)

        if export_baseline_stats and baseline_start is not None and baseline_end is not None:
            baselined_df = create_baselined_df(
                baseline_start, baseline_end, bin_size, filtered_export)
            baseline_sheet = f"Baseline_Stats ({baseline_start:.0f}s-{baseline_end:.0f}s)"
            baselined_df.to_excel(
                writer, sheet_name=baseline_sheet, index=False)

        if cck_df is not None and not cck_df.empty:
            _trim_cell_typing_df(cck_df).to_excel(
                writer, sheet_name="CCK_Cell_Typing", index=False)

        if pe_df is not None and not pe_df.empty:
            _trim_cell_typing_df(pe_df).to_excel(
                writer, sheet_name="PE_Cell_Typing", index=False)

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

            safe_name = ev["name"].replace(" ", "_")[:24]
            peri_df.to_excel(writer, sheet_name=f"Peri_{safe_name}"[
                             :31], index=False)

            # Peri mean-by-label sheet (if requested)
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
                        {"Time Intervals (s)": peri_raw_df["Time Intervals (s)"]})
                    for lbl, cids in sorted(lbl_cid_groups.items()):
                        cols = [
                            f"Cluster_{cid}" for cid in cids if f"Cluster_{cid}" in peri_raw_df.columns]
                        if cols:
                            mean_peri[f"Mean_{lbl}_Hz"] = peri_raw_df[cols].mean(
                                axis=1)
                    mean_peri.to_excel(writer, sheet_name=f"MeanPeri_{safe_name}"[
                                       :31], index=False)

        # Summary written last so warnings are included, then moved to sheet position 0
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
            key=lambda ws: 0 if ws.name == "Summary" else 1)

        if cluster_group_map:
            label_groups: dict[str, list[str]] = {}
            for cid, lbl in cluster_group_map.items():
                col = f"Cluster_{cid}"
                if col in df_raw.columns:
                    label_groups.setdefault(lbl, []).append(col)
            if label_groups:
                mean_df = df_raw[["Time Intervals (s)"]].copy()
                for lbl, cols in sorted(label_groups.items()):
                    mean_df[f"Mean_{lbl}_Hz"] = df_raw[cols].mean(axis=1)
                mean_df.to_excel(
                    writer, sheet_name="Mean_by_Label", index=False)

        # Binned_Firing_Rates is always the last sheet
        df_raw_renamed = _rename_cluster_columns(df_raw, label_map)
        df_raw_renamed.to_excel(
            writer, sheet_name="Binned_Firing_Rates", index=False)

    return export_dir, images_dir, df_raw


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
) -> None:
    """
    Write ISI and hazard data to Excel.

    Sheets written:
      Full_ISI / Full_Hazard / Full_Hazard_Summary  — whole recording
      Early_ISI (<label>) / Early_Hazard / Early_Hazard_Summary  — early window if provided
      Peri_<Drug>_ISI / Peri_<Drug>_Hazard  — per-drug pre+post combined, if provided
    """
    excel_path = export_dir / "isi_and_hazard_analysis.xlsx"
    with ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        isi_df.to_excel(writer, sheet_name="Full_ISI", index=False)
        hazard_df.to_excel(writer, sheet_name="Full_Hazard", index=False)
        hazard_summary_df.to_excel(
            writer, sheet_name="Full_Hazard_Summary", index=False)

        if early_isi_df is not None:
            lbl = early_hazard_label[:12]
            early_isi_df.to_excel(
                writer, sheet_name=f"Early_ISI ({lbl})"[:31], index=False)
        if early_hazard_df is not None:
            lbl = early_hazard_label[:12]
            early_hazard_df.to_excel(
                writer, sheet_name=f"Early_Hazard ({lbl})"[:31], index=False)
        if early_hazard_summary_df is not None:
            early_hazard_summary_df.to_excel(
                writer, sheet_name="Early_Hazard_Summary", index=False)

        for epoch in (peri_epochs or []):
            safe = epoch["name"].replace(" ", "_")[:16]

            pre_isi: pd.DataFrame = epoch["pre_isi_df"]
            post_isi: pd.DataFrame = epoch["post_isi_df"]
            combined_isi = pre_isi.join(post_isi.drop("Bin_Starts", axis=1))
            combined_isi.to_excel(
                writer, sheet_name=f"Peri_{safe}_ISI"[:31], index=False)

            pre_haz: pd.DataFrame = epoch["pre_hazard_df"]
            post_haz: pd.DataFrame = epoch["post_hazard_df"]
            combined_haz = pre_haz.join(post_haz.drop("Bin_Starts", axis=1))
            combined_haz.to_excel(
                writer, sheet_name=f"Peri_{safe}_Hazard"[:31], index=False)
