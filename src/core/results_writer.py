from pathlib import Path
import pandas as pd
from pandas import ExcelWriter
import numpy as np

from src.core.firing_rate import (
    calculate_firing_rate,
    create_firing_rate_dataframes,
    create_baselined_df,
)
from src.core.file_manager import make_output_folders


def create_export_dir(folder_path: Path, analysis_folder_name: str) -> Path:
    """
    Create an export directory for analysis results.

    Args
        folder_path: Base directory where the analysis folder will be created.
        analysis_folder_name: Name of the subdirectory for exported files.

    Returns
        Path object to the export directory.
    """
    export_dir = folder_path / analysis_folder_name
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def export_spike_times(
    data_export: dict[int, np.ndarray], csv_path: Path, txt_export_dir: Path
) -> None:
    """
    Export spike times for each cluster to a CSV file and individual text files.

    - Saves a CSV file named "spike_times_by_cluster_time_ms.csv" containing one column per cluster.
    - Saves individual text files in a subdirectory "txt_files_for_clampfit_import".

    Args
        data_export: Dictionary mapping cluster IDs to arrays of spike times (ms).
        csv_path: Path to the CSV file where spike times will be saved.
        txt_export_dir: Directory where individual text files will be saved.
    """
    max_length = max(len(arr) for arr in data_export.values())

    df_spikes = pd.DataFrame(
        {
            f"Cluster_{cid}": np.pad(
                arr.astype(float), (0, max_length - len(arr)), constant_values=np.nan
            )
            for cid, arr in data_export.items()
        }
    )

    df_spikes.to_csv(csv_path, index=False)
    print(f"Spike times exported to {csv_path}")

    for cid, arr in data_export.items():
        out_file = txt_export_dir / f"spike_times_Cluster_{cid}_time_ms.txt"
        np.savetxt(out_file, arr.astype(float), fmt="%.4f")

    print(f"Text files saved to {txt_export_dir}")


def export_data(
    data_export: dict[int, np.ndarray],
    baseline_fr_dict: dict[int, float | None],
    data_folder_path: Path,
    bin_size: float,
    start_time: float,
    end_time: float,
    baseline_start: float | None,
    baseline_end: float | None,
) -> tuple[Path, Path, pd.DataFrame | None]:
    """
    Export spike times and firing rate data.

    Baseline calculations are only performed if baseline_start and baseline_end are provided.

    Args
        data_export: Dictionary of processed spike times per cluster (ms).
        baseline_fr_dict: Dictionary of baseline firing rates per cluster.
        data_folder_path: Base folder for analysis.
        bin_size: Bin size (s) for firing rate calculations.
        start_time: Analysis start time (s).
        end_time: Analysis end time (s).
        baseline_start: Baseline start time (s) or None.
        baseline_end: Baseline end time (s) or None.

    Returns
        Export directories where files were saved.
        Raw firing rate DataFrame, or None if no spikes were available.
    """
    export_dir, images_dir, txt_export_dir = make_output_folders(data_folder_path)

    data_export = {cid: arr for cid, arr in data_export.items() if arr.size > 0}
    if not data_export:
        print("No spikes to export. Exiting.")
        return export_dir, images_dir, None

    export_spike_times(
        data_export, export_dir / "spike_times_by_cluster_time_ms.csv", txt_export_dir
    )

    raw_data, delta_data = calculate_firing_rate(
        data_export, bin_size, start_time, end_time, baseline_fr_dict
    )
    df_raw, df_delta = create_firing_rate_dataframes(raw_data, delta_data)

    xlsx_path = export_dir / "firing_rates_by_cluster.xlsx"

    with ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_raw.to_excel(writer, sheet_name="Firing_Rates_Raw", index=False)
        if df_delta is not None and len(df_delta.columns) > 1:
            df_delta.to_excel(writer, sheet_name="Delta_from_Baseline", index=False)

        if baseline_start is not None and baseline_end is not None:
            baselined_df = create_baselined_df(
                baseline_start, baseline_end, bin_size, data_export
            )
            baseline_sheet = (
                f"Baseline_Stats ({baseline_start:.2f}s-{baseline_end:.2f}s)"
            )
            baselined_df.to_excel(writer, sheet_name=baseline_sheet, index=False)

    print(f"Firing rates exported to {xlsx_path}")
    return export_dir, images_dir, df_raw


def export_hazard_excel(
    export_dir: Path,
    hazard_df: pd.DataFrame,
    hazard_summary_df: pd.DataFrame,
    isi_df: pd.DataFrame,
    baseline_isi_df: pd.DataFrame | None = None,
    baseline_hazard_df: pd.DataFrame | None = None,
    baseline_hazard_summary_df: pd.DataFrame | None = None,
) -> None:
    """
    Export hazard function and ISI analysis results to an Excel workbook.

    Args
        export_dir: Directory where the Excel file will be saved.
        hazard_df: DataFrame of hazard function values.
        hazard_summary_df: DataFrame of hazard summary metrics.
        isi_df: DataFrame of ISI histogram data.
        baseline_isi_df: Optional DataFrame of baseline ISI histogram data.
        baseline_hazard_df: Optional DataFrame of baseline hazard values.
        baseline_hazard_summary_df: Optional DataFrame of baseline hazard summary metrics.
    """
    excel_path = export_dir / "isi_and_hazard_analysis.xlsx"
    with ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        isi_df.to_excel(writer, sheet_name="ISI_Histogram", index=False)
        hazard_df.to_excel(writer, sheet_name="Hazard_Function", index=False)
        hazard_summary_df.to_excel(writer, sheet_name="Hazard_Summary", index=False)

        if baseline_isi_df is not None:
            baseline_isi_df.to_excel(
                writer, sheet_name="Baseline_ISI_Histogram", index=False
            )
        if baseline_hazard_df is not None:
            baseline_hazard_df.to_excel(
                writer, sheet_name="Baseline_Hazard_Function", index=False
            )
        if baseline_hazard_summary_df is not None:
            baseline_hazard_summary_df.to_excel(
                writer, sheet_name="Baseline_Hazard_Summary", index=False
            )

    print(f"ISI and hazard data exported to {excel_path}")
