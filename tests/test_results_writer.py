import numpy as np
import pandas as pd
import pytest

from src.core.results_writer import (
    export_data,
    export_hazard_excel,
)


# --- Test export_hazard_excel --- #

def test_export_hazard_excel(tmp_path):
    bin_starts = np.array([0.0, 0.1, 0.2])

    isi_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([5, 3, 2])})
    hazard_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([0.5, 0.3, 0.2])})
    hazard_summary_df = pd.DataFrame({
        "Cluster": ["Cluster_0"],
        "Peak Early Hazard": [0.5],
        "Mean Late Hazard": [0.2],
        "Hazard Ratio": [2.5],
    })

    export_dir = tmp_path / "export_hazard"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_hazard_excel(export_dir, hazard_df, hazard_summary_df, isi_df)

    assert (export_dir / "isi_and_hazard_analysis.xlsx").is_file()


def test_export_hazard_excel_with_early_window(tmp_path):
    bin_starts = np.array([0.0, 0.1, 0.2])

    isi_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([5, 3, 2])})
    hazard_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([0.5, 0.3, 0.2])})
    hazard_summary_df = pd.DataFrame({
        "Cluster": ["Cluster_0"],
        "Peak Early Hazard": [0.5],
        "Mean Late Hazard": [0.2],
        "Hazard Ratio": [2.5],
    })
    early_isi_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([2, 1, 1])})
    early_hazard_df = pd.DataFrame({"Bin_Starts": bin_starts, "Cluster_0": np.array([0.3, 0.2, 0.1])})
    early_hazard_summary_df = pd.DataFrame({
        "Cluster": ["Cluster_0"],
        "Peak Early Hazard": [0.3],
        "Mean Late Hazard": [0.1],
        "Hazard Ratio": [3.0],
    })

    export_dir = tmp_path / "export_hazard_early"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_hazard_excel(
        export_dir, hazard_df, hazard_summary_df, isi_df,
        early_isi_df=early_isi_df,
        early_hazard_df=early_hazard_df,
        early_hazard_summary_df=early_hazard_summary_df,
        early_hazard_label="0–600s",
    )

    excel_path = export_dir / "isi_and_hazard_analysis.xlsx"
    assert excel_path.is_file()
    with pd.ExcelFile(excel_path) as xls:
        assert any("Early_ISI" in s for s in xls.sheet_names)
        assert any("Early_Hazard" in s for s in xls.sheet_names)


# --- Test export_data --- #

def test_export_data_creates_files(tmp_path):
    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict={0: 5.0},
        data_folder_path=tmp_path,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    analysis_dir = tmp_path / "analysis_results"
    assert (analysis_dir / "firing_rates_by_cluster.xlsx").exists()
    assert (analysis_dir / "spike_times_by_cluster_time_ms.csv").exists()
    assert (analysis_dir / "firing_rate_images").exists()


def test_export_data_xlsx_sheet_name(tmp_path):
    """Firing rates sheet is named Binned_Firing_Rates, not Firing_Rates_Raw."""
    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict={0: 5.0},
        data_folder_path=tmp_path,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=0.0,
        baseline_end=0.5,
    )

    xlsx_path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    assert xlsx_path.exists()
    with pd.ExcelFile(xlsx_path) as xls:
        assert "Binned_Firing_Rates" in xls.sheet_names
        assert "Firing_Rates_Raw" not in xls.sheet_names
        df = pd.read_excel(xls, sheet_name="Binned_Firing_Rates")
        assert not df.empty


def test_export_data_no_spikes(tmp_path):
    export_dir, images_dir, df_raw = export_data(
        data_export={},
        baseline_fr_dict={},
        data_folder_path=tmp_path,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    assert export_dir.exists()
    assert images_dir.exists()
    assert df_raw is None


def test_export_data_summary_sheet_first(tmp_path):
    """Summary should be the first sheet in the output xlsx."""
    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict=None,
        data_folder_path=tmp_path,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    xlsx_path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    with pd.ExcelFile(xlsx_path) as xls:
        assert xls.sheet_names[0] == "Summary"


def test_export_data_binned_firing_rates_last(tmp_path):
    """Binned_Firing_Rates should be the last sheet in the output xlsx."""
    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict=None,
        data_folder_path=tmp_path,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    xlsx_path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    with pd.ExcelFile(xlsx_path) as xls:
        assert xls.sheet_names[-1] == "Binned_Firing_Rates"
