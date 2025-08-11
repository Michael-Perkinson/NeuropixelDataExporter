from pathlib import Path

from file_manager import validate_folder, REQUIRED_FILES
from spike_filter import prepare_filtered_data
from firing_rate import process_cluster_data
from isi_hazard import calculate_isi_histogram, calculate_hazard_function
from results_writer import export_data, export_hazard_excel
from interactive_plot import export_firing_rate_html
from terminal_prompts import (
    drug_application_time,
    start_and_end_time,
    prompt_for_baseline,
)


def run_analysis(folder_path: Path):
    """
    Run the full spike analysis pipeline, including data validation, spike filtering,
    firing rate calculation, ISI histogram and hazard function analysis, and exporting results.

    Steps
        1. Validate the input folder and required files.
        2. Load and filter spike data.
        3. Prompt for user-defined parameters (drug time, analysis window, baseline).
        4. Compute firing rates and baseline values.
        5. Generate ISI histograms and hazard functions.
        6. Export results and interactive plots.

    Args
        folder_path: Path to the directory containing spike data files.
    """
    print(f"Starting analysis on folder: {folder_path}")

    if not folder_path:
        print("No folder provided. Exiting analysis.")
        return

    # Validate required files
    try:
        file_paths = validate_folder(folder_path, REQUIRED_FILES)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    data_folder_path = Path(file_paths["spike_times.npy"]).parent
    print(f"Data folder path: {data_folder_path}")

    # Load and filter spike data
    recording_dataframe, max_time = prepare_filtered_data(file_paths)

    # Collect user-defined parameters
    drug_time = drug_application_time()
    start_time, end_time = start_and_end_time(max_time)

    bin_size = float(
        input(
            "Enter bin size for firing rate calculation (s), or press Enter for 1.0: "
        )
        or 1.0
    )
    baseline_start, baseline_end = prompt_for_baseline(end_time, start_time)

    # Compute firing rates
    raw_fr_dict, baseline_fr_dict = process_cluster_data(
        recording_dataframe,
        drug_time,
        start_time,
        end_time,
        sample_rate=30000,  # Fixed sample rate
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )

    # Export spike times and firing rates
    export_dir, images_dir, firing_rate_df = export_data(
        raw_fr_dict,
        baseline_fr_dict,
        data_folder_path,
        bin_size,
        start_time,
        end_time,
        baseline_start,
        baseline_end,
    )

    # Compute ISI histogram and hazard functions
    isi_df, baseline_isi_df = calculate_isi_histogram(
        raw_fr_dict, baseline_start, baseline_end
    )

    hazard_df, hazard_summary_df, baseline_hazard_df, baseline_hazard_summary_df = (
        calculate_hazard_function(isi_df, baseline_isi_df)
    )

    # Export hazard function results
    export_hazard_excel(
        export_dir,
        hazard_df,
        hazard_summary_df,
        isi_df,
        baseline_isi_df,
        baseline_hazard_df,
        baseline_hazard_summary_df,
    )

    # Export interactive plots
    export_firing_rate_html(
        firing_rate_df,
        images_dir,
        bin_size,
        drug_time,
    )

    print("Analysis complete. All files have been exported.")
