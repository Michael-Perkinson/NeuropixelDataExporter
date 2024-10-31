import re
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import openpyxl


def file_chooser():
    """
    Prompts the user to select a folder within the current script's directory.

    Returns:
        str or None: The path to the selected folder, or None if no folder is selected.
    """
    root = tk.Tk()
    root.withdraw()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = filedialog.askdirectory(
        initialdir=script_dir,
        title="Select a folder containing the required files"
    )

    if not folder_path:
        print("No folder selected.")
        return None

    print(f"Selected folder: {folder_path}")
    return folder_path


def read_npy_file(file_path):
    """
    Reads a .npy file from the specified file path.

    Parameters:
        file_path (str): Path to the .npy file.

    Returns:
        np.ndarray or None: The loaded data array, or None if no file path is provided.
    """
    if file_path:
        data = np.load(file_path)
        print(f"Loaded file: {file_path} with shape {data.shape}")
        return data
    print("No file selected.")
    return None


def load_selected_channels(spike_times_path, spike_clusters_path, channels):
    """
    Loads spike times and clusters, filtering by selected channels with spikes.

    Parameters:
        spike_times_path (str): Path to the .npy file containing spike times.
        spike_clusters_path (str): Path to the .npy file containing spike cluster IDs.
        channels (list): List of channels to filter.

    Returns:
        dict: A dictionary with channels as keys and spike times as values.
        int: Sample rate of the recording (Hz).
    """
    spike_times = read_npy_file(spike_times_path)
    spike_clusters = read_npy_file(spike_clusters_path)

    selected_spikes = {}
    channels_with_spikes = []

    for channel in channels:
        try:
            channel = int(channel)
            spikes = spike_times[spike_clusters == channel]
            if len(spikes) > 0:
                selected_spikes[channel] = spikes
                channels_with_spikes.append(channel)
                print(f"Channel {channel}: Loaded {len(spikes)} spikes")
            else:
                print(f"Channel {channel} has no spikes and will be skipped.")
        except ValueError:
            print(f"Invalid channel: {channel}. Skipping.")

    sample_rate = 30000  # Neuropixels sample rate (Hz)
    max_time = np.max(spike_times) / sample_rate if len(spike_times) > 0 else 0
    print(f"Max spike time in seconds: {max_time:.2f}s")
    return selected_spikes, sample_rate


def channels_to_export():
    """
    Prompts user for channels to export, retaining only numeric identifiers.

    Returns:
        list: Sorted list of unique numeric channel identifiers.
    """
    channels_input = input(
        "Enter the clusters you want to export (separated by commas): ")
    channels_cleaned = re.findall(r'\d+', channels_input)
    channels = sorted(set(int(channel) for channel in channels_cleaned))
    print(f"Selected channels: {channels}")
    return channels


def drug_application_time():
    """
    Prompts the user to enter the drug application time in seconds or skip if not provided.

    Returns:
        float or None: The entered drug application time in seconds, or None if skipped or invalid.
    """
    try:
        drug_time = input(
            "Enter the time at which the drug was applied (s), or press Enter to skip: ")
        return float(drug_time) if drug_time else None
    except ValueError:
        print("Invalid input. Ignoring drug application time.")
        return None


def start_and_end_time(max_time):
    """
    Prompts the user to enter start and end times for a plot, defaulting to the full range if omitted.

    Parameters:
        max_time (float): The maximum allowable time for the end of the plot.

    Returns:
        tuple: A tuple containing the start and end times in seconds.
    """
    try:
        start_time = input(
            "Enter the start time of the plot (s), or press Enter to start from 0: ")
        end_time = input(f"""Enter the end time of the plot (s),
            or press Enter to go to {max_time:.2f} s: """)
        start_time = float(start_time) if start_time else 0.0
        end_time = float(end_time) if end_time else max_time
        print(f"Start time: {start_time}s, End time: {end_time}s")
        return start_time, end_time
    except ValueError:
        print("Invalid input. Using default time range.")
        return 0.0, max_time


def extract_data(selected_spikes, drug_time, start_time, end_time, sample_rate):
    """
    Filters spike times by specified start and end times, adjusting relative to drug application if provided.

    Parameters:
        selected_spikes (dict): Dictionary with channels as keys and spike times as values.
        drug_time (float or None): The time of drug application in seconds, or None if not applicable.
        start_time (float): Start time for filtering in seconds.
        end_time (float): End time for filtering in seconds.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        dict: Dictionary with channels as keys and filtered, adjusted spike times in milliseconds as values.
    """
    data_export = {}
    for channel, spikes in selected_spikes.items():
        spikes_in_seconds = spikes / sample_rate
        filtered_spikes = spikes_in_seconds[(
            spikes_in_seconds >= start_time) & (spikes_in_seconds <= end_time)]

        if drug_time is not None:
            relative_spikes_ms = (filtered_spikes - drug_time) * 1000
            relative_spikes_ms = relative_spikes_ms[relative_spikes_ms >= 0]
            final_count = len(relative_spikes_ms)
        else:
            relative_spikes_ms = filtered_spikes * 1000
            final_count = len(relative_spikes_ms)

        data_export[channel] = relative_spikes_ms
        print(f"Channel {channel}: Total spikes = {len(spikes)}, In time range = {
              len(filtered_spikes)}, Exported = {final_count}")

    return data_export


def calculate_firing_rate(data_export, bin_size, max_time):
    """
    Calculates firing rates for each cluster using specified bin size.

    Parameters:
        data_export (dict): Dictionary with channels as keys and spike times in milliseconds as values.
        bin_size (float): Bin size for firing rate calculation, in seconds.
        max_time (float): Maximum time for binning, in seconds.

    Returns:
        pd.DataFrame: DataFrame with bin times and firing rates for each channel.
    """
    bins = np.arange(0, max_time + bin_size, bin_size)
    if bins[-1] > max_time:
        bins = bins[:-1]  # Adjust to remove incomplete bin

    bin_times = bins[:-1]
    firing_rates = {'Bin_Time_s': bin_times}

    for channel, spikes in data_export.items():
        spikes_in_seconds = spikes / 1000
        counts, _ = np.histogram(spikes_in_seconds, bins=bins)
        firing_rate = counts / bin_size
        firing_rates[channel] = firing_rate

    return pd.DataFrame(firing_rates)


def calculate_isi_histogram(data_export, time_bin=0.01, max_isi_time=0.75):
    """
    Calculates interspike interval (ISI) histograms for each channel.

    Parameters:
        data_export (dict): Dictionary with channels as keys and spike times in milliseconds as values.
        time_bin (float): Bin width for the ISI histogram, in seconds.
        max_isi_time (float): Maximum ISI time to consider for histogram, in seconds.

    Returns:
        pd.DataFrame: DataFrame with bin start times as the first column and ISI counts for each channel.
    """
    n_bins = int(max_isi_time / time_bin)
    bin_edges = np.arange(0, (n_bins + 1) * time_bin, time_bin)
    isi_data = {"Bin_Starts": bin_edges[:-1]}

    for channel, spikes in data_export.items():
        spikes_in_seconds = spikes / 1000  # Convert ms to seconds
        isis = np.diff(spikes_in_seconds)  # Calculate ISI
        isi_histogram, _ = np.histogram(isis, bins=bin_edges)
        isi_data[channel] = isi_histogram

    return pd.DataFrame(isi_data)


def calculate_hazard_function(isi_df, early_time=0.07, late_time_start=0.41, late_time_end=0.5):
    """
    Calculates hazard function values for each channel and key hazard metrics.

    Parameters:
        isi_df (pd.DataFrame): ISI histogram counts with bin start times as the first column.
        early_time (float): Threshold for early hazard calculation (default 0.07 s).
        late_time_start (float): Start of late interval (default 0.41 s).
        late_time_end (float): End of late interval (default 0.5 s).

    Returns:
        pd.DataFrame: Hazard function values for each channel.
        pd.DataFrame: Key hazard metrics (peak early hazard, mean late hazard, hazard ratio) for each channel.
    """
    bin_starts = isi_df['Bin_Starts']
    hazard_data = {'Bin_Starts': bin_starts}
    hazard_summary_data = {'Cluster': [], 'Peak Early Hazard': [
    ], 'Mean Late Hazard': [], 'Hazard Ratio': []}

    for channel in isi_df.columns[1:]:
        isi_counts = isi_df[channel]
        total_spikes = np.sum(isi_counts)

        hazard_values = [
            isi_counts[i] / (total_spikes -
                             np.cumsum(isi_counts)[i] + isi_counts[i])
            if np.cumsum(isi_counts)[i] < total_spikes else 0
            for i in range(len(bin_starts))
        ]
        hazard_data[channel] = hazard_values

        peak_early_hazard = max(hazard_values[i] for i in range(
            len(bin_starts)) if bin_starts[i] <= early_time)
        mean_late_hazard = np.mean([hazard_values[i] for i in range(len(bin_starts))
                                    if late_time_start <= bin_starts[i] <= late_time_end])
        hazard_ratio = peak_early_hazard / \
            mean_late_hazard if mean_late_hazard != 0 else np.nan

        hazard_summary_data['Cluster'].append(channel)
        hazard_summary_data['Peak Early Hazard'].append(peak_early_hazard)
        hazard_summary_data['Mean Late Hazard'].append(mean_late_hazard)
        hazard_summary_data['Hazard Ratio'].append(hazard_ratio)

    hazard_df = pd.DataFrame(hazard_data)
    hazard_summary_df = pd.DataFrame(hazard_summary_data)

    return hazard_df, hazard_summary_df


def export_firing_rate_html(firing_rate_df, images_export_dir, bin_size, drug_time=None):
    """
    Creates and exports interactive HTML plots for each cluster's firing rate with bin size in title and file name.

    Parameters:
        firing_rate_df (pd.DataFrame): DataFrame containing bin times and firing rates for each channel.
        images_export_dir (str): Directory to save the HTML files.
        bin_size (float): Bin size for firing rate in seconds.
        drug_time (float or None): Time of drug application in seconds, if applicable.
    """
    bin_times = firing_rate_df['Bin_Time_s']
    for channel in firing_rate_df.columns[1:]:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_times,
            y=firing_rate_df[channel],
            name=f"Cluster {channel}",
            marker_color='black'
        ))

        if drug_time is not None:
            fig.add_shape(
                type="line",
                x0=drug_time,
                y0=0,
                x1=drug_time,
                y1=max(firing_rate_df[channel]) * 1.1,
                line=dict(color="red", width=2, dash="dash")
            )
            fig.add_annotation(
                x=drug_time,
                y=max(firing_rate_df[channel]) * 1.1,
                text="Drug Application",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20
            )

        fig.update_layout(
            title=f"Firing Rate Histogram for Cluster {
                channel} (Bin Size: {bin_size}s)",
            xaxis_title="Time (s)",
            yaxis_title="Firing Rate (Hz)",
            plot_bgcolor='white',
            bargap=0,
            xaxis=dict(showline=True, linecolor='black', showgrid=False),
            yaxis=dict(showline=True, linecolor='black',
                       showgrid=False, ticks='outside')
        )

        html_path = os.path.join(images_export_dir, f"Firing_Rate_Cluster_{
                                 channel}_BinSize_{bin_size}s.html")
        fig.write_html(html_path)
    print(f"Interactive firing rate HTMLs for Channels saved to {html_path}")


def export_data(data_export, folder_path, bin_size, max_time, drug_time):
    """
    Exports filtered spike times and firing rates to structured CSV files, individual text files for each cluster, 
    and HTML plots.

    Parameters:
        data_export (dict): Dictionary of spike times per channel.
        folder_path (str): Base folder path for exporting data.
        bin_size (float): Bin size for firing rate calculation in seconds.
        max_time (float): Maximum time for firing rate calculation.
        drug_time (float or None): Time of drug application in seconds, if applicable.

    Returns:
        str: Directory path where data is exported.
    """
    analysis_folder_name = f"{os.path.basename(folder_path)}_analysed"
    export_dir = os.path.join(folder_path, analysis_folder_name)
    os.makedirs(export_dir, exist_ok=True)

    # Filter out channels with no spikes
    data_export = {channel: times for channel,
                   times in data_export.items() if len(times) > 0}

    if data_export:
        # Create overall CSV with padded spike times for all clusters
        max_length = max(len(times) for times in data_export.values())
        export_df = pd.DataFrame({
            channel: np.pad(times, (0, max_length - len(times)),
                            'constant', constant_values=np.nan)
            for channel, times in data_export.items()
        })
        export_df.to_csv(os.path.join(
            export_dir, "spike_times_by_cluster_time_ms.csv"), index=False)
        print(f"Spike times exported to {export_dir}")

        # Create folder for individual text files
        txt_export_dir = os.path.join(
            export_dir, 'txt_files_for_clampfit_import')
        os.makedirs(txt_export_dir, exist_ok=True)

        # Export each cluster's spike times as a separate text file
        for channel, times in data_export.items():
            txt_file_path = os.path.join(
                txt_export_dir, f"spike_times_cluster_{channel}_time_ms.txt")
            np.savetxt(txt_file_path, times, fmt='%0.4f')
        print(f"Individual text files for each cluster saved to {
              txt_export_dir}")

        # Export firing rate data
        firing_rate_df = calculate_firing_rate(data_export, bin_size, max_time)
        firing_rate_df.to_csv(os.path.join(
            export_dir, "firing_rates_by_cluster.csv"), index=False)
        print(f"Firing rates exported to {export_dir}")

        # Generate and export interactive HTML plots
        images_export_dir = os.path.join(export_dir, 'firing_rate_images')
        os.makedirs(images_export_dir, exist_ok=True)
        export_firing_rate_html(
            firing_rate_df, images_export_dir, bin_size, drug_time)

    return export_dir


def export_hazard_excel(folder_path, export_dir, hazard_df, hazard_summary_df, isi_df):
    """
    Exports hazard function data, summary metrics, and ISI data to a single Excel file.

    Parameters:
        folder_path (str): Path where the Excel file will be saved.
        export_dir (str): Directory for exporting the Excel file.
        hazard_df (pd.DataFrame): Hazard function values for each cluster.
        hazard_summary_df (pd.DataFrame): Summary metrics for each cluster.
        isi_df (pd.DataFrame): Raw ISI histogram data for each cluster.
    """
    excel_path = os.path.join(export_dir, "ISI_Hazard_Analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        hazard_summary_df.to_excel(
            writer, sheet_name='Summary Data', index=False)
        hazard_df.to_excel(writer, sheet_name='Hazard Function', index=False)
        isi_df.to_excel(writer, sheet_name='ISI Data', index=False)

    print("ISI and Hazard data exported to Excel file:", excel_path)


def main():
    # Choose folder and load files
    folder_path = file_chooser()
    if not folder_path:
        return

    spike_times_path = os.path.join(folder_path, 'spike_times.npy')
    spike_clusters_path = os.path.join(folder_path, 'spike_clusters.npy')

    # Load spike times and determine max_time
    channels = channels_to_export()
    selected_spikes, sample_rate = load_selected_channels(
        spike_times_path, spike_clusters_path, channels)
    max_time = np.max([np.max(spikes) for spikes in selected_spikes.values() if len(
        spikes) > 0]) / sample_rate

    # Get user input for drug application time, start/end time, and bin size
    drug_time = drug_application_time() or 0
    start_time, end_time = start_and_end_time(max_time)
    bin_size = float(input(
        "Enter the bin size for firing rate calculation (s), or press Enter for 1s: ") or 1.0)

    # Load and filter data for selected clusters
    data_export = extract_data(
        selected_spikes, drug_time, start_time, end_time, sample_rate)

    # Export spike times and firing rates
    export_dir = export_data(data_export, folder_path,
                             bin_size, max_time, drug_time)

    # ISI and hazard calculations
    isi_df = calculate_isi_histogram(data_export)
    hazard_df, hazard_summary_df = calculate_hazard_function(isi_df)

    # Export ISI and hazard data to Excel
    export_hazard_excel(folder_path, export_dir, hazard_df,
                        hazard_summary_df, isi_df)


if __name__ == "__main__":
    main()
