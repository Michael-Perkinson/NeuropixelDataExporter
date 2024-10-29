import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def file_chooser():
    """Prompts user to select a folder within the 'data' directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
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
    """Reads a .npy file from the specified file path."""
    if file_path:
        data = np.load(file_path)
        print(f"Loaded file: {file_path} with shape {data.shape}")
        return data
    else:
        print("No file selected.")
        return None


def load_selected_channels(spike_times_path, spike_clusters_path, channels):
    """Loads spike times and clusters and filters by selected channels."""
    spike_times = read_npy_file(spike_times_path)
    spike_clusters = read_npy_file(spike_clusters_path)

    selected_spikes = {}
    for channel in channels:
        if channel.strip():  # Ensures channel is not an empty string
            try:
                channel = int(channel)
                spikes = spike_times[spike_clusters == channel]
                if len(spikes) > 0:
                    selected_spikes[channel] = spikes
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
    """Prompts user for channels to export."""
    channels_input = input(
        "Enter the clusters you want to export (separated by commas): ")
    channels = list(set(channel.strip()
                    for channel in channels_input.split(',')))
    print(f"Selected channels: {channels}")
    return channels


def drug_application_time():
    """Prompts user to enter the drug application time or skip if not given."""
    try:
        drug_time = input(
            "Enter the time at which the drug was applied (s), or press Enter to skip: ")
        return float(drug_time) if drug_time else None
    except ValueError:
        print("Invalid input. Ignoring drug application time.")
        return None


def start_and_end_time(max_time):
    """Prompts user to enter the start and end times for the plot or defaults to full range if not given."""
    try:
        start_time = input(
            "Enter the start time of the plot (s), or press Enter to start from 0: ")
        end_time = input(
            f"Enter the end time of the plot (s), or press Enter to go to {max_time:.2f} s: ")
        start_time = float(start_time) if start_time else 0.0
        end_time = float(end_time) if end_time else max_time
        print(f"Start time: {start_time}s, End time: {end_time}s")
        return start_time, end_time
    except ValueError:
        print("Invalid input. Using default time range.")
        return 0.0, max_time


def extract_data(selected_spikes, drug_time, start_time, end_time, sample_rate):
    """Filters spike times by start and end times and adjusts relative to drug application."""
    data_export = {}
    for channel, spikes in selected_spikes.items():

        spikes_in_seconds = spikes / sample_rate

        filtered_spikes = spikes_in_seconds[(
            spikes_in_seconds >= start_time) & (spikes_in_seconds <= end_time)]

        if drug_time is not None:
            relative_spikes_ms = (filtered_spikes - drug_time) * 1000
            relative_spikes_ms = relative_spikes_ms[relative_spikes_ms >= 0]
        else:
            relative_spikes_ms = filtered_spikes * 1000

        data_export[channel] = relative_spikes_ms
        print(
            f"Channel {channel}: Original spikes count = {len(spikes)}, Filtered count = {len(filtered_spikes)}, Export count = {len(relative_spikes_ms)}")

    return data_export


def calculate_firing_rate(data_export, bin_size, max_time):
    """Calculates firing rates for each cluster in specified bin size (in seconds), with options for handling incomplete bins."""
    bins = np.arange(0, max_time + bin_size, bin_size)
    bin_times = bins[:-1]

    if bins[-1] > max_time:
        bins = bins[:-1]  # Exclude last bin if it's incomplete
        bin_times = bin_times[:-1]  # Also adjust bin_times to match
   
    firing_rates = {'Bin_Time_s': bin_times}

    for channel, spikes in data_export.items():
        spikes_in_seconds = spikes / 1000
        counts, _ = np.histogram(spikes_in_seconds, bins=bins)
        firing_rate = counts / bin_size
        firing_rates[channel] = firing_rate
        print(
            f"Channel {channel}: Firing rate calculated with {len(firing_rate)} bins")

    return pd.DataFrame(firing_rates)


def export_firing_rate_html(firing_rate_df, images_export_dir, bin_size, drug_time=None):
    """Creates and exports interactive HTML plots for each cluster's firing rate with bin size in title and file name."""
    bin_times = firing_rate_df['Bin_Time_s']
    for channel in firing_rate_df.columns[1:]:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_times,
            y=firing_rate_df[channel],
            name=f"Cluster {channel}",
            marker_color='black',
            marker_line_width=0
        ))
        # Add red dashed line at drug application time if provided
        if drug_time is not None:
            fig.add_shape(
                type="line",
                x0=drug_time,
                y0=0,
                x1=drug_time,
                y1=max(firing_rate_df[channel]) * 1.1,
                line=dict(color="red", width=2, dash="dash"),
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
            xaxis=dict(
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                showline=True,
                linecolor='black',
                showgrid=False,
                ticks='outside',
                tickwidth=2,
                tickcolor='black',
                ticklen=6
            )
        )

        html_path = os.path.join(
            images_export_dir, f"Firing_Rate_Cluster_{channel}_BinSize_{bin_size}s.html")
        fig.write_html(html_path)
        print(
            f"Interactive firing rate HTML for Channel {channel} saved to {html_path}")


def export_data(data_export, folder_path, bin_size, max_time, drug_time):
    """Exports the filtered spike times and firing rates to structured CSV files."""
    base_folder_name = os.path.basename(folder_path)
    analysis_folder_name = f"{base_folder_name}_analysed"
    export_dir = os.path.join(
        folder_path, 'exported_data', analysis_folder_name)
    os.makedirs(export_dir, exist_ok=True)

    # Filter out channels with no spikes
    data_export = {channel: times for channel,
                   times in data_export.items() if len(times) > 0}

    # Find the maximum length of spike times to pad columns for equal length
    max_length = max(len(times)
                     for times in data_export.values()) if data_export else 0

    # Create DataFrame where each cluster is a column if data exists
    if max_length > 0:
        export_df = pd.DataFrame({channel: np.pad(times, (0, max_length - len(times)), 'constant', constant_values=np.nan)
                                  for channel, times in data_export.items()})
        csv_path = os.path.join(
            export_dir, "spike_times_by_cluster_time_ms.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Spike times exported to {csv_path}")

        firing_rate_df = calculate_firing_rate(data_export, bin_size, max_time)
        firing_rate_csv_path = os.path.join(
            export_dir, "firing_rates_by_cluster.csv")
        firing_rate_df.to_csv(firing_rate_csv_path, index=False)
        print(f"Firing rates exported to {firing_rate_csv_path}")

        images_export_dir = os.path.join(export_dir, 'firing_rate_images')
        os.makedirs(images_export_dir, exist_ok=True)
        export_firing_rate_html(
            firing_rate_df, images_export_dir, bin_size, drug_time)


def main():
    # Step 1: Choose folder and load files
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

    # Step 2: Get user input for drug application time, start, end time, and bin size
    drug_time = drug_application_time() or 0
    start_time, end_time = start_and_end_time(max_time)
    bin_size_input = input(
        "Enter the bin size for firing rate calculation (s), or press Enter for 1s: ")
    # Default to 1 second if not provided
    bin_size = float(bin_size_input) if bin_size_input else 1.0
    print(f"Bin size for firing rate: {bin_size}s")

    # Step 3: Load and filter data for selected clusters
    data_export = extract_data(
        selected_spikes, drug_time, start_time, end_time, sample_rate)

    # Step 4: Export data including firing rates
    export_data(data_export, folder_path, bin_size, max_time, drug_time)


if __name__ == "__main__":
    main()
