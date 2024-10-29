
# Neuropixel Data Exporter

Neuropixel Data Exporter is a quick way to export Neuropixel spike data from Neuropixel recordings after Kilosort.
This script allows you to select specific clusters, filter spike times based on drug application time, and calculate firing rates in customizable time bins. Results are saved as CSV files, and interactive HTML plots of firing rates are generated for easy visualization of your recordings. Text files are exported per cluster to allow importing into Clampfit.

## Features

- **Cluster Selection**: Choose specific clusters to analyze.
- **Spike Time Filtering**: Filter spikes relative to drug application time.
- **Firing Rate Calculation**: Calculate firing rates for each cluster in user-defined time bins.
- **Data Export**: Save spike times and firing rates as structured CSV files.
- **Interactive Plots**: Generate HTML plots of firing rates for each cluster.

## Installation

1. **Clone the Repository**  
   First, clone this repository to your local machine:

    ```bash
   git clone https://github.com/yourusername/NeuropixelDataExporter.git
   cd NeuropixelDataExporter
   ```

2. **Create Virtual Environment**  
   It’s recommended to use a virtual environment to manage dependencies. Run the following command to create one, naming it `export_env` for easy identification:

   ```bash
   python -m venv export_env
   ```

3. **Activate the Virtual Environment**  
   - On Windows:

    ```bash
     export_env\Scripts\activate
    ```

   - On macOS and Linux:

    ```bash
    source export_env/bin/activate
    ```

4. **Install Dependencies**  
   Use `pip` to install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Activate the Environment**  
   If you have opened a new terminal since installation, run:

   ```bash
   export_env\Scripts\activate  # or source export_env/bin/activate on macOS/Linux
   ```

2. **Run the Script**  
   Start the analysis with:

   ```bash
   python export_clusters.py
   ```

3. **Select Data Folder**  
   When prompted, select the folder containing the following files:
   - `spike_times.npy`: Array of spike times in samples.
   - `spike_clusters.npy`: Array of cluster labels corresponding to each spike.

4. **Provide Input Parameters**  
   The script will prompt you to enter:
   - **Clusters to Export**: Enter the clusters you want to analyze (e.g., `5,7,12`).
   - **Drug Application Time**: Enter the time (in seconds) when the drug was applied, or press Enter to skip.
   - **Start and End Times**: Specify the time range for analysis, or press Enter to analyze the full range.
   - **Bin Size**: Enter the bin size in seconds for firing rate calculation, or press Enter to use the default of 1 second.

5. **Outputs**  
   A new folder will be created in the selected data directory:
   - **CSV Files**:
     - `spike_times_by_cluster_time_ms.csv`: Spike times (in ms) for each selected cluster.
     - `firing_rates_by_cluster.csv`: Firing rates (in Hz) for each cluster over time.
   - **HTML Plots**: Interactive HTML plots of firing rates for each cluster, saved in an `firing_rate_images` folder.

This tool provides a streamlined process for analyzing Neuropixel spike data, with outputs designed for easy data interpretation and visualization.
