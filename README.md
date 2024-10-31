
# Neuropixel Data Exporter

Neuropixel Data Exporter is a tool for analyzing and exporting Neuropixel spike data, post-Kilosort processing. This script enables users to select specific clusters, calculate firing rates, ISI histograms, and hazard functions, and export results in structured formats.

**Note**: If you download a new version of the script, remember to re-run pip install -r requirements.txt in the export_env to ensure all dependencies are up-to-date.

## Features

- **Cluster Selection**: Choose specific clusters for analysis.
- **Spike Time Filtering**: Filter spikes based on specified time ranges and drug application time.
- **Firing Rate Calculation**: Calculate firing rates for each cluster using user-defined time bins.
- **ISI Histogram & Hazard Function**: Generate ISI histograms and calculate hazard functions with summary metrics.
- **Data Export**: Save spike times, firing rates, ISI histograms, and hazard functions as structured CSV files, Excel files, and text files.
- **Interactive Plots**: Generate HTML plots of firing rates for each cluster.

## Installation

1. **Clone the Repository**  
   Clone this repository to your local system.

   ```bash
   git clone https://github.com/Michael-Perkinson/NeuropixelDataExporter.git
   cd NeuropixelDataExporter
   ```

   Or download and unzip the repository directly, then navigate to the folder:

   ```bash
   cd /path/to/NeuropixelDataExporter/
   ```

2. **Create a Virtual Environment**  
   It’s recommended to use a virtual environment to manage dependencies. Run the following command to create one, naming it `export_env`:

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
   Use `pip` to install the necessary packages from the requirements file:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Activate the Environment**  
   If you have opened a new terminal since installation, run:

   ```bash
   export_env\Scripts\activate
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
   A new folder will be created in the selected data directory containing:
   - **CSV Files**:
     - `spike_times_by_cluster_time_ms.csv`: Spike times (in ms) for each selected cluster.
     - `firing_rates_by_cluster.csv`: Firing rates (in Hz) for each cluster over time.
   - **HTML Plots**: Interactive HTML plots of firing rates for each cluster, saved in a `firing_rate_images` folder.
   - **TXT Files**: For importing directly into Clampfit for offline analysis, saved in a `txt_files_for_clampfit_import` folder.

This tool provides a streamlined process for analyzing Neuropixel spike data, with outputs designed for easy data interpretation and visualization.

## Dependencies

- Python 3.x
- numpy
- pandas
- plotly
- openpyxl
- tkinter (for GUI file selection)

Ensure all dependencies are installed to run the script successfully.
