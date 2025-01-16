
# Neuropixel Data Exporter

Neuropixel Data Exporter is a tool for analyzing and exporting Neuropixel spike data, post-Kilosort processing. This script enables users to select specific clusters, calculate firing rates, ISI histograms, and hazard functions, and export results in structured formats.

**Note**: If you download a new version of the script, remember to re-run pip install -r requirements.txt in the export_env to ensure all dependencies are up-to-date.

## Features

- **Cluster Selection**: Choose specific clusters for analysis (can use labels from Phy).
- **Spike Time Filtering**: Filter spikes based on specified time ranges and drug application time.
- **Firing Rate Calculation**: Calculate firing rates for each cluster using user-defined time bins.
- **ISI Histogram & Hazard Function**: Generate ISI histograms and calculate hazard functions with summary metrics.
- **Delta Firing Rates**: Compute delta firing rates relative to a baseline period, if specified.
- **Data Export**: Save spike times, firing rates, ISI histograms, and hazard functions as structured CSV files, Excel files, and text files.
- **Interactive Plots**: Generate HTML plots of firing rates for each cluster with optional drug application markers.

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
   It’s recommended to use a virtual environment to manage dependencies. Run one of the following commands to create one, naming it `export_env`:

   ### Conda Environment Setup

   - Create an environment named `export_env` using Conda:

     ```bash
     conda create -n export_env python=3.8
     conda activate export_env
     ```

   - Install the required dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   ### Standard Python (venv) Setup

   - Create a virtual environment named `export_env`:

     ```bash
     python -m venv export_env
     ```

   - Activate the environment:

     ```bash
     export_env\Scripts\activate
     ```

   - Install the required dependencies:

     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Activate the Environment**  
   Run one of the below codes depending on how you set up your environment:

   ### Conda Environment

      ```bash
      conda activate export_env 
      ```

   ### Standard Python (venv)

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
   - `cluster_group.tsv`: A tab-separated values file containing cluster IDs and their associated group labels, used to map clusters to specific groups for filtering.

4. **Provide Input Parameters**  
   The script will prompt you to enter:
   - **Clusters to Export**: Enter the clusters you want to analyze as specific channels (e.g., `5,7,12`) or labels (e.g., `good, mua`). If using labels, it's best to use the custom ones you can make when curating your data in Phy.
   - **Drug Application Time**: Enter the time (in seconds) when the drug was applied, or press Enter to skip.
   - **Start and End Times**: Specify the time range for analysis, or press Enter to analyze the full range.
   - **Bin Size**: Enter the bin size in seconds for firing rate calculation, or press Enter to use the default of 1 second.
   - **Baseline times**: Enter the start and end times (in seconds) for the baseline period, used to calculate the baseline firing rate for delta firing rate computation.

5. **Outputs**  
   A new folder will be created in the selected data directory containing:
   - **CSV Files**:
     - `spike_times_by_cluster_time_ms.csv`: Spike times (in ms) for each selected cluster.
     - `firing_rates_by_cluster.csv`: Firing rates (in Hz) for each cluster over time.
   - **HTML Plots**: Interactive HTML plots of firing rates for each cluster, saved in a `firing_rate_images` folder.
   - **TXT Files**: For importing directly into Clampfit for offline analysis, saved in a `txt_files_for_clampfit_import` folder.

This tool provides a streamlined process for analyzing Neuropixel spike data, with outputs designed for easy data interpretation and visualization.

## Dependencies

Ensure all dependencies are installed from `requirements.txt` to run the script successfully.

## License

This project is licensed under the GNU General Public License v3.0. © 2024 Michael Perkinson.
