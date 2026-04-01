# Neuropixel Data Exporter

A PySide6 desktop application for analysing and exporting Neuropixel spike data following Kilosort spike-sorting. The tool provides a graphical interface for cluster selection, firing rate analysis, cell-type classification (oxytocin vs vasopressin), ISI/hazard function analysis, and multi-sheet Excel export — replacing the original terminal-based script (`export_clusters.py`).

> **Upgrading from v1.x?** The tool is now a GUI application. See [What's New](#whats-new) and the [CHANGELOG](CHANGELOG.md) for a full breakdown.

---

## Features

### Graphical Interface
- Fixed-size (1080×720) two-column PySide6 window — analysis settings on the left, drug event management on the right
- Light theme with clear section grouping; dark mode toggle available under Settings
- Analysis runs on a background thread — the GUI stays responsive during long exports
- Session persistence — most settings are remembered between runs
- Import/export of full settings as a JSON file (Settings menu)

### Cluster Selection
- Enter cluster IDs numerically (`1, 5, 12`) or by Kilosort/Phy label (`good`, `mua`, or any custom label)
- Mix numeric and label inputs in a single field (e.g. `good, 14, 22`)
- Label dropdown auto-populates from the selected data folder

### Firing Rate Analysis
- Configurable analysis window (start/end time in seconds; leave end blank to use recording maximum)
- User-defined bin size (default 60 s)
- Optional firing rate baseline: specify a window (start/end) to compute mean baseline firing rate for delta-from-baseline export

### Cell-Type Classification
Two independent IV injection protocols for classifying magnocellular neurons as putative Oxytocin or Vasopressin:

**CCK (IV)** — Cholecystokinin protocol
- Compares mean firing rate 5 minutes before vs 5 minutes after injection (1-minute bins)
- delta ≥ +0.5 Hz → Putative Oxytocin
- delta < +0.5 Hz → Putative Vasopressin
- Pre-window linear regression flags baseline stability (Stable / Small / Medium / Large drift)

**PE (IV)** — Phenylephrine protocol
- Compares mean firing rate 1 minute before vs 1 minute after injection (10-second bins)
- delta ≤ −0.5 Hz → Putative Vasopressin
- delta > −0.5 Hz → Putative Oxytocin
- Pre-window linear regression flags baseline stability

Both protocols export a dedicated sheet with Pre_Mean_FR_Hz, Post_Mean_FR_Hz, Delta_FR_Hz, Classification, and a human-readable Baseline_Stability column.

### Drug Event Management
- Add named drug events with route (Microdialysis / IV), onset time, and optional offset time
- Enter `max` as the end time for a drug that runs to the end of the recording
- Set a peri-drug window per drug (e.g. `600/0` = 600 s pre, 0 s post; `60` = 60 s each side)
- Multiple drugs supported simultaneously; right-click a row to remove it

### ISI Histogram & Hazard Function
- Full-recording ISI histogram and hazard function
- Configurable early-recording reference window (default 0–600 s, "ISI Hazard Window" in the Analysis section) — useful as a pre-drug reference when the exact drug times vary
- Optional peri-drug hazard: for each drug, computes ISI histograms and hazard functions for one bin immediately before drug onset (PreDrug) and one bin at the end of drug application (EndDrug); each epoch is written to its own sheet
- All hazard output goes to a single `isi_and_hazard_analysis.xlsx` file

### Output Options (all toggleable)
| Checkbox | What it produces |
|---|---|
| Binned ISI & Hazard | `isi_and_hazard_analysis.xlsx` with full and early-window sheets |
| Peri-drug Hazard | Adds `<Drug>_PreDrug_ISI/Hazard` and `<Drug>_EndDrug_ISI/Hazard` sheets per drug |
| Export TXT (Clampfit) | `.txt` files in `txt_files_for_clampfit_import/` |
| Export All Graphs | Interactive HTML plots in `firing_rate_images/` |
| Mean by Label | `Mean_by_Label` sheet + `Mean_by_Label_Peri` sheet with all drugs combined |
| Peri-drug Sheets | `Peri_<Drug>` firing rate sheets time-aligned to each drug onset |

### Interactive Plots
- Per-cluster HTML bar charts (Plotly) with drug event overlays
- Point events (no end time) shown as dashed vertical lines with labels
- Continuous events (with start + end) shown as shaded regions
- `max` end times resolve to the actual recording end in the plot

---

## Project Structure

```
NeuropixelDataExporter/
├── main.py                        # Entry point — launches the GUI
├── src/
│   ├── core/
│   │   ├── cck_analysis.py        # CCK and PE cell-type classification engine
│   │   ├── file_manager.py        # Kilosort folder validation and file loading
│   │   ├── firing_rate.py         # Firing rate calculation and dataframe construction
│   │   ├── input_parser.py        # Cluster/label and drug event input parsing
│   │   ├── interactive_plot.py    # Plotly HTML export
│   │   ├── isi_hazard.py          # ISI histogram and hazard function calculation
│   │   ├── results_writer.py      # Excel multi-sheet export
│   │   └── spike_filter.py        # Spike data loading and filtering
│   └── gui/
│       ├── gui_controller.py      # Controller — input validation, threading, settings
│       ├── gui_themes.py          # QSS stylesheets (light/dark)
│       ├── view.py                # MainWindow — all UI construction
│       └── file_chooser.py        # Folder browser dialog
└── tests/
    └── test_firing_rate.py
```

---

## Installation

### Prerequisites
- Python 3.11 or later
- A Kilosort output folder containing:
  - `spike_times.npy`
  - `spike_clusters.npy`
  - `cluster_group.tsv` **or** `cluster_KSLabel.tsv`

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Michael-Perkinson/NeuropixelDataExporter.git
   cd NeuropixelDataExporter
   ```

2. **Create a virtual environment**

   **Conda:**
   ```bash
   conda create -n export_env python=3.11
   conda activate export_env
   pip install -r requirements.txt
   ```

   **Standard venv (Windows):**
   ```bash
   python -m venv export_env
   export_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python main.py
   ```

> **Note:** If you pull a new version, always re-run `pip install -r requirements.txt` inside your activated environment to pick up any new dependencies.

---

## Usage

### 1. Select a Data Folder
Click **Browse** and navigate to your Kilosort output folder. The app will validate that the required files are present and populate the cluster label dropdown automatically.

### 2. Configure Analysis

**Analysis section (left column, top):**
- **Clusters / Labels** — enter the clusters or labels to analyse
- **Start / End / Bin** — time window and bin size for firing rate calculation
- **FR Baseline** — tick and enter a window to compute a firing rate baseline for delta export
- **ISI Hazard Window** — set the early-recording reference window for hazard analysis (default 0–600 s); this is independent of the FR Baseline

**Cell Typing tabs:**
- Enter the CCK injection time (seconds) in the CCK tab to run the OT/VP classification protocol
- Enter the PE injection time in the PE tab to run the PE protocol
- Both are optional and independent; leave blank to skip

**Output Options (left column, bottom):**
Tick the outputs you want. All are on by default.

### 3. Add Drug Events

In the right column, fill in the drug name, route, start time (seconds), and optionally:
- **End time** — leave blank for an acute injection; enter a time in seconds for a continuous infusion; enter `max` to use the recording end
- **Peri (s)** — enter `60` for a symmetric 60 s window or `300/60` for 300 s pre / 60 s post

Click **Add Drug Event**. Events appear in the table below; right-click to remove. Multiple drugs can be added.

### 4. Run Analysis

Click **Run Analysis**. Progress appears in the log at the bottom. The button disables during the run and re-enables when complete.

---

## Outputs

All output is saved to a timestamped folder inside your data folder.

### `firing_rates_by_cluster.xlsx`

| Sheet | Contents |
|---|---|
| **Summary** | Recording parameters, protocol details, neuron counts (total / putative OT / putative VP per protocol), peri-drug window ranges, any clipping warnings |
| **Sheet_Guide** | First sheet — plain-English description of every tab, what it contains and the time window it covers |
| **CCK_Cell_Typing** | Pre/Post mean FR, delta, Classification (`Putative Oxytocin` / `Putative Vasopressin` / `Unclassifiable (Zero FR)`), Notes (baseline stability direction, zero pre-FR flag) |
| **PE_Cell_Typing** | Same format as CCK_Cell_Typing |
| **Baseline_Mean_and_SD (Xs–Ys)** | Mean and SD of firing rate across the baseline window (if FR Baseline enabled); mean is used to compute delta sheets |
| **Peri_\<Drug\>** | Binned firing rates over the peri-drug window, t=0 at drug onset |
| **Peri_\<Drug\>\_Delta** | Per-cluster firing rate change from baseline over the peri-drug window (if FR Baseline enabled) |
| **Mean_by_Label_Peri** | All peri-drug windows combined into one sheet, averaged by Phy group label, each drug separated by a header row (if Mean by Label enabled) |
| **Mean_by_Label** | Firing rates averaged per label group across the full recording window |
| **Binned_Firing_Rates** | Binned per-cluster firing rates across the full recording window (always last) |

### `isi_and_hazard_analysis.xlsx`

Cluster columns are labelled with putative cell-type where classification has been run (e.g. `Cluster_1 (Putative Oxytocin)`), matching the firing rate sheets.

| Sheet | Contents |
|---|---|
| **Full_ISI** | ISI histogram counts — full recording |
| **Full_Hazard** | Hazard function values — full recording |
| **Full_Hazard_Summary** | Peak early hazard, mean late hazard, and hazard ratio per cluster |
| **Early_ISI (X–Ys)** | ISI histogram for the configured early window (default 0–600 s) |
| **Early_Hazard (X–Ys)** | Hazard function for the early window |
| **Early_Hazard_Summary** | Summary metrics for the early window |
| **Summary** | Guide sheet listing every sheet, its section, and the time window it covers |
| **\<Drug\>\_PreDrug\_ISI** | ISI histogram for the 1-bin window immediately before drug onset (\_PreDrug suffix) |
| **\<Drug\>\_PreDrug\_Hazard** | Hazard function for the pre-drug bin |
| **\<Drug\>\_PreDrug\_HazSumm** | Hazard summary metrics for the pre-drug bin |
| **\<Drug\>\_EndDrug\_ISI** | ISI histogram for the 1-bin window at end of drug application (only when drug has an end time) |
| **\<Drug\>\_EndDrug\_Hazard** | Hazard function for the end-drug bin |
| **\<Drug\>\_EndDrug\_HazSumm** | Hazard summary metrics for the end-drug bin |

### `spike_times_by_cluster_time_ms.csv`
Raw spike times in milliseconds for each selected cluster.

### `firing_rate_images/`
One interactive HTML file per cluster with firing rate bars and drug event overlays (vertical lines for acute events, shaded regions for continuous events).

### `txt_files_for_clampfit_import/`
Per-cluster `.txt` files formatted for direct import into Clampfit (if Export TXT enabled).

---

## Classification Thresholds and Constants

All protocol thresholds are defined as module-level constants in `src/core/cck_analysis.py` for easy adjustment:

```python
CCK_WINDOW_S          = 300.0   # pre/post window (seconds)
CCK_BIN_S             = 60.0    # bin size (seconds)
CCK_THRESHOLD_HZ      = 0.5     # delta threshold for OT classification
CCK_STABILITY_HZ_PER_MIN = 0.1  # slope above which baseline is flagged

PE_WINDOW_S           = 60.0
PE_BIN_S              = 10.0
PE_THRESHOLD_HZ       = -0.5
PE_STABILITY_HZ_PER_MIN = 0.1
```

Baseline stability categories (applied to both protocols):

| Category | Slope magnitude |
|---|---|
| Stable | ≤ 0.1 Hz/min |
| Small drift | 0.1 – 0.3 Hz/min |
| Medium drift | 0.3 – 0.6 Hz/min |
| Large drift ⚠ | > 0.6 Hz/min |

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| PySide6 | GUI framework |
| numpy | Array operations |
| pandas | Dataframe construction and Excel export |
| xlsxwriter | Multi-sheet `.xlsx` writing |
| plotly | Interactive HTML plots |
| scipy | Linear regression for baseline stability |

---

## License

This project is licensed under the GNU General Public License v3.0. © 2024 Michael Perkinson.
