# PPAC Data Processing

This repository contains scripts for sorting raw SHREC data, building PPAC correlations and analysing the results.

## Requirements

- **Python** 3.9 or newer
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [psutil](https://pypi.org/project/psutil/)
- [PyYAML](https://pyyaml.org/)

Install dependencies with `pip install pandas numpy matplotlib psutil pyyaml`.

## Usage

1. **Prepare the file list**
   
   Create `files_to_sort.txt` containing paths to the raw CSV files to process.

2. **Sort and calibrate**

   Run `python sort_and_cal.py` to create per-run pickle files under
   `processed_data/<run_id>/`. The script options (around line 750 of
   `sort_and_cal.py`) allow adjusting:
   - `file_list_path` – text file of input CSVs
   - `data_folder` – location of calibration files
   - `output_folder` – where processed files are written
   - `energy_cut` – lower energy threshold (default 50)
   - `save_all_events` – keep intermediate events
   - `clear_outputs` – remove previous outputs before starting
   - `chunksize` – number of rows read per batch
   - `max_memory_mb` – limit memory usage

   Output files include `dssd_non_vetoed_events.pkl`, `ppac_events.pkl`,
   `rutherford_events.pkl`, a merged event pickle and processing logs. If no
   PPAC hits are present, `ppac_events.pkl` is not created.

3. **Configure correlations**

   Edit `correlation_config.yaml` to set PPAC coincidence windows and define correlation chains. A
   top-level `ppac_window` section provides defaults, but each chain may override these values by
   including its own `ppac_window` block.

4. **Build correlations**

   Run `python build_correlations.py [--run-dir RUN] [--base-dir DIR]` to create
   pickled correlation data. `--run-dir` defaults to the `RUN_DIR` environment
   variable (falling back to `long_run_4mbar_500V` if unset). `--base-dir`
   places the results under `correlations/<DIR>/`.

5. **Batch processing** (optional)

   `python run_correlations_from_list.py --run-list file_list_correlations.txt [--base-dir DIR]`
   will run `build_correlations.py` for each entry, merge the results and
   remove the per-run output folders.

6. **Analyse**
   
   Inspect the correlation outputs with `analysis_notebook.ipynb`.

## Outputs

- `processed_data/<run_id>/` – pickled detector data and logs from `sort_and_cal.py`
- `correlations/<base>/` – merged `coincident_imp.pkl`, `decay_candidates.pkl` and `final_correlated.pkl` from `run_correlations_from_list.py`

## Modifiable parameters

Key parameters such as memory limits (`max_memory_mb`), chunk size (`chunksize`), energy cuts (`energy_cut`) and output directory names can be customised directly in the Python scripts before running them.

