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
   
   Run `python sort_and_cal.py` to produce calibrated CSVs in `processed_data/`.
   The script options (lines 731‑748 of `sort_and_cal.py`) allow adjusting:
   - `file_list_path` – text file of input CSVs
   - `data_folder` – location of calibration files
   - `output_folder` – where processed files are written
   - `energy_cut` – lower energy threshold (default 50)
   - `save_all_events` – keep intermediate events
   - `clear_outputs` – remove previous outputs before starting
   - `chunksize` – number of rows read per batch
   - `max_memory_mb` – limit memory usage

   Output files include `dssd_non_vetoed_events.csv`, `ppac_events.csv`, `rutherford_events.csv`, a merged event file and processing logs.

3. **Configure correlations**
   
   Edit `correlation_config.yaml` to set PPAC coincidence windows and define correlation chains.

4. **Build correlations**
   
   Run `python build_correlations.py` to create pickled correlation data. Use the environment variable `RUN_DIR` if your processed data live in a different directory than the default `long_run_4mbar_500V`.

5. **Batch processing** (optional)
   
   `python run_correlations_from_list.py --run-list file_list_correlations.txt` will run `build_correlations.py` for each entry and merge the results.

6. **Analyse**
   
   Inspect the correlation outputs with `analysis_notebook.ipynb`.

## Outputs

- `processed_data/` – calibrated CSVs and processing logs from `sort_and_cal.py`
- `correlations/<RUN_DIR>/` – `coincident_imp.pkl`, `decay_candidates.pkl`, `final_correlated.pkl` produced by `build_correlations.py`

## Modifiable parameters

Key parameters such as memory limits (`max_memory_mb`), chunk size (`chunksize`), energy cuts (`energy_cut`) and output directory names can be customised directly in the Python scripts before running them.

