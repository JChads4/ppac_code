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
   PPAC hits are present, `ppac_events.pkl` is not created. The Rutherford
   events file is always produced (it will be empty if no hits occurred).

3. **Configure correlations**

   Edit `correlation_config.yaml` to set PPAC coincidence windows and define correlation chains. A
   top-level `ppac_window` section provides defaults, but each chain may override these values by
   including its own `ppac_window` block.

   The optional `box_events` section controls whether decay candidates include
   hits from the surrounding box detectors and how closely in time those hits
   must occur to be merged with the implant pixel. Set `enabled: false` to
   ignore box data or adjust `combine_window_ns` to change the merge window.

   If a run was recorded **without** a PPAC detector, set `min_hits: 0` in the
   relevant `ppac_window` so that all implant events are accepted. The code will
   automatically build decay candidates from all events in this mode.

4. **Build correlations**

   Run `python build_correlations.py [--run-dir RUN] [--base-dir DIR] [--max-memory-mb MB]` to create
   pickled correlation data. `--run-dir` defaults to the `RUN_DIR` environment
   variable (falling back to `long_run_4mbar_500V` if unset). `--base-dir`
   places the results under `correlations/<DIR>/`. The optional
   `--max-memory-mb` flag enforces a memory limit during processing.

5. **Batch processing** (optional)

   `python run_correlations_from_list.py --run-list file_list_correlations.txt [--base-dir DIR] [--max-memory-mb MB]`
   will run `build_correlations.py` for each entry, merge the results and
   remove the per-run output folders. The memory limit option is forwarded to
   each subprocess.

6. **Analyse**
   
   Inspect the correlation outputs with `analysis_notebook.ipynb`.

## Outputs

- `processed_data/<run_id>/` – pickled detector data and logs from `sort_and_cal.py`
- `correlations/<base>/` – merged `coincident_imp.pkl`, `decay_candidates.pkl` and `final_correlated.pkl` from `run_correlations_from_list.py`

## Modifiable parameters

Key parameters such as memory limits (`max_memory_mb`), chunk size (`chunksize`), energy cuts (`energy_cut`) and output directory names can be customised directly in the Python scripts before running them.

## Runs without PPAC

If your experiment did not include a PPAC detector, simply set
`min_hits: 0` in `correlation_config.yaml` (either at the top-level
`ppac_window` or within a specific chain). When `min_hits` is zero, all
implant events are considered valid recoils and decay candidates are
built from the full dataset.

