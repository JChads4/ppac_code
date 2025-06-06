import sys
import pandas as pd
import os
import numpy as np
import yaml
import time
import gc
from time_units import TO_S, TO_US, TO_NS


# ============================================================================
# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Configuration is provided via a YAML file so that users can easily
# customise PPAC coincidence windows and correlation chains.  The default
# file is ``correlation_config.yaml`` in the current directory.

CONFIG_PATH = 'correlation_config.yaml'

def _to_float_if_str(value):
    """Convert numeric strings to floats, leaving other values unchanged."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

ppac_cfg = {k: _to_float_if_str(v) for k, v in config.get('ppac_window', {}).items()}
window_before_ns = ppac_cfg.get('before_ns', 1700)
window_after_ns = ppac_cfg.get('after_ns', 0)
min_ppac_hits = ppac_cfg.get('min_hits', 3)

correlation_chains = config.get('chains', [])
for chain in correlation_chains:
    for step in chain.get('steps', []):
        for key in ['energy_min', 'energy_max', 'corr_min', 'corr_max']:
            if key in step:
                step[key] = _to_float_if_str(step[key])
if not correlation_chains:
    # Fallback to a simple alpha correlation if no config provided
    correlation_chains = [
        {
            'name': 'alpha_only',
            'steps': [
                {
                    'label': 'recoil',
                    'ppac_required': True,
                    'energy_min': 0,
                    'energy_max': np.inf,
                },
                {
                    'label': 'alpha',
                    'ppac_required': False,
                    'energy_min': 8100,
                    'energy_max': 8400,
                    'corr_min': 0.08,
                    'corr_max': 10,
                },
            ],
        }
    ]

# =============================================================================
# 1. DATA LOADING WITH MEMORY OPTIMIZATIONS
# =============================================================================

# Define memory-efficient dtypes for each CSV file.
dssd_dtypes = {
    'event_type': 'category',  # limited set of types
    'x': 'int32',
    'y': 'int32',
    'tagx': 'int64',
    'tagy': 'int64',
    'nfile': 'int16',
    'tdelta': 'float32',
    'nX': 'int16',
    'nY': 'int16',
    'xE': 'float32',
    'yE': 'float32',
    'xboard': 'int8',
    'yboard': 'int8',
}
ppac_dtypes = {
    'detector': 'category',
    'timetag': 'int64',
    'energy': 'float32',
    'board': 'int8',
    'channel': 'int8',
    'nfile': 'int16',
}
ruth_dtypes = {
    'detector': 'category',
    'timetag': 'int64',
    'energy': 'float32',
    'board': 'int8',
    'channel': 'int8',
    'nfile': 'int16',
}

# Read CSV files. (For very large files consider adding chunksize)
dssd = pd.read_csv('processed_data/dssd_non_vetoed_events.csv', dtype=dssd_dtypes)
ppac = pd.read_csv('processed_data/ppac_events.csv', dtype=ppac_dtypes)
ruth = pd.read_csv('processed_data/rutherford_events.csv', dtype=ruth_dtypes)

# =============================================================================
# 2. DATA SEGREGATION AND SORTING
# =============================================================================

# Split DSSD data into regions
imp = dssd[dssd['event_type'] == 'imp']
boxE = dssd[dssd['event_type'] == 'boxE']
boxW = dssd[dssd['event_type'] == 'boxW']
boxT = dssd[dssd['event_type'] == 'boxT']
boxB = dssd[dssd['event_type'] == 'boxB']

# Split PPAC data by detector type
cathode = ppac[ppac['detector'] == 'cathode']
anodeV = ppac[ppac['detector'] == 'anodeV']
anodeH = ppac[ppac['detector'] == 'anodeH']

# Split Rutherford data
ruth_E = ruth[ruth['detector'] == 'ruthE']
ruth_W = ruth[ruth['detector'] == 'ruthW']

# Define the coincidence window (in ns) and convert to ps
window_before_ps=window_before_ns*1000
window_after_ps=window_after_ns*1000
cathode_sorted = cathode.sort_values('timetag').reset_index(drop=True)
anodeV_sorted = anodeV.sort_values('timetag').reset_index(drop=True)
anodeH_sorted = anodeH.sort_values('timetag').reset_index(drop=True)
imp_sorted = imp.sort_values('tagx').reset_index(drop=True)

# Create a column 't' (time in seconds) for the IMP events (needed for later decay analysis)
imp_sorted['t'] = imp_sorted['tagx'] * TO_S  # converting picoseconds to seconds


# Cache timetag arrays for PPAC detectors for fast binary search
cathode_timetags = cathode_sorted['timetag'].values
anodeV_timetags = anodeV_sorted['timetag'].values
anodeH_timetags = anodeH_sorted['timetag'].values

# =============================================================================
# 3. DEFINE BINARY SEARCH FUNCTION FOR TIME WINDOWS
# =============================================================================

def find_events_in_window(imp_timetag, detector_timetags, window_before_ps, window_after_ps):
    """
    Find detector events within a specified time window around an IMP event using binary search.
    
    Parameters:
      imp_timetag (int): Timestamp of the IMP event (in ps).
      detector_timetags (ndarray): Sorted array of detector timestamps (in ps).
      window_before_ps (int): Time window before the IMP event (in ps).
      window_after_ps (int): Time window after the IMP event (in ps).
    
    Returns:
      list: Indices of events within the time window.
    """
    lower_bound = imp_timetag - window_before_ps
    upper_bound = imp_timetag + window_after_ps
    lower_idx = np.searchsorted(detector_timetags, lower_bound)
    upper_idx = np.searchsorted(detector_timetags, upper_bound)
    if upper_idx > lower_idx:
        return list(range(lower_idx, upper_idx))
    return []

# =============================================================================
# 4. COINCIDENCE SEARCH BETWEEN IMP AND PPAC EVENTS
# =============================================================================

start_time = time.time()
coincident_events = []
non_ppac_coincident_events = []
total_imp_events = len(imp_sorted)
print(f"Processing {total_imp_events} IMP events...")

for idx, imp_row in imp_sorted.iterrows():
    imp_timetag = imp_row['tagx']  # using tagx as the IMP time
    # Find PPAC events within the specified time window
    cathode_indices = find_events_in_window(imp_timetag, cathode_timetags, window_before_ps, window_after_ps)
    anodeV_indices = find_events_in_window(imp_timetag, anodeV_timetags, window_before_ps, window_after_ps)
    anodeH_indices = find_events_in_window(imp_timetag, anodeH_timetags, window_before_ps, window_after_ps)

    hits_found = int(bool(cathode_indices)) + int(bool(anodeV_indices)) + int(bool(anodeH_indices))

    if hits_found >= min_ppac_hits:
        # Find the closest PPAC event for each detector when available
        if cathode_indices:
            cathode_diffs = np.abs(cathode_timetags[cathode_indices] - imp_timetag)
            closest_cathode_idx = cathode_indices[np.argmin(cathode_diffs)]
            cathode_data = cathode_sorted.iloc[closest_cathode_idx]
            dt_cathode_ps = cathode_data['timetag'] - imp_timetag
        else:
            cathode_data = {k: np.nan for k in ['timetag','energy','board','channel','nfile']}
            dt_cathode_ps = np.nan

        if anodeV_indices:
            anodeV_diffs = np.abs(anodeV_timetags[anodeV_indices] - imp_timetag)
            closest_anodeV_idx = anodeV_indices[np.argmin(anodeV_diffs)]
            anodeV_data = anodeV_sorted.iloc[closest_anodeV_idx]
            dt_anodeV_ps = anodeV_data['timetag'] - imp_timetag
        else:
            anodeV_data = {k: np.nan for k in ['timetag','energy','board','channel','nfile']}
            dt_anodeV_ps = np.nan

        if anodeH_indices:
            anodeH_diffs = np.abs(anodeH_timetags[anodeH_indices] - imp_timetag)
            closest_anodeH_idx = anodeH_indices[np.argmin(anodeH_diffs)]
            anodeH_data = anodeH_sorted.iloc[closest_anodeH_idx]
            dt_anodeH_ps = anodeH_data['timetag'] - imp_timetag
        else:
            anodeH_data = {k: np.nan for k in ['timetag','energy','board','channel','nfile']}
            dt_anodeH_ps = np.nan
        
        event_data = {
            # IMP data
            'imp_timetag': imp_timetag,
            'imp_x': imp_row['x'],
            'imp_y': imp_row['y'],
            'imp_tagx': imp_row['tagx'],
            'imp_tagy': imp_row['tagy'],
            'imp_nfile': imp_row['nfile'],
            'imp_tdelta': imp_row['tdelta'],
            'imp_nX': imp_row['nX'],
            'imp_nY': imp_row['nY'],
            'imp_xE': imp_row['xE'],
            'imp_yE': imp_row['yE'],
            'xboard': imp_row['xboard'],
            'yboard': imp_row['yboard'],
            # PPAC data
            'cathode_timetag': cathode_data.get('timetag'),
            'cathode_energy': cathode_data.get('energy'),
            'cathode_board': cathode_data.get('board'),
            'cathode_channel': cathode_data.get('channel'),
            'cathode_nfile': cathode_data.get('nfile'),
            'anodeV_timetag': anodeV_data.get('timetag'),
            'anodeV_energy': anodeV_data.get('energy'),
            'anodeV_board': anodeV_data.get('board'),
            'anodeV_channel': anodeV_data.get('channel'),
            'anodeV_nfile': anodeV_data.get('nfile'),
            'anodeH_timetag': anodeH_data.get('timetag'),
            'anodeH_energy': anodeH_data.get('energy'),
            'anodeH_board': anodeH_data.get('board'),
            'anodeH_channel': anodeH_data.get('channel'),
            'anodeH_nfile': anodeH_data.get('nfile'),
            # Time differences
            'dt_cathode_ps': dt_cathode_ps,
            'dt_anodeV_ps': dt_anodeV_ps,
            'dt_anodeH_ps': dt_anodeH_ps,
            'dt_cathode_ns': dt_cathode_ps * TO_NS,
            'dt_anodeV_ns': dt_anodeV_ps * TO_NS,
            'dt_anodeH_ns': dt_anodeH_ps * TO_NS,
        }
        coincident_events.append(event_data)
    else:
        # Record IMP events with no PPAC coincidences
        non_coincident_data = {
            'timetag': imp_timetag,
            't': imp_timetag * TO_S,
            'x': imp_row['x'],
            'y': imp_row['y'],
            'tagx': imp_row['tagx'],
            'tagy': imp_row['tagy'],
            'nfile': imp_row['nfile'],
            'tdelta': imp_row['tdelta'],
            'nX': imp_row['nX'],
            'nY': imp_row['nY'],
            'xE': imp_row['xE'],
            'yE': imp_row['yE'],
            'xboard': imp_row['xboard'],
            'yboard': imp_row['yboard'],
        }
        non_ppac_coincident_events.append(non_coincident_data)
    
    # Print progress every 10,000 events
    if idx % 10000 == 0 and idx > 0:
        elapsed = time.time() - start_time
        events_per_sec = idx / elapsed
        remaining_time = (total_imp_events - idx) / events_per_sec 
        print(f"Processed {idx}/{total_imp_events} events ({idx/total_imp_events:.1%}) - ETA: {remaining_time:.1f} sec")

# Create DataFrames from the coincidence lists
coincident_imp_df = pd.DataFrame(coincident_events)
non_coincident_imp_df = pd.DataFrame(non_ppac_coincident_events)
print(f"Found {len(coincident_imp_df)} coincidences within the window")
elapsed_time = time.time() - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")
print(f"Processing rate: {total_imp_events/elapsed_time:.1f} events/sec")

# Free large objects that are no longer needed
del dssd, ppac, ruth, cathode, anodeV, anodeH, imp
gc.collect()

# =============================================================================
# 5. PLOTTING RAW E-ToF AND TIME CORRECTIONS
# =============================================================================

# Convert time differences from ps to µs for plotting convenience
coincident_imp_df['dt_cathode_us'] = coincident_imp_df['dt_cathode_ps'] * TO_US
coincident_imp_df['dt_anodeV_us'] = coincident_imp_df['dt_anodeV_ps'] * TO_US
coincident_imp_df['dt_anodeH_us'] = coincident_imp_df['dt_anodeH_ps'] * TO_US


# Apply manual board offsets using vectorized mapping
manual_offsets = {
    0: 0,
    1: -0.045e-6,
    2: -0.065e-6,
    3: -0.085e-6,
    4: -0.105e-6,
    5: -0.125e-6,
}
coincident_imp_df['dt_anodeH_us_corr'] = (coincident_imp_df['dt_anodeH_us'] +
                                          coincident_imp_df['xboard'].map(manual_offsets))
coincident_imp_df['dt_anodeV_us_corr'] = (coincident_imp_df['dt_anodeV_us'] +
                                          coincident_imp_df['xboard'].map(manual_offsets))
coincident_imp_df['dt_cathode_us_corr'] = (coincident_imp_df['dt_cathode_us'] +
                                           coincident_imp_df['xboard'].map(manual_offsets))

# (Additional plots by board can be inserted here following similar practices.)
# For brevity, only the raw E-ToF and correction steps are shown.

# =============================================================================
# 6. DECAY EVENT CANDIDATE IDENTIFICATION
# =============================================================================

# Build pixel history from the IMP events (using imp_sorted that we prepared earlier)
# Group by pixel (x, y) and ensure each group is sorted by time 't'
pixel_groups = imp_sorted.groupby(['x', 'y'])
pixel_history = {pixel: group.sort_values('t') for pixel, group in pixel_groups}

# Define decay time window (in seconds)
min_corr_time = 1e-8  # Minimum time after recoil to consider (10 ns)
max_corr_time = 20    # Maximum time after recoil (20 s)

decay_candidates = []  # List to store candidate decay events

# Loop through each recoil (coincident IMP event)
for recoil_idx, recoil in coincident_imp_df.iterrows():
    pixel = (recoil['imp_x'], recoil['imp_y'])
    recoil_time_sec = recoil['imp_timetag'] * TO_S  # convert to seconds
    if pixel not in pixel_history:
        continue  # No events for this pixel in history
    pixel_df = pixel_history[pixel]
    time_array = pixel_df['t'].values  # Already in seconds
    lower_bound = recoil_time_sec + min_corr_time
    upper_bound = recoil_time_sec + max_corr_time
    start_idx = np.searchsorted(time_array, lower_bound, side='left')
    end_idx = np.searchsorted(time_array, upper_bound, side='right')
    if start_idx < end_idx:
        candidate_events = pixel_df.iloc[start_idx:end_idx].copy()
        candidate_events['recoil_index'] = recoil_idx
        candidate_events['recoil_time_sec'] = recoil_time_sec
        decay_candidates.append(candidate_events)

if decay_candidates:
    decay_candidates_df = pd.concat(decay_candidates, ignore_index=True)
else:
    decay_candidates_df = pd.DataFrame()

print("First few decay candidates:")
print(decay_candidates_df.head())

# =============================================================================
# 7. PPAC ANTICOINCIDENCE CHECK FOR DECAYS
# =============================================================================

if not decay_candidates_df.empty and not non_coincident_imp_df.empty:
    non_coincident_clean = non_coincident_imp_df[['x', 'y']].drop_duplicates()
    decay_candidates_df = decay_candidates_df.merge(
        non_coincident_clean,
        on=['x', 'y'],
        how='left',
        indicator='ppac_flag'
    )
    decay_candidates_df['is_clean'] = decay_candidates_df['ppac_flag'] == 'left_only'
    print("PPAC anticoincidence check counts:")
    print(decay_candidates_df['is_clean'].value_counts())
    print(decay_candidates_df.head())
else:
    print("No decay candidates or non-coincident events available for PPAC anticoincidence check.")

# Calculate log time difference between decay candidate and recoil event
if not decay_candidates_df.empty:
    decay_candidates_df['log_dt'] = np.log(np.abs(decay_candidates_df['t'] - decay_candidates_df['recoil_time_sec']))


# =============================================================================
# 8. GENERIC CORRELATION BUILDING
# =============================================================================

def correlate_events(recoil_df, decay_df, chain):
    """Build correlations for a single chain configuration."""
    steps = chain['steps']
    first = steps[0]
    recoils = recoil_df[(recoil_df['xE'] >= first.get('energy_min', 0)) &
                        (recoil_df['xE'] <= first.get('energy_max', np.inf))].copy()
    recoils.rename(columns={'x': f"{first['label']}_x",
                            'y': f"{first['label']}_y",
                            't': f"{first['label']}_t",
                            'xE': f"{first['label']}_xE"}, inplace=True)
    stage_df = recoils
    prev_label = first['label']

    for step in steps[1:]:
        label = step['label']
        dataset = recoil_df if step.get('ppac_required') else decay_df
        e_min = step.get('energy_min', 0)
        e_max = step.get('energy_max', np.inf)
        dt_min = step.get('corr_min', 0)
        dt_max = step.get('corr_max', np.inf)
        results = []
        for _, row in stage_df.iterrows():
            pix = (row[f'{prev_label}_x'], row[f'{prev_label}_y'])
            pixel_events = dataset[(dataset['x'] == pix[0]) & (dataset['y'] == pix[1])]
            after = pixel_events[pixel_events['t'] > row[f'{prev_label}_t']]
            energy_sel = after[(after['xE'] >= e_min) & (after['xE'] <= e_max)]
            if energy_sel.empty:
                continue
            energy_sel = energy_sel.copy()
            energy_sel['dt'] = energy_sel['t'] - row[f'{prev_label}_t']
            window = energy_sel[(energy_sel['dt'] >= dt_min) & (energy_sel['dt'] <= dt_max)]
            if window.empty:
                continue
            evt = window.iloc[0]
            new_row = row.to_dict()
            new_row[f'{label}_x'] = evt['x']
            new_row[f'{label}_y'] = evt['y']
            new_row[f'{label}_t'] = evt['t']
            new_row[f'{label}_xE'] = evt['xE']
            new_row[f'{label}_dt'] = evt['dt']
            results.append(new_row)
        stage_df = pd.DataFrame(results)
        if stage_df.empty:
            break
        prev_label = label
    stage_df['chain'] = chain.get('name', 'chain')
    return stage_df

# Prepare simplified recoil and decay tables
recoil_df = coincident_imp_df[['imp_x', 'imp_y', 'imp_xE', 'imp_timetag']].copy()
recoil_df.rename(columns={'imp_x': 'x', 'imp_y': 'y', 'imp_xE': 'xE', 'imp_timetag': 'timetag'}, inplace=True)
recoil_df['t'] = recoil_df['timetag'] * TO_S

decay_df = non_coincident_imp_df[['x', 'y', 'xE', 'timetag']].copy()
decay_df['t'] = decay_df['timetag'] * TO_S

all_results = []
for chain in correlation_chains:
    res = correlate_events(recoil_df, decay_df, chain)
    if not res.empty:
        all_results.append(res)

if all_results:
    final_correlated_df = pd.concat(all_results, ignore_index=True)
else:
    final_correlated_df = pd.DataFrame()

# Save results
os.makedirs("analysis_output", exist_ok=True)
coincident_imp_df.to_pickle("analysis_output/coincident_imp.pkl")
decay_candidates_df.to_pickle("analysis_output/decay_candidates.pkl")
final_correlated_df.to_pickle("analysis_output/final_correlated.pkl")
