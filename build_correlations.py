"""Build PPAC correlation chains from processed detector data.

This script loads per-run pickle files, searches for PPAC coincidences with
implantation (IMP) events and then constructs decay correlation chains as
defined in ``correlation_config.yaml``.  Results are stored under
``correlations/<base-dir>/<run>``.

The implementation has been modularised so that the data loading, coincidence
search and correlation building steps can be reused elsewhere.
"""

import argparse
import os
import time
import gc
import numpy as np
import pandas as pd
import yaml
import sys
from contextlib import nullcontext

from memory_utils import memory_limit, MemoryLimitExceeded

from time_units import TO_S, TO_US, TO_NS

# Time window in picoseconds used to merge energy deposits from
# the IMP pixel and surrounding box regions into a single decay
# event. This is set to 1 microsecond by default.
COMBINE_WINDOW_PS = int(1e6)


# ============================================================================
# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Configuration is provided via a YAML file so that users can easily
# customise PPAC coincidence windows and correlation chains.  The default
# file is ``correlation_config.yaml`` in the current directory.

CONFIG_PATH = 'correlation_config.yaml'

parser = argparse.ArgumentParser(description="Build PPAC correlations")
parser.add_argument('--run-dir', default=os.environ.get('RUN_DIR', 'long_run_4mbar_500V'),
                    help='Name of run folder inside processed_data/')
parser.add_argument('--base-dir', default='',
                    help='Optional subdirectory under correlations/')
parser.add_argument('--max-memory-mb', type=float, default=None,
                    help='Optional maximum memory usage in MB')
args = parser.parse_args()

RUN_DIR = args.run_dir


def load_data(run_dir):
    """Return DSSD, PPAC and Rutherford DataFrames for *run_dir*.

    If ``ppac_events.pkl`` is missing (e.g. no PPAC detector was present), an
    empty DataFrame with the expected columns is returned instead of raising an
    error.
    """

    dssd = pd.read_pickle(
        f"processed_data/{run_dir}/dssd_non_vetoed_events.pkl"
    )

    ppac_path = f"processed_data/{run_dir}/ppac_events.pkl"
    if os.path.exists(ppac_path):
        ppac = pd.read_pickle(ppac_path)
    else:
        print(f"Warning: {ppac_path} not found - continuing without PPAC data")
        ppac = pd.DataFrame(columns=list(ppac_dtypes)).astype(ppac_dtypes)

    ruth = pd.read_pickle(f"processed_data/{run_dir}/rutherford_events.pkl")
    return dssd, ppac, ruth


def split_dssd_regions(dssd):
    """Split DSSD events into detector regions."""
    regions = {}
    for region in ["imp", "boxE", "boxW", "boxT", "boxB"]:
        regions[region] = dssd[dssd["event_type"] == region].copy()
    return regions


def split_ppac_detectors(ppac):
    """Return cathode, anodeV and anodeH DataFrames sorted by time."""
    cathode = ppac[ppac["detector"] == "cathode"].sort_values("timetag").reset_index(
        drop=True
    )
    anodeV = ppac[ppac["detector"] == "anodeV"].sort_values("timetag").reset_index(
        drop=True
    )
    anodeH = ppac[ppac["detector"] == "anodeH"].sort_values("timetag").reset_index(
        drop=True
    )
    return cathode, anodeV, anodeH


def prepare_decay_df(non_coincident_imp_df, box_regions, combine_window_ps=COMBINE_WINDOW_PS):
    """Return decay-like events combining energy from IMP and box regions.

    Parameters
    ----------
    non_coincident_imp_df : pd.DataFrame
        Implant-region events without PPAC coincidences.
    box_regions : dict[str, pd.DataFrame]
        Mapping of region name to DSSD DataFrame for the surrounding box.
    combine_window_ps : int, optional
        Maximum time difference between IMP and box hits to be merged
        into a single decay event.
    """

    frames = []
    imp_df = non_coincident_imp_df[["x", "y", "xE", "timetag"]].copy()
    frames.append(imp_df)
    for df in box_regions.values():
        tmp = df[["x", "y", "xE", "tagx"]].rename(columns={"tagx": "timetag"})
        frames.append(tmp)

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["x", "y", "timetag"], inplace=True)

    combined_rows = []
    for (x, y), grp in merged.groupby(["x", "y"]):
        grp = grp.sort_values("timetag").reset_index(drop=True)
        i = 0
        while i < len(grp):
            t0 = grp.at[i, "timetag"]
            e_sum = grp.at[i, "xE"]
            j = i + 1
            while j < len(grp) and grp.at[j, "timetag"] - t0 <= combine_window_ps:
                e_sum += grp.at[j, "xE"]
                j += 1
            combined_rows.append({"x": x, "y": y, "xE": e_sum, "timetag": t0})
            i = j

    decay = pd.DataFrame(combined_rows)
    decay["t"] = decay["timetag"] * TO_S
    return decay

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

# Pixel search mode for correlations
pixel_search_mode = config.get('pixel_search', 'single')
if pixel_search_mode not in ('single', 'square'):
    raise ValueError("pixel_search must be 'single' or 'square'")

correlation_chains = config.get('chains', [])
all_results = []
for chain in correlation_chains:
    for step in chain.get('steps', []):
        for key in ['energy_min', 'energy_max', 'corr_min', 'corr_max']:
            if key in step:
                step[key] = _to_float_if_str(step[key])
    chain_ppac = {k: _to_float_if_str(v) for k, v in chain.get('ppac_window', {}).items()}
    chain['ppac_before_ns'] = chain_ppac.get('before_ns', window_before_ns)
    chain['ppac_after_ns'] = chain_ppac.get('after_ns', window_after_ns)
    chain['ppac_min_hits'] = chain_ppac.get('min_hits', min_ppac_hits)

max_before_ns = max((c['ppac_before_ns'] for c in correlation_chains), default=window_before_ns)
max_after_ns = max((c['ppac_after_ns'] for c in correlation_chains), default=window_after_ns)
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

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def main():
    dssd, ppac, ruth = load_data(RUN_DIR)

    # ======================================================================
    # 2. DATA SEGREGATION AND SORTING
    # ======================================================================

    # Split DSSD data into regions
    regions = split_dssd_regions(dssd)
    imp = regions["imp"]
    boxE = regions["boxE"]
    boxW = regions["boxW"]
    boxT = regions["boxT"]
    boxB = regions["boxB"]

# Split PPAC data by detector type
    # Split PPAC data by detector type
    cathode, anodeV, anodeH = split_ppac_detectors(ppac)

    # Split Rutherford data
    ruth_E = ruth[ruth['detector'] == 'ruthE']
    ruth_W = ruth[ruth['detector'] == 'ruthW']

    # Prepare sorted PPAC data for fast time-window searches
    cathode_sorted = cathode
    anodeV_sorted = anodeV
    anodeH_sorted = anodeH
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
    
    # ---------------------------------------------------------------------------
    # 2. COINCIDENCE SEARCH BETWEEN IMP AND PPAC EVENTS
    # ---------------------------------------------------------------------------
    
    def find_ppac_hits(window_before_ns, window_after_ns):
        """Return PPAC hit information for all IMP events.
    
        The returned DataFrame contains one row per implantation event with the
        closest PPAC hit times and energies within the specified window.
        """
        start_time = time.time()
        rows = []
        total_imp_events = len(imp_sorted)
        print(f"Processing {total_imp_events} IMP events...")
    
        window_before_ps = int(window_before_ns * 1000)
        window_after_ps = int(window_after_ns * 1000)
    
        for idx, imp_row in imp_sorted.iterrows():
            imp_timetag = imp_row["tagx"]
    
            cathode_idx = find_events_in_window(
                imp_timetag, cathode_timetags, window_before_ps, window_after_ps
            )
            anodeV_idx = find_events_in_window(
                imp_timetag, anodeV_timetags, window_before_ps, window_after_ps
            )
            anodeH_idx = find_events_in_window(
                imp_timetag, anodeH_timetags, window_before_ps, window_after_ps
            )
    
            hits = int(bool(cathode_idx)) + int(bool(anodeV_idx)) + int(bool(anodeH_idx))
    
            if cathode_idx:
                diffs = np.abs(cathode_timetags[cathode_idx] - imp_timetag)
                closest = cathode_idx[np.argmin(diffs)]
                cathode_data = cathode_sorted.iloc[closest]
                dt_cathode_ps = cathode_data["timetag"] - imp_timetag
            else:
                cathode_data = {k: np.nan for k in ["timetag", "energy", "board", "channel", "nfile"]}
                dt_cathode_ps = np.nan
    
            if anodeV_idx:
                diffs = np.abs(anodeV_timetags[anodeV_idx] - imp_timetag)
                closest = anodeV_idx[np.argmin(diffs)]
                anodeV_data = anodeV_sorted.iloc[closest]
                dt_anodeV_ps = anodeV_data["timetag"] - imp_timetag
            else:
                anodeV_data = {k: np.nan for k in ["timetag", "energy", "board", "channel", "nfile"]}
                dt_anodeV_ps = np.nan
    
            if anodeH_idx:
                diffs = np.abs(anodeH_timetags[anodeH_idx] - imp_timetag)
                closest = anodeH_idx[np.argmin(diffs)]
                anodeH_data = anodeH_sorted.iloc[closest]
                dt_anodeH_ps = anodeH_data["timetag"] - imp_timetag
            else:
                anodeH_data = {k: np.nan for k in ["timetag", "energy", "board", "channel", "nfile"]}
                dt_anodeH_ps = np.nan
    
            rows.append(
                {
                    "imp_timetag": imp_timetag,
                    "imp_x": imp_row["x"],
                    "imp_y": imp_row["y"],
                    "imp_tagx": imp_row["tagx"],
                    "imp_tagy": imp_row["tagy"],
                    "imp_nfile": imp_row["nfile"],
                    "imp_tdelta": imp_row["tdelta"],
                    "imp_nX": imp_row["nX"],
                    "imp_nY": imp_row["nY"],
                    "imp_xE": imp_row["xE"],
                    "imp_yE": imp_row["yE"],
                    "xboard": imp_row["xboard"],
                    "yboard": imp_row["yboard"],
                    "cathode_timetag": cathode_data.get("timetag"),
                    "cathode_energy": cathode_data.get("energy"),
                    "anodeV_timetag": anodeV_data.get("timetag"),
                    "anodeV_energy": anodeV_data.get("energy"),
                    "anodeH_timetag": anodeH_data.get("timetag"),
                    "anodeH_energy": anodeH_data.get("energy"),
                    "dt_cathode_ps": dt_cathode_ps,
                    "dt_anodeV_ps": dt_anodeV_ps,
                    "dt_anodeH_ps": dt_anodeH_ps,
                    "dt_cathode_ns": dt_cathode_ps * TO_NS,
                    "dt_anodeV_ns": dt_anodeV_ps * TO_NS,
                    "dt_anodeH_ns": dt_anodeH_ps * TO_NS,
                    "num_hits": hits,
                }
            )
    
            if idx % 10000 == 0 and idx > 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total_imp_events - idx) / rate
                print(
                    f"Processed {idx}/{total_imp_events} events ({idx/total_imp_events:.1%}) - ETA: {remaining:.1f} sec"
                )
    
        hits_df = pd.DataFrame(rows)
    
        elapsed_time = time.time() - start_time
        print(f"Processed {total_imp_events} IMP events in {elapsed_time:.2f} s")
        if elapsed_time:
            print(f"Processing rate: {total_imp_events/elapsed_time:.1f} events/sec")
    
        return hits_df
    
    
    hits_df = find_ppac_hits(max_before_ns, max_after_ns)
    
    # Free large objects that are no longer needed
    del dssd, ppac, ruth, cathode, anodeV, anodeH
    gc.collect()
    
    
    # Build pixel history from all DSSD regions so escaping particles are included
    all_events = pd.concat([imp, boxE, boxW, boxT, boxB], ignore_index=True)
    all_events['t'] = all_events['tagx'] * TO_S
    pixel_groups = all_events.groupby(['x', 'y'])
    pixel_history = {pix: grp.sort_values('t') for pix, grp in pixel_groups}
    
    # Define decay time window (in seconds) with no upper bound.
    min_corr_time = 0.0
    max_corr_time = np.inf
    
    
    def build_results_for_chain(chain):
        """Return correlated events and intermediate DataFrames for a chain.
    
        The returned tuple contains ``(correlated, coincident_imp_df,
        decay_candidates_df)`` so that callers can persist the PPAC coincidence
        information and decay candidates in addition to the final correlation
        results.
        """
    
        before_ns = chain['ppac_before_ns']
        after_ns = chain['ppac_after_ns']
        min_hits_chain = chain['ppac_min_hits']
        before_ps = int(before_ns * 1000)
        after_ps = int(after_ns * 1000)
    
        def in_window(series):
            return (~series.isna()) & (series >= -before_ps) & (series <= after_ps)
    
        hits_in_window = (
            in_window(hits_df['dt_cathode_ps']).astype(int)
            + in_window(hits_df['dt_anodeV_ps']).astype(int)
            + in_window(hits_df['dt_anodeH_ps']).astype(int)
        )
    
        if min_hits_chain == 0:
            coincident_imp_df = hits_df.copy()
            non_coincident_imp_df = hits_df.copy()
        else:
            coincident_imp_df = hits_df[hits_in_window >= min_hits_chain].copy()
            non_coincident_imp_df = hits_df[hits_in_window < min_hits_chain].copy()
    
        # Use generic column names throughout
        rename_cols = {
            'imp_x': 'x',
            'imp_y': 'y',
            'imp_xE': 'xE',
            'imp_timetag': 'timetag',
        }
        coincident_imp_df = coincident_imp_df.rename(columns=rename_cols)
        non_coincident_imp_df = non_coincident_imp_df.rename(columns=rename_cols)
    
        if coincident_imp_df.empty:
            print(f"No coincidences found for chain {chain.get('name', 'chain')}")
            decay_candidates_df = pd.DataFrame()
            return pd.DataFrame(), coincident_imp_df, decay_candidates_df
    
        # Convert time differences from ps to µs
        coincident_imp_df['dt_cathode_us'] = coincident_imp_df['dt_cathode_ps'] * TO_US
        coincident_imp_df['dt_anodeV_us'] = coincident_imp_df['dt_anodeV_ps'] * TO_US
        coincident_imp_df['dt_anodeH_us'] = coincident_imp_df['dt_anodeH_ps'] * TO_US
    
        # Build decay candidates for this chain
        decay_candidates = []
        for recoil_idx, recoil in coincident_imp_df.iterrows():
            pixel = (recoil['x'], recoil['y'])
            recoil_time_sec = recoil['timetag'] * TO_S
            if pixel not in pixel_history:
                continue
            pixel_df = pixel_history[pixel]
            time_array = pixel_df['t'].values
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
    
        if not decay_candidates_df.empty and not non_coincident_imp_df.empty:
            non_coincident_clean = non_coincident_imp_df[['x', 'y']].drop_duplicates()
            decay_candidates_df = decay_candidates_df.merge(
                non_coincident_clean,
                on=['x', 'y'],
                how='left',
                indicator='ppac_flag'
            )
            decay_candidates_df['is_clean'] = decay_candidates_df['ppac_flag'] == 'left_only'
    
        if not decay_candidates_df.empty:
            decay_candidates_df['log_dt'] = np.log(
                np.abs(decay_candidates_df['t'] - decay_candidates_df['recoil_time_sec'])
            )
    
        box_regions = {"boxE": boxE, "boxW": boxW, "boxT": boxT, "boxB": boxB}
        decay_df = prepare_decay_df(non_coincident_imp_df, box_regions)
    
        recoil_df = coincident_imp_df[
            [
                'x',
                'y',
                'xE',
                'timetag',
                'cathode_timetag',
                'cathode_energy',
                'anodeV_timetag',
                'anodeV_energy',
                'anodeH_timetag',
                'anodeH_energy',
            ]
        ].copy()
        recoil_df['t'] = recoil_df['timetag'] * TO_S
    
        res = correlate_events(recoil_df, decay_df, chain, pixel_search_mode)
        return res, coincident_imp_df, decay_candidates_df
    
    def correlate_events(recoil_df, decay_df, chain, pixel_mode='single'):
        """Build correlations for a single chain configuration.
    
        Parameters
        ----------
        recoil_df : pd.DataFrame
            Must contain columns ``x``, ``y``, ``xE`` and ``t``.
        decay_df : pd.DataFrame
            Decay-like events with the same columns as ``recoil_df``.
        chain : dict
            Correlation chain definition from ``correlation_config.yaml``.
        pixel_mode : {'single', 'square'}, optional
            Pixel search mode specifying whether only the recoil pixel or the
            surrounding square is searched.
        """
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
    
        # Track which events from each dataset have been used so that the same decay
        # event is not matched to multiple recoil chains.
        used_recoil_idx = set()
        used_decay_idx = set()
    
        for step in steps[1:]:
            label = step['label']
            use_recoil = step.get('ppac_required')
            dataset = recoil_df if use_recoil else decay_df
            used_set = used_recoil_idx if use_recoil else used_decay_idx
            e_min = step.get('energy_min', 0)
            e_max = step.get('energy_max', np.inf)
            dt_min = step.get('corr_min', 0)
            dt_max = step.get('corr_max', np.inf)
            results = []
            for _, row in stage_df.iterrows():
                px = row[f'{prev_label}_x']
                py = row[f'{prev_label}_y']
                if pixel_mode == 'single':
                    pixel_events = dataset[(dataset['x'] == px) & (dataset['y'] == py)]
                elif pixel_mode == 'square':
                    pixel_events = dataset[(dataset['x'].between(px-1, px+1)) & (dataset['y'].between(py-1, py+1))]
                else:
                    raise ValueError(f"Unknown pixel_mode: {pixel_mode}")
    
                # Exclude events that have already been paired with another recoil
                pixel_events = pixel_events.loc[~pixel_events.index.isin(used_set)]
    
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
                used_set.add(evt.name)
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
    
    saved_imp = None
    saved_decay = None
    for chain in correlation_chains:
        res, imp_df_chain, decay_df_chain = build_results_for_chain(chain)
        if saved_imp is None:
            saved_imp = imp_df_chain
            saved_decay = decay_df_chain
        if not res.empty:
            all_results.append(res)

    if all_results:
        final_correlated_df = pd.concat(all_results, ignore_index=True)
    else:
        final_correlated_df = pd.DataFrame()

    # Save results
    out_dir = os.path.join("correlations", args.base_dir, RUN_DIR)
    os.makedirs(out_dir, exist_ok=True)
    final_correlated_df.to_pickle(os.path.join(out_dir, "final_correlated.pkl"))
    if saved_imp is None:
        saved_imp = pd.DataFrame()
    if saved_decay is None:
        saved_decay = pd.DataFrame()
    saved_imp.to_pickle(os.path.join(out_dir, "coincident_imp.pkl"))
    saved_decay.to_pickle(os.path.join(out_dir, "decay_candidates.pkl"))


if __name__ == "__main__":
    ctx = memory_limit(args.max_memory_mb) if args.max_memory_mb else nullcontext()
    try:
        with ctx:
            main()
    except MemoryLimitExceeded as e:
        print(f"Memory limit exceeded: {e}")
        sys.exit(1)

