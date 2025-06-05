"""
sort3.py

Ultra memory-efficient processing pipeline for SHREC, PPAC, and Rutherford detector data.
This version uses advanced memory protection to prevent system kills, including:
- Extremely conservative memory limits
- Incremental merging for large datasets
- Process-level resource monitoring
- Progressive data processing with checkpointing
"""

import pandas as pd
import numpy as np
import os
import time
import gc
from datetime import datetime
import warnings
import psutil
import csv
import signal
from contextlib import contextmanager, nullcontext
import sys
import tempfile
import uuid
import json
import multiprocessing as mp

def process_file_wrapper(args):
    csv_file, output_paths, shrec_map_path, calibration_path, energy_cut, chunksize, max_memory_mb = args
    # Call your existing file processing function
    summary = process_file(
        csv_file, 
        output_paths, 
        shrec_map_path, 
        calibration_path,
        save_all_events=False,  # or True if needed
        ecut=energy_cut,
        chunksize=chunksize,
        max_memory_mb=max_memory_mb
    )
    return summary


from shrec_utils2 import (
    mapimp, 
    mapboxE, 
    mapboxW, 
    mapboxT, 
    mapboxB, 
    mapveto,
    detmerge,
    extract_ppac_data,
    extract_rutherford_data
)

# Import sortcalSHREC without calling it directly - we'll use our own memory-safe version
from shrec_utils2 import sortcalSHREC as original_sortcalSHREC

# Disable pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Default folder locations
DATA_FOLDER = '../ppac_data/'
SHREC_MAP = os.path.join(DATA_FOLDER, 'r238_shrec_map.xlsx')
SHREC_CALIBRATION = os.path.join(DATA_FOLDER, 'r238_calibration_v0_copy-from-r237.txt')
OUTPUT_FOLDER = 'processed_data/'
TEMP_FOLDER = os.path.join(OUTPUT_FOLDER, 'temp')

# Memory monitoring functions
def get_memory_usage():
    """Return the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_memory_percent():
    """Return the current memory usage as a percentage of system memory."""
    process = psutil.Process(os.getpid())
    return process.memory_percent()

def get_system_memory():
    """Return total and available system memory in MB."""
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    available_mb = mem.available / (1024 * 1024)
    return total_mb, available_mb

def check_critical_memory():
    """Check if system memory is critically low."""
    mem = psutil.virtual_memory()
    # If less than 15% memory available, system is critical
    return mem.available / mem.total < 0.15

class MemoryLimitExceeded(Exception):
    """Exception raised when memory usage exceeds the specified limit."""
    pass

@contextmanager
def memory_limit(max_mb, safety_margin_mb=500):
    """
    Context manager that enforces a memory limit with an additional safety margin.
    
    Parameters:
    -----------
    max_mb : float
        Maximum memory limit in MB
    safety_margin_mb : float
        Additional safety margin in MB to trigger preventive action
    """
    adjusted_limit = max(100, max_mb - safety_margin_mb)
    
    def memory_check():
        used_mb = get_memory_usage()
        if used_mb > adjusted_limit:
            raise MemoryLimitExceeded(f"Memory limit approaching: {used_mb:.1f}MB used, limit is {adjusted_limit}MB")
        if check_critical_memory():
            raise MemoryLimitExceeded(f"System memory critically low")
    
    # Check memory before entering context
    memory_check()
    
    # Set up signal handler for periodic checking
    original_handler = None
    
    def sigalrm_handler(signum, frame):
        memory_check()
        # Reset the alarm for next check
        signal.alarm(1)  # Check every 1 second
    
    try:
        # Set up the signal handler
        original_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
        signal.alarm(1)  # Check every 1 second
        
        yield
    finally:
        # Restore original handler
        signal.alarm(0)
        if original_handler:
            signal.signal(signal.SIGALRM, original_handler)

def adjust_chunk_size(current_chunk_size, current_memory_mb, max_memory_mb, safety_factor=0.5):
    """
    Dynamically adjust chunk size based on memory usage.
    Uses a more conservative approach to prevent OOM kills.
    """
    # If we're using more than 60% of the max memory, reduce chunk size significantly
    if current_memory_mb > (max_memory_mb * 0.6):
        new_chunk_size = int(current_chunk_size * (max_memory_mb * safety_factor / current_memory_mb))
        # Ensure chunk size is at least 500 rows
        return max(500, new_chunk_size)
    
    # If we're using less than 30% of the max memory, increase chunk size cautiously
    if current_memory_mb < (max_memory_mb * 0.3):
        new_chunk_size = int(current_chunk_size * 1.1)  # Increase by just 10%
        # Don't let it grow too large
        return min(100000, new_chunk_size)
    
    # Otherwise, keep current chunk size
    return current_chunk_size

def get_output_paths(output_folder):
    """Generate standard output file paths."""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    return {
        'dssd_clean': os.path.join(output_folder, 'dssd_non_vetoed_events.csv'),
        'ppac': os.path.join(output_folder, 'ppac_events.csv'),
        'rutherford': os.path.join(output_folder, 'rutherford_events.csv'),
        'all_events': os.path.join(output_folder, 'all_events_merged.csv'),
        'summary': os.path.join(output_folder, 'processing_summary.csv'),
        'log': os.path.join(output_folder, 'processing_log.txt'),
        'temp': TEMP_FOLDER
    }

def log_message(message, log_file=None, include_memory=False, print_to_console=True):
    """Log a message to console and optionally to a file with memory usage info."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if include_memory:
        mem_usage = get_memory_usage()
        mem_percent = get_memory_percent()
        log_line = f"[{timestamp}] {message} (Memory: {mem_usage:.1f} MB, {mem_percent:.1f}%)"
    else:
        log_line = f"[{timestamp}] {message}"
    
    if print_to_console:
        print(log_line)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')

def load_file_list(file_list_path):
    """Reads the file paths from a text file and returns a list of CSV file paths."""
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files

def count_csv_rows(file_path):
    """Count the number of rows in a CSV file without loading it into memory."""
    try:
        with open(file_path, 'r') as f:
            # Count header plus rows
            return sum(1 for _ in f)
    except Exception as e:
        log_message(f"Warning: Could not count rows in {file_path}: {str(e)}")
        return 1000000  # Return a large default value to be safe

def csv_row_generator(file_path, chunksize=50000, max_memory_mb=None):
    """Generator that yields chunks of rows from a CSV file with dynamic chunk size adjustment."""
    current_chunk_size = chunksize
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=current_chunk_size)):
        yield chunk, current_chunk_size
        
        # Explicitly delete the chunk to help garbage collection
        del chunk
        gc.collect()
        
        # Adjust chunk size if memory limit is set
        if max_memory_mb is not None:
            current_memory = get_memory_usage()
            new_chunk_size = adjust_chunk_size(current_chunk_size, current_memory, max_memory_mb)
            
            if new_chunk_size != current_chunk_size:
                log_message(f"Adjusting chunk size: {current_chunk_size} → {new_chunk_size} rows " +
                          f"(Memory: {current_memory:.1f}MB / {max_memory_mb}MB)")
                current_chunk_size = new_chunk_size

def memory_safe_sortcalSHREC(xdata, ydata, calibration_path, ecut=50, max_memory_mb=None):
    """
    Memory-safe version of sortcalSHREC that splits large datasets into manageable chunks.
    
    If input data is too large, it will:
    1. Split the data into chunks
    2. Process each chunk separately
    3. Merge the results incrementally
    """
    # Check if the dataset size is small enough to process directly
    total_rows = len(xdata) + len(ydata)
    log_message(f"Processing sortcalSHREC with {len(xdata)} X rows and {len(ydata)} Y rows")
    
    # For very small datasets, use the original function directly
    if total_rows < 100000:
        return original_sortcalSHREC(xdata, ydata, calibration_path, ecut)
    
    # For larger datasets, process in chunks
    # First, apply energy cut to reduce data size
    xdata = xdata.query('energy >= @ecut').copy()
    ydata = ydata.query('energy >= @ecut').copy()
    
    # Load calibration
    calfile = pd.read_csv(calibration_path, sep='\t')
    
    # Convert timetags to time
    xdata['t'] = np.round(xdata['timetag'] * 1e-12, 6)
    ydata['t'] = np.round(ydata['timetag'] * 1e-12, 6)
    
    # Also prepare the t2 columns used in merging
    xdata['t2'] = np.round(xdata['timetag'] * 1e-12, 5)
    ydata['t2'] = np.round(ydata['timetag'] * 1e-12, 5)
    
    # Apply calibration to all data at once
    log_message(f"Applying calibration to X data...", include_memory=True)
    xdata = xdata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    xdata['calE'] = xdata['m'] * (xdata['energy'] - xdata['b'])
    xdata.drop(['energy', 'm', 'b'], axis=1, inplace=True)
    
    log_message(f"Applying calibration to Y data...", include_memory=True)
    ydata = ydata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    ydata['calE'] = ydata['m'] * (ydata['energy'] - ydata['b'])
    ydata.drop(['energy', 'm', 'b'], axis=1, inplace=True)
    
    # Create a temp directory for storing intermediate results
    temp_dir = os.path.join(TEMP_FOLDER, f"sortcal_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    log_message(f"Using temporary directory: {temp_dir}")
    
    # Sort data by time for efficient chunking
    xdata = xdata.sort_values('t')
    ydata = ydata.sort_values('t')
    
    # Find the time range
    t_min = min(xdata['t'].min(), ydata['t'].min())
    t_max = max(xdata['t'].max(), ydata['t'].max())
    t_range = t_max - t_min
    
    # Determine how many chunks we need
    max_chunk_rows = 50000  # A conservative estimate for chunk size
    max_rows_per_timeframe = max(len(xdata), len(ydata)) / (t_range or 1)  # Avoid division by zero
    
    # Aim for manageable chunks with a time window that would capture ~max_chunk_rows
    chunk_time_window = max_chunk_rows / max_rows_per_timeframe
    
    # Ensure we have at least 5 chunks to keep memory usage reasonable
    num_chunks = max(5, int(np.ceil(t_range / chunk_time_window)))
    chunk_time_window = t_range / num_chunks
    
    log_message(f"Processing time range [{t_min:.6f}, {t_max:.6f}] in {num_chunks} chunks of {chunk_time_window:.6f}s each")
    
    # Process each time window chunk separately
    results = []
    for i in range(num_chunks):
        chunk_start = t_min + i * chunk_time_window
        chunk_end = t_min + (i + 1) * chunk_time_window
        
        # Add overlap to ensure we don't miss coincidences at boundaries
        overlap = 0.000001  # 1 microsecond
        chunk_start_with_overlap = max(t_min, chunk_start - overlap)
        chunk_end_with_overlap = min(t_max, chunk_end + overlap)
        
        log_message(f"Processing chunk {i+1}/{num_chunks}: [{chunk_start_with_overlap:.6f}, {chunk_end_with_overlap:.6f}]", 
                   include_memory=True)
        
        # Filter data for this time chunk
        xdata_chunk = xdata[(xdata['t'] >= chunk_start_with_overlap) & (xdata['t'] < chunk_end_with_overlap)].copy()
        ydata_chunk = ydata[(ydata['t'] >= chunk_start_with_overlap) & (ydata['t'] < chunk_end_with_overlap)].copy()
        
        log_message(f"Chunk {i+1} has {len(xdata_chunk)} X rows and {len(ydata_chunk)} Y rows")
        
        # Skip empty chunks
        if len(xdata_chunk) == 0 or len(ydata_chunk) == 0:
            log_message(f"Skipping empty chunk {i+1}")
            continue
        
        # Find coincidences
        dfxy1 = pd.merge(xdata_chunk, ydata_chunk, on='t', suffixes=('_x', '_y'))
        dfxy2 = pd.merge(xdata_chunk, ydata_chunk, on='t2', suffixes=('_x', '_y'))
        
        # Fix column collisions
        dfxy1 = dfxy1.drop(['t2_y'], axis=1).rename(columns={'t2_x': 't2'})
        dfxy2 = dfxy2.drop(['t_y'], axis=1).rename(columns={'t_x': 't'})
        
        # Combine, drop duplicates
        dfxy = pd.concat([dfxy1, dfxy2], ignore_index=True).drop_duplicates().drop(['t2'], axis=1)
        
        # Free up memory
        del dfxy1, dfxy2
        gc.collect()
        
        # Rename columns as needed
        dfxy.columns = [
            'xboard','xchan','tagx','xflag','nfile','xid','x','t','xstripE',
            'yboard','ychan','tagy','yflag','nfiley','yid','y','ystripE'
        ]
        
        # Keep only relevant columns
        dfxy = dfxy[['t','x','y','xstripE','ystripE','tagx','tagy','nfile', 'xboard', 'yboard']]
        
        # Time difference cut
        dfxy['tdelta'] = dfxy['tagx'] - dfxy['tagy']
        dfxy = dfxy.loc[dfxy['tdelta'].abs() < 400000]
        
        # Skip empty results
        if len(dfxy) == 0:
            log_message(f"No coincidences found in chunk {i+1}")
            continue
        
        # Multiplicities
        dfxy['nX'] = dfxy.groupby(['t','x'])['t'].transform('count')
        dfxy['nY'] = dfxy.groupby(['t','y'])['t'].transform('count')
        
        # x / y differences
        dfxy['xdiff'] = dfxy.groupby('t')['x'].transform('max') - dfxy.groupby('t')['x'].transform('min')
        dfxy['ydiff'] = dfxy.groupby('t')['y'].transform('max') - dfxy.groupby('t')['y'].transform('min')
        
        # Summed energies
        dfxy['xE'] = dfxy.groupby(['t','nX'])['xstripE'].transform('sum') / dfxy['nX']
        dfxy['yE'] = dfxy.groupby(['t','nY'])['ystripE'].transform('sum') / dfxy['nY']
        
        # Filter on x/y differences
        temp3 = dfxy.loc[(dfxy['xdiff'] < 2) & (dfxy['ydiff'] < 2)]
        temp3.sort_values(by=['t','xstripE','ystripE'], ascending=[True,False,False], inplace=True)
        
        temp4 = temp3.drop(['xstripE','ystripE','xdiff','ydiff'], axis=1).reset_index(drop=True)
        
        # Save to a temporary file to free memory
        temp_file = os.path.join(temp_dir, f"chunk_{i}.csv")
        temp4.to_csv(temp_file, index=False)
        
        # Free memory
        del dfxy, temp3, temp4, xdata_chunk, ydata_chunk
        gc.collect()
        
        # Log progress
        log_message(f"Completed chunk {i+1}/{num_chunks} - saved to {temp_file}", include_memory=True)
    
    # Free original data
    del xdata, ydata
    gc.collect()
    
    # Now merge all the temp files
    temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("chunk_")]
    
    if not temp_files:
        log_message("Warning: No coincidences found in any chunk")
        return pd.DataFrame()
    
    log_message(f"Merging {len(temp_files)} chunk results", include_memory=True)
    
    # Read all CSV files and combine them
    dfs = []
    for temp_file in temp_files:
        df_chunk = pd.read_csv(temp_file)
        dfs.append(df_chunk)
        # Delete file after reading to free disk space
        os.remove(temp_file)
    
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.sort_values(by='t', inplace=True, ignore_index=True)
    
    # Clean up
    os.rmdir(temp_dir)
    
    # Print duration info
    if len(final_df) > 0:
        duration = final_df['t'].max() - final_df['t'].min()
        log_message(f"duration = {duration:.1f} s = {duration/3600:.2f} hr")
    
    return final_df

def process_shrec_data_chunked(csv_file, shrec_map_path, calibration_path, ecut=50, 
                           chunksize=50000, max_memory_mb=None):
    """Memory-efficient version of process_shrec_data that processes a file in chunks."""
    # Read the SHREC map once (dictionary of DataFrames)
    shrec_map = pd.read_excel(shrec_map_path, sheet_name=None)
    
    # Get an estimate of the file size
    row_count = count_csv_rows(csv_file) - 1  # Subtract header
    log_message(f"Processing {csv_file} with approximately {row_count} rows...")
    
    # Initialize containers for each detector type
    detector_data = {
        'impx': [], 'impy': [],
        'bex': [], 'bey': [],
        'bwx': [], 'bwy': [],
        'btx': [], 'bty': [],
        'bbx': [], 'bby': [],
        'vetox': [], 'vetoy': []
    }
    
    # Process the file in chunks
    chunk_count = 0
    results = {}
    
    # Use memory limit context manager if a limit is provided
    context = memory_limit(max_memory_mb) if max_memory_mb else nullcontext()
    
    try:
        with context:
            for chunk, used_chunk_size in csv_row_generator(csv_file, chunksize, max_memory_mb):
                chunk_count += 1
                log_message(f"Processing chunk {chunk_count} ({len(chunk)} rows, chunksize={used_chunk_size})...", 
                           include_memory=True)
                
                # Map each detector type for this chunk
                impx_chunk, impy_chunk = mapimp(chunk, shrec_map)
                bex_chunk, bey_chunk = mapboxE(chunk, shrec_map)
                bwx_chunk, bwy_chunk = mapboxW(chunk, shrec_map)
                btx_chunk, bty_chunk = mapboxT(chunk, shrec_map)
                bbx_chunk, bby_chunk = mapboxB(chunk, shrec_map)
                vetox_chunk, vetoy_chunk = mapveto(chunk, shrec_map)
                
                # Store results for each detector type
                detector_data['impx'].append(impx_chunk)
                detector_data['impy'].append(impy_chunk)
                detector_data['bex'].append(bex_chunk)
                detector_data['bey'].append(bey_chunk)
                detector_data['bwx'].append(bwx_chunk)
                detector_data['bwy'].append(bwy_chunk)
                detector_data['btx'].append(btx_chunk)
                detector_data['bty'].append(bty_chunk)
                detector_data['bbx'].append(bbx_chunk)
                detector_data['bby'].append(bby_chunk)
                detector_data['vetox'].append(vetox_chunk)
                detector_data['vetoy'].append(vetoy_chunk)
                
                # Delete chunk data to free memory
                del chunk, impx_chunk, impy_chunk, bex_chunk, bey_chunk
                del bwx_chunk, bwy_chunk, btx_chunk, bty_chunk, bbx_chunk, bby_chunk
                del vetox_chunk, vetoy_chunk
                gc.collect()
    except Exception as e:
        log_message(f"Error in chunk processing: {str(e)}")
        # Continue with what we have so far
    
    # Helper function to merge and process a detector type
    def merge_and_process(x_key, y_key, result_key):
        if detector_data[x_key] and detector_data[y_key]:
            try:
                # Merge all chunks for X and Y
                log_message(f"Merging {len(detector_data[x_key])} X chunks and {len(detector_data[y_key])} Y chunks for {result_key}...", 
                          include_memory=True)
                
                x_merged = pd.concat(detector_data[x_key], ignore_index=True)
                y_merged = pd.concat(detector_data[y_key], ignore_index=True)
                
                # Apply sortcalSHREC to get the clean data
                log_message(f"Running sortcalSHREC for {result_key}...", include_memory=True)
                results[result_key] = memory_safe_sortcalSHREC(
                    x_merged, y_merged, calibration_path, ecut=ecut, max_memory_mb=max_memory_mb
                )
                
                # Free memory immediately
                del detector_data[x_key], detector_data[y_key], x_merged, y_merged
                gc.collect()
                
            except Exception as e:
                log_message(f"Error processing {result_key}: {str(e)}")
                results[result_key] = pd.DataFrame()
        else:
            results[result_key] = pd.DataFrame()
    
    # Process each detector region - the order is important! Process largest regions last
    # Start with smaller regions
    merge_and_process('vetox', 'vetoy', 'veto')
    merge_and_process('btx', 'bty', 'boxT')
    merge_and_process('bbx', 'bby', 'boxB')
    merge_and_process('bwx', 'bwy', 'boxW')
    merge_and_process('bex', 'bey', 'boxE')
    
    # Process the largest region (imp) last to minimize memory pressure
    merge_and_process('impx', 'impy', 'imp')
    
    # Clean up remaining temporary data
    for key in list(detector_data.keys()):
        if key in detector_data:
            del detector_data[key]
    gc.collect()
    
    return results

def extract_ancillary_data_chunked(csv_file, chunksize=50000, max_memory_mb=None):
    """Extract PPAC and Rutherford detector data in a memory-efficient way."""
    ppac_chunks = []
    ruth_chunks = []
    
    # Use memory limit context manager if a limit is provided
    context = memory_limit(max_memory_mb) if max_memory_mb else nullcontext()
    
    try:
        with context:
            # Process the file in chunks
            for chunk, used_chunk_size in csv_row_generator(csv_file, chunksize, max_memory_mb):
                # Extract PPAC and Rutherford data from this chunk
                ppac_chunk = extract_ppac_data(chunk)
                ruth_chunk = extract_rutherford_data(chunk)
                
                # Store non-empty results
                if len(ppac_chunk) > 0:
                    ppac_chunks.append(ppac_chunk)
                if len(ruth_chunk) > 0:
                    ruth_chunks.append(ruth_chunk)
                
                # Cleanup
                del chunk, ppac_chunk, ruth_chunk
                gc.collect()
    except Exception as e:
        log_message(f"Error in ancillary data extraction: {str(e)}")
        # Continue with what we have so far
    
    # Merge results if any chunks were found
    ppac_data = pd.concat(ppac_chunks, ignore_index=True) if ppac_chunks else pd.DataFrame()
    ruth_data = pd.concat(ruth_chunks, ignore_index=True) if ruth_chunks else pd.DataFrame()
    
    # Final cleanup
    del ppac_chunks, ruth_chunks
    gc.collect()
    
    return ppac_data, ruth_data

def write_dataframe_to_csv(df, output_path, mode='w'):
    """Write DataFrame to CSV with proper header handling."""
    write_header = mode == 'w' or not os.path.exists(output_path)
    
    if len(df) > 0:
        df.to_csv(output_path, mode=mode, header=write_header, index=False)
        return len(df)
    return 0

def process_file(csv_file, output_paths, shrec_map_path, calibration_path, 
                save_all_events=False, ecut=50, chunksize=50000, max_memory_mb=None):
    """Process a single CSV file and extract all detector data with memory efficiency."""
    log_message(f"Processing {csv_file}...", output_paths['log'], include_memory=True)
    t_start = time.time()
    
    try:
        # 1. Extract ancillary data (PPAC and Rutherford) in a memory-efficient way
        log_message(f"Extracting PPAC and Rutherford data...", output_paths['log'])
        ppac_data, ruth_data = extract_ancillary_data_chunked(
            csv_file, 
            chunksize=chunksize,
            max_memory_mb=max_memory_mb
        )
        
        log_message(f"Found {len(ppac_data)} PPAC events and {len(ruth_data)} Rutherford events", 
                   output_paths['log'], include_memory=True)
        
        # 2. Process SHREC data in chunks
        log_message(f"Processing SHREC data in chunks...", output_paths['log'])
        shrec_results = process_shrec_data_chunked(
            csv_file, 
            shrec_map_path, 
            calibration_path, 
            ecut=ecut,
            chunksize=chunksize,
            max_memory_mb=max_memory_mb
        )
        
        # Prepare for summary statistics
        summary = {
            'filename': os.path.basename(csv_file),
            'ppac_events': len(ppac_data),
            'ruth_events': len(ruth_data)
        }
        
        total_events = len(ppac_data) + len(ruth_data)
        
        # 3. Extract veto events and apply detmerge incrementally
        veto_events = shrec_results.get('veto', pd.DataFrame())
        summary['veto_events'] = len(veto_events)
        total_events += len(veto_events)
        
        # 4. Process each DSSD region, save to disk, and clear memory
        dssd_regions = ['imp', 'boxE', 'boxW', 'boxT', 'boxB']
        clean_dssd_counts = 0
        all_dssd_counts = 0
        
        for region in dssd_regions:
            if region in shrec_results and len(shrec_results[region]) > 0:
                log_message(f"Processing {region} events ({len(shrec_results[region])} events)...", 
                           output_paths['log'], include_memory=True)
                
                # Apply detmerge to find veto coincidences
                region_data = detmerge(shrec_results[region], veto_events)
                
                # Count events
                region_count = len(region_data)
                total_events += region_count
                summary[f'{region}_events'] = region_count
                
                # Count vetoed events
                if 'is_vetoed' in region_data.columns:
                    vetoed_count = region_data['is_vetoed'].sum()
                    summary[f'{region}_vetoed'] = vetoed_count
                    summary[f'{region}_clean'] = region_count - vetoed_count
                    log_message(f"  {vetoed_count} of {region_count} {region} events are vetoed", 
                               output_paths['log'])
                    # Save all events if requested - do this incrementally to avoid memory pressure
                if save_all_events:
                    # Add event_type column
                    region_data['event_type'] = region
                    
                    # Append to all events file
                    write_mode = 'a' if os.path.exists(output_paths['all_events']) else 'w'
                    written = write_dataframe_to_csv(region_data, output_paths['all_events'], mode=write_mode)
                    all_dssd_counts += written
                
                # Filter for clean events and save
                if 'is_vetoed' in region_data.columns:
                    clean_events = region_data[~region_data['is_vetoed']].copy()
                    clean_events['event_type'] = region
                    clean_events = clean_events.drop('is_vetoed', axis=1)
                    
                    # Append to clean events file
                    write_mode = 'a' if os.path.exists(output_paths['dssd_clean']) else 'w'
                    written = write_dataframe_to_csv(clean_events, output_paths['dssd_clean'], mode=write_mode)
                    clean_dssd_counts += written
                    
                    # Free memory
                    del clean_events
                    gc.collect()
                
                # Free memory for this region
                del region_data
                del shrec_results[region]  # Remove from dictionary to free up memory
                gc.collect()
        
        # 5. Save PPAC and Rutherford data
        if len(ppac_data) > 0:
            write_mode = 'a' if os.path.exists(output_paths['ppac']) else 'w'
            write_dataframe_to_csv(ppac_data, output_paths['ppac'], mode=write_mode)
            # Free memory immediately
            del ppac_data
            gc.collect()
        
        if len(ruth_data) > 0:
            write_mode = 'a' if os.path.exists(output_paths['rutherford']) else 'w'
            write_dataframe_to_csv(ruth_data, output_paths['rutherford'], mode=write_mode)
            # Free memory immediately
            del ruth_data
            gc.collect()
        
        # 6. Free memory for remaining large objects
        if 'veto_events' in locals():
            del veto_events
        if 'shrec_results' in locals():
            del shrec_results
        gc.collect()
        
        # 7. Add processing stats to summary
        summary['total_events'] = total_events
        summary['clean_dssd_saved'] = clean_dssd_counts
        if save_all_events:
            summary['all_dssd_saved'] = all_dssd_counts
        
        # Add processing duration
        t_end = time.time()
        summary['processing_time_seconds'] = t_end - t_start
        
        # 8. Append to summary file
        summary_df = pd.DataFrame([summary])
        write_mode = 'a' if os.path.exists(output_paths['summary']) else 'w'
        write_dataframe_to_csv(summary_df, output_paths['summary'], mode=write_mode)
        
        log_message(f"Processing complete for {csv_file}. Time: {t_end - t_start:.1f} seconds", 
                   output_paths['log'], include_memory=True)
        return summary
        
    except Exception as e:
        log_message(f"Error processing {csv_file}: {str(e)}", output_paths['log'])
        return {
            'filename': os.path.basename(csv_file),
            'total_events': 0,
            'error': str(e)
        }

def main():
    """Main function to process all files with memory optimization."""
    # Configuration variables
    file_list_path = 'files_to_sort.txt'
    data_folder = '../ppac_data/'
    shrec_map_path = os.path.join(data_folder, 'r238_shrec_map.xlsx')
    calibration_path = os.path.join(data_folder, 'r238_calibration_v0_copy-from-r237.txt')
    output_folder = 'processed_data/'
    energy_cut = 50
    save_all_events = False
    clear_outputs = True
    chunksize = 20000  # More conservative initial chunk size
    
    # Memory limit settings
    # By default, use 50% of available system memory to be ultra conservative
    total_memory, available_memory = get_system_memory()
    max_memory_mb = int(available_memory * 0.5)  # Use only 50% of available memory
    # Uncomment and set a specific value to override
    # max_memory_mb = 2000  # Limit to 2GB of RAM
    
    # Set up output paths
    output_paths = get_output_paths(output_folder)
    
    # Clear existing output files if requested
    if clear_outputs:
        for path in output_paths.values():
            if path != output_paths['temp'] and os.path.exists(path):
                os.remove(path)
                log_message(f"Removed existing file: {path}")
    
    # Initialize log file
    log_message(f"Starting ultra memory-optimized SHREC data processing", output_paths['log'], include_memory=True)
    log_message(f"Initial memory usage: {get_memory_usage():.1f} MB", output_paths['log'])
    log_message(f"Memory limit set to: {max_memory_mb:.1f} MB (of {total_memory:.1f} MB total system memory)", 
              output_paths['log'])
    
    # Load list of files to process
    try:
        csv_files = load_file_list(file_list_path)
        log_message(f"Found {len(csv_files)} files to process", output_paths['log'])
    except Exception as e:
        log_message(f"Error loading file list: {str(e)}", output_paths['log'])
        return
    
    # Process each file
    total_events = 0
    
    for i, csv_file in enumerate(csv_files):
        log_message(f"\nProcessing file {i+1}/{len(csv_files)}: {csv_file}", 
                   output_paths['log'], include_memory=True)
        
        try:
            # Force garbage collection before processing each file
            gc.collect()
            
            # Prepare a checkpoint file to track processing status
            checkpoint_file = os.path.join(output_paths['temp'], f"{os.path.basename(csv_file)}.checkpoint")
            
            # Check if this file was previously processed
            if os.path.exists(checkpoint_file):
                log_message(f"File {csv_file} was already processed (checkpoint found)", output_paths['log'])
                
                # Read the summary from the checkpoint
                with open(checkpoint_file, 'r') as f:
                    import json
                    try:
                        summary = json.load(f)
                    except:
                        summary = {'filename': os.path.basename(csv_file), 'total_events': 0}
            else:
                # Process this file
                summary = process_file(
                    csv_file, 
                    output_paths,
                    shrec_map_path,
                    calibration_path,
                    save_all_events=save_all_events,
                    ecut=energy_cut,
                    chunksize=chunksize,
                    max_memory_mb=max_memory_mb
                )
                
                # Save checkpoint to mark as processed
                with open(checkpoint_file, 'w') as f:
                    import json
                    json.dump(summary, f)
            
            # Update statistics
            if 'total_events' in summary:
                total_events += summary['total_events']
            
            # Print progress
            log_message(f"Progress: {i+1}/{len(csv_files)} files processed", output_paths['log'])
            log_message(f"Running total: {total_events} events", output_paths['log'], include_memory=True)
            
            # Force garbage collection again
            gc.collect()
            
        except MemoryLimitExceeded as e:
            log_message(f"Memory limit exceeded while processing {csv_file}: {str(e)}", output_paths['log'])
            log_message("Reducing chunk size and retrying with more aggressive memory management...", output_paths['log'])
            
            # Try again with a smaller chunk size
            try:
                # Force garbage collection
                gc.collect()
                
                # Reduce chunk size to half
                smaller_chunksize = max(500, chunksize // 2)
                log_message(f"Retrying with smaller chunk size: {smaller_chunksize}", output_paths['log'])
                
                summary = process_file(
                    csv_file, 
                    output_paths,
                    shrec_map_path,
                    calibration_path,
                    save_all_events=save_all_events,
                    ecut=energy_cut,
                    chunksize=smaller_chunksize,
                    max_memory_mb=max_memory_mb
                )
                
                # Save checkpoint to mark as processed
                checkpoint_file = os.path.join(output_paths['temp'], f"{os.path.basename(csv_file)}.checkpoint")
                with open(checkpoint_file, 'w') as f:
                    json.dump(summary, f)
                
                # If successful, update the chunk size for future files
                chunksize = smaller_chunksize
                
            except Exception as retry_e:
                log_message(f"Failed retry for {csv_file}: {str(retry_e)}", output_paths['log'])
            
        except Exception as e:
            log_message(f"Error processing file {csv_file}: {str(e)}", output_paths['log'])
    
    # Print final summary
    log_message("\nProcessing complete!", output_paths['log'])
    log_message(f"Total files processed: {len(csv_files)}", output_paths['log'])
    log_message(f"Total events processed: {total_events}", output_paths['log'])
    log_message(f"Final memory usage: {get_memory_usage():.1f} MB", output_paths['log'])
    
    # Try to read file size information
    for name, path in output_paths.items():
        if name != 'log' and name != 'temp' and os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                log_message(f"{name} file size: {file_size_mb:.1f} MB", output_paths['log'])
            except:
                pass

if __name__ == "__main__":
    # Set up global exception handler to avoid crashes
    def handle_exception(exc_type, exc_value, exc_traceback):
        print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
        import traceback
        print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        
        # Force process to exit cleanly without kernel killing it
        sys.exit(1)
    
    sys.excepthook = handle_exception
    
    try:
        main()
    except Exception as e:
        print(f"Critical error in main function: {str(e)}")
        # Clean exit without kernel kill
        sys.exit(1)

# import multiprocessing as mp

# # Add this wrapper function before main()
# def process_file_wrapper(args):
#     """Wrapper function to unpack arguments for process_file when using multiprocessing."""
#     try:
#         csv_file, output_paths, shrec_map_path, calibration_path, energy_cut, chunksize, max_memory_mb = args
        
#         # Call the existing process_file function with unpacked arguments
#         return process_file(
#             csv_file=csv_file,
#             output_paths=output_paths,
#             shrec_map_path=shrec_map_path,
#             calibration_path=calibration_path,
#             save_all_events=False,  # Default value
#             ecut=energy_cut,
#             chunksize=chunksize,
#             max_memory_mb=max_memory_mb
#         )
#     except Exception as e:
#         # Handle any exceptions in the child process
#         import traceback
#         return {
#             'filename': os.path.basename(csv_file) if 'csv_file' in locals() else 'unknown',
#             'error': str(e),
#             'traceback': traceback.format_exc()
#         }

# def main():
#     # Your existing configuration and output path setup
#     file_list_path = 'files_to_sort.txt'
#     data_folder = '../ppac_data/'
#     shrec_map_path = os.path.join(data_folder, 'r238_shrec_map.xlsx')
#     calibration_path = os.path.join(data_folder, 'r238_calibration_v0_copy-from-r237.txt')
#     output_folder = 'processed_data/'
#     energy_cut = 50
#     chunksize = 5000  # Use a smaller chunk size for safer processing
    
#     # Memory limit settings (e.g., 60% of available system memory per process)
#     total_memory, available_memory = get_system_memory()
#     max_memory_mb = int(available_memory * 0.6)
    
#     # Ensure output directories exist
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(os.path.join(output_folder, 'temp'), exist_ok=True)
    
#     # Set up output paths and load file list
#     output_paths = get_output_paths(output_folder)
#     try:
#         csv_files = load_file_list(file_list_path)
#         log_message(f"Found {len(csv_files)} files to process", output_paths['log'])
#     except Exception as e:
#         log_message(f"Error loading file list: {str(e)}", output_paths['log'])
#         return
    
#     # Create a list of arguments for each file
#     tasks = [
#         (csv_file, output_paths, shrec_map_path, calibration_path, energy_cut, chunksize, max_memory_mb)
#         for csv_file in csv_files
#     ]
    
#     # Use a multiprocessing Pool
#     # To be extra cautious with memory, using processes=1 to run one file at a time
#     log_message(f"Starting processing with multiprocessing (1 process at a time)", output_paths['log'])
#     log_message(f"Memory limit per process: {max_memory_mb}MB", output_paths['log'])
    
#     with mp.Pool(processes=1) as pool:
#         summaries = pool.map(process_file_wrapper, tasks)
    
#     # Process the collected summaries
#     total_events = 0
#     for summary in summaries:
#         if 'error' in summary:
#             log_message(f"Error processing {summary.get('filename', 'unknown')}: {summary['error']}", 
#                        output_paths['log'])
#         elif 'total_events' in summary:
#             total_events += summary['total_events']
    
#     # Print final summary
#     log_message("\nProcessing complete!", output_paths['log'])
#     log_message(f"Total files processed: {len(csv_files)}", output_paths['log'])
#     log_message(f"Total events processed: {total_events}", output_paths['log'])
    
# if __name__ == "__main__":
#     # Force process to start fresh with released memory
#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn')
#     main()