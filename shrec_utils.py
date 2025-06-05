"""
shrec_utils.py

Comprehensive utility functions for processing SHREC experimental data.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def mapimp(dataframe, shrec_map):
    """
    Maps 'IMP' events in the DSSD (front vs back strips).
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Raw data containing columns ['board', 'channel', ...]
    shrec_map : dict of pd.DataFrames
        Dictionary of DataFrames (as loaded by pd.read_excel(..., sheet_name=None))

    Returns:
    --------
    impx : pd.DataFrame
        Front side events mapped to X strips.
    impy : pd.DataFrame
        Back side events mapped to Y strips.
    """
    impx = dataframe[((dataframe['board'] <=5) & (dataframe['channel'] <=31))]
    impy = dataframe[((dataframe['board'] == 6) | (dataframe['board'] == 7)) & (dataframe['channel'] <=31)]
    
    xmap = shrec_map['IMP X']
    ymap = shrec_map['IMP Y']
    
    cols = ['board', 'channel']
    impx = impx.join(xmap.set_index(cols), on=cols)
    impy = impy.join(ymap.set_index(cols), on=cols)
    
    return impx, impy


def mapboxE(dataframe, shrec_map):
    """
    Example function for mapping Box E. 
    This can remain unused if you don't need it yet, but it's stored here.
    """
    boxEx = dataframe[(dataframe['board'] == 6) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    boxEy = dataframe[(dataframe['board'] == 4) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['BOXE X']
    ymap = shrec_map['BOXE Y']
    
    cols = ['board', 'channel']
    boxEx = boxEx.join(xmap.set_index(cols), on=cols)
    boxEy = boxEy.join(ymap.set_index(cols), on=cols)
    
    return boxEx, boxEy


def mapboxW(dataframe, shrec_map):
    """
    Example function for mapping Box W.
    """
    boxWx = dataframe[(dataframe['board'] == 7) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    boxWy = dataframe[(dataframe['board'] == 5) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['BOXW X']
    ymap = shrec_map['BOXW Y']
    
    cols = ['board', 'channel']
    boxWx = boxWx.join(xmap.set_index(cols), on=cols)
    boxWy = boxWy.join(ymap.set_index(cols), on=cols)
    
    return boxWx, boxWy


def mapboxT(dataframe, shrec_map):
    """
    Example function for mapping Box T.
    """
    boxTx = dataframe[((dataframe['board'] == 8) & (dataframe['channel'] >= 32)) |
                      ((dataframe['board'] == 5) & (dataframe['channel'] >= 48))]
    boxTy = dataframe[(dataframe['board'] == 7) & (dataframe['channel'] >= 48)]
    
    xmap = shrec_map['BOXT X']
    ymap = shrec_map['BOXT Y']
    
    cols = ['board', 'channel']
    boxTx = boxTx.join(xmap.set_index(cols), on=cols)
    boxTy = boxTy.join(ymap.set_index(cols), on=cols)
    
    return boxTx, boxTy


def mapboxB(dataframe, shrec_map):
    """
    Example function for mapping Box B.
    """
    boxBx = dataframe[((dataframe['board'] == 8) & (dataframe['channel'] <=31)) |
                      ((dataframe['board'] == 4) & (dataframe['channel'] >=48))]
    boxBy = dataframe[(dataframe['board'] == 6) & (dataframe['channel'] >=48)]
    
    xmap = shrec_map['BOXB X']
    ymap = shrec_map['BOXB Y']
    
    cols = ['board', 'channel']
    boxBx = boxBx.join(xmap.set_index(cols), on=cols)
    boxBy = boxBy.join(ymap.set_index(cols), on=cols)
    
    return boxBx, boxBy


def mapveto(dataframe, shrec_map):
    """
    Example function for mapping VETO.
    """
    vetox = dataframe[(dataframe['board'] <= 2) & (dataframe['channel'] >= 32)]
    vetoy = dataframe[(dataframe['board'] == 3) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['VETO X']
    ymap = shrec_map['VETO Y']
    
    cols = ['board', 'channel']
    vetox = vetox.join(xmap.set_index(cols), on=cols)
    vetoy = vetoy.join(ymap.set_index(cols), on=cols)
    
    return vetox, vetoy

def sortcalSHREC(xdata, ydata, calibration_path, ecut=50):
    """
    Finds coincidences between the front and back strips within a detector,
    applies the energy calibration, and returns a cleaned DataFrame.

    Parameters:
    -----------
    xdata, ydata : pd.DataFrame
        The front and back data subsets for a particular detector region (e.g., IMP X / IMP Y).
    calibration_path : str
        Path to the SHREC calibration file
    ecut : float
        Energy cut to remove low-energy / noisy events.

    Returns:
    --------
    pd.DataFrame
        The cleaned and time-sorted DataFrame for this region.
    """
    # Time conversion using numpy array
    xdata['t'] = np.array(xdata['timetag']*1e-12).round(6)
    ydata['t'] = np.array(ydata['timetag']*1e-12).round(6)
    
    xdata['t2'] = np.array(xdata['timetag']*1e-12).round(5)
    ydata['t2'] = np.array(ydata['timetag']*1e-12).round(5)
    
    # Load calibration file
    calfile = pd.read_csv(calibration_path, sep='\t')
    
    # Apply energy cut
    xdata = xdata.query('energy >= @ecut')
    ydata = ydata.query('energy >= @ecut')
    
    # Join calibration columns
    xdata = xdata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    ydata = ydata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    
    # Compute calibrated energy
    xdata['calE'] = xdata['m'] * (xdata['energy'] - xdata['b'])
    ydata['calE'] = ydata['m'] * (ydata['energy'] - ydata['b'])
    
    # Drop unneeded columns
    xdata = xdata.drop(['energy', 'm', 'b'], axis=1)
    ydata = ydata.drop(['energy', 'm', 'b'], axis=1)
    
    # Merge on time and t2
    dfxy1 = xdata.merge(ydata, on=['t'])
    dfxy2 = xdata.merge(ydata, on=['t2'])
    
    # Fix column collisions
    dfxy1 = dfxy1.drop(['t2_y'], axis=1).rename(columns={'t2_x': 't2'})
    dfxy2 = dfxy2.drop(['t_y'], axis=1).rename(columns={'t_x': 't'})
    
    # Combine, drop duplicates
    dfxy = pd.concat([dfxy1, dfxy2]).drop_duplicates(ignore_index=True).drop(['t2'], axis=1).reset_index(drop=True)
    
    # Rename columns as needed
    dfxy.columns = [
        'xboard', 'xchan', 'tagx', 'xflag', 'nfile', 'xid', 'x', 't', 'xstripE', 
        'yboard', 'ychan', 'tagy', 'yflag', 'nfiley', 'yid', 'y', 'ystripE'
    ]
    
    # Keep only relevant columns
    dfxy = dfxy[['t', 'x', 'y', 'xstripE', 'ystripE', 'tagx', 'tagy', 'nfile', 'xboard', 'yboard']]
    
    # Time difference cut
    dfxy['tdelta'] = dfxy['tagx'] - dfxy['tagy']
    dfxy = dfxy.loc[abs(dfxy['tdelta']) < 400000]
    
    # Multiplicities
    dfxy['nX'] = dfxy.groupby(['t', 'x'])['t'].transform('count')
    dfxy['nY'] = dfxy.groupby(['t', 'y'])['t'].transform('count')
    
    # x / y differences
    dfxy['xdiff'] = dfxy.groupby(['t'])['x'].transform('max') - dfxy.groupby(['t'])['x'].transform('min')
    dfxy['ydiff'] = dfxy.groupby(['t'])['y'].transform('max') - dfxy.groupby(['t'])['y'].transform('min')
    
    # Summed energies
    dfxy['xE'] = dfxy.groupby(['t', 'nX'])['xstripE'].transform('sum') / dfxy['nX']
    dfxy['yE'] = dfxy.groupby(['t', 'nY'])['ystripE'].transform('sum') / dfxy['nY']
    
    # Filter on x/y differences
    temp3 = dfxy.loc[(dfxy['xdiff'].values < 2) & (dfxy['ydiff'].values < 2)]
    temp3 = temp3.sort_values(by=['t', 'xstripE', 'ystripE'], ascending=[True, False, False])
    
    # Drop unnecessary columns and reset index
    temp4 = temp3.drop(['xstripE', 'ystripE', 'xdiff', 'ydiff'], axis=1).reset_index(drop=True)
    
    # Calculate and print duration
    duration = max(temp4['t'].values) - min(temp4['t'].values)
    print("duration = %.1f s = %.2f hr" % (duration, duration/3600))
    
    return temp4

def process_shrec_data(csv_file, shrec_map_path, calibration_path, ecut=50):
    """
    Reads data, applies the SHREC mapping to all regions (IMP, Box E, etc.), then sorts
    and calibrates each region. Returns a dictionary of cleaned DataFrames.
    """
    # Read the raw data
    df = pd.read_csv(csv_file)

    # Read the SHREC map (dictionary of DataFrames)
    shrec_map = pd.read_excel(shrec_map_path, sheet_name=None)
    print(f"Processing {csv_file}... found {len(df)} raw events.")

    # 1) MAP all regions
    impx, impy   = mapimp(df, shrec_map)
    bex, bey     = mapboxE(df, shrec_map)
    bwx, bwy     = mapboxW(df, shrec_map)
    btx, bty     = mapboxT(df, shrec_map)
    bbx, bby     = mapboxB(df, shrec_map)
    vetox, vetoy = mapveto(df, shrec_map)

    # 2) Sort + calibrate each region using sortcalSHREC
    imp_clean   = sortcalSHREC(impx, impy,  calibration_path, ecut=ecut)
    boxE_clean  = sortcalSHREC(bex, bey,   calibration_path, ecut=ecut)
    boxW_clean  = sortcalSHREC(bwx, bwy,   calibration_path, ecut=ecut)
    boxT_clean  = sortcalSHREC(btx, bty,   calibration_path, ecut=ecut)
    boxB_clean  = sortcalSHREC(bbx, bby,   calibration_path, ecut=ecut)
    veto_clean  = sortcalSHREC(vetox, vetoy, calibration_path, ecut=ecut)

    # 3) Return them in a dict
    data = {
        'imp':   imp_clean,
        'boxE':  boxE_clean,
        'boxW':  boxW_clean,
        'boxT':  boxT_clean,
        'boxB':  boxB_clean,
        'veto':  veto_clean
    }

    return data

def extract_ppac_data(raw_df):
    """Extract PPAC detector data from raw dataframe."""
    try:
        # Define PPAC channels

        cathode = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 10)].copy()
        anodev = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 12)].copy()
        anodeh = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 8)].copy()
        
        # Add detector identifier
        if len(cathode) > 0:
            cathode['detector'] = 'cathode'
        if len(anodev) > 0:
            anodev['detector'] = 'anodeV'
        if len(anodeh) > 0:
            anodeh['detector'] = 'anodeH'
        
        # Combine all PPAC data
        ppac_data = pd.concat([cathode, anodev, anodeh], ignore_index=True)
        
        # Convert timetag to seconds
        if len(ppac_data) > 0:
            ppac_data['t'] = np.round(ppac_data['timetag'] * 1e-12, 6)
            
            # Sort by time
            ppac_data = ppac_data.sort_values(by='t').reset_index(drop=True)
            
            # Select columns for output
            ppac_data = ppac_data[['t', 'energy', 'board', 'channel', 'detector', 'timetag', 'nfile']]
            
        return ppac_data
        
    except Exception as e:
        print(f"Error extracting PPAC data: {str(e)}")
    return pd.DataFrame()

def extract_rutherford_data(raw_df):
    """Extract Rutherford detector data from raw dataframe."""
    try:
        # Define Rutherford detector channels
        ruthE = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 14)].copy()
        ruthW = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 15)].copy()
        
        # Add detector identifier
        if len(ruthE) > 0:
            ruthE['detector'] = 'ruthE'
        if len(ruthW) > 0:
            ruthW['detector'] = 'ruthW'
        
        # Combine all Rutherford data
        ruth_data = pd.concat([ruthE, ruthW], ignore_index=True)
        
        # Convert timetag to seconds
        if len(ruth_data) > 0:
            ruth_data['t'] = np.round(ruth_data['timetag'] * 1e-12, 6)
            
            # Sort by time
            ruth_data = ruth_data.sort_values(by='t').reset_index(drop=True)
            
            # Select columns for output
            ruth_data = ruth_data[['t', 'energy', 'board', 'channel', 'detector', 'timetag', 'nfile']]
            
        return ruth_data
        
    except Exception as e:
        print(f"Error extracting Rutherford data: {str(e)}")
        return pd.DataFrame()

def detmerge(dssd_events, veto_events, time_window=400000e-12):
    """
    Find coincidences between DSSD and veto events.
    """
    try:
        # If either dataframe is empty, return dssd_events with is_vetoed=False
        if len(dssd_events) == 0:
            return dssd_events
        if len(veto_events) == 0:
            dssd_events['is_vetoed'] = False
            return dssd_events
            
        # Convert timetags to time if not already done
        if 't' not in dssd_events.columns:
            dssd_events['t'] = np.round(dssd_events['tagx'] * 1e-12, 6)
        if 't' not in veto_events.columns:
            veto_events['t'] = np.round(veto_events['tagx'] * 1e-12, 6)
        
        # Create rounded time columns for merging
        dssd_events['t_round'] = np.round(dssd_events['t'], 5)
        veto_events['t_round'] = np.round(veto_events['t'], 5)
        
        # Find exact time matches
        merge_exact = pd.merge(
            dssd_events[['t', 'tagx', 't_round']], 
            veto_events[['t', 'tagx', 't_round']].rename(columns={'t': 't_veto', 'tagx': 'tagx_veto'}),
            on='t_round',
            how='left',
            suffixes=('', '_veto')
        )
        
        # Calculate time differences
        merge_exact['tdiff'] = np.abs(merge_exact['t'] - merge_exact['t_veto'])
        
        # Flag events within time window
        merge_exact['within_window'] = merge_exact['tdiff'] <= time_window
        
        # Group by dssd event and check if any veto event is within window
        vetoed_tags = merge_exact.groupby('tagx')['within_window'].any()
        
        # Add veto flag to original dataframe
        dssd_events['is_vetoed'] = dssd_events['tagx'].map(vetoed_tags).fillna(False)
        
        # Clean up
        dssd_events = dssd_events.drop('t_round', axis=1, errors='ignore')
        
        return dssd_events
        
    except Exception as e:
        print(f"Error in detmerge: {str(e)}")
        # If error occurs, assume no vetoed events
        dssd_events['is_vetoed'] = False
        return