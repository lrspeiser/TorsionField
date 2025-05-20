import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import streamlit as st
from io import BytesIO
import io

def load_local_gaia_files(file_paths, sample_size=50000, filter_params=None):
    """
    Load and sample from local Gaia CSV.gz files.
    
    Parameters:
    -----------
    file_paths : list
        List of paths to GaiaSource_*.csv.gz files
    sample_size : int
        Number of stars to randomly sample
    filter_params : dict
        Dictionary with filter parameters
    """
    
    # Set default filter parameters if not provided
    if filter_params is None:
        filter_params = {
            "pmra_error_max": 1.0,
            "pmdec_error_max": 1.0,
            "parallax_min": 0.1,
            "b_min": 10,
            "g_mag_max": 16.0
        }
    
    # Columns we need for frame dragging analysis
    required_cols = [
        'source_id', 'ra', 'dec', 'pmra', 'pmdec', 
        'radial_velocity', 'parallax', 'l', 'b',
        'pmra_error', 'pmdec_error', 'phot_g_mean_mag'
    ]
    
    all_data = []
    total_loaded = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for file_path in file_paths:
        status_text.text(f"Loading {file_path}...")
        
        try:
            # Read compressed CSV
            with gzip.open(file_path, 'rt') as f:
                # Read in chunks to manage memory
                chunk_size = 10000
                for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
                    
                    # Filter for good quality data
                    mask = (
                        chunk['pmra'].notna() & 
                        chunk['pmdec'].notna() &
                        (chunk['pmra_error'] < filter_params["pmra_error_max"]) &
                        (chunk['pmdec_error'] < filter_params["pmdec_error_max"]) &
                        (chunk['parallax'] > filter_params["parallax_min"]) &
                        (abs(chunk['b']) > filter_params["b_min"]) &
                        (chunk['phot_g_mean_mag'] < filter_params["g_mag_max"])
                    )
                    
                    good_data = chunk[mask][required_cols]
                    
                    if len(good_data) > 0:
                        all_data.append(good_data)
                        total_loaded += len(good_data)
                        
                        status_text.text(f"Loaded {total_loaded} good stars so far...")
                        progress_percent = min(total_loaded / sample_size, 1.0)
                        progress_bar.progress(progress_percent)
                        
                        # Stop if we have enough
                        if total_loaded >= sample_size:
                            break
            
            if total_loaded >= sample_size:
                break
                
        except Exception as e:
            status_text.error(f"Error processing file {file_path}: {str(e)}")
            # Continue with other files
            continue
    
    # Combine all data
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        
        # Randomly sample to exact size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        status_text.text(f"Final dataset: {len(df):,} stars")
        progress_bar.progress(1.0)
        return df
    else:
        raise ValueError("No good quality data found!")


def preprocess_stellar_data(df):
    """
    Preprocess stellar data for frame dragging analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stellar data from Gaia
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert coordinates from degrees to radians
    df_processed['ra_rad'] = np.radians(df_processed['ra'])
    df_processed['dec_rad'] = np.radians(df_processed['dec'])
    df_processed['l_rad'] = np.radians(df_processed['l'])
    df_processed['b_rad'] = np.radians(df_processed['b'])
    
    # Calculate distance in parsecs
    df_processed['distance_pc'] = 1000.0 / df_processed['parallax']
    
    # Calculate proper motion magnitude
    df_processed['pm_mag'] = np.sqrt(df_processed['pmra']**2 + df_processed['pmdec']**2)
    
    # Calculate proper motion position angle (North=0, East=90)
    df_processed['pm_angle'] = np.degrees(np.arctan2(df_processed['pmra'], df_processed['pmdec']))
    
    # Correct negative angles
    df_processed['pm_angle'] = df_processed['pm_angle'] % 360
    
    # Estimate 3D velocity components if radial velocity is available
    has_rv = df_processed['radial_velocity'].notna()
    
    if has_rv.any():
        # Constants
        k = 4.74057  # Conversion factor from mas/yr to km/s at 1 kpc
        
        # Calculate velocity components directly on the dataframe using .loc
        # This avoids the SettingWithCopyWarning
        # Calculate tangential velocity components directly
        df_processed.loc[has_rv, 'v_l'] = k * df_processed.loc[has_rv, 'distance_pc'] * df_processed.loc[has_rv, 'pmra'] / 1000.0  # km/s
        df_processed.loc[has_rv, 'v_b'] = k * df_processed.loc[has_rv, 'distance_pc'] * df_processed.loc[has_rv, 'pmdec'] / 1000.0  # km/s
        
        # Include radial velocity
        df_processed.loc[has_rv, 'v_r'] = df_processed.loc[has_rv, 'radial_velocity']  # km/s
    
    # For all stars, calculate projected galactocentric distance
    # Assuming R_0 = 8.122 kpc (distance to galactic center)
    R_0 = 8122.0  # pc
    
    # Calculate projected distance to galactic center
    df_processed['r_gc_proj'] = np.sqrt(
        (df_processed['distance_pc'] * np.cos(df_processed['b_rad']) * np.cos(df_processed['l_rad']) - R_0)**2 +
        (df_processed['distance_pc'] * np.cos(df_processed['b_rad']) * np.sin(df_processed['l_rad']))**2
    )
    
    # Calculate angular distance from galactic center
    df_processed['ang_gc'] = np.degrees(
        np.arccos(np.cos(df_processed['b_rad']) * np.cos(df_processed['l_rad']))
    )
    
    # Quality flags
    df_processed['quality_flag'] = 1  # Default good quality
    
    # Flag stars with high uncertainties relative to their parallax
    df_processed.loc[df_processed['parallax'] < 3 * df_processed['pmra_error'], 'quality_flag'] = 0
    df_processed.loc[df_processed['parallax'] < 3 * df_processed['pmdec_error'], 'quality_flag'] = 0
    
    return df_processed
