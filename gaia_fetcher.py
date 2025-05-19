import pandas as pd
import numpy as np
import os
import requests
import gzip
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
import streamlit as st
from astroquery.gaia import Gaia

def fetch_gaia_data(max_rows=50000, save_dir=None):
    """
    Fetch data directly from the Gaia Data Release 3 archive.
    
    Parameters:
    -----------
    max_rows : int
        Maximum number of rows to fetch
    save_dir : str or None
        Directory to save downloaded data, or None for temporary storage
    
    Returns:
    --------
    list
        List of paths to downloaded Gaia data files
    """
    # Set up progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Connecting to Gaia archive...")
    
    # Set Gaia table
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    
    # Create a temporary directory if save_dir is not provided
    if save_dir is None:
        save_dir = tempfile.mkdtemp()
    else:
        os.makedirs(save_dir, exist_ok=True)
    
    # Query for stars around galactic center with good proper motion measurements
    query = f"""
    SELECT TOP {max_rows}
        source_id, ra, dec, pmra, pmdec, radial_velocity,
        parallax, phot_g_mean_mag, l, b,
        pmra_error, pmdec_error, radial_velocity_error,
        bp_rp, phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE pmra IS NOT NULL 
        AND pmdec IS NOT NULL
        AND pmra_error < 1.0
        AND pmdec_error < 1.0
        AND parallax > 0.1
        AND abs(b) > 10  -- Avoid galactic plane extinction
        AND phot_g_mean_mag < 16  -- Bright enough for good measurements
        AND l BETWEEN -20 AND 20  -- Focus around the galactic center
        AND b BETWEEN -20 AND 20
    """
    
    try:
        progress_text.text("Submitting query to Gaia archive... This may take a few minutes.")
        job = Gaia.launch_job_async(query)
        
        progress_text.text("Query submitted. Waiting for results...")
        progress_bar.progress(0.3)
        
        # Get the results
        results = job.get_results()
        
        progress_text.text(f"Retrieved {len(results)} stars from Gaia. Processing data...")
        progress_bar.progress(0.7)
        
        # Convert to pandas dataframe and save to file
        df = results.to_pandas()
        
        # Save to gzipped CSV file
        output_file = os.path.join(save_dir, "gaia_results.csv.gz") 
        with gzip.open(output_file, 'wt') as f:
            df.to_csv(f, index=False)
        
        progress_text.text(f"Downloaded {len(df):,} stars from Gaia archive.")
        progress_bar.progress(1.0)
        
        return [output_file]
    
    except Exception as e:
        progress_text.error(f"Error fetching data from Gaia: {str(e)}")
        
        # Fall back to direct HTTP download from ESA CDN if astroquery fails
        return download_from_esa_cdn(save_dir, progress_text, progress_bar)

def download_from_esa_cdn(save_dir, progress_text, progress_bar):
    """
    Download Gaia data from ESA's CDN server as a fallback method.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save downloaded files
    progress_text : st.empty
        Streamlit text element for progress updates
    progress_bar : st.progress
        Streamlit progress bar element
    
    Returns:
    --------
    list
        List of paths to downloaded files
    """
    progress_text.text("Attempting direct download from ESA CDN...")
    progress_bar.progress(0.1)
    
    # Base URL for Gaia DR3 files
    base_url = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source"
    
    # Specific regions around the galactic center
    healpix_files = [
        "GaiaSource_4312.csv.gz",  # Central region
        "GaiaSource_4313.csv.gz",  # Adjacent region
    ]
    
    downloaded_files = []
    
    for i, filename in enumerate(healpix_files):
        file_url = f"{base_url}/{filename}"
        output_path = os.path.join(save_dir, filename)
        
        progress_text.text(f"Downloading {filename} ({i+1}/{len(healpix_files)})...")
        progress_percent = 0.1 + (i / len(healpix_files)) * 0.8
        progress_bar.progress(progress_percent)
        
        try:
            # Stream download to save memory
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            downloaded_files.append(output_path)
            
        except Exception as e:
            progress_text.error(f"Error downloading {filename}: {str(e)}")
            continue
    
    if downloaded_files:
        progress_text.text(f"Successfully downloaded {len(downloaded_files)} Gaia data files.")
        progress_bar.progress(1.0)
        return downloaded_files
    else:
        progress_text.error("Failed to download Gaia data.")
        return []

def cache_gaia_data():
    """
    Cache Gaia data for faster access.
    
    This function downloads data once and stores it in a persistent location.
    
    Returns:
    --------
    list
        List of paths to Gaia data files
    """
    # Create a cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), ".gaia_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if we already have cached data
    cached_files = list(Path(cache_dir).glob("*.csv.gz"))
    
    if cached_files:
        return [str(file) for file in cached_files]
    
    # No cached data, fetch new data
    return fetch_gaia_data(save_dir=cache_dir)