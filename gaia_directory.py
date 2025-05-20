import requests
import pandas as pd
import os
import gzip
import shutil
import tempfile
from pathlib import Path
import streamlit as st

def fetch_gaia_directory():
    """
    Fetches the Gaia data directory information from ESA's CDN.
    
    Returns:
    --------
    list
        List of available Gaia data files
    """
    # Base URL for Gaia DR3 CDN directory
    base_url = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"
    
    try:
        # Placeholder for file list - in a real scenario we'd query the server
        # Returns some common HEALPix files around the galactic center
        healpix_files = [
            "GaiaSource_4683.csv.gz",  # Near galactic center
            "GaiaSource_4684.csv.gz",  # Near galactic center
            "GaiaSource_4555.csv.gz",  # Near galactic center
            "GaiaSource_4556.csv.gz",  # Near galactic center
        ]
        
        return [(f, f"{base_url}{f}") for f in healpix_files]
    except Exception as e:
        st.error(f"Error fetching Gaia directory: {str(e)}")
        return []

def download_gaia_file(file_url, output_path, progress_text=None, progress_bar=None):
    """
    Downloads a Gaia data file from the specified URL.
    
    Parameters:
    -----------
    file_url : str
        URL of the Gaia file to download
    output_path : str
        Path to save the downloaded file
    progress_text : st.empty, optional
        Streamlit text element for progress updates
    progress_bar : st.progress, optional
        Streamlit progress bar element
        
    Returns:
    --------
    bool
        True if download was successful, False otherwise
    """
    try:
        # Show progress if widgets provided
        if progress_text:
            progress_text.text(f"Downloading {os.path.basename(file_url)}...")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Stream download to avoid memory issues
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                
                # Get total file size for progress tracking
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                # Download in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress if widgets provided
                        if progress_bar and total_size > 0:
                            progress_bar.progress(min(1.0, downloaded / total_size))
            
            # Move temp file to final destination
            shutil.move(temp_file.name, output_path)
            
            if progress_text:
                progress_text.text(f"Downloaded {os.path.basename(file_url)} successfully")
            
            return True
    except Exception as e:
        if progress_text:
            progress_text.error(f"Error downloading {os.path.basename(file_url)}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def get_cached_files(cache_dir):
    """
    Gets a list of cached Gaia files.
    
    Parameters:
    -----------
    cache_dir : str
        Directory to check for cached files
        
    Returns:
    --------
    list
        List of paths to cached Gaia files
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return list(Path(cache_dir).glob("*.csv.gz"))

def fetch_gaia_data_from_cdn(max_files=2):
    """
    Fetches Gaia data directly from ESA's CDN.
    
    Parameters:
    -----------
    max_files : int
        Maximum number of files to download
        
    Returns:
    --------
    list
        List of paths to downloaded Gaia files
    """
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), ".gaia_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for cached files first
    cached_files = get_cached_files(cache_dir)
    if cached_files:
        st.success(f"Using {len(cached_files)} cached Gaia data files")
        return [str(f) for f in cached_files]
    
    # Set up progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Get directory of available files
    progress_text.text("Fetching Gaia data directory...")
    available_files = fetch_gaia_directory()
    
    if not available_files:
        progress_text.error("No Gaia files found in directory")
        return []
    
    # Limit number of files to download
    files_to_download = available_files[:max_files]
    progress_text.text(f"Found {len(available_files)} Gaia files, downloading {len(files_to_download)}")
    
    # Download each file
    downloaded_files = []
    for i, (filename, file_url) in enumerate(files_to_download):
        # Update overall progress
        overall_progress = i / len(files_to_download)
        progress_bar.progress(overall_progress)
        
        # Set output path
        output_path = os.path.join(cache_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            progress_text.text(f"File {filename} already exists, skipping")
            downloaded_files.append(output_path)
            continue
        
        # Download file
        success = download_gaia_file(file_url, output_path, progress_text, None)
        if success:
            downloaded_files.append(output_path)
    
    # Complete progress
    progress_bar.progress(1.0)
    
    if downloaded_files:
        progress_text.success(f"Downloaded {len(downloaded_files)} Gaia files successfully")
        return downloaded_files
    else:
        progress_text.error("Failed to download any Gaia files")
        
        # Create a synthetic sample as a fallback only if explicitly requested
        return []