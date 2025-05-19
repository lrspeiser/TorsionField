import os
import gzip
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import tempfile
import shutil

def save_results(fig, filename, dpi=300):
    """
    Save figure to a file and return the file buffer.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename to save to
    dpi : int
        DPI for the figure
    
    Returns:
    --------
    BytesIO
        Buffer with the saved figure
    """
    buf = BytesIO()
    fig.savefig(buf, format="pdf", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def get_sample_file_path():
    """
    Generate a sample Gaia data file path or create a synthetic one if needed.
    
    Returns:
    --------
    str
        Path to a sample Gaia data file
    """
    # Check if sample file path is provided in environment
    sample_path = os.environ.get("GAIA_SAMPLE_PATH", "")
    
    if sample_path and os.path.exists(sample_path):
        return sample_path
    
    # If no sample file available, create a minimal synthetic file
    # Only for demonstration purposes
    print("Creating synthetic sample data file for demonstration")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "GaiaSource_sample.csv.gz")
    
    # Create synthetic data
    n_stars = 10000
    np.random.seed(42)
    
    # Generate random galactic coordinates centered around GC
    l = np.random.normal(0, 20, n_stars)
    b = np.random.normal(0, 15, n_stars)
    
    # Convert to RA/Dec (simplified approximation)
    ra = 266.4051 - 0.8 * l
    dec = -28.936175 + b
    
    # Generate proper motions with pattern
    # Distance to galactic center (approx)
    dist_gc = np.sqrt(l**2 + b**2)
    
    # Proper motion magnitude follows 1/sqrt(r) for Keplerian orbits
    pm_mag = 10.0 / np.sqrt(dist_gc + 1)
    
    # Angle is tangential to galactic center (+ some random scatter)
    pm_angle = np.arctan2(l, -b) + np.pi/2 + np.random.normal(0, 0.5, n_stars)
    
    # Convert to pmra, pmdec components
    pmra = pm_mag * np.sin(pm_angle)
    pmdec = pm_mag * np.cos(pm_angle)
    
    # Generate other parameters
    source_id = np.arange(1, n_stars + 1)
    parallax = 0.2 + np.random.exponential(0.3, n_stars)  # avg ~0.5 mas (~2kpc)
    
    # For some stars, add radial velocity (only 20%)
    has_rv = np.random.random(n_stars) < 0.2
    radial_velocity = np.full(n_stars, np.nan)
    radial_velocity[has_rv] = np.random.normal(0, 100, sum(has_rv))
    
    # Error parameters
    pmra_error = np.random.exponential(0.2, n_stars) + 0.1
    pmdec_error = np.random.exponential(0.2, n_stars) + 0.1
    
    # Magnitude
    phot_g_mean_mag = np.random.normal(15, 2, n_stars)
    
    # Create DataFrame
    df = pd.DataFrame({
        'source_id': source_id,
        'ra': ra,
        'dec': dec,
        'l': l,
        'b': b,
        'pmra': pmra,
        'pmdec': pmdec,
        'radial_velocity': radial_velocity,
        'parallax': parallax,
        'pmra_error': pmra_error,
        'pmdec_error': pmdec_error,
        'phot_g_mean_mag': phot_g_mean_mag
    })
    
    # Save to gzip CSV
    with gzip.open(file_path, 'wt') as f:
        df.to_csv(f, index=False)
    
    return file_path
