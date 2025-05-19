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
    
    # If no sample file available, create a synthetic file with frame dragging signal
    print("Creating enhanced synthetic dataset for frame dragging analysis")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "GaiaSource_sample.csv.gz")
    
    # Create synthetic data - larger dataset for better statistics
    n_stars = 50000
    np.random.seed(42)
    
    # Generate random galactic coordinates centered around GC
    # More concentration near galactic center for better frame dragging detection
    l = np.random.normal(0, 15, n_stars)  # Concentrate more around center
    b = np.random.normal(0, 10, n_stars)
    
    # Convert to RA/Dec (simplified approximation)
    ra = 266.4051 - 0.8 * l
    dec = -28.936175 + b
    
    # Generate proper motions with pattern
    # Distance to galactic center (approx)
    dist_gc = np.sqrt(l**2 + b**2)
    
    # Proper motion magnitude follows 1/sqrt(r) for Keplerian orbits
    pm_mag_keplerian = 10.0 / np.sqrt(dist_gc + 1)
    
    # Add frame dragging component (systematic rotation)
    # Frame dragging falls off with 1/r^3
    frame_dragging_strength = 0.3  # Strength parameter (μas/yr at 100 pc)
    fd_effect = frame_dragging_strength * 1e6 / np.power(dist_gc + 100, 3)
    
    # Generate angles for proper motion
    # Keplerian motion angle (tangential to GC)
    keplerian_angle = np.arctan2(l, -b) + np.pi/2 + np.random.normal(0, 0.3, n_stars)
    
    # Frame dragging angle (systematic around GC, perpendicular to radial)
    fd_angle = np.arctan2(l, -b) + np.pi + np.random.normal(0, 0.1, n_stars)
    
    # Combine Keplerian and frame dragging components into proper motion
    pmra_keplerian = pm_mag_keplerian * np.sin(keplerian_angle)
    pmdec_keplerian = pm_mag_keplerian * np.cos(keplerian_angle)
    
    pmra_fd = fd_effect * np.sin(fd_angle)
    pmdec_fd = fd_effect * np.cos(fd_angle)
    
    # Total proper motion
    pmra = pmra_keplerian + pmra_fd
    pmdec = pmdec_keplerian + pmdec_fd
    
    # Generate other parameters
    source_id = np.arange(1, n_stars + 1)
    
    # Distance distribution - more stars near GC for better signal
    parallax = 0.1 + np.random.exponential(0.2, n_stars)  # smaller parallax = greater distance
    
    # For some stars, add radial velocity (30%)
    has_rv = np.random.random(n_stars) < 0.3
    radial_velocity = np.full(n_stars, np.nan)
    radial_velocity[has_rv] = np.random.normal(0, 100, sum(has_rv))
    
    # Magnitude (with realistic distribution for Gaia)
    phot_g_mean_mag = np.random.normal(15, 2, n_stars)
    phot_g_mean_mag = np.clip(phot_g_mean_mag, 8, 20)  # constrain to realistic range
    
    # Generate realistic error parameters (based on magnitude and distance)
    # Fainter and more distant stars have larger errors
    base_error = 0.05  # minimum error in mas/yr
    g_mag_factor = np.exp((phot_g_mean_mag - 15) / 5)  # error increases with magnitude
    dist_factor = 1.0 / np.sqrt(parallax)  # error increases with distance
    
    # Calculate errors
    pmra_error = base_error * g_mag_factor * dist_factor * np.random.uniform(0.8, 1.2, n_stars)
    pmdec_error = base_error * g_mag_factor * dist_factor * np.random.uniform(0.8, 1.2, n_stars)
    
    # Add noise to proper motion based on errors
    pmra += np.random.normal(0, pmra_error, n_stars)
    pmdec += np.random.normal(0, pmdec_error, n_stars)
    
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
    
    # Apply quality cuts similar to what we'd do with real data
    df = df[
        (df['pmra'].notna()) & 
        (df['pmdec'].notna()) &
        (df['parallax'] > 0.1) &
        (abs(df['b']) > 5) &
        (df['phot_g_mean_mag'] < 18)
    ]
    
    # Save to gzip CSV
    with gzip.open(file_path, 'wt') as f:
        df.to_csv(f, index=False)
    
    return file_path
