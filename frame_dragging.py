import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
import streamlit as st

def calculate_frame_dragging_signatures(df, sgr_a_mass=4.152e6, sgr_a_distance=8.122, sgr_a_spin=0.9):
    """
    Calculate frame dragging signatures for stars around the galactic center.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed stellar data
    sgr_a_mass : float
        Mass of Sgr A* in solar masses
    sgr_a_distance : float
        Distance to galactic center in kpc
    sgr_a_spin : float
        Dimensionless spin parameter of Sgr A* (0 to 1)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and frame dragging calculations
    """
    # Create a copy of the dataframe
    df_result = df.copy()
    
    # Constants
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    c = const.c.value  # Speed of light in m/s
    M_sun = const.M_sun.value  # Solar mass in kg
    pc_to_m = const.pc.value  # Parsec to meters
    
    # Convert inputs to standard SI units
    M_sgr_a = sgr_a_mass * M_sun  # Mass in kg
    R_sgr_a = sgr_a_distance * 1000 * pc_to_m  # Distance in meters
    
    # Calculate Schwarzschild radius
    R_s = 2 * G * M_sgr_a / (c**2)  # Schwarzschild radius in meters
    
    # Process each star
    # Calculate 3D position (approximation, assuming galactic center is at l=0, b=0)
    df_result['x_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.cos(df_result['l_rad']) - sgr_a_distance * 1000
    df_result['y_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.sin(df_result['l_rad'])
    df_result['z_gc'] = df_result['distance_pc'] * np.sin(df_result['b_rad'])
    
    # Calculate 3D distance to galactic center in parsecs
    df_result['r_gc'] = np.sqrt(df_result['x_gc']**2 + df_result['y_gc']**2 + df_result['z_gc']**2)
    
    # Convert to meters for calculations
    r_gc_m = df_result['r_gc'] * pc_to_m
    
    # Calculate frame dragging effect (Lense-Thirring precession)
    # The effect is proportional to J/r^3 where J is angular momentum
    # J = a * M * G / c, where a is the dimensionless spin parameter
    J = sgr_a_spin * M_sgr_a * G / c
    
    # Calculate frame dragging angular velocity in rad/s
    omega_fd = 2 * G * J / (c**2 * r_gc_m**3)
    
    # Convert to mas/yr for comparison with Gaia proper motions
    # 1 rad/s = 206265000 mas/s
    mas_per_rad = 206265000
    seconds_per_year = 365.25 * 24 * 3600
    
    # Frame dragging proper motion in mas/yr
    fd_effect = omega_fd * mas_per_rad * seconds_per_year
    df_result['fd_effect'] = fd_effect
    
    # Calculate components of frame dragging effect
    # Assuming the black hole spin is aligned with the Galactic z-axis
    # The effect is perpendicular to both spin axis and radial direction
    
    # Normalize position vectors
    r_norm = np.sqrt(df_result['x_gc']**2 + df_result['y_gc']**2 + df_result['z_gc']**2)
    x_hat = df_result['x_gc'] / r_norm
    y_hat = df_result['y_gc'] / r_norm
    z_hat = df_result['z_gc'] / r_norm
    
    # Spin direction (assuming aligned with Galactic z-axis)
    spin_vec = np.array([0, 0, 1])
    
    # Calculate fd effect direction (spin × r_hat)
    fd_dir_x = spin_vec[1] * z_hat - spin_vec[2] * y_hat
    fd_dir_y = spin_vec[2] * x_hat - spin_vec[0] * z_hat
    fd_dir_z = spin_vec[0] * y_hat - spin_vec[1] * x_hat
    
    # Normalize direction vectors
    fd_norm = np.sqrt(fd_dir_x**2 + fd_dir_y**2 + fd_dir_z**2)
    # Avoid division by zero
    fd_norm = np.where(fd_norm > 0, fd_norm, 1)
    
    fd_dir_x = fd_dir_x / fd_norm
    fd_dir_y = fd_dir_y / fd_norm
    fd_dir_z = fd_dir_z / fd_norm
    
    # Calculate frame dragging components
    df_result['fd_effect_x'] = fd_effect * fd_dir_x
    df_result['fd_effect_y'] = fd_effect * fd_dir_y
    df_result['fd_effect_z'] = fd_effect * fd_dir_z
    
    # Project to observable proper motion components (ra, dec)
    # This is an approximation, assuming the fd effect is seen in the tangential plane
    # Proper projection would require more complex coordinate transformations
    
    # Convert to observable frame dragging components (pmra, pmdec)
    # Simplified projection: we project based on star's position on the sky
    l_rad = df_result['l_rad']
    b_rad = df_result['b_rad']
    
    # Project 3D frame dragging onto the sky plane
    # First convert to Galactic coordinates
    fd_l = -fd_dir_x * np.sin(l_rad) + fd_dir_y * np.cos(l_rad)
    fd_b = -fd_dir_x * np.cos(l_rad) * np.sin(b_rad) - fd_dir_y * np.sin(l_rad) * np.sin(b_rad) + fd_dir_z * np.cos(b_rad)
    
    # Convert Galactic proper motion to ICRS (ra, dec) proper motion
    # This is a simplified approximation
    # For a full transformation, we would need to use astropy coordinates
    
    # Approx conversion from galactic to equatorial proper motions
    # Precalculated transformation matrix elements for the galactic pole at J2000
    C1 = -0.054875539
    C2 = -0.873437105
    C3 = -0.483834992
    C4 = 0.494109454
    C5 = -0.444829594
    C6 = 0.746982249
    
    # Apply the transformation
    fd_pmra = fd_effect * (C1 * fd_l + C2 * fd_b)
    fd_pmdec = fd_effect * (C3 * fd_l + C4 * fd_b)
    
    df_result['fd_pmra'] = fd_pmra
    df_result['fd_pmdec'] = fd_pmdec
    
    # Calculate frame dragging magnitude
    df_result['fd_effect_mag'] = np.sqrt(df_result['fd_pmra']**2 + df_result['fd_pmdec']**2)
    
    # Calculate signal-to-noise ratio for detection
    df_result['fd_snr'] = df_result['fd_effect_mag'] / np.sqrt(df_result['pmra_error']**2 + df_result['pmdec_error']**2)
    
    # Expect stronger effect for stars closer to the galactic center
    # Check for expected correlation
    df_result['expected_correlation'] = -1.0 / df_result['r_gc']  # Should correlate with fd_effect
    
    return df_result


def calculate_relativistic_parameters(df_results, sgr_a_mass=4.152e6, sgr_a_distance=8.122):
    """
    Calculate relativistic parameters for frame dragging analysis.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame with frame dragging calculation results
    sgr_a_mass : float
        Mass of Sgr A* in solar masses
    sgr_a_distance : float
        Distance to galactic center in kpc
    
    Returns:
    --------
    dict
        Dictionary with relativistic parameters
    """
    # Constants
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    c = const.c.value  # Speed of light in m/s
    M_sun = const.M_sun.value  # Solar mass in kg
    pc_to_m = const.pc.value  # Parsec to meters
    
    # Convert inputs to standard SI units
    M_sgr_a = sgr_a_mass * M_sun  # Mass in kg
    R_sgr_a = sgr_a_distance * 1000 * pc_to_m  # Distance in meters
    
    # Calculate Schwarzschild radius
    R_s = 2 * G * M_sgr_a / (c**2)  # meters
    
    # Convert to parsecs for reference
    R_s_pc = R_s / pc_to_m
    
    # Calculate influence radius (where BH dominates stellar motions)
    # Typically ~1-2 pc for Sgr A*
    influence_radius_pc = 2.0  # parsecs
    
    # Calculate orbital periods and timescales
    # For stars at 1 pc distance from Sgr A*
    r_orbit = 1.0 * pc_to_m  # 1 pc in meters
    v_orbit = np.sqrt(G * M_sgr_a / r_orbit)  # orbital velocity in m/s
    orbital_period_1pc = 2 * np.pi * r_orbit / v_orbit  # seconds
    orbital_period_1pc_yr = orbital_period_1pc / (365.25 * 24 * 3600)  # years
    
    # Frame dragging timescale (time to precess 360 degrees)
    # Using median value from calculated dataset
    median_fd_effect = df_results['fd_effect'].median()  # rad/s
    if median_fd_effect > 0:
        fd_timescale = (2 * np.pi) / median_fd_effect  # seconds
        fd_timescale_yr = fd_timescale / (365.25 * 24 * 3600)  # years
    else:
        fd_timescale_yr = float('inf')
    
    # Einstein radius (for gravitational lensing)
    D_s = R_sgr_a  # Distance to source (galactic center)
    D_l = R_sgr_a / 2  # Example: Distance to lens (halfway to galactic center)
    D_ls = D_s - D_l  # Distance from lens to source
    einstein_radius = np.sqrt((4 * G * M_sgr_a / c**2) * (D_ls / (D_l * D_s)))  # meters
    einstein_radius_as = einstein_radius / (R_sgr_a * 4.8481e-9)  # arcseconds
    
    # Calculate average signal-to-noise ratio
    mean_snr = df_results['fd_snr'].mean()
    median_snr = df_results['fd_snr'].median()
    max_snr = df_results['fd_snr'].max()
    
    # Proportion of stars with detectable frame dragging
    detection_threshold = 3.0  # SNR > 3 is potentially detectable
    detectable_fraction = (df_results['fd_snr'] > detection_threshold).mean()
    
    # Overall signal-to-noise for the entire dataset
    # Using the fact that SNR adds in quadrature for independent measurements
    overall_snr = np.sqrt((df_results['fd_snr']**2).sum())
    
    # Proportion of stars with frame dragging effect larger than proper motion errors
    pm_error_mean = np.sqrt(df_results['pmra_error']**2 + df_results['pmdec_error']**2).mean()
    fd_larger_than_error = (df_results['fd_effect_mag'] > pm_error_mean).mean()
    
    # Return all parameters as a dictionary
    params = {
        'schwarzschild_radius_pc': R_s_pc,
        'influence_radius_pc': influence_radius_pc,
        'orbital_period_1pc_yr': orbital_period_1pc_yr,
        'frame_dragging_timescale_yr': fd_timescale_yr,
        'einstein_radius_as': einstein_radius_as,
        'mean_fd_snr': mean_snr,
        'median_fd_snr': median_snr,
        'max_fd_snr': max_snr,
        'detectable_fraction': detectable_fraction,
        'signal_to_noise': overall_snr,
        'fd_larger_than_error_fraction': fd_larger_than_error
    }
    
    return params
