import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, Galactic
import streamlit as st
from scipy import stats
import seaborn as sns


class FrameDraggingAnalyzer:
    def __init__(self):
        """Initialize the frame dragging analyzer for Gaia data."""
        # Constants
        self.G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        self.c = 299792458.0  # Speed of light in m/s
        self.M_sun = 1.989e30  # Solar mass in kg
        self.pc_to_m = 3.085677581e16  # Parsec to meters
        
    def calculate_galactic_velocities(self, df):
        """
        Convert proper motions to galactic coordinates and tangential velocities.
        """
        # Create a copy of the dataframe
        df_result = df.copy()
        
        # Calculate distance from parallax (in parsecs)
        df_result['distance_pc'] = 1000.0 / df_result['parallax']  # parallax in mas
        
        # Convert proper motions to velocity (km/s)
        # 1 mas/yr * distance_pc = 4.74 km/s tangential velocity
        v_l = df_result['pmra'] * df_result['distance_pc'] * 4.74 / 1000.0  # velocity in km/s
        v_b = df_result['pmdec'] * df_result['distance_pc'] * 4.74 / 1000.0  # velocity in km/s
        
        # Add velocities to dataframe
        df_result['v_l'] = v_l  # Tangential velocity in longitude direction
        df_result['v_b'] = v_b  # Tangential velocity in latitude direction
        
        return df_result
    
    def detect_frame_dragging_signature(self, df, sgr_a_distance=8.122):
        """
        Look for systematic rotation signatures that would indicate frame dragging.
        """
        # Create a copy of the dataframe
        df_result = df.copy()
        
        # Project stellar positions to galactocentric coordinates
        # Assuming Sun is at sgr_a_distance kpc from galactic center
        sun_distance = sgr_a_distance * 1000  # parsecs
        
        # Calculate galactocentric cartesian coordinates
        df_result['x_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.cos(df_result['l_rad']) - sun_distance
        df_result['y_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.sin(df_result['l_rad'])
        df_result['z_gc'] = df_result['distance_pc'] * np.sin(df_result['b_rad'])
        
        # Calculate 3D distance to galactic center in parsecs
        df_result['r_gc'] = np.sqrt(df_result['x_gc']**2 + df_result['y_gc']**2 + df_result['z_gc']**2)
        
        # Calculate azimuthal angle in galactocentric frame
        df_result['theta_gc'] = np.arctan2(df_result['y_gc'], df_result['x_gc'])
        
        # Frame dragging would create systematic velocity in azimuthal direction
        # Convert tangential velocities to galactocentric frame
        v_radial_gc = df_result['v_l'] * np.cos(df_result['theta_gc']) + df_result['v_b'] * np.sin(df_result['theta_gc'])
        v_azimuthal_gc = -df_result['v_l'] * np.sin(df_result['theta_gc']) + df_result['v_b'] * np.cos(df_result['theta_gc'])
        
        df_result['v_radial_gc'] = v_radial_gc
        df_result['v_azimuthal_gc'] = v_azimuthal_gc
        
        return df_result
    
    def analyze_rotation_pattern(self, df):
        """
        Analyze if there's a systematic rotation pattern indicating frame dragging.
        """
        # Bin by galactocentric radius
        r_bins = np.logspace(1, 3, 20)  # 10 to 1000 parsecs from center
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        mean_v_azimuthal = []
        std_v_azimuthal = []
        
        for i in range(len(r_bins)-1):
            mask = (df['r_gc'] >= r_bins[i]) & (df['r_gc'] < r_bins[i+1])
            if np.sum(mask) > 10:  # Need enough stars in bin
                v_az_bin = df[mask]['v_azimuthal_gc']
                mean_v_azimuthal.append(np.mean(v_az_bin))
                std_v_azimuthal.append(np.std(v_az_bin) / np.sqrt(len(v_az_bin)))
            else:
                mean_v_azimuthal.append(np.nan)
                std_v_azimuthal.append(np.nan)
        
        return r_centers, np.array(mean_v_azimuthal), np.array(std_v_azimuthal)
    
    def statistical_significance_test(self, df):
        """
        Test statistical significance of observed rotation.
        """
        v_azimuthal = df['v_azimuthal_gc'].dropna()
        
        # Test if mean is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(v_azimuthal, 0)
        
        # Calculate effect size (Cohen's d)
        cohen_d = np.abs(np.mean(v_azimuthal)) / np.std(v_azimuthal)
        
        results = {
            'sample_size': len(v_azimuthal),
            'mean_azimuthal_velocity': np.mean(v_azimuthal),
            'std_error': np.std(v_azimuthal)/np.sqrt(len(v_azimuthal)),
            'std_deviation': np.std(v_azimuthal),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohen_d
        }
        
        # Add interpretation
        if p_value < 0.001:
            results['significance'] = "Highly significant"
            if cohen_d > 0.1:
                results['interpretation'] = "Strong evidence of frame dragging"
            else:
                results['interpretation'] = "Statistically significant but small effect"
        elif p_value < 0.05:
            results['significance'] = "Significant"
            results['interpretation'] = "Moderate evidence of frame dragging"
        else:
            results['significance'] = "Not significant"
            results['interpretation'] = "No significant evidence of frame dragging"
            
        return results


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
    # Create analyzer instance
    analyzer = FrameDraggingAnalyzer()
    
    # First calculate galactic velocities
    df_result = analyzer.calculate_galactic_velocities(df)
    
    # Detect frame dragging signatures using the advanced analyzer
    df_result = analyzer.detect_frame_dragging_signature(df_result, sgr_a_distance=sgr_a_distance)
    
    # Get constants from the analyzer
    G = analyzer.G
    c = analyzer.c
    M_sun = analyzer.M_sun
    pc_to_m = analyzer.pc_to_m
    
    # Convert inputs to standard SI units
    M_sgr_a = sgr_a_mass * M_sun  # Mass in kg
    R_sgr_a = sgr_a_distance * 1000 * pc_to_m  # Distance in meters
    
    # Calculate Schwarzschild radius
    R_s = 2 * G * M_sgr_a / (c**2)  # Schwarzschild radius in meters
    
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
    # Create analyzer instance for constants
    analyzer = FrameDraggingAnalyzer()
    
    # Get constants
    G = analyzer.G
    c = analyzer.c
    M_sun = analyzer.M_sun
    pc_to_m = analyzer.pc_to_m
    
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
    
    # Add advanced statistical analysis
    if 'v_azimuthal_gc' in df_results.columns:
        # Get rotation pattern analysis
        analyzer = FrameDraggingAnalyzer()
        r_centers, mean_v_az, std_v_az = analyzer.analyze_rotation_pattern(df_results)
        rotation_stats = analyzer.statistical_significance_test(df_results)
        
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
            'fd_larger_than_error_fraction': fd_larger_than_error,
            'rotation_statistics': rotation_stats,
            'r_centers': r_centers,
            'mean_v_azimuthal': mean_v_az,
            'std_v_azimuthal': std_v_az
        }
    else:
        # Return all parameters without rotation analysis
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
