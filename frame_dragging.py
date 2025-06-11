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
        """Initialize the frame dragging analyzer for Gaia data (Galactic Center analysis)."""
        # Constants (SI units unless noted)
        self.G = 6.67430e-11   # Gravitational constant [m^3 kg^-1 s^-2]
        self.c = 299792458.0   # Speed of light [m/s]
        self.M_sun = 1.989e30  # Solar mass [kg]
        self.pc_to_m = 3.085677581e16  # Parsec to meters (exact)

    def calculate_galactic_velocities(self, df):
        """
        Convert stellar proper motions (from Gaia) into tangential velocities in Galactic coordinates.
        Returns a DataFrame with additional columns v_l and v_b (tangential velocities in km/s).
        """
        df_result = df.copy()  # Work on a copy to avoid modifying original data

        # Distance in parsec from parallax (in milliarcseconds). (d ≈ 1/parallax with parallax in arcseconds)
        df_result['distance_pc'] = 1000.0 / df_result['parallax']  # parallax is in mas, so 1000 mas = 1"

        # Convert proper motions (mas/yr) to linear velocities (km/s) at the star's distance.
        # 1 mas/yr at 1 pc corresponds to ~4.74 km/s.
        v_l = df_result['pmra']  * df_result['distance_pc'] * 4.74 / 1000.0  # velocity in Galactic longitude direction
        v_b = df_result['pmdec'] * df_result['distance_pc'] * 4.74 / 1000.0  # velocity in Galactic latitude direction

        # Store computed tangential velocities
        df_result['v_l'] = v_l   # [km/s] tangential velocity component along Galactic longitude
        df_result['v_b'] = v_b   # [km/s] tangential velocity component along Galactic latitude

        return df_result

    def detect_frame_dragging_signature(self, df, sgr_a_distance=8.122):
        """
        Project stellar motions into a Galactocentric frame to look for systematic rotation (frame dragging) signatures.
        Adds galactocentric Cartesian coords (x_gc, y_gc, z_gc) and radial/azimuthal velocities.
        """
        df_result = df.copy()

        # Assume Sun is at (–R0, 0, 0) in galactic coordinates, with R0 = sgr_a_distance (kpc) from Galactic Center.
        sun_distance = sgr_a_distance * 1000.0  # convert kpc to parsec for consistency

        # Convert each star's Galactic (l, b, distance) to Galactocentric Cartesian coordinates (in parsec).
        # x-axis points toward Galactic Center, y-axis along direction of Galactic rotation, z-axis toward North Galactic Pole.
        df_result['x_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.cos(df_result['l_rad']) - sun_distance
        df_result['y_gc'] = df_result['distance_pc'] * np.cos(df_result['b_rad']) * np.sin(df_result['l_rad'])
        df_result['z_gc'] = df_result['distance_pc'] * np.sin(df_result['b_rad'])

        # 3D distance from Galactic Center for each star (in parsec)
        df_result['r_gc'] = np.sqrt(df_result['x_gc']**2 + df_result['y_gc']**2 + df_result['z_gc']**2)

        # Azimuthal angle of star in Galactocentric cylindrical coordinates (in radians)
        df_result['theta_gc'] = np.arctan2(df_result['y_gc'], df_result['x_gc'])

        # Resolve the star's tangential velocity into Galactocentric radial and azimuthal components.
        v_radial_gc    = df_result['v_l'] * np.cos(df_result['theta_gc']) + df_result['v_b'] * np.sin(df_result['theta_gc'])
        v_azimuthal_gc = -df_result['v_l'] * np.sin(df_result['theta_gc']) + df_result['v_b'] * np.cos(df_result['theta_gc'])

        df_result['v_radial_gc'] = v_radial_gc       # [km/s] velocity toward/away from GC
        df_result['v_azimuthal_gc'] = v_azimuthal_gc # [km/s] tangential velocity around GC (rotational component)

        return df_result

    def analyze_rotation_pattern(self, df):
        """
        Analyze the average azimuthal velocity vs. radius to see if a rotation pattern (frame dragging) exists.
        Bins data in logarithmic radial bins and computes mean and std error of v_azimuthal in each bin.
        """
        # Define radial bins (10 pc to 1000 pc, logarithmically spaced)
        r_bins = np.logspace(1, 3, 20)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2  # midpoints of each bin for plotting

        mean_v_azimuthal = []
        std_v_azimuthal = []

        for i in range(len(r_bins) - 1):
            mask = (df['r_gc'] >= r_bins[i]) & (df['r_gc'] < r_bins[i+1])
            if np.sum(mask) > 10:  # Only consider bins with sufficient stars for a stable average
                v_az_bin = df.loc[mask, 'v_azimuthal_gc']
                mean_v_azimuthal.append(np.mean(v_az_bin))
                std_v_azimuthal.append(np.std(v_az_bin) / np.sqrt(len(v_az_bin)))  # standard error of mean
            else:
                mean_v_azimuthal.append(np.nan)
                std_v_azimuthal.append(np.nan)

        return r_centers, np.array(mean_v_azimuthal), np.array(std_v_azimuthal)

    def statistical_significance_test(self, df):
        """
        Perform a statistical test to check if the mean azimuthal velocity is significantly different from zero.
        Returns a dictionary of test results and effect size.
        """
        v_azimuthal = df['v_azimuthal_gc'].dropna()

        # One-sample t-test: null hypothesis that mean of v_azimuthal = 0
        t_stat, p_value = stats.ttest_1samp(v_azimuthal, 0.0)

        # Cohen's d (effect size) for the difference from zero
        cohen_d = np.abs(np.mean(v_azimuthal)) / np.std(v_azimuthal) if len(v_azimuthal) > 0 else 0.0

        results = {
            'sample_size': len(v_azimuthal),
            'mean_azimuthal_velocity': np.mean(v_azimuthal),
            'std_error': np.std(v_azimuthal) / np.sqrt(len(v_azimuthal)) if len(v_azimuthal) > 0 else np.nan,
            'std_deviation': np.std(v_azimuthal),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohen_d
        }

        # Add a human-readable interpretation of significance
        if p_value < 0.001:
            results['significance'] = "Highly significant"
            results['interpretation'] = ("Strong evidence of frame dragging"
                                         if cohen_d > 0.1 else
                                         "Statistically significant but effect size is small")
        elif p_value < 0.05:
            results['significance'] = "Significant"
            results['interpretation'] = "Moderate evidence of frame dragging"
        else:
            results['significance'] = "Not significant"
            results['interpretation'] = "No statistically significant evidence of frame dragging"

        return results

def calculate_frame_dragging_signatures(df, sgr_a_mass=4.152e6, sgr_a_distance=8.122, sgr_a_spin=0.9):
    """
    Compute expected frame-dragging (Lense-Thirring) signatures for stars orbiting the Galactic Center (Sgr A*).

    Parameters:
      df : pandas.DataFrame of preprocessed stellar data (including 'l_rad', 'b_rad', 'parallax', 'pmra', 'pmdec', etc.).
      sgr_a_mass : Sgr A* mass in solar masses (default 4.152e6 ~4.15 million M_sun).
      sgr_a_distance : Distance to Galactic Center in kpc (default ~8.122 kpc).
      sgr_a_spin : Dimensionless spin parameter of Sgr A* (0 to 1), indicating rotation (default 0.9).

    Returns:
      pandas.DataFrame with original data plus added columns for frame-dragging effect:
        - fd_effect, fd_effect_x/y/z: predicted frame-dragging proper motion effect (total and components)
        - fd_pmra, fd_pmdec: projected effect in equatorial (RA/Dec) proper motion components
        - fd_effect_mag: magnitude of frame dragging effect on sky (mas/yr)
        - fd_snr: signal-to-noise ratio of the effect given measurement errors
        - expected_correlation: an expected signature (e.g. -1/r dependence) for validation
    """
    analyzer = FrameDraggingAnalyzer()
    # Convert proper motions to tangential velocities and then project into Galactocentric frame
    df_result = analyzer.calculate_galactic_velocities(df)
    df_result = analyzer.detect_frame_dragging_signature(df_result, sgr_a_distance=sgr_a_distance)

    # Physical constants and conversions
    G = analyzer.G
    c = analyzer.c
    M_sun = analyzer.M_sun
    pc_to_m = analyzer.pc_to_m

    # Convert input mass and distance to SI units
    M_sgr_a = sgr_a_mass * M_sun                      # [kg] mass of Sgr A*
    R_sgr_a = sgr_a_distance * 1000.0 * pc_to_m       # [m] distance from us to GC

    # Schwarzschild radius of Sgr A* (for reference, not directly used in calculations here)
    R_s = 2 * G * M_sgr_a / (c**2)                    # [m]

    # Convert each star's distance to GC from parsec to meters
    r_gc_m = df_result['r_gc'] * pc_to_m

    # Compute Lense-Thirring frame-dragging angular velocity (ω_fd) at each star's position.
    # ω_fd ∝ J / r^3, where J is black hole angular momentum.
    # Dimensionless spin a defines J = a * (G M^2 / c) for a Kerr black hole.
    J = sgr_a_spin * G * M_sgr_a**2 / c  # [kg·m^2/s] Angular momentum of Sgr A*

    # Frame dragging angular speed at radius r: ω_fd = 2 G J / (c^2 * r^3)
    omega_fd = 2 * G * J / (c**2 * (r_gc_m**3))

    # Convert angular speed (rad/s) to an equivalent proper motion (mas/yr) observable by Gaia.
    mas_per_rad = 206265000            # [mas] in one radian
    seconds_per_year = 365.25 * 24 * 3600
    fd_effect = omega_fd * mas_per_rad * seconds_per_year  # [mas/yr] frame-dragging effect magnitude
    df_result['fd_effect'] = fd_effect

    # Log key calculation steps for transparency
    print(f"[INFO] Calculating frame dragging signatures for {len(df_result)} stars.")
    print(f"[DEBUG] Sgr A* mass = {sgr_a_mass:.3e} M☉, distance = {sgr_a_distance:.3f} kpc, spin = {sgr_a_spin:.2f}")
    print(f"[DEBUG] Schwarzschild radius R_s ≈ {R_s:.3e} m ({R_s / pc_to_m:.3e} pc)")
    print(f"[DEBUG] Angular momentum J = {J:.3e} kg·m^2/s (for a = {sgr_a_spin})")
    # Show sample values of ω_fd and fd_effect for sanity check
    if hasattr(omega_fd, '__len__') and len(omega_fd) > 3:
        # If omega_fd is a Series or array, print first 3 values
        print(f"[DEBUG] Sample ω_fd (rad/s): {list(omega_fd[:3])} ...")
    else:
        print(f"[DEBUG] ω_fd (rad/s) = {float(omega_fd):.3e}")
    if hasattr(fd_effect, '__len__') and len(fd_effect) > 0:
        fd_min = float(np.min(fd_effect))
        fd_max = float(np.max(fd_effect))
        print(f"[DEBUG] Frame-dragging proper motion effect range: {fd_min:.2e} to {fd_max:.2e} mas/yr")
    else:
        print(f"[DEBUG] Frame-dragging proper motion effect: {float(fd_effect):.2e} mas/yr")

    # Determine the 3D direction of the frame-dragging force for each star (perpendicular to spin axis and radial vector).
    r_norm = np.sqrt(df_result['x_gc']**2 + df_result['y_gc']**2 + df_result['z_gc']**2)
    x_hat = df_result['x_gc'] / r_norm
    y_hat = df_result['y_gc'] / r_norm
    z_hat = df_result['z_gc'] / r_norm

    spin_vec = np.array([0.0, 0.0, 1.0])  # Unit vector along spin (assuming spin aligned with Galactic z-axis)

    # Direction of frame dragging effect: proportional to (spin × r_hat)
    fd_dir_x = spin_vec[1] * z_hat - spin_vec[2] * y_hat
    fd_dir_y = spin_vec[2] * x_hat - spin_vec[0] * z_hat
    fd_dir_z = spin_vec[0] * y_hat - spin_vec[1] * x_hat

    # Normalize direction vectors (to unit length) to get direction of effect
    fd_norm = np.sqrt(fd_dir_x**2 + fd_dir_y**2 + fd_dir_z**2)
    fd_norm = np.where(fd_norm > 0, fd_norm, 1)  # avoid division by zero for any star exactly at spin axis
    fd_dir_x = fd_dir_x / fd_norm
    fd_dir_y = fd_dir_y / fd_norm
    fd_dir_z = fd_dir_z / fd_norm

    # Decompose the frame dragging effect vector into x, y, z components in Galactocentric frame
    df_result['fd_effect_x'] = fd_effect * fd_dir_x
    df_result['fd_effect_y'] = fd_effect * fd_dir_y
    df_result['fd_effect_z'] = fd_effect * fd_dir_z

    # Project the 3D frame dragging vectors back onto observable sky-plane (approximate).
    # We use Galactic (l, b) coordinates for projection, assuming small angles.
    l_rad = df_result['l_rad']
    b_rad = df_result['b_rad']
    # Components of frame dragging in Galactic coordinate directions:
    fd_l = -fd_dir_x * np.sin(l_rad) + fd_dir_y * np.cos(l_rad)
    fd_b = (-fd_dir_x * np.cos(l_rad) * np.sin(b_rad)
            - fd_dir_y * np.sin(l_rad) * np.sin(b_rad)
            + fd_dir_z * np.cos(b_rad))

    # Convert the Galactic (l*, b*) effect into equatorial (RA, Dec) proper motion components.
    # Using an approximate fixed rotation matrix from Galactic to ICRS:
    C1, C2, C3, C4 = -0.054875539, -0.873437105, -0.483834992, 0.494109454
    # (These coefficients correspond to elements of the rotation matrix for coordinate frames.)
    fd_pmra = fd_effect * (C1 * fd_l + C2 * fd_b)   # approximate Δ(ra) [mas/yr]
    fd_pmdec = fd_effect * (C3 * fd_l + C4 * fd_b)  # approximate Δ(dec) [mas/yr]
    df_result['fd_pmra'] = fd_pmra
    df_result['fd_pmdec'] = fd_pmdec

    # Total magnitude of frame dragging proper motion effect on sky (mas/yr)
    df_result['fd_effect_mag'] = np.sqrt(df_result['fd_pmra']**2 + df_result['fd_pmdec']**2)

    # Compute signal-to-noise ratio (SNR) of the effect for each star using its proper motion errors
    df_result['fd_snr'] = df_result['fd_effect_mag'] / np.sqrt(df_result['pmra_error']**2 + df_result['pmdec_error']**2)

    # Expected correlation: for true frame dragging, effect should be stronger at smaller r_gc (so anticorrelate with r)
    df_result['expected_correlation'] = -1.0 / df_result['r_gc']

    print("[DEBUG] Completed frame dragging calculations for all stars.")
    return df_result

def calculate_relativistic_parameters(df_results, sgr_a_mass=4.152e6, sgr_a_distance=8.122):
    """
    Compute aggregate relativistic parameters and detection metrics from the frame dragging analysis results.

    This includes Schwarzschild radius, frame-dragging precession timescales, lensing scale (Einstein radius),
    and statistics on the signal-to-noise and detection fractions.
    """
    analyzer = FrameDraggingAnalyzer()
    G = analyzer.G
    c = analyzer.c
    M_sun = analyzer.M_sun
    pc_to_m = analyzer.pc_to_m

    # Convert inputs to SI
    M_sgr_a = sgr_a_mass * M_sun                 # [kg]
    R_sgr_a = sgr_a_distance * 1000.0 * pc_to_m  # [m]

    # Schwarzschild radius of Sgr A* in parsec (for context)
    R_s = 2 * G * M_sgr_a / (c**2)   # [m]
    R_s_pc = R_s / pc_to_m          # [pc]

    # Influence radius ~ region where Sgr A* gravity dominates (assume ~2 pc for Sgr A*)
    influence_radius_pc = 2.0

    # Orbital period for a test star at 1 pc from Sgr A*
    r_orbit = 1.0 * pc_to_m        # [m]
    v_orbit = np.sqrt(G * M_sgr_a / r_orbit)  # [m/s]
    orbital_period_1pc = 2 * np.pi * r_orbit / v_orbit         # [s]
    orbital_period_1pc_yr = orbital_period_1pc / (365.25 * 24 * 3600)  # [yr]

    # Frame-dragging precession timescale (time for Lense-Thirring precession of ~360°) using median effect strength
    median_fd_effect = df_results['fd_effect'].median()  # [mas/yr]
    median_omega_fd = median_fd_effect / (206265000 * 365.25 * 24 * 3600)  # [rad/s], convert mas/yr back to rad/s
    if median_omega_fd > 0:
        fd_timescale = (2 * np.pi) / median_omega_fd           # [s] time for full precession
        fd_timescale_yr = fd_timescale / (365.25 * 24 * 3600)  # [yr]
    else:
        fd_timescale_yr = float('inf')

    # Einstein radius (lensing scale) if a mass M_sgr_a is halfway to the GC (just a rough reference calculation)
    D_s = R_sgr_a            # [m] distance to source (GC)
    D_l = R_sgr_a / 2.0      # [m] distance to lens (halfway to GC for this estimate)
    D_ls = D_s - D_l         # [m] distance from lens to source
    einstein_radius = np.sqrt((4 * G * M_sgr_a / c**2) * (D_ls / (D_l * D_s)))  # [m]
    einstein_radius_as = einstein_radius / (R_sgr_a * 4.8481e-9)               # [arcsec] (1 pc = 4.8481e-6 as at 1 pc)

    # SNR statistics for frame-dragging detections
    mean_snr = float(df_results['fd_snr'].mean())
    median_snr = float(df_results['fd_snr'].median())
    max_snr = float(df_results['fd_snr'].max())

    # Fraction of stars with frame-dragging SNR above a threshold (e.g., 3 for ~3σ detection)
    detection_threshold = 3.0
    detectable_fraction = float((df_results['fd_snr'] > detection_threshold).mean())

    # Combined SNR for the entire sample (treating all stars together, summing variances)
    overall_snr = float(np.sqrt(np.nansum(df_results['fd_snr']**2)))

    # Fraction of stars where FD effect exceeds the mean proper motion error
    pm_error_mean = float(np.sqrt(df_results['pmra_error']**2 + df_results['pmdec_error']**2).mean())
    fd_larger_than_error_fraction = float((df_results['fd_effect_mag'] > pm_error_mean).mean())

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
        'fd_larger_than_error_fraction': fd_larger_than_error_fraction
    }

    # If rotation pattern stats were computed, include them
    if 'v_azimuthal_gc' in df_results.columns:
        rotation_stats = analyzer.statistical_significance_test(df_results)
        r_centers, mean_v_az, std_v_az = analyzer.analyze_rotation_pattern(df_results)
        params.update({
            'rotation_statistics': rotation_stats,
            'r_centers': r_centers,
            'mean_v_azimuthal': mean_v_az,
            'std_v_azimuthal': std_v_az
        })

    # Log a summary of key parameters for insight
    print("[INFO] Relativistic parameter summary:")
    print(f"  Schwarzschild radius ~ {params['schwarzschild_radius_pc']:.3e} pc")
    print(f"  Influence radius ~ {params['influence_radius_pc']:.1f} pc")
    print(f"  Orbital period at 1 pc ~ {params['orbital_period_1pc_yr']:.2e} years")
    print(f"  Frame-dragging precession timescale ~ {params['frame_dragging_timescale_yr']:.2e} years")
    print(f"  Einstein radius ~ {params['einstein_radius_as']:.2e} arcseconds")
    print(f"  Mean FD SNR = {params['mean_fd_snr']:.2f}, Max FD SNR = {params['max_fd_snr']:.2f}")
    print(f"  {params['detectable_fraction']*100:.1f}% of stars have SNR > {detection_threshold}")
    print(f"  Combined SNR (all stars) = {params['signal_to_noise']:.2f}")
    print(f"  {params['fd_larger_than_error_fraction']*100:.1f}% of stars have FD effect larger than mean PM error")
    if 'rotation_statistics' in params:
        stats = params['rotation_statistics']
        print(f"  Rotation pattern p-value = {stats['p_value']:.3e} ({stats['interpretation']})")

    return params
