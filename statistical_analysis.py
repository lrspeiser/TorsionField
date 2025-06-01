import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

def run_monte_carlo_simulation(df_results, n_simulations=1000, null_model="Random stellar motions"):
    """
    Run Monte Carlo simulations to validate frame dragging detection.

    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame with frame dragging calculation results
    n_simulations : int
        Number of Monte Carlo simulations to run
    null_model : str
        Null hypothesis model type

    Returns:
    --------
    dict
        Dictionary with simulation results
    """
    print(f"[DEBUG] run_monte_carlo_simulation called with {len(df_results)} stars, {n_simulations} simulations")
    print(f"[DEBUG] Null model: {null_model}")

    try:
        # Validate input data
        if df_results.empty:
            print(f"[ERROR] Empty dataframe provided")
            raise ValueError("Empty dataframe provided")

        required_cols = ['pmra', 'pmdec', 'pmra_error', 'pmdec_error', 
                        'fd_pmra', 'fd_pmdec', 'fd_effect_mag', 'r_gc']
        missing_cols = [col for col in required_cols if col not in df_results.columns]

        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Copy key data to avoid modifying original
        pm_data = df_results[required_cols].copy()
        print(f"[DEBUG] Copied {len(pm_data)} rows with required columns")

        # Initialize arrays to store simulation results
        sim_fd_effects = np.zeros(n_simulations)
        sim_correlations = np.zeros(n_simulations)
        sim_alignments = np.zeros(n_simulations)

        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        print(f"[DEBUG] Starting Monte Carlo simulations")

        # Run simulations
        for i in range(n_simulations):
            if i % 10 == 0:
                status_text.text(f"Running simulation {i+1} of {n_simulations}...")
                progress_bar.progress((i+1) / n_simulations)
                print(f"[DEBUG] Simulation {i+1}/{n_simulations}")

            # Generate simulated data based on null model
            if null_model == "Random stellar motions":
                # Randomly shuffle proper motions to break any physical correlation
                shuffled_idx = np.random.permutation(len(pm_data))
                sim_pmra = pm_data['pmra'].values[shuffled_idx]
                sim_pmdec = pm_data['pmdec'].values[shuffled_idx]

            elif null_model == "Non-relativistic gravitational effects":
                # Generate proper motions based on Keplerian orbits around galactic center
                # with random orientations, but without frame dragging

                # Calculate expected velocity dispersion at each radius
                # v_disp ∝ 1/sqrt(r) in a simple model
                v_disp = 100.0 / np.sqrt(pm_data['r_gc'] / 1000)  # km/s

                # Convert to proper motion (mas/yr)
                # 1 mas/yr at 1 kpc ≈ 4.74 km/s
                pm_disp = v_disp / 4.74 * (1000 / pm_data['r_gc'])

                # Generate random position angles
                angles = np.random.uniform(0, 2*np.pi, len(pm_data))

                # Generate proper motions with expected dispersions
                sim_pmra = pm_disp * np.sin(angles)
                sim_pmdec = pm_disp * np.cos(angles)

            elif null_model == "Measurement errors only":
                # Generate random proper motions within measurement errors
                sim_pmra = np.random.normal(0, pm_data['pmra_error'])
                sim_pmdec = np.random.normal(0, pm_data['pmdec_error'])

            # Calculate simulated frame dragging metrics

            # 1. Mean simulated effect magnitude
            sim_effect_mag = np.sqrt(sim_pmra**2 + sim_pmdec**2)
            sim_fd_effects[i] = sim_effect_mag.mean()

            # 2. Correlation with 1/r (frame dragging should correlate with 1/r^3)
            inv_r = 1.0 / pm_data['r_gc']
            sim_corr = np.corrcoef(inv_r, sim_effect_mag)[0, 1]
            sim_correlations[i] = sim_corr

            # 3. Alignment with expected frame dragging direction
            # Calculate the angle between observed and predicted frame dragging
            dot_product = sim_pmra * pm_data['fd_pmra'] + sim_pmdec * pm_data['fd_pmdec']
            sim_mag = np.sqrt(sim_pmra**2 + sim_pmdec**2)
            fd_mag = np.sqrt(pm_data['fd_pmra']**2 + pm_data['fd_pmdec']**2)

            # Avoid division by zero
            valid_idx = (sim_mag > 0) & (fd_mag > 0)
            if valid_idx.any():
                cos_angle = dot_product[valid_idx] / (sim_mag[valid_idx] * fd_mag[valid_idx])
                # Clip to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles = np.arccos(cos_angle)
                # Average alignment (1=perfect, 0=random, -1=anti-aligned)
                sim_alignments[i] = np.cos(angles).mean()
            else:
                sim_alignments[i] = 0

        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Monte Carlo simulations completed")
        print(f"[DEBUG] Monte Carlo simulations completed successfully")

        # Actual observed values from data
        actual_fd_effect = pm_data['fd_effect_mag'].mean()
        actual_correlation = np.corrcoef(1.0 / pm_data['r_gc'], pm_data['fd_effect_mag'])[0, 1]

        # Calculate alignment of actual proper motions with predicted frame dragging
        dot_product = pm_data['pmra'] * pm_data['fd_pmra'] + pm_data['pmdec'] * pm_data['fd_pmdec']
        pm_mag = np.sqrt(pm_data['pmra']**2 + pm_data['pmdec']**2)
        fd_mag = np.sqrt(pm_data['fd_pmra']**2 + pm_data['fd_pmdec']**2)

        # Avoid division by zero
        valid_idx = (pm_mag > 0) & (fd_mag > 0)
        cos_angle = dot_product[valid_idx] / (pm_mag[valid_idx] * fd_mag[valid_idx])
        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.arccos(cos_angle)
        actual_alignment = np.cos(angles).mean()

        print(f"[DEBUG] Actual values - FD effect: {actual_fd_effect:.6f}, Correlation: {actual_correlation:.6f}, Alignment: {actual_alignment:.6f}")

        # Return all simulation results
        results = {
            'simulated_fd_effects': sim_fd_effects,
            'simulated_correlations': sim_correlations,
            'simulated_alignments': sim_alignments,
            'actual_fd_effect': actual_fd_effect,
            'actual_correlation': actual_correlation,
            'actual_alignment': actual_alignment,
            'null_model': null_model
        }

        print(f"[DEBUG] run_monte_carlo_simulation completed successfully")
        return results

    except Exception as e:
        print(f"[ERROR] run_monte_carlo_simulation failed: {str(e)}")
        st.error(f"Error in Monte Carlo simulation: {str(e)}")
        # Return empty results
        return {
            'simulated_fd_effects': np.array([]),
            'simulated_correlations': np.array([]),
            'simulated_alignments': np.array([]),
            'actual_fd_effect': 0.0,
            'actual_correlation': 0.0,
            'actual_alignment': 0.0,
            'null_model': null_model
        }


def calculate_confidence_intervals(mc_results, confidence_level=0.95):
    """
    Calculate confidence intervals from Monte Carlo simulation results.

    Parameters:
    -----------
    mc_results : dict
        Dictionary with Monte Carlo simulation results
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% confidence)

    Returns:
    --------
    dict
        Dictionary with confidence intervals
    """
    print(f"[DEBUG] calculate_confidence_intervals called with confidence level {confidence_level}")

    try:
        # Validate input
        if not mc_results or 'simulated_fd_effects' not in mc_results:
            print(f"[ERROR] Invalid mc_results provided")
            raise ValueError("Invalid Monte Carlo results")

        # Calculate percentiles for confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100

        print(f"[DEBUG] Using percentiles: {lower_percentile}% to {upper_percentile}%")

        # Calculate confidence intervals for frame dragging effect
        fd_effect_ci_lower = np.percentile(mc_results['simulated_fd_effects'], lower_percentile)
        fd_effect_ci_upper = np.percentile(mc_results['simulated_fd_effects'], upper_percentile)

        # Calculate confidence intervals for correlation
        corr_ci_lower = np.percentile(mc_results['simulated_correlations'], lower_percentile)
        corr_ci_upper = np.percentile(mc_results['simulated_correlations'], upper_percentile)

        # Calculate confidence intervals for alignment
        align_ci_lower = np.percentile(mc_results['simulated_alignments'], lower_percentile)
        align_ci_upper = np.percentile(mc_results['simulated_alignments'], upper_percentile)

        # Return results
        ci_results = {
            'confidence_level': confidence_level,
            'fd_effect_ci_lower': fd_effect_ci_lower,
            'fd_effect_ci_upper': fd_effect_ci_upper,
            'correlation_ci_lower': corr_ci_lower,
            'correlation_ci_upper': corr_ci_upper,
            'alignment_ci_lower': align_ci_lower,
            'alignment_ci_upper': align_ci_upper
        }

        print(f"[DEBUG] Confidence intervals calculated:")
        print(f"[DEBUG] FD effect: [{fd_effect_ci_lower:.6f}, {fd_effect_ci_upper:.6f}]")
        print(f"[DEBUG] Correlation: [{corr_ci_lower:.6f}, {corr_ci_upper:.6f}]")
        print(f"[DEBUG] Alignment: [{align_ci_lower:.6f}, {align_ci_upper:.6f}]")

        return ci_results

    except Exception as e:
        print(f"[ERROR] calculate_confidence_intervals failed: {str(e)}")
        st.error(f"Error calculating confidence intervals: {str(e)}")
        # Return default values
        return {
            'confidence_level': confidence_level,
            'fd_effect_ci_lower': 0.0,
            'fd_effect_ci_upper': 0.0,
            'correlation_ci_lower': 0.0,
            'correlation_ci_upper': 0.0,
            'alignment_ci_lower': 0.0,
            'alignment_ci_upper': 0.0
        }


def compute_p_values(mc_results):
    """
    Compute p-values to assess the significance of frame dragging detection.

    Parameters:
    -----------
    mc_results : dict
        Dictionary with Monte Carlo simulation results

    Returns:
    --------
    dict
        Dictionary with p-values
    """
    print(f"[DEBUG] compute_p_values called")

    try:
        # Validate input
        if not mc_results or 'simulated_fd_effects' not in mc_results:
            print(f"[ERROR] Invalid mc_results provided")
            raise ValueError("Invalid Monte Carlo results")

        # Calculate p-value for frame dragging effect
        # Null hypothesis: observed effect is due to chance
        # p-value = proportion of simulations with effect >= observed
        fd_effect_p = np.mean(mc_results['simulated_fd_effects'] >= mc_results['actual_fd_effect'])
        print(f"[DEBUG] Frame dragging effect p-value: {fd_effect_p:.6f}")

        # Calculate p-value for correlation with 1/r
        # For correlation, we want abs(sim_corr) >= abs(actual_corr)
        # since strong negative correlation could also be significant
        corr_p = np.mean(np.abs(mc_results['simulated_correlations']) >= 
                        np.abs(mc_results['actual_correlation']))
        print(f"[DEBUG] Correlation p-value: {corr_p:.6f}")

        # Calculate p-value for alignment
        # Higher alignment value is more significant
        align_p = np.mean(mc_results['simulated_alignments'] >= mc_results['actual_alignment'])
        print(f"[DEBUG] Alignment p-value: {align_p:.6f}")

        # Calculate combined p-value using Fisher's method
        # -2 * sum(ln(p_i)) follows chi-squared with 2k degrees of freedom
        # where k is the number of p-values being combined

        # Handle p-values of 0 (set to minimum possible value based on simulations)
        min_p = 1.0 / (len(mc_results['simulated_fd_effects']) + 1)
        print(f"[DEBUG] Minimum possible p-value: {min_p:.6f}")

        fd_effect_p = max(fd_effect_p, min_p)
        corr_p = max(corr_p, min_p)
        align_p = max(align_p, min_p)

        # Combine p-values using Fisher's method
        fisher_stat = -2 * (np.log(fd_effect_p) + np.log(corr_p) + np.log(align_p))
        combined_p = 1 - stats.chi2.cdf(fisher_stat, 2 * 3)  # 2k degrees of freedom, k=3
        print(f"[DEBUG] Fisher statistic: {fisher_stat:.6f}, Combined p-value: {combined_p:.6f}")

        # Return results
        p_values = {
            'fd_effect_p_value': fd_effect_p,
            'correlation_p_value': corr_p,
            'alignment_p_value': align_p,
            'combined_p_value': combined_p
        }

        print(f"[DEBUG] compute_p_values completed successfully")
        return p_values

    except Exception as e:
        print(f"[ERROR] compute_p_values failed: {str(e)}")
        st.error(f"Error computing p-values: {str(e)}")
        # Return default values
        return {
            'fd_effect_p_value': 1.0,
            'correlation_p_value': 1.0,
            'alignment_p_value': 1.0,
            'combined_p_value': 1.0
        }


def perform_hypothesis_testing(p_values, alpha=0.05):
    """
    Perform hypothesis testing based on computed p-values.

    Parameters:
    -----------
    p_values : dict
        Dictionary with p-values
    alpha : float
        Significance level

    Returns:
    --------
    dict
        Dictionary with hypothesis test results
    """
    print(f"[DEBUG] perform_hypothesis_testing called with alpha={alpha}")

    try:
        # Validate input
        if not p_values:
            print(f"[ERROR] Invalid p_values provided")
            raise ValueError("Invalid p-values")

        # Test null hypothesis for each metric
        fd_effect_significant = p_values['fd_effect_p_value'] < alpha
        correlation_significant = p_values['correlation_p_value'] < alpha
        alignment_significant = p_values['alignment_p_value'] < alpha
        combined_significant = p_values['combined_p_value'] < alpha

        print(f"[DEBUG] Significance results:")
        print(f"[DEBUG] FD effect significant: {fd_effect_significant}")
        print(f"[DEBUG] Correlation significant: {correlation_significant}")
        print(f"[DEBUG] Alignment significant: {alignment_significant}")
        print(f"[DEBUG] Combined significant: {combined_significant}")

        # Overall assessment
        # Frame dragging is considered detected if the combined p-value is significant
        # and at least 2 out of 3 individual metrics are significant
        individual_significant = [fd_effect_significant, correlation_significant, alignment_significant]
        frame_dragging_detected = combined_significant and (sum(individual_significant) >= 2)

        print(f"[DEBUG] Individual metrics significant: {sum(individual_significant)}/3")
        print(f"[DEBUG] Frame dragging detected: {frame_dragging_detected}")

        # Detection strength assessment
        if p_values['combined_p_value'] < 0.001:
            detection_strength = "Strong evidence"
        elif p_values['combined_p_value'] < 0.01:
            detection_strength = "Substantial evidence"
        elif p_values['combined_p_value'] < 0.05:
            detection_strength = "Moderate evidence"
        else:
            detection_strength = "Weak or no evidence"

        print(f"[DEBUG] Detection strength: {detection_strength}")

        # Return results
        test_results = {
            'fd_effect_significant': fd_effect_significant,
            'correlation_significant': correlation_significant,
            'alignment_significant': alignment_significant,
            'combined_significant': combined_significant,
            'frame_dragging_detected': frame_dragging_detected,
            'detection_strength': detection_strength,
            'significance_level': alpha
        }

        print(f"[DEBUG] perform_hypothesis_testing completed successfully")
        return test_results

    except Exception as e:
        print(f"[ERROR] perform_hypothesis_testing failed: {str(e)}")
        st.error(f"Error in hypothesis testing: {str(e)}")
        # Return default values
        return {
            'fd_effect_significant': False,
            'correlation_significant': False,
            'alignment_significant': False,
            'combined_significant': False,
            'frame_dragging_detected': False,
            'detection_strength': "Error in analysis",
            'significance_level': alpha
        }