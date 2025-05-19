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
    # Copy key data to avoid modifying original
    pm_data = df_results[['pmra', 'pmdec', 'pmra_error', 'pmdec_error', 
                         'fd_pmra', 'fd_pmdec', 'fd_effect_mag',
                         'r_gc']].copy()
    
    # Initialize arrays to store simulation results
    sim_fd_effects = np.zeros(n_simulations)
    sim_correlations = np.zeros(n_simulations)
    sim_alignments = np.zeros(n_simulations)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulations
    for i in range(n_simulations):
        if i % 10 == 0:
            status_text.text(f"Running simulation {i+1} of {n_simulations}...")
            progress_bar.progress((i+1) / n_simulations)
        
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
    
    return results


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
    # Calculate percentiles for confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    
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
    
    return ci_results


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
    # Calculate p-value for frame dragging effect
    # Null hypothesis: observed effect is due to chance
    # p-value = proportion of simulations with effect >= observed
    fd_effect_p = np.mean(mc_results['simulated_fd_effects'] >= mc_results['actual_fd_effect'])
    
    # Calculate p-value for correlation with 1/r
    # For correlation, we want abs(sim_corr) >= abs(actual_corr)
    # since strong negative correlation could also be significant
    corr_p = np.mean(np.abs(mc_results['simulated_correlations']) >= 
                    np.abs(mc_results['actual_correlation']))
    
    # Calculate p-value for alignment
    # Higher alignment value is more significant
    align_p = np.mean(mc_results['simulated_alignments'] >= mc_results['actual_alignment'])
    
    # Calculate combined p-value using Fisher's method
    # -2 * sum(ln(p_i)) follows chi-squared with 2k degrees of freedom
    # where k is the number of p-values being combined
    
    # Handle p-values of 0 (set to minimum possible value based on simulations)
    min_p = 1.0 / (len(mc_results['simulated_fd_effects']) + 1)
    
    fd_effect_p = max(fd_effect_p, min_p)
    corr_p = max(corr_p, min_p)
    align_p = max(align_p, min_p)
    
    # Combine p-values using Fisher's method
    fisher_stat = -2 * (np.log(fd_effect_p) + np.log(corr_p) + np.log(align_p))
    combined_p = 1 - stats.chi2.cdf(fisher_stat, 2 * 3)  # 2k degrees of freedom, k=3
    
    # Return results
    p_values = {
        'fd_effect_p_value': fd_effect_p,
        'correlation_p_value': corr_p,
        'alignment_p_value': align_p,
        'combined_p_value': combined_p
    }
    
    return p_values


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
    # Test null hypothesis for each metric
    fd_effect_significant = p_values['fd_effect_p_value'] < alpha
    correlation_significant = p_values['correlation_p_value'] < alpha
    alignment_significant = p_values['alignment_p_value'] < alpha
    combined_significant = p_values['combined_p_value'] < alpha
    
    # Overall assessment
    # Frame dragging is considered detected if the combined p-value is significant
    # and at least 2 out of 3 individual metrics are significant
    individual_significant = [fd_effect_significant, correlation_significant, alignment_significant]
    frame_dragging_detected = combined_significant and (sum(individual_significant) >= 2)
    
    # Detection strength assessment
    if p_values['combined_p_value'] < 0.001:
        detection_strength = "Strong evidence"
    elif p_values['combined_p_value'] < 0.01:
        detection_strength = "Substantial evidence"
    elif p_values['combined_p_value'] < 0.05:
        detection_strength = "Moderate evidence"
    else:
        detection_strength = "Weak or no evidence"
    
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
    
    return test_results
