#!/usr/bin/env python3
"""
Test script for the universal rotation redshift hypothesis.
Run this to verify the universal_rotation_redshift module works correctly
before integrating with the main application.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Assuming your new code is in 'universal_rotation_redshift.py' or similar
# and contains UniversalRotationRedshift, generate_mock_gaia_data, apply_rotation_to_stellar_data
from universal_rotation_redshift import UniversalRotationRedshift, generate_mock_gaia_data, apply_rotation_to_stellar_data

# --- Default Parameters for Tests (can be overridden by fitting results) ---
# These are initial guesses and might need significant tuning based on model behavior
DEFAULT_A_ROT = 1.0  # Dimensionless coupling constant
DEFAULT_OMEGA_U_INV_GYR = 10.0 # Universal angular velocity in 1/Gyr (e.g., 1 rotation per ~628 Myr)
DEFAULT_R_OBSERVER_MPC = 10000 # Our distance from the rotation center
H0_FOR_COMPARISON = 70 # km/s/Mpc

# Store fitted parameters globally
fitted_params_global = {}

def test_basic_rotational_calculations():
    """Test basic rotational redshift calculations."""
    print("\n=== Testing Basic Rotational Redshift Calculations ===")
    urs = UniversalRotationRedshift() # URS = Universal Rotation Redshift

    # Use fitted params if available, else defaults
    A_rot = fitted_params_global.get('A_rot', DEFAULT_A_ROT)
    omega_U_inv_Gyr = fitted_params_global.get('omega_U_inv_Gyr', DEFAULT_OMEGA_U_INV_GYR)
    R_observer_Mpc = fitted_params_global.get('R_observer_Mpc', DEFAULT_R_OBSERVER_MPC)

    print(f"Using parameters: A_rot={A_rot:.2e}, omega_U={omega_U_inv_Gyr:.2f} /Gyr, R_observer={R_observer_Mpc:.0f} Mpc")

    # Test single distance of an object from the rotation center
    r_object_from_center_Mpc_single = R_observer_Mpc + 1000
    z_single = urs.rotational_redshift_model(
        r_object_from_center_Mpc_single * urs.Mpc_to_m,
        R_observer_Mpc * urs.Mpc_to_m,
        A_rot,
        omega_U_inv_Gyr
    )
    print(f"Rotational redshift for object at {r_object_from_center_Mpc_single:.0f} Mpc from Center: z = {z_single:.6e}")

    # Test array of distances of objects from the rotation center
    object_distances_from_center_Mpc = np.logspace(
        np.log10(max(100, R_observer_Mpc / 10)), # Avoid too close to center if formula behaves poorly
        np.log10(R_observer_Mpc * 2 + 5000), # Extend a bit beyond
        50) 

    redshifts = urs.rotational_redshift_model(
        object_distances_from_center_Mpc * urs.Mpc_to_m,
        R_observer_Mpc * urs.Mpc_to_m,
        A_rot,
        omega_U_inv_Gyr
    )

    plt.figure(figsize=(10, 6))
    plt.plot(object_distances_from_center_Mpc, redshifts, 'b.-', linewidth=2, markersize=5) # loglog might hide detail if z is not always positive
    plt.axvline(R_observer_Mpc, color='r', linestyle='--', label=f'Observer Dist from Center ({R_observer_Mpc:.0f} Mpc)')
    plt.xlabel('Object Distance from Universal Rotation Center (Mpc)')
    plt.ylabel('Predicted Rotational Redshift (z_rot)')
    plt.title('Rotational Redshift vs. Object Distance from Center')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    if redshifts.max() > 1e-9 : plt.yscale('symlog', linthresh=1e-5) # Use symlog if wide range
    plt.savefig('test_rot_redshift.png', dpi=150, bbox_inches='tight')
    plt.close()

    min_z_val, max_z_val = redshifts.min(), redshifts.max()
    print(f"Redshift range for objects {object_distances_from_center_Mpc.min():.0f}-{object_distances_from_center_Mpc.max():.0f} Mpc from Center: {min_z_val:.3e} to {max_z_val:.3e}")
    print("Saved plot: test_rot_redshift.png")

def test_hubble_law_comparison_rotation():
    """Test comparison with Hubble's law using the rotational model."""
    print("\n=== Testing Hubble Law Comparison (Rotational Model) ===")
    urs = UniversalRotationRedshift()

    A_rot = fitted_params_global.get('A_rot', DEFAULT_A_ROT)
    omega_U_inv_Gyr = fitted_params_global.get('omega_U_inv_Gyr', DEFAULT_OMEGA_U_INV_GYR)
    R_observer_Mpc = fitted_params_global.get('R_observer_Mpc', DEFAULT_R_OBSERVER_MPC)
    print(f"Using parameters: A_rot={A_rot:.2e}, omega_U={omega_U_inv_Gyr:.2f} /Gyr, R_observer={R_observer_Mpc:.0f} Mpc")

    # These are distances of objects FROM THE OBSERVER (Earth)
    object_distances_from_observer_Mpc = np.linspace(10, 4000, 100)

    v_rot, z_rot = urs.hubble_law_from_rotation(
        object_distances_from_observer_Mpc, A_rot, omega_U_inv_Gyr, R_observer_Mpc
    )
    v_hubble = H0_FOR_COMPARISON * object_distances_from_observer_Mpc

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(object_distances_from_observer_Mpc, v_hubble, 'b-', label=f'Std. Hubble Law (H0={H0_FOR_COMPARISON})', linewidth=2)
    plt.plot(object_distances_from_observer_Mpc, v_rot, 'r.--', label='Rotational Model', linewidth=2, markersize=5)
    plt.xlabel('Distance from Observer (Mpc)')
    plt.ylabel('Apparent Recession Velocity (km/s)')
    plt.title('Velocity-Distance (Observer Frame)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = np.zeros_like(v_rot)
    valid_res_mask = np.abs(v_hubble) > 1e-3 # Avoid division by zero
    residuals[valid_res_mask] = (v_rot[valid_res_mask] - v_hubble[valid_res_mask]) / v_hubble[valid_res_mask] * 100

    plt.plot(object_distances_from_observer_Mpc[valid_res_mask], residuals[valid_res_mask], 'g.-', linewidth=2, markersize=5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Distance from Observer (Mpc)')
    plt.ylabel('Residual (%)')
    plt.title('Rotational Model vs Hubble Law Residuals')
    plt.grid(True, alpha=0.3)
    # Dynamic y-lim based on actual residuals
    if np.any(valid_res_mask):
      min_res, max_res = residuals[valid_res_mask].min(), residuals[valid_res_mask].max()
      plt.ylim(min(min_res*1.1 if min_res < 0 else -10, -10), 
              max(max_res*1.1 if max_res > 0 else 10, 10) )


    plt.tight_layout()
    plt.savefig('test_rot_hubble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    if np.any(valid_res_mask):
        print(f"Max velocity difference: {np.max(np.abs(v_rot[valid_res_mask] - v_hubble[valid_res_mask])):.1f} km/s")
        rms_res = np.sqrt(np.mean(residuals[valid_res_mask]**2))
        print(f"RMS residual: {rms_res:.1f}%")
    else:
        print("Could not calculate residuals (v_hubble likely zero).")
    print("Saved plot: test_rot_hubble_comparison.png")

def test_rotational_parameter_fitting():
    """Test rotational parameter fitting to mock SNe data."""
    global fitted_params_global
    print("\n=== Testing Rotational Parameter Fitting (Mock SNe Data) ===")
    urs = UniversalRotationRedshift()

    n_sne = 100
    true_distances_from_observer_Mpc = np.logspace(np.log10(30), np.log10(3500), n_sne)
    distance_errors_fraction = 0.1
    obs_distances_from_observer_Mpc = true_distances_from_observer_Mpc * (1 + np.random.normal(0, distance_errors_fraction, n_sne))
    obs_distances_from_observer_Mpc = np.maximum(1.0, obs_distances_from_observer_Mpc)

    v_mock = H0_FOR_COMPARISON * true_distances_from_observer_Mpc
    obs_z_true = v_mock / (urs.c / 1000)
    z_scatter = 0.02
    obs_z = obs_z_true + np.random.normal(0, z_scatter, n_sne)
    obs_z = np.maximum(0.001, obs_z)

    print(f"Generated {n_sne} mock Type Ia supernovae.")
    print(f"Observed Distance (from Earth) range: {obs_distances_from_observer_Mpc.min():.0f} to {obs_distances_from_observer_Mpc.max():.0f} Mpc")
    print(f"Observed Redshift range: {obs_z.min():.3f} to {obs_z.max():.3f}")

    print("\nFitting Universal Rotational parameters...")
    initial_R_observer_guess = DEFAULT_R_OBSERVER_MPC
    try:
        best_params = urs.fit_rotational_parameters(obs_z, obs_distances_from_observer_Mpc,
                                                    R_observer_guess_Mpc=initial_R_observer_guess)
        print("\nBest-fit rotational parameters from mock SNe data:")
        print(f"  Log10(A_rot): {best_params['log10_A_rot']:.3f} (A_rot: {best_params['A_rot']:.2e})")
        print(f"  Log10(omega_U_inv_Gyr): {best_params['log10_omega_U_inv_Gyr']:.3f} (omega_U: {best_params['omega_U_inv_Gyr']:.2f} /Gyr)")
        print(f"  Earth's Distance from Rotation Center (R_observer): {best_params['R_observer_Mpc']:.0f} Mpc")
        print(f"  Chi-squared: {best_params['chi2']:.2f}")
        print(f"  Fit successful: {best_params['success']}")

        if best_params['success']:
            fitted_params_global = best_params
            print("Stored fitted parameters for subsequent tests.")
        else:
            print("Fit was not successful. Subsequent tests will use default parameters.")
            fitted_params_global = {}
    except Exception as e:
        print(f"Error during parameter fitting: {e}")
        print("Subsequent tests will use default parameters.")
        fitted_params_global = {}

def test_cmb_predictions_kinematic(): # Name changed for clarity
    """Test CMB dipole predictions (kinematic due to peculiar velocity)."""
    print("\n=== Testing CMB Dipole Predictions (Kinematic) ===")
    urs = UniversalRotationRedshift() # Instantiation needed for constants like c

    # R_observer_Mpc is not directly used by the kinematic dipole calculation in the new model
    # R_observer_Mpc = fitted_params_global.get('R_observer_Mpc', DEFAULT_R_OBSERVER_MPC)
    # print(f"Using Earth's distance from Rotation Center (R_observer): {R_observer_Mpc:.0f} Mpc (for context only)")

    peculiar_velocity_sun_kms = 369.8
    print(f"Solar peculiar velocity: {peculiar_velocity_sun_kms:.1f} km/s")

    cmb_pred = urs.calculate_cmb_dipole_prediction(peculiar_velocity_kms=peculiar_velocity_sun_kms)

    kinematic_dipole_mK = cmb_pred['kinematic_dipole_amplitude_mK']
    print(f"Predicted MAX kinematic CMB dipole amplitude: {kinematic_dipole_mK:.3f} mK")

    observed_cmb_dipole_mK = 3.3621
    print(f"Observed CMB dipole amplitude: {observed_cmb_dipole_mK:.4f} ± 0.0010 mK")

    # Plotting logic for T_obs vs angle remains the same as it's standard kinematics
    angles_relative_to_v_pec = np.linspace(0, 180, 90)
    T_cmb_K = 2.725
    beta = peculiar_velocity_sun_kms * 1000 / urs.c
    gamma = 1 / np.sqrt(1 - beta**2)
    T_observed_K = T_cmb_K / (gamma * (1 - beta * np.cos(np.radians(angles_relative_to_v_pec))))
    delta_T_mK_plot = (T_observed_K - T_cmb_K) * 1000

    plt.figure(figsize=(8, 6))
    plt.plot(angles_relative_to_v_pec, delta_T_mK_plot, 'b-', linewidth=2)
    plt.axhline(y=observed_cmb_dipole_mK, color='r', linestyle='--', label=f'Observed Max Amp ({observed_cmb_dipole_mK:.3f} mK)')
    plt.axhline(y=-observed_cmb_dipole_mK, color='r', linestyle='--')
    plt.xlabel('Angle from Peculiar Velocity Vector (degrees)')
    plt.ylabel('CMB Temperature Fluctuation (mK)')
    plt.title('Predicted Kinematic CMB Dipole vs. Observation Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_cmb_dipole_kinematic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot: test_cmb_dipole_kinematic.png")

def test_stellar_rotation_integration():
    """Test integration with mock Gaia-like stellar data using rotational redshift."""
    print("\n=== Testing Rotational Redshift with Mock Gaia-like Stellar Data ===")
    urs = UniversalRotationRedshift()

    A_rot = fitted_params_global.get('A_rot', DEFAULT_A_ROT)
    omega_U_inv_Gyr = fitted_params_global.get('omega_U_inv_Gyr', DEFAULT_OMEGA_U_INV_GYR)
    R_observer_Mpc = fitted_params_global.get('R_observer_Mpc', DEFAULT_R_OBSERVER_MPC)
    print(f"Using parameters: A_rot={A_rot:.2e}, omega_U={omega_U_inv_Gyr:.2f} /Gyr, R_observer={R_observer_Mpc:.0f} Mpc")

    num_stars = 500
    mock_stars_df = generate_mock_gaia_data(num_stars=num_stars, dist_range_pc=(100, 10000))
    print(f"Generated {len(mock_stars_df)} mock stars (distances from Earth: "
          f"{mock_stars_df['distance_pc'].min():.0f} - {mock_stars_df['distance_pc'].max():.0f} pc).")

    stars_with_rot_df = apply_rotation_to_stellar_data(
        mock_stars_df, urs, A_rot, omega_U_inv_Gyr, R_observer_Mpc
    )
    print("Applied rotational model to mock stellar data. Sample output:")
    print(stars_with_rot_df[['l_deg', 'b_deg', 'distance_pc', 'dist_to_center_Mpc', 'z_rot']].head())

    plt.figure(figsize=(12, 7))

    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(stars_with_rot_df['distance_pc'] / 1e3, stars_with_rot_df['z_rot'], 
                     c=stars_with_rot_df['dist_to_center_Mpc'], cmap='viridis', s=15, alpha=0.7)
    cbar1 = plt.colorbar(sc1, label='Distance to Rotation Center (Mpc)')
    plt.xlabel('Distance from Earth (kpc)')
    plt.ylabel('Predicted Rotational Redshift (z_rot)')
    plt.title('Rotational Redshift for Mock Stars')
    plt.grid(True, alpha=0.3)
    min_z, max_z = stars_with_rot_df['z_rot'].min(), stars_with_rot_df['z_rot'].max()
    # Adjust y-scale and limits based on z_rot values
    if not np.all(np.isclose(min_z, max_z)): # Avoid issues if all z are same
        if max_z > 1e-9 or min_z < -1e-9: plt.yscale('symlog', linthresh=1e-5)
        plt.ylim(min_z - 0.1*abs(min_z) if min_z!=0 else -0.01, max_z + 0.1*abs(max_z) if max_z!=0 else 0.01)
    else: plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    plt.subplot(1, 2, 2)
    sorted_df = stars_with_rot_df.sort_values(by='dist_to_center_Mpc')
    sc2 = plt.scatter(sorted_df['dist_to_center_Mpc'], sorted_df['z_rot'], 
                     c=sorted_df['distance_pc']/1e3, cmap='coolwarm', s=15, alpha=0.7)
    cbar2 = plt.colorbar(sc2, label='Distance from Earth (kpc)')
    plt.axvline(R_observer_Mpc, color='k', linestyle='--', label=f'Observer Dist from Center ({R_observer_Mpc:.0f} Mpc)')
    plt.xlabel('Distance from Rotation Center (Mpc)')
    plt.ylabel('Predicted Rotational Redshift (z_rot)')
    plt.title('z_rot vs. Dist from Rotation Center')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if not np.all(np.isclose(min_z, max_z)):
        if max_z > 1e-9 or min_z < -1e-9: plt.yscale('symlog', linthresh=1e-5)
        plt.ylim(min_z - 0.1*abs(min_z) if min_z!=0 else -0.01, max_z + 0.1*abs(max_z) if max_z!=0 else 0.01)
    else: plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    plt.suptitle(f'Rotational Model (A_rot={A_rot:.1e}, omega_U={omega_U_inv_Gyr:.1f}/Gyr, R_obs={R_observer_Mpc:.0f} Mpc)', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('test_rot_stellar_redshifts.png', dpi=150)
    print("Saved plot: test_rot_stellar_redshifts.png")
    plt.close()

    if stars_with_rot_df['z_rot'].isnull().any() or np.isinf(stars_with_rot_df['z_rot']).any():
        print("WARNING: NaNs or Infs found in predicted stellar redshifts (z_rot).")
        print(stars_with_rot_df[stars_with_rot_df['z_rot'].isnull() | np.isinf(stars_with_rot_df['z_rot'])])

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Universal Rotation Redshift Hypothesis") # Updated title
    print("=" * 60)

    # Run tests in order
    test_rotational_parameter_fitting() 
    test_basic_rotational_calculations()
    test_hubble_law_comparison_rotation()
    test_cmb_predictions_kinematic()
    test_stellar_rotation_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)

if __name__ == "__main__":
    main()