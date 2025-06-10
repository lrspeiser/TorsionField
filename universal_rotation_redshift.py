# universal_rotation_redshift.py

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not strictly needed in the module for calculations
from scipy import integrate, optimize # optimize is used
# from scipy.interpolate import interp1d # Not currently used in this version

# --- Helper Functions ---
def galactic_to_cartesian(l, b, distance):
    """
    Convert Galactic coordinates (l, b, distance) to Cartesian coordinates.
    Assumes Earth is at the origin.
    l, b in degrees. distance in arbitrary units (e.g., pc or Mpc).
    Output: x, y, z in the same units as distance.
    Standard convention: x towards Galactic Center, y in direction of rotation, z towards North Galactic Pole.
    """
    # print(f"DEBUG: galactic_to_cartesian called with l={l}, b={b}, distance={distance}") # Optional debug
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    x = distance * np.cos(b_rad) * np.cos(l_rad)
    y = distance * np.cos(b_rad) * np.sin(l_rad)
    z = distance * np.sin(b_rad)
    return x, y, z

def cartesian_to_galactic(x, y, z):
    """
    Convert Cartesian coordinates (x,y,z) to Galactic coordinates (l,b,distance)
    """
    # print(f"DEBUG: cartesian_to_galactic called with x={x}, y={y}, z={z}") # Optional debug
    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance == 0: return 0,0,0 # Handle origin case
    # Ensure z/distance is within [-1, 1] for arcsin due to potential floating point inaccuracies
    ratio = np.clip(z / distance, -1.0, 1.0)
    b_rad = np.arcsin(ratio)
    l_rad = np.arctan2(y, x)
    l_deg = np.degrees(l_rad) % 360 # Ensure l is in [0, 360)
    b_deg = np.degrees(b_rad)
    return l_deg, b_deg, distance

class UniversalRotationRedshift:
    """
    Theoretical framework for cosmological redshift caused by global
    rotation of the universe, inspired by Gödel-like concepts.
    Redshift is NOT due to Doppler effect from expansion or peculiar motion in this context,
    but from photon propagation in a rotating spacetime.
    """

    def __init__(self):
        print("DEBUG: UniversalRotationRedshift.__init__() called")
        # Physical constants
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.c = 299792458.0  # m/s
        self.pc_to_m = 3.0857e16  # meters
        self.Mpc_to_m = 3.0857e22  # meters
        self.year_to_s = 365.25 * 24 * 3600  # seconds
        self.s_to_Gyr = 1 / (1e9 * self.year_to_s) # For omega_U units

        # CMB "Axis of Evil" direction - hypothesized axis/center of Universal Rotation
        self.rot_axis_l_deg = 260.0
        self.rot_axis_b_deg = 60.0
        print(f"DEBUG: Rotation axis set to l={self.rot_axis_l_deg}, b={self.rot_axis_b_deg}")

        # Cartesian unit vector pointing towards the center of rotation from Earth
        uc_x, uc_y, uc_z = galactic_to_cartesian(self.rot_axis_l_deg, self.rot_axis_b_deg, 1.0)
        self.rotation_center_direction_vector = np.array([uc_x, uc_y, uc_z])
        print(f"DEBUG: Rotation center direction vector (from Earth): {self.rotation_center_direction_vector}")


    def rotational_redshift_model(self,
                                 r_object_to_center_m,
                                 R_observer_to_center_m,
                                 A_rot, # Dimensionless rotation coupling constant
                                 omega_U_inv_Gyr, # Universal angular velocity in 1/Gyr
                                 debug_log=False): 
        if debug_log:
            print(f"DEBUG: rotational_redshift_model called with:")
            print(f"  r_object_to_center_m (sample): {r_object_to_center_m if np.isscalar(r_object_to_center_m) else r_object_to_center_m[:min(3, len(r_object_to_center_m))]} m")
            print(f"  R_observer_to_center_m: {R_observer_to_center_m:.3e} m")
            print(f"  A_rot: {A_rot:.3e}")
            print(f"  omega_U_inv_Gyr: {omega_U_inv_Gyr:.3f} /Gyr")

        if np.isscalar(r_object_to_center_m):
            r_object_to_center_m = np.array([r_object_to_center_m])
            scalar_input = True
        else:
            scalar_input = False
            r_object_to_center_m = np.asarray(r_object_to_center_m)

        omega_U_inv_s = omega_U_inv_Gyr / (1e9 * self.year_to_s)
        if debug_log: print(f"  omega_U_inv_s: {omega_U_inv_s:.3e} rad/s")

        def z_potential(r_m, A_rot_val, omega_U_val_inv_s):
            term_v_over_c = (r_m * omega_U_val_inv_s / self.c)
            # if debug_log: print(f"    DEBUG z_potential: r_m={r_m:.2e}, (r*omega/c)={term_v_over_c:.2e}, (r*omega/c)^2={(term_v_over_c**2):.2e}")
            return A_rot_val * term_v_over_c**2

        z_source_pot = z_potential(r_object_to_center_m, A_rot, omega_U_inv_s)
        z_observer_pot = z_potential(R_observer_to_center_m, A_rot, omega_U_inv_s)

        if debug_log:
            print(f"  z_source_pot (sample): {z_source_pot if np.isscalar(z_source_pot) else z_source_pot[:min(3, len(z_source_pot))]}")
            print(f"  z_observer_pot: {z_observer_pot:.3e}")

        z = np.abs(z_source_pot - z_observer_pot)
        if debug_log: print(f"  Predicted z (sample): {z if np.isscalar(z) else z[:min(3, len(z))]}")

        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print("WARNING: NaN or Inf detected in rotational_redshift_model output z!")
            print(f"  Input r_object_to_center_m (sample): {r_object_to_center_m if np.isscalar(r_object_to_center_m) else r_object_to_center_m[:min(3, len(r_object_to_center_m))]}")
            print(f"  Input R_observer_to_center_m: {R_observer_to_center_m}")
            print(f"  Input A_rot: {A_rot}, omega_U_inv_Gyr: {omega_U_inv_Gyr}")
            z = np.nan_to_num(z, nan=1e9, posinf=1e9, neginf=-1e9) # Replace with large finite number

        if scalar_input:
            return z[0]
        return z

    def hubble_law_from_rotation(self,
                                 object_distances_from_observer_Mpc,
                                 A_rot, omega_U_inv_Gyr, R_observer_to_center_Mpc):
        print("DEBUG: hubble_law_from_rotation called")
        r_objects_to_center_Mpc = R_observer_to_center_Mpc + object_distances_from_observer_Mpc
        r_obj_m = r_objects_to_center_Mpc * self.Mpc_to_m
        R_obs_m = R_observer_to_center_Mpc * self.Mpc_to_m
        z = self.rotational_redshift_model(r_obj_m, R_obs_m, A_rot, omega_U_inv_Gyr, debug_log=False) # Limit debug for many calls
        v_recession = self.c * ((1 + z)**2 - 1) / ((1 + z)**2 + 1) / 1000 # km/s
        return v_recession, z

    def fit_rotational_parameters(self,
                                  observed_z,
                                  observed_distances_from_observer_Mpc,
                                  R_observer_guess_Mpc=10000):
        print(f"DEBUG: fit_rotational_parameters called with R_observer_guess_Mpc={R_observer_guess_Mpc}")

        iteration_count = [0] # Using a list to modify in nested scope

        def objective(params):
            iteration_count[0] += 1
            log10_A_rot, log10_omega_U_inv_Gyr, R_observer_Mpc = params
            A_rot = 10**log10_A_rot
            omega_U_inv_Gyr = 10**log10_omega_U_inv_Gyr

            if iteration_count[0] % 50 == 0: # Log every 50 iterations
                print(f"  DEBUG objective iter {iteration_count[0]}: log10A={log10_A_rot:.2f}, log10om={log10_omega_U_inv_Gyr:.2f}, Robs={R_observer_Mpc:.0f} Mpc")

            if not (100 <= R_observer_Mpc <= 20000):
                 if iteration_count[0] % 50 == 0: print(f"    R_observer_Mpc out of bounds: {R_observer_Mpc}")
                 return 1e12 # Penalty for out of bounds

            r_objects_to_center_Mpc = R_observer_Mpc + observed_distances_from_observer_Mpc
            r_obj_m = r_objects_to_center_Mpc * self.Mpc_to_m
            R_obs_m = R_observer_Mpc * self.Mpc_to_m
            z_pred = self.rotational_redshift_model(r_obj_m, R_obs_m, A_rot, omega_U_inv_Gyr, debug_log=(iteration_count[0] % 200 == 0)) # More detailed log less frequently

            if np.any(np.isnan(z_pred)) or np.any(np.isinf(z_pred)):
                print(f"WARNING objective: NaN/Inf in z_pred for params: A_rot={A_rot:.2e}, omega={omega_U_inv_Gyr:.2f}, Robs={R_observer_Mpc:.0f}")
                return 1e15 # Very large penalty

            weights = 1.0 / (0.01 + np.abs(observed_z))**2
            chi2 = np.sum(weights * (observed_z - z_pred)**2)
            if iteration_count[0] % 50 == 0: print(f"    chi2: {chi2:.2f}")
            if np.isnan(chi2) or np.isinf(chi2):
                print(f"ERROR objective: chi2 is NaN/Inf!")
                return 1e20 # Max penalty
            return chi2

        x0 = [np.log10(1.0), np.log10(10.0), R_observer_guess_Mpc]
        print(f"DEBUG: Initial guess x0 for optimizer: {x0}")
        bounds = [(-6, 6), (-2, 3), (1000, 15000)] # log10(A_rot), log10(omega_U_inv_Gyr), R_observer_Mpc
        print(f"DEBUG: Bounds for optimizer: {bounds}")

        result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B', options={'disp': True, 'maxiter': 500}) # Display optimizer messages, increase maxiter

        print(f"DEBUG: Optimization result: {result}")
        best_params = {
            'log10_A_rot': result.x[0],
            'A_rot': 10**result.x[0],
            'log10_omega_U_inv_Gyr': result.x[1],
            'omega_U_inv_Gyr': 10**result.x[1],
            'R_observer_Mpc': result.x[2],
            'chi2': result.fun,
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev
        }
        return best_params

    def calculate_cmb_dipole_prediction(self, peculiar_velocity_kms):
        print("DEBUG: calculate_cmb_dipole_prediction called")
        v = peculiar_velocity_kms * 1000
        if v >= self.c: v = 0.99 * self.c
        T_cmb = 2.725
        beta = v / self.c
        delta_T_mK = T_cmb * beta * 1000
        return {
            'peculiar_velocity_kms': peculiar_velocity_kms,
            'beta': beta,
            'kinematic_dipole_amplitude_mK': delta_T_mK,
            'expected_dipole_direction': "Aligned with peculiar velocity vector"
        }

# --- Global Scope Helper Functions for Data Handling ---
# These are defined at the module level so they can be imported.

def generate_mock_gaia_data(num_stars=1000, dist_range_pc=(10, 10000)):
    """
    Generates a DataFrame of mock stellar data similar to Gaia.
    """
    print(f"DEBUG: generate_mock_gaia_data called for {num_stars} stars, range {dist_range_pc} pc")
    l_deg = np.random.uniform(0, 360, num_stars)
    sin_b = np.random.uniform(-1, 1, num_stars) # Correct way to get uniform points on sphere
    b_deg = np.degrees(np.arcsin(sin_b))

    # Log-uniform distribution of distances for better spread
    log_dist_min, log_dist_max = np.log10(dist_range_pc[0]), np.log10(dist_range_pc[1])
    distances_pc = 10**np.random.uniform(log_dist_min, log_dist_max, num_stars)
    parallax_mas = 1000.0 / distances_pc

    # Simulate parallax errors (e.g., 10% of parallax for faint/distant, smaller for bright/nearby)
    parallax_error_mas = parallax_mas * np.random.uniform(0.01, 0.2, num_stars)

    df = pd.DataFrame({
        'source_id': range(num_stars),
        'l_deg': l_deg,
        'b_deg': b_deg,
        'parallax_mas': parallax_mas,
        'parallax_error_mas': parallax_error_mas,
        'distance_pc': distances_pc
    })
    return df

def apply_rotation_to_stellar_data(stellar_df, rot_model_instance,
                                   A_rot, omega_U_inv_Gyr, R_observer_Mpc):
    """
    Applies the Universal Rotation model to stellar data.
    Calculates predicted redshift for each star.
    """
    print(f"DEBUG: apply_rotation_to_stellar_data called for {len(stellar_df)} stars")
    print(f"  Using A_rot={A_rot:.2e}, omega_U={omega_U_inv_Gyr:.2f}/Gyr, R_obs={R_observer_Mpc:.0f} Mpc")
    df = stellar_df.copy()

    # Vector from Rotation Center to Earth.
    # rot_model_instance.rotation_center_direction_vector points from Earth TO Center.
    # So, if Center is origin, Earth is at -R_observer_Mpc * (direction_Earth_to_Center)
    vec_earth_coords_if_center_is_origin_Mpc = -rot_model_instance.rotation_center_direction_vector * R_observer_Mpc

    # Star positions relative to Earth (Cartesian, Mpc)
    star_x_earth_Mpc, star_y_earth_Mpc, star_z_earth_Mpc = galactic_to_cartesian(
        df['l_deg'].values, df['b_deg'].values, df['distance_pc'].values / 1e6 # convert pc to Mpc
    )
    # Vector from Earth to Star
    vec_star_coords_if_earth_is_origin_Mpc = np.vstack((star_x_earth_Mpc, star_y_earth_Mpc, star_z_earth_Mpc)).T

    # Vector from Rotation Center to Star = vec(Earth_coords_if_Center_is_Origin) + vec(Star_coords_if_Earth_is_Origin)
    vec_star_coords_if_center_is_origin_Mpc = vec_earth_coords_if_center_is_origin_Mpc + vec_star_coords_if_earth_is_origin_Mpc

    dist_stars_to_center_Mpc = np.linalg.norm(vec_star_coords_if_center_is_origin_Mpc, axis=1)
    df['dist_to_center_Mpc'] = dist_stars_to_center_Mpc

    R_observer_m = R_observer_Mpc * rot_model_instance.Mpc_to_m
    dist_stars_to_center_m = dist_stars_to_center_Mpc * rot_model_instance.Mpc_to_m

    df['z_rot'] = rot_model_instance.rotational_redshift_model(
        dist_stars_to_center_m, R_observer_m, A_rot, omega_U_inv_Gyr, debug_log=False # Limit debug here
    )
    if df['z_rot'].isnull().any() or np.isinf(df['z_rot']).any():
        print("WARNING in apply_rotation_to_stellar_data: z_rot contains NaNs or Infs AFTER model call.")
    return df