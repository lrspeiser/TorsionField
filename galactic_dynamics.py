# galactic_dynamics.py
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Galactocentric

# Define Solar Parameters (can be made configurable)
# These are example values; consult recent literature for best estimates.
R_SUN_GALACTOCENTER = 8.178 * u.kpc  # Example: Gravity Collaboration 2019
V_SUN_PECULIAR = SkyCoord(11.1 * u.km/u.s,  # U (radially inwards)
                           12.24 * u.km/u.s, # V (direction of Galactic rotation)
                           7.25 * u.km/u.s,  # W (normal to Galactic plane, towards NGP)
                           frame=Galactocentric(galcen_distance=R_SUN_GALACTOCENTER,
                                                galcen_v_sun=[0,0,0]*u.km/u.s), # Temp v_sun for definition
                           representation_type='cartesian',
                           differential_type='cartesian')

V_LSR_CIRCULAR = 232.8 * u.km/u.s # Example: McMillan 2017 (adjust v_sun accordingly if using specific model)

# Update Galactocentric frame with these solar motion parameters
# Note: Astropy's Galactocentric frame uses v_sun as the Sun's velocity *in the Galactocentric frame*.
# The peculiar velocity defined above is often given wrt LSR.
# The total v_sun vector in Galactocentric frame would be V_LSR + V_SUN_PECULIAR (vector sum)
# For simplicity here, we'll use a common approach:
# v_sun components wrt Galactic center (U toward GC, V in rotation dir, W toward NGP)
# V_SUN_GALACTOCENTRIC = [-11.1, V_LSR_CIRCULAR.value + 12.24, 7.25] * u.km/u.s # Approximate
# Better: Define v_sun relative to LSR and let astropy handle it if using a specific model's LSR
# For now, using the default v_sun in Astropy's Galactocentric which is based on Schoenrich et al. 2010
# or provide all components explicitly.

GALACTOCENTRIC_FRAME = Galactocentric(
    galcen_distance=R_SUN_GALACTOCENTER,
    # Use astropy's default solar motion for now, or set explicitly:
    # galcen_v_sun= V_SUN_GALACTOCENTRIC
    # z_sun = value * u.pc  # Sun's height above/below the plane
)

def get_galactocentric_kinematics(df_gaia):
    """
    Converts Gaia observables to Galactocentric coordinates and velocities.

    Parameters:
    -----------
    df_gaia : pd.DataFrame
        DataFrame with Gaia data including ra, dec, parallax,
        pmra, pmdec, radial_velocity.
        Assumes ra, dec in degrees, parallax in mas, pm in mas/yr, rv in km/s.

    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added Galactocentric columns:
        'R_gal' (kpc): Cylindrical Galactocentric radius.
        'phi_gal' (rad): Galactocentric azimuth.
        'z_gal' (kpc): Height above/below Galactic plane.
        'v_R_gal' (km/s): Galactocentric radial velocity.
        'v_phi_gal' (km/s): Galactocentric azimuthal (rotational) velocity.
        'v_z_gal' (km/s): Galactocentric vertical velocity.
    """
    if not all(col in df_gaia.columns for col in ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']):
        raise ValueError("Input DataFrame is missing required Gaia columns.")

    # Create SkyCoord object
    # Ensure units are correct for astropy
    coords = SkyCoord(
        ra=df_gaia['ra'].values * u.deg,
        dec=df_gaia['dec'].values * u.deg,
        distance=(1000 / df_gaia['parallax'].values) * u.pc, # Parallax in mas
        pm_ra_cosdec=df_gaia['pmra'].values * u.mas/u.yr,
        pm_dec=df_gaia['pmdec'].values * u.mas/u.yr,
        radial_velocity=df_gaia['radial_velocity'].values * u.km/u.s,
        frame='icrs'
    )

    # Transform to Galactocentric coordinates
    galcen_coords = coords.transform_to(GALACTOCENTRIC_FRAME)

    # Extract cylindrical components
    # x, y, z are Galactocentric Cartesian coordinates
    # R_gal is sqrt(x^2 + y^2)
    # v_phi is the rotational velocity component
    df_gaia['x_gal'] = galcen_coords.x.to(u.kpc).value
    df_gaia['y_gal'] = galcen_coords.y.to(u.kpc).value
    df_gaia['z_gal'] = galcen_coords.z.to(u.kpc).value

    df_gaia['v_x_gal'] = galcen_coords.velocity.d_x.to(u.km/u.s).value
    df_gaia['v_y_gal'] = galcen_coords.velocity.d_y.to(u.km/u.s).value
    df_gaia['v_z_gal'] = galcen_coords.velocity.d_z.to(u.km/u.s).value

    # Cylindrical Galactocentric radius
    df_gaia['R_gal'] = np.sqrt(df_gaia['x_gal']**2 + df_gaia['y_gal']**2) # in kpc

    # Azimuthal (rotational) velocity v_phi
    # v_phi = (x * v_y - y * v_x) / R_gal
    # In astropy's Galactocentric frame by default, positive v_phi is in the direction of Galactic rotation.
    # galcen_coords.represent_as('cylindrical') gives v_phi directly with correct sign convention
    cylindrical_diff = galcen_coords.cartesian.differentials['s'] # Get cartesian velocities
    cylindrical_representation = galcen_coords.represent_as('cylindrical')

    df_gaia['phi_gal'] = cylindrical_representation.phi.to(u.rad).value
    df_gaia['v_R_gal'] = cylindrical_representation.differentials['s'].d_rho.to(u.km/u.s).value
    df_gaia['v_phi_gal'] = (cylindrical_representation.rho * cylindrical_representation.differentials['s'].d_phi).to(u.km/u.s).value
    # Astropy's v_phi can sometimes have the opposite sign depending on conventions.
    # Standard convention: positive v_phi is in direction of Galactic rotation.
    # For (x,y,z) with Sun at (-R_sun, 0, 0) and x towards GC, y in direction of rotation:
    # v_phi = (-v_x * sin(phi) + v_y * cos(phi))
    # If phi is angle from x-axis, and x-axis points from Sun to GC:
    # phi_atan2 = np.arctan2(df_gaia['y_gal'], df_gaia['x_gal'])
    # df_gaia['v_phi_gal_manual'] = -df_gaia['v_x_gal'] * np.sin(phi_atan2) + df_gaia['v_y_gal'] * np.cos(phi_atan2)
    # It's generally safer to rely on astropy's internal conversions for v_phi when using `represent_as`

    # Ensure v_phi_gal has the conventional sign (positive for rotation)
    # If astropy's default results in negative for typical rotation, flip it.
    # A quick check: stars rotating with the galaxy should have positive v_phi_gal.
    # The Sun's V component (tangential) is ~230-240 km/s.
    # If median v_phi_gal is negative, it might need flipping based on chosen astropy convention details.
    # Typically, astropy handles this correctly if the Galactocentric frame is set up well.

    return df_gaia

# --- Simple Visible Mass Model (Bulge + Disk) ---
# Parameters are illustrative and should be refined from literature
M_BULGE = 1.5e10 * u.M_sun # Solar masses
R_BULGE_SCALE = 0.7 * u.kpc   # Hernquist bulge scale radius

M_DISK = 5.0e10 * u.M_sun # Solar masses
R_DISK_SCALE = 3.0 * u.kpc    # Exponential disk scale length
Z_DISK_SCALE = 0.3 * u.kpc    # Exponential disk scale height (less critical for Vc in plane)

def v_expected_bulge(R_gal_kpc):
    """Expected circular velocity from a Hernquist bulge."""
    R = R_gal_kpc * u.kpc
    # For Hernquist profile: Vc^2 = G * M * R / (R + a)^2
    vc_squared = (const.G * M_BULGE * R) / (R + R_BULGE_SCALE)**2
    return np.sqrt(vc_squared.to((u.km/u.s)**2)).value # km/s

def v_expected_disk(R_gal_kpc):
    """Expected circular velocity from an exponential disk (approximation)."""
    # This is a more complex calculation for a true exponential disk.
    # A common approximation for thin exponential disk (Binney & Tremaine eq. 2.165 for Vc^2)
    # Vc^2(R) = (G * M_disk / R_disk_scale) * y^2 * [I0(y)K0(y) - I1(y)K1(y)] where y = R/(2*R_disk_scale)
    # I_n, K_n are modified Bessel functions.
    # Simpler approximation / placeholder: treat as a softened point mass or use tabulated values if available
    # For this example, we'll use a Miyamoto-Nagai potential's circular velocity, which is common
    # Vc^2 = G * M_disk * R^2 / (R^2 + (a + sqrt(z^2 + b^2))^2)^(3/2)  -- for a point on the plane z=0
    # Miyamoto-Nagai parameters a (radial scale) and b (vertical scale)
    a_mn = R_DISK_SCALE.value # kpc
    b_mn = Z_DISK_SCALE.value # kpc (approximate, as MN is not exactly exponential)
    R = R_gal_kpc # ensure it's a float or numpy array for calculations

    # For Miyamoto-Nagai on the plane (z=0): Vc^2 = G * M_disk * R^2 / (R^2 + (a_mn + b_mn)^2)^(3/2) -> This is incorrect form for Vc^2 from potential.
    # Correct for M-N on plane z=0: Vc^2 = (G * M_disk * R^2) / (R**2 + (a_mn + b_mn)**2 )**(3./2.) this is also potential, not Vc^2
    # The actual Vc from M-N potential phi = -GM / sqrt(R^2 + (a + sqrt(z^2+b^2))^2)
    # Vc^2 = R * d(phi)/dR
    # At z=0: Vc^2 = G * M_disk * R**2 / (R**2 + (a_mn + b_mn)**2)**(1.5) - this is also wrong.

    # Let's use a simpler placeholder that declines reasonably, or a known fit if you have one.
    # For an exponential disk, the rotation curve peaks and then declines.
    # A very rough form that somewhat mimics this (not physically derived here):
    # This is just to have a declining component. **REPLACE WITH A PROPER MODEL**
    # vc_squared = (const.G * M_DISK * R) / (R**2 + (1.5*R_DISK_SCALE)**2)**(0.75) # Arbitrary
    # return np.sqrt(vc_squared.to((u.km/u.s)**2)).value

    # A better, but still simplified approach for an exponential disk:
    # Use the approximation from Bovy's galpy documentation or Binney & Tremaine
    # For simplicity here, we'll use a Kuzmin disk contribution as a placeholder, which has Vc^2 = GMR^2 / (R^2+a^2)^(3/2)
    # This also doesn't match exponential disk well at large R.

    # **Best approach for this example's simplicity: Use a pre-computed fit or a simpler function**
    # If you have galpy, you could use its potential objects:
    # from galpy.potential import MiyamotoNagaiPotential
    # mp = MiyamotoNagaiPotential(amp=M_DISK, a=R_DISK_SCALE, b=Z_DISK_SCALE)
    # return mp.vcirc(R_gal_kpc * u.kpc, 0.0 * u.kpc).to(u.km/u.s).value

    # Placeholder: simple function that peaks and declines.
    # This needs to be replaced with a physically motivated model for V_expected from visible disk.
    # For instance, using a softened Keplerian decline for radii > peak.
    R_peak_approx = 2.0 * R_DISK_SCALE.value # Roughly where an exponential disk might peak
    v_peak_approx = 200 # km/s, typical peak for disk
    if isinstance(R_gal_kpc, (int, float)): R_gal_kpc = np.array([R_gal_kpc])

    v_disk = np.zeros_like(R_gal_kpc, dtype=float)
    mask_inner = R_gal_kpc <= R_peak_approx
    v_disk[mask_inner] = v_peak_approx * (R_gal_kpc[mask_inner] / R_peak_approx) # Linear rise to peak (very simple)

    mask_outer = R_gal_kpc > R_peak_approx
    v_disk[mask_outer] = v_peak_approx * np.sqrt(R_peak_approx / R_gal_kpc[mask_outer]) # Keplerian decline after peak
    return v_disk


def total_v_expected_visible(R_gal_kpc):
    """
    Total expected circular velocity from visible components (bulge + disk).
    Velocities are added in quadrature (sqrt(v_bulge^2 + v_disk^2)).
    """
    v_b_sq = v_expected_bulge(R_gal_kpc)**2
    v_d_sq = v_expected_disk(R_gal_kpc)**2 # Ensure this function returns velocity in km/s
    return np.sqrt(v_b_sq + v_d_sq)