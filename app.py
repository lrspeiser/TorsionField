import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import time
import warnings

# --- Global Helper for Universal Rotation Redshift Module ---
# This will store the module once imported successfully to avoid re-import attempts
# that might fail if the path changes or there are issues during subsequent Streamlit reruns.
_URS_MODULE = None
_URS_MODULE_ERROR = None

def get_urs_module():
    """Safely imports and returns the universal_rotation_redshift module components."""
    global _URS_MODULE, _URS_MODULE_ERROR
    if _URS_MODULE:
        return _URS_MODULE['class'], _URS_MODULE['apply_func'], _URS_MODULE['generate_data_func']
    if _URS_MODULE_ERROR: # Don't retry if a persistent error occurred
        return None, None, None

    try:
        from universal_rotation_redshift import (
            UniversalRotationRedshift,
            apply_rotation_to_stellar_data,
            generate_mock_gaia_data # Assuming generate_mock_gaia_data is in this module too
        )
        _URS_MODULE = {
            'class': UniversalRotationRedshift,
            'apply_func': apply_rotation_to_stellar_data,
            'generate_data_func': generate_mock_gaia_data
        }
        print("[DEBUG] Successfully imported universal_rotation_redshift module globally.")
        return _URS_MODULE['class'], _URS_MODULE['apply_func'], _URS_MODULE['generate_data_func']
    except ImportError as e:
        _URS_MODULE_ERROR = str(e)
        print(f"[ERROR] CRITICAL: Failed to import universal_rotation_redshift: {_URS_MODULE_ERROR}")
        return None, None, None
    except Exception as e_gen: # Catch other potential errors during import
        _URS_MODULE_ERROR = str(e_gen)
        print(f"[ERROR] CRITICAL: Unexpected error importing universal_rotation_redshift: {_URS_MODULE_ERROR}")
        return None, None, None


try:
    from scipy import integrate, optimize
    from scipy.interpolate import interp1d
    print("[DEBUG] Successfully imported scipy optimization modules")
except ImportError as e:
    print(f"[WARNING] Failed to import scipy modules: {str(e)}")
    st.warning("Some scipy modules are missing. Universal frame dragging analysis may be limited.")

# CRITICAL FIX: Suppress warnings and configure matplotlib
warnings.filterwarnings('ignore', 'Tight layout not applied.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib') # Be more specific
plt.rcParams['figure.constrained_layout.use'] = True

print("[DEBUG] Starting TorsionField application")
print("[DEBUG] Importing custom modules...")

# Import custom modules with error handling
# It's good practice to also wrap these in functions if they are complex or might fail,
# similar to get_urs_module, but for now, keeping existing structure.
try:
    from data_loader import load_local_gaia_files, preprocess_stellar_data
    print("[DEBUG] Successfully imported data_loader")
except ImportError as e:
    print(f"[ERROR] Failed to import data_loader: {str(e)}")
    # st.error(f"Failed to import data_loader: {str(e)}") # Avoid st.error at import time if possible
    DATA_LOADER_AVAILABLE = False
else:
    DATA_LOADER_AVAILABLE = True


try:
    from galactic_dynamics import get_galactocentric_kinematics, total_v_expected_visible
    GALACTIC_DYNAMICS_AVAILABLE = True
    print("[DEBUG] Successfully imported galactic_dynamics module")
except ImportError as e:
    GALACTIC_DYNAMICS_AVAILABLE = False
    print(f"[ERROR] Failed to import galactic_dynamics: {str(e)}")
    # st.error(f"Galactic dynamics module not found: {str(e)}") # Avoid error at import time if app can run without


try:
    from frame_dragging import (
        calculate_frame_dragging_signatures,
        calculate_relativistic_parameters
    )
    print("[DEBUG] Successfully imported frame_dragging")
except ImportError as e:
    print(f"[ERROR] Failed to import frame_dragging: {str(e)}")
    # st.error(f"Failed to import frame_dragging: {str(e)}")
    FRAME_DRAGGING_AVAILABLE = False
else:
    FRAME_DRAGGING_AVAILABLE = True

try:
    from statistical_analysis import (
        run_monte_carlo_simulation,
        calculate_confidence_intervals,
        compute_p_values,
        perform_hypothesis_testing
    )
    print("[DEBUG] Successfully imported statistical_analysis")
except ImportError as e:
    print(f"[ERROR] Failed to import statistical_analysis: {str(e)}")
    # st.error(f"Failed to import statistical_analysis: {str(e)}")
    STATISTICAL_ANALYSIS_AVAILABLE = False
else:
    STATISTICAL_ANALYSIS_AVAILABLE = True


try:
    from visualization import (
        plot_stellar_proper_motions,
        plot_frame_dragging_vectors,
        plot_statistical_results,
        create_3d_visualization,
        create_publication_figure
    )
    print("[DEBUG] Successfully imported visualization")
except ImportError as e:
    print(f"[ERROR] Failed to import visualization: {str(e)}")
    # st.error(f"Failed to import visualization: {str(e)}")
    VISUALIZATION_AVAILABLE = False
else:
    VISUALIZATION_AVAILABLE = True

try:
    from utils import save_results, get_sample_file_path
    print("[DEBUG] Successfully imported utils")
except ImportError as e:
    print(f"[ERROR] Failed to import utils: {str(e)}")
    # st.error(f"Failed to import utils: {str(e)}")
    UTILS_AVAILABLE = False
else:
    UTILS_AVAILABLE = True

try:
    from gaia_fetcher import fetch_gaia_data, cache_gaia_data
    print("[DEBUG] Successfully imported gaia_fetcher")
except ImportError as e:
    print(f"[ERROR] Failed to import gaia_fetcher: {str(e)}")
    # st.error(f"Failed to import gaia_fetcher: {str(e)}")
    GAIA_FETCHER_AVAILABLE = False
else:
    GAIA_FETCHER_AVAILABLE = True


try:
    from database import (
        init_db,
        save_analysis_results,
        get_all_analyses,
        analyses_to_dataframe,
        get_analysis_by_id
    )
    print("[DEBUG] Successfully imported database")
except ImportError as e:
    print(f"[ERROR] Failed to import database: {str(e)}")
    # st.error(f"Failed to import database: {str(e)}")
    DATABASE_AVAILABLE = False
else:
    DATABASE_AVAILABLE = True


print("[DEBUG] All imports completed (or attempted)")

# Set page configuration (should be the first Streamlit command)
if 'page_config_set' not in st.session_state:
    try:
        st.set_page_config(
            page_title="Frame Dragging Detection in Galactic Center",
            page_icon="🌌",
            layout="wide"
        )
        print("[DEBUG] Page configuration set successfully")
        st.session_state.page_config_set = True
    except st.errors.StreamlitAPIException as e_config:
        if "st.set_page_config() can only be called once per app" in str(e_config):
            print("[DEBUG] Page config already set.")
            st.session_state.page_config_set = True
        else:
            print(f"[ERROR] Failed to set page configuration: {str(e_config)}")
    except Exception as e:
        print(f"[ERROR] Unexpected error setting page configuration: {str(e)}")


# Define the sidebar
print("[DEBUG] Setting up sidebar")
st.sidebar.title("Frame Dragging Analysis")
try:
    st.sidebar.image("https://pixabay.com/get/g938cb325df28fac9908bd3f403b5d188532a8be60777d2b0d8bccddcc6cee83b5b2187bf8a4cd1258731e1950b35cb60b9ba0f674d2172955985e96987708a87_1280.jpg", use_container_width=True)
    print("[DEBUG] Sidebar image loaded successfully")
except Exception as e:
    print(f"[WARNING] Failed to load sidebar image: {str(e)}")

# Main page content
print("[DEBUG] Setting up main page content")
st.title("Galactic Center Frame Dragging Analysis")
st.write("""
This application analyzes stellar motion around the galactic center to detect signatures of frame dragging -
a relativistic effect predicted by Einstein's General Theory of Relativity where a massive rotating object
drags the fabric of spacetime around it.
""")

# Display explanatory content
col1, col2 = st.columns([2, 1])
with col1:
    st.write("""
    ## What is Frame Dragging?

    Frame dragging is a relativistic effect where a massive rotating object (like Sagittarius A*, our galaxy's central black hole)
    drags the spacetime fabric around it, causing nearby objects to be pulled along in the direction of rotation.

    This application analyzes stellar motion data from the Gaia catalog to detect these subtle signatures by:

    1. Filtering high-quality stellar data
    2. Calculating expected frame dragging effects
    3. Performing statistical analysis
    4. Visualizing the results
    """)

with col2:
    st.write("""
    ### Key Concepts

    **Einstein's General Relativity** predicts that massive rotating objects like black holes drag spacetime itself around them.

    **Detection Method:** We analyze the tiny systematic motions of stars around Sagittarius A* to detect this effect.

    **Challenge:** The effect is extremely small (microarcseconds per year) requiring precise measurements from the Gaia space telescope.
    """)
# Data loading section
print("[DEBUG] Setting up data loading section")
st.header("1. Data Loading and Preprocessing")
st.write("Choose a data source or upload your own Gaia files")

# Add tabs for different data sources
data_source_options = ["Upload Files", "Sample Data"]
if GAIA_FETCHER_AVAILABLE:
    data_source_options.append("Fetch from Gaia Archive")

data_source = st.radio(
    "Select data source:",
    data_source_options,
    horizontal=True,
    key="data_source_radio"
)
print(f"[DEBUG] Data source selected: {data_source}")

uploaded_files = None # Initialize
if data_source == "Upload Files":
    uploaded_files = st.file_uploader("Upload Gaia data files (.csv.gz)",
                                      type=["csv.gz"],
                                      accept_multiple_files=True)
    print(f"[DEBUG] Upload files mode, {len(uploaded_files) if uploaded_files else 0} files uploaded")

elif data_source == "Sample Data":
    st.info("Using synthetic sample data (includes a frame dragging signal)")
    print("[DEBUG] Sample data mode selected")

elif data_source == "Fetch from Gaia Archive" and GAIA_FETCHER_AVAILABLE:
    st.warning("This will download data directly from the ESA Gaia archive. It may take several minutes.")
    st.info("The data will be cached for future use to avoid repeated downloads.")
    print("[DEBUG] Gaia archive fetch mode selected")

sample_size = st.slider("Sample size", min_value=1000, max_value=100000, value=10000, step=1000, key="sample_size_slider")
print(f"[DEBUG] Sample size set to: {sample_size}")

# Data filtering parameters
st.subheader("Data Quality Filters")
col1_filter, col2_filter = st.columns(2)

with col1_filter:
    pmra_error_max = st.slider("Max proper motion RA error (mas/yr)", 0.1, 2.0, 1.0, 0.1, key="pmra_error_slider")
    pmdec_error_max = st.slider("Max proper motion Dec error (mas/yr)", 0.1, 2.0, 1.0, 0.1, key="pmdec_error_slider")
    parallax_min = st.slider("Min parallax (mas)", 0.05, 1.0, 0.1, 0.05, key="parallax_min_slider")
    print(f"[DEBUG] Filter params set - pmra_error_max: {pmra_error_max}, pmdec_error_max: {pmdec_error_max}, parallax_min: {parallax_min}")

with col2_filter:
    b_min = st.slider("Min absolute galactic latitude |b| (deg)", 0, 30, 10, 1, key="b_min_slider")
    g_mag_max = st.slider("Max G-band magnitude", 10.0, 20.0, 16.0, 0.5, key="g_mag_max_slider")
    print(f"[DEBUG] Filter params set - b_min: {b_min}, g_mag_max: {g_mag_max}")

filter_params = {
    "pmra_error_max": pmra_error_max,
    "pmdec_error_max": pmdec_error_max,
    "parallax_min": parallax_min,
    "b_min": b_min,
    "g_mag_max": g_mag_max
}
print(f"[DEBUG] Combined filter params: {filter_params}")

# Load data button
if st.button("Load and Preprocess Data", key="load_data_button"):
    print("[DEBUG] Load and Preprocess Data button clicked")
    # Reset relevant session state variables
    for key in ['file_paths', 'raw_data', 'processed_data', 'analysis_results', 'relativistic_parameters', 
                'mc_results', 'confidence_intervals', 'p_values', 'hypothesis_results', 
                'fitted_urs_params', 'universal_analysis']:
        if key in st.session_state:
            del st.session_state[key]

    with st.spinner("Loading and preprocessing data..."):
        try:
            st.session_state.using_sample = False
            st.session_state.using_online = False

            if data_source == "Upload Files" and uploaded_files:
                print(f"[DEBUG] Processing {len(uploaded_files)} uploaded files")
                temp_file_paths = []
                for uploaded_file_item in uploaded_files: # Renamed to avoid conflict
                    temp_path = f"/tmp/{uploaded_file_item.name}" # Use a more robust temp dir if needed
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file_item.getbuffer())
                    temp_file_paths.append(temp_path)
                    print(f"[DEBUG] Saved uploaded file to: {temp_path}")
                st.session_state.file_paths = temp_file_paths

            elif data_source == "Sample Data" and UTILS_AVAILABLE:
                print("[DEBUG] Using synthetic sample data")
                sample_path = get_sample_file_path()
                if sample_path and Path(sample_path).exists():
                    st.session_state.file_paths = [str(sample_path)] # Ensure it's a string
                    st.session_state.using_sample = True
                    print(f"[DEBUG] Sample data path: {sample_path}")
                else:
                    st.error("Sample data file not found. Please check configuration.")
                    print(f"[ERROR] Sample data file not found at {sample_path}")
                    st.stop()


            elif data_source == "Fetch from Gaia Archive" and GAIA_FETCHER_AVAILABLE:
                print("[DEBUG] Fetching data from Gaia Archive")
                st.info("Fetching data from Gaia Archive (this may take a while)")
                cached_files = cache_gaia_data() # This function needs to handle query params, etc.

                if cached_files:
                    st.session_state.file_paths = [str(cf) for cf in cached_files] # Ensure strings
                    st.session_state.using_online = True
                    print(f"[DEBUG] Using {len(cached_files)} cached/fetched files: {st.session_state.file_paths}")
                else:
                    st.error("Failed to fetch data from Gaia archive.")
                    if UTILS_AVAILABLE:
                        st.warning("Falling back to sample data.")
                        sample_path = get_sample_file_path()
                        if sample_path and Path(sample_path).exists():
                             st.session_state.file_paths = [str(sample_path)]
                             st.session_state.using_sample = True
                             print(f"[WARNING] Fallback to sample data: {sample_path}")
                        else:
                            print("[ERROR] Fallback sample data also not found.")
                            st.stop()
                    else:
                        print("[ERROR] No fallback available as utils is not loaded.")
                        st.stop()

            if not hasattr(st.session_state, 'file_paths') or not st.session_state.file_paths:
                st.error("No data files specified to load.")
                print("[ERROR] No file paths available for loading data.")
                st.stop()

            print(f"[DEBUG] File paths to load: {st.session_state.file_paths}")

            if not DATA_LOADER_AVAILABLE:
                st.error("Data loader module is not available. Cannot proceed.")
                st.stop()

            df = load_local_gaia_files(st.session_state.file_paths, sample_size=sample_size, filter_params=filter_params)
            if df is None or df.empty:
                st.error("Failed to load data or data is empty after filtering. Please check files and filters.")
                print("[ERROR] load_local_gaia_files returned empty or None DataFrame.")
                st.stop()

            st.session_state.raw_data = df
            print(f"[DEBUG] Loaded {len(df)} raw data rows")

            df_processed = preprocess_stellar_data(df)
            if df_processed is None or df_processed.empty:
                st.error("Preprocessing resulted in an empty dataset. Check preprocessing steps or input data.")
                print("[ERROR] preprocess_stellar_data returned empty or None DataFrame.")
                st.stop()
            st.session_state.processed_data = df_processed
            print(f"[DEBUG] Processed data has {len(df_processed)} rows")

            st.success(f"Successfully loaded and preprocessed {len(df_processed):,} stars")

            st.subheader("Preview of processed data")
            st.dataframe(df_processed.head(10))

            st.subheader("Data Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Stars", f"{len(df_processed):,}")
            if 'parallax' in df_processed.columns and not df_processed['parallax'].empty:
                with col_stat2:
                    # Avoid division by zero or issues with non-positive parallaxes if any slip through
                    valid_parallax = df_processed[df_processed['parallax'] > 0]['parallax']
                    if not valid_parallax.empty:
                        avg_distance = 1000 / valid_parallax.mean()
                        st.metric("Avg. Distance (pc)", f"{avg_distance:.1f}")
                        print(f"[DEBUG] Average distance: {avg_distance:.1f} pc")
                    else:
                        st.metric("Avg. Distance (pc)", "N/A (no valid parallaxes)")
                        print("[DEBUG] No valid parallaxes for average distance calculation.")
            else:
                 with col_stat2:
                    st.metric("Avg. Distance (pc)", "N/A (no parallax data)")

            if all(col in df_processed.columns for col in ['pmra', 'pmdec']):
                with col_stat3:
                    avg_pm = np.sqrt(df_processed['pmra']**2 + df_processed['pmdec']**2).mean()
                    st.metric("Avg. Proper Motion (mas/yr)", f"{avg_pm:.2f}")
                    print(f"[DEBUG] Average proper motion: {avg_pm:.2f} mas/yr")
            else:
                with col_stat3:
                    st.metric("Avg. Proper Motion (mas/yr)", "N/A (no PM data)")


            print("[DEBUG] Data loading and preprocessing completed successfully")

        except FileNotFoundError as fnf_error:
            print(f"[ERROR] File not found during data loading: {str(fnf_error)}")
            st.error(f"File not found: {str(fnf_error)}. Please ensure the file exists or check the path.")
        except pd.errors.EmptyDataError as ede_error:
            print(f"[ERROR] Empty data file encountered: {str(ede_error)}")
            st.error(f"One of the data files is empty: {str(ede_error)}")
        except Exception as e_load:
            print(f"[ERROR] Error loading data: {str(e_load)}")
            st.error(f"An unexpected error occurred during data loading: {str(e_load)}")


# Frame Dragging Analysis section (local Sgr A* effect)
if 'processed_data' in st.session_state and FRAME_DRAGGING_AVAILABLE:
    print("[DEBUG] Setting up Frame Dragging Analysis section (Local Sgr A*)")
    st.header("2. Local Frame Dragging Analysis (Sgr A*)")

    col_fd1, col_fd2 = st.columns(2)
    with col_fd1:
        st.write("Configure Sgr A* parameters:")
        sgr_a_mass = st.number_input("Sgr A* mass (million solar masses)",
                                     min_value=3.0, max_value=5.0, value=4.152, step=0.001, key="sgr_a_mass_input")
        sgr_a_distance = st.number_input("Distance to Galactic Center (kpc)",
                                         min_value=7.0, max_value=9.0, value=8.122, step=0.001, key="sgr_a_dist_input")
        sgr_a_spin = st.slider("Sgr A* dimensionless spin parameter", 0.0, 0.99, 0.9, 0.01, key="sgr_a_spin_slider")
        print(f"[DEBUG] Local FD params - mass: {sgr_a_mass}M☉, distance: {sgr_a_distance} kpc, spin: {sgr_a_spin}")

    with col_fd2:
        try:
            st.image("https://pixabay.com/get/g8d6d9ce5602b0b5cc8229dc15f5da7eb502ae610faec062b2a447853ba4d252edcec32bb808e6c81af5c850dcc2a32abfed4dbf6eddcd18e08755d25e1657139_1280.jpg",
                     caption="Galactic center visualization", use_container_width=True)
            print("[DEBUG] Galactic center image loaded successfully for Section 2")
        except Exception as e_img_s2:
            print(f"[WARNING] Failed to load galactic center image for Section 2: {str(e_img_s2)}")

    if st.button("Run Local Sgr A* Frame Dragging Analysis", key="run_local_fd_button"):
        print("[DEBUG] Run Local Sgr A* Frame Dragging Analysis button clicked")
        with st.spinner("Calculating local frame dragging signatures..."):
            try:
                if st.session_state.processed_data.empty:
                    st.warning("Processed data is empty. Cannot run analysis.")
                    st.stop()
                print(f"[DEBUG] Starting local FD calculation with {len(st.session_state.processed_data)} stars")

                df_results_local_fd = calculate_frame_dragging_signatures(
                    st.session_state.processed_data,
                    sgr_a_mass=sgr_a_mass * 1e6,
                    sgr_a_distance=sgr_a_distance,
                    sgr_a_spin=sgr_a_spin
                )
                st.session_state.analysis_results = df_results_local_fd # This will be used by subsequent sections
                print(f"[DEBUG] Local FD signatures calculated for {len(df_results_local_fd)} stars")

                rel_params_local_fd = calculate_relativistic_parameters(df_results_local_fd,
                                                              sgr_a_mass=sgr_a_mass * 1e6,
                                                              sgr_a_distance=sgr_a_distance)
                st.session_state.relativistic_parameters = rel_params_local_fd
                print(f"[DEBUG] Local FD relativistic parameters calculated")
                st.success("Local Sgr A* frame dragging analysis completed successfully")

                st.subheader("Local Frame Dragging Signature Results")
                col_res1, col_res2, col_res3 = st.columns(3)
                avg_fd_effect = df_results_local_fd['fd_effect_mag'].mean() if 'fd_effect_mag' in df_results_local_fd else np.nan
                max_fd_effect = df_results_local_fd['fd_effect_mag'].max() if 'fd_effect_mag' in df_results_local_fd else np.nan
                snr = rel_params_local_fd.get('signal_to_noise', np.nan)

                with col_res1: st.metric("Avg. FD Effect (μas/yr)", f"{avg_fd_effect:.3f}")
                with col_res2: st.metric("Max FD Effect (μas/yr)", f"{max_fd_effect:.3f}")
                with col_res3: st.metric("Signal-to-Noise Ratio", f"{snr:.2f}")
                print(f"[DEBUG] Local FD Results - Avg FD: {avg_fd_effect:.3f}, Max FD: {max_fd_effect:.3f}, SNR: {snr:.2f}")

                if 'rotation_statistics' in rel_params_local_fd:
                    # ... (your existing code for displaying rotation_statistics) ...
                    pass # Placeholder - keep your existing code here

                st.subheader("Detailed Local FD Results (sample)")
                st.dataframe(df_results_local_fd.head(5))

            except KeyError as ke:
                print(f"[ERROR] KeyError in local Sgr A* frame dragging analysis: {str(ke)}. Potentially missing column.")
                st.error(f"A required data column is missing: {str(ke)}. Please check data preprocessing.")
            except Exception as e_local_fd:
                print(f"[ERROR] Error in local Sgr A* frame dragging analysis: {str(e_local_fd)}")
                st.error(f"Error in local Sgr A* frame dragging analysis: {str(e_local_fd)}")
elif 'processed_data' in st.session_state and not FRAME_DRAGGING_AVAILABLE:
    st.header("2. Local Frame Dragging Analysis (Sgr A*)")
    st.warning("Local Frame Dragging module (`frame_dragging.py`) not available. This section is disabled.")


# Database initialization (Moved earlier, but can be re-checked or used here if needed)
db_session = st.session_state.get('db_session', None)
if not db_session and DATABASE_AVAILABLE:
    print("[DEBUG] Attempting to re-initialize database connection if needed")
    try:
        db_session = init_db()
        if db_session:
            st.session_state.db_session = db_session
            st.sidebar.success("Database connected successfully (re-check).")
            print("[DEBUG] Database re-connected successfully.")
        else:
            st.sidebar.warning("Database connection failed (re-check).")
            print("[WARNING] Database re-connection failed.")
    except Exception as e_db_reinit:
        print(f"[ERROR] Database re-initialization failed: {str(e_db_reinit)}")
        st.sidebar.error(f"Database error (re-check): {str(e_db_reinit)}")
        db_session = None


# Statistical validation section
if 'analysis_results' in st.session_state and STATISTICAL_ANALYSIS_AVAILABLE:
    print("[DEBUG] Setting up Statistical Validation section")
    st.header("3. Statistical Validation")
    # ... (Your existing Statistical Validation code - Section 3) ...
    # Ensure it uses st.session_state.analysis_results (which is set by Section 2)
    col1_statval, col2_statval = st.columns(2)
    with col1_statval:
        st.write("Configure statistical validation parameters:")
        confidence_level = st.slider("Confidence level (%)", 90, 99, 95, 1, key="conf_level_slider")
        num_simulations = st.slider("Number of Monte Carlo simulations", 100, 10000, 1000, 100, key="mc_sim_slider")
        null_hypothesis = st.selectbox("Null hypothesis model",
                                      ["Random stellar motions",
                                       "Non-relativistic gravitational effects",
                                       "Measurement errors only"], key="null_hypo_select")
    # ... and the rest of section 3 ...
    if st.button("Run Statistical Validation", key="run_stat_val_button"):
        # ...
        pass

elif 'analysis_results' in st.session_state and not STATISTICAL_ANALYSIS_AVAILABLE:
    st.header("3. Statistical Validation")
    st.warning("Statistical Analysis module (`statistical_analysis.py`) not available. This section is disabled.")


# Visualization section
if 'analysis_results' in st.session_state and VISUALIZATION_AVAILABLE:
    print("[DEBUG] Setting up Interactive Visualization section")
    st.header("4. Interactive Visualization")
    # ... (Your existing Visualization code - Section 4) ...
    # Ensure it uses st.session_state.analysis_results
    visualization_type = st.selectbox(
        "Select visualization type",
        ["Proper Motion Vectors", "Frame Dragging Effect", "3D Visualization", "Combined Analysis"],
        key="viz_type_select"
    )
    # ... and the rest of section 4 ...
    if st.button("Generate Publication Figure", key="gen_pub_fig_button"):
        # ...
        pass

elif 'analysis_results' in st.session_state and not VISUALIZATION_AVAILABLE:
    st.header("4. Interactive Visualization")
    st.warning("Visualization module (`visualization.py`) not available. This section is disabled.")


# Footer (Moved before dynamic sections to ensure it always shows)
print("[DEBUG] Setting up footer")
st.markdown("---")
st.write("""
### Frame Dragging Detection Platform
This application was developed to analyze stellar motion around the galactic center and detect the relativistic frame dragging effect.
It also includes experimental features for Universal Rotation Redshift hypotheses.
""")
st.sidebar.markdown("---")
st.sidebar.info("""
**About Frame Dragging**
Frame dragging, also known as the Lense-Thirring effect, is a prediction of Einstein's theory of general relativity.
It describes how a massive rotating object drags the fabric of spacetime around it.
""")


# --- Universal Rotation Redshift Analysis section (Section 5) ---
# This section will use the UniversalRotationRedshift model
UniversalRotationRedshift, urs_apply_rotation_func, urs_generate_data_func = get_urs_module()

if 'processed_data' in st.session_state:
    print("[DEBUG] Setting up Universal Rotation Redshift Analysis section (Section 5)")
    st.header("5. Universal Rotation Redshift Analysis")
    st.write("""
    This section tests the hypothesis that cosmological redshift
    could be caused by a global rotation of the universe.
    It uses a separate model (`universal_rotation_redshift.py`).
    """)

    if UniversalRotationRedshift is None:
        st.error(f"Universal Rotation Redshift module (`universal_rotation_redshift.py`) failed to import: {_URS_MODULE_ERROR}. This section is disabled.")
        print(f"[ERROR] Section 5 disabled because URS module failed to import: {_URS_MODULE_ERROR}")
    else:
        print("[DEBUG] UniversalRotationRedshift module successfully loaded for Section 5.")
        col_urs1, col_urs2 = st.columns(2)
        with col_urs1:
            st.write("Configure universal rotation parameters:")
            log10_A_rot_default_s5 = np.log10(1.0)
            log10_omega_U_default_s5 = np.log10(10.0)
            R_observer_default_s5 = 10000.0

            log10_A_rot_input_s5 = st.slider("Log10(A_rot) - Coupling Constant",
                                         -8.0, 8.0, log10_A_rot_default_s5, 0.5, key="s5_log10_a_rot")
            log10_omega_U_input_s5 = st.slider("Log10(Omega_U) - Ang. Vel. (/Gyr)",
                                          -3.0, 4.0, log10_omega_U_default_s5, 0.1, key="s5_log10_omega")
            R_observer_input_Mpc_s5 = st.number_input("Our Distance from Rotation Center (Mpc)",
                                                 min_value=500.0, max_value=20000.0,
                                                 value=R_observer_default_s5, step=500.0, key="s5_r_observer")
            A_rot_val_s5 = 10**log10_A_rot_input_s5
            omega_U_val_inv_Gyr_s5 = 10**log10_omega_U_input_s5
            print(f"[DEBUG] S5 URS Params - A_rot: {A_rot_val_s5:.2e}, omega_U: {omega_U_val_inv_Gyr_s5:.2f}/Gyr, R_obs: {R_observer_input_Mpc_s5} Mpc")

        with col_urs2:
            st.write("Analysis options (URS):")
            analyze_cmb_urs_s5 = st.checkbox("Analyze CMB dipole implications (URS)", value=True, key="s5_urs_analyze_cmb")

        if st.button("Run Universal Rotation Redshift Analysis & Fit Parameters", key="s5_run_urs_analysis"):
            print("[DEBUG] S5: Run URS Analysis button clicked")
            with st.spinner("Analyzing universal rotation hypothesis..."):
                try:
                    urs_s5 = UniversalRotationRedshift()
                    print("[DEBUG] S5: UniversalRotationRedshift instance created.")

                    n_sne_s5 = 100
                    H0_s5 = 70
                    true_distances_obs_Mpc_s5 = np.logspace(np.log10(30), np.log10(3500), n_sne_s5)
                    obs_distances_Mpc_s5 = true_distances_obs_Mpc_s5 * (1 + np.random.normal(0, 0.1, n_sne_s5))
                    v_mock_s5 = H0_s5 * true_distances_obs_Mpc_s5
                    obs_z_mock_s5 = v_mock_s5 / (urs_s5.c / 1000) + np.random.normal(0, 0.02, n_sne_s5)
                    obs_z_mock_s5 = np.maximum(0.001, obs_z_mock_s5)

                    st.info("S5: Fitting universal rotation parameters to mock SNe data...")
                    print("[DEBUG] S5: Calling fit_rotational_parameters...")
                    best_urs_params_s5 = urs_s5.fit_rotational_parameters(
                        obs_z_mock_s5,
                        obs_distances_Mpc_s5,
                        R_observer_guess_Mpc=R_observer_input_Mpc_s5
                    )
                    st.session_state.fitted_urs_params = best_urs_params_s5 # CRITICAL: Save to session state
                    print(f"[DEBUG] S5: URS Parameter fitting completed. Chi2: {best_urs_params_s5['chi2']:.2f}, Success: {best_urs_params_s5['success']}")

                    st.subheader("S5: Fitted Universal Rotation Parameters")
                    col_fit_s5_1, col_fit_s5_2, col_fit_s5_3 = st.columns(3)
                    with col_fit_s5_1: st.metric("Best-fit A_rot", f"{best_urs_params_s5['A_rot']:.2e}")
                    with col_fit_s5_2: st.metric("Best-fit Omega_U (/Gyr)", f"{best_urs_params_s5['omega_U_inv_Gyr']:.3f}")
                    with col_fit_s5_3: st.metric("Best-fit R_observer (Mpc)", f"{best_urs_params_s5['R_observer_Mpc']:.0f}")
                    st.success(f"S5 Fit: χ² = {best_urs_params_s5['chi2']:.2f}. Optimizer Msg: {best_urs_params_s5.get('message', 'N/A')}")

                    st.subheader("S5: Simplified Hubble Diagram (Fitted URS Model)")
                    plot_distances_Mpc_s5 = np.linspace(10, 4000, 50)
                    v_rot_fitted_s5, _ = urs_s5.hubble_law_from_rotation(
                        plot_distances_Mpc_s5,
                        best_urs_params_s5['A_rot'],
                        best_urs_params_s5['omega_U_inv_Gyr'],
                        best_urs_params_s5['R_observer_Mpc']
                    )
                    v_hubble_comp_s5 = H0_s5 * plot_distances_Mpc_s5
                    fig_hubble_urs_s5, ax_hubble_urs_s5 = plt.subplots(figsize=(8,5), layout='constrained')
                    ax_hubble_urs_s5.plot(plot_distances_Mpc_s5, v_hubble_comp_s5, 'b-', label=f'Std. Hubble Law (H0={H0_s5})')
                    ax_hubble_urs_s5.plot(plot_distances_Mpc_s5, v_rot_fitted_s5, 'r.--', label='Fitted URS Model')
                    ax_hubble_urs_s5.set_xlabel('Distance from Observer (Mpc)')
                    ax_hubble_urs_s5.set_ylabel('Apparent Recession Velocity (km/s)')
                    ax_hubble_urs_s5.legend(); ax_hubble_urs_s5.grid(True, alpha=0.3)
                    st.pyplot(fig_hubble_urs_s5)
                    print("[DEBUG] S5: URS Hubble diagram plotted.")

                    if analyze_cmb_urs_s5:
                        st.subheader("S5: CMB Dipole Analysis (URS context)")
                        peculiar_velocity_sun_kms_s5 = 369.8
                        cmb_results_urs_s5 = urs_s5.calculate_cmb_dipole_prediction(peculiar_velocity_kms=peculiar_velocity_sun_kms_s5)
                        st.metric("Predicted Kinematic CMB Dipole (URS)", f"{cmb_results_urs_s5['kinematic_dipole_amplitude_mK']:.3f} mK")
                        st.info("Observed CMB dipole: 3.362 ± 0.001 mK (Planck 2018)")
                        print(f"[DEBUG] S5: URS CMB analysis - dipole: {cmb_results_urs_s5['kinematic_dipole_amplitude_mK']:.3f} mK")
                    st.success("S5: Universal Rotation Redshift analysis and fitting completed.")
                except Exception as e_urs_s5:
                    print(f"[ERROR] S5: Error in URS analysis: {str(e_urs_s5)}")
                    st.error(f"Error in S5 URS analysis: {str(e_urs_s5)}")
        # else:
        #     st.info("Click the button above to run the Universal Rotation Redshift analysis.")

# --- Section 6: Co-moving Group Differential Redshift Analysis ---
if 'processed_data' in st.session_state and 'fitted_urs_params' in st.session_state and UniversalRotationRedshift is not None:
    print("[DEBUG] Setting up Co-moving Group Differential Redshift Analysis section (Section 6)")
    st.header("6. Co-moving Group Differential Redshift Analysis")
    st.write("""
    This section analyzes stars within a (simulated) co-moving group
    to look for subtle redshift patterns after accounting for the group's bulk motion,
    using the parameters fitted for the Universal Rotation Redshift model from Section 5.
    """)

    urs_model_instance_s6 = UniversalRotationRedshift()
    apply_rotation_func_s6 = urs_apply_rotation_func # From get_urs_module()

    fitted_urs_params_s6 = st.session_state.fitted_urs_params
    A_rot_group_s6 = fitted_urs_params_s6['A_rot']
    omega_U_inv_Gyr_group_s6 = fitted_urs_params_s6['omega_U_inv_Gyr']
    R_observer_Mpc_group_s6 = fitted_urs_params_s6['R_observer_Mpc']
    print(f"[DEBUG] S6: Group Analysis using URS params: A_rot={A_rot_group_s6:.2e}, omega_U={omega_U_inv_Gyr_group_s6:.2f}/Gyr, R_obs={R_observer_Mpc_group_s6:.0f} Mpc")

    st.subheader("Define Simulated Co-moving Group for Section 6")
    df_full_processed_s6 = st.session_state.processed_data.copy() # Use a copy
    if 'radial_velocity' not in df_full_processed_s6.columns:
        st.error("S6: Radial velocity data ('radial_velocity') required but not found.")
        st.warning("S6: Generating DUMMY radial velocities. Results will NOT be meaningful.")
        df_full_processed_s6['radial_velocity'] = np.random.normal(0, 50, len(df_full_processed_s6))
        df_full_processed_s6['radial_velocity_error'] = np.random.uniform(1, 5, len(df_full_processed_s6))

    col_g1_s6, col_g2_s6 = st.columns(2)
    with col_g1_s6:
        group_center_l_s6 = st.slider("S6 Group Center L (deg)", 0.0, 360.0, 10.0, 1.0, key="s6_grp_l")
        group_center_b_s6 = st.slider("S6 Group Center B (deg)", -90.0, 90.0, 5.0, 1.0, key="s6_grp_b")
        group_radius_deg_s6 = st.slider("S6 Group Angular Radius (deg)", 0.1, 5.0, 0.5, 0.1, key="s6_grp_rad")
    with col_g2_s6:
        group_dist_pc_min_s6 = st.number_input("S6 Group Min Dist (pc)", value=7000.0, min_value=100.0, key="s6_grp_dmin")
        group_dist_pc_max_s6 = st.number_input("S6 Group Max Dist (pc)", value=9000.0, min_value=group_dist_pc_min_s6 + 100, key="s6_grp_dmax")
        min_stars_in_group_s6 = st.number_input("S6 Min Stars", value=10, min_value=3, key="s6_grp_minstars")

    if st.button("Analyze Selected Co-moving Group (Section 6)", key="s6_analyze_group_btn"):
        print("[DEBUG] S6: Analyze Co-moving Group button clicked")
        with st.spinner("S6: Analyzing group..."):
            # Filter logic (ensure df_full_processed_s6 has l_deg, b_deg, distance_pc)
            l_deg_col = 'l_deg' if 'l_deg' in df_full_processed_s6.columns else 'l'
            b_deg_col = 'b_deg' if 'b_deg' in df_full_processed_s6.columns else 'b'

            l_lower_s6 = (group_center_l_s6 - group_radius_deg_s6 + 360) % 360
            l_upper_s6 = (group_center_l_s6 + group_radius_deg_s6 + 360) % 360
            b_lower_s6 = group_center_b_s6 - group_radius_deg_s6
            b_upper_s6 = group_center_b_s6 + group_radius_deg_s6

            if l_lower_s6 < l_upper_s6:
                l_condition_s6 = (df_full_processed_s6[l_deg_col] >= l_lower_s6) & (df_full_processed_s6[l_deg_col] <= l_upper_s6)
            else:
                l_condition_s6 = (df_full_processed_s6[l_deg_col] >= l_lower_s6) | (df_full_processed_s6[l_deg_col] <= l_upper_s6)
            b_condition_s6 = (df_full_processed_s6[b_deg_col] >= b_lower_s6) & (df_full_processed_s6[b_deg_col] <= b_upper_s6)
            dist_condition_s6 = (df_full_processed_s6['distance_pc'] >= group_dist_pc_min_s6) & \
                             (df_full_processed_s6['distance_pc'] <= group_dist_pc_max_s6)
            df_group_s6 = df_full_processed_s6[l_condition_s6 & b_condition_s6 & dist_condition_s6].copy()
            print(f"[DEBUG] S6: Filtered group: {len(df_group_s6)} stars. Required: {min_stars_in_group_s6}")

            if len(df_group_s6) < min_stars_in_group_s6:
                st.warning(f"S6: Found only {len(df_group_s6)} stars. Need {min_stars_in_group_s6}.")
            else:
                st.success(f"S6: Selected {len(df_group_s6)} stars for co-moving group analysis.")
                st.dataframe(df_group_s6[['source_id', l_deg_col, b_deg_col, 'distance_pc', 'radial_velocity']].head())

                mean_rv_group_s6 = df_group_s6['radial_velocity'].mean()
                std_rv_group_s6 = df_group_s6['radial_velocity'].std()
                st.metric("S6: Mean RV of Group (km/s)", f"{mean_rv_group_s6:.2f} ± {std_rv_group_s6:.2f}")

                df_group_s6['v_residual_kms'] = df_group_s6['radial_velocity'] - mean_rv_group_s6
                df_group_s6['z_doppler_residual'] = df_group_s6['v_residual_kms'] / urs_model_instance_s6.c * 1000

                df_group_with_zrot_s6 = apply_rotation_func_s6(
                    df_group_s6, urs_model_instance_s6, A_rot_group_s6, omega_U_inv_Gyr_group_s6, R_observer_Mpc_group_s6
                )
                print(f"[DEBUG] S6: Calculated z_rot. Sample: {df_group_with_zrot_s6['z_rot'].head().values}")

                st.subheader("S6: Comparison of Residual Doppler z and Predicted Rotational z")
                fig_group_s6 = go.Figure()
                fig_group_s6.add_trace(go.Scatter(x=df_group_with_zrot_s6['dist_to_center_Mpc'], y=df_group_with_zrot_s6['z_doppler_residual'], mode='markers', name='z_Doppler_residual'))
                fig_group_s6.add_trace(go.Scatter(x=df_group_with_zrot_s6['dist_to_center_Mpc'], y=df_group_with_zrot_s6['z_rot'], mode='markers', name='z_rot (Predicted URS)'))
                mean_z_rot_group_s6 = df_group_with_zrot_s6['z_rot'].mean()
                df_group_with_zrot_s6['z_rot_residual'] = df_group_with_zrot_s6['z_rot'] - mean_z_rot_group_s6
                fig_group_s6.update_layout(title='S6: Residuals vs. Dist from URS Center', xaxis_title='Dist from URS Center (Mpc)', yaxis_title='Redshift (z)')
                st.plotly_chart(fig_group_s6, use_container_width=True)

                if len(df_group_with_zrot_s6) > 1:
                    try:
                        correlation_s6 = df_group_with_zrot_s6['z_doppler_residual'].corr(df_group_with_zrot_s6['z_rot_residual'])
                        st.metric("S6: Corr (z_Doppler_res vs z_rot_res)", f"{correlation_s6:.3f}")
                    except Exception: st.text("S6: Correlation N/A.")
                    avg_z_dop_res_s6 = df_group_with_zrot_s6['z_doppler_residual'].abs().mean()
                    avg_z_rot_s6 = df_group_with_zrot_s6['z_rot'].abs().mean()
                    avg_z_rot_res_s6 = df_group_with_zrot_s6['z_rot_residual'].abs().mean()
                    st.markdown(f"S6 Avg: |z_Dop_res|: `{avg_z_dop_res_s6:.2e}`, |z_rot|: `{avg_z_rot_s6:.2e}`, |z_rot_res|: `{avg_z_rot_res_s6:.2e}`")
                    if avg_z_rot_res_s6 < 1e-7 and avg_z_dop_res_s6 > 1e-6:
                         st.info("S6: Predicted URS variation in group << internal velocity dispersion.")
elif 'processed_data' in st.session_state and UniversalRotationRedshift is None:
    st.header("6. Co-moving Group Differential Redshift Analysis")
    st.warning(f"Section 6 disabled because Universal Rotation Redshift module failed to import: {_URS_MODULE_ERROR}")


print("[DEBUG] End of Streamlit application script.")