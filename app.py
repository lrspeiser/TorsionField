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

# CRITICAL FIX: Suppress warnings and configure matplotlib
warnings.filterwarnings('ignore', 'Tight layout not applied.*')
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.constrained_layout.use'] = True

print("[DEBUG] Starting TorsionField application")
print("[DEBUG] Importing custom modules...")

# Import custom modules with error handling
try:
    from data_loader import load_local_gaia_files, preprocess_stellar_data
    print("[DEBUG] Successfully imported data_loader")
except ImportError as e:
    print(f"[ERROR] Failed to import data_loader: {str(e)}")
    st.error(f"Failed to import data_loader: {str(e)}")

try:
    from frame_dragging import (
        calculate_frame_dragging_signatures,
        calculate_relativistic_parameters
    )
    print("[DEBUG] Successfully imported frame_dragging")
except ImportError as e:
    print(f"[ERROR] Failed to import frame_dragging: {str(e)}")
    st.error(f"Failed to import frame_dragging: {str(e)}")

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
    st.error(f"Failed to import statistical_analysis: {str(e)}")

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
    st.error(f"Failed to import visualization: {str(e)}")

try:
    from utils import save_results, get_sample_file_path
    print("[DEBUG] Successfully imported utils")
except ImportError as e:
    print(f"[ERROR] Failed to import utils: {str(e)}")
    st.error(f"Failed to import utils: {str(e)}")

try:
    from gaia_fetcher import fetch_gaia_data, cache_gaia_data
    print("[DEBUG] Successfully imported gaia_fetcher")
except ImportError as e:
    print(f"[ERROR] Failed to import gaia_fetcher: {str(e)}")
    st.error(f"Failed to import gaia_fetcher: {str(e)}")

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
    st.error(f"Failed to import database: {str(e)}")

print("[DEBUG] All imports completed")

# Set page configuration
try:
    st.set_page_config(
        page_title="Frame Dragging Detection in Galactic Center",
        page_icon="🌌",
        layout="wide"
    )
    print("[DEBUG] Page configuration set successfully")
except Exception as e:
    print(f"[ERROR] Failed to set page configuration: {str(e)}")

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
data_source = st.radio(
    "Select data source:",
    ["Upload Files", "Sample Data", "Fetch from Gaia Archive"],
    horizontal=True
)
print(f"[DEBUG] Data source selected: {data_source}")

if data_source == "Upload Files":
    uploaded_files = st.file_uploader("Upload Gaia data files (.csv.gz)", 
                                      type=["csv.gz"], 
                                      accept_multiple_files=True)
    use_sample_data = False
    fetch_from_gaia = False
    print(f"[DEBUG] Upload files mode, {len(uploaded_files) if uploaded_files else 0} files uploaded")

elif data_source == "Sample Data":
    st.info("Using synthetic sample data (includes a frame dragging signal)")
    uploaded_files = None
    use_sample_data = True
    fetch_from_gaia = False
    print("[DEBUG] Sample data mode selected")

else:  # Fetch from Gaia Archive
    st.warning("This will download data directly from the ESA Gaia archive. It may take several minutes.")
    uploaded_files = None
    use_sample_data = False
    fetch_from_gaia = True
    st.info("The data will be cached for future use to avoid repeated downloads.")
    print("[DEBUG] Gaia archive fetch mode selected")

sample_size = st.slider("Sample size", min_value=1000, max_value=100000, value=10000, step=1000)
print(f"[DEBUG] Sample size set to: {sample_size}")

# Data filtering parameters
st.subheader("Data Quality Filters")
col1, col2 = st.columns(2)

with col1:
    pmra_error_max = st.slider("Max proper motion RA error (mas/yr)", 0.1, 2.0, 1.0, 0.1)
    pmdec_error_max = st.slider("Max proper motion Dec error (mas/yr)", 0.1, 2.0, 1.0, 0.1)
    parallax_min = st.slider("Min parallax (mas)", 0.05, 1.0, 0.1, 0.05)
    print(f"[DEBUG] Filter params set - pmra_error_max: {pmra_error_max}, pmdec_error_max: {pmdec_error_max}, parallax_min: {parallax_min}")

with col2:
    b_min = st.slider("Min absolute galactic latitude |b| (deg)", 0, 30, 10, 1)
    g_mag_max = st.slider("Max G-band magnitude", 10.0, 20.0, 16.0, 0.5)
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
if st.button("Load and Preprocess Data"):
    print("[DEBUG] Load and Preprocess Data button clicked")
    with st.spinner("Loading and preprocessing data..."):
        try:
            if data_source == "Upload Files" and uploaded_files:
                print(f"[DEBUG] Processing {len(uploaded_files)} uploaded files")
                # Save uploaded files temporarily
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_file_paths.append(temp_path)
                    print(f"[DEBUG] Saved uploaded file to: {temp_path}")

                st.session_state.file_paths = temp_file_paths
                st.session_state.using_sample = False
                st.session_state.using_online = False

            elif data_source == "Sample Data":
                print("[DEBUG] Using synthetic sample data")
                # Use synthetic sample data
                st.info("Using synthetic sample data with frame dragging signal")
                sample_path = get_sample_file_path()
                st.session_state.file_paths = [sample_path]
                st.session_state.using_sample = True
                st.session_state.using_online = False
                print(f"[DEBUG] Sample data path: {sample_path}")

            elif data_source == "Fetch from Gaia Archive":
                print("[DEBUG] Fetching data from Gaia Archive")
                # Fetch data from Gaia archive
                st.info("Fetching data from Gaia Archive (this may take a while)")

                # Try to use cached data first
                cached_files = cache_gaia_data()

                if cached_files:
                    print(f"[DEBUG] Using {len(cached_files)} cached files")
                    st.session_state.file_paths = cached_files
                    st.session_state.using_sample = False
                    st.session_state.using_online = True
                else:
                    print("[WARNING] Failed to fetch from Gaia, falling back to sample data")
                    st.error("Failed to fetch data from Gaia archive. Using sample data instead.")
                    st.session_state.file_paths = [get_sample_file_path()]
                    st.session_state.using_sample = True
                    st.session_state.using_online = False

            print(f"[DEBUG] File paths to load: {st.session_state.file_paths}")

            # Load the data
            df = load_local_gaia_files(st.session_state.file_paths, sample_size=sample_size, filter_params=filter_params)
            st.session_state.raw_data = df
            print(f"[DEBUG] Loaded {len(df)} raw data rows")

            # Preprocess the data
            df_processed = preprocess_stellar_data(df)
            st.session_state.processed_data = df_processed
            print(f"[DEBUG] Processed data has {len(df_processed)} rows")

            st.success(f"Successfully loaded and preprocessed {len(df_processed):,} stars")

            # Show a preview of the data
            st.subheader("Preview of processed data")
            st.dataframe(df_processed.head(10))

            # Basic statistics
            st.subheader("Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stars", f"{len(df_processed):,}")
            with col2:
                avg_distance = 1000/df_processed['parallax'].mean()
                st.metric("Avg. Distance (pc)", f"{avg_distance:.1f}")
                print(f"[DEBUG] Average distance: {avg_distance:.1f} pc")
            with col3:
                avg_pm = np.sqrt(df_processed['pmra']**2 + df_processed['pmdec']**2).mean()
                st.metric("Avg. Proper Motion (mas/yr)", f"{avg_pm:.2f}")
                print(f"[DEBUG] Average proper motion: {avg_pm:.2f} mas/yr")

            print("[DEBUG] Data loading and preprocessing completed successfully")

        except Exception as e:
            print(f"[ERROR] Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")

# Frame Dragging Analysis section (only show if data is loaded)
if 'processed_data' in st.session_state:
    print("[DEBUG] Setting up Frame Dragging Analysis section")
    st.header("2. Frame Dragging Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Configure frame dragging detection parameters:")
        sgr_a_mass = st.number_input("Sgr A* mass (million solar masses)", 
                                     min_value=3.0, max_value=5.0, value=4.152, step=0.001)
        sgr_a_distance = st.number_input("Distance to Galactic Center (kpc)", 
                                         min_value=7.0, max_value=9.0, value=8.122, step=0.001)
        sgr_a_spin = st.slider("Sgr A* dimensionless spin parameter", 0.0, 0.99, 0.9, 0.01)
        print(f"[DEBUG] Frame dragging params - mass: {sgr_a_mass}M☉, distance: {sgr_a_distance} kpc, spin: {sgr_a_spin}")

    with col2:
        try:
            st.image("https://pixabay.com/get/g8d6d9ce5602b0b5cc8229dc15f5da7eb502ae610faec062b2a447853ba4d252edcec32bb808e6c81af5c850dcc2a32abfed4dbf6eddcd18e08755d25e1657139_1280.jpg", 
                     caption="Galactic center visualization", use_container_width=True)
            print("[DEBUG] Galactic center image loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load galactic center image: {str(e)}")

    # Run frame dragging analysis
    if st.button("Run Frame Dragging Analysis"):
        print("[DEBUG] Run Frame Dragging Analysis button clicked")
        with st.spinner("Calculating frame dragging signatures..."):
            try:
                print(f"[DEBUG] Starting frame dragging calculation with {len(st.session_state.processed_data)} stars")

                # Calculate frame dragging signatures
                df_results = calculate_frame_dragging_signatures(
                    st.session_state.processed_data,
                    sgr_a_mass=sgr_a_mass * 1e6,  # Convert to solar masses
                    sgr_a_distance=sgr_a_distance,
                    sgr_a_spin=sgr_a_spin
                )

                st.session_state.analysis_results = df_results
                print(f"[DEBUG] Frame dragging signatures calculated for {len(df_results)} stars")

                # Calculate relativistic parameters
                rel_params = calculate_relativistic_parameters(df_results, 
                                                              sgr_a_mass=sgr_a_mass * 1e6,
                                                              sgr_a_distance=sgr_a_distance)

                st.session_state.relativistic_parameters = rel_params
                print(f"[DEBUG] Relativistic parameters calculated")

                st.success("Frame dragging analysis completed successfully")

                # Show main results
                st.subheader("Frame Dragging Signature Results")
                col1, col2, col3 = st.columns(3)

                avg_fd_effect = df_results['fd_effect_mag'].mean()
                max_fd_effect = df_results['fd_effect_mag'].max()
                snr = rel_params['signal_to_noise']

                with col1:
                    st.metric("Avg. Frame Dragging Effect (μas/yr)", f"{avg_fd_effect:.3f}")
                with col2:
                    st.metric("Max Frame Dragging Effect (μas/yr)", f"{max_fd_effect:.3f}")
                with col3:
                    st.metric("Signal-to-Noise Ratio", f"{snr:.2f}")

                print(f"[DEBUG] Results - Avg FD: {avg_fd_effect:.3f}, Max FD: {max_fd_effect:.3f}, SNR: {snr:.2f}")

                # Display rotation analysis results if available
                if 'rotation_statistics' in rel_params:
                    print("[DEBUG] Displaying rotation analysis results")
                    rot_stats = rel_params['rotation_statistics']
                    st.subheader("Galactic Rotation Analysis")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Azimuthal Velocity (km/s)", 
                                f"{rot_stats['mean_azimuthal_velocity']:.3f}",
                                delta=f"±{rot_stats['std_error']:.3f}")
                        st.metric("Statistical Significance", 
                                f"{rot_stats['significance']}")
                    with col2:
                        st.metric("p-value", 
                                f"{rot_stats['p_value']:.6f}")
                        st.metric("Effect Size (Cohen's d)", 
                                f"{rot_stats['cohens_d']:.3f}")

                    st.info(f"Interpretation: {rot_stats['interpretation']}")
                    print(f"[DEBUG] Rotation stats - velocity: {rot_stats['mean_azimuthal_velocity']:.3f}, p-value: {rot_stats['p_value']:.6f}")

                    # Plot rotation curve if available
                    if 'r_centers' in rel_params and 'mean_v_azimuthal' in rel_params:
                        print("[DEBUG] Creating rotation curve plot")
                        st.subheader("Rotation Curve")

                        # CRITICAL FIX: Use constrained_layout instead of tight_layout
                        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

                        # Filter out NaN values
                        mask = ~np.isnan(rel_params['mean_v_azimuthal'])
                        r_centers = rel_params['r_centers'][mask]
                        mean_v_az = rel_params['mean_v_azimuthal'][mask]
                        std_v_az = rel_params['std_v_azimuthal'][mask]

                        ax.errorbar(r_centers, mean_v_az, yerr=std_v_az, 
                                  fmt='bo-', capsize=5)
                        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

                        # Add theoretical frame dragging curve (1/r)
                        if len(r_centers) > 0:
                            r_theory = np.linspace(r_centers.min(), r_centers.max(), 100)
                            v_theory = mean_v_az[0] * r_centers[0] / r_theory
                            ax.plot(r_theory, v_theory, 'r--', alpha=0.7, 
                                  label='Expected Frame Dragging (∝ 1/r)')
                            ax.legend()

                        ax.set_xlabel('Galactocentric Radius (pc)')
                        ax.set_ylabel('Mean Azimuthal Velocity (km/s)')
                        ax.set_title('Evidence of Frame Dragging: Systematic Rotation')
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)
                        print("[DEBUG] Rotation curve plot created successfully")

                # Show detailed results table
                st.subheader("Detailed Results (sample)")
                st.dataframe(df_results.head(5))
                print("[DEBUG] Frame dragging analysis completed successfully")

            except Exception as e:
                print(f"[ERROR] Error in frame dragging analysis: {str(e)}")
                st.error(f"Error in frame dragging analysis: {str(e)}")

# Database initialization
print("[DEBUG] Initializing database connection")
try:
    db_session = init_db()
    if db_session:
        st.sidebar.success("Database connected successfully")
        print("[DEBUG] Database connected successfully")
    else:
        st.sidebar.warning("Database connection failed. Analysis results won't be saved.")
        print("[WARNING] Database connection failed")
except Exception as e:
    print(f"[ERROR] Database initialization failed: {str(e)}")
    st.sidebar.error(f"Database error: {str(e)}")
    db_session = None

# Statistical validation section (only show if analysis is completed)
if 'analysis_results' in st.session_state:
    print("[DEBUG] Setting up Statistical Validation section")
    st.header("3. Statistical Validation")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Configure statistical validation parameters:")
        confidence_level = st.slider("Confidence level (%)", 90, 99, 95, 1)
        num_simulations = st.slider("Number of Monte Carlo simulations", 100, 10000, 1000, 100)
        null_hypothesis = st.selectbox("Null hypothesis model", 
                                      ["Random stellar motions", 
                                       "Non-relativistic gravitational effects", 
                                       "Measurement errors only"])
        print(f"[DEBUG] Statistical params - confidence: {confidence_level}%, simulations: {num_simulations}, null model: {null_hypothesis}")

    with col2:
        try:
            st.image("https://pixabay.com/get/g8f1501f5e51954f4f87faddf9a3cdd718acbe85b3ee7fc7e71f69d89b72e4babc9274805c5e90979bda8b2eedf36524ca63103c0ff9808060e3b484c58b25b87_1280.jpg", 
                     caption="Stellar motion patterns", use_container_width=True)
            print("[DEBUG] Stellar motion patterns image loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load stellar motion patterns image: {str(e)}")

    # Save analysis to database option
    if 'relativistic_parameters' in st.session_state and db_session:
        print("[DEBUG] Setting up database save option")
        st.subheader("Save Analysis to Database")
        analysis_description = st.text_input("Analysis description", "Frame dragging analysis of galactic center stars")

        if st.button("Save Analysis Results"):
            print("[DEBUG] Save Analysis Results button clicked")
            with st.spinner("Saving analysis to database..."):
                try:
                    # Determine source type
                    if 'using_sample' in st.session_state and st.session_state.using_sample:
                        source_type = "synthetic_sample"
                    elif 'using_online' in st.session_state and st.session_state.using_online:
                        source_type = "gaia_archive"
                    else:
                        source_type = "uploaded_files"

                    print(f"[DEBUG] Saving analysis with source type: {source_type}")

                    # Get parameters for saving
                    sgr_a_mass_save = st.session_state.relativistic_parameters.get('schwarzschild_radius_pc', 0) / 2.95e-13  # Convert back to solar masses
                    sgr_a_distance_save = 8.122  # Default value in kpc
                    sgr_a_spin_save = 0.9  # Default spin parameter

                    # Get hypothesis results and p-values if statistical validation was done
                    p_values = st.session_state.get('p_values', None)
                    hypothesis_results = st.session_state.get('hypothesis_results', None)

                    # Save to database
                    success = save_analysis_results(
                        description=analysis_description,
                        source_type=source_type,
                        sample_size=len(st.session_state.analysis_results),
                        sgr_a_mass=sgr_a_mass_save,
                        sgr_a_distance=sgr_a_distance_save,
                        sgr_a_spin=sgr_a_spin_save,
                        analysis_results=st.session_state.analysis_results,
                        rel_params=st.session_state.relativistic_parameters,
                        p_values=p_values,
                        hypothesis_results=hypothesis_results
                    )

                    if success:
                        st.success("Analysis saved to database successfully!")
                        print("[DEBUG] Analysis saved to database successfully")
                    else:
                        st.error("Failed to save analysis. Please check database connection.")
                        print("[ERROR] Failed to save analysis to database")

                except Exception as e:
                    print(f"[ERROR] Error saving to database: {str(e)}")
                    st.error(f"Error saving to database: {str(e)}")

    # Run statistical validation
    if st.button("Run Statistical Validation"):
        print("[DEBUG] Run Statistical Validation button clicked")
        with st.spinner("Performing statistical validation..."):
            try:
                print(f"[DEBUG] Starting Monte Carlo simulation with {num_simulations} simulations")

                # Run Monte Carlo simulations
                mc_results = run_monte_carlo_simulation(
                    st.session_state.analysis_results,
                    n_simulations=num_simulations,
                    null_model=null_hypothesis
                )

                st.session_state.mc_results = mc_results
                print("[DEBUG] Monte Carlo simulation completed")

                # Calculate confidence intervals
                ci_results = calculate_confidence_intervals(
                    mc_results, 
                    confidence_level=confidence_level/100.0
                )

                st.session_state.confidence_intervals = ci_results
                print("[DEBUG] Confidence intervals calculated")

                # Compute p-values
                p_values = compute_p_values(mc_results)
                st.session_state.p_values = p_values
                print(f"[DEBUG] P-values computed: {p_values}")

                # Perform hypothesis testing
                hypothesis_results = perform_hypothesis_testing(
                    p_values, 
                    alpha=(1 - confidence_level/100.0)
                )

                st.session_state.hypothesis_results = hypothesis_results
                print(f"[DEBUG] Hypothesis testing completed: {hypothesis_results}")

                st.success("Statistical validation completed successfully")

                # Show statistical results
                st.subheader("Statistical Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P-value", f"{p_values['fd_effect_p_value']:.4f}")
                with col2:
                    result_text = "Significant" if hypothesis_results["fd_effect_significant"] else "Not Significant"
                    result_delta = "↑" if hypothesis_results["fd_effect_significant"] else "↓"
                    st.metric("Frame Dragging Detection", result_text, delta=result_delta)
                with col3:
                    st.metric(f"{confidence_level}% CI Lower", 
                              f"{ci_results['fd_effect_ci_lower']:.4f}")
                    st.metric(f"{confidence_level}% CI Upper", 
                              f"{ci_results['fd_effect_ci_upper']:.4f}")

                print(f"[DEBUG] Statistical results displayed")

                # Plot statistical results
                st.subheader("Statistical Visualization")
                try:
                    fig = plot_statistical_results(mc_results, ci_results, p_values)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        print("[DEBUG] Statistical visualization created successfully")
                    else:
                        st.error("Failed to create statistical visualization")
                        print("[ERROR] Statistical visualization failed")
                except Exception as viz_error:
                    print(f"[ERROR] Error in statistical visualization: {str(viz_error)}")
                    st.error(f"Error in statistical visualization: {str(viz_error)}")

            except Exception as e:
                print(f"[ERROR] Error in statistical validation: {str(e)}")
                st.error(f"Error in statistical validation: {str(e)}")

# Visualization section (only show if analysis is completed)
if 'analysis_results' in st.session_state:
    print("[DEBUG] Setting up Interactive Visualization section")
    st.header("4. Interactive Visualization")

    visualization_type = st.selectbox(
        "Select visualization type",
        ["Proper Motion Vectors", "Frame Dragging Effect", "3D Visualization", "Combined Analysis"]
    )
    print(f"[DEBUG] Visualization type selected: {visualization_type}")

    try:
        if visualization_type == "Proper Motion Vectors":
            print("[DEBUG] Creating proper motion vectors plot")
            fig = plot_stellar_proper_motions(st.session_state.analysis_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                print("[DEBUG] Proper motion vectors plot displayed successfully")
            else:
                st.error("Failed to create proper motion plot")

        elif visualization_type == "Frame Dragging Effect":
            print("[DEBUG] Creating frame dragging effect plot")
            fig = plot_frame_dragging_vectors(st.session_state.analysis_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                print("[DEBUG] Frame dragging effect plot displayed successfully")
            else:
                st.error("Failed to create frame dragging plot")

        elif visualization_type == "3D Visualization":
            print("[DEBUG] Creating 3D visualization")
            fig = create_3d_visualization(st.session_state.analysis_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                print("[DEBUG] 3D visualization displayed successfully")
            else:
                st.error("Failed to create 3D visualization")

        elif visualization_type == "Combined Analysis":
            print("[DEBUG] Creating combined analysis plots")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = plot_stellar_proper_motions(st.session_state.analysis_results)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                    print("[DEBUG] Combined analysis - proper motion plot displayed")
                else:
                    st.error("Failed to create proper motion plot")
            with col2:
                fig2 = plot_frame_dragging_vectors(st.session_state.analysis_results)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                    print("[DEBUG] Combined analysis - frame dragging plot displayed")
                else:
                    st.error("Failed to create frame dragging plot")

    except Exception as e:
        print(f"[ERROR] Error in visualization: {str(e)}")
        st.error(f"Error creating visualization: {str(e)}")

    # Publication quality figure export
    print("[DEBUG] Setting up publication figure export")
    st.subheader("Export Publication-Quality Figure")

    col1, col2, col3 = st.columns(3)
    with col1:
        dpi = st.slider("DPI", 100, 600, 300, 10)
    with col2:
        fig_width = st.slider("Figure width (inches)", 4, 12, 8)
    with col3:
        fig_height = st.slider("Figure height (inches)", 3, 10, 6)

    include_statistical = st.checkbox("Include statistical results", value=True)
    fig_title = st.text_input("Figure title", "Frame Dragging Effect in Galactic Center Stars")

    print(f"[DEBUG] Publication figure params - DPI: {dpi}, size: {fig_width}x{fig_height}, include_stats: {include_statistical}")

    if st.button("Generate Publication Figure"):
        print("[DEBUG] Generate Publication Figure button clicked")
        with st.spinner("Generating publication-quality figure..."):
            try:
                # Create publication figure
                fig_data = {
                    'analysis_results': st.session_state.analysis_results,
                    'mc_results': st.session_state.mc_results if 'mc_results' in st.session_state else None,
                    'confidence_intervals': st.session_state.confidence_intervals if 'confidence_intervals' in st.session_state else None,
                    'p_values': st.session_state.p_values if 'p_values' in st.session_state else None,
                    'include_statistical': include_statistical,
                    'dpi': dpi,
                    'figsize': (fig_width, fig_height),
                    'title': fig_title
                }

                print("[DEBUG] Creating publication figure with provided data")
                fig = create_publication_figure(fig_data)

                if fig:
                    # Show publication figure
                    st.pyplot(fig)
                    print("[DEBUG] Publication figure displayed successfully")

                    # Save option for figure
                    try:
                        buf = save_results(fig, "publication_figure.pdf", dpi=dpi)
                        st.download_button(
                            label="Download Figure (PDF)",
                            data=buf,
                            file_name="frame_dragging_figure.pdf",
                            mime="application/pdf"
                        )
                        print("[DEBUG] Download button created successfully")
                    except Exception as save_error:
                        print(f"[ERROR] Error creating download: {str(save_error)}")
                        st.error(f"Error creating download: {str(save_error)}")
                else:
                    st.error("Failed to create publication figure")
                    print("[ERROR] Publication figure creation failed")

            except Exception as e:
                print(f"[ERROR] Error generating publication figure: {str(e)}")
                st.error(f"Error generating publication figure: {str(e)}")

# Footer
print("[DEBUG] Setting up footer")
st.markdown("---")
st.write("""
### Frame Dragging Detection Platform

This application was developed to analyze stellar motion around the galactic center and detect the relativistic frame dragging effect.

The analysis includes:
- Data quality filtering based on parallax, proper motion errors, and other parameters
- Calculation of expected frame dragging signatures
- Statistical validation with Monte Carlo simulations
- Interactive visualization of results
- Export capabilities for publication-quality figures
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**About Frame Dragging**

Frame dragging, also known as the Lense-Thirring effect, is a prediction of Einstein's theory of general relativity. It describes how a massive rotating object drags the fabric of spacetime around it.

The effect is extremely small but can be detected through precise measurements of stellar proper motions around the supermassive black hole at the center of our galaxy.
""")

print("[DEBUG] TorsionField application setup completed")