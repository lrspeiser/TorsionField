import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

print("[DEBUG] Initializing database module")

# Initialize SQLAlchemy base
Base = declarative_base()

# Define the table for storing frame dragging analysis results
class FrameDraggingAnalysis(Base):
    __tablename__ = 'frame_dragging_analyses'

    id = Column(Integer, primary_key=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    description = Column(String(255))
    source_type = Column(String(50))  # e.g., 'sample', 'uploaded', 'gaia_archive'
    data_sample_size = Column(Integer)

    # Black hole parameters
    sgr_a_mass = Column(Float)  # in solar masses
    sgr_a_distance = Column(Float)  # in kpc
    sgr_a_spin = Column(Float)  # dimensionless

    # Analysis results
    avg_frame_dragging_effect = Column(Float)  # in μas/yr
    max_frame_dragging_effect = Column(Float)  # in μas/yr
    signal_to_noise_ratio = Column(Float)

    # Statistical analysis
    p_value = Column(Float)
    cohens_d = Column(Float)
    significance_level = Column(Float)
    significant_detection = Column(Boolean)
    interpretation = Column(String(255))

    # Additional information can be stored as JSON
    additional_info = Column(Text)  # JSON data for other parameters

    def __repr__(self):
        return f"<FrameDraggingAnalysis(id={self.id}, date={self.analysis_date}, significant={self.significant_detection})>"

# CRITICAL FIX: Function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types to avoid database errors.

    Parameters:
    -----------
    obj : any
        Object that may contain NumPy types

    Returns:
    --------
    any
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Function to initialize the database and create tables if they don't exist
def init_db():
    """
    Initialize database connection and create tables.

    Returns:
    --------
    sqlalchemy.orm.session.Session or None
        Database session if successful, None if failed
    """
    print("[DEBUG] init_db called")

    try:
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            print("[WARNING] DATABASE_URL not found in environment, using fallback")
            st.warning("Database URL not found in environment. Using fallback connection.")
            # Try PostgreSQL default credentials
            db_user = os.environ.get('PGUSER', 'postgres')
            db_pass = os.environ.get('PGPASSWORD', 'postgres')
            db_host = os.environ.get('PGHOST', 'localhost')
            db_port = os.environ.get('PGPORT', '5432')
            db_name = os.environ.get('PGDATABASE', 'postgres')

            db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            print(f"[DEBUG] Using constructed DB URL: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}")

        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        print("[DEBUG] Database initialized successfully")
        return Session()

    except Exception as e:
        print(f"[ERROR] Error initializing database: {str(e)}")
        st.error(f"Error initializing database: {str(e)}")
        return None

# Function to save analysis results to the database
def save_analysis_results(
    description, 
    source_type, 
    sample_size, 
    sgr_a_mass, 
    sgr_a_distance, 
    sgr_a_spin, 
    analysis_results,
    rel_params,
    p_values=None,
    hypothesis_results=None
):
    """
    Save analysis results to the database with NumPy type conversion.

    Parameters:
    -----------
    description : str
        Description of the analysis
    source_type : str
        Type of data source used
    sample_size : int
        Number of stars in the analysis
    sgr_a_mass : float
        Mass of Sgr A* in solar masses
    sgr_a_distance : float
        Distance to galactic center in kpc
    sgr_a_spin : float
        Dimensionless spin parameter
    analysis_results : pandas.DataFrame
        DataFrame with analysis results
    rel_params : dict
        Dictionary with relativistic parameters
    p_values : dict, optional
        Dictionary with p-values
    hypothesis_results : dict, optional
        Dictionary with hypothesis test results

    Returns:
    --------
    bool
        True if successful, False if failed
    """
    print(f"[DEBUG] save_analysis_results called with description: '{description}'")

    session = init_db()
    if not session:
        print("[ERROR] Failed to initialize database session")
        return False

    try:
        # CRITICAL FIX: Convert all NumPy types to Python native types
        print("[DEBUG] Converting NumPy types to Python native types")

        # Extract required values and convert NumPy types
        avg_fd_effect = convert_numpy_types(analysis_results['fd_effect_mag'].mean())
        max_fd_effect = convert_numpy_types(analysis_results['fd_effect_mag'].max())
        snr = convert_numpy_types(rel_params.get('signal_to_noise', 0.0))

        print(f"[DEBUG] Converted values - avg_fd: {avg_fd_effect}, max_fd: {max_fd_effect}, snr: {snr}")

        # Statistical values if available
        p_value = None
        if p_values and 'combined_p_value' in p_values:
            p_value = convert_numpy_types(p_values['combined_p_value'])

        significance_level = 0.05
        if hypothesis_results and 'significance_level' in hypothesis_results:
            significance_level = convert_numpy_types(hypothesis_results['significance_level'])

        significant = False
        if hypothesis_results and 'frame_dragging_detected' in hypothesis_results:
            significant = convert_numpy_types(hypothesis_results['frame_dragging_detected'])

        print(f"[DEBUG] Statistical values - p_value: {p_value}, significant: {significant}")

        # Check for rotation statistics
        interpretation = ""
        cohens_d = 0.0
        if 'rotation_statistics' in rel_params:
            rot_stats = rel_params['rotation_statistics']
            cohens_d = convert_numpy_types(rot_stats.get('cohens_d', 0.0))
            interpretation = str(rot_stats.get('interpretation', ""))

            # If we don't have p-value from hypothesis testing, use the one from rotation stats
            if p_value is None and 'p_value' in rot_stats:
                p_value = convert_numpy_types(rot_stats['p_value'])

        print(f"[DEBUG] Rotation stats - cohens_d: {cohens_d}, interpretation: '{interpretation}'")

        # Store additional info as JSON with NumPy type conversion
        additional_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'relativistic_parameters': {}
        }

        # Convert relativistic parameters, excluding complex objects
        excluded_keys = ['rotation_statistics', 'r_centers', 'mean_v_azimuthal', 'std_v_azimuthal']
        for k, v in rel_params.items():
            if k not in excluded_keys:
                additional_info['relativistic_parameters'][k] = convert_numpy_types(v)

        print(f"[DEBUG] Additional info prepared with {len(additional_info['relativistic_parameters'])} parameters")

        # Convert all parameters to ensure no NumPy types remain
        params = {
            'description': str(description),
            'source_type': str(source_type),
            'data_sample_size': int(sample_size),
            'sgr_a_mass': float(sgr_a_mass),
            'sgr_a_distance': float(sgr_a_distance),
            'sgr_a_spin': float(sgr_a_spin),
            'avg_frame_dragging_effect': float(avg_fd_effect),
            'max_frame_dragging_effect': float(max_fd_effect),
            'signal_to_noise_ratio': float(snr),
            'p_value': float(p_value) if p_value is not None else None,
            'cohens_d': float(cohens_d),
            'significance_level': float(significance_level),
            'significant_detection': bool(significant),
            'interpretation': str(interpretation),
            'additional_info': json.dumps(additional_info)
        }

        print(f"[DEBUG] All parameters converted to native Python types")

        # Add the analysis to the database
        analysis = FrameDraggingAnalysis(**params)

        session.add(analysis)
        session.commit()
        print(f"[DEBUG] Analysis saved to database successfully with ID: {analysis.id}")
        return True

    except Exception as e:
        print(f"[ERROR] Error saving analysis to database: {str(e)}")
        st.error(f"Error saving analysis to database: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()
        print("[DEBUG] Database session closed")

# Function to retrieve all saved analyses
def get_all_analyses():
    """
    Retrieve all saved analyses from the database.

    Returns:
    --------
    list
        List of FrameDraggingAnalysis objects
    """
    print("[DEBUG] get_all_analyses called")

    session = init_db()
    if not session:
        print("[ERROR] Failed to initialize database session")
        return []

    try:
        analyses = session.query(FrameDraggingAnalysis).order_by(FrameDraggingAnalysis.analysis_date.desc()).all()
        print(f"[DEBUG] Retrieved {len(analyses)} analyses from database")
        return analyses
    except Exception as e:
        print(f"[ERROR] Error retrieving analyses: {str(e)}")
        st.error(f"Error retrieving analyses: {str(e)}")
        return []
    finally:
        session.close()
        print("[DEBUG] Database session closed")

# Function to convert analyses to a DataFrame for display
def analyses_to_dataframe(analyses):
    """
    Convert list of analyses to pandas DataFrame for display.

    Parameters:
    -----------
    analyses : list
        List of FrameDraggingAnalysis objects

    Returns:
    --------
    pandas.DataFrame
        DataFrame with analysis data
    """
    print(f"[DEBUG] analyses_to_dataframe called with {len(analyses)} analyses")

    try:
        data = []
        for analysis in analyses:
            data.append({
                'ID': analysis.id,
                'Date': analysis.analysis_date,
                'Description': analysis.description,
                'Source': analysis.source_type,
                'Sample Size': analysis.data_sample_size,
                'Sgr A* Mass (Msun)': analysis.sgr_a_mass,
                'Sgr A* Spin': analysis.sgr_a_spin,
                'Avg FD Effect (μas/yr)': analysis.avg_frame_dragging_effect,
                'SNR': analysis.signal_to_noise_ratio,
                'p-value': analysis.p_value,
                'Significant': "Yes" if analysis.significant_detection else "No",
                'Interpretation': analysis.interpretation
            })

        df = pd.DataFrame(data)
        print(f"[DEBUG] Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        print(f"[ERROR] Error converting analyses to DataFrame: {str(e)}")
        return pd.DataFrame()

# Function to get a specific analysis by ID
def get_analysis_by_id(analysis_id):
    """
    Retrieve a specific analysis by ID.

    Parameters:
    -----------
    analysis_id : int
        ID of the analysis to retrieve

    Returns:
    --------
    FrameDraggingAnalysis or None
        Analysis object if found, None otherwise
    """
    print(f"[DEBUG] get_analysis_by_id called with ID: {analysis_id}")

    session = init_db()
    if not session:
        print("[ERROR] Failed to initialize database session")
        return None

    try:
        analysis = session.query(FrameDraggingAnalysis).filter(FrameDraggingAnalysis.id == analysis_id).first()
        if analysis:
            print(f"[DEBUG] Found analysis with ID: {analysis_id}")
        else:
            print(f"[WARNING] No analysis found with ID: {analysis_id}")
        return analysis
    except Exception as e:
        print(f"[ERROR] Error retrieving analysis: {str(e)}")
        st.error(f"Error retrieving analysis: {str(e)}")
        return None
    finally:
        session.close()
        print("[DEBUG] Database session closed")

print("[DEBUG] Database module loaded successfully")