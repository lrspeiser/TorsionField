import os
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

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

# Function to initialize the database and create tables if they don't exist
def init_db():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        st.warning("Database URL not found in environment. Using fallback connection.")
        # Try PostgreSQL default credentials
        db_user = os.environ.get('PGUSER', 'postgres')
        db_pass = os.environ.get('PGPASSWORD', 'postgres')
        db_host = os.environ.get('PGHOST', 'localhost')
        db_port = os.environ.get('PGPORT', '5432')
        db_name = os.environ.get('PGDATABASE', 'postgres')
        
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
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
    session = init_db()
    if not session:
        return False
    
    try:
        # Extract required values
        avg_fd_effect = analysis_results['fd_effect_mag'].mean()
        max_fd_effect = analysis_results['fd_effect_mag'].max()
        snr = rel_params['signal_to_noise'] if 'signal_to_noise' in rel_params else 0.0
        
        # Statistical values if available
        p_value = p_values['combined_p_value'] if p_values and 'combined_p_value' in p_values else None
        significance_level = hypothesis_results['significance_level'] if hypothesis_results and 'significance_level' in hypothesis_results else 0.05
        significant = hypothesis_results['frame_dragging_detected'] if hypothesis_results and 'frame_dragging_detected' in hypothesis_results else False
        
        # Check for rotation statistics
        interpretation = ""
        cohens_d = 0.0
        if 'rotation_statistics' in rel_params:
            rot_stats = rel_params['rotation_statistics']
            cohens_d = rot_stats.get('cohens_d', 0.0)
            interpretation = rot_stats.get('interpretation', "")
            
            # If we don't have p-value from hypothesis testing, use the one from rotation stats
            if p_value is None and 'p_value' in rot_stats:
                p_value = rot_stats['p_value']
        
        # Store additional info as JSON
        additional_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'relativistic_parameters': {k: v for k, v in rel_params.items() if k not in ['rotation_statistics', 'r_centers', 'mean_v_azimuthal', 'std_v_azimuthal']}
        }
        
        # Add the analysis to the database
        analysis = FrameDraggingAnalysis(
            description=description,
            source_type=source_type,
            data_sample_size=sample_size,
            sgr_a_mass=sgr_a_mass,
            sgr_a_distance=sgr_a_distance,
            sgr_a_spin=sgr_a_spin,
            avg_frame_dragging_effect=avg_fd_effect,
            max_frame_dragging_effect=max_fd_effect,
            signal_to_noise_ratio=snr,
            p_value=p_value,
            cohens_d=cohens_d,
            significance_level=significance_level,
            significant_detection=significant,
            interpretation=interpretation,
            additional_info=json.dumps(additional_info)
        )
        
        session.add(analysis)
        session.commit()
        return True
    
    except Exception as e:
        st.error(f"Error saving analysis to database: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()

# Function to retrieve all saved analyses
def get_all_analyses():
    session = init_db()
    if not session:
        return []
    
    try:
        analyses = session.query(FrameDraggingAnalysis).order_by(FrameDraggingAnalysis.analysis_date.desc()).all()
        return analyses
    except Exception as e:
        st.error(f"Error retrieving analyses: {str(e)}")
        return []
    finally:
        session.close()

# Function to convert analyses to a DataFrame for display
def analyses_to_dataframe(analyses):
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
    
    return pd.DataFrame(data)

# Function to get a specific analysis by ID
def get_analysis_by_id(analysis_id):
    session = init_db()
    if not session:
        return None
    
    try:
        analysis = session.query(FrameDraggingAnalysis).filter(FrameDraggingAnalysis.id == analysis_id).first()
        return analysis
    except Exception as e:
        st.error(f"Error retrieving analysis: {str(e)}")
        return None
    finally:
        session.close()