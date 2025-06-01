# TorsionField

A Python application for analyzing stellar motion around the galactic center to detect signatures of **frame dragging** - a relativistic effect predicted by Einstein's General Theory of Relativity where a massive rotating object drags the fabric of spacetime around it.

## Overview

TorsionField processes astronomical data from the Gaia mission to search for subtle gravitational effects that could provide evidence of spacetime distortion around Sagittarius A*, the supermassive black hole at the center of our galaxy. The application implements sophisticated statistical methods to distinguish potential frame dragging signatures from other sources of stellar motion.

## Features

- **Data Processing**: Load and preprocess stellar data from Gaia source files
- **Quality Control**: Advanced filtering based on measurement errors and stellar parameters
- **Frame Dragging Analysis**: Calculate potential frame dragging signatures from proper motion data
- **Statistical Framework**: Comprehensive statistical analysis with confidence intervals and p-value calculations
- **Visualization**: Interactive plotting of stellar motion patterns and analysis results
- **Publication Ready**: Export capabilities for high-quality scientific figures
- **Hypothesis Testing**: Robust framework with null model comparison for statistical validation

## Scientific Background

Frame dragging (also known as the Lense-Thirring effect) is a phenomenon predicted by General Relativity where a massive rotating body causes nearby spacetime to be "dragged" along with its rotation. For the supermassive black hole at our galaxy's center, this effect would manifest as systematic patterns in the proper motions of nearby stars that differ from purely Newtonian orbital mechanics.

## Installation

### Prerequisites

- Python 3.8+
- Required dependencies (install via pip or conda):

```bash
pip install -r requirements.txt
```

### Key Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `astropy` - Astronomical calculations and coordinate systems
- `matplotlib` - Plotting and visualization
- `scipy` - Statistical analysis
- `astroquery` - Gaia data access (if downloading directly)

## Usage

### Basic Analysis Workflow

1. **Data Loading**: Import Gaia stellar catalog data
2. **Quality Filtering**: Apply measurement error thresholds and parameter cuts
3. **Coordinate Transformation**: Convert to galactic-centered coordinate system
4. **Motion Analysis**: Calculate proper motion patterns and potential frame dragging signatures
5. **Statistical Testing**: Perform hypothesis testing against null models
6. **Visualization**: Generate plots and export results

### Example

```python
# Basic usage example
from torsionfield import StellarAnalyzer

# Initialize analyzer
analyzer = StellarAnalyzer()

# Load and process data
analyzer.load_gaia_data('path/to/gaia_catalog.fits')
analyzer.apply_quality_cuts()

# Perform frame dragging analysis
results = analyzer.analyze_frame_dragging()

# Generate visualization
analyzer.plot_motion_patterns()
analyzer.export_results('output_directory/')
```

## Data Requirements

The analysis requires high-precision astrometric data, typically from:

- **Gaia DR3** or later releases
- **Proper motion measurements** with uncertainties < 0.1 mas/yr
- **Distance estimates** (parallax or photometric)
- **Stellar coordinates** in the galactic center region

## Output

The application generates:

- Statistical summaries of frame dragging analysis
- Confidence intervals and significance tests
- Publication-quality plots of stellar motion patterns
- Data tables suitable for further analysis
- Diagnostic plots for quality assessment

## Scientific Applications

This tool is designed for:

- **Relativistic Astrophysics**: Testing General Relativity in strong gravitational fields
- **Galactic Center Studies**: Understanding dynamics near Sagittarius A*
- **Precision Astrometry**: Exploiting high-precision stellar motion measurements
- **Dark Matter Research**: Distinguishing relativistic effects from dark matter signatures

## Contributing

Contributions are welcome! Areas of particular interest:

- Improved statistical methods
- Additional visualization capabilities
- Performance optimizations
- Extended compatibility with other catalogs

## Acknowledgments

- **Gaia Mission**: European Space Agency's Gaia mission for providing unprecedented astrometric precision
- **Einstein's Legacy**: Built upon the theoretical framework of General Relativity
- **Astronomical Community**: Various contributors to galactic center research methodologies
