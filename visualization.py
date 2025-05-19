import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import streamlit as st

def plot_stellar_proper_motions(df, max_stars=2000):
    """
    Plot stellar proper motions as vectors on the sky.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stellar data and proper motions
    max_stars : int
        Maximum number of stars to plot (for performance)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with proper motion vectors
    """
    # Sample stars if needed
    if len(df) > max_stars:
        df_sample = df.sample(max_stars, random_state=42)
    else:
        df_sample = df
    
    # Create quiver plot in galactic coordinates
    fig = go.Figure()
    
    # Scale vectors for better visualization
    scale_factor = 0.2  # Adjust based on data
    
    # Add scatter points for stars
    fig.add_trace(go.Scatter(
        x=df_sample['l'],
        y=df_sample['b'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_sample['phot_g_mean_mag'],
            colorscale='Viridis_r',
            opacity=0.7,
            colorbar=dict(
                title="G magnitude"
            )
        ),
        text=df_sample['source_id'],
        name='Stars'
    ))
    
    # Add vector arrows for proper motions
    for i, row in df_sample.iterrows():
        # The arrow starting point is the star position
        x0, y0 = row['l'], row['b']
        
        # The arrow components are the proper motion
        dx = row['pmra'] * scale_factor * np.cos(np.radians(row['b']))  # Adjust for cos(dec)
        dy = row['pmdec'] * scale_factor
        
        # Only add arrows for stars with non-zero proper motion
        if np.sqrt(dx**2 + dy**2) > 0:
            fig.add_trace(go.Scatter(
                x=[x0, x0 + dx],
                y=[y0, y0 + dy],
                mode='lines',
                line=dict(
                    color='rgba(50, 50, 200, 0.3)',
                    width=1
                ),
                showlegend=False
            ))
    
    # Add a reference vector
    ref_value = 10  # mas/yr
    ref_x0, ref_y0 = 10, -10  # Position of reference vector
    ref_dx = ref_value * scale_factor
    ref_dy = 0
    
    fig.add_trace(go.Scatter(
        x=[ref_x0, ref_x0 + ref_dx],
        y=[ref_y0, ref_y0],
        mode='lines',
        line=dict(color='red', width=2),
        name=f'{ref_value} mas/yr'
    ))
    
    # Update layout with axis ranges
    fig.update_layout(
        title="Stellar Proper Motions in Galactic Coordinates",
        xaxis_title="Galactic Longitude (l) [deg]",
        yaxis_title="Galactic Latitude (b) [deg]",
        hovermode='closest',
        height=700,
        xaxis=dict(range=[40, -40], autorange="reversed"),  # Center on l=0
        yaxis=dict(range=[-30, 30]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_frame_dragging_vectors(df, max_stars=2000):
    """
    Plot frame dragging effect vectors on the sky.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stellar data and frame dragging calculations
    max_stars : int
        Maximum number of stars to plot (for performance)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with frame dragging vectors
    """
    # Sample stars if needed
    if len(df) > max_stars:
        df_sample = df.sample(max_stars, random_state=42)
    else:
        df_sample = df
    
    # Create quiver plot in galactic coordinates
    fig = go.Figure()
    
    # Scale vectors for better visualization
    # The frame dragging effect is tiny, so we need a large scale factor
    scale_factor = 2e6  # Adjust based on data
    
    # Add scatter points for stars
    fig.add_trace(go.Scatter(
        x=df_sample['l'],
        y=df_sample['b'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_sample['fd_effect_mag'],
            colorscale='Plasma',
            opacity=0.7,
            colorbar=dict(
                title="Frame dragging magnitude (μas/yr)"
            )
        ),
        text=df_sample['source_id'],
        name='Stars'
    ))
    
    # Add vector arrows for frame dragging effect
    for i, row in df_sample.iterrows():
        # The arrow starting point is the star position
        x0, y0 = row['l'], row['b']
        
        # The arrow components are the frame dragging proper motion
        dx = row['fd_pmra'] * scale_factor * np.cos(np.radians(row['b']))  # Adjust for cos(dec)
        dy = row['fd_pmdec'] * scale_factor
        
        # Only add arrows for stars with non-zero frame dragging effect
        if np.sqrt(dx**2 + dy**2) > 0:
            fig.add_trace(go.Scatter(
                x=[x0, x0 + dx],
                y=[y0, y0 + dy],
                mode='lines',
                line=dict(
                    color='rgba(255, 100, 50, 0.3)',
                    width=1
                ),
                showlegend=False
            ))
    
    # Add a reference vector
    ref_value = 1  # μas/yr
    ref_x0, ref_y0 = 10, -10  # Position of reference vector
    ref_dx = ref_value * scale_factor
    ref_dy = 0
    
    fig.add_trace(go.Scatter(
        x=[ref_x0, ref_x0 + ref_dx],
        y=[ref_y0, ref_y0],
        mode='lines',
        line=dict(color='red', width=2),
        name=f'{ref_value} μas/yr'
    ))
    
    # Update layout with axis ranges
    fig.update_layout(
        title="Frame Dragging Effect Vectors in Galactic Coordinates",
        xaxis_title="Galactic Longitude (l) [deg]",
        yaxis_title="Galactic Latitude (b) [deg]",
        hovermode='closest',
        height=700,
        xaxis=dict(range=[40, -40], autorange="reversed"),  # Center on l=0
        yaxis=dict(range=[-30, 30]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_statistical_results(mc_results, ci_results, p_values):
    """
    Plot statistical results from Monte Carlo simulations.
    
    Parameters:
    -----------
    mc_results : dict
        Dictionary with Monte Carlo simulation results
    ci_results : dict
        Dictionary with confidence interval results
    p_values : dict
        Dictionary with p-values
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with statistical results
    """
    # Create subplots with 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Frame Dragging Effect Distribution", 
            "Correlation with 1/r Distribution",
            "Alignment Distribution",
            "Statistical Summary"
        )
    )
    
    # Histogram of frame dragging effect from simulations
    fig.add_trace(
        go.Histogram(
            x=mc_results['simulated_fd_effects'],
            name='Simulated',
            opacity=0.7,
            marker_color='blue',
            nbinsx=30
        ),
        row=1, col=1
    )
    
    # Add vertical line for actual observed value
    fig.add_vline(
        x=mc_results['actual_fd_effect'],
        line_dash="dash",
        line_color="red",
        row=1, col=1
    )
    
    # Add annotation for p-value
    fig.add_annotation(
        text=f"p-value: {p_values['fd_effect_p_value']:.4f}",
        x=0.95, y=0.95,
        xref="x1 domain", yref="y1 domain",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        row=1, col=1
    )
    
    # Histogram of correlation from simulations
    fig.add_trace(
        go.Histogram(
            x=mc_results['simulated_correlations'],
            name='Simulated',
            opacity=0.7,
            marker_color='green',
            nbinsx=30
        ),
        row=1, col=2
    )
    
    # Add vertical line for actual observed correlation
    fig.add_vline(
        x=mc_results['actual_correlation'],
        line_dash="dash",
        line_color="red",
        row=1, col=2
    )
    
    # Add annotation for p-value
    fig.add_annotation(
        text=f"p-value: {p_values['correlation_p_value']:.4f}",
        x=0.95, y=0.95,
        xref="x2 domain", yref="y2 domain",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        row=1, col=2
    )
    
    # Histogram of alignment from simulations
    fig.add_trace(
        go.Histogram(
            x=mc_results['simulated_alignments'],
            name='Simulated',
            opacity=0.7,
            marker_color='purple',
            nbinsx=30
        ),
        row=2, col=1
    )
    
    # Add vertical line for actual observed alignment
    fig.add_vline(
        x=mc_results['actual_alignment'],
        line_dash="dash",
        line_color="red",
        row=2, col=1
    )
    
    # Add annotation for p-value
    fig.add_annotation(
        text=f"p-value: {p_values['alignment_p_value']:.4f}",
        x=0.95, y=0.95,
        xref="x3 domain", yref="y3 domain",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        row=2, col=1
    )
    
    # Summary table for statistical results
    summary_data = {
        'Metric': ['Frame Dragging Effect', 'Correlation with 1/r', 'Alignment', 'Combined'],
        'Observed Value': [
            f"{mc_results['actual_fd_effect']:.4g}",
            f"{mc_results['actual_correlation']:.4g}",
            f"{mc_results['actual_alignment']:.4g}",
            "N/A"
        ],
        'p-value': [
            f"{p_values['fd_effect_p_value']:.4g}",
            f"{p_values['correlation_p_value']:.4g}",
            f"{p_values['alignment_p_value']:.4g}",
            f"{p_values['combined_p_value']:.4g}"
        ],
        'Significant': [
            "Yes" if p_values['fd_effect_p_value'] < 0.05 else "No",
            "Yes" if p_values['correlation_p_value'] < 0.05 else "No",
            "Yes" if p_values['alignment_p_value'] < 0.05 else "No",
            "Yes" if p_values['combined_p_value'] < 0.05 else "No"
        ]
    }
    
    # Create table
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_data.keys()),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[summary_data[k] for k in summary_data.keys()],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f"Statistical Analysis Results (Null model: {mc_results['null_model']})",
        showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Frame Dragging Effect (μas/yr)", row=1, col=1)
    fig.update_xaxes(title_text="Correlation with 1/r", row=1, col=2)
    fig.update_xaxes(title_text="Alignment", row=2, col=1)
    
    return fig


def create_3d_visualization(df, max_stars=1000):
    """
    Create a 3D visualization of stars and frame dragging effects.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stellar data and frame dragging calculations
    max_stars : int
        Maximum number of stars to plot (for performance)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with 3D visualization
    """
    # Sample stars if needed
    if len(df) > max_stars:
        df_sample = df.sample(max_stars, random_state=42)
    else:
        df_sample = df
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add the galactic center as a special point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            symbol='diamond'
        ),
        name='Galactic Center (Sgr A*)'
    ))
    
    # Add stars as scatter points
    fig.add_trace(go.Scatter3d(
        x=df_sample['x_gc'],
        y=df_sample['y_gc'],
        z=df_sample['z_gc'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_sample['fd_effect_mag'],
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(
                title="Frame dragging magnitude (μas/yr)"
            )
        ),
        text=[f"Star ID: {id}<br>Frame dragging: {fd:.4g} μas/yr"
              for id, fd in zip(df_sample['source_id'], df_sample['fd_effect_mag'])],
        name='Stars'
    ))
    
    # Update layout
    fig.update_layout(
        title="3D Distribution of Stars Around Galactic Center with Frame Dragging Effects",
        scene=dict(
            xaxis_title="X (pc)",
            yaxis_title="Y (pc)",
            zaxis_title="Z (pc)",
            aspectmode='data'
        ),
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_publication_figure(fig_data):
    """
    Create a publication-quality figure for the frame dragging analysis.
    
    Parameters:
    -----------
    fig_data : dict
        Dictionary with data for creating the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    # Extract data
    df = fig_data['analysis_results']
    mc_results = fig_data['mc_results']
    confidence_intervals = fig_data['confidence_intervals']
    p_values = fig_data['p_values']
    include_statistical = fig_data['include_statistical']
    dpi = fig_data['dpi']
    figsize = fig_data['figsize']
    title = fig_data['title']
    
    # Create figure with subplots
    if include_statistical and mc_results is not None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        
    # Make sure axes is an array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    # Plot proper motions
    ax = axes[0]
    
    # Sample for plotting if needed
    max_stars = 500
    if len(df) > max_stars:
        df_sample = df.sample(max_stars, random_state=42)
    else:
        df_sample = df
    
    # Convert to galactic coordinates
    l = df_sample['l']
    b = df_sample['b']
    
    # Scale factor for proper motion vectors
    pm_scale = 0.2
    
    # Plot stars
    sc = ax.scatter(l, b, c=df_sample['phot_g_mean_mag'], 
                  cmap='viridis_r', s=10, alpha=0.7)
    
    # Plot proper motion vectors
    for i, row in df_sample.iterrows():
        x0, y0 = row['l'], row['b']
        dx = row['pmra'] * pm_scale * np.cos(np.radians(row['b']))
        dy = row['pmdec'] * pm_scale
        
        # Only plot vectors for stars with significant proper motion
        if np.sqrt(dx**2 + dy**2) > 0.01:
            ax.arrow(x0, y0, dx, dy, head_width=0.5, 
                    head_length=0.5, fc='blue', ec='blue', alpha=0.3)
    
    # Add reference vector
    ref_value = 10  # mas/yr
    ref_x0, ref_y0 = 20, -25
    ref_dx = ref_value * pm_scale
    ref_dy = 0
    ax.arrow(ref_x0, ref_y0, ref_dx, ref_dy, head_width=0.5, 
            head_length=0.5, fc='blue', ec='blue')
    ax.text(ref_x0 + ref_dx/2, ref_y0 - 2, f"{ref_value} mas/yr", 
           ha='center', va='center', color='blue')
    
    # Set axis limits
    ax.set_xlim(40, -40)
    ax.set_ylim(-30, 30)
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('G magnitude')
    
    # Add labels
    ax.set_xlabel('Galactic Longitude (l) [deg]')
    ax.set_ylabel('Galactic Latitude (b) [deg]')
    ax.set_title('Stellar Proper Motions')
    
    # Plot frame dragging vectors
    ax = axes[1]
    
    # Scale factor for frame dragging vectors (much larger as effect is tiny)
    fd_scale = 1e6
    
    # Plot stars
    sc = ax.scatter(l, b, c=df_sample['fd_effect_mag'], 
                  cmap='plasma', s=10, alpha=0.7)
    
    # Plot frame dragging vectors
    for i, row in df_sample.iterrows():
        x0, y0 = row['l'], row['b']
        dx = row['fd_pmra'] * fd_scale * np.cos(np.radians(row['b']))
        dy = row['fd_pmdec'] * fd_scale
        
        # Only plot vectors for stars with significant frame dragging
        if np.sqrt(dx**2 + dy**2) > 0.01:
            ax.arrow(x0, y0, dx, dy, head_width=0.5, 
                    head_length=0.5, fc='red', ec='red', alpha=0.3)
    
    # Add reference vector
    ref_value = 1  # μas/yr
    ref_x0, ref_y0 = 20, -25
    ref_dx = ref_value * fd_scale
    ref_dy = 0
    ax.arrow(ref_x0, ref_y0, ref_dx, ref_dy, head_width=0.5, 
            head_length=0.5, fc='red', ec='red')
    ax.text(ref_x0 + ref_dx/2, ref_y0 - 2, f"{ref_value} μas/yr", 
           ha='center', va='center', color='red')
    
    # Set axis limits
    ax.set_xlim(40, -40)
    ax.set_ylim(-30, 30)
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Frame dragging magnitude (μas/yr)')
    
    # Add labels
    ax.set_xlabel('Galactic Longitude (l) [deg]')
    ax.set_ylabel('Galactic Latitude (b) [deg]')
    ax.set_title('Frame Dragging Effect')
    
    # Add statistical plots if requested
    if include_statistical and mc_results is not None and len(axes) > 2:
        # Plot histogram of frame dragging effect
        ax = axes[2]
        ax.hist(mc_results['simulated_fd_effects'], bins=30, alpha=0.7, color='blue')
        ax.axvline(mc_results['actual_fd_effect'], color='red', linestyle='--')
        ax.set_xlabel('Frame Dragging Effect (μas/yr)')
        ax.set_ylabel('Frequency')
        ax.set_title('Frame Dragging Effect Distribution')
        ax.text(0.95, 0.95, f"p-value: {p_values['fd_effect_p_value']:.4f}", 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Plot statistical summary
        ax = axes[3]
        ax.axis('off')  # Turn off axes
        
        # Create a table with statistical results
        table_data = [
            ['Metric', 'Observed', 'p-value', 'Significant'],
            ['Frame Dragging', f"{mc_results['actual_fd_effect']:.4g}", 
             f"{p_values['fd_effect_p_value']:.4g}", 
             "Yes" if p_values['fd_effect_p_value'] < 0.05 else "No"],
            ['Correlation', f"{mc_results['actual_correlation']:.4g}", 
             f"{p_values['correlation_p_value']:.4g}", 
             "Yes" if p_values['correlation_p_value'] < 0.05 else "No"],
            ['Alignment', f"{mc_results['actual_alignment']:.4g}", 
             f"{p_values['alignment_p_value']:.4g}", 
             "Yes" if p_values['alignment_p_value'] < 0.05 else "No"],
            ['Combined', "N/A", f"{p_values['combined_p_value']:.4g}", 
             "Yes" if p_values['combined_p_value'] < 0.05 else "No"]
        ]
        
        # Create the table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                       loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Set title for this subplot
        ax.set_title('Statistical Summary')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig
