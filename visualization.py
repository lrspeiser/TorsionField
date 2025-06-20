import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import streamlit as st
import warnings

# CRITICAL FIX: Suppress matplotlib warnings and use constrained layout
warnings.filterwarnings('ignore', 'Tight layout not applied.*')
plt.rcParams['figure.constrained_layout.use'] = True

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
    print(f"[DEBUG] plot_stellar_proper_motions called with {len(df)} stars")

    try:
        # Sample stars if needed
        if len(df) > max_stars:
            df_sample = df.sample(max_stars, random_state=42)
            print(f"[DEBUG] Sampled {max_stars} stars from {len(df)} total")
        else:
            df_sample = df
            print(f"[DEBUG] Using all {len(df_sample)} stars")

        # Create quiver plot in galactic coordinates
        fig = go.Figure()

        # Scale vectors for better visualization
        scale_factor = 0.2  # Adjust based on data
        print(f"[DEBUG] Using scale factor: {scale_factor}")

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
        print(f"[DEBUG] Added {len(df_sample)} star markers")

        # Add vector arrows for proper motions
        arrow_count = 0
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
                arrow_count += 1
        print(f"[DEBUG] Added {arrow_count} proper motion arrows")

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
        print(f"[DEBUG] Added reference vector")

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
        print(f"[DEBUG] plot_stellar_proper_motions completed successfully")

        return fig

    except Exception as e:
        print(f"[ERROR] plot_stellar_proper_motions failed: {str(e)}")
        st.error(f"Error creating proper motion plot: {str(e)}")
        return None


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
    print(f"[DEBUG] plot_frame_dragging_vectors called with {len(df)} stars")

    try:
        # Sample stars if needed
        if len(df) > max_stars:
            df_sample = df.sample(max_stars, random_state=42)
            print(f"[DEBUG] Sampled {max_stars} stars from {len(df)} total")
        else:
            df_sample = df
            print(f"[DEBUG] Using all {len(df_sample)} stars")

        # Create quiver plot in galactic coordinates
        fig = go.Figure()

        # Scale vectors for better visualization
        # The frame dragging effect is tiny, so we need a large scale factor
        scale_factor = 2e6  # Adjust based on data
        print(f"[DEBUG] Using scale factor: {scale_factor}")

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
        print(f"[DEBUG] Added {len(df_sample)} star markers with frame dragging coloring")

        # Add vector arrows for frame dragging effect
        arrow_count = 0
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
                arrow_count += 1
        print(f"[DEBUG] Added {arrow_count} frame dragging arrows")

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
        print(f"[DEBUG] Added reference vector")

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
        print(f"[DEBUG] plot_frame_dragging_vectors completed successfully")

        return fig

    except Exception as e:
        print(f"[ERROR] plot_frame_dragging_vectors failed: {str(e)}")
        st.error(f"Error creating frame dragging plot: {str(e)}")
        return None


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
    print(f"[DEBUG] plot_statistical_results called")

    try:
        # CRITICAL FIX: Create separate figures instead of subplots with tables
        # The error was caused by mixing 'table' trace type with 'xy' subplot type

        # Create a single figure with multiple histograms only
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Frame Dragging Effect Distribution", 
                "Correlation with 1/r Distribution",
                "Alignment Distribution",
                "P-Values Summary"
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]  # All xy subplots
        )
        print(f"[DEBUG] Created subplot structure with xy types only")

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
        print(f"[DEBUG] Added frame dragging effect histogram")

        # Add vertical line for actual observed value
        fig.add_vline(
            x=mc_results['actual_fd_effect'],
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )
        print(f"[DEBUG] Added observed value line")

        # CRITICAL FIX: Use "paper" instead of "x1 domain" for xref
        fig.add_annotation(
            text=f"p-value: {p_values['fd_effect_p_value']:.4f}",
            x=0.25, y=0.95,  # Position in first quadrant
            xref="paper", yref="paper",  # FIXED: was "x1 domain"
            showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        print(f"[DEBUG] Added annotation for p-value")

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
        print(f"[DEBUG] Added correlation histogram")

        # Add vertical line for actual observed correlation
        fig.add_vline(
            x=mc_results['actual_correlation'],
            line_dash="dash",
            line_color="red",
            row=1, col=2
        )

        # CRITICAL FIX: Use "paper" instead of "x2" for xref
        fig.add_annotation(
            text=f"p-value: {p_values['correlation_p_value']:.4f}",
            x=0.75, y=0.95,  # Position in second quadrant
            xref="paper", yref="paper",  # FIXED: was "x2"
            showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        print(f"[DEBUG] Added correlation annotation")

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
        print(f"[DEBUG] Added alignment histogram")

        # Add vertical line for actual observed alignment
        fig.add_vline(
            x=mc_results['actual_alignment'],
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )

        # CRITICAL FIX: Use "paper" instead of "x3" for xref
        fig.add_annotation(
            text=f"p-value: {p_values['alignment_p_value']:.4f}",
            x=0.25, y=0.45,  # Position in third quadrant
            xref="paper", yref="paper",  # FIXED: was "x3"
            showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        print(f"[DEBUG] Added alignment annotation")

        # CRITICAL FIX: Instead of table, create a bar chart for p-values
        metrics = ['Frame Dragging', 'Correlation', 'Alignment', 'Combined']
        p_vals = [
            p_values['fd_effect_p_value'],
            p_values['correlation_p_value'],
            p_values['alignment_p_value'],
            p_values['combined_p_value']
        ]

        # Color bars based on significance
        colors = ['red' if p < 0.05 else 'blue' for p in p_vals]

        fig.add_trace(
            go.Bar(
                x=metrics,
                y=p_vals,
                marker_color=colors,
                name='P-values',
                text=[f"{p:.4f}" for p in p_vals],
                textposition='auto'
            ),
            row=2, col=2
        )

        # Add significance line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            row=2, col=2
        )

        print(f"[DEBUG] Added p-values bar chart")

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
        fig.update_xaxes(title_text="Metric", row=2, col=2)

        # Update y-axis labels
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="P-value", row=2, col=2)

        print(f"[DEBUG] plot_statistical_results completed successfully")
        return fig

    except Exception as e:
        print(f"[ERROR] plot_statistical_results failed: {str(e)}")
        st.error(f"Error creating statistical plot: {str(e)}")
        # Return simple fallback plot
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Statistical Analysis Error<br>P-value: {p_values.get('fd_effect_p_value', 'N/A'):.4f}",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Statistical Analysis Results (Error)")
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
    print(f"[DEBUG] create_3d_visualization called with {len(df)} stars")

    try:
        # Sample stars if needed
        if len(df) > max_stars:
            df_sample = df.sample(max_stars, random_state=42)
            print(f"[DEBUG] Sampled {max_stars} stars from {len(df)} total")
        else:
            df_sample = df
            print(f"[DEBUG] Using all {len(df_sample)} stars")

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
        print(f"[DEBUG] Added galactic center marker")

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
        print(f"[DEBUG] Added {len(df_sample)} star markers in 3D")

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

        print(f"[DEBUG] create_3d_visualization completed successfully")
        return fig

    except Exception as e:
        print(f"[ERROR] create_3d_visualization failed: {str(e)}")
        st.error(f"Error creating 3D visualization: {str(e)}")
        return None


# Fix for the publication figure binning error in visualization.py

def create_publication_figure(fig_data):
    """
    Create a publication-quality figure for the frame dragging analysis.
    FIXED: Better handling of histogram binning for small data ranges.

    Parameters:
    -----------
    fig_data : dict
        Dictionary with data for creating the figure

    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    print(f"[DEBUG] create_publication_figure called")

    try:
        # Extract data
        df = fig_data['analysis_results']
        mc_results = fig_data['mc_results']
        confidence_intervals = fig_data['confidence_intervals']
        p_values = fig_data['p_values']
        include_statistical = fig_data['include_statistical']
        dpi = fig_data['dpi']
        figsize = fig_data['figsize']
        title = fig_data['title']

        print(f"[DEBUG] Extracted figure data: {len(df)} stars, include_statistical={include_statistical}")

        # Make sure we have sufficient data for histograms
        if mc_results and 'simulated_fd_effects' in mc_results:
            sim_values = mc_results['simulated_fd_effects']
            if len(sim_values) < 10:
                include_statistical = False
                print(f"[DEBUG] Not enough simulation data, disabling statistical plots")

        # CRITICAL FIX: Use constrained_layout instead of tight_layout
        if include_statistical and mc_results is not None:
            fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, layout='constrained')
            axes = axes.flatten()
            print(f"[DEBUG] Created 2x2 subplot layout with constrained layout")
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, layout='constrained')
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            print(f"[DEBUG] Created 1x2 subplot layout with constrained layout")

        # Make sure axes is an array
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Plot proper motions
        ax = axes[0]
        print(f"[DEBUG] Starting proper motion plot")

        # Sample for plotting if needed
        max_stars = 500
        if len(df) > max_stars:
            df_sample = df.sample(max_stars, random_state=42)
            print(f"[DEBUG] Sampled {max_stars} stars for plotting")
        else:
            df_sample = df
            print(f"[DEBUG] Using all {len(df_sample)} stars for plotting")

        # Convert to galactic coordinates
        l = df_sample['l']
        b = df_sample['b']

        # Scale factor for proper motion vectors
        pm_scale = 0.2
        print(f"[DEBUG] Using proper motion scale factor: {pm_scale}")

        # Plot stars
        sc = ax.scatter(l, b, c=df_sample['phot_g_mean_mag'], 
                      cmap='viridis_r', s=10, alpha=0.7)
        print(f"[DEBUG] Added {len(df_sample)} stars to proper motion plot")

        # Plot proper motion vectors
        arrow_count = 0
        for i, row in df_sample.iterrows():
            x0, y0 = row['l'], row['b']
            dx = row['pmra'] * pm_scale * np.cos(np.radians(row['b']))
            dy = row['pmdec'] * pm_scale

            # Only plot vectors for stars with significant proper motion
            if np.sqrt(dx**2 + dy**2) > 0.01:
                ax.arrow(x0, y0, dx, dy, head_width=0.5, 
                        head_length=0.5, fc='blue', ec='blue', alpha=0.3)
                arrow_count += 1
        print(f"[DEBUG] Added {arrow_count} proper motion arrows")

        # Add reference vector
        ref_value = 10  # mas/yr
        ref_x0, ref_y0 = 20, -25
        ref_dx = ref_value * pm_scale
        ref_dy = 0
        ax.arrow(ref_x0, ref_y0, ref_dx, ref_dy, head_width=0.5, 
                head_length=0.5, fc='blue', ec='blue')
        ax.text(ref_x0 + ref_dx/2, ref_y0 - 2, f"{ref_value} mas/yr", 
               ha='center', va='center', color='blue')
        print(f"[DEBUG] Added reference vector")

        # Set axis limits
        ax.set_xlim(40, -40)
        ax.set_ylim(-30, 30)

        # Add colorbar
        try:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('G magnitude')
            print(f"[DEBUG] Added colorbar to proper motion plot")
        except Exception as e:
            print(f"[WARNING] Could not add colorbar: {str(e)}")

        # Add labels
        ax.set_xlabel('Galactic Longitude (l) [deg]')
        ax.set_ylabel('Galactic Latitude (b) [deg]')
        ax.set_title('Stellar Proper Motions')

        # Plot frame dragging vectors
        ax = axes[1]
        print(f"[DEBUG] Starting frame dragging plot")

        # Scale factor for frame dragging vectors (much larger as effect is tiny)
        fd_scale = 1e6
        print(f"[DEBUG] Using frame dragging scale factor: {fd_scale}")

        # Plot stars
        sc = ax.scatter(l, b, c=df_sample['fd_effect_mag'], 
                      cmap='plasma', s=10, alpha=0.7)
        print(f"[DEBUG] Added {len(df_sample)} stars to frame dragging plot")

        # Plot frame dragging vectors
        arrow_count = 0
        for i, row in df_sample.iterrows():
            x0, y0 = row['l'], row['b']
            dx = row['fd_pmra'] * fd_scale * np.cos(np.radians(row['b']))
            dy = row['fd_pmdec'] * fd_scale

            # Only plot vectors for stars with significant frame dragging
            if np.sqrt(dx**2 + dy**2) > 0.01:
                ax.arrow(x0, y0, dx, dy, head_width=0.5, 
                        head_length=0.5, fc='red', ec='red', alpha=0.3)
                arrow_count += 1
        print(f"[DEBUG] Added {arrow_count} frame dragging arrows")

        # Add reference vector
        ref_value = 1  # μas/yr
        ref_x0, ref_y0 = 20, -25
        ref_dx = ref_value * fd_scale
        ref_dy = 0
        ax.arrow(ref_x0, ref_y0, ref_dx, ref_dy, head_width=0.5, 
                head_length=0.5, fc='red', ec='red')
        ax.text(ref_x0 + ref_dx/2, ref_y0 - 2, f"{ref_value} μas/yr", 
               ha='center', va='center', color='red')
        print(f"[DEBUG] Added frame dragging reference vector")

        # Set axis limits
        ax.set_xlim(40, -40)
        ax.set_ylim(-30, 30)

        # Add colorbar
        try:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Frame dragging magnitude (μas/yr)')
            print(f"[DEBUG] Added colorbar to frame dragging plot")
        except Exception as e:
            print(f"[WARNING] Could not add colorbar: {str(e)}")

        # Add labels
        ax.set_xlabel('Galactic Longitude (l) [deg]')
        ax.set_ylabel('Galactic Latitude (b) [deg]')
        ax.set_title('Frame Dragging Effect')

        # Add statistical plots if requested
        if include_statistical and mc_results is not None and len(axes) > 2:
            print(f"[DEBUG] Adding statistical plots")

            # Plot histogram of frame dragging effect
            ax = axes[2]

            # CRITICAL FIX: Better histogram binning logic
            sim_values = mc_results['simulated_fd_effects']
            print(f"[DEBUG] Simulation values range: {np.min(sim_values):.2e} to {np.max(sim_values):.2e}")

            # Check if we have valid data for histogram
            if len(sim_values) > 0:
                data_range = np.max(sim_values) - np.min(sim_values)
                unique_values = len(np.unique(sim_values))

                print(f"[DEBUG] Data range: {data_range:.2e}, Unique values: {unique_values}")

                # FIXED: More robust bin calculation
                if data_range > 0 and unique_values > 1:
                    # Use the minimum of several bin calculation methods
                    sturges_bins = max(1, int(np.log2(len(sim_values)) + 1))
                    sqrt_bins = max(1, int(np.sqrt(len(sim_values))))
                    unique_bins = min(unique_values, 50)  # Cap at 50 bins

                    # Choose the most appropriate bin count
                    num_bins = min(30, max(3, min(sturges_bins, sqrt_bins, unique_bins)))

                    print(f"[DEBUG] Calculated bins - Sturges: {sturges_bins}, Sqrt: {sqrt_bins}, Unique: {unique_bins}, Final: {num_bins}")

                    try:
                        # Try to create histogram with calculated bins
                        n, bins, patches = ax.hist(sim_values, bins=num_bins, alpha=0.7, color='blue')
                        print(f"[DEBUG] Successfully created histogram with {len(bins)-1} bins")

                        # Add vertical line for actual value
                        ax.axvline(mc_results['actual_fd_effect'], color='red', linestyle='--', 
                                 label=f"Observed: {mc_results['actual_fd_effect']:.2e}")

                    except Exception as hist_error:
                        print(f"[WARNING] Histogram with {num_bins} bins failed: {str(hist_error)}")
                        # Fallback: try with fewer bins
                        try:
                            n, bins, patches = ax.hist(sim_values, bins=3, alpha=0.7, color='blue')
                            ax.axvline(mc_results['actual_fd_effect'], color='red', linestyle='--')
                            print(f"[DEBUG] Fallback histogram with 3 bins succeeded")
                        except Exception as fallback_error:
                            print(f"[ERROR] Even fallback histogram failed: {str(fallback_error)}")
                            # Last resort: just plot the data points
                            ax.scatter(range(len(sim_values)), sim_values, alpha=0.5, s=1)
                            ax.axhline(mc_results['actual_fd_effect'], color='red', linestyle='--')
                            print(f"[DEBUG] Using scatter plot as final fallback")

                else:
                    print(f"[WARNING] Data range too small for histogram, using scatter plot")
                    # If data range is too small, show a scatter plot instead
                    ax.scatter(range(len(sim_values)), sim_values, alpha=0.5, s=1)
                    ax.axhline(mc_results['actual_fd_effect'], color='red', linestyle='--')

            else:
                print(f"[WARNING] No simulation values available")
                ax.text(0.5, 0.5, 'No simulation data available', 
                       transform=ax.transAxes, ha='center', va='center')

            ax.set_xlabel('Frame Dragging Effect (μas/yr)')
            ax.set_ylabel('Frequency')
            ax.set_title('Frame Dragging Effect Distribution')

            # Add p-value annotation
            if p_values and 'fd_effect_p_value' in p_values:
                ax.text(0.95, 0.95, f"p-value: {p_values['fd_effect_p_value']:.4f}", 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

            print(f"[DEBUG] Added frame dragging histogram with fallback handling")

            # Plot statistical summary
            ax = axes[3]
            ax.axis('off')  # Turn off axes
            print(f"[DEBUG] Creating statistical summary table")

            # Create a simple text summary instead of a complex table
            if p_values:
                summary_text = f"""Statistical Summary

Frame Dragging Effect:
  Observed: {mc_results.get('actual_fd_effect', 0):.2e}
  p-value: {p_values.get('fd_effect_p_value', 1):.4f}
  Significant: {'Yes' if p_values.get('fd_effect_p_value', 1) < 0.05 else 'No'}

Correlation with 1/r:
  Observed: {mc_results.get('actual_correlation', 0):.4f}
  p-value: {p_values.get('correlation_p_value', 1):.4f}
  Significant: {'Yes' if p_values.get('correlation_p_value', 1) < 0.05 else 'No'}

Combined Analysis:
  p-value: {p_values.get('combined_p_value', 1):.4f}
  Result: {'SIGNIFICANT' if p_values.get('combined_p_value', 1) < 0.05 else 'NOT SIGNIFICANT'}
"""
                ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No statistical results available', 
                       transform=ax.transAxes, ha='center', va='center')

            ax.set_title('Statistical Summary')
            print(f"[DEBUG] Added statistical summary text")

        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)

        print(f"[DEBUG] create_publication_figure completed successfully")

        return fig

    except Exception as e:
        print(f"[ERROR] create_publication_figure failed: {str(e)}")
        st.error(f"Error creating publication figure: {str(e)}")
        # Return a simple fallback figure
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        ax.text(0.5, 0.5, f"Error creating publication figure\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Publication Figure Error")
        return fig