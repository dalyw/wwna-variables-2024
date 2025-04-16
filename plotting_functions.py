"""Plotting helper functions for wastewater analysis."""
from typing import Dict, List, Union
import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.gridspec as gridspec
import re
# LIST OF FUNCTIONS:
# - plot_pie_counts
# - plot_facilities_map

# Common plotting settings
FIGURE_DPI = 300
DEFAULT_CMAP = 'viridis'

def setup_figure(figsize: tuple = (10, 6)) -> tuple:
    """Create and setup a new figure with common settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_and_close(path: str, dpi: int = FIGURE_DPI) -> None:
    """Save figure to path and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_pie_counts(df: pd.DataFrame, title: str) -> None:
    """
    Plot pie chart of parameter categories.
    
    Args:
        df: DataFrame containing PARENT_CATEGORY column
        title: Title for the plot
    """
    category_counts = df['PARENT_CATEGORY'].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(category_counts, 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 4 else '', 
            startangle=140)
    plt.title(title)
    plt.legend(category_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    save_and_close(f'processed_data/step1/{title.lower().replace(" ", "_")}.png')

def plot_facilities_map(num_parameters_per_facility: Dict[str, int], 
                       legend_label: str, 
                       label_threshold: int) -> None:
    """
    Plot facilities on a map of CA.
    
    Args:
        num_parameters_per_facility: Dictionary mapping facility IDs to parameter counts
        legend_label: Label for the legend
        label_threshold: Threshold for labeling facilities
    """
    ca_counties = gpd.read_file('data/ca_counties/CA_Counties.shp')
    facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
    
    # Prepare facilities data
    facilities_with_coords = facilities_list.copy()[
        ['NPDES # CA#', 'LATITUDE DECIMAL DEGREES', 'LONGITUDE DECIMAL DEGREES']
    ].rename(columns={
        'NPDES # CA#': 'NPDES_CODE',
        'LATITUDE DECIMAL DEGREES': 'LATITUDE',
        'LONGITUDE DECIMAL DEGREES': 'LONGITUDE'
    })
    
    facilities_with_coords_merged = pd.DataFrame({
        'NPDES_CODE': list(num_parameters_per_facility.keys())
    }).merge(facilities_with_coords, on='NPDES_CODE', how='left')
    
    facilities_gdf = gpd.GeoDataFrame(
        facilities_with_coords_merged,
        geometry=gpd.points_from_xy(
            facilities_with_coords_merged['LONGITUDE'], 
            facilities_with_coords_merged['LATITUDE']
        ),
        crs="EPSG:4326"
    )

    # Handle CRS
    if facilities_gdf.crs != ca_counties.crs:
        if ca_counties.crs is None:
            ca_counties = ca_counties.set_crs(facilities_gdf.crs)
        else:
            facilities_gdf = facilities_gdf.to_crs(ca_counties.crs)
    
    # Create plot
    fig, ax = setup_figure(figsize=(8, 5))
    ca_counties.plot(ax=ax, color='lightgray')
    facilities_gdf['num_parameters'] = facilities_gdf['NPDES_CODE'].map(num_parameters_per_facility)
    
    # Setup colormap
    norm = plt.Normalize(
        vmin=facilities_gdf['num_parameters'].min(), 
        vmax=facilities_gdf['num_parameters'].max()
    )
    cmap = plt.cm.get_cmap(DEFAULT_CMAP)
    
    facilities_gdf.plot(
        ax=ax, 
        column='num_parameters', 
        cmap=cmap, 
        norm=norm, 
        markersize=10, 
        alpha=0.7
    )

    # Add facility labels
    top_facilities = facilities_gdf[
        facilities_gdf['num_parameters'] >= label_threshold
    ].sort_values('num_parameters', ascending=False).head(10)
    
    top_facilities_sorted = top_facilities.sort_values('LATITUDE', ascending=False)
    
    # Calculate label positions
    label_x = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    label_y_start = ax.get_ylim()[1] - 0.57 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    label_y_step = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for idx, (_, row) in enumerate(top_facilities_sorted.iterrows()):
        if not row.geometry.is_empty and pd.notna(row.geometry.x) and pd.notna(row.geometry.y):
            label_y = label_y_start - idx * label_y_step
            
            ax.annotate(
                f"{row['NPDES_CODE']}", 
                xy=(label_x, label_y),
                xytext=(0, 0), 
                textcoords="offset points",
                fontsize=8,
                ha='left',
                va='center'
            )
            
            ax.plot(
                [row.geometry.x, label_x + 2.5*1e5], 
                [row.geometry.y, label_y], 
                color='black', 
                linewidth=0.5, 
                alpha=0.5
            )

    # Add legend
    unique_params = sorted(facilities_gdf['num_parameters'].unique())
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   label=str(int(value)),
                   markerfacecolor=cmap(norm(value)), 
                   markersize=10)
        for value in unique_params
    ]
    ax.legend(
        handles=legend_elements, 
        title=legend_label, 
        loc='upper right', 
        frameon=False
    )
    
    # Clean up axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    
    save_and_close('processed_data/step3/figures_py/facilities_map.png')

def plot_population_distribution(merged_pop: pd.DataFrame) -> None:
    """Plot distribution of population served."""
    fig, ax = setup_figure()
    plt.hist(merged_pop['population_mean'].dropna(), bins=50)
    plt.xlabel('Population Served')
    plt.ylabel('Number of Facilities')
    plt.title('Distribution of Population Served by CA Wastewater Facilities')
    save_and_close('processed_data/step2/population_distribution.png')

def plot_population_source_comparison(merged_pop: pd.DataFrame) -> None:
    """Plot comparison of population data sources."""
    pop_sources = [col for col in merged_pop.columns if 'population_' in col 
                  and col not in ['population_mean', 'population_std', 'population_cv']]
    fig, ax = setup_figure()
    merged_pop[pop_sources].boxplot()
    plt.yscale('log')
    plt.ylabel('Population (log scale)')
    plt.title('Population Estimates by Data Source')
    plt.xticks(rotation=45)
    save_and_close('processed_data/step2/figures_py/population_source_comparison.png')

def plot_facilities_summary(num_parameters_per_facility: Dict[str, int], 
                          title: str = "Facilities with Exceedances") -> None:
    """Plot summary of facilities without geographic data."""
    df = pd.DataFrame(
        list(num_parameters_per_facility.items()), 
        columns=['Facility', 'Parameters']
    ).sort_values('Parameters', ascending=True)
    
    fig, ax = setup_figure(figsize=(12, 6))
    bars = plt.barh(df['Facility'], df['Parameters'])
    plt.xlabel('Number of Parameters with Slope and Near-Exceedance')
    plt.ylabel('Facility ID')
    plt.title(title)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width, 
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}', 
            ha='left', 
            va='center', 
            fontsize=8
        )
    
    save_and_close('processed_data/step3/figures_py/facilities_summary.png')

def plot_future_limits_summary(df_sorted: pd.DataFrame) -> None:
    """Plot summary of facilities with future limits."""
    fig, ax = setup_figure()
    x = np.arange(len(df_sorted.index))
    width = 0.35
    
    colors = [plt.colormaps[DEFAULT_CMAP](val) for val in [0.2, 0.8]]
    ax.bar(x - width/2, df_sorted['Discharges to Listed'], width, 
           label='Discharging to Listed\nWater Body', color=colors[0])
    ax.bar(x + width/2, df_sorted['Newly Listed and Not Yet Limited'], width,
           label='Discharging to Newly Listed\nWater Body and\nNot Yet Limited', 
           color=colors[1])
    
    plt.ylabel('Number of Facilities', fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.xticks(x, df_sorted.index, rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(df_sorted['Discharges to Listed']):
        ax.text(i - width/2, v, str(int(v)), ha='center', va='bottom')
    for i, v in enumerate(df_sorted['Newly Listed and Not Yet Limited']):
        ax.text(i + width/2, v, str(int(v)), ha='center', va='bottom')
    
    save_and_close('processed_data/step4/figures_py/facilities_with_future_limits_efficient.png')

def plot_facilities_scatter(facilities_with_coords: pd.DataFrame) -> None:
    """Plot scatter plot of facilities when map is unavailable."""
    fig, ax = setup_figure(figsize=(10, 8))
    scatter = plt.scatter(
        facilities_with_coords['LONGITUDE'], 
        facilities_with_coords['LATITUDE'],
        c=facilities_with_coords['Parameters'],
        cmap=DEFAULT_CMAP,
        alpha=0.6
    )
    plt.colorbar(scatter, label='Number of Parameters')
    plt.title('Facilities by Number of Parameters\nwith Possible Future Limits')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    save_and_close('processed_data/step4/figures_py/facilities_summary_scatter.png')

def generate_facility_plots(facilities_list: pd.DataFrame, 
                          limits_2023: pd.DataFrame) -> None:
    """Generate detailed plots for each facility."""
    impaired_facilities = facilities_list[
        facilities_list['Discharges to Impaired Water Bodies and Not Limited'] != ''
    ]
    
    for _, facility in impaired_facilities.iterrows():
        npdes_code = facility['NPDES # CA#']
        impaired_categories = facility['Discharges to Impaired Water Bodies and Not Limited'].split(', ')
        
        if not impaired_categories:
            continue
            
        n_params = len(impaired_categories)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols)
        
        for idx, category in enumerate(impaired_categories):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            param_data = limits_2023[
                (limits_2023['EXTERNAL_PERMIT_NMBR'] == npdes_code) &
                (limits_2023['SUB_CATEGORY'] == category)
            ]
            
            if len(param_data) == 0:
                continue
                
            for _, param_row in param_data.iterrows():
                param_desc = param_row['PARAMETER_DESC']
                limit_value = param_row['LIMIT_VALUE_STANDARD_UNITS']
                
                ax.axhline(y=limit_value, color='r', linestyle='--', alpha=0.5)
                ax.fill_between([-1, 1], [0, 0], [limit_value, limit_value], 
                              color='lightgreen', alpha=0.3)
                ax.fill_between([-1, 1], [limit_value, limit_value], 
                              [limit_value*2, limit_value*2], 
                              color='lightcoral', alpha=0.3)
                
                ax.set_title(f'{param_desc}\n{category}', fontsize=10)
                ax.set_ylabel(param_row['STANDARD_UNIT_DESC'])
        
        save_and_close(f'processed_data/step4/figures_py/{npdes_code}_parameters.png')