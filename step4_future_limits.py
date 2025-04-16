import pandas as pd
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
from helper_functions import *
import json
from collections import defaultdict
from shapely.geometry import Point
import numpy as np
import os
import matplotlib.gridspec as gridspec
from plotting_functions import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs('processed_data/step4', exist_ok=True)

def load_and_process_data():
    """Load and process all required data."""
    logger.info("Loading data...")
    
    # Load facilities list
    facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
    
    # Load and process 2023 limits
    limits_2023 = read_limits(2023)
    
    # Load IR parameter data and create category maps
    ir_parameter_df = pd.read_csv('processed_data/step1/ir_parameter_df.csv')
    
    # Debug: print columns to see what we have
    # logger.info(f"IR parameter columns: {ir_parameter_df.columns.tolist()}")
    
    # Map categories to limits using parameter descriptions
    # First, map parameter codes to descriptions using REF_PARAMETER
    parameter_desc_map = dict(zip(ref_parameter['PARAMETER_CODE'].str.lstrip('0'), 
                                ref_parameter['PARAMETER_DESC']))
    
    # Then map descriptions to categories
    desc_to_parent = dict(zip(ir_parameter_df['IR_PARAMETER_DESC'], ir_parameter_df['PARENT_CATEGORY']))
    desc_to_sub = dict(zip(ir_parameter_df['IR_PARAMETER_DESC'], ir_parameter_df['SUB_CATEGORY']))
    
    # Apply mappings to limits
    limits_2023['PARAMETER_DESC_CLEAN'] = limits_2023['PARAMETER_CODE'].str.lstrip('0').map(parameter_desc_map)
    limits_2023['PARENT_CATEGORY'] = limits_2023['PARAMETER_DESC_CLEAN'].map(desc_to_parent)
    limits_2023['SUB_CATEGORY'] = limits_2023['PARAMETER_DESC_CLEAN'].map(desc_to_sub)
    
    # Log unmapped parameters
    unmapped_params = limits_2023[limits_2023['PARENT_CATEGORY'].isna()]['PARAMETER_DESC'].unique()
    if len(unmapped_params) > 0:
        logger.warning(f"Unmapped parameters: {unmapped_params}")
    
    # Get unique categories, removing any None values
    parent_categories = [cat for cat in ir_parameter_df['PARENT_CATEGORY'].unique() if pd.notna(cat)]
    sub_categories = [cat for cat in ir_parameter_df['SUB_CATEGORY'].unique() if pd.notna(cat)]
    
    logger.info(f"Found {len(parent_categories)} parent categories and {len(sub_categories)} sub categories")
    
    # Load 303d lists
    columns_to_keep = [
        'Water Body CALWNUMS', 'Pollutant', 'Pollutant Category', 
        'Decision Status', 'TMDL Requirement Status',
        'Sources', 'Expected TMDL Completion Date', 'Expected Attainment Date'
    ]
    
    impaired_303d = {}
    for year, rows_to_skip in [(2018, 2), (2024, 1)]:
        impaired_303d[year] = pd.read_csv(
            f'data/ir/{year}-303d.csv', 
            skiprows=rows_to_skip
        )[columns_to_keep].dropna(subset=['Water Body CALWNUMS'])
        
        # Map categories directly from pollutant names
        impaired_303d[year]['PARENT_CATEGORY'] = impaired_303d[year]['Pollutant'].map(desc_to_parent)
        impaired_303d[year]['SUB_CATEGORY'] = impaired_303d[year]['Pollutant'].map(desc_to_sub)
        
        # Debug: check mapping coverage
        unmapped_pollutants = set(impaired_303d[year]['Pollutant']) - set(desc_to_parent.keys())
        if unmapped_pollutants:
            logger.warning(f"Unmapped pollutants in {year} data: {unmapped_pollutants}")
    
    # Load parameter sorting dictionary
    with open('data/manual_updates/parameter_sorting_dict.json', 'r') as f:
        parameter_sorting_dict = json.load(f)
        
    return (facilities_list, limits_2023, impaired_303d, parameter_sorting_dict, 
            parent_categories, sub_categories)

def analyze_impaired_waters(facilities_list, limits_2023, impaired_303d, parameter_sorting_dict, 
                          parent_categories, sub_categories):
    """Analyze impaired waters and identify facilities requiring future limits."""
    logger.info("Analyzing impaired waters...")
    
    # Create dictionaries for impaired water bodies
    newly_impaired_water_bodies = defaultdict(set)
    impaired_water_bodies = defaultdict(set)
    
    for category in sub_categories:
        logger.debug(f"Processing category: {category}")
        impaired_set_2018 = set(impaired_303d[2018].loc[impaired_303d[2018]['SUB_CATEGORY'] == category, 'Water Body CALWNUMS'])
        impaired_set_2024 = set(impaired_303d[2024].loc[impaired_303d[2024]['SUB_CATEGORY'] == category, 'Water Body CALWNUMS'])
        newly_impaired_water_bodies[category] = impaired_set_2024 - impaired_set_2018
        impaired_water_bodies[category] = impaired_set_2024
        logger.debug(f"Category {category}: {len(newly_impaired_water_bodies[category])} newly impaired water bodies")
    
    # Create all column names first
    all_categories = parent_categories + sub_categories
    column_names = []
    for category in all_categories:
        column_names.extend([
            f'Discharges to Newly {category} Impaired',
            f'Discharges to {category} Impaired',
            f'Discharges to Newly {category} Impaired and Not Limited'
        ])
    column_names.extend([
        'Discharges to Impaired Water Bodies and Not Limited',
        'Discharges to Impaired and Not Limited: Number of Parameters'
    ])
    
    # Pre-allocate all columns with zeros/empty strings
    new_data = pd.DataFrame(0, index=facilities_list.index, columns=column_names)
    new_data['Discharges to Impaired Water Bodies and Not Limited'] = ''
    
    # Process all categories at once
    for category in all_categories:
        logger.debug(f"Processing category for masks: {category}")
        # Calculate masks for the whole dataset at once
        newly_impaired_mask = facilities_list['CAL WATERSHED NAME'].apply(
            lambda x: any(wb in str(x) for wb in newly_impaired_water_bodies[category]) if pd.notna(x) else False
        )
        impaired_mask = facilities_list['CAL WATERSHED NAME'].apply(
            lambda x: any(wb in str(x) for wb in impaired_water_bodies[category]) if pd.notna(x) else False
        )
        
        logger.debug(f"Category {category} - Newly impaired mask sum: {newly_impaired_mask.sum()}")
        logger.debug(f"Category {category} - Impaired mask sum: {impaired_mask.sum()}")
        
        # Update columns using masks
        new_data[f'Discharges to Newly {category} Impaired'] = newly_impaired_mask.astype(int)
        new_data[f'Discharges to {category} Impaired'] = impaired_mask.astype(int)
        
        # Check limits for facilities with newly impaired waters
        for index in facilities_list[newly_impaired_mask].index:
            sub_limits_2023 = limits_2023[limits_2023['EXTERNAL_PERMIT_NMBR'] == facilities_list.loc[index, 'NPDES # CA#']]
            if not any((sub_limits_2023['SUB_CATEGORY'] == category) & 
                      (sub_limits_2023['LIMIT_VALUE_NMBR'].notna()) & 
                      (sub_limits_2023['LIMIT_VALUE_NMBR'] != '') & 
                      (sub_limits_2023['LIMIT_VALUE_NMBR'] != 'nan')):
                new_data.loc[index, f'Discharges to Newly {category} Impaired and Not Limited'] = 1
    
    # Calculate summary columns
    impaired_categories = []
    for category in sub_categories:
        col_name = f'Discharges to Newly {category} Impaired and Not Limited'
        total = new_data[col_name].sum()
        logger.debug(f"Category {category} - Column {col_name} - Total: {total}")
        logger.debug(f"Type of total: {type(total)}")
        logger.debug(f"Total value: {total}")
        
        # Handle different types of total values
        if isinstance(total, pd.Series):
            logger.debug("Total is a Series, checking size")
            if len(total) == 1:
                total = total.iloc[0]
            else:
                logger.warning(f"Unexpected Series size for category: {len(total)}")
                total = total.sum()
        elif isinstance(total, (np.ndarray, np.generic)):
            logger.debug("Total is a numpy array")
            total = total.item() if total.size == 1 else total.sum()
        
        logger.debug(f"Final total value: {total}, type: {type(total)}")
        
        if total > 0:
            impaired_categories.append(category)
            logger.debug(f"Added {category} to impaired_categories")
    
    # Update summary columns using vectorized operations
    def get_impaired_categories(row):
        categories = []
        for cat in sub_categories:
            col_name = f'Discharges to Newly {cat} Impaired and Not Limited'
            value = row[col_name]
            logger.debug(f"Processing row for {cat} - Value: {value}, Type: {type(value)}")
            if isinstance(value, (pd.Series, np.ndarray)):
                value = value.iloc[0] if isinstance(value, pd.Series) else value.item()
            if value > 0:
                categories.append(cat)
        return ', '.join(categories) if categories else ''
    
    new_data['Discharges to Impaired Water Bodies and Not Limited'] = new_data.apply(get_impaired_categories, axis=1)
    
    # Calculate total parameters per facility
    parameter_cols = [f'Discharges to Newly {category} Impaired and Not Limited' for category in sub_categories]
    logger.debug(f"Parameter columns: {parameter_cols}")
    logger.debug(f"New data columns: {new_data.columns}")
    logger.debug(f"New data shape: {new_data.shape}")
    
    # Ensure all columns are numeric before summing
    for col in parameter_cols:
        if col in new_data.columns:
            logger.debug(f"Converting column {col} to numeric")
            logger.debug(f"Column type before conversion: {new_data[col].dtypes}")
            logger.debug(f"Column values before conversion: {new_data[col].head()}")
            
            # Convert to numeric using astype instead of pd.to_numeric
            try:
                new_data[col] = new_data[col].astype(float)
            except Exception as e:
                logger.error(f"Error converting column {col} to float: {str(e)}")
                logger.error(f"Column values causing error: {new_data[col].head()}")
                # If conversion fails, try to clean the data first
                new_data[col] = new_data[col].replace([np.inf, -np.inf], np.nan)
                new_data[col] = new_data[col].fillna(0)
                new_data[col] = new_data[col].astype(float)
    
    new_data['Discharges to Impaired and Not Limited: Number of Parameters'] = new_data[parameter_cols].sum(axis=1)
    
    # Combine original data with new columns efficiently
    facilities_list = pd.concat([facilities_list, new_data], axis=1)
    
    return facilities_list, newly_impaired_water_bodies

def generate_visualizations(facilities_list, newly_impaired_water_bodies, sub_categories, parent_categories, limits_2023):
    """Generate visualizations of the analysis results."""
    logger.info("Generating visualizations...")
    
    try:
        # Create bar plot
        all_categories = sub_categories + parent_categories
        data = {}
        for category in all_categories:
            if len(newly_impaired_water_bodies[category]) > 0:
                data[category] = {
                    'Discharges to Listed': facilities_list[f'Discharges to {category} Impaired'].sum(),
                    'Newly Listed and Not Yet Limited': facilities_list[f'Discharges to Newly {category} Impaired and Not Limited'].sum()
                }
        
        if not data:
            logger.warning("No data available for visualization")
            return
            
        df = pd.DataFrame(data).T
        df_sorted = df.sort_values(by=df.columns.tolist(), ascending=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_sorted.index))
        width = 0.35
        
        # Use colormaps directly instead of get_cmap
        ax.bar(x - width/2, df_sorted['Discharges to Listed'], width, 
               label='Discharging to Listed\nWater Body', color=plt.colormaps['viridis'](0.2))
        ax.bar(x + width/2, df_sorted['Newly Listed and Not Yet Limited'], width,
               label='Discharging to Newly Listed\nWater Body and\nNot Yet Limited', 
               color=plt.colormaps['viridis'](0.8))
        
        plt.ylabel('Number of Facilities', fontsize=14)
        plt.legend(fontsize=12, frameon=False)
        plt.xticks(x, df_sorted.index, rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(df_sorted['Discharges to Listed']):
            ax.text(i - width/2, v, str(int(v)), ha='center', va='bottom')
        for i, v in enumerate(df_sorted['Newly Listed and Not Yet Limited']):
            ax.text(i + width/2, v, str(int(v)), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('processed_data/step4/facilities_with_future_limits_efficient.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate facility plots
        os.makedirs('processed_data/step4/facility_plots', exist_ok=True)
        generate_facility_plots(facilities_list, newly_impaired_water_bodies, limits_2023, sub_categories)
        
        # Try to create facilities map, fall back to simple plot if shapefile missing
        try:
            num_parameters_per_facility = dict(zip(
                facilities_list['NPDES # CA#'],
                facilities_list['Discharges to Impaired and Not Limited: Number of Parameters']
            ))
            num_parameters_per_facility = {k: v for k, v in num_parameters_per_facility.items() if v >= 1}
            
            plot_facilities_map(num_parameters_per_facility, 
                           '# of Parameters with\nPossible Future Limits', 6)
        except Exception as e:
            logger.warning(f"Could not generate map plot: {str(e)}")
            logger.info("Generating alternative summary plot...")
            
            # Create simple scatter plot of facilities
            facilities_with_coords = facilities_list[['NPDES # CA#', 'LATITUDE DECIMAL DEGREES', 'LONGITUDE DECIMAL DEGREES']].copy()
            facilities_with_coords = facilities_with_coords.rename(columns={
                'NPDES # CA#': 'NPDES_CODE',
                'LATITUDE DECIMAL DEGREES': 'LATITUDE',
                'LONGITUDE DECIMAL DEGREES': 'LONGITUDE'
            })
            
            # Add parameter counts
            facilities_with_coords['Parameters'] = facilities_with_coords['NPDES_CODE'].map(
                lambda x: num_parameters_per_facility.get(x, 0)
            )
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(facilities_with_coords['LONGITUDE'], 
                       facilities_with_coords['LATITUDE'],
                       c=facilities_with_coords['Parameters'],
                       cmap='viridis',
                       alpha=0.6)
            plt.colorbar(label='Number of Parameters')
            plt.title('Facilities by Number of Parameters\nwith Possible Future Limits')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.savefig('processed_data/step4/figures_py/facilities_summary_scatter.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def main(generate_plots=True):
    try:
        # Load and process data with updated return values
        (facilities_list, limits_2023, impaired_303d, parameter_sorting_dict,
         parent_categories, sub_categories) = load_and_process_data()
        
        # Analyze impaired waters with updated parameters
        facilities_list, newly_impaired_water_bodies = analyze_impaired_waters(
            facilities_list, limits_2023, impaired_303d, parameter_sorting_dict,
            parent_categories, sub_categories
        )
        
        # Save results
        facilities_list.to_csv('processed_data/step4/facilities_with_future_limits.csv', index=False)
        
        # Generate visualizations if requested
        if generate_plots:
            generate_visualizations(facilities_list, newly_impaired_water_bodies, 
                                 sub_categories, parent_categories, limits_2023)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in future limits analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main(generate_plots=True) 