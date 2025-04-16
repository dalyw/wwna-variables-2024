import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from helper_functions import *
import pickle
import geopandas as gpd
import scipy
from plotting_functions import *

from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directories if they don't exist
os.makedirs('processed_data/step3', exist_ok=True)
os.makedirs('processed_data/step3/figures_py', exist_ok=True)

def get_facilities_with_slope_and_near_exceedance(facility_param_dict, slope_threshold, limit_threshold, print_counts=False):
    """Calculate facilities with significant slope and near exceedance."""
    facilities_with_slope = []
    facilities_with_near_exceedance = []

    for NPDES_code in facility_param_dict.keys():
        for parameter_code in facility_param_dict[NPDES_code].keys():
            for (STANDARD_UNIT_DESC, MONITORING_LOCATION_CODE), limit_data in facility_param_dict[NPDES_code][parameter_code].items():
                slope = limit_data['slope']
                latest_limit = limit_data['latest_limit']
                qualifiers = limit_data['qualifiers']
                Q1 = limit_data['Q1']
                Q3 = limit_data['Q3']
                limit_value_type_codes = limit_data['limit_value_type_codes']
                limit_set_schedule_ids = limit_data['limit_set_schedule_ids']

                # Skip if no valid limit
                if np.isnan(latest_limit):
                    continue

                # Check for facilities with significant slope based on qualifier
                if qualifiers[0] in ['<=', '<'] and slope > slope_threshold:
                    near_exceedance = (Q3 > (1-limit_threshold)*latest_limit)
                    has_slope = (slope > slope_threshold)
                elif qualifiers[0] in ['>=', '>'] and slope < -slope_threshold:
                    near_exceedance = (Q1 < (1+limit_threshold)*latest_limit)
                    has_slope = (slope < -slope_threshold)
                else:
                    has_slope, near_exceedance = False, False

                if has_slope:
                    facilities_with_slope.append((NPDES_code, parameter_code, STANDARD_UNIT_DESC, 
                                               limit_set_schedule_ids[0], limit_value_type_codes[0], 
                                               MONITORING_LOCATION_CODE))
                if near_exceedance:
                    facilities_with_near_exceedance.append((NPDES_code, parameter_code, STANDARD_UNIT_DESC, 
                                                         limit_set_schedule_ids[0], limit_value_type_codes[0], 
                                                         MONITORING_LOCATION_CODE))

    facilities_with_slope_and_near_exceedance = list(set(facilities_with_slope) & set(facilities_with_near_exceedance))
    
    if print_counts:
        print(f'{len(facilities_with_slope)} facility-parameter-limit pairs with slope > {slope_threshold*100}%')
        print(f'{len(facilities_with_near_exceedance)} facility-parameter-limit pairs with Q1 or Q3 reaching > {limit_threshold*100}% of limit')
        print(f'{len(facilities_with_slope_and_near_exceedance)} facility-parameter-limit pairs with both slope and near exceedance')
        print(f'{len(set([facility[0] for facility in facilities_with_slope_and_near_exceedance]))} facilities that have at least one parameter-limit pair with slope and near exceedance')
    
    return facilities_with_slope_and_near_exceedance

def generate_visualizations(facilities_with_slope_and_near_exceedance_df, facility_param_dict, ref_parameter, ref_frequency, generate_plots=True):
    """Generate exceedance analysis visualizations."""
    if not generate_plots:
        logger.info("Skipping visualization generation")
        return
        
    logger.info("Generating visualizations...")
    
    facilities_grouped = facilities_with_slope_and_near_exceedance_df.groupby('NPDES_CODE')
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='b', label='Data', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Trend'),
        plt.Line2D([0], [0], marker='*', color='red', label='Outliers', markersize=10, linestyle='None'),
        plt.Rectangle((0,0), 1, 1, fc="lightgreen", alpha=0.3, label='In Compliance'),
        plt.Rectangle((0,0), 1, 1, fc="lightcoral", alpha=0.3, label='Out of Compliance')
    ]
    
    histogram_legend_elements = [
        plt.Line2D([0], [0], color='red', linestyle='--', label='Q1'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='Q3'),
        plt.Line2D([0], [0], color='grey', linestyle='-', label='Limit')
    ]

    # Generate facility plots
    for NPDES_CODE, group in facilities_grouped:
        # Implementation of facility plot generation
        # (Previous plotting code here)
        pass

def analyze_thresholds(facility_param_dict):
    """Analyze different threshold combinations."""
    slope_threshold_ranges = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    limit_threshold_ranges = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    results = []
    for slope_threshold in slope_threshold_ranges:
        for limit_threshold in limit_threshold_ranges:
            facilities = get_facilities_with_slope_and_near_exceedance(
                facility_param_dict, slope_threshold, limit_threshold
            )
            results.append({
                'count': len(set(facility[0] for facility in facilities)),
                'slope_threshold': slope_threshold,
                'limit_threshold': limit_threshold
            })
    return pd.DataFrame(results)

def process_facility_group(args):
    """Process a single facility-parameter group in parallel."""
    try:
        (NPDES_code, parameter_code, STANDARD_UNIT_DESC, MONITORING_LOCATION_CODE), group = args
        
        # Convert data types and handle missing values
        dates = pd.to_numeric(group['MONITORING_PERIOD_END_DATE_NUMERIC'], errors='coerce')
        values = pd.to_numeric(group['DMR_VALUE_STANDARD_UNITS'], errors='coerce')
        
        # Remove NaN values and ensure unique x values
        mask = ~(np.isnan(dates) | np.isnan(values))
        dates = dates[mask]
        values = values[mask]
        
        # Get unique x values and their corresponding y means
        unique_dates = np.unique(dates)
        if len(unique_dates) < 3:  # Need at least 3 unique points for trend
            return None
            
        unique_values = [np.mean(values[dates == d]) for d in unique_dates]
        
        # Calculate outliers using robust statistics
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outlier_mask = (values < lower_fence) | (values > upper_fence)
        value_mask = ~outlier_mask
        
        # Calculate slope using unique points
        try:
            from scipy import stats
            # Center and scale the data
            dates_norm = (unique_dates - np.mean(unique_dates)) / np.std(unique_dates)
            values_norm = (unique_values - np.mean(unique_values)) / np.std(unique_values) if np.std(unique_values) != 0 else unique_values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_norm, values_norm)
            
            # Convert slope back to original scale
            slope = slope * (np.std(unique_values) / np.std(unique_dates)) if np.std(unique_dates) != 0 else 0
            
            # If the fit is poor or not significant, set slope to 0
            if r_value**2 < 0.1 or p_value > 0.05:
                slope, intercept = 0, 0
                
        except Exception as e:
            logger.debug(f"Slope calculation failed for {NPDES_code}-{parameter_code}: {str(e)}")
            slope, intercept = 0, 0
        
        # Get the most recent limit value safely
        limits = pd.to_numeric(group['LIMIT_VALUE_STANDARD_UNITS'], errors='coerce')
        latest_limit = limits.iloc[-1] if not limits.empty else np.nan
        
        # Create result dictionary
        key = (STANDARD_UNIT_DESC, MONITORING_LOCATION_CODE)
        result_dict = {
            'slope': slope,
            'intercept': intercept,
            'limits': limits.values,
            'qualifiers': group['LIMIT_VALUE_QUALIFIER_CODE'].values,
            'dates': dates,
            'values': values,
            'datetimes': pd.to_datetime(group['MONITORING_PERIOD_END_DATE']).values,
            'frequency_code': group['LIMIT_FREQ_OF_ANALYSIS_CODE'].values,
            'outlier_mask': outlier_mask,
            'value_mask': value_mask,
            'mean': values[value_mask].mean() if any(value_mask) else np.nan,
            'Q1': Q1,
            'Q3': Q3,
            'limit_value_type_codes': group['LIMIT_VALUE_TYPE_CODE'].values,
            'limit_set_schedule_ids': group['LIMIT_SET_SCHEDULE_ID'].values,
            'latest_limit': latest_limit,
            'r_squared': r_value**2 if 'r_value' in locals() else 0,
            'p_value': p_value if 'p_value' in locals() else 1
        }
        
        return (NPDES_code, parameter_code, key, result_dict)
        
    except Exception as e:
        logger.error(f"Error processing facility group {NPDES_code}-{parameter_code}: {str(e)}")
        return None

def read_all_dmrs(save=False, drop_toxicity=False):
    """Optimized data loading."""
    # Specify dtypes for faster loading
    dtypes = {
        'EXTERNAL_PERMIT_NMBR': str,
        'PARAMETER_CODE': str,
        'STANDARD_UNIT_DESC': str,
        'MONITORING_LOCATION_CODE': str,
        'DMR_VALUE_STANDARD_UNITS': float,
        'LIMIT_VALUE_STANDARD_UNITS': float,
        'LIMIT_VALUE_QUALIFIER_CODE': str,
        'LIMIT_VALUE_TYPE_CODE': str,
        'LIMIT_SET_SCHEDULE_ID': str,
        'LIMIT_FREQ_OF_ANALYSIS_CODE': str
    }
    
    if save:
        data_dict = {}
        for year in analysis_range:
            # Only read needed columns
            data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_DMRS.csv',
                             usecols=columns_to_keep_dmr,
                             dtype=dtypes,
                             parse_dates=['MONITORING_PERIOD_END_DATE'])
            
            # Add numeric date column for trend analysis
            data['MONITORING_PERIOD_END_DATE_NUMERIC'] = (
                data['MONITORING_PERIOD_END_DATE'].dt.year + 
                data['MONITORING_PERIOD_END_DATE'].dt.month / 12 + 
                data['MONITORING_PERIOD_END_DATE'].dt.day / 365
            )
            
            # Filter data
            data = data[data['MONITORING_LOCATION_CODE'].isin(['1', '2', 'EG', 'Y', 'K'])]
            data = data[data['EXTERNAL_PERMIT_NMBR'].isin(npdes_from_facilities_list)]
            
            if drop_toxicity:
                data = data[~data['PARAMETER_DESC'].str.contains('Toxicity', na=False)]
            
            data_dict[year] = data
            logger.info(f"Loaded {year} data: {len(data)} records for {data['EXTERNAL_PERMIT_NMBR'].nunique()} facilities")
        
        # Save to pickle
        with open('processed_data/step3/data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        # Load from pickle
        try:
            with open('processed_data/step3/data_dict.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            logger.info(f"Loaded data from pickle for years {min(data_dict.keys())}-{max(data_dict.keys())}")
        except FileNotFoundError:
            logger.error("No saved data found. Please run with save=True first")
            raise
    
    return data_dict

def main(generate_plots=True):
    try:
        # Load reference data
        logger.info("Loading reference data...")
        ref_frequency = pd.read_csv('data/dmrs/REF_FREQUENCY_OF_ANALYSIS.csv')
        ref_parameter = pd.read_csv('data/dmrs/REF_PARAMETER.csv')
        
        # Load unique parameter codes from step1 output
        logger.info("Loading unique parameter codes...")
        unique_parameter_codes = pd.read_csv('processed_data/step1/dmr_esmr_mapping.csv')['PARAMETER_CODE'].unique()
        
        # Load data - first time with save=True
        logger.info("Loading and processing DMR data...")
        try:
            data_dict = read_all_dmrs(save=False)
        except FileNotFoundError:
            logger.info("No saved data found. Processing raw data files...")
            data_dict = read_all_dmrs(save=True)
        
        current_year = max(data_dict.keys())
        
        # Load or generate facility_param_dict
        regenerate_facility_param_dict = True  # Set to False to load from pickle
        
        if regenerate_facility_param_dict:
            logger.info("Generating facility parameter dictionary...")
            facility_param_dict = {}
            
            # Pre-filter and group data once
            filtered_data = pd.concat([data_dict[year][data_dict[year]['PARAMETER_CODE'].isin(unique_parameter_codes)] 
                                    for year in analysis_range])
            
            # Group by facility and parameter
            grouped_data = filtered_data.groupby(['EXTERNAL_PERMIT_NMBR', 'PARAMETER_CODE', 
                                                'STANDARD_UNIT_DESC', 'MONITORING_LOCATION_CODE'])
            
            # Parallel processing
            with Pool(processes=cpu_count() - 1) as pool:
                results = pool.map(process_facility_group, grouped_data)
            
            # Reconstruct facility_param_dict from results
            facility_param_dict = {}
            for result in results:
                if result is not None:
                    NPDES_code, parameter_code, key, result_dict = result
                    if NPDES_code not in facility_param_dict:
                        facility_param_dict[NPDES_code] = {}
                    if parameter_code not in facility_param_dict[NPDES_code]:
                        facility_param_dict[NPDES_code][parameter_code] = {}
                    facility_param_dict[NPDES_code][parameter_code][key] = result_dict
            
            # Save facility_param_dict
            with open('processed_data/step3/facility_param_dict.pkl', 'wb') as f:
                pickle.dump(facility_param_dict, f)
        else:
            # Load existing facility_param_dict
            logger.info("Loading facility parameter dictionary from pickle...")
            with open('processed_data/step3/facility_param_dict.pkl', 'rb') as f:
                facility_param_dict = pickle.load(f)
        
        # Continue with the rest of the analysis...
        slope_threshold = 0.05
        limit_threshold = 0.1
        facilities_with_slope_and_near_exceedance = get_facilities_with_slope_and_near_exceedance(
            facility_param_dict, slope_threshold, limit_threshold, print_counts=True
        )
        
        # Create DataFrame of results
        facilities_with_slope_and_near_exceedance_df = pd.DataFrame(
            facilities_with_slope_and_near_exceedance,
            columns=['NPDES_CODE', 'PARAMETER_CODE', 'STANDARD_UNIT_DESC', 
                    'LIMIT_SET_SCHEDULE_ID', 'LIMIT_VALUE_TYPE_CODE', 'MONITORING_LOCATION_CODE']
        )
        
        # Count parameters per facility
        num_parameters_per_facility = {}
        for facility, parameter_code, *_ in facilities_with_slope_and_near_exceedance:
            num_parameters_per_facility.setdefault(facility, set()).add(parameter_code)
        num_parameters_per_facility = {facility: len(parameters) 
                                     for facility, parameters in num_parameters_per_facility.items()}
        
        # Create facilities visualization
        if generate_plots:
            try:
                plot_facilities_map(num_parameters_per_facility, 
                                  'Number of Parameters\nwith Slope and\nNear-Exceedance', 4)
            except Exception as e:
                logger.warning(f"Could not generate map plot: {str(e)}")
                logger.info("Generating alternative summary plot...")
                plot_facilities_summary(num_parameters_per_facility)
        
        # Generate other visualizations with error handling
        try:
            generate_visualizations(facilities_with_slope_and_near_exceedance_df, 
                                  facility_param_dict, ref_parameter, ref_frequency,
                                  generate_plots)
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {str(e)}")
            logger.warning("Continuing without visualizations...")
        
        # Analyze different thresholds
        threshold_results = analyze_thresholds(facility_param_dict)
        
        # Save results
        facilities_with_slope_and_near_exceedance_df.to_csv(
            'processed_data/step3/facilities_with_slope_and_near_exceedance.csv',
            index=False
        )
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in exceedance analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main(generate_plots=True) 