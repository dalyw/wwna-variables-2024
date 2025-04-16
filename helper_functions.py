import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import re
import logging
import os
import glob
# LIST OF FUNCTIONS:
# - read_dmr
# - read_all_dmrs
# - read_limits
# - read_esmr
# - categorize_parameters
# - normalize_param_desc
# - match_parameter_desc

### IMPORT DMR AND ESMR DATA
analysis_range = range(2014, 2024)
save = False
load = True

# import list of npdes codes for permits in the filtered full facilities flat file
facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
npdes_from_facilities_list = facilities_list[facilities_list['NPDES # CA#'].notna()]['NPDES # CA#'].unique().tolist()

ref_parameter = pd.read_csv('data/dmrs/REF_PARAMETER.csv')

columns_to_keep_limits = [
        'EXTERNAL_PERMIT_NMBR',
        'LIMIT_SET_ID',
        'PARAMETER_CODE',
        'PARAMETER_DESC',
        'MONITORING_LOCATION_CODE',
        'LIMIT_VALUE_TYPE_CODE',
        'LIMIT_VALUE_NMBR',
        'LIMIT_VALUE_STANDARD_UNITS',
        'LIMIT_UNIT_CODE',
        'STANDARD_UNIT_CODE',
        'STANDARD_UNIT_DESC',
        'STATISTICAL_BASE_CODE',
        'STATISTICAL_BASE_TYPE_CODE',
        'LIMIT_VALUE_QUALIFIER_CODE',
        'LIMIT_FREQ_OF_ANALYSIS_CODE',
        'LIMIT_TYPE_CODE',
        'LIMIT_SET_DESIGNATOR',
        'LIMIT_SET_SCHEDULE_ID',
        'LIMIT_UNIT_DESC',
        'LIMIT_SAMPLE_TYPE_CODE',
    ]

columns_to_keep_dmr = columns_to_keep_limits + [
    'MONITORING_PERIOD_END_DATE',
    'DMR_VALUE_ID',
    'DMR_VALUE_NMBR',
    'DMR_UNIT_CODE',
    'DMR_UNIT_DESC',
    'DMR_VALUE_STANDARD_UNITS',
    'NODI_CODE'
]

columns_to_keep_esmr = [
       'parameter', 'qualifier', 
       'result', 'units', 'mdl', 'ml', 'rl',
       'sampling_date', 'sampling_time', 
       'review_priority_indicator', 
       'qa_codes', 'comments', 'facility_name',
       'facility_place_id', 'report_name',
       'location_desc'
]

def read_dmr(year, drop_no_limit=False):
    """
    Reads the CA DMR data for the given year
    - Keeps only the columns that are needed
    - Drops rows where the limit value is not present (if drop_no_limit is True) and where the No Data Indicator is present
    - Filters for monitoring locations 1, 2, EG, Y, or K
    - Filters for permits in the npdes_list from CIWQS flat file

    Returns the cleaned data
    """
    data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_DMRS.csv', low_memory=False)
    print(f'{year} DMR data has {len(data)} DMR events and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits')
    data = data[columns_to_keep_dmr] 
    if drop_no_limit:
        data = data[data['LIMIT_VALUE_NMBR'].notna()] # drop rows for monitoring without a permit limit
    data = data[data['NODI_CODE'].isna()] # drop rows where No Data Indicator is present
    data = data[data['MONITORING_LOCATION_CODE'].isin(['1', '2', 'EG', 'Y', 'K'])]
    data = data[data['EXTERNAL_PERMIT_NMBR'].isin(npdes_from_facilities_list)]
    # if data['PARAMETER_CODE'] has leading 0s, remove them
    data['PARAMETER_CODE'] = data['PARAMETER_CODE'].str.lstrip('0')
    data['POLLUTANT_CODE'] = data['PARAMETER_CODE'].map(ref_parameter.copy().set_index('PARAMETER_CODE')['POLLUTANT_CODE'])
    data['MONITORING_PERIOD_END_DATE'] = pd.to_datetime(data['MONITORING_PERIOD_END_DATE'])
    data['DMR_VALUE_STANDARD_UNITS'] = pd.to_numeric(data['DMR_VALUE_STANDARD_UNITS'], errors='coerce')
    data['MONITORING_PERIOD_END_DATE_NUMERIC'] = data['MONITORING_PERIOD_END_DATE'].dt.year + data['MONITORING_PERIOD_END_DATE'].dt.month / 12 + data['MONITORING_PERIOD_END_DATE'].dt.day / 365
    print(f'{year} DMR data has {len(data)} DMR events and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits after filtering')
    return data

def read_all_dmrs(save=False, drop_toxicity=False):
    """
    Uses read_dmr to read all the CA DMR data for years in the analysis range
    - Changes the POLLUTANT_DESC on all rows with POLLUTANT_CODE starting with T or W into 'Toxicity'
    - Drops rows where the parameter description contains 'Toxicity' if drop_toxicity is True
    - Saves the data to a pickle
    """
    if save:
        data_dict = {}
        for year in analysis_range:
            data_dict[year] = read_dmr(year, drop_no_limit=True)
            # for data_dict[year], change the POLLUTANT_DESC on all rows with POLLUTANT_CODE starting with T or W into 'Toxicity'
            data_dict[year].loc[data_dict[year]['PARAMETER_CODE'].str.startswith(('T', 'W')), 'PARAMETER_DESC'] = 'Toxicity'
            if drop_toxicity:
                data_dict[year] = data_dict[year][~data_dict[year]['PARAMETER_DESC'].str.contains('Toxicity')]
        # save the data_dict to a pickle
        with open('processed_data/step3/data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else: # load the data_dict from the pickle
        with open('processed_data/step3/data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    return data_dict

def read_limits(year):
    """
    Reads the CA DMR data for the given year
    """
    data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_LIMITS.csv', low_memory=False)
    len_orig = len(data)
    data = data[data['EXTERNAL_PERMIT_NMBR'].isin(npdes_from_facilities_list)]
    print(f'{year} has {len(data)} limits and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits after filtering ({len_orig} limits before filtering)')
    return data[columns_to_keep_limits]


def read_esmr(save=False):
    """
    Read eSMR data with minimal required columns.
    
    Args:
        save (bool): Whether to save processed data to CSV
        
    Returns:
        pd.DataFrame: ESMR data with only required columns
    """
    logger = logging.getLogger(__name__)
    
    # Define only the essential columns we need
    required_columns = [
        'parameter',
        'result',
        'sampling_date',
        'facility_place_id'
    ]
    
    try:
        # Read specific ESMR file
        file_path = 'data/esmr/esmr-analytical-export_years-2006-2024_2024-09-03.csv'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ESMR data file not found: {file_path}")
        
        # Read only required columns
        data = pd.read_csv(file_path, usecols=required_columns)
        
        if save:
            data.to_csv('processed_data/esmr_data.csv', index=False)
            logger.info(f"Saved {len(data)} ESMR records")
            
        return data
        
    except Exception as e:
        logger.error(f"Error reading ESMR data: {str(e)}")
        raise


### CATEGORIZE PARAMETERS
with open('data/manual_updates/parameter_sorting_dict.json', 'r') as f:
    parameter_sorting_dict = json.load(f)

def categorize_parameters(df, parameter_sorting_dict, desc_column):
    """
    Categorize parameters in a dataframe based on a sorting dictionary.
    
    Args:
    df (pd.DataFrame): The dataframe containing parameters to categorize.
    parameter_sorting_dict (dict): Dictionary containing categories and their associated keywords.
    desc_column (str): Name of the column or index containing parameter descriptions.
    
    Returns:
    pd.DataFrame: The input dataframe with additional 'PARENT_CATEGORY' and 'SUB_CATEGORY' columns.
    """
    df['PARENT_CATEGORY'] = 'Uncategorized'
    df['SUB_CATEGORY'] = 'Uncategorized'
    # iterate through the parameter sorting dictionary
    for key, value in parameter_sorting_dict.items():
        if 'values' in value:
            mask = df[desc_column].str.contains('|'.join(map(re.escape, value['values'])), case=value.get('case', False))
            df.loc[mask, 'PARENT_CATEGORY'] = key
            df.loc[mask, 'SUB_CATEGORY'] = key
        else:
            for sub_key, sub_value in value.items():
                mask = df[desc_column].str.contains('|'.join(sub_value['values']), case=False)
                df.loc[mask, 'PARENT_CATEGORY'] = key
                df.loc[mask, 'SUB_CATEGORY'] = sub_key
    return df

def normalize_param_desc(desc):
    """
    Normalize the parameter description by removing commas, brackets, spaces, apostrophes, and dots,
      converting to lowercase, and removing "sum" and "total"
    """
    return desc.replace(", sum", "").replace(", total", "").replace(", tot.", "")\
               .replace(", Sum", "").replace(", Total", "")\
               .replace(',', '').replace('[', '(').replace(']', ')').replace(' ', '')\
               .replace("'", '').replace(".", '').replace("&", 'and')\
               .lower()

def match_parameter_desc(row, target_df, target_desc_column):
    """
    Match the parameter description in the target dataframe to the parameter description in the row.
    """
    normalized_desc = normalize_param_desc(str(row['PARAMETER_DESC']))
    match = target_df[target_df['normalized_desc'] == normalized_desc]
    return match[target_desc_column].iloc[0] if not match.empty else ''