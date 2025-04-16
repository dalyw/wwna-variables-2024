import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from plotting_functions import *
from us_sewersheds import (
    load_cwns_data,
    process_facility_types,
    build_sewershed_map
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs('processed_data/step2', exist_ok=True)

def load_and_process_cwns_data():
    """Load and process CWNS data for California facilities"""
    logger.info("Loading CWNS data...")
    cwns_data = load_cwns_data(data_dir = 'data/cwns/CA_2022CWNS_APR2024/')
    
    # Filter for California facilities
    facilities = cwns_data['facilities']
    facilities = facilities[facilities['STATE_CODE'] == 'CA']
    
    # Merge with other CWNS datasets
    merge_columns = {
        'permits': ['CWNS_ID', 'PERMIT_NUMBER'],
        'counties': ['CWNS_ID', 'COUNTY_NAME'],
        'types': ['CWNS_ID', 'FACILITY_TYPE'],
        'flow': ['CWNS_ID', 'CURRENT_DESIGN_FLOW'],
        'population': ['CWNS_ID', 'TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2042']
    }
    
    for df_name, columns in merge_columns.items():
        logger.info(f"Merging {df_name} data...")
        df = cwns_data[df_name][columns]
        facilities = facilities.merge(df, on='CWNS_ID', how='left')
    
    return facilities

def load_covid_monitoring_data():
    """Load COVID monitoring dataset with population information"""
    logger.info("Loading COVID monitoring data...")
    covid_data = pd.read_csv('data/ww_surveillance/wastewatersurveillancecalifornia.csv')
    # print(covid_data[['epaid', 'site_id', 'epa_registry_id']])
    # print(covid_data.columns)
    return covid_data

def load_sso_data():
    """Load SSO Annual Report data with service population"""
    logger.info("Loading SSO data...")
    sso_data = pd.read_csv('data/sso/Questionnaire.txt', sep='\t')
    return sso_data

def merge_population_data(facilities_df, covid_data, sso_data):
    """
    Merge population data from multiple sources and analyze discrepancies
    """
    logger.info("Merging population data from multiple sources...")
    
    # Create merged dataset with population from all sources
    merged_pop = facilities_df[['CWNS_ID', 'PERMIT_NUMBER', 'FACILITY_NAME', 'TOTAL_RES_POPULATION_2022']].copy()
    merged_pop = merged_pop.rename(columns={'TOTAL_RES_POPULATION_2022': 'population_cwns'})
    
    # Merge COVID monitoring population data
    logger.info("Processing COVID monitoring population data...")
    if 'population_served' in covid_data.columns:
        # Check if epaid column contains lists instead of strings
        if covid_data['epaid'].apply(lambda x: isinstance(x, list)).any():
            logger.info("Found list values in epaid column, exploding to separate rows")
            # Explode the epaid column if it contains lists
            covid_data = covid_data.explode('epaid')
        
        # Ensure epaid is string type for merging
        covid_data['epaid'] = covid_data['epaid'].astype(str)
        merged_pop['PERMIT_NUMBER'] = merged_pop['PERMIT_NUMBER'].astype(str)
        
        # Create copy to avoid SettingWithCopyWarning
        covid_pop = covid_data[['epaid', 'population_served']].copy()
        covid_pop = covid_pop.rename(columns={'population_served': 'population_covid'})
        
        # Drop duplicates if any exist after exploding
        covid_pop = covid_pop.drop_duplicates(subset=['epaid'])
        
        # Merge with facilities data
        merged_pop = merged_pop.merge(covid_pop, left_on='PERMIT_NUMBER', right_on='epaid', how='left')
        logger.info(f"Merged COVID population data: {len(merged_pop)} rows")

    # Merge SSO questionnaire population data
    if 'service_population' in sso_data.columns:
        sso_pop = sso_data[['permit_number', 'service_population']].copy()
        sso_pop = sso_pop.rename(columns={'service_population': 'population_sso'})
        merged_pop = merged_pop.merge(sso_pop, left_on='PERMIT_NUMBER', right_on='permit_number', how='left')
    
    # Calculate statistics and identify discrepancies
    pop_columns = [col for col in merged_pop.columns if 'population' in col]
    merged_pop['population_mean'] = merged_pop[pop_columns].mean(axis=1)
    merged_pop['population_std'] = merged_pop[pop_columns].std(axis=1)
    merged_pop['population_cv'] = merged_pop['population_std'] / merged_pop['population_mean']
    
    # Save merged population data
    merged_pop.to_csv('processed_data/step2/merged_population_data.csv', index=False)
    
    return merged_pop

def generate_visualizations(merged_pop):
    """Generate population data visualizations"""
    logger.info("Generating population visualizations...")
        
    # Create population distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(merged_pop['population_mean'].dropna(), bins=50)
    plt.xlabel('Population Served')
    plt.ylabel('Number of Facilities')
    plt.title('Distribution of Population Served by CA Wastewater Facilities')
    plt.savefig('processed_data/step2/population_distribution.png')
    plt.close()
    
    # Create population source comparison plot
    pop_sources = [col for col in merged_pop.columns if 'population_' in col and col not in ['population_mean', 'population_std', 'population_cv']]
    plt.figure(figsize=(10, 6))
    merged_pop[pop_sources].boxplot()
    plt.yscale('log')
    plt.ylabel('Population (log scale)')
    plt.title('Population Estimates by Data Source')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processed_data/step2/population_source_comparison.png')
    plt.close()

def main():
    try:
        # Load and process all data sources
        facilities_df = load_and_process_cwns_data()
        covid_data = load_covid_monitoring_data()
        sso_data = load_sso_data()
        
        # Merge population data from all sources
        merged_pop = merge_population_data(facilities_df, covid_data, sso_data)
        
        # Generate visualizations
        generate_visualizations(merged_pop)
        
        logger.info("Population analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in population analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 