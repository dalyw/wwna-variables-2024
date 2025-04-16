import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from helper_functions import *
from plotting_functions import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs("processed_data/step1", exist_ok=True)

def load_dmr_data():
    """Load DMR datasets from 2014-2023."""
    logger.info("Loading DMR data...")
    
    dmr_files = list(Path("data/dmrs").glob("*.csv"))
    if not dmr_files:
        raise FileNotFoundError("No DMR data files found in data/dmrs directory")
    
    dfs = []
    for file in dmr_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    dmr_data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(dmr_data)} DMR records")
    return dmr_data

def load_ir_data():
    """Load California Integrated Report 303d list."""
    logger.info("Loading IR data...")
    
    ir_file = Path("data/ir/303d_list.csv")
    if not ir_file.exists():
        raise FileNotFoundError("303d list file not found in data/ir directory")
    
    ir_data = pd.read_csv(ir_file)
    logger.info(f"Loaded {len(ir_data)} IR records")
    return ir_data

def load_esmr_data():
    """Load eSMR data from CIWQS database."""
    logger.info("Loading eSMR data...")
    
    esmr_files = list(Path("data/esmr").glob("*.csv"))
    if not esmr_files:
        raise FileNotFoundError("No eSMR data files found in data/esmr directory")
    
    dfs = []
    for file in esmr_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    esmr_data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(esmr_data)} eSMR records")
    return esmr_data

def create_parameter_mapping():
    """Create standardized parameter name mapping."""
    return {
        # DMR parameter mappings
        'BOD, 5-DAY': 'BOD',
        'BOD, CARBONACEOUS, 5-DAY': 'CBOD',
        'SOLIDS, TOTAL SUSPENDED': 'TSS',
        'NITROGEN, AMMONIA TOTAL': 'NH3-N',
        'COLIFORM, FECAL': 'FC',
        'CHLORINE, TOTAL RESIDUAL': 'CL2',
        
        # eSMR parameter mappings
        'Biochemical Oxygen Demand': 'BOD',
        'Carbonaceous BOD': 'CBOD',
        'Total Suspended Solids': 'TSS',
        'Ammonia as N': 'NH3-N',
        'Fecal Coliform': 'FC',
        'Total Residual Chlorine': 'CL2',
        
        # IR parameter mappings
        'Biochemical Oxygen Demand (BOD)': 'BOD',
        'Suspended Solids': 'TSS',
        'Ammonia': 'NH3-N',
        'Fecal Coliform Bacteria': 'FC',
        'Chlorine': 'CL2'
    }

def standardize_parameters(dmr_data, ir_data, esmr_data):
    """Standardize parameter names across different data sources."""
    logger.info("Standardizing parameter names...")
    
    param_mapping = create_parameter_mapping()
    
    # Create standardized parameter columns
    dmr_standardized = dmr_data.copy()
    ir_standardized = ir_data.copy()
    esmr_standardized = esmr_data.copy()
    
    # Apply mapping to each dataset
    dmr_standardized['parameter_standardized'] = dmr_standardized['parameter_name'].map(param_mapping)
    ir_standardized['parameter_standardized'] = ir_standardized['pollutant'].map(param_mapping)
    esmr_standardized['parameter_standardized'] = esmr_standardized['parameter'].map(param_mapping)
    
    # Create summary of standardization
    summary = {
        'dmr_unique_params': len(dmr_data['parameter_name'].unique()),
        'ir_unique_params': len(ir_data['pollutant'].unique()),
        'esmr_unique_params': len(esmr_data['parameter'].unique()),
        'standardized_params': len(set(param_mapping.values()))
    }
    
    # Save standardization summary
    pd.DataFrame([summary]).to_csv('processed_data/step1/parameter_standardization_summary.csv', index=False)
    
    # Save mapping dictionary
    pd.DataFrame(list(param_mapping.items()), 
                columns=['original_name', 'standardized_name']
                ).to_csv('processed_data/step1/parameter_mapping.csv', index=False)
    
    return dmr_standardized, ir_standardized, esmr_standardized

def save_standardized_data(dmr_data, ir_data, esmr_data):
    """Save standardized datasets."""
    output_dir = Path("processed_data/step1")
    
    dmr_data.to_csv(output_dir / "dmr_standardized.csv", index=False)
    ir_data.to_csv(output_dir / "ir_standardized.csv", index=False)
    esmr_data.to_csv(output_dir / "esmr_standardized.csv", index=False)
    
    logger.info("Saved standardized datasets to processed_data/step1/")

def main():
    # Import DMR Parameter Data
    dmrs_2023 = read_dmr(2023, drop_no_limit=True)
    unique_parameter_codes_dmrs = dmrs_2023['PARAMETER_CODE'].unique()
    dmr_parameter_df = pd.read_csv('data/dmrs/REF_PARAMETER.csv')
    dmr_parameter_df = dmr_parameter_df[dmr_parameter_df['PARAMETER_CODE'].isin(unique_parameter_codes_dmrs)]
    print(f'{len(dmr_parameter_df)} parameters and {len(dmr_parameter_df["POLLUTANT_CODE"].unique())} unique pollutants in DMR limit and monitoring datasets')

    # Import ESMR Data
    esmr_data = read_esmr(save=False)
    esmr_parameter_df = pd.DataFrame({'ESMR_PARAMETER_DESC': esmr_data['parameter'].unique()})

    # Import IR Data
    impaired_303d_2024 = pd.read_csv('data/ir/2024-303d.csv', skiprows=1)
    ir_parameter_df = impaired_303d_2024[['Pollutant']].drop_duplicates().reset_index(drop=True)
    ir_parameter_df.rename(columns={'Pollutant': 'IR_PARAMETER_DESC'}, inplace=True)

    # Import CA toxics rule data
    toxics_df = pd.read_csv('data/toxics/criteria_for_toxics.csv')

    # Import CA toxics rule data
    toxics_parameter_df = pd.read_csv('data/toxics/criteria_for_toxics.csv')
    toxics_parameter_df = toxics_parameter_df.rename(columns={'Number compound': 'TOXICS_PARAMETER_DESC'})
    toxics_parameter_df['TOXICS_PARAMETER_DESC'] = toxics_parameter_df['TOXICS_PARAMETER_DESC'].str.replace(r'^\d+\.\s*', '', regex=True)

    # Categorize parameters
    dmr_parameter_df = categorize_parameters(dmr_parameter_df, parameter_sorting_dict, 'PARAMETER_DESC')
    ir_parameter_df = categorize_parameters(ir_parameter_df, parameter_sorting_dict, 'IR_PARAMETER_DESC')
    esmr_parameter_df = categorize_parameters(esmr_parameter_df, parameter_sorting_dict, 'ESMR_PARAMETER_DESC')
    toxics_parameter_df = categorize_parameters(toxics_parameter_df, parameter_sorting_dict, 'TOXICS_PARAMETER_DESC')

    # additional categorization of Total Toxics
    mask = dmr_parameter_df['PARAMETER_CODE'].str.startswith(('T', 'W'))
    dmr_parameter_df.loc[mask, 'PARENT_CATEGORY'] = 'Total Toxics'
    dmr_parameter_df.loc[mask, 'SUB_CATEGORY'] = ''

    # save ir_parameter_df
    ir_parameter_df.to_csv('processed_data/step1/ir_parameter_df.csv', index=False)

    # Plot category distributions
    plot_pie_counts(dmr_parameter_df, 'REF_Parameter Categories')
    plot_pie_counts(ir_parameter_df, 'ir_parameter_df Categories')
    plot_pie_counts(esmr_parameter_df, 'esmr_parameter_df Categories')
    plot_pie_counts(toxics_parameter_df, 'toxics_parameter_df Categories')

    # Parameter name matching
    esmr_parameter_df['normalized_desc'] = esmr_parameter_df['ESMR_PARAMETER_DESC'].apply(normalize_param_desc)
    toxics_parameter_df['normalized_desc'] = toxics_parameter_df['TOXICS_PARAMETER_DESC'].apply(normalize_param_desc)
    dmr_parameter_df['ESMR_PARAMETER_DESC_MATCHED'] = dmr_parameter_df.apply(lambda row: match_parameter_desc(row, esmr_parameter_df, 'ESMR_PARAMETER_DESC'), axis=1)
    dmr_parameter_df['TOXICS_PARAMETER_DESC'] = dmr_parameter_df.apply(lambda row: match_parameter_desc(row, toxics_parameter_df, 'TOXICS_PARAMETER_DESC'), axis=1)
    print(f'{len(dmr_parameter_df["ESMR_PARAMETER_DESC_MATCHED"].unique()) - 1} out of {len(dmr_parameter_df)} parameter names automatically matched to ESMR PARAMETER_DESC')
    print(f'{len(dmr_parameter_df["TOXICS_PARAMETER_DESC"].unique()) - 1} out of {len(dmr_parameter_df)} parameter names automatically matched to TOXICS_PARAMETER_DESC')

    # Add manual mappings
    dmr_esmr_mapping_manual = pd.read_csv('data/manual_updates/dmr_esmr_mapping_manual.csv')
    manual_mapping = dict(zip(dmr_esmr_mapping_manual['PARAMETER_CODE'], dmr_esmr_mapping_manual['ESMR_PARAMETER_DESC_MANUAL']))
    dmr_parameter_df['ESMR_PARAMETER_DESC_MANUAL'] = dmr_parameter_df['PARAMETER_CODE'].map(manual_mapping).fillna('')

    # Combine automatic and manual mappings
    dmr_parameter_df['ESMR_PARAMETER_DESC'] = dmr_parameter_df.apply(
        lambda row: row['ESMR_PARAMETER_DESC_MATCHED'] if row['ESMR_PARAMETER_DESC_MATCHED'] != '' 
        else (row['ESMR_PARAMETER_DESC_MANUAL'] if row['ESMR_PARAMETER_DESC_MANUAL'] != '' else 'No Match (unconfirmed)'),
        axis=1)

    # Final cleanup and save
    dmr_parameter_df.drop(columns=['ESMR_PARAMETER_DESC_MATCHED', 'ESMR_PARAMETER_DESC_MANUAL'], inplace=True)
    dmr_parameter_df.rename(columns={'PARAMETER_DESC': 'DMR_PARAMETER_DESC'}, inplace=True)
    dmr_parameter_df.to_csv('processed_data/step1/dmr_esmr_mapping.csv', index=False)

if __name__ == "__main__":
    main() 