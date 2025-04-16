# wwna-variables-2024

This repository includes code to analyze and visualize risk variables associated with California wastewater treatment plants. The top-level folder includes five Jupyter notebook files which should be run in sequence to build a comprehensive risk assessment for wastewater facilities.

The facilities list of interest for the CA Wastewater Needs Assessment is housed under "data/facilities_list"

## Analysis Pipeline

The .ipynb scripts perform the following analyses:

### 1. Parameter Categorization (`1_parameter_categorization.ipynb`)
    Standardizes parameter names from different data sources to create a unified naming convention.
    This standardization is critical for the subsequent analysis steps that combine multiple data sources.

    Data used:
     -- data/dmrs: EPA ICIS DMR datasets. (Files are too large for github: Must be downloaded from ICIS for years 2014-2023 and added to this folder)
     -- data/ir: California Integrated Report 303d list of impaired water bodies
     -- data/esmr: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database. (Files are too large for github: Must be downloaded from CIWQS and added to this folder)

### 2. Population Served (`2_population_served.ipynb`)
    Merges multiple sources for population served into the primary facilities list.
    Creates a reliable estimate of the population served by each facility by cross-referencing multiple datasets.
    Outputs population data visualizations to processed_data/step2/.

    Data used:
     -- data/cwns: Clean Watersheds Needs Survey 2022 dataset
     -- data/ww_surveillance: COVID monitoring dataset which also includes facility population served
     -- data/sso: SSO Annual Report ("Questionnaire") data with service population information

### 3. Near Exceedance Analysis (`3_near_exceedance.ipynb`)
    Analyzes historical effluent data to determine which facilities are frequently at or near their permitted limits for various parameters.
    Calculates the percentage of measurements that exceed specific thresholds of the permitted limits.
    Generates visualizations stored in processed_data/step3/.

    Data used:
     -- data/dmrs: EPA ICIS DMR datasets
     -- data/esmr: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database

### 4. Future Limits (Proximity to Impaired Waters) (`4_future_limits.ipynb`)
    Assesses which facilities discharge into newly-listed impaired water bodies but do not yet have a permitted limit for the listed parameters.
    Identifies facilities that may face stricter regulatory requirements in the future.
    Creates maps and visualizations in processed_data/step4/.

    Data used:
     -- data/ir: California Integrated Report 303d list

### 5. Generate Updated Facilities List (`5_generate_updated_facilities_list.ipynb`)
    Uses outputs from steps 2, 3, and 4 to generate an updated facilities list with risk assessment results.
    Combines all risk factors into a single dataset for prioritization and decision-making.
    Produces the final output file with comprehensive risk variables for each facility.
