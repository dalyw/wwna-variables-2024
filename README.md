# wwna-variables-2024

This repository includes code to analyze and visualize risk variables associated with California wastewater treatment plants. The analysis is available in both Python (.py) and R (.R) scripts which can be run in sequence to build compliance risk variables for WWTPs. 

The facilities list of interest for the CA Wastewater Needs Assessment is housed under "data/facilities_list"

## Installation

1. Create a conda environment:
```bash
conda create -n wwna-variables-2024 python=3.12 -y
conda activate wwna-variables-2024
```

2. Install the package and dependencies:
```bash
pip install -e .
```

3. For development tools (black, flake8):
```bash
pip install -e ".[dev]"
```

## Getting Started

### 1. Download Data

First, download the large data files (~25GB):
```bash
python wwna_variables_2024/step0_download_data.py
```

This downloads all required data files from public sources to `data/` directory. Large files (>100MB) are excluded from git.

### 2. Run the Analysis

### Python Scripts

Run all analysis steps in sequence:
```bash
python wwna_variables_2024/RUN_ALL.py
```

Or run individual steps:
```bash
python step1_parameter_categorization.py
python step2_population_served.py
python step3_near_exceedance.py
python step4_future_limits.py
```

### R Scripts

Ensure you have the required packages installed, then run:
```r
source("RUN_ALL.R")
```

Or run individual steps:
```r
source("step1_parameter_categorization.R")
source("step2_population_served.R")
source("step3_near_exceedence.R")  # Note: different spelling in R version
source("step4_future_limits.R")
source("step5_update_facilities_list.R")
```

## Analysis Pipeline

The analysis scripts perform the following steps:

### 1. Parameter Categorization
**Files:** `step1_parameter_categorization.py` / `step1_parameter_categorization.R`

Standardizes parameter names from different data sources to create a unified naming convention. This standardization is critical for the subsequent analysis steps that combine multiple data sources.

**Data used:**
- `data/dmr`: EPA ICIS DMR datasets (files too large for GitHub; must be downloaded from ICIS for years 2014-2023)
- `data/ir`: California Integrated Report 303d list of impaired water bodies
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database
(files too large for GitHub;
must be downloaded from https://lab.data.ca.gov/dataset/water-quality-effluent-electronic-self-monitoring-report-esmr-data)

### 2. Population Served
**Files:** `step2_population_served.py` / `step2_population_served.R`

Merges multiple sources for population served into the primary facilities list. Creates a reliable estimate of the population served by each facility by cross-referencing multiple datasets. Outputs population data visualizations to `processed_data/step2/`.

**Data used:**
- `data/cwns`: Clean Watersheds Needs Survey 2022 dataset
- `data/ww_surveillance`: COVID monitoring dataset which also includes facility population served
- `data/sso`: SSO Annual Report ("Questionnaire") data with service population information

### 3. Near Exceedance Analysis
**Files:** `step3_near_exceedance.py` / `step3_near_exceedence.R`

Analyzes historical effluent data to determine which facilities are frequently at or near their permitted limits for various parameters. Calculates the percentage of measurements that exceed specific thresholds of the permitted limits. Generates visualizations stored in `processed_data/step3/`.

**Data used:**
- `data/dmr`: EPA ICIS DMR datasets
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database

### 4. Future Limits Analysis (Potential Permit Tightening)
**Files:** `step4_future_limits.py` / `step4_future_limits.R`

Identifies facilities that may face stricter regulatory requirements based on newly-listed impaired water bodies. This analysis flags facilities for potential future permit tightening when:

1. **Watershed was newly listed**: The facility's discharge location (CAL WATERSHED NAME) was added to the California Integrated Report 303(d) list between 2018 and 2024 for a specific pollutant category
2. **Facility lacks limits**: The facility discharges parameters in that category but does not currently have a permitted limit for those parameters

**Methodology:**
- Compares Integrated Report from 2018 vs 2024 to identify newly impaired water bodies
- For each newly listed water body and pollutant category:
  - Identifies facilities discharging into that watershed
  - Checks if facility monitors parameters in that category
  - Flags facilities that have parameters but no limits for that category
- Excludes specific categories that are unlikely to have wastewater discharge limits (configured in `data/manual_updates/categories_to_exclude_from_future_limits.csv`)

POTWs are typically subject to secondary treatment standards unless:
- They discharge into effluent-dominated water bodies
- They cannot provide 20:1 or more dilution
- Tertiary treatment is needed to protect beneficial uses

When facilities cannot meet seasonal dilution requirements, they become subject to additional limits based on TMDLs applied to the water body. This analysis helps identify which facilities may need additional limits.

**Output:** Creates maps and visualizations in `processed_data/step4/` showing facilities that may need future limits by pollutant category.

**Data used:**
- `data/ir`: California Integrated Report 303(d) lists for 2018 and 2024
- `data/dmr`: NPDES permit limits for 2023 (to check existing limits)
- Parameter categorization from step 1 (to group pollutants by category)

### 5. Generate Updated Facilities List
**Files:** `RUN_ALL.py` (Python) / `step5_update_facilities_list.R` (R only)

Uses outputs from steps 2, 3, and 4 to generate an updated facilities list with risk assessment results. Combines all risk factors into a single dataset for prioritization and decision-making. Produces the final output file with comprehensive risk variables for each facility.

**Note:** In Python, this step is integrated into `RUN_ALL.py`. In R, it's a separate script `step5_update_facilities_list.R`.

## Dependencies

This project uses the outputs of the [us-sewersheds](https://github.com/dalyw/us-sewersheds) package for CWNS data population consolidation.

## Data Loading and Filtering

All data loading is handled through the centralized `file_configs.json` configuration file, which defines:
- Column selection (dtypes)
- Date parsing
- Data filters (dropna, drop_notna, isin)
- Data transformations
- Column renames

### DMR Data Filtering
When loading DMR data, the following filters are applied:
1. **Column selection** - Only loads necessary columns for analysis
2. **Data quality filters**:
   - Removes rows where No Data Indicator (NODI_CODE) is present
   - Removes rows where limit values are missing (LIMIT_VALUE_NMBR is null)
3. **Monitoring location filter** - Keeps only locations: 1, 2, EG, Y, or K
4. **Permit filter** - Includes only permits from the CA Wastewater Needs Assessment facilities list
5. **Parameter transformations**:
   - Strips leading zeros from parameter codes
   - Marks toxicity parameters (codes starting with T or W)

### CWNS Data Filtering
When loading CWNS data:
1. **State filter** - Includes only California facilities (STATE_CODE = "CA")
2. **Column selection** - Loads facility identifiers and population data
3. **Column rename** - Renames TOTAL_RES_POPULATION_2022 to population_cwns

### Step 4: Future Limits Exclusion Configuration
To exclude specific pollutant categories that are unlikely to have wastewater discharge limits, edit `data/manual_updates/categories_to_exclude_from_future_limits.csv`:

e.g.
```csv
SUB_CATEGORY,REASON
Silver,Not typically regulated
Lead,Not typically regulated
...
```

Categories listed in this file are excluded from the future limits analysis in step 4. The file can be edited without modifying code.

## Python Functions Reference

### helper_functions.py
- `load_data(data_type, year=None, file_path=None)` - Generic function to load config-driven data
- `load_data(year, type, drop_toxicity=False)` - Reads DMR/ESMR data for a given year
- `apply_filters(data, config)` - Applies config-defined filters (dropna, drop_notna, isin)
- `apply_transformations(data, config)` - Applies config-defined transformations

### plotting_functions.py
- `setup_fig(figsize=(10, 6))` - Creates and sets up a new figure with common settings
- `save_fig(path)` - Saves figure to path and closes it
- `plot_pie_counts(df, title)` - Plots pie chart of parameter categories
- `plot_map(num_params_per_facility, label_threshold)` - Plots facilities on CA map
- `plot_population_distribution(merged_pop)` - Plots distribution of population served
- `plot_population_source_comparison(merged_pop)` - Plots comparison of population data sources
- `plot_facilities_summary(num_params_per_facility)` - Plots summary without geographic data
- `plot_future_limits_summary(df_sorted)` - Plots summary of facilities with future limits
- `plot_facilities_scatter(facilities_with_coords)` - Plots scatter plot when map unavailable
- `generate_facility_plots(facilities_list, limits_2024)` - Generates detailed plots for each facility
