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

## Running the Analysis

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
- `data/dmrs`: EPA ICIS DMR datasets (files too large for GitHub; must be downloaded from ICIS for years 2014-2023)
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
- `data/dmrs`: EPA ICIS DMR datasets
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database

### 4. Future Limits (Proximity to Impaired Waters)
**Files:** `step4_future_limits.py` / `step4_future_limits.R`

Assesses which facilities discharge into newly-listed impaired water bodies but do not yet have a permitted limit for the listed parameters. Identifies facilities that may face stricter regulatory requirements in the future. Creates maps and visualizations in `processed_data/step4/`.

**Data used:**
- `data/ir`: California Integrated Report 303d list

### 5. Generate Updated Facilities List
**Files:** `RUN_ALL.py` (Python) / `step5_update_facilities_list.R` (R only)

Uses outputs from steps 2, 3, and 4 to generate an updated facilities list with risk assessment results. Combines all risk factors into a single dataset for prioritization and decision-making. Produces the final output file with comprehensive risk variables for each facility.

**Note:** In Python, this step is integrated into `RUN_ALL.py`. In R, it's a separate script `step5_update_facilities_list.R`.

## Dependencies

This project uses the outputs of the [us-sewersheds](https://github.com/dalyw/us-sewersheds) package for CWNS data population consolidation.

## Python Functions Reference

### helper_functions.py
- `read_dmr(year, drop_no_limit=False)` - Reads CA DMR data for a given year
- `read_all_dmrs(save=False, drop_toxicity=False)` - Reads all CA DMR data for analysis range
- `read_limits(year)` - Reads CA DMR limits data for a given year
- `read_esmr(save=False)` - Reads eSMR data with minimal required columns
- `categorize_parameters(df, parameter_sorting_dict, desc_column)` - Categorizes parameters based on sorting dictionary
- `normalize_param_desc(desc)` - Normalizes parameter descriptions for matching
- `match_parameter_desc(row, target_df, target_desc_column)` - Matches parameter descriptions across datasets

### plotting_functions.py
- `setup_figure(figsize=(10, 6))` - Creates and sets up a new figure with common settings
- `save_and_close(path, dpi=FIGURE_DPI)` - Saves figure to path and closes it
- `plot_pie_counts(df, title)` - Plots pie chart of parameter categories
- `plot_facilities_map(num_parameters_per_facility, legend_label, label_threshold)` - Plots facilities on CA map
- `plot_population_distribution(merged_pop)` - Plots distribution of population served
- `plot_population_source_comparison(merged_pop)` - Plots comparison of population data sources
- `plot_facilities_summary(num_parameters_per_facility)` - Plots summary without geographic data
- `plot_future_limits_summary(df_sorted)` - Plots summary of facilities with future limits
- `plot_facilities_scatter(facilities_with_coords)` - Plots scatter plot when map unavailable
- `generate_facility_plots(facilities_list, limits_2024)` - Generates detailed plots for each facility
