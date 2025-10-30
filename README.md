# wwna-variables-2024

This repository includes code to analyze and visualize risk variables associated with California wastewater treatment plants. The analysis is available in both Python (.py) and R (.R) scripts which can be run in sequence to build compliance risk variables for WWTPs.

The facilities list of interest for the CA Wastewater Needs Assessment is housed under `data/wwna_list/`.

## Analysis Pipeline

The analysis consists of five steps that aggregate data sources to inform 3 risk variables for wastewater treatment facilities. Each step produces intermediate outputs that feed into the final consolidated facilities list.

### Step 0: Data Download
**Files:** `step0_download_data.py` / `step0_download_data.R`

Downloads required data files from public sources. Large files are not stored in git and must be downloaded before running the analysis.
URLs used for downloading data are stored in `file_configs.json`.

### Step 1: Parameter Categorization
**Files:** `step1_parameter_categorization.py` / `step1_parameter_categorization.R`

Standardizes parameter names from different data sources to create a unified naming convention. This standardization is used for the subsequent analysis steps that combine multiple data sources.

**Inputs:**
- `data/dmr`: EPA ICIS DMR datasets (files too large for GitHub; must be downloaded from ICIS for years 2014-2023)
- `data/ir`: California Integrated Report 303d list of impaired water bodies
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS
- `data/manual_updates/parameters_manual_additions.csv`: Manual parameter additions
- `data/manual_updates/parameter_sorting_dict.json`: Parameter categorization dictionary

**Outputs:**
- `processed_data/step1/ref_parameter_merged_py.csv` / `ref_parameter_merged_R.csv`: Merged parameter reference with categories
- `processed_data/step1/dmr_esmr_mapping_py.csv` / `dmr_esmr_mapping_R.csv`: Mapping between DMR and eSMR parameter codes

### Step 2: Population Served Risk Variable
**Files:** `step2_population_served.py` / `step2_population_served.R`

This step merged multiple sources for population served to get the population served by each WWTP. The manual permit number matching is used to address facilities whose ORDER NO or Permit Number changed after the 2022 CWNS, which has most of the population data. Prioritizes PERMIT_NUMBER values that have manual matches when a CWNS_ID has multiple associated PERMIT_NUMBERs.
The output figure shows facilities by data source availability (CWNS only, COVID only, SSO only, combinations, or Unmatched)

**Inputs:**
- `data/cwns`: Clean Watersheds Needs Survey 2022 dataset
- `data/ww_surveillance`: COVID monitoring dataset which also includes facility population served
- `data/sso`: SSO Annual Report ("Questionnaire") data with service population information
- `data/manual_updates/cwns_facilities_match_manual.csv`: Manual mappings for CWNS facilities

When loading CWNS data, we filter to include only California facilities (STATE_CODE = "CA").

**Outputs:**
- `processed_data/step2/merged_population_data_py.csv` / `merged_population_data_R.csv`: Consolidated population data with source classification

### Step 3: Near Exceedance Analysis
**Files:** `step3_near_exceedance.py` / `step3_near_exceedance.R`

Analyzes historical effluent data to determine which facilities are frequently at or near their permitted limits for various parameters. Calculates the percentage of measurements that exceed specific thresholds of the permitted limits using statistical analysis (slope calculation and quartile analysis).

**Inputs:**
- `data/dmr`: EPA ICIS DMR datasets
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database
- Parameter categorization from step 1
- `data/wwna_list/NPDES+WDR Facilities List_20240906.csv`: WWNA facilities list

**Outputs:**
- `processed_data/step3/flagged_facilities_step3_py.csv` / `flagged_facilities_step3_R.csv`: Facilities flagged for near exceedance
- `processed_data/step3/figures_py/`: Visualizations showing facilities with exceedances

**Methodology:** 
We analyze each facility–parameter by grouping on permit, parameter code, monitoring location and units, and we always compare against the most recent non‑null permit limit and its qualifier ("≤", "<", ">=", ">"). 

We remove outliers with a two‑sided IQR rule, then fit a simple linear trend and save the slope and intercept for plotting. 
  - IQR = Q3 - Q1 (difference between 75th and 25th percentiles)
  - Outlier threshold = Q3 + (iqr_multiplier × IQR), where `iqr_multiplier` defaults to 2.5 in 
  `analysis_config.json` (less strict than standard 1.5 × IQR Tukey method)

 Flags are driven by two signals:
 (1) a near‑exceedance check computed from the most recent three [can be varied] years of data (max limits: Q3_recent > (1 − limit_threshold) × limit; min limits: Q1_recent < (1 + limit_threshold) × limit), which keeps the decision focused on current performance. (controlled by `limit_threshold` in `analysis_config.json`)
(2) a time‑to‑limit indicator defined as distance from the series median to the limit divided by |slope|

A facility-parameter is flagged when near‑exceedance is true and the time‑to‑limit is within the configured number of years. Slope is saved for visualization. Because slope on its own is unit-dependent and the threshold value could vary across e.g. pH, concentration, and percent removal parameters, the "time to limit" approach is more unit‑agnostic and interpretable across parameters.

When loading DMR data, the following filters are applied:
1. **Column selection** - Only loads necessary columns for analysis
2. **Data quality filters**:
   - Removes rows where No Data Indicator (NODI_CODE) is present
   - Removes rows where limit values are missing (LIMIT_VALUE_STANDARD_UNITS is null)
3. **Monitoring location filter** - Keeps only locations: 1, 2, EG, Y, or K
4. **Permit filter** - Includes only permits from the CA Wastewater Needs Assessment facilities list
5. **Parameter transformations**:
   - Strips leading zeros from parameter codes
   - Marks toxicity parameters (codes starting with T or W)

### Step 4: Future Limits Analysis (Potential Permit Tightening)
**Files:** `step4_future_limits.py` / `step4_future_limits.R`

Identifies facilities that may face stricter regulatory requirements based on newly-listed impaired water bodies. This analysis flags facilities for potential future permit tightening.

**Inputs:**
- `data/ir`: California Integrated Report 303(d) lists for 2018 and 2024
- `data/dmr`: NPDES permit limits for 2023 (to check existing limits)
- Parameter categorization from step 1 (to group pollutants by category)
- `data/wwna_list/NPDES+WDR Facilities List_20240906.csv`: WWNA facilities list

**Outputs:**
- `processed_data/step4/flagged_facilities_step4_py.csv` / `flagged_facilities_step4_R.csv`: Facilities flagged for potential future limits

**Methodology:**
Step 4 analyzes only pollutant categories that appear in the California DMR dataset against the  Integrated Report (IR) from 2018 vs 2024 to identify newly impaired water bodies. POTWs are typically subject to secondary treatment standards unless:
- They discharge into effluent-dominated water bodies
- They cannot provide 20:1 or more dilution
- Tertiary treatment is needed to protect beneficial uses
When facilities cannot meet seasonal dilution requirements, they become subject to additional limits based on TMDLs applied to the water body. This analysis helps identify which facilities may have additional limits imposed in the coming years.


For each newly listed water body from the IR and pollutant category:
- Identify facilities discharging into that watershed based on CAL WATERSHED NAME
- Check if facility monitors parameters in that category. (Parameter categories not appearing anywhere in LIMITS data with valid numerical limits are excluded)
- Flag facilities that have parameters but no limits for that category

We filter the IR data to only include pollutants in these regulated categories.
When using the DMR LIMITS file, we filter the data data to only include rows with valid numerical `LIMIT_VALUE_STANDARD_UNITS` values (excludes monitoring-only parameters). We extract unique `SUB_CATEGORY` values from this filtered LIMITS dataset


### RUN_ALL: Generate Updated Facilities List
**Files:** `RUN_ALL.py` (Python) / `step5_update_facilities_list.R` (R only)

Combines outputs from steps 2, 3, and 4 to generate an updated facilities list with risk assessment results added as new columns.

**Inputs:**
- Original WWNA facilities list: `data/wwna_list/NPDES+WDR Facilities List_20240906.csv`
- Population data from step 2
- Near exceedance data from step 3
- Future limits data from step 4

**Outputs:**
- `processed_data/WWNA_LIST_FINAL_updated_py.csv` / `WWNA_LIST_FINAL_updated_R.csv`: Final facilities list with all risk variables


## Running the Analysis

This repository provides **two equivalent workflows** in Python and R. You can use either workflow to produce the same results.

### Set up the environment

#### Python
Create a conda environment:
```bash
conda create -n wwna-variables-2024 python=3.12 -y
conda activate wwna-variables-2024
```
Install the package and dependencies:
```bash
pip install -e .
```

#### R
Install the required packages
```r
install.packages(c("tidyverse", "jsonlite", "readxl", "sf", "raster", "viridis", "gridExtra", "scales", "httr", "lubridate", "parallel"))
```


### Adjust configurations

The analysis uses two main configuration files to control data loading and analysis parameters:

#### `file_configs.json`

All data loading and download configuration is handled through the centralized `wwna_variables_2024/file_configs.json` configuration file, which defines:
- Column selection (dtypes) and rows to skip (skiprows)
- Data filters (dropna, drop_notna, isin) and transformations
- Download URLs for DMR (+LIMITS), ESMR, IR, SSO, TOXICS, CWNS
  - Plus `year_config` key for ESMR and IR which have URLs varying by year
- Size thresholds for detecting a valid file of this type 


#### `analysis_config.json`

Analysis parameters, thresholds, and some plotting settings are controlled by `wwna_variables_2024/analysis_config.json`. You can adjust these values to customize the analysis:

- `year_range`: DMR data year range for analysis (end year is inclusive)
- `wwna_list_path`: Path to the WWNA facilities list CSV file
- `grouping_columns`: Columns used to group data for trend analysis (typically should not be changed)
- `time_to_limit_years`: Global threshold in years for how soon a trend would reach the permit limit. The pipeline flags when the trend moves toward non-compliance, the near-exceedance test passes, and the average time-to-limit ≤ 5 yr.
- `limit_threshold`: Threshold for near-exceedance detection (default: 0.1 = 10%). For maximum limits, flags if Q3 > 90% of limit. For minimum limits, flags if Q1 < limit × 1.1. Decrease to be more selective, increase to flag more facilities.
- `iqr_multiplier`: Multiplier for outlier detection (default: 2.5). Outliers are values > Q3 + (multiplier × IQR). Increase to flag fewer outliers, decrease to flag more.


### Run all analysis steps in sequence:
#### Python
```bash
python wwna_variables_2024/RUN_ALL.py
```

#### R
```r
source("wwna_variables_2024/RUN_ALL.R")
```