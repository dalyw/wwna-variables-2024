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
- `data/dmr/REF_Parameter.csv`: Reference parameter file with all DMR parameter codes and descriptions
- `data/ir`: California Integrated Report 303d list of impaired water bodies
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS
- `data/manual_updates/parameters_manual_additions.csv`: Manual parameter additions
- `data/manual_updates/parameter_sorting_dict.json`: Parameter categorization dictionary

**Outputs:**
- `processed_data/step1/ref_parameter_merged_py.csv` / `ref_parameter_merged_R.csv`: Merged parameter reference with categories
- `processed_data/step1/dmr_esmr_mapping_py.csv` / `dmr_esmr_mapping_R.csv`: Mapping between DMR and eSMR parameter codes

For DMR parameters, all parameter codes from `REF_Parameter.csv` are included in the mapping to maintain unique `PARAMETER_CODE -> PARAMETER_DESC` relationships. ESMR and IR parameter descriptions are mapped to DMR parameter codes, with deduplication prioritizing manual mappings and canonical codes to ensure one `PARAMETER_CODE` per ESMR parameter description. These will be used in steps 3 and 4 for ESMR and IR, respectively

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

Population growth rate is calculated as the annualized growth rate from 2022 to 2042 (20-year period), normalized to a per-year CAGR percentage: `((population_2042 / population_2022)^(1/20) - 1) * 100`.

### Step 3: Near Exceedance Analysis
**Files:** `step3_near_exceedance.py` / `step3_near_exceedance.R`

Identifies facilities that are at risk of exceeding their current permit limit in the near future, based on trends from historical monitoring data. This analysis flags facility-parameter combinations where the median value, combined with the historical trend slope, indicates the limit will be reached within the specified time threshold.

**Inputs:**
- `data/dmr`: EPA ICIS DMR datasets (LIMITS and DMR files for each year)
- `data/esmr`: Analytical results from electronic self-monitoring reports (eSMRs) from CIWQS database
- Parameter categorization from step 1 (for parameter name mapping)
- `data/wwna_list/NPDES+WDR Facilities List_20240906.csv`: WWNA facilities list
- `data/manual_updates/stat_base_code_mapping.json`: Mapping of ESMR calcultion methods to DMR stat base codes
- `data/manual_updates/unit_aliases.csv`: Unit normalization aliases

**Outputs:**
- `processed_data/step3/flagged_facilities_step3_py.csv` / `flagged_facilities_step3_R.csv`: Facilities flagged for near exceedance
- `processed_data/step3/figures_py/`: Visualizations showing facilities with near-exceedance patterns

**Methodology:**

A facility-parameter combination is flagged when the time-to-limit check indicates the limit will be reached within the specified threshold. The time-to-limit is calculated by projecting the median value forward using the historical trend slope. The slope must be positive for maximum limits (trending upward) or negative for minimum limits (trending downward). The time-to-limit is computed as the distance from the median to the limit divided by the absolute value of the slope. If this projected time is less than or equal to the `time_to_limit_years` threshold (default: 8 years), the facility-parameter combination is flagged. This forward-looking approach captures cases where values are trending toward limits even if current percentiles haven't reached near-exceedance thresholds yet.

Before trend analysis, values are filtered using a two-sided IQR rule. Outliers are values outside Q1 - (iqr_multiplier × IQR) and Q3 + (iqr_multiplier × IQR), where `iqr_multiplier` defaults to 3.0 in `analysis_config.json`. Only groups with at least one non-NA limit value entry after the recent year threshold (default: 2024) are analyzed, and groups must have at least 3 years of data span (difference between minimum and maximum monitoring dates) to ensure sufficient historical data for trend analysis. All historical data for qualifying groups is included in the analysis, not just recent data.

Several column normalizations are applied before merging DMR with LIMITS data to ensure consistent matching and grouping across years. `LIMIT_VALUE_TYPE_CODE` is truncated to the first character (C, Q, etc.) before merging, so that that DMR codes 'C1', 'C2', are treated the same. `PERM_FEATURE_NMBR` values are normalized to combine historical and recent naming conventions (e.g., '001' → 'EFF1', 'INF' → 'INF1') to ensure consistent grouping across years. `MONITORING_LOCATION_CODE` values are normalized in LIMITS data before merging (effluent codes '1', '2', 'EG', 'Y', 'K' are mapped to '1'). 

DMR and ESMR data are combined, with ESMR providing additional monitoring records. Both datasets are normalized to base units using the unit alias table. ESMR unit strings are normalized using the alias table (`unit_to_py` for Python, `unit_to_r` for R) in `data/manual_updates/unit_aliases.csv`, ensuring both the Python (pint) and R (units) pipelines resolve raw strings to a canonical unit before conversion. The resolved base-unit string plus the resolved `STATISTICAL_BASE_CODE` (derived directly from ESMR `calculated_method`) form a unique pair that is mapped to a single `LIMIT_VALUE_TYPE_CODE` via `data/manual_updates/stat_base_code_mapping.json`. The mapping file lists every allowed `(base_unit, STATISTICAL_BASE_CODE)` combination for each limit type; if an ESMR record does not match one of those combinations, the pipeline errors so the JSON can be updated explicitly. This guarantees Python and R runs share the same authoritative mapping of statistical bases to limit types before any DMR/ESMR merge or unit conversion happens.

Data is grouped by `LIMIT_GROUP_COLS` to identify similar monitoring scenarios:
- `EXTERNAL_PERMIT_NMBR`: Facility permit number
- `PARAMETER_CODE`: Parameter code
- `MONITORING_LOCATION_CODE`: Monitoring location (e.g., effluent discharge point)
- `PERM_FEATURE_NMBR`: Permit feature number (e.g., "001" for EFF-001)
- `STATISTICAL_BASE_CODE`: Statistical basis (daily max, monthly average, etc.)
- `STANDARD_UNIT_DESC`: Units (normalized to base units)
- `LIMIT_VALUE_TYPE_CODE`: Limit type code (concentration, flow, etc.)

DMR column descriptions are noted here: https://echo.epa.gov/tools/data-downloads/icis-npdes-dmr-summary

These columns together define a unique monitoring scenario (e.g., "Weekly maximum ammonia concentration (mg/L) as NH4-N at the effluent discharge point"). Historical monitoring data is aggregated across all limit changes for the same group, but comparisons are made against the most recent limit value.

The DMR dataset includes two files per year. The LIMITS file contains unique limit values with detailed information about monitoring schedules, applicable months, and limit periods. The DMR file contains individual monitoring event records with limit values and violation information. DMR and LIMITS data from all years are loaded, then LIMITS are deduplicated on `UNIQUE_LIMIT_COLS` (`LIMIT_SET_SCHEDULE_ID`, `LIMIT_VALUE_ID`, `LIMIT_VALUE_TYPE_CODE`). This allows DMR records from any year to match LIMITS records from any year, ensuring historical data can be matched with current limit definitions. For each `LIMIT_GROUP_COLS` combination, the most recent valid (non-NA) limit is determined by sorting by monitoring date and selecting the first record per group.

Parameter descriptions are looked up from the step1 mapping file. 

Analysis parameters are controlled by `analysis_config.json`. The `time_to_limit_years` parameter sets the maximum years for the median value to reach the limit based on the trend slope (default: 8). The `iqr_multiplier` parameter controls the multiplier for IQR outlier detection (default: 3.0). 

**Note:** This analysis aggregates historical monitoring data across all limit periods without filtering by monitoring active months. Since monitoring periods may change between permit renewals, data from previous permits may include months that are not relevant to the current limit's active monitoring period.

The `recent_violation_years` parameter sets the years to look back for excluding noncompliant facilities (default: 5). When `remove_noncompliant` is set to `TRUE`, facilities with recent violations (exceedances with `REPORTED_EXCURSION_NMBR > 0`, `VIOLATION_CODE == "E90"`, and `EXCEEDENCE_PCT > 0`) within the `recent_violation_years` period are excluded from near-exceedance flags. This avoids double-counting facilities that already have compliance issues.

When loading DMR data, the following filters are applied. 
1. Column selection loads only necessary columns for analysis. 
2. Data quality filters remove rows where No Data Indicator (NODI_CODE) is present and remove rows with no data (DMR_VALUE_NMBR is null). 
3. The monitoring location filter keeps only locations: 1, 2, EG, Y, or K (effluent or % removal). 
4. Parameter transformations strip leading zeros from parameter codes and mark toxicity parameters (codes starting with T or W) for optional filtering.

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
- `time_to_limit_years`: Threshold in years for how soon the median value would reach the permit limit based on the trend slope (default: 8). The pipeline flags when the trend moves toward non-compliance (positive slope for maximum limits, negative for minimum limits) and the projected time-to-limit is ≤ 8 years. Decrease to be more selective (flag only faster-approaching trends), increase to flag more facilities.
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