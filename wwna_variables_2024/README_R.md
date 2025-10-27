Install all required packages by running this single command in R:

```r
install.packages(c("tidyverse", "jsonlite", "readxl", "sf", "raster", "viridis", "gridExtra", "scales", "httr"))
```

## Required Packages

### Core Packages
- **tidyverse** - Includes dplyr, readr, tidyr, ggplot2, stringr, lubridate, purrr, forcats
- **jsonlite** - For reading JSON configuration files
- **ggplot2** - For creating visualizations

### Data I/O
- **readxl** - For reading Excel files (IR 303d lists, manual updates)
- **readr** - For reading CSV files (part of tidyverse)

### Spatial Analysis
- **sf** - For reading and working with shapefiles (CA counties)
- **raster** - For spatial data operations
- **viridis** - For color scales in maps

### Visualization
- **gridExtra** - For arranging multiple plots on one page
- **scales** - For formatting axes and legends

### Other Utilities
- **parallel** - For parallel processing (part of base R)
- **httr** - For downloading files from URLs
- **lubridate** - For date/time operations (part of tidyverse)

## Running the Analysis

### Step 0: Download Data
```r
source('wwna_variables_2024/step0_download_data.R')
main()
```

### Step 1: Parameter Categorization
```r
source('wwna_variables_2024/step1_parameter_categorization.R')
main()
```

### Step 2: Population Served
```r
source('wwna_variables_2024/step2_population_served.R')
main()
```

### Step 3: Near Exceedance Analysis
```r
source('wwna_variables_2024/step3_near_exceedence.R')
main()
```

### Step 4: Future Limits
```r
source('wwna_variables_2024/step4_future_limits.R')
main()
```

## Troubleshooting

### Package Installation Issues

If you encounter errors installing packages, try:

1. **Update R to the latest version**
   ```r
   update.packages(checkBuilt = TRUE, ask = FALSE)
   ```

2. **Install from source if binary fails**
   ```r
   install.packages("package_name", type = "source")
   ```

3. **Install system dependencies for spatial packages (macOS/Linux)**
   
   On macOS with Homebrew:
   ```bash
   brew install gdal proj geos
   ```
   
   On Ubuntu/Debian:
   ```bash
   sudo apt-get install libgdal-dev libproj-dev libgeos-dev
   ```

### File Path Issues

If you get "cannot open file" errors:

1. Check your working directory:
   ```r
   getwd()
   ```

2. Set it to the project root:
   ```r
   setwd('/Users/dalywettermark/Documents/git/wwna-variables-2024')
   ```

### Permission Issues

If you get permission denied errors:

1. Create a personal library directory:
   ```r
   Sys.setenv(R_LIBS_USER = "~/R/%p-%v")
   .libPaths(c("~/R/%p-%v", .libPaths()))
   ```

## File Structure

The R scripts expect data files in these locations:

- `data/dmr/` - DMR (Discharge Monitoring Report) data files
- `data/esmr/` - eSMR (electronic Self-Monitoring Report) data files
- `data/ir/` - Integrated Report 303d list files
- `data/cwns/` - CWNS (Clean Watersheds Needs Survey) data
- `data/sso/` - SSO Questionnaire data
- `data/toxics/` - Toxics criteria data
- `data/wwna_list/` - WWNA facilities list
- `data/manual_updates/` - Manual parameter mappings and additions
- `processed_data/step1/` - Output from step 1
- `processed_data/step2/` - Output from step 2
- `processed_data/step3/` - Output from step 3
- `processed_data/step4/` - Output from step 4

## Notes

- The R scripts have been updated to match the Python methodology
- All steps use the same `helper_functions.R` and `file_configs.json` as the Python scripts
- The scripts are designed to be run in sequence (step 0 → step 1 → step 2 → step 3 → step 4)

## Getting Help

If you encounter issues:

1. Check that all packages are installed: `installed.packages()`
2. Verify your working directory: `getwd()`
3. Check that data files exist in the expected locations
4. Review the error messages - they often provide helpful clues

For more information, see the main README.md file.

