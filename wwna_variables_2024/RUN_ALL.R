setwd('/home/daly/git/wwna-variables-2024')

# Load required libraries
library(tidyverse)
library(jsonlite)
library(readxl)
library(sf)
library(ggplot2)

# Load helper functions to get STEP_DIRS
source('wwna_variables_2024/helper_functions.R')

# Create processed_data directory and subdirectories if they don't exist
dir.create("processed_data", showWarnings = FALSE, recursive = TRUE)
for (i in 1:4) {
  dir.create(STEP_DIRS[[as.character(i)]], showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(STEP_DIRS[[as.character(i)]], "figures_R"), showWarnings = FALSE, recursive = TRUE)
}
# Create csvs_py and csvs_R directories for step3
dir.create(file.path(STEP_DIRS[["3"]], "csvs_R"), showWarnings = FALSE, recursive = TRUE)

# Step 0: Download data (optional - skip if data already exists)
source('wwna_variables_2024/step0_download_data.R')
main()
cat("\nStep 0 complete\n\n")

# Step 1: Parameter Categorization
source('wwna_variables_2024/step1_parameter_categorization.R')
main()
cat("\n Step 1 complete\n\n")

# # Step 2: Population Served
# source('wwna_variables_2024/step2_population_served.R')
# main()
# cat("\n Step 2 complete\n\n")

# Step 3: Near Exceedance Analysis
source('wwna_variables_2024/step3_near_exceedance.R')
main(exclude_noncompliant = TRUE)
cat("\n Step 3 complete\n\n")

# Step 4: Future Limits
source('wwna_variables_2024/step4_future_limits.R')
main()
cat("\n Step 4 complete\n\n")

# Final merge: Combine all step results into WWNA_LIST
cat("Final merge: combining all step results\n")

# Load the original WWNA_LIST
WWNA_LIST_FINAL <- read_csv_tibble('data/wwna_list/NPDES+WDR Facilities List_20240906.csv')
cat(sprintf("Original WWNA_LIST length: %d\n", nrow(WWNA_LIST_FINAL)))

# Load results from previous steps (use _R suffix for R outputs)
population <- read_csv_tibble(file.path(STEP_DIRS[["2"]], "merged_population_data_R.csv"))
exceedance <- read_csv_tibble(file.path(STEP_DIRS[["3"]], "flagged_facilities_step3_R.csv"))
future_limits <- read_csv_tibble(file.path(STEP_DIRS[["4"]], "flagged_facilities_step4_R.csv"))

# Merge population data
WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
  left_join(population, by = c("NPDES # CA#" = "PERMIT_NUMBER"))

cat(sprintf("After population merge: %d rows\n", nrow(WWNA_LIST_FINAL)))

# Merge exceedance data  
WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
  left_join(exceedance, by = c("NPDES # CA#" = "EXTERNAL_PERMIT_NMBR"))

cat(sprintf("After exceedance merge: %d rows\n", nrow(WWNA_LIST_FINAL)))

# Merge future limits data
future_limits_subset <- future_limits %>%
  dplyr::select(`NPDES # CA#`, 
         one_of(c(AGG_STRINGS[["4"]]$COUNT, AGG_STRINGS[["4"]]$PARAM)))

WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
  left_join(future_limits_subset, by = "NPDES # CA#")

cat(sprintf("After future limits merge: %d rows\n", nrow(WWNA_LIST_FINAL)))

# Fill NA values with defaults (0 for numeric, "" for character)
num_cols <- names(WWNA_LIST_FINAL)[sapply(WWNA_LIST_FINAL, is.numeric)]
char_cols <- names(WWNA_LIST_FINAL)[sapply(WWNA_LIST_FINAL, is.character)]

# Fill with appropriate defaults
for (col in names(WWNA_LIST_FINAL)) {
  if (col %in% num_cols) {
    WWNA_LIST_FINAL[[col]][is.na(WWNA_LIST_FINAL[[col]])] <- 0
  } else if (col %in% char_cols) {
    WWNA_LIST_FINAL[[col]][is.na(WWNA_LIST_FINAL[[col]])] <- ""
  }
}

# Save the final merged list
write_csv(WWNA_LIST_FINAL, "processed_data/WWNA_LIST_updated_R.csv")
cat("Saved updated facilities list\n")
