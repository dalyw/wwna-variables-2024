setwd('/Users/dalywettermark/Documents/git/wwna-variables-2024')

# Load required libraries
library(tidyverse)
library(jsonlite)
library(readxl)
library(sf)
library(ggplot2)

# Step 0: Download data (optional - skip if data already exists)
cat("----------------------------------------\n")
source('wwna_variables_2024/step0_download_data.R')
main()
cat("\nStep 0 complete\n\n")

# Step 1: Parameter Categorization
cat("----------------------------------------\n")
source('wwna_variables_2024/step1_parameter_categorization.R')
main()
cat("\n Step 1 complete\n\n")

# Step 2: Population Served
cat("----------------------------------------\n")
source('wwna_variables_2024/step2_population_served.R')
main()
cat("\n Step 2 complete\n\n")

# Step 3: Near Exceedance Analysis
cat("----------------------------------------\n")
source('wwna_variables_2024/step3_near_exceedence.R')
main()
cat("\n Step 3 complete\n\n")

# Step 4: Future Limits
cat("----------------------------------------\n")
source('wwna_variables_2024/step4_future_limits.R')
main()
cat("\n Step 4 complete\n\n")

# Final merge: Combine all step results into WWNA_LIST
cat("Final merge: combining all step results\n")

# Load the original WWNA_LIST
WWNA_LIST_FINAL <- suppressMessages(read_csv('data/wwna_list/NPDES+WDR Facilities List_20240906.csv'))
cat(sprintf("Original WWNA_LIST length: %d\n", nrow(WWNA_LIST_FINAL)))

# Load results from previous steps (use _R suffix for R outputs)
population <- suppressMessages(read_csv(file.path(STEP_DIRS[["2"]], "merged_population_data_R.csv")))
exceedance <- suppressMessages(read_csv(file.path(STEP_DIRS[["3"]], "flagged_facilities_step3_R.csv")))
future_limits <- suppressMessages(read_csv(file.path(STEP_DIRS[["4"]], "flagged_facilities_step4_R.csv")))

# Deduplicate population data before merging (some facilities have multiple CWNS records)
population <- population %>% 
  distinct(PERMIT_NUMBER, .keep_all = TRUE)
cat(sprintf("After deduplication: %d rows\n", nrow(population)))

# Merge population data
WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
  left_join(population, by = c("NPDES # CA#" = "PERMIT_NUMBER"))

cat(sprintf("After population merge: %d rows\n", nrow(WWNA_LIST_FINAL)))

# Merge exceedance data  
if (nrow(exceedance) > 0) {
  WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
    left_join(exceedance, by = c("NPDES # CA#" = "EXTERNAL_PERMIT_NMBR"))
}

cat(sprintf("After exceedance merge: %d rows\n", nrow(WWNA_LIST_FINAL)))

# Merge future limits data
if (nrow(future_limits) > 0) {
  future_limits_subset <- future_limits %>%
    dplyr::select(`NPDES # CA#`, 
           one_of(c(AGG_STRINGS[["4"]]$COUNT, AGG_STRINGS[["4"]]$PARAM)))
  
  WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
    left_join(future_limits_subset, by = "NPDES # CA#")
}

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

# Remove duplicates based on FACILITY ID
initial_count <- nrow(WWNA_LIST_FINAL)
WWNA_LIST_FINAL <- WWNA_LIST_FINAL %>%
  distinct(`FACILITY ID`, .keep_all = TRUE)

if (initial_count > nrow(WWNA_LIST_FINAL)) {
  cat(sprintf("Removed %d duplicates\n", initial_count - nrow(WWNA_LIST_FINAL)))
}

# Save the final merged list
write_csv(WWNA_LIST_FINAL, "processed_data/WWNA_LIST_FINAL_updated_R.csv")
cat("Saved updated facilities list\n")
