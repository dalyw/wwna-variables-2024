# step2_population_served.R
# Updated to match Python step2_population_served.py methodology
# with support from Claude 4.0

library(tidyverse)
library(ggplot2)

source('wwna_variables_2024/helper_functions.R')

main <- function() {
  # Load and process all data sources
  cwns_df <- load_data("CWNS")
  covid_data <- load_data("WW_SURVEILLANCE")
  sso_data <- load_data("SSO")
  
  cat(sprintf("Loaded %d California facilities from CWNS data\n", nrow(cwns_df)))
  
  # Load manual matches to prioritize PERMIT_NUMBERs that match
  manual_matches <- suppressMessages(read_csv("data/manual_updates/cwns_facilities_match_manual.csv"))
  manual_permit_numbers <- unique(manual_matches$PERMIT_NUMBER[!is.na(manual_matches$PERMIT_NUMBER)])
  manual_permit_no_clean <- unique(manual_matches$PERMIT_NO_clean[!is.na(manual_matches$PERMIT_NO_clean)])
  all_manual_permits <- c(manual_permit_numbers, manual_permit_no_clean)
  
  # Debug: Check for CWNS_IDs with multiple PERMIT_NUMBERs
  # Filter out NA, empty, and invalid CWNS_IDs
  cwns_df_filtered <- cwns_df %>% 
    filter(!is.na(CWNS_ID), !is.na(PERMIT_NUMBER)) %>%
    filter(nchar(as.character(CWNS_ID)) > 0)
  
  cwns_id_counts <- cwns_df_filtered %>%
    group_by(CWNS_ID) %>%
    summarise(n_permits = n_distinct(PERMIT_NUMBER))
  
  multi_permit <- cwns_id_counts %>% 
    filter(n_permits > 1) %>%
    filter(!is.na(CWNS_ID))
  
  # Prefer PERMIT_NUMBERs that appear in manual matches
  if (nrow(multi_permit) > 0) {
    for (i in 1:min(5, nrow(multi_permit))) {
      cwns_id <- multi_permit$CWNS_ID[i]
      if (is.na(cwns_id) || length(cwns_id) == 0) next  # Skip NA or empty values
      
      row <- cwns_df_filtered %>%
        filter(CWNS_ID == cwns_id) %>%
        dplyr::select(CWNS_ID, PERMIT_NUMBER, FACILITY_NAME, population_cwns) %>%
        mutate(has_manual_match = PERMIT_NUMBER %in% all_manual_permits)
      
      cat(sprintf("  CWNS_ID %s:\n", cwns_id))
      print(row)
      
      matched_permits <- row %>% filter(has_manual_match)
      if (nrow(matched_permits) > 1) {
        cat(sprintf("    Multiple matched permits, keeping first: %s\n", matched_permits$PERMIT_NUMBER[1]))
      }
    }
  }
  
  # Aggregate CWNS data: group by CWNS_ID, sum population
  cwns_agg <- cwns_df_filtered %>%
  group_by(CWNS_ID) %>%
  summarise(
      population_cwns = sum(population_cwns, na.rm = TRUE),
      PERMIT_NUMBER = first(PERMIT_NUMBER)
    ) %>%
    filter(population_cwns > 0)
  
  cat(sprintf("After dropping duplicate CWNS_IDs: %d rows\n", nrow(cwns_agg)))
  
  # Handle cases where one CWNS_ID has multiple PERMIT_NUMBERs
  # For each CWNS_ID, prefer PERMIT_NUMBERs that are in manual matches
  cwns_agg_processed <- cwns_df_filtered %>%
    group_by(CWNS_ID) %>%
    slice_head(n = 1) %>%
    ungroup() %>%
    mutate(
      PERMIT_NUMBER_prio = ifelse(PERMIT_NUMBER %in% all_manual_permits, PERMIT_NUMBER, NA)
    )
  
  # For multi-permit CWNS_IDs, keep the one with manual match if available
  if (nrow(multi_permit) > 0) {
    for (cwns_id in multi_permit$CWNS_ID) {
      if (is.na(cwns_id) || length(cwns_id) == 0) next  # Skip NA or empty values
      group <- cwns_df_filtered %>% filter(CWNS_ID == cwns_id)
      has_match <- group$PERMIT_NUMBER %in% all_manual_permits
      
      if (any(has_match)) {
        # Keep first PERMIT_NUMBER with match
        keep_row <- group[which(has_match)[1], ]
        cwns_agg_processed <- cwns_agg_processed %>%
          filter(!(CWNS_ID == cwns_id)) %>%
          bind_rows(keep_row)
      }
    }
  }
  
  # Sum population by CWNS_ID
  cwns_agg <- cwns_agg_processed %>%
    group_by(CWNS_ID) %>%
    summarise(
      population_cwns = sum(population_cwns, na.rm = TRUE),
      PERMIT_NUMBER = first(PERMIT_NUMBER)
    ) %>%
    filter(population_cwns > 0)
  
  cat(sprintf("After dropping duplicate CWNS_IDs: %d rows\n", nrow(cwns_agg)))
  
  # Apply manual permit number mappings before merges
  if (file.exists("data/manual_updates/cwns_facilities_match_manual.csv")) {
    manual_map <- manual_matches %>%
      dplyr::select(PERMIT_NUMBER, PERMIT_NO_clean) %>%
      filter(!is.na(PERMIT_NO_clean)) %>%
      distinct(PERMIT_NUMBER, .keep_all = TRUE)
    
    if (nrow(manual_map) > 0) {
      cwns_agg <- cwns_agg %>%
        left_join(manual_map, by = "PERMIT_NUMBER") %>%
        mutate(PERMIT_NUMBER = ifelse(!is.na(PERMIT_NO_clean), PERMIT_NO_clean, PERMIT_NUMBER)) %>%
        dplyr::select(-PERMIT_NO_clean)
      
      cat(sprintf("Applied %d manual permit number mappings\n", nrow(manual_map)))
    }
  }
  
  # Merge COVID surveillance then SSO questionnaire population data
  merged_df <- cwns_agg %>%
    left_join(covid_data, by = c("PERMIT_NUMBER" = "epaid")) %>%
    left_join(sso_data, by = c("PERMIT_NUMBER" = "permit_number"))
  
  # Rename population columns for clarity
  merged_df <- merged_df %>%
    rename(
      population_covid = population_covid,
      population_sso = population_sso
    )
  
  # Classify facilities by data source
  has_cwns <- !is.na(merged_df$population_cwns)
  has_covid <- !is.na(merged_df$population_covid)
  has_sso <- !is.na(merged_df$population_sso)
  
  get_source <- function(i) {
    sources <- c()
    if (has_cwns[i]) sources <- c(sources, "CWNS")
    if (has_covid[i]) sources <- c(sources, "COVID")
    if (has_sso[i]) sources <- c(sources, "SSO")
    if (length(sources) == 0) return("Unmatched")
    return(paste(sources, collapse = "+"))
  }
  
  merged_df$source <- sapply(1:nrow(merged_df), get_source)
  pie_data <- table(merged_df$source)
  
  # Create figures directory if it doesn't exist
  figures_dir <- file.path(STEP_DIRS[["2"]], "figures_R")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create pie chart
  png(file.path(STEP_DIRS[["2"]], "figures_R", "population_source_comparison.png"), 
      width = 800, height = 800)
  pie(pie_data, labels = names(pie_data), main = "Population Data Sources for Facilities")
  dev.off()
  
  # Calculate statistics and identify discrepancies
  pop_columns <- c("population_cwns", "population_covid", "population_sso")
  available_cols <- pop_columns[pop_columns %in% names(merged_df)]
  
  # Calculate mean population (excluding NA values)
  merged_df$`Population Served` <- apply(
    merged_df[, available_cols, drop = FALSE], 
    1, 
    function(x) mean(x, na.rm = TRUE)
  )
  
  # Population Histogram
  png(file.path(STEP_DIRS[["2"]], "figures_R", "population_distribution.png"), 
      width = 800, height = 600)
  hist(merged_df$`Population Served`, breaks = 50, 
       main = "Population Distribution", 
       xlab = "Population Served",
       ylab = "Number of Facilities")
dev.off()

  # Save merged population data (only selected columns)
  merged_df_save <- merged_df %>%
    dplyr::select(CWNS_ID, PERMIT_NUMBER, population_cwns, population_covid, population_sso, source, `Population Served`)
  write_csv(merged_df_save, file.path(STEP_DIRS[["2"]], "merged_population_data_R.csv"))
}
