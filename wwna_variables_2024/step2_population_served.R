# step2_population_served.R
# Updated to match Python step2_population_served.py methodology
# with support from Claude 4.0

# Note - NOT UPDATED compared to current .py file version as of Feb 2026

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
  manual_matches <- read_csv_tibble("data/manual_updates/cwns_facilities_match_manual.csv")
  manual_permit_numbers <- unique(manual_matches$PERMIT_NUMBER[!is.na(manual_matches$PERMIT_NUMBER)])
  manual_permit_no_clean <- unique(manual_matches$PERMIT_NO_clean[!is.na(manual_matches$PERMIT_NO_clean)])
  all_manual_permits <- c(manual_permit_numbers, manual_permit_no_clean)
  
  # Create mapping dictionary from PERMIT_NUMBER to PERMIT_NO_clean
  manual_map <- manual_matches %>%
    dplyr::select(PERMIT_NUMBER, PERMIT_NO_clean) %>%
    filter(!is.na(PERMIT_NO_clean)) %>%
    distinct(PERMIT_NUMBER, .keep_all = TRUE)
  
  # Aggregate CWNS data: group by CWNS_ID, prefer PERMIT_NUMBERs that match manual permits
  cwns_agg_list <- list()
  
  for (cwns_id in unique(cwns_df$CWNS_ID)) {
    group <- cwns_df %>% filter(CWNS_ID == cwns_id)
    
    if (nrow(group) > 1) {
      # Check which PERMIT_NUMBERs have manual matches
      has_match <- group$PERMIT_NUMBER %in% all_manual_permits
      if (any(has_match)) {
        # Keep the first one that has a match
        group <- group %>% filter(PERMIT_NUMBER %in% all_manual_permits) %>% slice_head(n = 1)
      } else {
        # Keep first if no matches
        group <- group %>% slice_head(n = 1)
      }
    }
    cwns_agg_list[[length(cwns_agg_list) + 1]] <- group
  }
  
  cwns_df <- bind_rows(cwns_agg_list)
  cat(sprintf("After dropping duplicate CWNS_IDs: %d rows\n", nrow(cwns_df)))
  
  # Apply manual permit number mappings before merges
  # Only apply non-identity mappings (where PERMIT_NUMBER != PERMIT_NO_clean)
  if (nrow(manual_map) > 0) {
    non_identity_count <- 0
    for (i in 1:nrow(manual_map)) {
      permit <- manual_map$PERMIT_NUMBER[i]
      cleaned <- manual_map$PERMIT_NO_clean[i]
      if (permit != cleaned) {
        cwns_df <- cwns_df %>%
          mutate(PERMIT_NUMBER = ifelse(PERMIT_NUMBER == permit, cleaned, PERMIT_NUMBER))
        non_identity_count <- non_identity_count + 1
      }
    }
    if (non_identity_count > 0) {
      cat(sprintf("Applied %d manual permit number mappings (from %d total mappings)\n", 
                  non_identity_count, nrow(manual_map)))
    }
  }
  
  # Merge COVID surveillance then SSO questionnaire population data
  merged_df <- cwns_df %>%
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
  
  # Create pie chart
  fig_path <- save_fig("population_source_comparison.png", step = 2, width = 8, height = 8)
  png(fig_path, width = 8, height = 8, units = "in", res = 150)
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
  
  # Calculate standard deviation between population sources
  merged_df$pop_std <- apply(
    merged_df[, available_cols, drop = FALSE], 
    1, 
    function(x) round(sd(x, na.rm = TRUE), 2)
  )
  merged_df$pop_std[is.na(merged_df$pop_std)] <- NA
  
  # Calculate annualized population growth rate from 2022 to 2042 (20-year period)
  # Using compound annual growth rate (CAGR): ((end/start)^(1/years) - 1) * 100
  from_cwns <- grepl("CWNS", merged_df$source)
  years <- 20  # 2022 to 2042
  merged_df$population_growth_rate <- NA
  merged_df$population_growth_rate[from_cwns] <- round(
    ((merged_df$population_cwns_2042[from_cwns] / merged_df$population_cwns[from_cwns])^(1 / years) - 1) * 100,
    2
  )
  
  # Population Histogram
  fig_path <- save_fig("population_distribution.png", step = 2, width = 10, height = 6)
  png(fig_path, width = 10, height = 6, units = "in", res = 150)
  hist(merged_df$`Population Served`, breaks = 50, 
       main = "Population Distribution", 
       xlab = "Population Served",
       ylab = "Number of Facilities")
  dev.off()

  # Deduplicate by PERMIT_NUMBER before saving
  # (some facilities have multiple CWNS records after merging)
  initial_rows <- nrow(merged_df)
  merged_df <- merged_df %>%
    distinct(PERMIT_NUMBER, .keep_all = TRUE)
  if (initial_rows != nrow(merged_df)) {
    cat(sprintf("Deduplicated population data: %d -> %d rows\n", initial_rows, nrow(merged_df)))
  }
  
  # Save merged population data (only selected columns)
  merged_df_save <- merged_df %>%
    dplyr::select(CWNS_ID, PERMIT_NUMBER, population_cwns, population_cwns_2042, population_covid, population_sso, source, `Population Served`, pop_std, population_growth_rate)
  write_csv(merged_df_save, file.path(STEP_DIRS[["2"]], "merged_population_data_R.csv"))
}
