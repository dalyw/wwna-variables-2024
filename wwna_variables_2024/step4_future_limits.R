# step4_future_limits.R
# Updated to match Python step4_future_limits.py methodology

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(sf)

source('wwna_variables_2024/helper_functions.R')

# Load data
limits_2024 <- load_data("LIMITS", 2024)
parameter_reference <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "ref_parameter_merged_R.csv")))

# Merge parameter categories into NPDES limits
limits_2024 <- limits_2024 %>%
  mutate(PARAMETER_CODE_CLEAN = sub("^0+", "", PARAMETER_CODE)) %>%
  left_join(
    parameter_reference %>% dplyr::select(PARAMETER_CODE_CLEAN, PARENT_CATEGORY, SUB_CATEGORY),
    by = "PARAMETER_CODE_CLEAN"
  )

# Filter to only limits with valid numerical LIMIT_VALUE_STANDARD_UNITS (exclude monitoring-only)
limits_with_values <- limits_2024 %>%
  filter(!is.na(LIMIT_VALUE_STANDARD_UNITS) & LIMIT_VALUE_STANDARD_UNITS != "")

# Extract sub-categories that appear in CA LIMITS data with valid limits
# Only analyze IR data for categories that have actual numerical limits (not monitoring-only)
sub_categories <- unique(limits_with_values$SUB_CATEGORY[!is.na(limits_with_values$SUB_CATEGORY)])
cat(sprintf("%d categories in CA NPDES (with numerical limits): %s\n", 
            length(sub_categories), paste(sort(sub_categories), collapse = ", ")))

# Log unmapped parameters for categories that appear in LIMITS (with valid limits)
limits_with_categories <- limits_with_values %>% filter(!is.na(SUB_CATEGORY))
unmapped_params <- limits_with_categories %>%
  filter(is.na(PARENT_CATEGORY)) %>%
  pull(PARAMETER_DESC) %>%
  unique()
if (length(unmapped_params) > 0) {
  cat(sprintf("Unmapped parameters in LIMITS (with numerical limits): %s\n", 
              paste(unmapped_params, collapse = ", ")))
}


# Load IR data (Integrated Report 303(d) lists)
# Compares years from config to identify newly impaired water bodies
ir_parameter_df <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv")))
ir_303d <- list()
ir_keys <- sort(as.integer(names(FILE_CONFIGS$IR$year_config)))
ir_years <- c(ir_keys[1], ir_keys[length(ir_keys)])

for (year in ir_years) {
  df_year <- load_data("IR", year = year)
  
  df_year <- df_year %>%
    left_join(
      ir_parameter_df %>% dplyr::select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY),
      by = c("Pollutant" = "IR_PARAMETER_DESC")
    ) %>%
    # Only keep pollutants in regulated categories (those that appear in LIMITS)
    filter(SUB_CATEGORY %in% sub_categories)
  
  unmapped <- df_year %>%
    filter(is.na(PARENT_CATEGORY)) %>%
    pull(Pollutant) %>%
    unique()
  if (length(unmapped) > 0) {
    cat(sprintf("Unmapped pollutants in %d data (in regulated categories): %s\n", 
                year, paste(unmapped, collapse = ", ")))
  }
  
  ir_303d[[as.character(year)]] <- df_year
}

# Analyze impaired waters
facilities <- WWNA_LIST

# Create dictionaries for impaired water bodies
newly_impaired_bodies <- list()
impaired_water_bodies <- list()

for (category in sub_categories) {
  # Get water bodies from comparison years
  first_year <- ir_years[1]
  last_year <- ir_years[length(ir_years)]
  
  impaired_set_first <- ir_303d[[as.character(first_year)]] %>%
    filter(SUB_CATEGORY == category) %>%
    pull("Water Body CALWNUMS") %>%
    unique()
  
  impaired_set_last <- ir_303d[[as.character(last_year)]] %>%
    filter(SUB_CATEGORY == category) %>%
    pull("Water Body CALWNUMS") %>%
    unique()
  
  newly_impaired_bodies[[category]] <- setdiff(impaired_set_last, impaired_set_first)
  impaired_water_bodies[[category]] <- impaired_set_last
}

# Helper function to check if watershed contains impaired water body and return matching IDs
check_impaired <- function(x, water_bodies) {
  if (is.na(x)) return(list(matches = FALSE, ids = character(0)))
  x_str <- as.character(x)
  matching_ids <- sapply(water_bodies, function(wb) {
    if (str_detect(x_str, fixed(wb))) wb else NA_character_
  })
  matching_ids <- matching_ids[!is.na(matching_ids)]
  list(matches = length(matching_ids) > 0, ids = matching_ids)
}

# Find facilities discharging into newly impaired waters that are not yet limited
FLAGGED_STEP4_LIST <- list()

for (category in sub_categories) {
  # Check each facility for newly impaired waterbodies in this category
  for (idx in 1:nrow(facilities)) {
    watershed_name <- facilities$`CAL WATERSHED NAME`[idx]
    result <- check_impaired(watershed_name, newly_impaired_bodies[[category]])
    
    if (!result$matches) next
    
    npdes <- facilities$`NPDES # CA#`[idx]
    
    # Skip if npdes is NA or empty
    if (is.na(npdes) || length(npdes) == 0 || npdes == "") next
    
    sub_limits <- limits_2024 %>% filter(EXTERNAL_PERMIT_NMBR == npdes)
    
    # Check if facility monitors parameters in this category
    has_params_in_category <- any(sub_limits$SUB_CATEGORY == category, na.rm = TRUE)
    
    # Check if they have limits for those parameters
    has_limit <- any(
      (sub_limits$SUB_CATEGORY == category) & 
      !is.na(sub_limits$LIMIT_VALUE_STANDARD_UNITS) & 
      (sub_limits$LIMIT_VALUE_STANDARD_UNITS != ""),
      na.rm = TRUE
    )
    
    # Flag facilities that monitor parameters in category but lack limits
    if (has_params_in_category && !has_limit) {
      FLAGGED_STEP4_LIST[[length(FLAGGED_STEP4_LIST) + 1]] <- list(
        "NPDES # CA#" = npdes,
        SUB_CATEGORY = category,
        "Water Body ID" = paste(sort(unique(result$ids)), collapse = ", ")
      )
    }
  }
}

# Create DataFrame and aggregate flagged facilities
if (length(FLAGGED_STEP4_LIST) > 0) {
  flagged_df <- bind_rows(FLAGGED_STEP4_LIST)
  
  # Aggregate categories per facility
  aggregated <- aggregate_flagged_params(
    flagged_df,
    "NPDES # CA#",
    "SUB_CATEGORY",
    AGG_STRINGS[["4"]]
  )
  
  # Aggregate water body IDs per facility
  water_body_agg <- flagged_df %>%
    group_by(`NPDES # CA#`) %>%
    summarise(
      water_body_list = paste(sort(unique(unlist(strsplit(`Water Body ID`, ", ")))), collapse = ", "),
      .groups = "drop"
    ) %>%
    mutate(water_body_list = ifelse(water_body_list == "", "", water_body_list))
  
  # Merge water body IDs into aggregated results
  aggregated <- aggregated %>%
    left_join(water_body_agg, by = "NPDES # CA#") %>%
    mutate(`Water Body ID` = ifelse(is.na(water_body_list), "", water_body_list)) %>%
    dplyr::select(-water_body_list)
  
  # Save aggregated results for RUN_ALL merge
  write_csv(aggregated, file.path(STEP_DIRS[["4"]], "flagged_facilities_step4_R.csv"))
  
  # Merge with WWNA_LIST for full facility data needed for visualizations
  # Use proper column name for join
  join_col <- "NPDES # CA#"
  flagged_facilities <- WWNA_LIST %>%
    filter(!!sym(join_col) %in% aggregated[[join_col]]) %>%
    left_join(aggregated, by = join_col)
  
  # Generate visualizations
  # Bar plot of category counts
  all_categories <- unlist(strsplit(
    paste(flagged_facilities[[AGG_STRINGS[["4"]]$PARAM]], collapse = ", "),
    ", "
  ))
  
  category_counts <- table(trimws(all_categories)) %>%
    as.data.frame() %>%
    rename(Category = Var1, count = Freq) %>%
    arrange(count) %>%
    mutate(Category = factor(Category, levels = Category))
  
  plot_barh(
    category_counts,
    x_col = "count",
    y_col = "Category",
    xlabel = "Number of Facilities",
    path = "figures_R/category_summary.png",
    step = 4
  )
  
  # Map of facilities with parameter counts
  param_counts <- setNames(
    flagged_facilities[[AGG_STRINGS[["4"]]$COUNT]],
    flagged_facilities$`NPDES # CA#`
  )
  
  plot_map(param_counts, label_threshold = 4, step = 4)
} else {
  cat("No facilities flagged\n")
}
