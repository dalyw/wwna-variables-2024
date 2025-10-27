# step4_future_limits.R
# Updated to match Python step4_future_limits.py methodology

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(sf)

source('wwna_variables_2024/helper_functions.R')

# Load data
limits_2023 <- load_data("LIMITS", 2023)
parameter_reference <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "ref_parameter_merged_R.csv")))

# Merge parameter categories into NPDES limits
limits_2023 <- limits_2023 %>%
  mutate(PARAMETER_CODE_CLEAN = sub("^0+", "", PARAMETER_CODE)) %>%
  left_join(
    parameter_reference %>% dplyr::select(PARAMETER_CODE_CLEAN, PARENT_CATEGORY, SUB_CATEGORY),
    by = "PARAMETER_CODE_CLEAN"
  )

sub_categories <- unique(parameter_reference$SUB_CATEGORY[!is.na(parameter_reference$SUB_CATEGORY)])

# Load categories to exclude from future limits analysis
if (file.exists("data/manual_updates/categories_to_exclude_from_future_limits.csv")) {
  exclude_df <- suppressMessages(read_csv("data/manual_updates/categories_to_exclude_from_future_limits.csv"))
  exclude_categories <- unique(exclude_df$SUB_CATEGORY)
  sub_categories <- sub_categories[!sub_categories %in% exclude_categories]
  
  if (length(exclude_categories) > 0) {
    cat(sprintf("Excluding %d categories: %s\n", length(exclude_categories), 
                paste(sort(exclude_categories), collapse = ", ")))
  }
}

# Filter out excluded categories before checking unmapped parameters
limits_filtered <- limits_2023 %>% 
  filter(!SUB_CATEGORY %in% exclude_categories)


# Load IR data (Integrated Report 303(d) lists)
# Compares 2018 vs 2024 to identify newly impaired water bodies
ir_parameter_df <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv")))
ir_303d <- list()

for (year in c(2018, 2024)) {
  df_year <- load_data("IR", year = year)
  
  df_year <- df_year %>%
    left_join(
      ir_parameter_df %>% dplyr::select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY),
      by = c("Pollutant" = "IR_PARAMETER_DESC")
    )
  
  ir_303d[[as.character(year)]] <- df_year
}

# Analyze impaired waters
facilities <- WWNA_LIST

# Create dictionaries for impaired water bodies
newly_impaired_bodies <- list()
impaired_water_bodies <- list()

for (category in sub_categories) {
  # Get water bodies from 2018 and 2024
  impaired_set_2018 <- ir_303d[["2018"]] %>%
    filter(SUB_CATEGORY == category) %>%
    pull("Water Body CALWNUMS") %>%
    unique()
  
  impaired_set_2024 <- ir_303d[["2024"]] %>%
    filter(SUB_CATEGORY == category) %>%
    pull("Water Body CALWNUMS") %>%
    unique()
  
  newly_impaired_bodies[[category]] <- setdiff(impaired_set_2024, impaired_set_2018)
  impaired_water_bodies[[category]] <- impaired_set_2024
}

# Helper function to check if watershed contains impaired water body
check_impaired <- function(x, water_bodies) {
  if (is.na(x)) return(FALSE)
  any(sapply(water_bodies, function(wb) str_detect(as.character(x), fixed(wb))))
}

# Find facilities discharging into newly impaired waters that are not yet limited
FLAGGED_STEP4_LIST <- list()

for (category in sub_categories) {
  # Filter to facilities discharging into newly impaired waterbodies for this category
  newly_impaired_mask <- sapply(facilities$`CAL WATERSHED NAME`, 
                                 check_impaired, 
                                 water_bodies = newly_impaired_bodies[[category]])
  
  # Check each affected facility
  affected_facilities <- facilities[newly_impaired_mask, ]
  
  for (idx in 1:nrow(affected_facilities)) {
    npdes <- affected_facilities$`NPDES # CA#`[idx]
    
    # Skip if npdes is NA or empty
    if (is.na(npdes) || length(npdes) == 0 || npdes == "") next
    
    sub_limits <- limits_2023 %>% filter(EXTERNAL_PERMIT_NMBR == npdes)
    
    # Check if facility monitors parameters in this category
    has_params_in_category <- any(sub_limits$SUB_CATEGORY == category, na.rm = TRUE)
    
    # Check if they have limits for those parameters
    has_limit <- any(
      (sub_limits$SUB_CATEGORY == category) & 
      !is.na(sub_limits$LIMIT_VALUE_NMBR) & 
      (sub_limits$LIMIT_VALUE_NMBR != ""),
      na.rm = TRUE
    )
    
    # Flag facilities that monitor parameters in category but lack limits
    if (has_params_in_category && !has_limit) {
      FLAGGED_STEP4_LIST[[length(FLAGGED_STEP4_LIST) + 1]] <- list(
        "NPDES # CA#" = npdes,
        SUB_CATEGORY = category
      )
    }
  }
}

# Create DataFrame and aggregate flagged facilities
if (length(FLAGGED_STEP4_LIST) > 0) {
  flagged_df <- bind_rows(FLAGGED_STEP4_LIST)
  
  aggregated <- aggregate_flagged_params(
    flagged_df,
    "NPDES # CA#",
    "SUB_CATEGORY",
    AGG_STRINGS[["4"]]
  )
  
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
    rename(Category = Var1, count = Freq)
  
  # Create bar plot
  p <- ggplot(category_counts, aes(x = reorder(Category, count), y = count)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    xlab("Category") +
    ylab("Number of Facilities") +
    ggtitle("Facilities Needing Limits by Category") +
    theme_minimal()
  
  figures_dir <- file.path(STEP_DIRS[["4"]], "figures_R")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  ggsave(file.path(STEP_DIRS[["4"]], "figures_R", "category_summary.png"), p, 
         width = 10, height = 6, units = "in")
  
  # Map of facilities with parameter counts
  param_counts <- setNames(
    flagged_facilities[[AGG_STRINGS[["4"]]$COUNT]],
    flagged_facilities$`NPDES # CA#`
  )
  
  plot_map(param_counts, label_threshold = 4, step = 4)
} else {
  cat("No facilities flagged\n")
}
