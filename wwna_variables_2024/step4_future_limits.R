# step4_future_limits.R
# Updated to match Python step4_future_limits.py methodology

library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(sf)

source('wwna_variables_2024/helper_functions.R')

# Helper function to check if watershed contains impaired water body
check_impaired <- function(x, water_bodies) {
  if (is.na(x)) return(list(matches = FALSE, ids = character(0)))
  x_str <- as.character(x)
  matching_ids <- sapply(water_bodies, function(wb) {
    if (str_detect(x_str, fixed(wb))) wb else NA_character_
  })
  matching_ids <- matching_ids[!is.na(matching_ids)]
  list(matches = length(matching_ids) > 0, ids = matching_ids)
}

# Helper function to combine water body IDs
combine_water_bodies <- function(series) {
  all_ids <- character(0)
  for (wb_str in series) {
    if (!is.na(wb_str) && wb_str != "") {
      all_ids <- c(all_ids, trimws(strsplit(wb_str, ", ")[[1]]))
    }
  }
  paste(sort(unique(all_ids)), collapse = ", ")
}

# Load and merge parameter categories into NPDES limits
main <- function() {
  limits_2024 <- load_data("LIMITS", 2024, dropna = FALSE)
  param_ref <- read_csv_tibble(file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv"))
  limits_2024 <- limits_2024 %>%
    left_join(
      param_ref %>% dplyr::select(PARAMETER_CODE, PARENT_CATEGORY, SUB_CATEGORY),
      by = "PARAMETER_CODE"
    )

  # Extract categories with valid limits (for IR filtering)
  limits_with_values <- limits_2024 %>%
    filter(!is.na(LIMIT_VALUE_STANDARD_UNITS) & LIMIT_VALUE_STANDARD_UNITS != "")
  sub_categories <- unique(limits_with_values$SUB_CATEGORY[!is.na(limits_with_values$SUB_CATEGORY)])
  cat(sprintf("%d categories in CA NPDES: %s\n", 
              length(sub_categories), paste(sort(sub_categories), collapse = ", ")))

  # Log unmapped parameters
  unmapped <- limits_with_values %>%
    filter(is.na(PARENT_CATEGORY)) %>%
    pull(PARAMETER_DESC) %>%
    unique()
  if (length(unmapped) > 0) {
    cat(sprintf("Unmapped parameters in LIMITS: %s\n", paste(unmapped, collapse = ", ")))
  }

  # Load IR data and merge categories
  ir_param_df <- read_csv_tibble(file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv"))
  ir_keys <- sort(as.integer(names(FILE_CONFIGS$IR$year_config)))
  ir_years <- c(ir_keys[1], ir_keys[length(ir_keys)])
  ir_303d <- list()
  for (year in ir_years) {
    df_year <- load_data("IR", year = year, rename = FALSE) %>%
      left_join(
        ir_param_df %>% dplyr::select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY),
        by = c("Pollutant" = "IR_PARAMETER_DESC")
      ) %>%
      filter(SUB_CATEGORY %in% sub_categories)
    ir_303d[[as.character(year)]] <- df_year
    unmapped <- df_year %>%
      filter(is.na(PARENT_CATEGORY)) %>%
      pull(Pollutant) %>%
      unique()
    if (length(unmapped) > 0) {
      cat(sprintf("Unmapped pollutants in %d data: %s\n", year, paste(unmapped, collapse = ", ")))
    }
  }

  # Identify newly impaired water bodies
  facilities <- WWNA_LIST
  newly_impaired_bodies <- list()
  first_year <- ir_years[1]
  last_year <- ir_years[length(ir_years)]
  for (category in sub_categories) {
    impaired_first <- ir_303d[[as.character(first_year)]] %>%
      filter(SUB_CATEGORY == category) %>%
      pull("Water Body CALWNUMS") %>%
      unique()
    impaired_last <- ir_303d[[as.character(last_year)]] %>%
      filter(SUB_CATEGORY == category) %>%
      pull("Water Body CALWNUMS") %>%
      unique()
    newly_impaired_bodies[[category]] <- setdiff(impaired_last, impaired_first)
  }

  # Find facilities discharging into newly impaired waters that are not yet limited
  FLAGGED_STEP4_LIST <- list()
  for (category in sub_categories) {
    impaired_set <- newly_impaired_bodies[[category]]
    for (idx in 1:nrow(facilities)) {
      watershed_name <- facilities$`CAL WATERSHED NAME`[idx]
      result <- check_impaired(watershed_name, impaired_set)
      if (!result$matches) next
      
      npdes <- facilities$`NPDES # CA#`[idx]
      facility_limits <- limits_2024 %>% filter(EXTERNAL_PERMIT_NMBR == npdes)
      if (any(facility_limits$SUB_CATEGORY == category, na.rm = TRUE)) {
        FLAGGED_STEP4_LIST[[length(FLAGGED_STEP4_LIST) + 1]] <- list(
          "NPDES # CA#" = npdes,
          SUB_CATEGORY = category,
          "Water Body ID" = paste(sort(unique(result$ids)), collapse = ", ")
        )
      }
    }
  }

  # Aggregate flagged facilities
  flagged_df <- bind_rows(FLAGGED_STEP4_LIST) %>% as_regular_tibble()
  print(head(flagged_df))

  aggregated <- aggregate_flags(
    flagged_df,
    "NPDES # CA#",
    "SUB_CATEGORY",
    AGG_STRINGS[["4"]]
  )

  # Aggregate water body IDs
  water_body_agg <- flagged_df %>%
    group_by(`NPDES # CA#`) %>%
    summarise(
      "Water Body ID" = combine_water_bodies(`Water Body ID`),
      .groups = "drop"
    )
  aggregated <- aggregated %>%
    left_join(water_body_agg, by = "NPDES # CA#") %>%
    mutate(`Water Body ID` = ifelse(is.na(`Water Body ID`), "", `Water Body ID`))

  # Save and visualize
  write_csv(aggregated, file.path(STEP_DIRS[["4"]], "flagged_facilities_step4_R.csv"))

  # Merge with WWNA_LIST for full facility data needed for visualizations
  flagged_facilities <- WWNA_LIST %>%
    filter(`NPDES # CA#` %in% aggregated$`NPDES # CA#`) %>%
    inner_join(aggregated, by = "NPDES # CA#")

  # Bar plot of category counts
  all_categories <- unlist(strsplit(
    paste(flagged_facilities[[AGG_STRINGS[["4"]]$PARAM]], collapse = ", "),
    ", "
  ))
  category_counts <- table(trimws(all_categories)) %>%
    as.data.frame() %>%
    rename(Category = Var1, count = Freq)
  plot_barh(
    category_counts,
    x_col = "count",
    y_col = "Category",
    xlabel = "Number of Facilities with Possible Future Limits",
    path = "category_summary.png",
    step = 4
  )

  # Map of facilities with parameter counts
  plot_map(
    setNames(flagged_facilities[[AGG_STRINGS[["4"]]$COUNT]], flagged_facilities$`NPDES # CA#`),
    6,
    4
  )
}
