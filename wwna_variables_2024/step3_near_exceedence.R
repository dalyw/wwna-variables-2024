# step3_near_exceedence.R
# Updated to match Python step3_near_exceedance.py methodology
# with support from Claude 4.0

library(tidyverse)
library(lubridate)
library(scales)
library(parallel)

source('wwna_variables_2024/helper_functions.R')

# Grouping columns (DMR dataframe column names)
GROUP_COLS <- c(
  "EXTERNAL_PERMIT_NMBR",
  "PARAMETER_CODE",
  "STANDARD_UNIT_DESC",
  "MONITORING_LOCATION_CODE"
)

# Output columns
OUTPUT_COLS <- c(GROUP_COLS, "LIMIT_SET_SCHEDULE_ID", "LIMIT_VALUE_TYPE_CODE")

# Function to get flagged facilities (simplified to match Python structure)
get_flagged_facilities <- function(facility_records, slope_threshold = 0.05, limit_threshold = 0.1) {
  flagged_slope <- list()
  flagged_near_exceedance <- list()
  
  for (rec in facility_records) {
    # Skip if no valid limit
    if (is.null(rec$latest_limit) || is.na(rec$latest_limit)) next
    
    # Check for facilities with significant slope
    if (rec$qualifier %in% c("<=", "<") && rec$slope > slope_threshold) {
      near_exceedance <- rec$Q3 > (1 - limit_threshold) * rec$latest_limit
      has_slope <- rec$slope > slope_threshold
    } else if (rec$qualifier %in% c(">=", ">") && rec$slope < -slope_threshold) {
      near_exceedance <- rec$Q1 < (1 + limit_threshold) * rec$latest_limit
      has_slope <- rec$slope < -slope_threshold
    } else {
      has_slope <- FALSE
      near_exceedance <- FALSE
    }
    
    if (has_slope) {
      flagged_slope[[length(flagged_slope) + 1]] <- rec
    }
    if (near_exceedance) {
      flagged_near_exceedance[[length(flagged_near_exceedance) + 1]] <- rec
    }
  }
  
  # Find records in both lists by comparing GROUP_COLS
  slope_keys <- sapply(flagged_slope, function(rec) {
    paste(rec[GROUP_COLS], collapse = "|")
  })
  exceedance_keys <- sapply(flagged_near_exceedance, function(rec) {
    paste(rec[GROUP_COLS], collapse = "|")
  })
  
  common_keys <- intersect(slope_keys, exceedance_keys)
  flagged_all <- lapply(which(slope_keys %in% common_keys), function(i) flagged_slope[[i]])
  
  cat(sprintf("%d w/ slope>slope_threshold\n", length(flagged_slope)))
  cat(sprintf("%d pairs with Q1/Q3 > %f\n", length(flagged_near_exceedance), limit_threshold))
  cat(sprintf("%d pairs with both\n", length(flagged_all)))
  
  if (length(flagged_all) > 0) {
    cat(sprintf("%d facilities affected\n", length(unique(sapply(flagged_all, function(r) r$EXTERNAL_PERMIT_NMBR)))))
  }
  
  return(flagged_all)
}

# Function to process facility group
process_facility_group <- function(args) {
  key_tuple <- args$key_tuple
  group <- args$group
  
  # Convert data types and handle missing values
  dates <- as.numeric(group$MONITORING_PERIOD_END_DATE_NUMERIC)
  values <- as.numeric(group$DMR_VALUE_STANDARD_UNITS)
  
  # Remove NaN values and ensure unique x values
  mask <- !is.na(dates) & !is.na(values)
  dates <- dates[mask]
  values <- values[mask]
  
  # Get unique x values and their corresponding y means
  unique_dates <- unique(dates)
  if (length(unique_dates) < 3) {
    return(NULL)
  }
  
  unique_vals <- sapply(unique_dates, function(d) mean(values[dates == d]))
  
  # Calculate quartiles
  Q1_percentile <- quantile(values, 0.25)
  Q3_percentile <- quantile(values, 0.75)
  
  # Center and scale the data, then get linear regression
  dates_norm <- (unique_dates - mean(unique_dates)) / sd(unique_dates)
  vals_mean <- mean(unique_vals)
  vals_std <- ifelse(sd(unique_vals) > 0, sd(unique_vals), 1)
  values_norm <- (unique_vals - vals_mean) / vals_std
  
  fit <- lm(values_norm ~ dates_norm)
  slope <- coef(fit)[2]
  r_value <- summary(fit)$r.squared
  p_value <- summary(fit)$coefficients[2, 4]
  
  # Convert slope back to original scale and reset if poor fit
  if (r_value < 0.1 || is.na(p_value) || p_value > 0.05) {
    slope <- 0
  } else {
    slope <- slope * (vals_std / sd(unique_dates))
  }
  
  # Get the most recent limit value
  limits <- as.numeric(group$LIMIT_VALUE_STANDARD_UNITS)
  latest_limit <- ifelse(length(limits) > 0, limits[length(limits)], NA)
  
  result <- list(
    slope = slope,
    latest_limit = latest_limit,
    qualifier = group$LIMIT_VALUE_QUALIFIER_CODE[1],
    Q1 = Q1_percentile,
    Q3 = Q3_percentile,
    LIMIT_VALUE_TYPE_CODE = group$LIMIT_VALUE_TYPE_CODE[1],
    LIMIT_SET_SCHEDULE_ID = group$LIMIT_SET_SCHEDULE_ID[1]
  )
  
  # Add grouping columns
  for (i in seq_along(GROUP_COLS)) {
    result[[GROUP_COLS[i]]] <- key_tuple[[i]]
  }
  
  return(result)
}

main <- function(save = FALSE, drop_toxicity = FALSE) {
  # Load unique parameter codes from step1 output
  unique_param_codes <- read_csv(file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv")) %>%
    pull(PARAMETER_CODE) %>%
    unique()
  
  # Load and filter DMR data
  data_dict <- list()
  
  for (year in analysis_range) {
    data <- load_data("DMR", year = year, drop_toxicity = drop_toxicity)
    data_dict[[as.character(year)]] <- data
  }
  
  if (save) {
    # Concatenate all years and save as CSV
    # Ensure consistent types before binding
    all_data <- bind_rows(lapply(names(data_dict), function(y) {
      df <- data_dict[[y]]
      # Convert RNC_RESOLUTION_CODE to character to avoid type mismatches
      if ("RNC_RESOLUTION_CODE" %in% names(df)) {
        df$RNC_RESOLUTION_CODE <- as.character(df$RNC_RESOLUTION_CODE)
      }
      return(df)
    }))
    filename <- file.path(STEP_DIRS[["3"]], paste0("dmr_all_years_R.csv"))
    write_csv(all_data, filename)
    cat(sprintf("Saved %d records from %d years\n", nrow(all_data), length(data_dict)))
  }
  
  # Filter by unique parameter codes and ensure consistent types
  filtered_data <- bind_rows(lapply(names(data_dict), function(y) {
    df <- data_dict[[y]] %>% filter(PARAMETER_CODE %in% unique_param_codes)
    # Convert RNC_RESOLUTION_CODE to character to avoid type mismatches
    if ("RNC_RESOLUTION_CODE" %in% names(df)) {
      df$RNC_RESOLUTION_CODE <- as.character(df$RNC_RESOLUTION_CODE)
    }
    return(df)
  }))
  
  # Group data
  grouped_data <- filtered_data %>%
    group_by(across(all_of(GROUP_COLS)))
  
  # Parallel processing
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  clusterEvalQ(cl, {
    library(dplyr)
    library(tidyr)
  })
  # Export GROUP_COLS to cluster workers
  clusterExport(cl, "GROUP_COLS")
  
  # Prepare data for parallel processing
  grouped_list <- grouped_data %>%
    group_split()
  
  key_tuples <- grouped_data %>%
    group_keys()
  
  args_list <- lapply(1:nrow(key_tuples), function(i) {
    list(key_tuple = key_tuples[i, ], group = grouped_list[[i]])
  })
  
  # Process in parallel
  results <- parLapply(cl, args_list, process_facility_group)
  stopCluster(cl)
  
  # Filter out NULL results and get flagged facilities (keep as list of dicts like Python)
  facility_records <- lapply(results, function(r) {
    if (is.null(r)) return(NULL)
    r
  })
  facility_records <- facility_records[!sapply(facility_records, is.null)]
  
  # Get flagged facilities (returns list of records, not DataFrame)
  flagged_facilities <- get_flagged_facilities(facility_records, slope_threshold = 0.05, limit_threshold = 0.1)
  
  # Merge flagged facilities with actual data for plotting
  if (length(flagged_facilities) > 0) {
    # Convert flagged facilities list to DataFrame
    flagged_facilities_df <- bind_rows(lapply(flagged_facilities, as.data.frame))
    
    flagged_data <- filtered_data %>%
      inner_join(flagged_facilities_df, by = GROUP_COLS)
    
    # Count parameters per facility
    param_counts <- flagged_facilities_df %>%
      group_by(EXTERNAL_PERMIT_NMBR) %>%
      summarise(num_parameters = n_distinct(PARAMETER_CODE))
    
    # Generate visualizations (simplified - full plotting would go here)
    
    # Save detailed results (only selected columns)
    flagged_facilities_save <- flagged_facilities_df %>%
      dplyr::select(EXTERNAL_PERMIT_NMBR, PARAMETER_CODE, slope, latest_limit, qualifier, Q1, Q3)
    write_csv(flagged_facilities_save, file.path(STEP_DIRS[["3"]], "flagged_facilities_R.csv"))
    
    # Save aggregated results
    aggregated <- aggregate_flagged_params(
      flagged_facilities_df,
      "EXTERNAL_PERMIT_NMBR",
      "PARAMETER_CODE",
      AGG_STRINGS[["3"]]
    )
    write_csv(aggregated, file.path(STEP_DIRS[["3"]], "flagged_facilities_step3_R.csv"))
  } else {
    cat("No facilities flagged\n")
  }
}

