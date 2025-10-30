# step3_near_exceedance.R
# Updated to match Python step3_near_exceedance.py methodology
# with support from Claude 4.0

library(tidyverse)
library(lubridate)
library(scales)
library(parallel)
library(gridExtra)

source('wwna_variables_2024/helper_functions.R')

# Grouping columns (DMR dataframe column names) 
GROUP_COLS <- unlist(ANALYSIS_CONFIG$step3$grouping_columns)
OUTPUT_COLS <- c(GROUP_COLS, "LIMIT_SET_SCHEDULE_ID", "LIMIT_VALUE_TYPE_CODE")

# Thresholds from config
TIME_TO_LIMIT_YEARS <- ANALYSIS_CONFIG$step3$time_to_limit_years
LIMIT_THRESHOLD <- ANALYSIS_CONFIG$step3$limit_threshold

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
  
  # Check if we have any valid data points
  if (length(values) == 0) {
    return(NULL)
  }
  
  # Calculate quartiles first (needed for outlier detection)
  Q1_percentile <- quantile(values, 0.25)
  Q3_percentile <- quantile(values, 0.75)
  
  # Two-sided IQR filter before calculating slope
  iqr <- Q3_percentile - Q1_percentile
  iqr_multiplier <- ANALYSIS_CONFIG$step3$iqr_multiplier
  lower_thr <- Q1_percentile - iqr_multiplier * iqr
  upper_thr <- Q3_percentile + iqr_multiplier * iqr
  # Keep non-outlier values for slope calculation
  non_outlier_mask <- values >= lower_thr & values <= upper_thr
  dates_filtered <- dates[non_outlier_mask]
  values_filtered <- values[non_outlier_mask]
  
  # Get unique x values and their corresponding y means
  unique_dates <- unique(dates_filtered)
  if (length(unique_dates) < 3) {
    return(NULL)
  }
  
  unique_vals <- sapply(unique_dates, function(d) mean(values_filtered[dates_filtered == d]))
  
  # Center and scale the data, then get linear regression (using non-outlier data)
  dates_norm <- (unique_dates - mean(unique_dates)) / sd(unique_dates)
  vals_mean <- mean(unique_vals)
  vals_std <- ifelse(sd(unique_vals) > 0, sd(unique_vals), 1)
  values_norm <- (unique_vals - vals_mean) / vals_std
  
  fit <- lm(values_norm ~ dates_norm)
  slope <- coef(fit)[2]
  r_value <- summary(fit)$r.squared
  p_value <- summary(fit)$coefficients[2, 4]
  
  if (sd(unique_dates) > 0) {
    slope <- slope * (vals_std / sd(unique_dates))
  }
  
  # Near-exceedance should use the most recent 3 years of data
  recent_threshold <- max(dates_filtered, na.rm = TRUE) - 3
  recent_values <- values_filtered[dates_filtered >= recent_threshold]
  if (length(recent_values) == 0) {
    return(NULL)
  }
  Q1_recent <- quantile(recent_values, 0.25, na.rm = TRUE)
  Q3_recent <- quantile(recent_values, 0.75, na.rm = TRUE)

  # Pick the most recent non-null limit and its qualifier (simple, explicit)
  limits <- as.numeric(group$LIMIT_VALUE_STANDARD_UNITS)
  quals  <- as.character(group$LIMIT_VALUE_QUALIFIER_CODE)
  types  <- as.character(group$LIMIT_VALUE_TYPE_CODE)
  dates  <- as.numeric(group$MONITORING_PERIOD_END_DATE_NUMERIC)
  ord <- order(dates)
  limits_sorted <- limits[ord]
  quals_sorted  <- quals[ord]
  types_sorted  <- types[ord]
  valid_idx <- which(!is.na(limits_sorted))
  if (length(valid_idx) == 0) {
    stop(sprintf("No valid limits for group: %s", paste(paste(GROUP_COLS, key_tuple, sep = "="), collapse = ", ")))
  }
  last_pos <- tail(valid_idx, 1)
  latest_limit <- limits_sorted[last_pos]
  chosen_qual <- quals_sorted[last_pos]
  chosen_type <- types_sorted[last_pos]
  
  result <- list(
    slope = slope,
    latest_limit = latest_limit,
    median = as.numeric(median(values_filtered, na.rm = TRUE)),
    qualifier = chosen_qual,
    Q1 = Q1_percentile,
    Q3 = Q3_percentile,
    Q1_recent = as.numeric(Q1_recent),
    Q3_recent = as.numeric(Q3_recent),
    LIMIT_VALUE_TYPE_CODE = as.character(chosen_type),
    LIMIT_SET_SCHEDULE_ID = group$LIMIT_SET_SCHEDULE_ID[1]
  )
  
  # Add grouping columns
  for (i in seq_along(GROUP_COLS)) {
    result[[GROUP_COLS[i]]] <- key_tuple[[i]]
  }
  
  return(result)
}

# Determine flagged facilities (match Python logic)
get_flagged_facilities <- function(facility_records) {
  flagged_all <- list()

  for (rec in facility_records) {
    if (is.null(rec$latest_limit) || is.na(rec$latest_limit)) next
    qualifier <- rec$qualifier
    is_minimum_limit <- qualifier %in% c(">=", ">")

    near_cutoff <- if (qualifier %in% c("<=", "<")) {
      (1 - LIMIT_THRESHOLD) * rec$latest_limit
    } else if (qualifier %in% c(">=", ">")) {
      (1 + LIMIT_THRESHOLD) * rec$latest_limit
    } else {
      NA_real_
    }
    near_exceedance <- if (qualifier %in% c("<=", "<")) {
      rec$Q3 > near_cutoff
    } else if (qualifier %in% c(">=", ">")) {
      rec$Q1 < near_cutoff
    } else {
      FALSE
    }

    # Time-to-limit (Inf if slope missing/zero)
    time_to_limit <- Inf
    if (!is.null(rec$slope) && !is.na(rec$slope) && rec$slope != 0) {
      distance <- if (qualifier %in% c("<=", "<")) rec$latest_limit - rec$median else rec$median - rec$latest_limit
      if (!is.na(distance)) time_to_limit <- abs(distance) / abs(rec$slope)
    }

    if (near_exceedance && time_to_limit <= TIME_TO_LIMIT_YEARS) {
      rec$is_minimum_limit <- is_minimum_limit
      rec$near_cutoff <- near_cutoff
      rec$time_to_limit_years <- time_to_limit
      flagged_all[[length(flagged_all) + 1]] <- rec
    }
  }

  cat(sprintf("%d pairs with Q1/Q3_recent > %f\n", sum(sapply(facility_records, function(r) {
    if (is.null(r$latest_limit) || is.na(r$latest_limit)) return(FALSE)
    if (is.null(r$Q1_recent) || is.null(r$Q3_recent)) return(FALSE)
    q <- r$qualifier
    cutoff <- if (q %in% c("<=", "<")) (1 - LIMIT_THRESHOLD) * r$latest_limit else if (q %in% c(">=", ">")) (1 + LIMIT_THRESHOLD) * r$latest_limit else NA_real_
    if (is.na(cutoff)) return(FALSE)
    if (q %in% c("<=", "<")) r$Q3_recent > cutoff else r$Q1_recent < cutoff
  })), LIMIT_THRESHOLD))
  cat(sprintf("%d pairs with both (and within %d yrs)\n", length(flagged_all), TIME_TO_LIMIT_YEARS))
  if (length(flagged_all) > 0) {
    cat(sprintf("%d facilities affected\n", length(unique(sapply(flagged_all, function(r) r$EXTERNAL_PERMIT_NMBR)))))
  }

  return(flagged_all)
}

# Function to create individual facility-parameter plot
step3_facility_param_plot <- function(npdes_code, param_desc, data) {
  # Extract using bracket notation to avoid any partial matching issues
  dates_raw <- data[["MONITORING_PERIOD_END_DATE_NUMERIC"]]
  values_raw <- data[["DMR_VALUE_STANDARD_UNITS"]]
  
  # Convert to numeric, handling any character/factor types
  dates <- as.numeric(as.character(dates_raw))
  values <- as.numeric(as.character(values_raw))  
  limits <- data[["LIMIT_VALUE_STANDARD_UNITS"]]
  limit_value <- if (length(limits) > 0 && !all(is.na(limits))) limits[!is.na(limits)][1] else NA
  limit_value <- as.numeric(limit_value)
  
  # Get qualifier to determine if this is a minimum or maximum limit
  qualifier <- if ("LIMIT_VALUE_QUALIFIER_CODE" %in% names(data)) {
    data[["LIMIT_VALUE_QUALIFIER_CODE"]][1]
  } else if ("qualifier" %in% names(data)) {
    data[["qualifier"]][1]
  } else {
    "<="
  }
  is_minimum_limit <- qualifier %in% c(">=", ">")  
  unit_desc <- data[["STANDARD_UNIT_DESC"]][1]
  
  # Calculate statistics
  q1 <- quantile(values, 0.25, na.rm = TRUE)
  q3 <- quantile(values, 0.75, na.rm = TRUE)
  
  # Create time series plot - ensure all data is numeric
  # Remove any non-numeric values
  valid_indices <- !is.na(dates) & !is.na(values) & 
                   is.finite(dates) & is.finite(values) &
                   !is.nan(dates) & !is.nan(values)
  
  if (sum(valid_indices) == 0) {
    return(invisible(NULL))
  }
  
  # Create plot_data with explicit numeric conversion and verification
  plot_x <- dates[valid_indices]
  plot_y <- values[valid_indices]
  
  # Final numeric conversion and validation
  plot_x <- as.numeric(as.character(plot_x))
  plot_y <- as.numeric(as.character(plot_y))
  
  # Filter out any remaining non-numeric/invalid values
  final_valid <- !is.na(plot_x) & !is.na(plot_y) & 
                 is.finite(plot_x) & is.finite(plot_y) &
                 !is.nan(plot_x) & !is.nan(plot_y)
  
  plot_x <- plot_x[final_valid]
  plot_y <- plot_y[final_valid]
  
  # Final check if we have any data
  if (length(plot_x) == 0 || length(plot_y) == 0) {
    return(invisible(NULL))
  }
  
  # Create clean data frame with only numeric data
  if (!is.numeric(plot_x) || !is.numeric(plot_y)) {
    return(invisible(NULL))
  }
  plot_data <- data.frame(
    x = plot_x,
    y = plot_y,
    stringsAsFactors = FALSE
  )
  
  # Double-check the data frame
  if (nrow(plot_data) == 0 || !is.numeric(plot_data$x) || !is.numeric(plot_data$y)) {
    return(invisible(NULL))
  }
  
  # Verify plot_data only has x and y columns
  if (length(names(plot_data)) != 2 || !all(names(plot_data) %in% c("x", "y"))) {
    return(invisible(NULL))
  }
  
  # Filter out non-finite values
  if (any(!is.finite(plot_data$x)) || any(!is.finite(plot_data$y))) {
    plot_data <- plot_data[is.finite(plot_data$x) & is.finite(plot_data$y), ]
  }
  
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  
  # explicit column references with .data and auto-legend mapping
  p1 <- ggplot(plot_data, aes(x = .data[["x"]], y = .data[["y"]])) +
    geom_point(aes(color = "Data"), alpha = 0.7, size = 3, show.legend = TRUE) +
    labs(x = "Time", y = unit_desc, 
         title = paste0(param_desc, "\nFacility: ", npdes_code)) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(alpha = 0.3))
  
  # Add trend line
  if (nrow(plot_data) > 1) {
    trend_fit <- lm(y ~ x, data = plot_data)
    x_seq <- seq(min(plot_data$x), max(plot_data$x), length.out = 100)
    trend_line <- data.frame(
      x = as.numeric(x_seq),
      y = as.numeric(predict(trend_fit, newdata = data.frame(x = x_seq))),
      stringsAsFactors = FALSE
    )
    if (is.numeric(trend_line$x) && is.numeric(trend_line$y)) {
      p1 <- p1 + geom_line(data = trend_line, aes(x = .data$x, y = .data$y, color = "Trend", linetype = "Trend"),
                           linewidth = 0.8, show.legend = TRUE)
    }
  }
  
  # Add compliance zones if limit exists
  if (!is.na(limit_value)) {
    y_range <- range(plot_data$y, na.rm = TRUE)
    y_data_min <- y_range[1]
    y_data_max <- y_range[2]
    
    if (is_minimum_limit) {
      # Minimum limit (>=, >): values ABOVE limit are in compliance (e.g., percent removal)
      if (y_data_min >= limit_value) {
        # All data is compliant, but trending toward limit
        y_min <- min(y_data_min * 0.95, limit_value * 0.95)
      } else {
        y_min <- min(y_data_min * 0.95, 0)
      }
      y_max <- max(y_data_max * 1.05, limit_value * 1.05)
      
      zones <- data.frame(
        xmin = -Inf, xmax = Inf,
        ymin = c(limit_value, y_min),
        ymax = c(y_max, limit_value),
        category = factor(c("In Compliance", "Out of Compliance"),
                          levels = c("In Compliance", "Out of Compliance"))
      )
      p1 <- p1 +
        geom_rect(data = zones, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = category),
                  alpha = 0.3, inherit.aes = FALSE, show.legend = TRUE)
    } else {
      # Maximum limit (<=, <): values BELOW limit are in compliance
      y_min <- min(y_data_min * 0.95, limit_value * 0.9)
      if (y_data_max <= limit_value) {
        y_max <- max(y_data_max * 1.1, limit_value * 1.1)
      } else {
        y_max <- max(y_data_max * 1.05, limit_value * 1.05)
      }
      zones <- data.frame(
        xmin = -Inf, xmax = Inf,
        ymin = c(y_min, limit_value),
        ymax = c(limit_value, y_max),
        category = factor(c("In Compliance", "Out of Compliance"),
                          levels = c("In Compliance", "Out of Compliance"))
      )
      p1 <- p1 +
        geom_rect(data = zones, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = category),
                  alpha = 0.3, inherit.aes = FALSE, show.legend = TRUE)
    }
    p1 <- p1 + 
      geom_hline(aes(yintercept = limit_value, color = "Limit"), linetype = "solid", linewidth = 1.5, show.legend = TRUE) +
      coord_cartesian(ylim = c(y_min, y_max))
  } else {
  }
  
  # Mark outliers (two-sided IQR rule)
  iqr <- q3 - q1
  iqr_multiplier <- ANALYSIS_CONFIG$step3$iqr_multiplier
  lower_thr <- q1 - iqr_multiplier * iqr
  upper_thr <- q3 + iqr_multiplier * iqr
  outlier_data <- plot_data %>% filter(y < lower_thr | y > upper_thr)
  
  if (nrow(outlier_data) > 0) {
    p1 <- p1 + 
      geom_point(data = outlier_data, aes(x = x, y = y, shape = "Outliers"),
                 size = 5, color = "red", show.legend = TRUE)
  }

  # Scales for auto legend
  p1 <- p1 +
    scale_color_manual(name = NULL, values = c(Data = "blue", Trend = "black", Limit = "gray")) +
    scale_linetype_manual(name = NULL, values = c(Trend = "dashed")) +
    scale_fill_manual(name = NULL, values = c("In Compliance" = "lightgreen", "Out of Compliance" = "lightcoral")) +
    scale_shape_manual(name = NULL, values = c(Outliers = 8)) +
    guides(shape = guide_legend(override.aes = list(color = "red")))
  
  # Histogram plot - ensure y is numeric
  p2 <- ggplot(plot_data, aes(x = as.numeric(y))) +
    geom_histogram(bins = 20, fill = "blue", alpha = 0.7, color = "black") +
    labs(x = unit_desc, y = "Frequency", title = "Distribution") +
    theme_minimal() +
    theme(plot.title = element_text(size = 10),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(alpha = 0.3))
  
  # Add statistical markers with legend
  p2 <- p2 + 
    geom_vline(xintercept = q1, color = "red", linetype = "dashed", linewidth = 1) +
    annotate("text", x = q1, y = Inf, label = "Q1", color = "red", vjust = -0.5) +
    geom_vline(xintercept = q3, color = "orange", linetype = "dashed", linewidth = 1) +
    annotate("text", x = q3, y = Inf, label = "Q3", color = "orange", vjust = -0.5)
  
  if (!is.na(limit_value)) {
    p2 <- p2 + 
      geom_vline(xintercept = limit_value, color = "gray", 
                linetype = "solid", linewidth = 1) +
      annotate("text", x = limit_value, y = Inf, label = "Limit", color = "gray", vjust = -0.5)
  }
  
  # Create filename (sanitize parameter name)
  safe_param <- gsub(" ", "_", as.character(param_desc))
  safe_param <- gsub("[^A-Za-z0-9_]+", "_", safe_param)
  safe_param <- gsub("_+", "_", safe_param)              # collapse repeats
  filename <- paste0(npdes_code, "_", safe_param, ".png")
  
  combined_plot <- grid.arrange(p1, p2, ncol = 2, widths = c(3, 1))
  save_fig(filename, step = 3, plot_obj = combined_plot, width = 20, height = 6, res = 150)
  
  return(invisible(NULL))
}

# Function to generate all step3 plots
main <- function(save = FALSE, drop_toxicity = FALSE) {
  # Load unique parameter codes from step1 output
  unique_param_codes <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv"))) %>%
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
    all_data <- bind_rows(data_dict)
    filename <- file.path(STEP_DIRS[["3"]], paste0("dmr_all_years_R.csv"))
    write_csv(all_data, filename)
    cat(sprintf("Saved %d records from %d years\n", nrow(all_data), length(data_dict)))
  }
  
  # Filter by unique parameter codes
  filtered_data <- bind_rows(lapply(names(data_dict), function(y) {
    data_dict[[y]] %>% filter(PARAMETER_CODE %in% unique_param_codes)
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
  # Export required variables to cluster workers
  clusterExport(cl, c("GROUP_COLS", "ANALYSIS_CONFIG"))
  
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
  # Uses default thresholds from config
  flagged_facilities <- get_flagged_facilities(facility_records)
  
  # Merge flagged facilities with actual data for plotting (inline plotting)
  if (length(flagged_facilities) > 0) {
    flagged_facilities_df <- bind_rows(lapply(flagged_facilities, as.data.frame))

    # Ensure column types match before joining
    for (col in GROUP_COLS) {
      if (col %in% names(filtered_data) && col %in% names(flagged_facilities_df)) {
        filtered_data[[col]] <- as.character(filtered_data[[col]])
        flagged_facilities_df[[col]] <- as.character(flagged_facilities_df[[col]])
      }
    }

    flagged_data <- filtered_data %>%
      inner_join(flagged_facilities_df, by = GROUP_COLS, suffix = c("", "_flagged"))

    # Attach parameter descriptions once for plotting titles
    param_ref <- suppressMessages(read_csv(file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv"))) %>%
      dplyr::select(PARAMETER_CODE, DMR_PARAMETER_DESC) %>% distinct()
    flagged_data <- flagged_data %>%
      left_join(param_ref, by = "PARAMETER_CODE") %>%
      mutate(PARAMETER_DESC = coalesce(PARAMETER_DESC, DMR_PARAMETER_DESC, as.character(PARAMETER_CODE)))

    # Count parameters per facility (nunique)
    param_counts_list <- list()
    for (rec in flagged_facilities) {
      facility <- rec$EXTERNAL_PERMIT_NMBR
      param <- rec$PARAMETER_CODE
      if (is.null(param_counts_list[[facility]])) param_counts_list[[facility]] <- character(0)
      param_counts_list[[facility]] <- c(param_counts_list[[facility]], param)
    }
    flagged_param_counts <- setNames(sapply(param_counts_list, function(x) length(unique(x))),
                                     names(param_counts_list))

    # Map of counts
    plot_map(flagged_param_counts, 4, step = 3)

    # Generate plots per unique combination (avoid mixing schedule/type/unit/location)
    # Derive plot grouping from GROUP_COLS (minus facility id) + description
    plot_group_cols <- c(setdiff(GROUP_COLS, "EXTERNAL_PERMIT_NMBR"), "PARAMETER_DESC")
    for (npdes_code in unique(flagged_data$EXTERNAL_PERMIT_NMBR)) {
      facility_indices <- which(flagged_data$EXTERNAL_PERMIT_NMBR == npdes_code)
      facility_data <- flagged_data[facility_indices, , drop = FALSE]
      # Group by and split using dplyr
      grouped_list <- facility_data %>% group_by(across(all_of(plot_group_cols))) %>% group_split()
      for (i in seq_along(grouped_list)) {
        param_subset <- grouped_list[[i]]
        if (nrow(param_subset) == 0) next
        param_desc <- param_subset$PARAMETER_DESC[1]
        step3_facility_param_plot(npdes_code, param_desc, param_subset)
      }
    }

    # Bar plot of counts
    param_counts_df <- data.frame(
      Facility = names(flagged_param_counts),
      Parameters = unlist(flagged_param_counts)
    ) %>%
      arrange(Parameters) %>%
      mutate(Facility = factor(Facility, levels = Facility))

    plot_barh(
      param_counts_df,
      x_col = "Parameters",
      y_col = "Facility",
      xlabel = "Number of Parameters with Slope and Near-Exceedance",
      path = "facilities_summary.png",
      step = 3
    )

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
