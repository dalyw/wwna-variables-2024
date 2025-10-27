# helper_functions.R
# Updated to match Python helper_functions.py methodology
# with support from Claude 4.0

library(readr)
library(dplyr)
library(lubridate)
library(jsonlite)
library(ggplot2)
library(gridExtra)
library(sf)
library(raster)
library(viridis)

# Analysis configuration
analysis_range <- 2014:2023

# Import WWNA facilities list
WWNA_LIST <- read_csv('data/wwna_list/NPDES+WDR Facilities List_20240906.csv')
NPDES_FROM_WWNA_LIST <- WWNA_LIST %>%
  filter(!is.na(`NPDES # CA#`)) %>%
  pull(`NPDES # CA#`) %>%
  unique()

# Column name constants for aggregated results
AGG_STRINGS <- list(
  "3" = list(
    COUNT = "Parameters with Slope and Near Exceedance: Number of Parameters",
    PARAM = "Parameters with Slope and Near Exceedance: List of Parameters"
  ),
  "4" = list(
    COUNT = "Discharges to Impaired and Not Limited: Number of Parameters",
    PARAM = "Discharges to Impaired and Not Limited: List of Parameters"
  )
)

# Step directories
STEP_DIRS <- c(
  "1" = "processed_data/step1",
  "2" = "processed_data/step2",
  "3" = "processed_data/step3",
  "4" = "processed_data/step4"
)

# File configuration
FILE_CONFIGS <- jsonlite::fromJSON('wwna_variables_2024/file_configs.json', simplifyVector = FALSE)

# ESMR Resource IDs
ESMR_RESOURCE_IDS <- c(
  "2014" = "c0f64b3f-d921-4eb9-aa95-af1827e5033e",
  "2015" = "81c399d4-f661-4808-8e6b-8e543281f1c9",
  "2016" = "aacfe728-f063-452c-9dca-63482cc994ad",
  "2017" = "44d1f39c-f21b-4060-8225-c175eaea129d",
  "2018" = "bb3b3d85-44eb-4813-bbf9-ea3a0e623bb7",
  "2019" = "2eaa2d55-9024-431e-b902-9676db949174",
  "2020" = "4fa56f3f-7dca-4dbd-bec4-fe53d5823905",
  "2021" = "28d3a164-7cec-4baf-9b11-7a9322544cd6",
  "2022" = "8c6296f7-e226-42b7-9605-235cd33cdee2",
  "2023" = "65eb7023-86b6-4960-b714-5f6574d43556",
  "2024" = "7adb8aea-62fb-412f-9e67-d13b0729222f",
  "2025" = "176a58bf-6f5d-4e3f-9ed9-592a509870eb"
)

get_data_file_path <- function(data_type, year = NULL) {
  config <- FILE_CONFIGS[[data_type]]
  base_dir <- if (!is.null(config$base_dir_type)) config$base_dir_type else data_type
  dir_path <- paste0('data/', tolower(base_dir))
  
  if (!is.null(config$subdir_pattern) && !is.null(year)) {
    dir_path <- file.path(dir_path, gsub('\\{year\\}', year, config$subdir_pattern))
  }
  
  if (!is.null(config$file_pattern)) {
    filename <- if (!is.null(year)) gsub('\\{year\\}', year, config$file_pattern) else config$file_pattern
    return(file.path(dir_path, filename))
  }
  
  return(dir_path)
}

load_data <- function(data_type, year = NULL, drop_toxicity = FALSE) {
  config <- FILE_CONFIGS[[data_type]]
  file_path <- get_data_file_path(data_type, year)
  
  # Handle skiprows
  skiprows <- 0
  if (!is.null(config$skiprows)) {
    if (is.list(config$skiprows) && !is.null(year)) {
      skiprows <- ifelse(is.null(config$skiprows[[as.character(year)]]), 0, config$skiprows[[as.character(year)]])
    } else {
      skiprows <- config$skiprows
    }
  }
  
  suppressMessages({
    data <- read_csv(file_path, skip = skiprows, show_col_types = FALSE)
  })
  
  # Drop NA on specified columns
  if (!is.null(config$dropna)) {
    for (col in config$dropna) {
      if (col %in% names(data)) {
        data <- data %>% filter(!is.na(.data[[col]]))
      }
    }
  }
  
  # Drop NOT NA on specified columns
  if (!is.null(config$drop_notna)) {
    for (col in config$drop_notna) {
      if (col %in% names(data)) {
        data <- data %>% filter(is.na(.data[[col]]))
      }
    }
  }
  
  # Apply filters
  if (!is.null(config$filters)) {
    for (col in names(config$filters)) {
      if (col %in% names(data)) {
        filter_values <- config$filters[[col]]
        if (is.character(filter_values)) {
          data <- data %>% filter(.data[[col]] == filter_values)
        } else if (is.list(filter_values) || is.atomic(filter_values)) {
          data <- data %>% filter(.data[[col]] %in% filter_values)
        }
      }
    }
  }
  
  # Strip leading zeros
  if (!is.null(config$strip_leading_zeros)) {
    col <- config$strip_leading_zeros
    if (col %in% names(data)) {
      data[[col]] <- sub("^0+", "", data[[col]])
    }
  }
  
  # Mark toxicity
  if (!is.null(config$mark_toxicity)) {
    mt <- config$mark_toxicity
    if (mt$pattern == "startswith" && mt$column %in% names(data)) {
      for (val in mt$values) {
        data[[mt$column]][startsWith(data[[mt$column]], val)] <- mt$set_value
      }
    }
  }
  
  # Explode (for handling list columns - not needed for WW_SURVEILLANCE which is already exploded)
  # Note: Python's explode only works on actual list columns, which this CSV doesn't have
  # The "explode" config here is just for documentation - the data is already in correct format
  if (!is.null(config$explode)) {
    # Skip explode for WW_SURVEILLANCE - data is already in correct format
    if (data_type != "WW_SURVEILLANCE") {
      for (col in config$explode) {
        if (col %in% names(data)) {
          # Only unnest if the column actually contains lists
          sample_val <- data[[col]][1]
          if (is.list(sample_val) && length(sample_val) > 1) {
            data <- data %>% tidyr::unnest_wider(!!sym(col), names_sep = "_")
          }
        }
      }
    }
  }
  
  # Drop duplicates
  if (!is.null(config$drop_duplicates)) {
    dup_cols <- if (is.list(config$drop_duplicates)) unlist(config$drop_duplicates) else config$drop_duplicates
    
    # For WW_SURVEILLANCE, we need to get the old column name before rename
    actual_cols <- dup_cols
    if (data_type == "WW_SURVEILLANCE" && "epaid" %in% dup_cols) {
      # epaid exists in the data before rename, so keep it as is
      actual_cols <- dup_cols
    }
    
    # Only use columns that exist in the data
    actual_cols <- actual_cols[actual_cols %in% names(data)]
    
    if (length(actual_cols) > 0) {
      data <- data %>%
        group_by(across(all_of(actual_cols))) %>%
        slice(1) %>%
        ungroup()
    }
  }
  
  # Rename
  if (!is.null(config$rename)) {
    # In R, rename(oldname = newname) format
    # But config has newname: oldname format, so we need to flip it
    rename_list <- list()
    for (oldname in names(config$rename)) {
      newname <- config$rename[[oldname]]
      if (oldname %in% names(data)) {
        rename_list[[newname]] <- as.name(oldname)
      }
    }
    if (length(rename_list) > 0) {
      data <- data %>% rename(!!!rename_list)
    }
  }
  
  # Apply DMR-specific transformations
  if (data_type == "DMR") {
    if (drop_toxicity && "PARAMETER_DESC" %in% names(data)) {
      data <- data %>% filter(!str_detect(PARAMETER_DESC, "Toxicity"))
    }
    
    if ("MONITORING_PERIOD_END_DATE" %in% names(data)) {
      data <- data %>%
        mutate(
          MONITORING_PERIOD_END_DATE = parse_date_time(MONITORING_PERIOD_END_DATE, 
                                                        orders = c("mdy", "ymd", "dmy"),
                                                        quiet = TRUE),
          MONITORING_PERIOD_END_DATE_NUMERIC = ifelse(
            !is.na(MONITORING_PERIOD_END_DATE),
            year(MONITORING_PERIOD_END_DATE) + month(MONITORING_PERIOD_END_DATE) / 12 + day(MONITORING_PERIOD_END_DATE) / 365,
            NA_real_
          )
        )
    }
    
    cat(sprintf('%d %s: %d records, %d facilities\n', 
                year, data_type, nrow(data), n_distinct(data$EXTERNAL_PERMIT_NMBR)))
  }
  
  # Apply WWNA facilities list filter for DMR and LIMITS
  if (data_type %in% c("DMR", "LIMITS")) {
    data <- data %>% filter(EXTERNAL_PERMIT_NMBR %in% NPDES_FROM_WWNA_LIST)
  }
  
  if (data_type == "LIMITS") {
    cat(sprintf('%d has %d limits, %d unique permits\n', 
                year, nrow(data), n_distinct(data$EXTERNAL_PERMIT_NMBR)))
  }
  
  return(data)
}

aggregate_flagged_params <- function(data, group_col, value_col, col_names) {
  if (nrow(data) == 0) {
    result <- data.frame(
      col1 = character(0),
      col2 = integer(0),
      col3 = character(0)
    )
    names(result) <- c(group_col, col_names$COUNT, col_names$PARAM)
    return(result)
  }
  
  agg_data <- data %>%
    group_by(!!sym(group_col)) %>%
    summarise(
      count = n(),
      values = paste(unique(!!sym(value_col)), collapse = ", ")
    ) %>%
    ungroup()
  
  names(agg_data) <- c(group_col, col_names$COUNT, col_names$PARAM)
  
  return(agg_data)
}

setup_fig <- function(figsize = c(10, 6)) {
  list(width = figsize[1], height = figsize[2])
}

save_fig <- function(path, step = NULL) {
  full_path <- if (!is.null(step)) {
    file.path(STEP_DIRS[[as.character(step)]], path)
  } else {
    path
  }
  
  # Create directory if it doesn't exist
  dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
  
  ggsave(full_path, width = 10, height = 6, units = "in")
  dev.off()
}

plot_barh <- function(data, x_col, y_col, xlabel, title, figsize = c(12, 6), path = NULL, step = NULL) {
  # Create bar plot
  p <- ggplot(data, aes_string(x = x_col, y = y_col)) +
    geom_barh(stat = "identity") +
    xlab(xlabel) +
    ggtitle(title) +
    theme_minimal()
  
  if (!is.null(path)) {
    save_fig(path, step)
  }
  
  return(p)
}

plot_map <- function(num_params_per_facility, label_threshold, step = 3) {
  ca_counties <- st_read('data/ca_counties/CA_Counties.shp')
  
  # Create DataFrame with facility IDs and merge
  facilities_df <- data.frame("NPDES # CA#" = names(num_params_per_facility)) %>%
    left_join(WWNA_LIST, by = "NPDES # CA#")
  
  facilities_gdf <- st_as_sf(
    facilities_df,
    coords = c("LONGITUDE DECIMAL DEGREES", "LATITUDE DECIMAL DEGREES"),
    crs = 4326
  )
  
  # Reproject if needed
  ca_counties <- st_transform(ca_counties, st_crs(4326))
  facilities_gdf <- st_transform(facilities_gdf, st_crs(4326))
  
  # Map param counts
  facilities_gdf$param_count <- sapply(facilities_gdf$`NPDES # CA#`, 
    function(x) {
      x_char <- as.character(x)
      if (x_char %in% names(num_params_per_facility)) {
        num_params_per_facility[[x_char]]
      } else {
        NA
      }
    }
  )
  
  ggplot() +
    geom_sf(data = ca_counties, fill = "lightgray", color = "white") +
    geom_sf(data = facilities_gdf, aes(color = param_count, size = param_count)) +
    scale_color_viridis_c() +
    theme_minimal() +
    labs(title = "Facilities with Parameters Having Slope and Near Exceedance",
         color = "Number of Parameters",
         size = "Number of Parameters")
  
  save_fig("figures_R/facilities_map.png", step)
}
