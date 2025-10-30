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

# Load analysis configuration
ANALYSIS_CONFIG <- jsonlite::fromJSON('wwna_variables_2024/analysis_config.json', simplifyVector = FALSE)
year_range_config <- ANALYSIS_CONFIG$year_range
analysis_range <- seq(as.numeric(year_range_config[[1]]), as.numeric(year_range_config[[2]]))

# Import WWNA facilities list
WWNA_LIST_PATH <- ANALYSIS_CONFIG$wwna_list_path
WWNA_LIST <- suppressMessages(read_csv(WWNA_LIST_PATH))
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
ESMR_RESOURCE_IDS <- FILE_CONFIGS$ESMR$year_config

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

  # Convert RNC_RESOLUTION_CODE to character to handle 'B' values
  if ("RNC_RESOLUTION_CODE" %in% names(data)) {
    data$RNC_RESOLUTION_CODE <- as.character(data$RNC_RESOLUTION_CODE)
  }
  
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
    # PARAMETER_DESC and MONITORING_PERIOD_END_DATE are guaranteed for DMR (from file_configs)
    if (drop_toxicity) {
      data <- data %>% filter(!str_detect(PARAMETER_DESC, "Toxicity"))
    }
    
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

save_fig <- function(path, step, plot_obj = NULL, width = 10, height = 6, res = 150) {
  full_path <- file.path(STEP_DIRS[[as.character(step)]], "figures_R", path)
  
  # Save ggplot objects
  if (!is.null(plot_obj)) {
    ggsave(full_path, plot_obj, width = width, height = height, units = "in", dpi = res)
  }
  
  return(full_path)
}

plot_barh <- function(data, x_col, y_col, xlabel, figsize = c(12, 6), path = NULL, step = NULL, ylabel = NULL) {
  # Ensure data is a clean data frame
  if (!is.data.frame(data)) {
    data <- as.data.frame(data, stringsAsFactors = FALSE)
  }
  
  # Verify columns exist
  if (!x_col %in% names(data) || !y_col %in% names(data)) {
    return(NULL)
  }
  
  # Ensure x_col is numeric (the values/width of bars)
  if (!is.numeric(data[[x_col]])) {
    data[[x_col]] <- as.numeric(as.character(data[[x_col]]))
  }
  
  # Ensure y_col (Facility) is treated as factor for proper ordering
  # For horizontal bars, y_col should be discrete (factor)
  if (!is.factor(data[[y_col]])) {
    # Preserve order by converting to factor with current order
    data[[y_col]] <- factor(data[[y_col]], levels = rev(unique(data[[y_col]])))
  }
  
  # Create horizontal bar chart
  # discrete categories on y-axis, numeric values on x-axis  
  data_clean <- data.frame(
    y_axis = factor(data[[y_col]], levels = unique(data[[y_col]])),  # Discrete categories
    x_axis = as.numeric(as.character(data[[x_col]])),  # Numeric values
    stringsAsFactors = FALSE
  )
  
  p <- ggplot(data_clean, aes(x = y_axis, y = x_axis)) +
    geom_col() +
    coord_flip() +
    xlab(ifelse(is.null(ylabel), "", ylabel)) +
    ylab(xlabel) +
    theme_minimal()
  
  # Add labels to bars - after coord_flip, categories are on horizontal axis
  for (i in 1:nrow(data_clean)) {
    bar_height <- data_clean$x_axis[i]  # The numeric value (bar width/height)
    category_level <- data_clean$y_axis[i]  # The category/factor level
    p <- p + annotate("text", 
                     x = category_level,  # The category (on horizontal axis after flip)
                     y = bar_height,      # At the end of the bar
                     label = as.character(bar_height), 
                     hjust = -0.1, vjust = 0.5, size = 2.5)
  }
  
  if (!is.null(path)) {
    save_fig(path, step)
  }
  
  return(p)
}

plot_map <- function(num_params_per_facility, label_threshold, step = 3) {
  source_crs <- "EPSG:3857"
  target_crs <- st_crs(3310)  # EPSG:3310 (NAD83 California Albers)
  ca_counties <- st_read("data/ca_counties/CA_Counties.shp")
  
  # Counties are in source CRS - set if not already set
  source_crs_num <- 3857
  if (is.na(st_crs(ca_counties))) {
    st_crs(ca_counties) <- st_crs(source_crs_num)
  }
  
  # Transform counties to target CRS
  ca_counties_proj <- st_transform(ca_counties, target_crs)
  
  # Create DataFrame with facility IDs and merge
  facility_ids <- names(num_params_per_facility)
  facilities_df <- data.frame(facility_ids, stringsAsFactors = FALSE)
  names(facilities_df) <- "NPDES # CA#"
  
  facilities_df <- facilities_df %>%
    left_join(WWNA_LIST, by = "NPDES # CA#", relationship = "many-to-many")
  
  # Filter out facilities with missing coordinates
  facilities_df <- facilities_df %>%
    filter(!is.na(`LONGITUDE DECIMAL DEGREES`) & !is.na(`LATITUDE DECIMAL DEGREES`))
  
  # Create facilities GDF from lat/lon (EPSG:4326)
  facilities_gdf <- st_as_sf(
    facilities_df,
    coords = c("LONGITUDE DECIMAL DEGREES", "LATITUDE DECIMAL DEGREES"),
    crs = 4326
  )
  
  # Transform facilities to target CRS for plotting
  facilities_gdf_proj <- st_transform(facilities_gdf, target_crs)
  
  # Map param counts
  facilities_gdf_proj$param_count <- sapply(facilities_gdf_proj$`NPDES # CA#`, 
    function(x) {
      x_char <- as.character(x)
      if (x_char %in% names(num_params_per_facility)) {
        num_params_per_facility[[x_char]]
      } else {
        NA
      }
    }
  )
  
  # Get bounds for axis limits
  county_bounds <- st_bbox(ca_counties_proj)
  
  # Create plot
  p <- ggplot() +
    geom_sf(data = ca_counties_proj, fill = "lightgray", 
            color = "white", linewidth = 0.5) +
    geom_sf(data = facilities_gdf_proj, aes(color = param_count, size = param_count), 
            inherit.aes = FALSE) +
    scale_color_viridis_c(name = "# Parameters Flagged") +
    scale_size_continuous(name = "# Parameters Flagged", guide = "none") +
    coord_sf(xlim = c(county_bounds[["xmin"]], county_bounds[["xmax"]]),
             ylim = c(county_bounds[["ymin"]], county_bounds[["ymax"]]),
             expand = FALSE) +
    theme_minimal() +
    theme(axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank()) +
    labs(title = "# Parameters Flagged")
  
  # Save plot
  save_fig("facilities_map.png", step = step, plot_obj = p, 
         width = 8, height = 5)
}
