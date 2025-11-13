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
library(scales)

select <- dplyr::select  # Ensure dplyr::select is used (raster masks select)

YEAR_RANGE <- c(2015, 2025)
analysis_range <- seq(YEAR_RANGE[1], YEAR_RANGE[2])

# Load analysis configuration
ANALYSIS_CONFIG <- jsonlite::fromJSON('wwna_variables_2024/analysis_config.json', simplifyVector = FALSE)

# Helper functions to return regular tibbles (not spec_tbl_df) for csv
read_csv_tibble <- function(...) {
  suppressMessages(read_csv(...)) %>% as_tibble()
}

# Helper to ensure regular tibble after dplyr operations
as_regular_tibble <- function(x) {
  # Convert to data.frame first to strip any special classes, then to tibble
  if (inherits(x, "data.frame")) {
    class(x) <- "data.frame"
  }
  as_tibble(x)
}

# WWNA FACILITY LIST
WWNA_LIST_PATH <- "data/wwna_list/NPDES+WDR Facilities List_20240906.csv"
WWNA_LIST <- read_csv_tibble(WWNA_LIST_PATH)
NPDES_FROM_WWNA_LIST <- WWNA_LIST %>%
  filter(!is.na(`NPDES # CA#`)) %>%
  pull(`NPDES # CA#`) %>%
  unique()
cat(sprintf("%d of %d WWNA facilities have NPDES\n", 
            length(NPDES_FROM_WWNA_LIST), nrow(WWNA_LIST)))

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

# Path constants for processed data directories
STEP_DIRS <- list()
for (i in 1:4) {
  STEP_DIRS[[as.character(i)]] <- paste0("processed_data/step", i)
}

FILE_CONFIGS <- jsonlite::fromJSON('wwna_variables_2024/file_configs.json', simplifyVector = FALSE)

# Data download configuration
ESMR_RESOURCE_IDS <- FILE_CONFIGS$ESMR$year_config
names(ESMR_RESOURCE_IDS) <- as.character(as.integer(names(ESMR_RESOURCE_IDS)))

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

load_data <- function(data_type, year = NULL, drop_toxicity = FALSE, rename = TRUE, dropna = TRUE) {
  config <- FILE_CONFIGS[[data_type]]
  file_path <- get_data_file_path(data_type, year)
  
  # Read data with configured columns and dtypes
  # Exclude date columns from dtype dict since parse_dates will handle them
  parse_dates_list <- if (!is.null(config$parse_dates)) config$parse_dates else character(0)
  drop_notna_list <- if (!is.null(config$drop_notna)) config$drop_notna else character(0)
  dtype_dict <- if (!is.null(config$dtype)) config$dtype else list()
  
  # Remove date columns from dtype so parse_dates can work
  for (date_col in parse_dates_list) {
    dtype_dict[[date_col]] <- NULL
  }
  
  # Handle skiprows
  skiprows <- 0
  if (!is.null(config$skiprows)) {
    if (is.list(config$skiprows) && !is.null(year)) {
      skiprows <- ifelse(is.null(config$skiprows[[as.character(year)]]), 0, config$skiprows[[as.character(year)]])
    } else {
      skiprows <- config$skiprows
    }
  }
  
  # Build usecols_list (col_select in readr)
  usecols_list <- c(names(dtype_dict), parse_dates_list, drop_notna_list)
  
  # Read separator (for readr, default is comma for read_csv)
  separator <- if (!is.null(config$separator)) config$separator else ","
  
  # Read data with specified columns
  suppressMessages({
    if (length(usecols_list) > 0) {
      # Convert dtype strings to R col_types specification
      col_types_spec <- list()
      for (col in names(dtype_dict)) {
        dtype_str <- dtype_dict[[col]]
        if (dtype_str == "str") {
          col_types_spec[[col]] <- col_character()
        } else if (dtype_str == "float") {
          col_types_spec[[col]] <- col_double()
        } else if (dtype_str == "int") {
          col_types_spec[[col]] <- col_integer()
        } else if (dtype_str == "bool") {
          col_types_spec[[col]] <- col_logical()
        } else {
          col_types_spec[[col]] <- col_guess()
        }
      }
      # Add date columns
      # DMR dates are in "mdy" format (e.g., "12/31/2024"), read as character then parse
      for (date_col in parse_dates_list) {
        if ((data_type == "DMR" && date_col == "MONITORING_PERIOD_END_DATE") ||
            (data_type == "LIMITS" && date_col %in% c("LIMIT_BEGIN_DATE", "LIMIT_END_DATE"))) {
          # Read as character, will parse manually (col_datetime() doesn't handle mdy format)
          col_types_spec[[date_col]] <- col_character()
        } else {
          # Other date columns can be parsed as datetime
          col_types_spec[[date_col]] <- col_datetime()
        }
      }
      # Add drop_notna columns (read as character to check for NA)
      for (col in drop_notna_list) {
        col_types_spec[[col]] <- col_character()
      }
      
      col_types <- do.call(cols_only, col_types_spec)
      # Use read_delim if separator is not comma, otherwise use read_csv
      if (separator != ",") {
        data <- read_delim(file_path, delim = separator, skip = skiprows, col_types = col_types,
                          locale = locale(date_names = "en"))
      } else {
        data <- read_csv(file_path, skip = skiprows, col_types = col_types, 
                         locale = locale(date_names = "en"))
      }
    } else {
      # No column selection - read all columns
      if (separator != ",") {
        data <- read_delim(file_path, delim = separator, skip = skiprows, show_col_types = FALSE)
      } else {
        data <- read_csv(file_path, skip = skiprows, show_col_types = FALSE)
      }
    }
  })
  
  data <- as_tibble(data)  # regular tibble (not spec_tbl_df)
  
  # Automatically coerce numeric columns (float/int) to numeric
  if (!is.null(config$dtype)) {
    numeric_cols <- names(config$dtype)[config$dtype %in% c("float", "int")]
    numeric_cols <- intersect(numeric_cols, names(data))
    for (col in numeric_cols) {
      data[[col]] <- suppressWarnings(as.numeric(data[[col]]))
    }
  }

  # dropna: drop rows where column IS NA or empty string
  dropna_cols <- intersect(if (is.null(config$dropna)) character(0) else config$dropna, names(data))
  if (length(dropna_cols) > 0 && dropna) {
    # Drop NaN values
    for (col in dropna_cols) {
      data <- data %>% filter(!is.na(.data[[col]]))
    }
    # Then drop empty strings (after converting to string to handle mixed types)
    for (col in dropna_cols) {
      empty_mask <- trimws(as.character(data[[col]])) == ""
      if (any(empty_mask, na.rm = TRUE)) {
        data <- data %>% filter(!empty_mask)
      }
    }
  }
  
  # drop_notna: drop rows where column IS NOT NA, then drop the column itself
  for (col in drop_notna_list) {
    if (col %in% names(data)) {
      data <- data %>% filter(is.na(.data[[col]]))
    }
  }
  # Drop the columns themselves
  for (col in drop_notna_list) {
    if (col %in% names(data)) {
      data <- data %>% dplyr::select(-all_of(col))
    }
  }
  
  # Apply filters - handle wildcard matching with * prefix or suffix
  if (!is.null(config$filters)) {
    for (col in names(config$filters)) {
      if (col %in% names(data)) {
        filter_values <- config$filters[[col]]
        if (is.list(filter_values) || is.atomic(filter_values)) {
          mask <- rep(FALSE, nrow(data))
          for (fv in filter_values) {
            fv_str <- as.character(fv)
            if (substr(fv_str, nchar(fv_str), nchar(fv_str)) == "*") {
              # Match strings starting with pattern (remove * suffix)
              pattern <- substr(fv_str, 1, nchar(fv_str) - 1)
              mask <- mask | startsWith(as.character(data[[col]]), pattern)
            } else if (substr(fv_str, 1, 1) == "*") {
              # Match strings ending with pattern (remove * prefix)
              pattern <- substr(fv_str, 2, nchar(fv_str))
              mask <- mask | endsWith(as.character(data[[col]]), pattern)
            } else {
              # Exact matching
              mask <- mask | (trimws(as.character(data[[col]])) == fv_str)
            }
          }
          data <- data %>% filter(mask)
        }
      }
    }
  }
  
  # Apply transformations from config
  if (!is.null(config$strip_leading_zeros)) {
    transform <- config$strip_leading_zeros
    if (transform %in% names(data)) {
      data[[transform]] <- sub("^0+", "", data[[transform]])
    }
  } else if (!is.null(config$mark_toxicity)) {
    transform <- config$mark_toxicity
    if (transform$pattern == "startswith" && transform$column %in% names(data)) {
      # Check if string starts with any of the values (like Python's tuple)
      mask <- rep(FALSE, nrow(data))
      for (val in transform$values) {
        mask <- mask | startsWith(as.character(data[[transform$column]]), val)
      }
      data[[transform$column]][mask] <- transform$set_value
    }
  }
  
  # Apply post-processing (explode, drop_duplicates)
  if (!is.null(config$explode)) {
    for (col in config$explode) {
      if (col %in% names(data)) {
        # Check if column contains lists (like Python's isinstance(x, list).any())
        # In R, check if any value is a list
        if (any(sapply(data[[col]], is.list))) {
          data <- data %>% tidyr::unnest_longer(!!sym(col))
        }
      }
    }
  }
  
  drop_dup_cols <- config$drop_duplicates
  if (!is.null(drop_dup_cols)) {
    # Ensure drop_dup_cols is a character vector (JSON arrays become lists in R)
    drop_dup_cols <- unlist(drop_dup_cols)
    actual_cols <- drop_dup_cols[drop_dup_cols %in% names(data)]
    if (length(actual_cols) > 0) {
      data <- data %>% distinct(across(all_of(actual_cols)), .keep_all = TRUE)
    }
  }
  
  # Apply renames from config to df and parse_dates list
  rename_map <- if (!is.null(config$rename)) config$rename else list()
  if (length(rename_map) > 0 && rename) {
    # In R, rename(newname = oldname) format
    # Config has newname: oldname format
    rename_list <- list()
    for (oldname in names(rename_map)) {
      newname <- rename_map[[oldname]]
      if (oldname %in% names(data)) {
        rename_list[[newname]] <- as.name(oldname)
      }
    }
    if (length(rename_list) > 0) {
      data <- data %>% rename(!!!rename_list)
    }
    # Update parse_dates_list after rename
    parse_dates_list <- sapply(parse_dates_list, function(col) {
      if (col %in% names(rename_map)) rename_map[[col]] else col
    })
  }
  
  if (data_type == "DMR") {  # Apply DMR-specific transformations
    if (drop_toxicity && "PARAMETER_DESC" %in% names(data)) {
      data <- data %>% filter(!grepl("Toxicity", PARAMETER_DESC))
    }
  }
  
  if (data_type %in% c("ESMR", "DMR", "LIMITS")) {  # Create numeric date columns
    for (date_col in parse_dates_list) {
      # Parse date columns - DMR dates are read as character, others as datetime
      if (!inherits(data[[date_col]], c("POSIXct", "POSIXt", "Date"))) {
        # Parse using parse_date_time (handles mdy format for DMR)
        data[[date_col]] <- parse_date_time(data[[date_col]], 
                                             orders = c("mdy", "ymd", "dmy", "ymd HMS", "mdy HMS"),
                                             quiet = TRUE)
      }
      
      numeric_col <- paste0(date_col, "_NUMERIC")
      # Create numeric date: year + month/12 + day/365
      data[[numeric_col]] <- ifelse(
        !is.na(data[[date_col]]),
        year(data[[date_col]]) + month(data[[date_col]]) / 12 + day(data[[date_col]]) / 365,
        NA_real_
      )
    }
  }
  
  # Apply WWNA facilities list filter for LIMITS
  if (data_type == "LIMITS") {
    data <- data %>% filter(EXTERNAL_PERMIT_NMBR %in% NPDES_FROM_WWNA_LIST)
    unique_count <- n_distinct(data$EXTERNAL_PERMIT_NMBR)
    cat(sprintf("%d has %d limits, %d unique permits\n", 
                year, nrow(data), unique_count))
  }
  
  return(data)
}

aggregate_flags <- function(data, group_col, value_col, col_names) {
  agg_data <- data %>%
    group_by(!!sym(group_col)) %>%
    summarise(
      count = n(),
      values = paste(unique(!!sym(value_col)), collapse = ", "),
      .groups = "drop"
    ) %>%
    ungroup()
  
  names(agg_data) <- c(group_col, col_names$COUNT, col_names$PARAM)
  
  # Fill NA values with defaults
  agg_data[[col_names$COUNT]] <- ifelse(is.na(agg_data[[col_names$COUNT]]), 0, agg_data[[col_names$COUNT]])
  agg_data[[col_names$COUNT]] <- as.integer(agg_data[[col_names$COUNT]])
  agg_data[[col_names$PARAM]] <- ifelse(is.na(agg_data[[col_names$PARAM]]), "", agg_data[[col_names$PARAM]])
  
  return(agg_data)
}

setup_fig <- function(figsize = c(10, 6)) {
  list(width = figsize[1], height = figsize[2])
}

save_fig <- function(path, step, plot_obj = NULL, width = 10, height = 6, res = 150) {
  fig_dir <- file.path(STEP_DIRS[[as.character(step)]], "figures_R")
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Handle paths that might contain subdirectories (create them if needed)
  full_path <- file.path(fig_dir, path)
  path_dir <- dirname(full_path)
  if (path_dir != fig_dir) {
    dir.create(path_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  # Save plot objects (handles both ggplot and grid.arrange objects)
  if (!is.null(plot_obj)) {
    # Use png() to explicitly save as PNG (not PDF)
    png(full_path, width = width, height = height, units = "in", res = res)
    # Handle grid.arrange (gtable) objects vs ggplot objects
    plot_type <- if (inherits(plot_obj, "gtable") || inherits(plot_obj, "gTree") || inherits(plot_obj, "grob")) {
      # Ensure grid package is available
      if (!requireNamespace("grid", quietly = TRUE)) {
        stop("grid package required for saving gtable objects")
      }
      grid::grid.newpage()
      grid::grid.draw(plot_obj)
      "gtable"
    } else if (inherits(plot_obj, "ggplot")) {
      print(plot_obj)
      "ggplot"
    } else {
      # Try grid.draw for any grid object
      tryCatch({
        grid::grid.newpage()
        grid::grid.draw(plot_obj)
        "grid_object"
      }, error = function(e) {
        print(plot_obj)
        "fallback_print"
      })
    }
    dev.off()  # Close PNG device
    message(sprintf("Saved %s plot to %s", plot_type, full_path))
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
    save_fig(path, step, plot_obj = p, width = figsize[1], height = figsize[2])
  }
  
  return(p)
}

plot_map <- function(num_params_per_facility, label_threshold, step = 3) {
  source_crs <- st_crs(3857)  # EPSG:3857
  target_crs <- st_crs(3310)  # EPSG:3310 (NAD83 California Albers)
  ca_counties <- st_read("data/ca_counties/CA_Counties.shp", quiet = TRUE)
  
  # Counties are in source CRS, transform to target CRS
  # Note: st_set_crs sets CRS without transforming (like Python's set_crs with allow_override=True)
  if (is.na(st_crs(ca_counties))) {
    st_crs(ca_counties) <- source_crs
  }
  ca_counties_proj <- st_transform(ca_counties, target_crs)
  
  # Create DataFrame with facility IDs and merge
  facility_names <- names(num_params_per_facility)
  if (is.null(facility_names)) {
    stop("num_params_per_facility must be a named vector")
  }
  facilities_with_coords_merged <- tibble(
    `NPDES # CA#` = facility_names
  ) %>%
    left_join(WWNA_LIST, by = "NPDES # CA#", relationship = "many-to-many") %>%
    filter(!is.na(`LONGITUDE DECIMAL DEGREES`), !is.na(`LATITUDE DECIMAL DEGREES`))
  
  # Create facilities GDF from lat/lon (EPSG:4326)
  facilities_gdf <- st_as_sf(
    facilities_with_coords_merged,
    coords = c("LONGITUDE DECIMAL DEGREES", "LATITUDE DECIMAL DEGREES"),
    crs = 4326
  )
  
  # Convert facilities to projected CRS for plotting
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
  
  # Setup colormap
  param_count_min <- min(facilities_gdf_proj$param_count, na.rm = TRUE)
  param_count_max <- max(facilities_gdf_proj$param_count, na.rm = TRUE)
  
  # Create base plot
  p <- ggplot() +
    geom_sf(data = ca_counties_proj, fill = "lightgray", 
            color = "white", linewidth = 0.5) +
    geom_sf(data = facilities_gdf_proj, aes(color = param_count), 
            inherit.aes = FALSE) +
    scale_color_viridis_c(name = "# Parameters Flagged",
                         limits = c(param_count_min, param_count_max),
                         option = "viridis") +
    coord_sf(xlim = c(county_bounds[["xmin"]], county_bounds[["xmax"]]),
             ylim = c(county_bounds[["ymin"]], county_bounds[["ymax"]]),
             expand = FALSE) +
    theme_minimal() +
    theme(axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank())
  
  # Add facility labels
  top_facilities <- facilities_gdf_proj %>%
    filter(param_count >= label_threshold) %>%
    arrange(desc(param_count)) %>%
    head(10)
  
  if (nrow(top_facilities) > 0) {
    # Sort by projected y coordinates for vertical ordering
    # Extract Y coordinates for sorting
    coords_all <- st_coordinates(top_facilities)
    # Ensure coords_all is a matrix with at least one row
    if (is.vector(coords_all)) {
      coords_all <- matrix(coords_all, nrow = 1, ncol = length(coords_all))
    }
    top_facilities_sorted <- top_facilities %>%
      mutate(geometry_y = coords_all[, 2]) %>%
      arrange(desc(geometry_y))
    
    # Get plot limits (need to build plot first to get limits)
    # We'll use coord_sf limits for label positioning
    x_range <- county_bounds[["xmax"]] - county_bounds[["xmin"]]
    y_range <- county_bounds[["ymax"]] - county_bounds[["ymin"]]
    label_x <- county_bounds[["xmin"]] + 0.02 * x_range
    label_y_start <- county_bounds[["ymax"]] - 0.57 * y_range
    label_y_step <- 0.03 * y_range
    
    # Add labels and connecting lines
    # Extract all coordinates at once for efficiency
    coords_all <- st_coordinates(top_facilities_sorted)
    # Ensure coords_all is a matrix
    if (is.vector(coords_all)) {
      coords_all <- matrix(coords_all, nrow = 1, ncol = length(coords_all))
    }
    # Ensure we don't exceed bounds
    n_labels <- min(nrow(top_facilities_sorted), nrow(coords_all))
    for (idx in 1:n_labels) {
      row <- top_facilities_sorted[idx, ]
      facility_x <- coords_all[idx, 1]
      facility_y <- coords_all[idx, 2]
      label_y <- label_y_start - (idx - 1) * label_y_step
      
      # Add label
      p <- p + annotate("text",
                       x = label_x,
                       y = label_y,
                       label = as.character(row$`NPDES # CA#`),
                       hjust = 0,
                       vjust = 0.5,
                       size = 2.2)
      
      # Add connecting line
      p <- p + annotate("segment",
                       x = facility_x,
                       y = facility_y,
                       xend = label_x + 2.5 * 1e5,
                       yend = label_y,
                       color = "black",
                       linewidth = 0.5)
    }
  }
  
  # Add legend with custom markers
  unique_params <- sort(unique(facilities_gdf_proj$param_count))
  legend_data <- data.frame(
    value = unique_params,
    color = scales::viridis_pal(option = "viridis")(length(unique_params))
  )
  
  # Create legend manually with points
  p <- p + guides(color = guide_legend(
    title = "# Parameters Flagged",
    override.aes = list(size = 3)
  ))
  
  # Save plot
  save_fig("facilities_map.png", step = step, plot_obj = p, 
         width = 8, height = 5)
}
