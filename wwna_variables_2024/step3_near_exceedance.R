# step3_near_exceedance.R
# Updated to mirror the Python implementation in step3_near_exceedance.py

library(tidyverse)
library(gridExtra)
library(units)
library(jsonlite)

source("wwna_variables_2024/helper_functions.R")

`%||%` <- function(x, y) {
  if (is.null(x) || (is.atomic(x) && length(x) == 0)) y else x
}

# CONFIGURATION AND CONSTANTS

STEP3_CONFIG <- list(
  limit_threshold = ANALYSIS_CONFIG$limit_threshold,
  time_to_limit_years = ANALYSIS_CONFIG$time_to_limit_years,
  recent_violation_years = ANALYSIS_CONFIG$recent_violation_years,
  iqr_multiplier = ANALYSIS_CONFIG$iqr_multiplier,
  current_limit_year = ANALYSIS_CONFIG$current_limit_year
)

permit_col <- "EXTERNAL_PERMIT_NMBR"
param_code_col <- "PARAMETER_CODE"
unit_desc_col <- "STANDARD_UNIT_DESC"
unit_base_col <- "STANDARD_UNIT_BASE"
location_col <- "MONITORING_LOCATION_CODE"
perm_feature_nmbr_col <- "PERM_FEATURE_NMBR"
monitor_date_col <- "MONITORING_PERIOD_END_DATE_NUMERIC"
limit_begin_col <- "LIMIT_BEGIN_DATE_NUMERIC"
limit_end_col <- "LIMIT_END_DATE_NUMERIC"
qualifier_col <- "LIMIT_VALUE_QUALIFIER_CODE"
limit_val_col <- "LIMIT_VALUE_STANDARD_UNITS"
limit_val_base_col <- "LIMIT_VALUE_BASE_UNITS"
dmr_val_col <- "DMR_VALUE_STANDARD_UNITS"
dmr_val_base_col <- "DMR_VALUE_BASE_UNITS"
stat_base_col <- "STATISTICAL_BASE_CODE"
limit_type_code_col <- "LIMIT_VALUE_TYPE_CODE"
limit_value_id_col <- "LIMIT_VALUE_ID"
limit_set_schedule_col <- "LIMIT_SET_SCHEDULE_ID"

MONTH_COLS <- c("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# UNIQUE_LIMIT_COLS uniquely identify a single row in LIMITS, for merging into DMR
UNIQUE_LIMIT_COLS <- c(limit_set_schedule_col, limit_value_id_col, limit_type_code_col)

# LIMIT_GROUP_COLS identify similarly-monitored data across different permits
LIMIT_GROUP_COLS <- c(permit_col, param_code_col, location_col, perm_feature_nmbr_col, stat_base_col, unit_desc_col, limit_type_code_col)
PLOT_GROUP_COLS <- setdiff(LIMIT_GROUP_COLS, permit_col)

LIMIT_THRESHOLD <- STEP3_CONFIG$limit_threshold
RECENT_VIOLATION_YEARS <- STEP3_CONFIG$recent_violation_years
CURRENT_LIMIT_YEARS <- STEP3_CONFIG$current_limit_year
IQR_MULTIPLIER <- STEP3_CONFIG$iqr_multiplier
TIME_TO_LIMIT_YEARS <- STEP3_CONFIG$time_to_limit_years

# UNIT ALIASING AND DIMENSION MAPPING

UNIT_ALIASES_df <- read_csv_tibble("data/manual_updates/unit_aliases.csv", show_col_types = FALSE)
# Map original unit names (unit_from) to R units (unit_to_r)
UNIT_ALIAS_LOOKUP <- setNames(UNIT_ALIASES_df$unit_to_r, tolower(UNIT_ALIASES_df$unit_from))
# Map R units to themselves (for units already in correct format)
UNIT_ALIAS_LOOKUP <- c(UNIT_ALIAS_LOOKUP, setNames(UNIT_ALIASES_df$unit_to_r, tolower(UNIT_ALIASES_df$unit_to_r)))

normalize_unit_string <- function(unit) {
  if (is.null(unit) || (is.atomic(unit) && length(unit) == 0)) return("dimensionless")
  unit_str <- trimws(as.character(unit))
  if (!nzchar(unit_str)) return("dimensionless")
  unit_str <- gsub("[\u00b5\u03bc]", "u", unit_str)
  unit_str <- gsub("\\s*/\\s*", "/", unit_str)
  unit_str <- gsub("\\s+", " ", unit_str)
  unit_lower <- tolower(unit_str)
  alias <- if (unit_lower %in% names(UNIT_ALIAS_LOOKUP)) UNIT_ALIAS_LOOKUP[[unit_lower]] else NULL
  alias %||% unit_str
}

prepare_units_package_string <- function(unit_str) {
  if (tolower(unit_str) %in% c("dimensionless", "1")) return("1")
  
  # Use CSV lookup for unit conversion
  unit_lower <- tolower(unit_str)
  if (unit_lower %in% names(UNIT_ALIAS_LOOKUP)) {
    cleaned <- UNIT_ALIAS_LOOKUP[[unit_lower]]
    return(cleaned)
  }
  
  # Unit not found in CSV - return as-is (will likely fail in set_units, but that's expected)
  return(unit_str)
}

safe_set_units <- function(value, unit_str) {
  if (is.null(unit_str) || !nzchar(unit_str) || tolower(unit_str) == "dimensionless") {
    return(set_units(value, "1"))
  }
  unit_cleaned <- prepare_units_package_string(unit_str)
  # Check if cleaned unit is empty or invalid
  if (!nzchar(unit_cleaned) || tolower(unit_cleaned) == "dimensionless") {
    return(set_units(value, "1"))
  }
  
  # Handle special dimensionless units
  if (tolower(unit_cleaned) == "percent") {
    # percent is 1e-2 dimensionless
    return(set_units(value * 1e-2, "1"))
  }
  if (tolower(unit_cleaned) == "permille") {
    # permille is 1e-3 dimensionless
    return(set_units(value * 1e-3, "1"))
  }
  
  # Copy to new variable to avoid NSE issues with set_units
  u <- unit_cleaned
  
  # Try to set units
  tryCatch({
    set_units(value, u)
  }, error = function(e) {
    # Return dimensionless as fallback
    set_units(value, "1")
  })
}

compute_dimension_key <- function(unit_str) {
  if (is.null(unit_str) || !nzchar(unit_str) || tolower(unit_str) %in% c("dimensionless", "1")) {
    return("dimensionless")
  }
  q <- safe_set_units(1, unit_str)
  qb <- convert_to_base(q)
  base_unit_str <- as.character(units(qb))
  # map dimensionless to "dimensionless" instead of empty string
  if (is.null(base_unit_str) || base_unit_str == "" || trimws(base_unit_str) == "") {
    base_unit_str <- "dimensionless"
  }
  base_unit_str
}

convert_to_base_units <- function(df, convert_cols) {
  if (is.null(df) || !nrow(df)) return(df)
  
  normalized_units <- vapply(df[[unit_desc_col]], normalize_unit_string, character(1))
  df[[unit_desc_col]] <- normalized_units
  
  factor_map <- list("dimensionless" = 1.0)
  base_unit_map <- list("dimensionless" = "dimensionless")
  
  for (unit in unique(normalized_units)) {
    if (!is.null(factor_map[[unit]])) next  # Already processed
    
    if (tolower(unit) %in% c("dimensionless", "1")) {
      factor_map[[unit]] <- 1.0
      base_unit_map[[unit]] <- "dimensionless"
    } else {
      # Convert unit to base units
      q <- safe_set_units(1, unit)
      qb <- convert_to_base(q)
      base_unit_str <- as.character(units(qb))
      
      # map dimensionless to "dimensionless" instead of empty string
      if (is.null(base_unit_str) || base_unit_str == "" || trimws(base_unit_str) == "") {
        base_unit_str <- "dimensionless"
      }
      
      factor_map[[unit]] <- as.numeric(drop_units(qb))
      base_unit_map[[unit]] <- base_unit_str
    }
  }
  
  # Store base units in separate columns (don't overwrite STANDARD_UNIT_DESC)
  df[[unit_base_col]] <- vapply(normalized_units, function(u) base_unit_map[[u]], character(1))
  df[["LIMIT_BASE_UNIT_DESC"]] <- df[[unit_base_col]]
  
  # Convert values to base units
  for (col in convert_cols) {
    base_col <- gsub("STANDARD", "BASE", col)
    factors <- vapply(normalized_units, function(u) factor_map[[u]], numeric(1))
    df[[base_col]] <- df[[col]] * factors
    df[[col]] <- df[[base_col]]
  }
  
  df
}

stat_base_mapping <- fromJSON("data/manual_updates/statistical_base_code_mapping.json")

# Build stat_patterns from stat_base_mapping.json (sorted by length descending)
# stat_base_mapping has code -> list of patterns
stat_patterns <- purrr::map2(
  names(stat_base_mapping),
  stat_base_mapping,
  function(code, patterns) {
    if (length(patterns) == 0) return(NULL)
    tibble(code = code, pattern = tolower(patterns))
  }
) |>
  purrr::compact() |>
    purrr::list_rbind() |>
    filter(nzchar(pattern)) |>
    mutate(length = nchar(pattern)) |>
    arrange(desc(length))


# Map unit dimensionality to LIMIT_VALUE_TYPE_CODE
# Dimensionless -> "C", Concentration -> "C", Flow/Quantity -> "Q"
get_limit_type_code_from_dimension <- function(unit_str) {
  dim_key <- compute_dimension_key(unit_str)
  if (dim_key == "dimensionless") {
    return("C")  # Dimensionless concentration (pH, etc.)
  }
  # Check if it's concentration-like (kg/m^3) or flow-like (kg/s)
  # For simplicity, we'll use the dimension key to determine
  # In Python: _CONC_DIM = kg/m**3, _FLOW_DIM = kg/s, _TEMP_DIM = K
  # We'll check the dimension string representation
  if (grepl("\\[mass\\]/\\[length\\]\\^3", dim_key) || 
      grepl("kg.*m.*-3", dim_key) ||
      grepl("kg/m", dim_key)) {
    return("C")  # Concentration
  }
  if (grepl("\\[mass\\]/\\[time\\]", dim_key) ||
      grepl("kg.*s.*-1", dim_key) ||
      grepl("kg/s", dim_key)) {
    return("Q")  # Flow/Quantity
  }
  # Check for temperature dimensionality
  if (grepl("\\[temperature\\]", dim_key) ||
      grepl("kelvin", dim_key, ignore.case = TRUE) ||
      grepl("K", dim_key)) {
    return("Q")  # Temperature (mapped to Q type)
  }
  # No default, enforce mapping to "C" for unknown dimensions
}

# ANALYSIS HELPER FUNCTIONS

build_month_flags <- function(row) {
  values <- sapply(MONTH_COLS, function(col) row[[col]])
  flags <- MONTH_COLS[str_to_upper(as.character(values)) == "Y"]
  if (!length(flags) || length(flags) == length(MONTH_COLS)) return(NULL)
  match(MONTH_COLS, flags, nomatch = 0L) |> which() |> as.integer()
}

# MAIN ANALYSIS

main <- function(drop_toxicity = FALSE, exclude_noncompliant = FALSE) {
  # LOAD PARAMETER MAPPING FOR ESMR AND DMR PARAMETER DESCRIPTIONS
  mapping_py <- file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_py.csv")
  mapping_r <- file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv")
  mapping_path <- if (file.exists(mapping_py)) mapping_py else mapping_r
  param_mapping <- read_csv_tibble(mapping_path)

  param_desc_lookup <- param_mapping %>%
    select(all_of(c(param_code_col, "DMR_PARAMETER_DESC"))) %>%
    filter(!is.na(.data$DMR_PARAMETER_DESC)) %>%
    deframe()

  # LOAD DMR HISTORY
  # Load DMR and LIMITS data for all years, then concatenate and merge
  t_start_dmr <- Sys.time()
  dmr_parts <- list()
  limits_parts <- list()
  for (y in analysis_range) {
    # Load DMR data
    dmr_year <- load_data("DMR", year = y, drop_toxicity = drop_toxicity)
    # Drop MONITORING_LOCATION_CODE from DMR - it will come from LIMITS after merge
    dmr_year <- dmr_year %>% select(-any_of("MONITORING_LOCATION_CODE"))
    dmr_parts[[as.character(y)]] <- dmr_year
    
    # Load LIMITS data
    limits_year <- load_data("LIMITS", year = y, drop_toxicity = drop_toxicity)
    limits_year <- limits_year %>% select(-any_of("PARAMETER_DESC"))
    
    # Normalize MONITORING_LOCATION_CODE before concatenation/deduplication
    location_str <- trimws(as.character(limits_year[[location_col]]))
    effluent_codes <- c("1", "2", "EG", "Y", "K")
    limits_year[[location_col]][location_str %in% effluent_codes] <- "1"
    
    limits_parts[[as.character(y)]] <- limits_year
  }
  
  # Combine all DMR years into single DataFrame
  dmr_all <- bind_rows(dmr_parts)
  message(sprintf("  Total DMR records: %s", nrow(dmr_all)))
  
  # Truncate LIMIT_VALUE_TYPE_CODE to first character (C, Q, etc.) BEFORE merging
  # both DMR and LIMITS must be truncated before merge
  dmr_all[[limit_type_code_col]] <- substr(as.character(dmr_all[[limit_type_code_col]]), 1, 1)
  
  # Combine all LIMITS years and deduplicate on UNIQUE_LIMIT_COLS
  limits_all <- bind_rows(limits_parts)
  # Truncate LIMIT_VALUE_TYPE_CODE to first character BEFORE deduplication
  limits_all[[limit_type_code_col]] <- substr(as.character(limits_all[[limit_type_code_col]]), 1, 1)
  limits_all <- limits_all %>%
    distinct(across(all_of(UNIQUE_LIMIT_COLS)), .keep_all = TRUE)
  
  # Merge all DMR data with deduplicated LIMITS
  # DMR rows will match LIMITS rows if they share the same UNIQUE_LIMIT_COLS
  dmr_all <- inner_join(dmr_all, limits_all, by = UNIQUE_LIMIT_COLS)
  message(sprintf("  Total merged records: %s", nrow(dmr_all)))
  
  # Convert DMR values and limit values to base units for comparison
  dmr_all <- convert_to_base_units(dmr_all, c(dmr_val_col, limit_val_col))
  
  # Combine similar statistical base codes to enable use of historical data
  stat_base_str <- trimws(dmr_all[[stat_base_col]])
  dmr_all[[stat_base_col]][stat_base_str == "IA"] <- "MB"
  dmr_all[[stat_base_col]][stat_base_str == "IB"] <- "ME"
  
  # Normalize PERM_FEATURE_NMBR to combine historical and recent naming conventions
  # '001', '002', etc. -> 'EFF1', 'EFF2', etc.
  feature_nmbr_str <- trimws(as.character(dmr_all[[perm_feature_nmbr_col]]))
  numeric_pattern <- grepl("^0*\\d+$", feature_nmbr_str)
  if (sum(numeric_pattern) > 0) {
    numeric_values <- gsub("^0*", "", feature_nmbr_str[numeric_pattern])
    dmr_all[[perm_feature_nmbr_col]][numeric_pattern] <- paste0("EFF", numeric_values)
  }
  # Normalize 'INF' to 'INF1' for consistency
  dmr_all[[perm_feature_nmbr_col]][feature_nmbr_str == "INF"] <- "INF1"
  
  # Filter to groups with at least one non-NA limit_value entry after 2024
  # and at least 3 years of data span
  recent_year_threshold <- as.numeric(max(analysis_range)) - 1  # 2025 - 1 = 2024
  groups_with_recent_limit <- dmr_all %>%
    filter(
      .data[[monitor_date_col]] >= recent_year_threshold,
      !is.na(.data[[limit_val_col]]),
      !is.na(.data[[qualifier_col]])
    ) %>%
    distinct(across(all_of(LIMIT_GROUP_COLS))) %>%
    mutate(key = paste(!!!syms(LIMIT_GROUP_COLS), sep = "|")) %>%
    pull(key) %>%
    unique()
  
  dmr_filtered <- dmr_all %>%
    group_by(across(all_of(LIMIT_GROUP_COLS))) %>%
    filter({
      group_key <- paste(!!!syms(LIMIT_GROUP_COLS), sep = "|")
      group_key[1] %in% groups_with_recent_limit &&
        max(.data[[monitor_date_col]], na.rm = TRUE) - min(.data[[monitor_date_col]], na.rm = TRUE) >= 3.0
    }) %>%
    ungroup()
  
  # Get most recent valid (non-NA) limit per LIMIT_GROUP_COLS for lookup
  # This is only for the lookup dictionary - dmr_filtered still has all rows
  most_recent_limit_filtered <- dmr_filtered %>%
    filter(!is.na(.data[[limit_val_col]]), !is.na(.data[[qualifier_col]])) %>%
    arrange(desc(.data[[monitor_date_col]])) %>%
    group_by(across(all_of(LIMIT_GROUP_COLS))) %>%
    slice(1) %>%
    ungroup()
  t_end_dmr <- Sys.time()
  message(sprintf("Time to load LIMITS/DMR data: %.2f seconds", as.numeric(difftime(t_end_dmr, t_start_dmr, units = "secs"))))

  # OPTIONALLY DROP RECENT NONCOMPLIANT GROUPS
  # TO avoid double-counting with violations data for risk assessment
  if (exclude_noncompliant) {
    violation_years <- tail(sort(analysis_range), RECENT_VIOLATION_YEARS)
    recent_violation_data <- dmr_all[
      as.integer(dmr_all[[monitor_date_col]]) %in% violation_years,
    ]
    violations <- recent_violation_data[
      recent_violation_data$REPORTED_EXCURSION_NMBR > 0 |
        recent_violation_data$VIOLATION_CODE == "E90" |
        recent_violation_data$EXCEEDENCE_PCT > 0,
    ]
    noncompliant <- violations[LIMIT_GROUP_COLS] %>% distinct()
    message(sprintf(" %s noncompliant facility+parameter combos", nrow(noncompliant)))
    noncompliant_keys <- noncompliant %>%
      mutate(key = paste(!!!syms(LIMIT_GROUP_COLS), sep = "|")) %>%
      pull(key)
    dmr_filtered <- dmr_filtered %>%
      mutate(key = paste(!!!syms(LIMIT_GROUP_COLS), sep = "|")) %>%
      filter(!key %in% noncompliant_keys) %>%
      select(-key)
  }
  
  # Sort by monitoring date. Identify unique LIMIT_GROUP_COLS for ESMR matching
  dmr_filtered <- dmr_filtered %>%
    arrange(.data[[monitor_date_col]])
  dmr_group_filter <- dmr_filtered %>%
    distinct(across(all_of(LIMIT_GROUP_COLS)))
  message(sprintf("  DMR LIMIT_GROUP_COLS combos: %s", nrow(dmr_group_filter)))
  
  # Free memory - delete large DataFrames that are no longer needed
  rm(dmr_parts, limits_parts, limits_all, dmr_all, groups_with_recent_limit)
  if (exclude_noncompliant) {
    rm(noncompliant, noncompliant_keys)
  }
  gc()
  
  # REFERENCE DATA FOR FACILITY + PARAMETER DESCRIPTIONS
  wwna_facilities <- WWNA_LIST %>%
    select(`FACILITY ID`, `NPDES # CA#`) %>%
    rename(facility_place_id = `FACILITY ID`, !!permit_col := `NPDES # CA#`) %>%
    mutate(facility_place_id = as.character(facility_place_id)) %>%
    filter(!is.na(.data[[permit_col]]))
  # Create set of matching facility_place_id values for early ESMR filtering
  wwna_facility_ids <- unique(wwna_facilities$facility_place_id)
  message(sprintf(" Loaded WWNA facilities: %s", nrow(wwna_facilities)))

  # LOAD ESMR DATA WITH NORMALIZATION + METADATA
  t_start_esmr <- Sys.time()
  esmr_dataframes <- list()

  for (y in analysis_range) {
    # Load ESMR data for this year and filter to only WWNA facilities
    esmr_year <- load_data("ESMR", year = y)
    message(sprintf("  Loaded %s records", nrow(esmr_year)))
    esmr_year <- esmr_year %>%
      filter(facility_place_id %in% wwna_facility_ids)
    message(sprintf("  After WWNA filter: %s records", nrow(esmr_year)))
    
    # Convert ESMR values to base units for comparison with DMR limits
    esmr_year <- convert_to_base_units(esmr_year, c(dmr_val_col))
    message(sprintf("  After unit conversion: %s records", nrow(esmr_year)))
    
    # Map ESMR calculated_method to DMR STATISTICAL_BASE_CODE
    calc_normalized <- esmr_year$calculated_method %>%
      tolower() %>%
      trimws() %>%
      gsub("\\s+", " ", .)
    
    # create matrix of matches (rows = calc values, cols = patterns)
    # Then find first match per row (patterns are sorted by length descending)
    if (nrow(stat_patterns) > 0 && length(calc_normalized) > 0) {
      # each row is a calc value, each column is a pattern
      match_matrix <- vapply(
        stat_patterns$pattern,
        function(pattern) stringr::str_detect(calc_normalized, stringr::fixed(pattern)),
        logical(length(calc_normalized))
      )
      
      # Find first match per row: multiply by column indices, then use max.col
      # This gives us the first (leftmost) TRUE value per row
      col_indices <- matrix(rep(seq_len(ncol(match_matrix)), each = nrow(match_matrix)), 
                           nrow = nrow(match_matrix), ncol = ncol(match_matrix))
      match_matrix_idx <- match_matrix * col_indices
      
      first_match_idx <- max.col(match_matrix_idx, ties.method = "first")
      # Set to NA if no match (all zeros in row)
      first_match_idx[rowSums(match_matrix) == 0] <- NA_integer_
      
      esmr_year[[stat_base_col]] <- ifelse(
        is.na(first_match_idx),
        NA_character_,
        stat_patterns$code[first_match_idx]
      )
    } else {
      esmr_year[[stat_base_col]] <- NA_character_
    }
    
    esmr_year <- esmr_year %>% filter(!is.na(.data[[stat_base_col]]))
    message(sprintf("  After stat base mapping: %s records", nrow(esmr_year)))

    # Map unit dimensionality to LIMIT_VALUE_TYPE_CODE prefix (C, Q, etc.)
    # Use STANDARD_UNIT_DESC (original units) to determine C vs Q type
    unit_dims <- {
      unique_units <- unique(esmr_year[[unit_desc_col]])
      setNames(
        lapply(unique_units, compute_dimension_key),
        unique_units
      )
    }
    esmr_year[[limit_type_code_col]] <- esmr_year[[unit_desc_col]] %>%
      map_chr(function(unit) {
        dim_key <- unit_dims[[unit]]
        if (dim_key == "dimensionless") {
          return("C")  # Dimensionless concentration (pH, etc.)
        }
        if (grepl("\\[mass\\]/\\[length\\]\\^3", dim_key) || 
            grepl("kg.*m.*-3", dim_key) ||
            grepl("kg/m", dim_key)) {
          return("C")  # Concentration
        }
        if (grepl("\\[mass\\]/\\[time\\]", dim_key) ||
            grepl("kg.*s.*-1", dim_key) ||
            grepl("kg/s", dim_key)) {
          return("Q")  # Flow/Quantity
        }
        if (grepl("\\[temperature\\]", dim_key) ||
            grepl("kelvin", dim_key, ignore.case = TRUE) ||
            grepl("K", dim_key)) {
          return("Q")  # Temperature (mapped to Q type)
        }
        return("C")  # Default to "C"
      })
    esmr_year <- esmr_year %>% filter(!is.na(.data[[limit_type_code_col]]))
    
    # Add permit number (EXTERNAL_PERMIT_NMBR) needed for merging with DMR data
    esmr_year <- esmr_year %>%
      inner_join(
        wwna_facilities %>% select(all_of(c(permit_col, "facility_place_id"))),
        by = "facility_place_id"
      )
    message(sprintf("  After permit join: %s records", nrow(esmr_year)))
    
    # Map ESMR parameter names to DMR PARAMETER_CODE
    esmr_year <- esmr_year %>%
      inner_join(
        param_mapping %>% select(all_of(c(param_code_col, "ESMR_PARAMETER_DESC"))),
        by = c("PARAMETER_DESC" = "ESMR_PARAMETER_DESC")
      ) %>%
      select(-any_of("ESMR_PARAMETER_DESC"))
    message(sprintf("  After parameter mapping: %s records", nrow(esmr_year)))
    
    # Parse location codes: "EFF-001" -> MONITORING_LOCATION_CODE="1", PERM_FEATURE_NMBR="001"
    location_upper <- toupper(as.character(esmr_year$location))
    
    parse_location <- function(loc) {
      loc_code <- ifelse(startsWith(loc, "INF"), "0", "1")
      if (" " %in% loc) loc <- strsplit(loc, " ")[[1]][1]
      if ("-" %in% loc) {
        feature_nmbr <- strsplit(loc, "-", fixed = TRUE)[[1]][2]
        feature_nmbr <- gsub("-", "", feature_nmbr)
        return(list(loc_code = loc_code, feature_nmbr = feature_nmbr))
      }
      return(list(loc_code = loc_code, feature_nmbr = ""))
    }
    
    parsed <- lapply(location_upper, parse_location)
    esmr_year[[location_col]] <- vapply(parsed, function(p) p$loc_code, character(1))
    esmr_year[[perm_feature_nmbr_col]] <- vapply(parsed, function(p) p$feature_nmbr, character(1))
    
    # Drop rows where we couldn't parse the location
    esmr_year <- esmr_year %>% filter(!is.na(.data[[location_col]]), !is.na(.data[[perm_feature_nmbr_col]]))
    message(sprintf("  After location parsing: %s records", nrow(esmr_year)))
    
    # Filter to only ESMR records that match existing DMR LIMIT_GROUP_COLS combos
    esmr_year <- esmr_year %>%
      inner_join(dmr_group_filter, by = LIMIT_GROUP_COLS)
    message(sprintf("  After DMR group join: %s records", nrow(esmr_year)))
    
    esmr_dataframes[[as.character(y)]] <- esmr_year
    message(sprintf(" Year %s complete: %s records", y, nrow(esmr_year)))
    rm(esmr_year, location_upper, parsed)  # Free memory after appending
    gc()
  }
  
  esmr_data <- bind_rows(esmr_dataframes)
  rm(esmr_dataframes)  # Free memory
  gc()
  t_end_esmr <- Sys.time()
  message(sprintf("Time to load ESMR data: %.2f seconds", as.numeric(difftime(t_end_esmr, t_start_esmr, units = "secs"))))
  
  # COMBINE DMR + ESMR INTO A SINGLE TABLE W/ SOURCE
  dmr_filtered$`_DATA_SOURCE` <- "DMR"
  esmr_data$`_DATA_SOURCE` <- "ESMR"
  data <- bind_rows(dmr_filtered, esmr_data) %>%
    arrange(.data[[monitor_date_col]]) %>%
    as_tibble()
  message(sprintf("Combined %s DMR + %s ESMR", nrow(dmr_filtered), nrow(esmr_data)))
  
  # Create lookup dictionary indexed by LIMIT_GROUP_COLS tuple for fast access
  recent_limit_lookup <- setNames(
    split(most_recent_limit_filtered, seq_len(nrow(most_recent_limit_filtered))),
    most_recent_limit_filtered %>%
      mutate(key = paste(!!!syms(LIMIT_GROUP_COLS), sep = "|")) %>%
      pull(key)
  )
  rm(most_recent_limit_filtered, esmr_data)  # Free memory
  gc()
  
  # Pre-compute group keys for faster lookup
  data$group_key <- do.call(paste, c(data[LIMIT_GROUP_COLS], sep = "|"))
  
  # Process groups sequentially and generate plots immediately for flagged groups
  t_start_analysis <- Sys.time()
  grouped_data <- data %>%
    group_by(across(all_of(LIMIT_GROUP_COLS))) %>%
    group_split()
  
  flagged_records <- list()
  non_flagged_count <- 0  # Track how many non-flagged examples we've plotted
  
  message(sprintf("Processing %s groups sequentially", length(grouped_data)))
  
  flagged_count <- 0
  for (group_df in grouped_data) {
    key_tuple_list <- as.list(group_df[1, LIMIT_GROUP_COLS])
    key_tuple_char <- paste(sapply(key_tuple_list, as.character), collapse = "|")
    
    dates <- group_df[[monitor_date_col]]
    values <- group_df[[dmr_val_col]]
    
    if (all(is.na(values)) || all(is.na(dates))) {
      next
    }
    
    # Quartiles and two-sided outlier filtering with intraquartile range
    q1 <- quantile(values, 0.25, na.rm = TRUE, names = FALSE)
    q3 <- quantile(values, 0.75, na.rm = TRUE, names = FALSE)
    iqr <- q3 - q1
    lower_thr <- q1 - IQR_MULTIPLIER * iqr
    upper_thr <- q3 + IQR_MULTIPLIER * iqr
    outlier_mask <- !is.na(values) & !is.na(dates) & values >= lower_thr & values <= upper_thr
    filtered_dates <- dates[outlier_mask]
    filtered_values <- values[outlier_mask]
    
    if (length(unique(filtered_dates)) < 8) {
      next  # Don't analyze (fewer than 8 unique dates)
    }
    
    # linear regression using .lm.fit
    X <- cbind(1, filtered_dates)
    trend_fit <- .lm.fit(X, filtered_values)
    trend_slope <- trend_fit$coefficients[2]
    trend_intercept <- trend_fit$coefficients[1]
    
    # Get current limit values from recent_limit_lookup for flagging
    current_limit_row <- recent_limit_lookup[[key_tuple_char]]
    if (is.null(current_limit_row)) {
      next  # No current limit found, skip
    }
    
    current_limit_val <- current_limit_row[[limit_val_col]]
    qualifier_val <- trimws(current_limit_row[[qualifier_col]])
    is_maximum_limit <- qualifier_val %in% c("<=", "<")
    
    # Flagging logic
    # Use time_to_limit based on median + slope (no Q3/Q1 check needed)
    # This captures cases where values are trending toward limits
    slope_toward <- if (is_maximum_limit) {
      trend_slope > 0
    } else {
      trend_slope < 0
    }
    
    # Calculate time to limit
    distance <- abs(median(filtered_values, na.rm = TRUE) - current_limit_val)
    if (distance == 0 || !(slope_toward && abs(trend_slope) > 0)) {
      time_to_limit <- Inf
    } else {
      time_to_limit <- distance / abs(trend_slope)
    }
    
    # Check if flagged: time_to_limit check (removed Q3/Q1 near-exceedance check)
    is_flagged <- time_to_limit <= TIME_TO_LIMIT_YEARS
    
    # Determine if we should plot this group
    should_plot <- FALSE
    is_non_flagged_example <- FALSE
    
    if (is_flagged) {
      should_plot <- TRUE
      flagged_count <- flagged_count + 1
    } else if (non_flagged_count < 5 && runif(1) < 0.1) {  # ~10% chance per non-flagged group
      should_plot <- TRUE
      is_non_flagged_example <- TRUE
      non_flagged_count <- non_flagged_count + 1
    }
    
    if (!should_plot) {
      next  # Skip plotting
    }
    
    # Generate plot (for flagged or selected non-flagged)
    # Extract values for plotting
    first_data_row <- group_df[1, ]
    param_code_val <- key_tuple_list[[param_code_col]]
    param_code_char <- as.character(param_code_val)
    param_desc <- if (param_code_char %in% names(param_desc_lookup)) {
      param_desc_lookup[[param_code_char]]
    } else {
      paste("Parameter", param_code_val)
    }
    
    permit_code <- key_tuple_list[[permit_col]]
    unit_desc <- current_limit_row[[unit_base_col]]
    
    # Determine filename and save path
    base_filename <- sprintf(
      "%s_%s_%s_%s_Loc%s.png",
      permit_code,
      param_desc,
      first_data_row[[unit_desc_col]],
      first_data_row[[stat_base_col]],
      first_data_row[[location_col]]
    )
    # Clean filename: remove invalid characters for file paths
    base_filename <- gsub("[ ,\\[\\]%/:]", "_", base_filename)
    base_filename <- gsub("[^A-Za-z0-9._-]", "_", base_filename)
    
    # Add subfolder prefix for non-flagged examples
    filename <- if (is_non_flagged_example) {
      file.path("not_flagged_examples", base_filename)
    } else {
      base_filename
    }
    
    # Generate plot
    plot_data <- tibble(date = dates, value = values)
    outliers_plot <- (values < lower_thr) | (values > upper_thr)
    outliers_plot[is.na(outliers_plot)] <- FALSE
    
    trend_line <- tibble(date = dates, value = trend_slope * dates + trend_intercept)
    
    # Calculate y-axis limits
    limit_val <- current_limit_row[[limit_val_col]]
    val_min <- min(values[!outliers_plot], na.rm = TRUE)
    val_max <- max(values[!outliers_plot], na.rm = TRUE)
    low_buf <- if (is_maximum_limit) 0.95 else 0.90
    high_buf <- if (is_maximum_limit) 1.05 else 1.10
    y_min <- min(val_min, limit_val, val_min * 0.95, limit_val * low_buf, na.rm = TRUE)
    y_max <- max(val_max, limit_val, val_max * 1.05, limit_val * high_buf, na.rm = TRUE)
    
    # Create segments based on actual monitoring dates where limit values change
    # Extract monitoring date and limit value, sort by date, collapse consecutive same values
    limit_segments <- group_df %>%
      filter(`_DATA_SOURCE` == "DMR") %>%
      arrange(.data[[monitor_date_col]]) %>%
      dplyr::select(all_of(c(monitor_date_col, limit_val_col))) %>%
      drop_na(all_of(limit_val_col))
    
    if (nrow(limit_segments) > 0) {
      # Identify where limit value changes between consecutive rows
      limit_segments <- limit_segments %>%
        mutate(
          limit_changed = .data[[limit_val_col]] != lag(.data[[limit_val_col]], default = first(.data[[limit_val_col]])),
          segment = cumsum(limit_changed)
        )
      
      # Group by segment and get first row's date and limit value
      segment_df <- limit_segments %>%
        group_by(segment) %>%
        summarise(
          start = as.numeric(first(.data[[monitor_date_col]])),
          limit = as.numeric(first(.data[[limit_val_col]])),
          .groups = "drop"
        )
      
      # Set end date: next segment's start date (or end of plot range for last segment)
      segment_df$end <- c(segment_df$start[-1], 2026.0)
      
      # Ensure end doesn't exceed plot range
      segment_df$end <- pmin(segment_df$end, 2026.0)
    } else {
      # No limit data available
      segment_df <- tibble(start = numeric(0), end = numeric(0), limit = numeric(0))
    }
    
    # Create compliance zones
    zone_colors <- c("In Compliance" = "#d4f8d4", "Out of Compliance" = "#fad8d8")
    
    p1 <- ggplot(plot_data, aes(x = date, y = value))
    
    # Add compliance zones
    if (nrow(segment_df) > 0) {
      for (i in seq_len(nrow(segment_df))) {
        seg <- segment_df[i, ]
        if (is_maximum_limit) {
          p1 <- p1 +
            geom_rect(inherit.aes = FALSE, aes(xmin = seg$start, xmax = seg$end, ymin = y_min, ymax = seg$limit),
                     fill = zone_colors[["In Compliance"]], zorder = 0) +
            geom_rect(inherit.aes = FALSE, aes(xmin = seg$start, xmax = seg$end, ymin = seg$limit, ymax = y_max),
                     fill = zone_colors[["Out of Compliance"]], zorder = 0)
        } else {
          p1 <- p1 +
            geom_rect(inherit.aes = FALSE, aes(xmin = seg$start, xmax = seg$end, ymin = seg$limit, ymax = y_max),
                     fill = zone_colors[["In Compliance"]], zorder = 0) +
            geom_rect(inherit.aes = FALSE, aes(xmin = seg$start, xmax = seg$end, ymin = y_min, ymax = seg$limit),
                     fill = zone_colors[["Out of Compliance"]], zorder = 0)
        }
      }
    }
    
    # Add data points and trend line
    p1 <- p1 +
      geom_point(data = filter(plot_data, !outliers_plot), colour = "blue", alpha = 0.7, size = 2) +
      geom_point(data = filter(plot_data, outliers_plot), colour = "red", shape = 8, size = 3) +
      geom_line(data = trend_line, aes(y = value), linetype = "dashed", colour = "black", linewidth = 0.8) +
      coord_cartesian(ylim = c(y_min, y_max), xlim = c(2015, 2026)) +
      labs(x = "Time", y = unit_desc, title = paste0(param_desc, "\n", permit_code)) +
      scale_x_continuous(breaks = 2015:2026) +
      theme_minimal()
    
    # Histogram plot
    p2 <- ggplot(filter(plot_data, !outliers_plot), aes(x = value)) +
      geom_histogram(bins = 20, fill = "blue", alpha = 0.7, colour = "black") +
      geom_vline(xintercept = q1, colour = "red", linetype = "dashed") +
      geom_vline(xintercept = q3, colour = "orange", linetype = "dashed") +
      geom_vline(xintercept = current_limit_row[[limit_val_col]], colour = "grey40") +
      labs(x = unit_desc, y = "Frequency") +
      theme_minimal()
    
    combined <- arrangeGrob(p1, p2, widths = c(3, 1), ncol = 2)
    invisible(suppressMessages(save_fig(filename, 3, plot_obj = combined, width = 15, height = 6, res = 150)))
    
    # Only save CSV and track flagged records for flagged groups
    if (!is_non_flagged_example) {
      # Save CSV with all columns for this flagged group
      csv_filename <- gsub("\\.png$", ".csv", base_filename)
      csv_path <- file.path(STEP_DIRS[["3"]], "csvs_R", csv_filename)
      write_csv(group_df, csv_path)
      
      # Store the key tuple for flagged groups
      flagged_records[[length(flagged_records) + 1]] <- key_tuple_list
    }
  }
  
  rm(grouped_data)  # Free memory
  gc()
  t_end_analysis <- Sys.time()
  message(sprintf("Time for group analysis: %.2f seconds", as.numeric(difftime(t_end_analysis, t_start_analysis, units = "secs"))))
  
  if (non_flagged_count > 0) {
    message(sprintf("Generated %d example plots for non-flagged groups in not_flagged_examples/", non_flagged_count))
  }
  
  if (length(flagged_records) == 0) {
    message("No facilities flagged")
    return(invisible(NULL))
  }

  flagged_df <- bind_rows(flagged_records)
  message(sprintf("%s pairs with both near-exceedance and time-to-limit", nrow(flagged_df)))
  message(sprintf("%s unique facilities", n_distinct(flagged_df[[permit_col]])))
  message("Generated facility-parameter plots")
  
  # Map and bar plot of flagged parameter counts per facility
  flagged_counts <- flagged_df %>%
    group_by(.data[[permit_col]]) %>%
    summarise(parameters = n_distinct(.data[[param_code_col]]), .groups = "drop") %>%
    deframe()
  
  plot_map(flagged_counts, 4, step = 3)

  param_counts_df <- tibble(
    Facility = names(flagged_counts),
    Parameters = unname(flagged_counts)
  ) %>% arrange(Parameters)

  plot_barh(
    param_counts_df,
    x_col = "Parameters",
    y_col = "Facility",
    xlabel = "Number of Parameters with Slope and Near-Exceedance",
    path = "facilities_summary.png",
    step = 3
  )

  # Save aggregated results
  aggregated_df <- aggregate_flags(flagged_df, permit_col, param_code_col, AGG_STRINGS[["3"]])
  write_csv(aggregated_df, file.path(STEP_DIRS[["3"]], "flagged_facilities_step3_R.csv"))
}
