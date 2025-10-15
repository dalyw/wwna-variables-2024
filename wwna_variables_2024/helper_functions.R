# converted from .ipynb files using Claude 3.5 Sonnet
library(readr)
library(dplyr)
library(lubridate)

analysis_range <- 2014:2023
save <- FALSE
load <- TRUE

# import list of npdes codes for permits in the filtered full facilities flat file
facilities_list <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
npdes_from_facilities_list <- facilities_list %>%
  filter(!is.na(`NPDES # CA#`)) %>%
  pull(`NPDES # CA#`) %>%
  unique()
facility_place_id_from_facilities_list <- facilities_list %>% pull(`FACILITY ID`)

ref_parameter <- read_csv('data/dmrs/REF_PARAMETER.csv')

columns_to_keep_dmr <- c(
  'EXTERNAL_PERMIT_NMBR',
  'LIMIT_SET_ID',
  'PARAMETER_CODE',
  'PARAMETER_DESC',
  'MONITORING_LOCATION_CODE',
  'LIMIT_VALUE_TYPE_CODE',
  'LIMIT_VALUE_NMBR',
  'LIMIT_VALUE_STANDARD_UNITS',
  'LIMIT_UNIT_CODE',
  'LIMIT_VALUE_QUALIFIER_CODE',
  'STANDARD_UNIT_CODE',
  'STATISTICAL_BASE_CODE',
  'STATISTICAL_BASE_TYPE_CODE',
  'LIMIT_FREQ_OF_ANALYSIS_CODE',
  'LIMIT_TYPE_CODE',
  'MONITORING_PERIOD_END_DATE',
  'DMR_VALUE_ID',
  'DMR_VALUE_NMBR',
  'DMR_UNIT_CODE',
  'DMR_UNIT_DESC',
  'DMR_VALUE_STANDARD_UNITS',
  'VALUE_RECEIVED_DATE',
  'NODI_CODE',
  'EXCEEDENCE_PCT'
)

columns_to_keep_esmr <- c(
  'parameter', 
  'qualifier', 
  'result', 
  'units', 'mdl', 'ml', 'rl',
  'sampling_date', 'sampling_time', 
  'review_priority_indicator', 
  'qa_codes', 'comments', 'facility_name',
  'facility_place_id', 'report_name',
  'location_desc'
)

read_dmr <- function(year, drop_no_limit = FALSE) {
  # Reads the CA DMR data for the given year
  # - Keeps only the columns that are needed
  # - Drops rows where the limit value is not present (if drop_no_limit is TRUE) and where the No Data Indicator is present
  # - Filters for monitoring locations 1, 2, EG, Y, or K
  # - Filters for permits in the npdes_list from CIWQS flat file
  #
  # Returns the cleaned data
  
  data <- read_csv(sprintf('data/dmrs/CA_FY%d_NPDES_DMRS_LIMITS/CA_FY%d_NPDES_DMRS.csv', year, year), 
                   col_types = cols(.default = "c"))
  cat(sprintf('%d DMR data has %d DMR events and %d unique permits\n', 
              year, nrow(data), n_distinct(data$EXTERNAL_PERMIT_NMBR)))
  
  data <- data %>% select(all_of(columns_to_keep_dmr))
  
  if (drop_no_limit) {
    data <- data %>% filter(!is.na(LIMIT_VALUE_NMBR)) # drop rows for monitoring without a permit limit
  }
  
  data <- data %>%
    filter(is.na(NODI_CODE)) %>% # drop rows where No Data Indicator is present
    filter(MONITORING_LOCATION_CODE %in% c('1', '2', 'EG', 'Y', 'K')) %>%
    filter(EXTERNAL_PERMIT_NMBR %in% npdes_from_facilities_list)
  
  # if data$PARAMETER_CODE has leading 0s, remove them
  data$PARAMETER_CODE <- sub("^0+", "", data$PARAMETER_CODE)
  
  data <- data %>%
    left_join(ref_parameter %>% select(PARAMETER_CODE, POLLUTANT_CODE), by = "PARAMETER_CODE") %>%
    mutate(
    MONITORING_PERIOD_END_DATE = as.Date(MONITORING_PERIOD_END_DATE, format = "%m/%d/%Y"),
    DMR_VALUE_STANDARD_UNITS = as.numeric(DMR_VALUE_STANDARD_UNITS),
    MONITORING_PERIOD_END_DATE_NUMERIC = year(MONITORING_PERIOD_END_DATE) + 
        month(MONITORING_PERIOD_END_DATE) / 12 + day(MONITORING_PERIOD_END_DATE) / 365
    )
  
  cat(sprintf('%d DMR data has %d DMR events and %d unique permits after filtering\n', 
              year, nrow(data), n_distinct(data$EXTERNAL_PERMIT_NMBR)))
  
  return(data)
}

read_all_dmrs <- function(save = FALSE, load = TRUE) {
  # Reads all DMR data for the given years and saves it to a list
  
  if (save) {
    data_list <- list()
    drop_toxicity <- FALSE
    for (year in analysis_range) {
      data_list[[as.character(year)]] <- read_dmr(year, drop_no_limit = TRUE)
      # Change the POLLUTANT_DESC on all rows with PARAMETER_CODE starting with T or W into 'Toxicity'
      data_list[[as.character(year)]]$POLLUTANT_DESC[startsWith(data_list[[as.character(year)]]$PARAMETER_CODE, "T") | 
                                                     startsWith(data_list[[as.character(year)]]$PARAMETER_CODE, "W")] <- "Toxicity"
      if (drop_toxicity) {
        data_list[[as.character(year)]] <- data_list[[as.character(year)]][!grepl("Toxicity", data_list[[as.character(year)]]$POLLUTANT_DESC), ]
      }
    }
    saveRDS(data_list, "processed_data/step1/data_list.rds")
  }
  
  if (load) {
    data_list <- readRDS("processed_data/step1/data_list.rds")
  }
  
  return(data_list)
}

read_limits <- function(year) {
  # Reads the CA DMR data for the given year
  
  data <- read.csv(sprintf('data/dmrs/CA_FY%d_NPDES_DMRS_LIMITS/CA_FY%d_NPDES_LIMITS.csv', year, year), 
                   stringsAsFactors = FALSE)
  
  cat(sprintf('%d limits data has %d limits and %d unique permits\n', 
              year, nrow(data), length(unique(data$EXTERNAL_PERMIT_NMBR))))
  
  columns_to_keep <- c(
    'EXTERNAL_PERMIT_NMBR',
    'LIMIT_SET_ID',
    'PARAMETER_CODE',
    'PARAMETER_DESC',
    'MONITORING_LOCATION_CODE',
    'LIMIT_VALUE_TYPE_CODE',
    'LIMIT_VALUE_NMBR',
    'LIMIT_VALUE_STANDARD_UNITS',
    'LIMIT_UNIT_CODE',
    'STANDARD_UNIT_CODE',
    'STATISTICAL_BASE_CODE',
    'STATISTICAL_BASE_TYPE_CODE',
    'LIMIT_VALUE_QUALIFIER_CODE',
    'LIMIT_FREQ_OF_ANALYSIS_CODE',
    'LIMIT_TYPE_CODE'
  )
  
  data <- data[, columns_to_keep]
  
  return(data)
}

read_esmr <- function(save = FALSE) {
  # Reads the CA ESMR data for all years since 2006
  
  if (save) {
    # Use data_dict to specify data types for reading the csv
    data_dict <- read.csv('data/esmr/esmr_data_dictionary.csv')
    dtype_dict <- setNames(data_dict$type, data_dict$column)
    dtype_dict <- sapply(dtype_dict, function(dtype) if (dtype %in% c('text', 'timestamp')) 'character' else 'numeric')
    
    # Read the csv with the dtype_dict
    data <- read.csv('data/esmr/esmr-analytical-export_years-2006-2024_2024-09-03.csv', 
                     colClasses = dtype_dict)
    
    # Convert timestamp columns to datetime
    timestamp_columns <- data_dict$column[data_dict$type == 'timestamp']
    data[timestamp_columns] <- lapply(data[timestamp_columns], as.POSIXct, format = "%Y-%m-%d %H:%M:%S")
    
    data <- data[, columns_to_keep_esmr]
    
    cat(sprintf('ESMR data has %d ESMR events and %d unique facilities\n', 
                nrow(data), length(unique(data$facility_place_id))))
    
    data <- data[!is.na(data$result), ]
    
    cat(sprintf('ESMR data has %d ESMR events and %d unique facilities that are not NA\n', 
                nrow(data), length(unique(data$facility_place_id))))
    
    data$sampling_date_datetime <- as.Date(data$sampling_date)
    data <- data[format(data$sampling_date_datetime, "%Y") %in% as.character(analysis_range), ]
    
    cat(sprintf('ESMR data has %d ESMR events and %d unique facilities that are not NA and are in the analysis date range\n', 
                nrow(data), length(unique(data$facility_place_id))))
    
    write.csv(data, 'processed_data/step1/esmr_data.csv', row.names = FALSE)
  } else {
    # Load the data from the csv
    data <- read.csv('processed_data/step1/esmr_data.csv')
  }
  
  return(data)
}

categorize_parameters <- function(df, parameter_sorting_dict, desc_column) {
  # Categorize parameters in a dataframe based on a sorting dictionary.
  #
  # Args:
  # df: A data frame containing parameters to categorize.
  # parameter_sorting_dict: A list containing categories and their associated keywords.
  # desc_column: Name of the column containing parameter descriptions.
  #
  # Returns:
  # A data frame with additional 'PARENT_CATEGORY' and 'SUB_CATEGORY' columns.

  # Initialize the PARENT_CATEGORY and SUB_CATEGORY columns
  df$PARENT_CATEGORY <- "Uncategorized"
  df$SUB_CATEGORY <- "Uncategorized"
  
  # Iterate through the parameter sorting dictionary
  for (key in names(parameter_sorting_dict)) {
    value <- parameter_sorting_dict[[key]]
    if ("values" %in% names(value)) {
      mask <- grepl(paste(value$values, collapse = "|"), df[[desc_column]], 
                    ignore.case = !isTRUE(value$case))
      df$PARENT_CATEGORY[mask] <- key
      df$SUB_CATEGORY[mask] <- key
    } else {
      for (sub_key in names(value)) {
        sub_value <- value[[sub_key]]
        mask <- grepl(paste(sub_value$values, collapse = "|"), df[[desc_column]], 
                      ignore.case = TRUE)
        df$PARENT_CATEGORY[mask] <- key
        df$SUB_CATEGORY[mask] <- sub_key
      }
    }
  }
  
  return(df)
}
normalize_param_desc <- function(desc) {
  desc %>%
    str_replace_all(",", "") %>%
    str_replace_all("\\[", "(") %>%
    str_replace_all("\\]", ")") %>%
    str_replace_all(" ", "") %>%
    str_replace_all("'", "") %>%
    str_replace_all("\\.", "") %>%
    str_replace_all("&", "and") %>%
    tolower()
}

match_parameter_desc <- function(row, target_df) {
  normalized_desc <- normalize_param_desc(tolower(as.character(row$PARAMETER_DESC)))
  print(paste("Normalized description:", normalized_desc))
  print(paste("Data type:", class(normalized_desc)))
  
  match <- target_df %>% filter(tolower(normalized_desc) == !!normalized_desc)
  
  if (nrow(match) == 0) {
    normalized_desc_no_sum <- normalized_desc %>%
      str_replace_all(", sum|, total", "")
    print(paste("Normalized description without sum/total:", normalized_desc_no_sum))
    print(paste("Data type:", class(normalized_desc_no_sum)))
    
    match <- target_df %>%
      filter(tolower(str_replace_all(normalized_desc, ", sum|, total", "")) == normalized_desc_no_sum)
  }
  
  ifelse(nrow(match) > 0, match$PARAMETER_DESC[1], "")
}