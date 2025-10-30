# step0_download_data.R
# Updated to match Python step0_download_data.py methodology
# with support from Claude 4.0

library(httr)
library(readr)
library(dplyr)

# Set up directories
for (data_type in c("DMR", "ESMR", "IR", "SSO", "TOXICS", "CWNS")) {
  dir_path <- file.path("data", tolower(data_type))
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
}

# Load file configs
source('wwna_variables_2024/helper_functions.R')

# Helper function to download a file
download_file <- function(url, file_path) {
  tryCatch({
    response <- GET(url, write_disk(file_path, overwrite = TRUE), progress())
    stop_for_status(response)
    return(TRUE)
  }, error = function(e) {
    return(FALSE)
  })
}

# Download DMR data for a year
download_dmr_year <- function(year) {
  zipname <- sprintf("CA_FY%d_NPDES_DMRS_LIMITS.zip", year)
  url <- sprintf("%s/%s", FILE_CONFIGS$DMR$download$url, zipname)
  zip_path <- file.path("data", "dmr", zipname)
    
  if (!download_file(url, zip_path)) {
    return(FALSE)
  }
  
  # Extract zip file then delete it
  unzip(zip_path, exdir = dirname(zip_path), overwrite = TRUE)
  unlink(zip_path)
  
  return(TRUE)
}

# Download ESMR data for a year
download_esmr_year <- function(year) {  
  resource_id <- FILE_CONFIGS$ESMR$year_config[[as.character(year)]]
  file_path <- file.path("data", "esmr", sprintf("esmr-analytical-export_year-%d.csv", year))
  # Use CKAN datastore dump endpoint for csv
  url <- sprintf("https://data.ca.gov/datastore/dump/%s?bom=True", resource_id)
  
  return(download_file(url, file_path))
}

# Download IR data for a year
download_ir_year <- function(year) {  
  xlsx_filename <- FILE_CONFIGS$IR$year_config[[as.character(year)]]
  url <- sprintf("%s/%s", FILE_CONFIGS$IR$download$url, xlsx_filename)
  
  temp_xlsx <- file.path("data", "ir", sprintf("temp_%d-303d.xlsx", year))
  
  if (!download_file(url, temp_xlsx)) {
    return(FALSE)
  }
  
  # Read and convert to CSV
  library(readxl)
  df <- read_excel(temp_xlsx)
  csv_path <- file.path(BASE_DIRS[["IR"]]$dir, sprintf("%d-303d.csv", year))
  write_csv(df, csv_path)
  
  # Delete temporary xlsx
  unlink(temp_xlsx)
  
  return(TRUE)
}

# Download SSO data
download_sso <- function() {
  
  url <- FILE_CONFIGS$SSO$download$url
  csv_path <- file.path("data", "sso", "Questionnaire.csv")
  
  # Download txt file
  txt_path <- file.path("data", "sso", "Questionnaire.txt")
  
  if (!download_file(url, txt_path)) {
    return(FALSE)
  }
  
  # Convert to CSV
  df <- read_tsv(txt_path, locale = locale(encoding = "UTF-8"))
  
  # Convert numeric columns that have comma-separated values
  for (col in names(df)) {
    if (str_detect(col, "^SSOq") && str_detect(col, "Population")) {
      df[[col]] <- as.numeric(str_replace_all(as.character(df[[col]]), ",", ""))
    }
  }
  
  write_csv(df, csv_path)
  return(TRUE)
}

# Download data by type
download_data_by_type <- function(data_type, year_range = NULL) {
  if (is.null(year_range)) {
    year_range <- list(NA)
  }
  for (item in year_range) {
    year_arg <- if (is.na(item)) NULL else item
    path_to_check <- get_data_file_path(data_type, year_arg)

    if (file.exists(path_to_check)) {
      file_size <- file.info(path_to_check)$size
      size_threshold <- FILE_CONFIGS[[data_type]]$download$size_threshold
      if (file_size > size_threshold) {
        next
      }
      # Corrupted - remove and re-download
      unlink(path_to_check)
      cat(sprintf("%s %s corrupted. Re-downloading\n", data_type, ifelse(is.null(year_arg), "", item)))
    }

    # Print once when actually downloading (match Python)
    cat(sprintf("Downloading %s\n", data_type))

    success <- FALSE
    if (data_type == "DMR") {
      success <- download_dmr_year(item)
    } else if (data_type == "ESMR") {
      success <- download_esmr_year(item)
    } else if (data_type == "IR") {
      success <- download_ir_year(item)
    } else if (data_type == "SSO") {
      success <- download_sso()
    } else {  # TOXICS or CWNS
      success <- download_file(FILE_CONFIGS[[data_type]]$download$url, path_to_check)
    }

    if (!success) {
      if (is.null(year_arg)) {
        cat(sprintf("Error processing %s\n", data_type))
      } else {
        cat(sprintf("Error processing %s %s\n", data_type, item))
      }
    }
  }
}

# Main function to download all data
main <- function() {
  # Load helper functions
  source('wwna_variables_2024/helper_functions.R')
    
  # Download all data types
  download_data_by_type("DMR", year_range = analysis_range)
  download_data_by_type("ESMR", year_range = tail(analysis_range, 1))
  download_data_by_type("IR", year_range = c(2018, 2022, 2024))
  download_data_by_type("SSO")
  download_data_by_type("TOXICS")
  download_data_by_type("CWNS")  
}

