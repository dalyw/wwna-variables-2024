# step0_download_data.R
# Updated to match Python step0_download_data.py methodology
# with support from Claude 4.0

library(httr)
library(readr)
library(dplyr)

# Download configuration
BASE_DIRS <- list(
  DMR = list(
    url = "https://echo.epa.gov/files/echodownloads/NPDES_by_state_year",
    size_threshold = 1000000
  ),
  ESMR = list(
    url = "https://data.ca.gov/dataset/203e5d1f-ec9d-4d07-93aa-d8b74d3fe71f/resource",
    size_threshold = 10000000
  ),
  IR = list(
    url = "https://www.waterboards.ca.gov/water_issues/programs/tmdl",
    size_threshold = 1000000
  ),
  SSO = list(
    url = "https://www.waterboards.ca.gov/water_issues/programs/sso/docs/data_files/Questionnaire.txt",
    size_threshold = 100000
  ),
  TOXICS = list(
    url = "https://data.ca.gov/dataset/7b2b5d26-9407-4368-8744-d5b659024dd7/resource/0d417a2b-6559-4725-820f-add7c57a8bc9/download/oehha-toxicity-criteria-database-20250408.csv",
    size_threshold = 1000
  ),
  CWNS = list(
    url = "https://raw.githubusercontent.com/dalyw/us-sewersheds/refs/heads/main/processed_data/facilities_merged.csv",
    size_threshold = 100000
  )
)

# Set up directories
for (data_type in names(BASE_DIRS)) {
  dir_path <- file.path("data", tolower(data_type))
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  BASE_DIRS[[data_type]]$dir <- dir_path
}

# IR 303d paths
IR_303D_PATHS <- list(
  "2018" = "2018state_ir_reports_final/app_a_2018303d.xlsx",
  "2022" = "2020_2022state_ir_reports_revised_final/apx-a-303d-list.xlsx",
  "2024" = "2023_2024state_ir_reports/apx-a-2024-303d-list-final.xlsx"
)

# ESMR Resource IDs
ESMR_RESOURCE_IDS <- list(
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

# Helper function to download a file
download_file <- function(url, file_path) {
  tryCatch({
    response <- GET(url, write_disk(file_path, overwrite = TRUE), progress())
    stop_for_status(response)
    cat(sprintf("Downloaded: %s\n", file_path))
    return(TRUE)
  }, error = function(e) {
    cat(sprintf("Error downloading %s: %s\n", url, e$message))
    return(FALSE)
  })
}

# Download DMR data for a year
download_dmr_year <- function(year) {
  zipname <- sprintf("CA_FY%d_NPDES_DMRS_LIMITS.zip", year)
  url <- sprintf("%s/%s", BASE_DIRS[["DMR"]]$url, zipname)
  zip_path <- file.path(BASE_DIRS[["DMR"]]$dir, zipname)
  
  cat(sprintf("Downloading DMR for %d\n", year))
  
  if (!download_file(url, zip_path)) {
    return(FALSE)
  }
  
  # Extract zip file
  unzip(zip_path, exdir = dirname(zip_path), overwrite = TRUE)
  
  # Delete the zip file
  unlink(zip_path)
  
  return(TRUE)
}

# Download ESMR data for a year
download_esmr_year <- function(year) {
  cat(sprintf("Downloading eSMR for %d\n", year))
  
  resource_id <- ESMR_RESOURCE_IDS[[as.character(year)]]
  file_path <- file.path(BASE_DIRS[["ESMR"]]$dir, sprintf("esmr-analytical-export_year-%d.csv", year))
  filename_no_ext <- basename(file_path)
  url <- sprintf("%s/%s/download/%s_2025-10-06.csv", 
                 BASE_DIRS[["ESMR"]]$url, resource_id, filename_no_ext)
  
  return(download_file(url, file_path))
}

# Download IR data for a year
download_ir_year <- function(year) {
  cat(sprintf("Downloading IR for %d\n", year))
  
  xlsx_filename <- IR_303D_PATHS[[as.character(year)]]
  url <- sprintf("%s/%s", BASE_DIRS[["IR"]]$url, xlsx_filename)
  
  temp_xlsx <- file.path(BASE_DIRS[["IR"]]$dir, sprintf("temp_%d-303d.xlsx", year))
  
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
  cat("Downloading SSO\n")
  
  url <- BASE_DIRS[["SSO"]]$url
  csv_path <- file.path(BASE_DIRS[["SSO"]]$dir, "Questionnaire.csv")
  
  # Download txt file
  txt_path <- file.path(BASE_DIRS[["SSO"]]$dir, "Questionnaire.txt")
  
  if (!download_file(url, txt_path)) {
    return(FALSE)
  }
  
  # Convert to CSV
  cat("Converting SSO from txt to csv\n")
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
  for (item in year_range) {
    # Get paths using helper function
    path_to_check <- get_data_file_path(data_type, item)
    
    # Check if file exists and is of sufficient size
    if (file.exists(path_to_check)) {
      file_size <- file.info(path_to_check)$size
      if (file_size > BASE_DIRS[[data_type]]$size_threshold) {
        next
      }
    }
    
    cat(sprintf("%s %s missing or corrupted\n", data_type, item))
    
    # Delete corrupted file if exists
    if (file.exists(path_to_check)) {
      unlink(path_to_check)
    }
    
    # Download based on data type
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
      success <- download_file(BASE_DIRS[[data_type]]$url, path_to_check)
    }
    
    if (!success) {
      cat(sprintf("Error processing %s %s\n", data_type, item))
    }
  }
}

# Main function to download all data
main <- function() {
  # Make sure helper functions are loaded
  if (!exists("get_data_file_path")) {
    source('wwna_variables_2024/helper_functions.R')
  }
  
  # Analysis range
  analysis_range <- 2014:2023
  
  # Download all data types
  download_data_by_type("DMR", year_range = analysis_range)
  download_data_by_type("ESMR", year_range = analysis_range)
  download_data_by_type("IR", year_range = c(2018, 2022, 2024))
  download_data_by_type("SSO")
  download_data_by_type("TOXICS")
  download_data_by_type("CWNS")
  
  cat("All data download completed!\n")
}

