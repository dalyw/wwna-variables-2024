# step1_parameter_categorization.R
# Updated to match Python step1_parameter_categorization.py methodology
# with support from Claude 4.0

library(tidyverse)
library(jsonlite)
library(ggplot2)

source('wwna_variables_2024/helper_functions.R')

# Load parameter sorting dictionary
parameter_sorting_dict <- jsonlite::fromJSON('data/manual_updates/parameter_sorting_dict.json')
ref_parameter <- suppressMessages(read_csv('data/dmr/REF_PARAMETER.csv'))

# Function to normalize parameter descriptions
normalize_param_desc <- function(desc) {
  str_replace_all(tolower(as.character(desc)), 
    c("," = "", " " = "", "\\[" = "(", "\\]" = ")", "'" = "", "\\." = "", "&" = "and")) %>%
    str_replace_all("(, sum)|(, total)|(, Sum)|(, Total)|(tot\\.)", "")
}

# Function to categorize parameters
categorize_parameters <- function(df, parameter_sorting_dict, desc_column) {
  df$PARENT_CATEGORY <- "Uncategorized"
  df$SUB_CATEGORY <- "Uncategorized"
  
  # Helper function to recursively apply categories
  apply_categories <- function(d, parent = NULL) {
    for (key in names(d)) {
      value <- d[[key]]
      if ("values" %in% names(value)) {
        # Leaf node: apply category
        pattern <- paste(value$values, collapse = "|")
        case_insensitive <- ifelse(is.null(value$case) || !value$case, TRUE, FALSE)
        if (case_insensitive) {
          mask <- str_detect(df[[desc_column]], regex(pattern, ignore_case = TRUE))
        } else {
          mask <- str_detect(df[[desc_column]], pattern)
        }
        df$PARENT_CATEGORY[mask] <<- ifelse(is.null(parent), key, parent)
        df$SUB_CATEGORY[mask] <<- key
      } else if (is.list(value)) {
        # Branch node: recurse
        apply_categories(value, parent = key)
      }
    }
  }
  
  apply_categories(parameter_sorting_dict)
  return(df)
}

# Function to match parameter descriptions
match_param_desc <- function(row, target_df, target_desc_column) {
  normalized_desc <- normalize_param_desc(row$PARAMETER_DESC)
  match <- target_df[target_df$normalized_desc == normalized_desc, ]
  if (nrow(match) > 0) {
    return(match[[target_desc_column]][1])
  } else {
    return("")
  }
}

# Function to plot pie counts
plot_pie_counts <- function(df, title, step = 1) {
  category_counts <- table(df$PARENT_CATEGORY)
  pct_labels <- ifelse(category_counts / sum(category_counts) > 0.04,
                       paste0(round(100 * category_counts / sum(category_counts), 1), "%"),
                       "")
  
  figures_dir <- file.path(STEP_DIRS[["1"]], "figures_R")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  png(file.path(STEP_DIRS[["1"]], "figures_R", paste0(tolower(gsub(" ", "_", title)), ".png")), 
      width = 800, height = 800)
  pie(category_counts, 
      labels = pct_labels,
      main = title,
      init.angle = 140)
  legend("topleft", 
         legend = names(category_counts), 
         fill = rainbow(length(category_counts)),
         cex = 0.8)
  dev.off()
}

# Main function
main <- function() {
  # Load and process each data source
  dataframes <- list()
  
  for (key in c("DMR", "ESMR", "IR", "TOXICS")) {
    if (key == "DMR") {
      data <- load_data("DMR", 2023)
      
      # Add POLLUTANT_CODE from ref_parameter for step1 processing
      data <- data %>%
        mutate(PARAMETER_CODE_CLEAN = sub("^0+", "", PARAMETER_CODE)) %>%
        left_join(ref_parameter %>% dplyr::select(PARAMETER_CODE, POLLUTANT_CODE), 
                  by = c("PARAMETER_CODE_CLEAN" = "PARAMETER_CODE"))
      
      processed <- data %>%
        dplyr::select(PARAMETER_CODE, PARAMETER_DESC, POLLUTANT_CODE) %>%
        distinct(PARAMETER_CODE, .keep_all = TRUE)
      
      cat(sprintf("%d unique parameters in DMR 2023 data\n", nrow(processed)))
      dataframes[[key]] <- processed
      next
    }
    
    config <- FILE_CONFIGS[[key]]$step1
    if (is.null(config)) next
    
    # Load CSV data
    if (key == "ESMR") {
      data <- load_data("ESMR", 2023)
    } else {
      data <- load_data(key, config$year)
    }
    
    # Extract column and rename
    processed <- data %>%
      dplyr::select(all_of(config$column)) %>%
      distinct() %>%
      rename(!!config$desc_col := all_of(config$column))
    
    # Apply post-processing if specified
    if (!is.null(config$post_process)) {
      pp <- config$post_process$strip_prefix
      if (!is.null(pp)) {
        pattern <- pp$pattern
        processed[[config$desc_col]] <- str_replace(processed[[config$desc_col]], pattern, "")
      }
    }
    
    dataframes[[key]] <- processed
  }
  
  # Categorize parameters
  category_cols <- list(
    "DMR" = "PARAMETER_DESC",
    "IR" = "IR_PARAMETER_DESC",
    "ESMR" = "ESMR_PARAMETER_DESC",
    "TOXICS" = "TOXICS_PARAMETER_DESC"
  )
  
  for (key in names(dataframes)) {
    desc_col <- category_cols[[key]]
    dataframes[[key]] <- categorize_parameters(dataframes[[key]], parameter_sorting_dict, desc_col)
  }
  
  # Save ir_parameter_df
  write_csv(dataframes[["IR"]], file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv"))
  
  # Merge with manually added parameters if they exist
  manual_params_path <- "data/manual_updates/parameters_manual_additions.csv"
  if (file.exists(manual_params_path)) {
    manual_params <- suppressMessages(read_csv(manual_params_path))
    existing_params <- unique(dataframes[["IR"]]$IR_PARAMETER_DESC)
    new_params <- manual_params %>%
      filter(!IR_PARAMETER_DESC %in% existing_params)
    
    cat(sprintf("Adding %d manually added parameters to ir_parameter_df\n", nrow(new_params)))
    
    combined <- bind_rows(
      dataframes[["IR"]],
      new_params %>% dplyr::select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY)
    )
  } else {
    combined <- dataframes[["IR"]]
  }
  
  # Add unmapped DMR parameters to ir_parameter_df
  dmr_params <- dataframes[["DMR"]] %>%
    dplyr::select(PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY) %>%
    rename(IR_PARAMETER_DESC = PARAMETER_DESC) %>%
    filter(!is.na(IR_PARAMETER_DESC))
  
  existing_dmr <- unique(combined$IR_PARAMETER_DESC)
  new_dmr_params <- dmr_params %>%
    filter(!IR_PARAMETER_DESC %in% existing_dmr)
  
  # Fill in Uncommon for new DMR params
  new_dmr_params$PARENT_CATEGORY[is.na(new_dmr_params$PARENT_CATEGORY)] <- "Uncommon"
  new_dmr_params$SUB_CATEGORY[is.na(new_dmr_params$SUB_CATEGORY)] <- "Uncommon"
  
  if (nrow(new_dmr_params) > 0) {
    cat(sprintf("Adding %d DMR parameters to ir_parameter_df\n", nrow(new_dmr_params)))
    combined <- bind_rows(combined, new_dmr_params)
  }
  
  # Add LIMITS parameters that aren't in DMR or IR
  limits_data <- load_data("LIMITS", 2023)
  limits_params <- data.frame(IR_PARAMETER_DESC = unique(limits_data$PARAMETER_DESC))
  
  # Apply keyword-based mapping
  limits_params$PARENT_CATEGORY <- "Uncommon"
  limits_params$SUB_CATEGORY <- "Uncommon"
  
  existing_combined <- unique(combined$IR_PARAMETER_DESC)
  new_limits_params <- limits_params %>%
    filter(!IR_PARAMETER_DESC %in% existing_combined)
  
  if (nrow(new_limits_params) > 0) {
    cat(sprintf("Adding %d LIMITS parameters to ir_parameter_df\n", nrow(new_limits_params)))
    combined <- bind_rows(combined, new_limits_params)
  }
  
  write_csv(combined, file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv"))
  
  # Create parameter reference by merging ref_parameter with combined categories
  ref_parameter_merged <- ref_parameter %>%
    left_join(combined %>% 
              dplyr::select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY) %>%
              distinct(IR_PARAMETER_DESC, .keep_all = TRUE),
              by = c("PARAMETER_DESC" = "IR_PARAMETER_DESC"), 
              relationship = "many-to-one")
  
  # Clean PARAMETER_CODE for matching
  ref_parameter_merged$PARAMETER_CODE_CLEAN <- sub("^0+", "", ref_parameter_merged$PARAMETER_CODE)
  
  # Save consolidated reference
  write_csv(ref_parameter_merged, file.path(STEP_DIRS[["1"]], "ref_parameter_merged_R.csv"))
  cat(sprintf("Saved consolidated parameter reference with %d parameters\n", nrow(ref_parameter_merged)))
  
  # Plot category distributions
  for (key in names(category_cols)) {
    if (key %in% names(dataframes)) {
      plot_pie_counts(dataframes[[key]], paste(key, "Categories"), step = 1)
    }
  }
  
  # Parameter name matching
  for (key in c("ESMR", "TOXICS")) {
    if (!key %in% names(dataframes)) next
    
    target_df <- dataframes[[key]]
    
    # Normalize target descriptions
    target_df$normalized_desc <- sapply(target_df[[paste(key, "_PARAMETER_DESC", sep = "")]], 
                                         normalize_param_desc)
    
    # Match DMR to target
    matched_col <- paste(key, "_PARAMETER_DESC_MATCHED", sep = "")
    dataframes[["DMR"]][[matched_col]] <- sapply(1:nrow(dataframes[["DMR"]]), 
      function(i) {
        match_param_desc(dataframes[["DMR"]][i, ], target_df, 
                         paste(key, "_PARAMETER_DESC", sep = ""))
      })
    
    # Print match statistics
    unique_count <- length(unique(dataframes[["DMR"]][[matched_col]])) - 1
    cat(sprintf("%d of %d auto matched to %s\n", unique_count, nrow(dataframes[["DMR"]]), key))
  }
  
  # Add manual mappings for ESMR
  if (file.exists("data/manual_updates/dmr_esmr_mapping_manual.csv")) {
    manual_mapping <- suppressMessages(read_csv("data/manual_updates/dmr_esmr_mapping_manual.csv")) %>%
      dplyr::select(PARAMETER_CODE, ESMR_PARAMETER_DESC_MANUAL)
    
    dataframes[["DMR"]] <- dataframes[["DMR"]] %>%
      left_join(manual_mapping, by = "PARAMETER_CODE") %>%
      mutate(ESMR_PARAMETER_DESC_MANUAL = replace_na(ESMR_PARAMETER_DESC_MANUAL, ""))
    
    dataframes[["DMR"]] <- dataframes[["DMR"]] %>%
      mutate(ESMR_PARAMETER_DESC = case_when(
        !is.na(ESMR_PARAMETER_DESC_MATCHED) & ESMR_PARAMETER_DESC_MATCHED != "" ~ ESMR_PARAMETER_DESC_MATCHED,
        !is.na(ESMR_PARAMETER_DESC_MANUAL) & ESMR_PARAMETER_DESC_MANUAL != "" ~ ESMR_PARAMETER_DESC_MANUAL,
        TRUE ~ "No Match (unconfirmed)"
      )) %>%
      dplyr::select(-ESMR_PARAMETER_DESC_MATCHED, -ESMR_PARAMETER_DESC_MANUAL)
  } else {
    dataframes[["DMR"]] <- dataframes[["DMR"]] %>%
      mutate(ESMR_PARAMETER_DESC = replace_na(ESMR_PARAMETER_DESC_MATCHED, "No Match (unconfirmed)")) %>%
      dplyr::select(-ESMR_PARAMETER_DESC_MATCHED)
  }
  
  # Final cleanup and save
  dataframes[["DMR"]] <- dataframes[["DMR"]] %>%
    rename(DMR_PARAMETER_DESC = PARAMETER_DESC)
  
  write_csv(dataframes[["DMR"]], file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv"))
}

