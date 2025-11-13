# step1_parameter_categorization.R
# Updated to match Python step1_parameter_categorization.py methodology
# with support from Claude 4.0

library(tidyverse)
library(jsonlite)
library(ggplot2)

source('wwna_variables_2024/helper_functions.R')

# CATEGORIZE PARAMETERS
PARAMETER_SORTING_DICT <- jsonlite::fromJSON('data/manual_updates/parameter_sorting_dict.json')

# Function to clean parameter codes
clean_param_code <- function(series) {
  sub("^0+", "", as.character(series))
}

ref_parameter <- read_csv_tibble(
  'data/dmr/REF_Parameter.csv',
  col_types = cols(POLLUTANT_CODE = col_integer())
) %>%
  mutate(PARAMETER_CODE = clean_param_code(PARAMETER_CODE))

ref_parameter_no_desc <- ref_parameter %>%
  select(-PARAMETER_DESC)

# Function to normalize parameter descriptions
normalize_param_desc <- function(desc) {
  desc <- as.character(desc)
  replacements <- list(
    "," = "", " " = "", "'" = "", "." = "",
    "[" = "(", "]" = ")", "&" = "and"
  )
  for (word in c("sum", "total", "tot.")) {
    replacements[[paste0(", ", word)]] <- ""
    capitalized <- paste0(toupper(substr(word, 1, 1)), substr(word, 2, nchar(word)))
    replacements[[paste0(", ", capitalized)]] <- ""
  }
  desc <- tolower(desc)
  for (old in names(replacements)) {
    desc <- gsub(old, replacements[[old]], desc, fixed = TRUE)
  }
  desc
}

# Recursive function to build category rules list
iter_category_rules <- function(tree, parent = NULL) {
  rules <- list()
  for (key in names(tree)) {
    node <- tree[[key]]
    if (!is.list(node)) next
    if ("values" %in% names(node)) {
      rules[[length(rules) + 1]] <- list(
        parent = if (is.null(parent)) key else parent,
        child = key,
        values = node$values,
        case = ifelse(is.null(node$case), FALSE, node$case)
      )
    } else {
      rules <- c(rules, iter_category_rules(node, parent = key))
    }
  }
  rules
}

# Build CATEGORY_RULES list
category_rules_list <- iter_category_rules(PARAMETER_SORTING_DICT)
CATEGORY_RULES <- purrr::map(category_rules_list, function(rule) {
  # Escape special regex characters
  escaped_values <- gsub("([.|()\\^{}+$*?\\[\\\\])", "\\\\\\1", rule$values)
  pattern <- paste(escaped_values, collapse = "|")
  list(
    parent = rule$parent,
    child = rule$child,
    pattern = pattern,
    case = rule$case
  )
})

# Load manual mapping
manual_mapping_df <- read_csv_tibble("data/manual_updates/dmr_esmr_mapping_manual.csv") %>%
  mutate(PARAMETER_CODE = clean_param_code(PARAMETER_CODE))
MANUAL_MAPPING <- setNames(
  manual_mapping_df$ESMR_PARAMETER_DESC_MANUAL,
  manual_mapping_df$PARAMETER_CODE
)

# Function to plot pie counts
plot_pie_counts <- function(df, title) {
  category_counts <- table(df$PARENT_CATEGORY)
  pct_labels <- ifelse(
    category_counts / sum(category_counts) > 0.04,
    paste0(round(100 * category_counts / sum(category_counts), 1), "%"),
    ""
  )
  
  filename <- paste0(tolower(gsub(" ", "_", title)), ".png")
  fig_path <- file.path(STEP_DIRS[["1"]], "figures_R", filename)
  png(fig_path, width = 5, height = 5, units = "in", res = 150)
  pie(category_counts, 
      labels = pct_labels,
      main = title,
      init.angle = 140)
  legend("center left", 
         legend = names(category_counts), 
         fill = rainbow(length(category_counts)),
         x = 1.2, y = 0.5)
  dev.off()
}

# Function to load and apply categories 
load_and_apply_categories <- function(name, year) {
  column <- paste0(name, "_PARAMETER_DESC")
  
  if (name == "DMR") {
    # For DMR, use all parameter codes from REF_Parameter.csv
    df <- ref_parameter %>%
      select(PARAMETER_CODE, PARAMETER_DESC, POLLUTANT_CODE) %>%
    rename(!!column := PARAMETER_DESC)
  } else {
    df <- load_data(name, year)
    df <- df %>%
      rename(!!column := PARAMETER_DESC) %>%
      distinct(across(all_of(column))) %>%
      as_regular_tibble()
  
  if (name == "TOXICS") {
    df[[column]] <- str_replace(df[[column]], "^\\d+\\.\\s*", "")
    }
  }
  
  df <- df %>%
    mutate(
      !!column := ifelse(is.na(.data[[column]]), "", .data[[column]]),
      PARENT_CATEGORY = "Uncategorized",
      SUB_CATEGORY = "Uncategorized"
    )
  
  descriptions <- as.character(df[[column]])
  for (rule in CATEGORY_RULES) {
    if (rule$case) {
      mask <- str_detect(descriptions, regex(rule$pattern))
    } else {
      mask <- str_detect(descriptions, regex(rule$pattern, ignore_case = TRUE))
    }
    if (any(mask)) {
      df$PARENT_CATEGORY[mask] <- rule$parent
      df$SUB_CATEGORY[mask] <- rule$child
    }
  }
  
  df
}

# Main function
main <- function() {
  recent_year <- tail(YEAR_RANGE, 1)
  
  dataframes <- list(
    DMR = load_and_apply_categories("DMR", recent_year),
    ESMR = load_and_apply_categories("ESMR", recent_year),
    IR = load_and_apply_categories("IR", 2024),
    TOXICS = load_and_apply_categories("TOXICS", recent_year)
  )
  
  dmr <- dataframes[["DMR"]]
  manual_params <- read_csv_tibble("data/manual_updates/parameters_manual_additions.csv")
  
  # Build IR parameter catalog from multiple sources
  ir_df <- dataframes[["IR"]]
  ir_df <- bind_rows(
    ir_df,
    manual_params %>% select(IR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY),
    dmr %>%
      select(DMR_PARAMETER_DESC, PARENT_CATEGORY, SUB_CATEGORY) %>%
      rename(IR_PARAMETER_DESC = DMR_PARAMETER_DESC) %>%
      mutate(
        PARENT_CATEGORY = ifelse(is.na(PARENT_CATEGORY), "Uncommon", PARENT_CATEGORY),
        SUB_CATEGORY = ifelse(is.na(SUB_CATEGORY), "Uncommon", SUB_CATEGORY)
      ),
    map_dfr(
      seq(YEAR_RANGE[1], YEAR_RANGE[2] - 1),
      function(year) {
        load_data("LIMITS", year) %>% select(PARAMETER_DESC)
      }
    ) %>%
      distinct() %>%
      filter(!is.na(PARAMETER_DESC)) %>%
      rename(IR_PARAMETER_DESC = PARAMETER_DESC) %>%
      mutate(
        PARENT_CATEGORY = "Uncommon",
        SUB_CATEGORY = "Uncommon"
      )
  ) %>%
    distinct(IR_PARAMETER_DESC, .keep_all = TRUE) %>%
    as_regular_tibble()
  
  ir_df %>%
    write_csv(file.path(STEP_DIRS[["1"]], "ir_parameter_df_R.csv"))
  dataframes[["IR"]] <- ir_df
  
  # Plot pie charts
  for (key in names(dataframes)) {
    plot_pie_counts(dataframes[[key]], paste(key, "Categories"))
  }
  
  # Match DMR parameters to ESMR and TOXICS using normalized descriptions
  for (source_name in c("ESMR", "TOXICS")) {
    source_col <- paste0(source_name, "_PARAMETER_DESC")
    normalized <- vapply(
      ifelse(is.na(dataframes[[source_name]][[source_col]]), "", dataframes[[source_name]][[source_col]]),
      normalize_param_desc,
      character(1)
    )
    lookup <- setNames(
      ifelse(is.na(dataframes[[source_name]][[source_col]]), "", dataframes[[source_name]][[source_col]]),
      normalized
    )
    lookup <- lookup[nchar(names(lookup)) > 0]
    lookup <- lookup[!duplicated(names(lookup))]
    normalized_dmr <- vapply(
      ifelse(is.na(dmr$DMR_PARAMETER_DESC), "", dmr$DMR_PARAMETER_DESC),
      normalize_param_desc,
      character(1)
    )
    matched <- lookup[normalized_dmr]
    matched[is.na(matched)] <- ""
    dmr[[paste0(source_name, "_PARAMETER_DESC_MATCHED")]] <- matched
    unique_count <- length(unique(matched[matched != ""]))
    message(sprintf("%d of %d exact matched to %s", unique_count, nrow(dmr), source_name))
  }
  
  # Apply manual mappings first, then exact normalized matches
  dmr$ESMR_PARAMETER_DESC_MATCHED[dmr$ESMR_PARAMETER_DESC_MATCHED == ""] <- NA_character_
  manual_codes <- names(MANUAL_MAPPING)
  dmr$ESMR_PARAMETER_DESC <- ifelse(
    dmr$PARAMETER_CODE %in% manual_codes,
    MANUAL_MAPPING[dmr$PARAMETER_CODE],
    dmr$ESMR_PARAMETER_DESC_MATCHED
  )
  
  # Apply similarity-based matching for remaining unmatched (>0.9 similarity)
  # Only for parameters not already matched through exact or manual
  message("Running similarity-based matching")
  unmatched_idx <- which(is.na(dmr$ESMR_PARAMETER_DESC))
  
  # Get ESMR descriptions already mapped (by manual or exact)
  already_mapped_esmr <- unique(dmr$ESMR_PARAMETER_DESC[!is.na(dmr$ESMR_PARAMETER_DESC)])
  
  if (length(unmatched_idx) > 0) {
    esmr_descs <- unique(dataframes[["ESMR"]]$ESMR_PARAMETER_DESC)
    esmr_descs <- esmr_descs[!is.na(esmr_descs) & esmr_descs != "" & !esmr_descs %in% already_mapped_esmr]
    
    similarity_matches <- list()
    for (idx in unmatched_idx) {
      dmr_desc <- dmr$DMR_PARAMETER_DESC[idx]
      dmr_normalized <- normalize_param_desc(dmr_desc)
      
      # Find best match by similarity
      best_match <- NULL
      best_sim <- 0.0
      for (esmr_desc in esmr_descs) {
        esmr_normalized <- normalize_param_desc(esmr_desc)
        # Calculate similarity using Levenshtein distance
        max_len <- max(nchar(dmr_normalized), nchar(esmr_normalized))
        if (max_len > 0) {
          dist <- adist(dmr_normalized, esmr_normalized)[1, 1]
          sim <- 1 - (dist / max_len)
        } else {
          sim <- if (dmr_normalized == esmr_normalized) 1.0 else 0.0
        }
        if (sim > best_sim) {
          best_sim <- sim
          best_match <- esmr_desc
        }
      }
      
      # Only use if similarity >0.9
      if (best_sim > 0.9) {
        similarity_matches[[length(similarity_matches) + 1]] <- list(idx = idx, match = best_match, sim = best_sim)
      }
    }
    
    # Apply similarity matches
    if (length(similarity_matches) > 0) {
      for (match_info in similarity_matches) {
        dmr$ESMR_PARAMETER_DESC[match_info$idx] <- match_info$match
      }
      message(sprintf("Applied %d similarity-based matches (>0.9)", length(similarity_matches)))
    }
  }
  
  dmr <- dmr %>% select(-ESMR_PARAMETER_DESC_MATCHED)
  
  # Remove ambiguous auto-matched mappings (multiple DMR codes -> same ESMR desc)
  # This catches any remaining conflicts (e.g., multiple similarity matches to same ESMR)
  is_auto_matched <- !is.na(dmr$ESMR_PARAMETER_DESC) & !(dmr$PARAMETER_CODE %in% manual_codes)
  esmr_counts <- table(dmr$ESMR_PARAMETER_DESC[is_auto_matched])
  ambiguous_esmr <- names(esmr_counts[esmr_counts > 1])
  if (length(ambiguous_esmr) > 0) {
    dmr$ESMR_PARAMETER_DESC[is_auto_matched & dmr$ESMR_PARAMETER_DESC %in% ambiguous_esmr] <- NA_character_
    message(sprintf("Removed %d ambiguous auto-matched ESMR mappings", length(ambiguous_esmr)))
  }
  
  # Fill remaining NAs
  dmr$ESMR_PARAMETER_DESC[is.na(dmr$ESMR_PARAMETER_DESC)] <- "No Match (unconfirmed)"
  
  # Save
  dmr %>%
    write_csv(file.path(STEP_DIRS[["1"]], "dmr_esmr_mapping_R.csv"))
  
  dataframes[["DMR"]] <- dmr
}

