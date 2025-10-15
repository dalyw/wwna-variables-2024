# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(tidyverse)
library(readr)

source('helper_functions.R')

# Import DMR Parameter Data
dmrs_2023 <- read_dmr(2023, drop_no_limit = TRUE)
unique_parameter_codes_dmrs <- unique(dmrs_2023$PARAMETER_CODE)
dmr_parameter_df <- read_csv('data/dmrs/REF_PARAMETER.csv', show_col_types = FALSE)
dmr_parameter_df <- dmr_parameter_df[dmr_parameter_df$PARAMETER_CODE %in% unique_parameter_codes_dmrs,]
cat(sprintf("%d parameters and %d unique pollutants in DMR limit and monitoring datasets\n", 
            nrow(dmr_parameter_df), length(unique(dmr_parameter_df$POLLUTANT_CODE))))

# Import ESMR Data
esmr_data <- read_esmr(save = FALSE)
esmr_parameter_df <- data.frame(ESMR_PARAMETER_DESC = unique(esmr_data$parameter))

# Import IR Data
impaired_303d_2024 <- read_csv('data/ir/2024-303d.csv', skip = 1, show_col_types = FALSE)
ir_parameter_df <- impaired_303d_2024 %>%
  select(Pollutant) %>%
  distinct() %>%
  rename(IR_PARAMETER_DESC = Pollutant)

# Import CA toxics rule data
toxics_parameter_df <- read_csv('data/toxics/criteria_for_toxics.csv', show_col_types = FALSE) %>%
  rename(TOXICS_PARAMETER_DESC = `Number compound`) %>%
  mutate(TOXICS_PARAMETER_DESC = str_replace(TOXICS_PARAMETER_DESC, "^\\d+\\.\\s*", ""))

# Load parameter sorting dictionary
parameter_sorting_dict <- jsonlite::fromJSON('processed_data/step1/parameter_sorting_dict.json')

# Categorize parameters using the parameter sorting dictionary
dmr_parameter_df <- categorize_parameters(dmr_parameter_df, parameter_sorting_dict, 'PARAMETER_DESC')
ir_parameter_df <- categorize_parameters(ir_parameter_df, parameter_sorting_dict, 'IR_PARAMETER_DESC')
esmr_parameter_df <- categorize_parameters(esmr_parameter_df, parameter_sorting_dict, 'ESMR_PARAMETER_DESC')
toxics_parameter_df <- categorize_parameters(toxics_parameter_df, parameter_sorting_dict, 'TOXICS_PARAMETER_DESC')

# Additional categorization of Total Toxics
dmr_parameter_df <- dmr_parameter_df %>%
  mutate(PARENT_CATEGORY = ifelse(str_starts(PARAMETER_CODE, "T") | str_starts(PARAMETER_CODE, "W"), 
                                  "Total Toxics", PARENT_CATEGORY),
         SUB_CATEGORY = ifelse(str_starts(PARAMETER_CODE, "T") | str_starts(PARAMETER_CODE, "W"), 
                               "", SUB_CATEGORY))

plot_pie_counts <- function(df, title) {
  # Inputs: df
  # Returns: none, plots figure
  
  category_counts <- table(df$PARENT_CATEGORY)
  
  pie(category_counts, 
      labels = ifelse(category_counts/sum(category_counts) > 0.04, 
                      paste0(round(100 * category_counts/sum(category_counts), 1), "%"), 
                      ""),
      main = title,
      col = rainbow(length(category_counts)),
      init.angle = 140)
  
  legend("topright", 
         legend = names(category_counts), 
         fill = rainbow(length(category_counts)),
         cex = 0.8,
         bty = "n")
}
        
plot_pie_counts(dmr_parameter_df, 'REF_Parameter Categories')
plot_pie_counts(ir_parameter_df, 'ir_parameter_df Categories')
plot_pie_counts(esmr_parameter_df, 'esmr_parameter_df Categories')
plot_pie_counts(toxics_parameter_df, 'toxics_parameter_df Categories')

### Create list of parameter codes, names, and matched ESMR parameter names

esmr_parameter_df <- esmr_parameter_df %>%
  mutate(normalized_desc = sapply(ESMR_PARAMETER_DESC, normalize_param_desc))


# print out the normalized desc and their data types for the first 20 rows
print(esmr_parameter_df %>% 
  head(20) %>% 
  select(ESMR_PARAMETER_DESC, normalized_desc) %>% 
  mutate(
    ESMR_PARAMETER_DESC_type = sapply(ESMR_PARAMETER_DESC, class),
    normalized_desc_type = sapply(normalized_desc, class)
  )
)

dmr_parameter_df <- dmr_parameter_df %>%
  rowwise() %>%
  mutate(ESMR_PARAMETER_DESC_MATCHED = match_parameter_desc(., esmr_params_df)) %>%
  ungroup()

cat(sprintf("%d out of %d parameter names automatically matched to ESMR PARAMETER_DESC in REF_PARAMETER.csv\n",
            length(unique(dmr_parameter_df$ESMR_PARAMETER_DESC_MATCHED)) - 1, nrow(dmr_parameter_df)))

dmr_esmr_mapping_manual <- read_csv('processed_data/step1/dmr_esmr_mapping_manual.csv', show_col_types = FALSE)
dmr_parameter_df <- dmr_parameter_df %>%
  left_join(dmr_esmr_mapping_manual %>% select(PARAMETER_CODE, ESMR_PARAMETER_DESC_MANUAL),
            by = "PARAMETER_CODE") %>%
  mutate(ESMR_PARAMETER_DESC_MANUAL = replace_na(ESMR_PARAMETER_DESC_MANUAL, ""))

cat(sprintf("%d out of %d parameter names manually mapped to ESMR PARAMETER_DESC in REF_PARAMETER.csv\n",
            length(unique(dmr_parameter_df$ESMR_PARAMETER_DESC_MANUAL)) - 1, nrow(dmr_parameter_df)))

dmr_parameter_df <- dmr_parameter_df %>%
  mutate(ESMR_PARAMETER_DESC = case_when(
    ESMR_PARAMETER_DESC_MATCHED != "" ~ ESMR_PARAMETER_DESC_MATCHED,
    ESMR_PARAMETER_DESC_MANUAL != "" ~ ESMR_PARAMETER_DESC_MANUAL,
    TRUE ~ ""
  )) %>%
  select(-ESMR_PARAMETER_DESC_MATCHED, -ESMR_PARAMETER_DESC_MANUAL) %>%
  rename(DMR_PARAMETER_DESC = PARAMETER_DESC)

write_csv(dmr_parameter_df, 'processed_data/step1/dmr_esmr_mapping.csv')