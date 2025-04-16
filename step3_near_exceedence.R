# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(tidyverse)
library(lubridate)
library(sf)

# Source helper functions
source("helper_functions.R")

# Load data
unique_parameter_codes <- read_csv('processed_data/step1/dmr_esmr_mapping.csv') %>% 
  pull(PARAMETER_CODE) %>% 
  unique()

# Define thresholds
slope_threshold <- 0.05
limit_threshold <- 0.1
fraction_threshold <- 0.5

current_year <- '2023'

create_facility_dict <- function(data_dict) {
  facility_dict <- list()

  # convert analysis_range to characters
  analysis_range <- as.character(analysis_range)
  # Pre-filter the data for selected pollutants
  filtered_data_dict <- map(analysis_range, ~data_dict[[current_year]] %>% 
                              filter(PARAMETER_CODE %in% unique_parameter_codes))
  
  count <- 0
  for (NPDES_code in unique(data_dict[[current_year]]$EXTERNAL_PERMIT_NMBR)) {
    count <- count + 1
    if (count %% 50 == 0) {
      cat(sprintf('%d facilities processed\n', count))
    }
    
    facility_dict[[NPDES_code]] <- list()
    
    for (parameter_code in unique_parameter_codes) {
      master_pollutant_data <- list()
      percent_near_exceedance <- list()
      
      for (year in analysis_range) {
        if (!parameter_code %in% unique(filtered_data_dict[[year]]$PARAMETER_CODE)) {
          next
        }
        
        year_pollutant_data <- filtered_data_dict[[year]] %>%
          filter(EXTERNAL_PERMIT_NMBR == NPDES_code, PARAMETER_CODE == parameter_code)
        
        master_pollutant_data <- append(master_pollutant_data, list(year_pollutant_data))
        
        if (nrow(year_pollutant_data) > 0) {
          percent_of_limit <- year_pollutant_data$DMR_VALUE_STANDARD_UNITS / year_pollutant_data$LIMIT_VALUE_STANDARD_UNITS
          qualifier <- year_pollutant_data$LIMIT_VALUE_QUALIFIER_CODE[1]
          
          if (qualifier %in% c('<=', '<')) {
            percent_near_exceedance[[as.character(year)]] <- sum(percent_of_limit > (1 - limit_threshold)) / length(percent_of_limit)
          } else if (qualifier %in% c('>=', '>')) {
            percent_near_exceedance[[as.character(year)]] <- sum(percent_of_limit < (1 + limit_threshold)) / length(percent_of_limit)
          }
        } else {
          percent_near_exceedance[[as.character(year)]] <- 0
        }
      }
      
      master_pollutant_data_df <- bind_rows(master_pollutant_data)
      date <- as.numeric(ymd(master_pollutant_data_df$MONITORING_PERIOD_END_DATE))
      datetimes <- ymd(master_pollutant_data_df$MONITORING_PERIOD_END_DATE)
      values <- as.numeric(master_pollutant_data_df$DMR_VALUE_STANDARD_UNITS)
      
      if (max(year(datetimes)) <= 2022) {
        date <- numeric(0)
        datetimes <- ymd(character(0))
        values <- numeric(0)
      }
      
      if (length(date) > 1) {
        fit <- lm(values ~ date)
        slope <- coef(fit)[2]
        intercept <- coef(fit)[1]
      } else {
        slope <- 0
        intercept <- 0
      }
      
      facility_dict[[NPDES_code]][[parameter_code]] <- list(
        percent_near_exceedance = percent_near_exceedance,
        slope = slope,
        intercept = intercept,
        data = master_pollutant_data_df,
        slope_dates = date,
        slope_datetimes = datetimes,
        slope_values = values,
        slope_intercept = intercept,
        limits = as.numeric(master_pollutant_data_df$LIMIT_VALUE_STANDARD_UNITS),
        qualifiers = master_pollutant_data_df$LIMIT_VALUE_QUALIFIER_CODE
      )
    }
  }
  
  return(facility_dict)
}

# Load or create facility_dict
if (file.exists('processed_data/step3/facility_dict.rds')) {
  facility_dict <- readRDS('processed_data/step3/facility_dict.rds')
} else {
  data_dict <- read_all_dmrs(save = FALSE, load = TRUE)
  facility_dict <- create_facility_dict(data_dict)
  saveRDS(facility_dict, 'processed_data/step3/facility_dict.rds')
}

# Print the combined length of data_dict[analysis_range]
sum <- 0
for (key in as.character(analysis_range)) {
  if (!is.null(data_dict[[key]])) {
    sum <- sum + nrow(data_dict[[key]])
  }
}
cat(sprintf('%s DMR events in data_dict\n', format(sum, big.mark = ",")))

# Analyze facilities
facilities_with_slope <- list()
facilities_with_near_exceedance <- list()

for (NPDES_code in unique(data_dict[[current_year]]$EXTERNAL_PERMIT_NMBR)) {
  for (parameter_code in unique_parameter_codes) {
    for (year in analysis_range) {
      current_year <- as.integer(format(Sys.Date(), "%Y"))
      if (max(facility_dict[[NPDES_code]][[parameter_code]]$data$MONITORING_PERIOD_END_DATE_NUMERIC) < current_year) {
        next
      }
      
      slope <- facility_dict[[NPDES_code]][[parameter_code]]$slope
      intercept <- facility_dict[[NPDES_code]][[parameter_code]]$intercept
      percent_near_exceedance <- facility_dict[[NPDES_code]][[parameter_code]]$percent_near_exceedance
      master_pollutant_data_df <- facility_dict[[NPDES_code]][[parameter_code]]$data
      date <- facility_dict[[NPDES_code]][[parameter_code]]$slope_dates
      values <- facility_dict[[NPDES_code]][[parameter_code]]$slope_values
      limits <- facility_dict[[NPDES_code]][[parameter_code]]$limits
      qualifiers <- facility_dict[[NPDES_code]][[parameter_code]]$qualifiers
      
      # Check for facilities with significant slope based on qualifier
      if (length(qualifiers) > 0) {
        if (qualifiers[1] %in% c('<=', '<') && slope > slope_threshold) {
          facilities_with_slope <- append(facilities_with_slope, list(c(NPDES_code, parameter_code, limits[1])))
        } else if (qualifiers[1] %in% c('>=', '>') && slope < -slope_threshold) {
          facilities_with_slope <- append(facilities_with_slope, list(c(NPDES_code, parameter_code, limits[1])))
        }
      }
      
      # Check for facilities with near exceedance
      if (any(unlist(percent_near_exceedance) > 0.5)) {
        facilities_with_near_exceedance <- append(facilities_with_near_exceedance, list(c(NPDES_code, parameter_code, limits[1])))
      }
    }
  }
}

facilities_with_slope_and_near_exceedance <- intersect(facilities_with_slope, facilities_with_near_exceedance)

cat(sprintf('%d facility-parameter pairs with slope > %s%%\n', length(facilities_with_slope), slope_threshold*100))
cat(sprintf('%d facility-parameter pairs with near exceedance > %s%% of limit more than %s%% of the time\n', 
            length(facilities_with_near_exceedance), limit_threshold*100, fraction_threshold*100))
cat(sprintf('%d facility-parameter pairs with both slope and near exceedance\n', length(facilities_with_slope_and_near_exceedance)))
cat(sprintf('%d facilities that have at least one parameter with slope and near exceedance\n', 
            length(unique(map_chr(facilities_with_slope_and_near_exceedance, ~.x[1])))))

facilities_with_slope_and_near_exceedance_df <- as_tibble(do.call(rbind, facilities_with_slope_and_near_exceedance)) %>%
  rename(NPDES_CODE = V1, PARAMETER_CODE = V2, LIMIT_VALUE_STANDARD_UNITS = V3)

# Plot timeseries for each facility-parameter combination
facilities_grouped <- facilities_with_slope_and_near_exceedance_df %>% group_by(NPDES_CODE)

for (NPDES_CODE in unique(facilities_grouped$NPDES_CODE)) {
  group <- facilities_grouped %>% filter(NPDES_CODE == !!NPDES_CODE)
  num_parameters <- nrow(group)
  
  p <- ggplot() +
    labs(title = sprintf('Facility: %s', NPDES_CODE)) +
    theme_minimal()
  
  for (i in 1:num_parameters) {
    row <- group[i,]
    PARAMETER_CODE <- as.character(row$PARAMETER_CODE)
    LIMIT_VALUE <- as.numeric(row$LIMIT_VALUE_STANDARD_UNITS)
    facility_data <- facility_dict[[NPDES_CODE]][[PARAMETER_CODE]]
    parameter_desc <- ref_parameter$PARAMETER_DESC[ref_parameter$PARAMETER_CODE == PARAMETER_CODE][1]
    qualifier <- facility_data$qualifiers[1]
    
    p <- p +
      geom_point(data = tibble(x = facility_data$slope_dates, y = facility_data$slope_values),
                 aes(x = x, y = y), alpha = 0.5) +
      geom_smooth(data = tibble(x = facility_data$slope_dates, y = facility_data$slope_values),
                  aes(x = x, y = y), method = "lm", se = FALSE, color = "red") +
      geom_hline(yintercept = LIMIT_VALUE, linetype = "dashed", color = "blue") +
      labs(subtitle = parameter_desc,
           x = "Date",
           y = "Value") +
      theme(legend.position = "none")
    
    if (i < num_parameters) {
      p <- p + theme(axis.title.x = element_blank(),
                     axis.text.x = element_blank())
    }
  }
  
  ggsave(sprintf('processed_data/step3/figures_R/%s.png', NPDES_CODE), p, width = 10, height = 2*num_parameters)
}

# Barplot of facilities frequently above limits
buffer_values <- c(limit_threshold)

# Count the number of parameters for each facility with slope and near exceedance
num_parameters_per_facility <- facilities_with_slope_and_near_exceedance_df %>%
  group_by(NPDES_CODE) %>%
  summarise(num_parameters = n_distinct(PARAMETER_CODE))

# Plot the facilities on a map of CA
ca_counties <- st_read('data/ca_counties/CA_Counties.shp')
facilities_list <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
facilities_with_coords <- facilities_list %>%
  select(`NPDES # CA#`, `LATITUDE DECIMAL DEGREES`, `LONGITUDE DECIMAL DEGREES`) %>%
  rename(NPDES_CODE = `NPDES # CA#`, LATITUDE = `LATITUDE DECIMAL DEGREES`, LONGITUDE = `LONGITUDE DECIMAL DEGREES`)

facilities_with_slope_and_near_exceedance_df <- facilities_with_slope_and_near_exceedance_df %>%
  left_join(facilities_with_coords, by = "NPDES_CODE")

facilities_gdf <- st_as_sf(facilities_with_slope_and_near_exceedance_df, 
                           coords = c("LONGITUDE", "LATITUDE"), 
                           crs = 4326)

if (st_crs(facilities_gdf) != st_crs(ca_counties)) {
  if (is.na(st_crs(ca_counties))) {
    ca_counties <- st_set_crs(ca_counties, st_crs(facilities_gdf))
  } else {
    facilities_gdf <- st_transform(facilities_gdf, st_crs(ca_counties))
  }
}

facilities_gdf <- facilities_gdf %>%
  left_join(num_parameters_per_facility, by = "NPDES_CODE")

ggplot() +
  geom_sf(data = ca_counties, fill = "lightgray", color = "white") +
  geom_sf(data = facilities_gdf, aes(color = num_parameters, size = num_parameters)) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Facilities with Parameters Having Slope and Near Exceedance",
       color = "Number of Parameters",
       size = "Number of Parameters")

ggsave("processed_figures/facilities_map.png", width = 10, height = 8)

# Make a df with NPDES_CODE and the number of parameters with slope and near exceedance
df <- num_parameters_per_facility %>%
  right_join(tibble(NPDES_CODE = unique(data_dict[[current_year]]$EXTERNAL_PERMIT_NMBR)), by = "NPDES_CODE") %>%
  replace_na(list(num_parameters = 0))

write_csv(df, 'processed_data/step3/num_parameters_per_facility.csv')

# Import ciwqs_facilities and merge df into ciwqs_facilities based on NPDES_CODE
ciwqs_facilities <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
ciwqs_facilities <- ciwqs_facilities %>%
  left_join(df, by = c("NPDES # CA#" = "NPDES_CODE"))

write_csv(ciwqs_facilities, 'processed_data/step3/facilities_with_near_exceedance.csv')

# Create pie chart
ggplot(ciwqs_facilities, aes(x = "", y = num_parameters)) +
  geom_bar(width = 1, stat = "count") +
  coord_polar("y", start = 0) +
  theme_void() +
  labs(title = "Number of Facilities in CIWQS facilities list with Slope and Near Exceedance")

ggsave("processed_figures/facilities_pie_chart.png", width = 8, height = 6)

# Create pie charts for each program category
program_categories <- ciwqs_facilities %>%
  filter(!is.na(`REG MEASURE TYPE`)) %>%
  pull(`REG MEASURE TYPE`) %>%
  unique()

plots <- list()

for (category in program_categories) {
  category_facilities <- ciwqs_facilities %>% filter(`REG MEASURE TYPE` == category)
  
  p <- ggplot(category_facilities, aes(x = "", y = num_parameters)) +
    geom_bar(width = 1, stat = "count") +
    coord_polar("y", start = 0) +
    theme_void() +
    labs(title = category,
         subtitle = sprintf("Total Facilities: %d", nrow(category_facilities)))
  
  plots[[category]] <- p
}

# Arrange plots in a grid
library(gridExtra)
do.call(grid.arrange, c(plots, ncol = 3))

ggsave("processed_figures/facilities_pie_charts_by_category.png", width = 12, height = 12)

## Print the most frequent parameter codes and their corresponding descriptions with value counts
dmr_esmr_mapping <- read_csv('processed_data/step1/dmr_esmr_mapping.csv')
most_frequent_parameter_codes <- facilities_with_slope_and_near_exceedance_df %>%
  count(PARAMETER_CODE, sort = TRUE)

for (i in 1:nrow(most_frequent_parameter_codes)) {
  code <- most_frequent_parameter_codes$PARAMETER_CODE[i]
  value_count <- most_frequent_parameter_codes$n[i]
  
  desc <- dmr_esmr_mapping %>%
    filter(PARAMETER_CODE == code) %>%
    pull(ESMR_PARAMETER_DESC)
  
  if (length(desc) > 0) {
    if (desc[1] == 'No Match') {
      parameter_desc <- dmr_esmr_mapping %>%
        filter(PARAMETER_CODE == code) %>%
        pull(DMR_PARAMETER_DESC)
      cat(sprintf('No Match: %d\n', value_count))
      cat(sprintf('    DMR: %s\n', parameter_desc))
    } else {
      cat(sprintf('%s: %d\n', desc[1], value_count))
    }
  } else {
    cat(sprintf('Unknown parameter (code %s): %d\n', code, value_count))
  }
}