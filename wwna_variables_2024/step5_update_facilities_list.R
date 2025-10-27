# converted from .ipynb files using Claude 3.5 Sonnet

library(dplyr)
library(readr)

# Read original facilities list and processed data
facilities_list <- read_csv('data/wwna_list/NPDES+WDR Facilities List_20240906.csv')
facilities_list_with_population_served <- read_csv('processed_data/step2/facilities_list_with_population_served.csv')
facilities_list_with_num_parameters <- read_csv('processed_data/step3/facilities_with_near_exceedance.csv')
facilities_list_with_future_limits <- read_csv('processed_data/step4/flagged_facilities_step4.csv')

# Map the population served to the facilities_list
facilities_list <- facilities_list %>%
  left_join(facilities_list_with_population_served %>% 
              select(`FACILITY ID`, POPULATION_SERVED, POPULATION_SOURCE),
            by = "FACILITY ID") %>%
  rename(`Population Served` = POPULATION_SERVED,
         `Population Data Source` = POPULATION_SOURCE)

# Map the num_parameters to the facilities_list
facilities_list <- facilities_list %>%
  left_join(facilities_list_with_num_parameters %>% 
              select(`FACILITY ID`, num_parameters),
            by = "FACILITY ID") %>%
  rename(`Number of Parameters with Slope and Near Exceedance` = num_parameters)

# Map the future limits to the facilities_list
facilities_list <- facilities_list %>%
  left_join(facilities_list_with_future_limits %>% 
              select(`FACILITY ID`, `Discharges to Impaired Water Bodies and Not Limited`),
            by = "FACILITY ID")

# Save the updated facilities_list
write_csv(facilities_list, 'processed_data/wwna_list_updated.csv')
