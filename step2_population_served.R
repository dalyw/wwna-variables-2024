# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(tidyverse)
# Install and load the 'sf' package
if (!requireNamespace("sf", quietly = TRUE)) {
  install.packages("sf")
}
library(sf)

# Import Data

# CIWQS
ciwqs_facilities <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
cat(sprintf("Length of full list: %d\n", nrow(ciwqs_facilities)))
cat(sprintf(" Unique FACILITY ID: %d\n", n_distinct(ciwqs_facilities$`FACILITY ID`)))
cat(sprintf(" Unique WDID: %d\n", n_distinct(ciwqs_facilities$WDID)))
cat(sprintf(" Unique ORDER #: %d\n", n_distinct(ciwqs_facilities$`ORDER #`)))
ciwqs_facilities <- ciwqs_facilities %>% distinct(WDID, .keep_all = TRUE)

# CWNS
facilities <- read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITIES.csv', locale = locale(encoding = "latin1")) %>%
  select(CWNS_ID, FACILITY_NAME)

facility_permits <- read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITY_PERMIT.csv', locale = locale(encoding = "latin1")) %>%
  select(-FACILITY_ID, -STATE_CODE)

pop_served_cwns <- bind_rows(
  read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_WASTEWATER.csv', locale = locale(encoding = "latin1")),
  read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_WASTEWATER_CONFIRMED.csv', locale = locale(encoding = "latin1")),
  read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_DECENTRALIZED.csv', locale = locale(encoding = "latin1"))
)

# WW Surveillance
pop_served_ww_surveillance <- read_csv('data/ww_surveillance/wastewatersurveillancecalifornia.csv') %>%
  select(epaid, population_served) %>%
  drop_na() %>%
  distinct(epaid, .keep_all = TRUE)

# SSO Questionnaire
questionnaire <- read_tsv('data/sso/Questionnaire.txt') %>%
  select(Wdid, `SSOq Population Served`)

# Merge Facilities list with CWNS and WW Surveillance Datasets

# clean CWNS population data
cat(sprintf("%d CWNS facilities\n", nrow(facilities)))
facilities <- facilities %>% left_join(facility_permits, by = "CWNS_ID")
cat(sprintf("%d CWNS facilities after merging with facility_permits\n", nrow(facilities)))
facilities <- facilities %>% filter(!str_detect(tolower(FACILITY_NAME), "stormwater"))
cat(sprintf("%d CWNS facilities after dropping stormwater in name\n", nrow(facilities)))

patterns_to_remove <- c("WDR ", "WDR-", "WDR", "Order WQ ", "WDR Order No. ", "Order No. ", "Order ", "NO. ", "ORDER NO. ", "NO.", "ORDER ", "DWQ- ", "NO.·", ". ")
replacements <- c("·" = "-", "\\?" = "-")

facilities <- facilities %>%
  mutate(PERMIT_NUMBER_cwns_clean = str_replace_all(as.character(PERMIT_NUMBER), paste(patterns_to_remove, collapse = "|"), "")) %>%
  mutate(PERMIT_NUMBER_cwns_clean = str_replace_all(PERMIT_NUMBER_cwns_clean, replacements)) %>%
  filter(PERMIT_NUMBER != "2006-0003-DWQ")

cat(sprintf("%d CWNS facilities after dropping 2006-0003-DWQ from PERMIT_NUMBER\n", nrow(facilities)))

# POPULATION
# This code block processes the population data from the Clean Watersheds Needs Survey (CWNS)
pop_served_cwns <- pop_served_cwns %>%
  # Join the population data with facility information
  left_join(facilities, by = "CWNS_ID") %>%
  # Group by CWNS_ID to aggregate data for each facility
  group_by(CWNS_ID) %>%
  summarise(
    # Calculate total population served by summing 2022 residential population, ignoring NAs
    POPULATION_SERVED = sum(TOTAL_RES_POPULATION_2022, na.rm = TRUE),
    # Take the first permit number (assuming it's the same for all rows in a group)
    PERMIT_NUMBER = first(PERMIT_NUMBER),
    # Take the first cleaned permit number
    PERMIT_NUMBER_cwns_clean = first(PERMIT_NUMBER_cwns_clean)
  ) %>%
  # Filter out facilities with zero population served
  filter(POPULATION_SERVED > 0)

cat(sprintf("%d CWNS facilities after merging with pop served and cleaning\n", nrow(pop_served_cwns)))

# Merge with CIWQS facilities list
cat(sprintf("%d CIWQS WDIDs\n", n_distinct(ciwqs_facilities$WDID)))

pop_served_cwns_check <- ciwqs_facilities %>%
  left_join(pop_served_cwns, by = c("NPDES # CA#" = "PERMIT_NUMBER_cwns_clean")) %>%
  mutate(POPULATION_SOURCE = "CWNS")

cat(sprintf("%d CIWQS WDIDs after merge with cwns_facilities\n", n_distinct(pop_served_cwns_check$WDID)))

# Merge with WW Surveillance data
pop_served_cwns_check <- pop_served_cwns_check %>%
  left_join(pop_served_ww_surveillance, by = c("NPDES # CA#" = "epaid")) %>%
  mutate(
    POPULATION_SERVED = coalesce(POPULATION_SERVED, population_served),
    POPULATION_SOURCE = if_else(is.na(POPULATION_SERVED) & !is.na(population_served), "WW Surveillance", POPULATION_SOURCE)
  ) %>%
  select(-population_served)

cat(sprintf("%d CIWQS WDIDs after merging with pop_served_ww_surveillance\n", n_distinct(pop_served_cwns_check$WDID)))

# Merge with SSO Questionnaire data
pop_served_cwns_check <- pop_served_cwns_check %>%
  left_join(questionnaire, by = c("WDID" = "Wdid")) %>%
  mutate(
    POPULATION_SERVED = coalesce(POPULATION_SERVED, `SSOq Population Served`),
    POPULATION_SOURCE = if_else(is.na(POPULATION_SERVED) & !is.na(`SSOq Population Served`), "SSO Questionnaire", POPULATION_SOURCE)
  ) %>%
  select(-`SSOq Population Served`)

# Fill in missing population data with design flow * 100
pop_served_cwns_check <- pop_served_cwns_check %>%
  mutate(
    POPULATION_SERVED = if_else(is.na(POPULATION_SERVED), `DESIGN FLOW` * 100, POPULATION_SERVED),
    POPULATION_SOURCE = if_else(is.na(POPULATION_SOURCE) & !is.na(POPULATION_SERVED), "Design Flow * 100", POPULATION_SOURCE)
  )

# Plotting
library(ggplot2)

# Pie chart of fraction of population served by source
pop_counts <- pop_served_cwns_check %>%
  count(POPULATION_SOURCE) %>%
  mutate(percentage = n / sum(n) * 100)

ggplot(pop_counts, aes(x = "", y = percentage, fill = POPULATION_SOURCE)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  theme_void() +
  labs(title = "Fraction of Population Served by Source") +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), position = position_stack(vjust = 0.5)) +
  annotate("text", x = 0, y = 0, label = sprintf("Total Facilities: %d", nrow(pop_served_cwns_check)))

# Get unique program categories
program_categories <- pop_served_cwns_check %>%
  filter(!is.na(`REG MEASURE TYPE`)) %>%
  pull(`REG MEASURE TYPE`) %>%
  unique()

# Create a list to store plots
plots <- list()

for (category in program_categories) {
  category_facilities <- pop_served_cwns_check %>% filter(`REG MEASURE TYPE` == category)
  category_pop_counts <- category_facilities %>%
    count(POPULATION_SOURCE) %>%
    mutate(percentage = n / sum(n) * 100)
  
  p <- ggplot(category_pop_counts, aes(x = "", y = percentage, fill = POPULATION_SOURCE)) +
    geom_bar(stat = "identity", width = 1) +
    coord_polar("y", start = 0) +
    theme_void() +
    labs(title = category) +
    geom_text(aes(label = sprintf("%.1f%%", percentage)), position = position_stack(vjust = 0.5)) +
    annotate("text", x = 0, y = -1, label = sprintf("Total Facilities: %d", nrow(category_facilities)))
  
  plots[[category]] <- p
}

# Arrange plots in a grid
# Install and load gridExtra package
if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra")
}
library(gridExtra)
do.call(grid.arrange, c(plots, ncol = 3))

# Sum up population values
cat(sprintf("Total Population Served for facilities with non-nan population served: %s\n",
            format(sum(pop_served_cwns_check$POPULATION_SERVED[!is.na(pop_served_cwns_check$POPULATION_SERVED)], na.rm = TRUE), big.mark = ",")))

cat(sprintf("Total Design Flow for all facilities: %.0f MGD\n",
            sum(pop_served_cwns_check$`DESIGN FLOW`, na.rm = TRUE)))

cat(sprintf("Total Design Flow for facilities with non-nan population served: %.0f MGD\n",
            sum(pop_served_cwns_check$`DESIGN FLOW`[!is.na(pop_served_cwns_check$POPULATION_SERVED)], na.rm = TRUE)))

cat(sprintf("Percentage of total design flow where population served is not nan: %.2f%%\n",
            sum(pop_served_cwns_check$`DESIGN FLOW`[!is.na(pop_served_cwns_check$POPULATION_SERVED)], na.rm = TRUE) / 
            sum(pop_served_cwns_check$`DESIGN FLOW`, na.rm = TRUE) * 100))

# Save plots to file
pdf('processed_data/step2/population_served_plots.pdf')
for (category in program_categories) {
  print(plots[[category]])
}
dev.off()

# Save the results
write_csv(pop_served_cwns_check, 'processed_data/step2/facilities_list_with_population_served.csv')