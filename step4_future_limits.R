# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(sf)
library(gridExtra)

source("helper_functions.R")

# Read and process data
dmr_esmr_mapping <- read_csv('processed_data/step1/dmr_esmr_mapping.csv')
parent_category_map <- setNames(dmr_esmr_mapping$PARENT_CATEGORY, dmr_esmr_mapping$PARAMETER_CODE)
sub_category_map <- setNames(dmr_esmr_mapping$SUB_CATEGORY, dmr_esmr_mapping$PARAMETER_CODE)

categories <- c('Temperature', 'Metals', 'Dissolved Solids', 'Nitrogen', 'Phosphorus', 
                'Disinfectants', 'Dissolved Oxygen', 'Pathogens', 'Toxic Inorganics',
                'Turbidity', 'Color')

facilities_list <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')

limits_2023 <- read_limits(2023) %>%
  mutate(
    PARENT_CATEGORY = parent_category_map[PARAMETER_CODE],
    SUB_CATEGORY = sub_category_map[PARAMETER_CODE]
  )

# Import and process 303d lists for 2018 and 2024
columns_to_keep <- c('Water Body CALWNUMS', 'Pollutant', 'Pollutant Category', 'Decision Status', 
                     'TMDL Requirement Status', 'Sources', 'Expected TMDL Completion Date', 
                     'Expected Attainment Date')

impaired_303d_2018 <- read_csv('data/ir/2018-303d.csv', skip = 2) %>%
  select(all_of(columns_to_keep)) %>%
  filter(!is.na(`Water Body CALWNUMS`))

impaired_303d_2024 <- read_csv('data/ir/2024-303d.csv', skip = 1) %>%
  select(all_of(columns_to_keep)) %>%
  filter(!is.na(`Water Body CALWNUMS`))

# Create lists to store the newly impaired water bodies for each category
newly_impaired_water_bodies <- list()
impaired_water_bodies <- list()

for (category in categories) {
  impaired_set_2018 <- impaired_303d_2018 %>%
    filter(`Pollutant Category` == category) %>%
    pull(`Water Body CALWNUMS`) %>%
    unique()
  
  impaired_set_2024 <- impaired_303d_2024 %>%
    filter(`Pollutant Category` == category) %>%
    pull(`Water Body CALWNUMS`) %>%
    unique()
  
  newly_impaired_water_bodies[[category]] <- setdiff(impaired_set_2024, impaired_set_2018)
  impaired_water_bodies[[category]] <- impaired_set_2024
}

# Helper function to check if watershed contains impaired water body
contains_impaired_water_body <- function(watershed_name, impaired_water_bodies) {
  if (is.na(watershed_name)) return(FALSE)
  any(sapply(impaired_water_bodies, function(x) grepl(x, watershed_name)))
}

# Check for facilities discharging to newly impaired water bodies
for (category in categories) {
  print(sprintf('Analyzing %s', category))
  
  facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]] <- 
    sapply(facilities_list$`CAL WATERSHED NAME`, 
           contains_impaired_water_body, 
           impaired_water_bodies = newly_impaired_water_bodies[[category]]) %>%
    as.logical()
  
  facilities_list[[paste0('Discharges to ', category, ' Impaired')]] <- 
    sapply(facilities_list$`CAL WATERSHED NAME`, 
           contains_impaired_water_body, 
           impaired_water_bodies = impaired_water_bodies[[category]]) %>%
    as.logical()
  
  facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]] <- FALSE
  
  for (i in 1:nrow(facilities_list)) {
    sub_limits_2023 <- limits_2023 %>% 
      filter(EXTERNAL_PERMIT_NMBR == facilities_list$`NPDES # CA#`[i])
    
    if (isTRUE(facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]][i]) || 
        facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]][i] == 'TRUE') {
      for (j in 1:nrow(sub_limits_2023)) {
        if (isTRUE(sub_limits_2023$SUB_CATEGORY[j] == category) && 
            (is.na(sub_limits_2023$LIMIT_VALUE_NMBR[j]) || 
             identical(sub_limits_2023$LIMIT_VALUE_NMBR[j], ''))) {
          facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i] <- TRUE
          break
        }
      }
      if (!facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i]) {
        facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i] <- TRUE
      }
    }
  }
}

# Consolidate to a single column
facilities_list <- facilities_list %>%
  mutate(`Discharges to Impaired Water Bodies and Not Limited` = 
           apply(select(., starts_with("Discharges to Newly") & ends_with("Impaired and Not Limited")), 1, 
                 function(x) paste(names(x)[x], collapse = " and ")))

# Save the processed data
write_csv(facilities_list, 'processed_data/step4/facilities_with_future_limits.csv')

# Create plots in R and save to file
num_categories <- length(categories)
row_titles <- c('Discharges into\nListed', 'Discharges into\nNewly Listed', 'Newly Listed and\nnot Yet Limited')

plot_list <- list()

for (i in seq_along(categories)) {
  category <- categories[i]
  
  # Plot facilities discharging to any impaired water bodies
  data1 <- table(facilities_list[[paste0('Discharges to ', category, ' Impaired')]])
  p1 <- ggplot(as.data.frame(data1), aes(x="", y=Freq, fill=Var1)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y", start=0) +
    theme_void() +
    ggtitle(row_titles[1]) +
    scale_fill_manual(values=c("TRUE"="skyblue", "FALSE"="pink"))
  
  # Plot facilities discharging to newly impaired water bodies
  data2 <- table(facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]])
  p2 <- ggplot(as.data.frame(data2), aes(x="", y=Freq, fill=Var1)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y", start=0) +
    theme_void() +
    ggtitle(row_titles[2]) +
    scale_fill_manual(values=c("TRUE"="skyblue", "FALSE"="pink"))
  
  # Plot facilities discharging to newly impaired water bodies and not limited
  data3 <- table(facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]])
  p3 <- ggplot(as.data.frame(data3), aes(x="", y=Freq, fill=Var1)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y", start=0) +
    theme_void() +
    ggtitle(row_titles[3]) +
    scale_fill_manual(values=c("TRUE"="skyblue", "FALSE"="pink"))
  
  plot_list[[i]] <- arrangeGrob(p1, p2, p3, ncol=1, top=grid::textGrob(category, gp=grid::gpar(fontsize=12, fontface="bold")))
}

# Combine all plots
final_plot <- do.call(grid.arrange, c(plot_list, ncol=num_categories))

# Save the plot
ggsave("processed_data/step4/figures/impaired_water_bodies_plot.png", final_plot, width=2.5*num_categories, height=8, units="in")


## 
plot_facilities_discharge <- function(sub_category) {
  # Separate facilities with and without coordinates
  facilities_with_coords <- facilities_list %>%
    filter(!is.na(`LONGITUDE DECIMAL DEGREES`) & !is.na(`LATITUDE DECIMAL DEGREES`))
  
  facilities_without_coords <- facilities_list %>%
    filter(is.na(`LONGITUDE DECIMAL DEGREES`) | is.na(`LATITUDE DECIMAL DEGREES`))
  
  # Create an sf object from facilities with coordinates
  gdf <- st_as_sf(facilities_with_coords, 
                  coords = c("LONGITUDE DECIMAL DEGREES", "LATITUDE DECIMAL DEGREES"),
                  crs = 4326) %>%
    mutate(missing_coords = FALSE)
  
  # Create a data frame for facilities without coordinates to be placed at the top
  top_row <- st_sf(
    geometry = st_sfc(st_point(c(-124, 42)), crs = 4326),  # Northwest corner of California
    missing_coords = TRUE
  )
  
  # Add all other columns from gdf to top_row, filling with NA
  top_row[setdiff(names(gdf), names(top_row))] <- NA
  # Combine the sf object and the top row
  gdf <- rbind(
    st_sf(top_row),
    gdf %>% mutate(missing_coords = FALSE)
  ) %>%
    st_sf()
  
  # Add a column to track facilities without coordinates
  gdf$missing_coords[1] <- nrow(facilities_without_coords) > 0

  california <- st_read('data/ca_counties/CA_Counties.shp')

  # If CRS don't match, reproject one to match the other
  if (st_crs(gdf) != st_crs(california)) {
    if (is.na(st_crs(california))) {
      california <- st_set_crs(california, st_crs(gdf))
    } else {
      gdf <- st_transform(gdf, st_crs(california))
    }
  }

  # plot county outlines and facilities
  ggplot() +
    geom_sf(data = california, fill = 'lightgrey', color = 'black') +
    geom_sf(data = gdf[gdf[[paste0('Discharges to Newly ', sub_category, ' Impaired and Not Limited')]] == FALSE, ], 
            color = 'blue', size = 1, alpha = 0.5) +
    geom_sf(data = gdf[gdf[[paste0('Discharges to Newly ', sub_category, ' Impaired and Not Limited')]] == TRUE, ], 
            color = 'red', size = 1, alpha = 0.5) +
    labs(title = paste('Facilities Discharging to Newly', sub_category, 'Impaired Water Bodies')) +
    theme_void() +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(override.aes = list(size = 3)))

  ggsave(paste0("processed_data/step4/figures/facilities_discharging_", sub_category, ".png"), 
         width = 6, height = 6, units = "in")
}


# Generate plots for all sub-categories
sub_categories <- c('Temperature', 'Metals', 'Dissolved Solids', 'Nitrogen', 'Phosphorus', 
                    'Disinfectants', 'Dissolved Oxygen', 'Pathogens', 'Toxic Inorganics',
                    'Turbidity', 'Color')

for (sub_category in sub_categories) {
  plot_facilities_discharge(sub_category)
}