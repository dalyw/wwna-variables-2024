import matplotlib.pyplot as plt
from helper_functions import load_data, save_and_close, setup_figure


def merge_population_source(merged_pop, pop_data, right_on):
    merged_pop = merged_pop.merge(
        pop_data, left_on="PERMIT_NUMBER", right_on=right_on, how="left"
    )
    return merged_pop


def plot_population_source_comparison(merged_pop):
    """Plot comparison of population data sources."""
    pop_sources = [
        col
        for col in merged_pop.columns
        if "population_" in col and col not in ["population_mean", "population_std"]
    ]
    fig, ax = setup_figure()
    merged_pop[pop_sources].boxplot()
    plt.yscale("log")
    plt.ylabel("Population (log scale)")
    plt.title("Population Estimates by Data Source")
    plt.xticks(rotation=45)
    save_and_close("figures_py/population_source_comparison.png", 2)


def generate_visualizations(merged_pop):
    """Generate population data visualizations"""

    # Population Histogram
    fig, ax = setup_figure()
    plt.hist(merged_pop["Population Served"].dropna(), bins=50)
    plt.xlabel("Population Served")
    plt.ylabel("Number of Facilities")
    plt.title("Distribution of Population Served by CA Wastewater Facilities")
    save_and_close("population_distribution.png", 2)

    # Population data source comparison
    exclude_cols = {"Population Served", "pop_std"}
    pop_sources = [
        col for col in merged_pop.columns if "pop" in col and col not in exclude_cols
    ]
    plt.figure(figsize=(10, 6))
    merged_pop[pop_sources].boxplot()
    plt.yscale("log")
    plt.ylabel("Population (log scale)")
    plt.title("Population Estimates by Data Source")
    plt.xticks(rotation=45)
    save_and_close("population_source_comparison.png", 2)


def main():
    # Load and process all data sources
    try:
        facilities_df = load_data("CWNS")
        covid_data = load_data("WW_SURVEILLANCE")
        sso_data = load_data("SSO")
        print(f"Loaded {len(facilities_df)} California facilities from CWNS data")

        # Merge COVID surveillance then SSO questionnaire population data
        merged_pop = facilities_df.copy()
        merged_pop = merge_population_source(merged_pop, covid_data, "epaid")
        merged_pop = merge_population_source(merged_pop, sso_data, "permit_number")

        # Calculate statistics and identify discrepancies
        pop_columns = [col for col in merged_pop.columns if "population" in col]
        merged_pop["Population Served"] = merged_pop[pop_columns].mean(axis=1)
        merged_pop["pop_std"] = merged_pop[pop_columns].std(axis=1)

        # Save merged population data
        merged_pop.to_csv(f"STEP_DIRS{2}/merged_population_data.csv", index=False)

        generate_visualizations(merged_pop)

    except Exception as e:
        print(f"Error in population analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
