import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs("processed_data/step2", exist_ok=True)


def load_cwns_data():
    """Download and load CWNS facilities data from GitHub"""
    logger.info("Downloading CWNS facilities data from GitHub...")

    # URL for the CWNS data file
    url = "https://raw.githubusercontent.com/dalyw/us-sewersheds/refs/heads/main/processed_data/facilities_2022_merged.csv"  # noqa: E501

    # Local file path
    local_file = "data/cwns/facilities_2022_merged.csv"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_file), exist_ok=True)

    # Download file if it doesn't exist locally
    if not os.path.exists(local_file):
        logger.info(f"Downloading {url} to {local_file}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(local_file, "wb") as f:
            f.write(response.content)
        logger.info(f"Successfully downloaded CWNS data to {local_file}")
    else:
        logger.info(f"Using existing CWNS data file: {local_file}")

    # Load the CSV file
    logger.info("Loading CWNS facilities data...")
    facilities_df = pd.read_csv(local_file)

    # Filter for California facilities only
    ca_facilities = facilities_df[facilities_df["STATE_CODE"] == "CA"].copy()
    logger.info(f"Loaded {len(ca_facilities)} California facilities from CWNS data")

    return ca_facilities


def load_covid_monitoring_data():
    """Load COVID monitoring dataset with population information"""
    logger.info("Loading COVID monitoring data...")
    return pd.read_csv("data/ww_surveillance/wastewatersurveillancecalifornia.csv")


def load_sso_data():
    """Load SSO Annual Report data with service population"""
    logger.info("Loading SSO data...")
    return pd.read_csv("data/sso/Questionnaire.txt", sep="\t")


def merge_population_data(facilities_df, covid_data, sso_data):
    """
    Merge population data from multiple sources and analyze discrepancies
    """
    logger.info("Merging population data from multiple sources...")

    # Create merged dataset with population from all sources
    merged_pop = (
        facilities_df[
            [
                "CWNS_ID",
                "PERMIT_NUMBER",
                "FACILITY_NAME",
                "TOTAL_RES_POPULATION_2022",
            ]
        ]
        .copy()
        .rename(columns={"TOTAL_RES_POPULATION_2022": "population_cwns"})
    )

    # Merge COVID monitoring population data
    logger.info("Processing COVID monitoring population data...")
    if "population_served" in covid_data.columns:
        # Check if epaid column contains lists instead of strings
        if covid_data["epaid"].apply(lambda x: isinstance(x, list)).any():
            logger.info("Found list values in epaid column, exploding to separate rows")
            # Explode the epaid column if it contains lists
            covid_data = covid_data.explode("epaid")

        # Ensure epaid is string type for merging
        covid_data["epaid"] = covid_data["epaid"].astype(str)
        merged_pop["PERMIT_NUMBER"] = merged_pop["PERMIT_NUMBER"].astype(str)

        # Create copy to avoid SettingWithCopyWarning
        covid_pop = (
            covid_data[["epaid", "population_served"]]
            .copy()
            .rename(columns={"population_served": "population_covid"})
        )

        # Drop duplicates if any exist after exploding
        covid_pop = covid_pop.drop_duplicates(subset=["epaid"])

        # Merge with facilities data
        merged_pop = merged_pop.merge(
            covid_pop, left_on="PERMIT_NUMBER", right_on="epaid", how="left"
        )
        logger.info(f"Merged COVID population data: {len(merged_pop)} rows")

    # Merge SSO questionnaire population data
    if "service_population" in sso_data.columns:
        sso_pop = (
            sso_data[["permit_number", "service_population"]]
            .copy()
            .rename(columns={"service_population": "population_sso"})
        )
        merged_pop = merged_pop.merge(
            sso_pop,
            left_on="PERMIT_NUMBER",
            right_on="permit_number",
            how="left",
        )

    # Calculate statistics and identify discrepancies
    pop_columns = [col for col in merged_pop.columns if "population" in col]
    merged_pop["population_mean"] = merged_pop[pop_columns].mean(axis=1)
    merged_pop["population_std"] = merged_pop[pop_columns].std(axis=1)
    merged_pop["population_cv"] = (
        merged_pop["population_std"] / merged_pop["population_mean"]
    )

    # Save merged population data
    merged_pop.to_csv("processed_data/step2/merged_population_data.csv", index=False)

    return merged_pop


def generate_visualizations(merged_pop):
    """Generate population data visualizations"""
    logger.info("Generating population visualizations...")

    # Create population distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(merged_pop["population_mean"].dropna(), bins=50)
    plt.xlabel("Population Served")
    plt.ylabel("Number of Facilities")
    plt.title("Distribution of Population Served by CA Wastewater Facilities")
    plt.savefig("processed_data/step2/population_distribution.png")
    plt.close()

    # Create population source comparison plot
    exclude_cols = {"population_mean", "population_std", "population_cv"}
    pop_sources = [
        col
        for col in merged_pop.columns
        if "population_" in col and col not in exclude_cols
    ]
    plt.figure(figsize=(10, 6))
    merged_pop[pop_sources].boxplot()
    plt.yscale("log")
    plt.ylabel("Population (log scale)")
    plt.title("Population Estimates by Data Source")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("processed_data/step2/population_source_comparison.png")
    plt.close()


def main():
    try:
        # Load and process all data sources
        facilities_df = load_cwns_data()
        covid_data = load_covid_monitoring_data()
        sso_data = load_sso_data()

        # Merge population data from all sources
        merged_pop = merge_population_data(facilities_df, covid_data, sso_data)

        # Generate visualizations
        generate_visualizations(merged_pop)

        logger.info("Population analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in population analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
