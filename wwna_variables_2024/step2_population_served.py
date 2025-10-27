import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import load_data, save_fig, setup_fig, STEP_DIRS


def main():
    # Load and process all data sources
    cwns_df = load_data("CWNS")
    covid_data = load_data("WW_SURVEILLANCE")
    sso_data = load_data("SSO")
    print(f"Loaded {len(cwns_df)} California facilities from CWNS data")

    # Load manual matches to prioritize PERMIT_NUMBERs that match
    manual_matches = pd.read_csv("data/manual_updates/cwns_facilities_match_manual.csv")
    manual_permit_numbers = set(manual_matches["PERMIT_NUMBER"].dropna().unique())
    manual_permit_no_clean = set(manual_matches["PERMIT_NO_clean"].dropna().unique())
    all_manual_permits = manual_permit_numbers | manual_permit_no_clean

    # Debug: Check for CWNS_IDs with multiple PERMIT_NUMBERs
    cwns_id_counts = cwns_df.groupby("CWNS_ID")["PERMIT_NUMBER"].nunique()
    multi_permit = cwns_id_counts[cwns_id_counts > 1]
    print(f"\nFound {len(multi_permit)} CWNS_IDs with multiple PERMIT_NUMBERs:")

    # Prefer PERMIT_NUMBERs that appear in manual matches
    for cwns_id in multi_permit.index[:5]:  # Show first 5 examples
        row = cwns_df[cwns_df["CWNS_ID"] == cwns_id][
            ["CWNS_ID", "PERMIT_NUMBER", "FACILITY_NAME", "population_cwns"]
        ].copy()
        # Check which permits have manual matches
        row["has_manual_match"] = row["PERMIT_NUMBER"].isin(all_manual_permits)

        matched_permits = row[row["has_manual_match"]]
        print(f"  CWNS_ID {cwns_id}:")
        print(
            row[
                [
                    "CWNS_ID",
                    "PERMIT_NUMBER",
                    "FACILITY_NAME",
                    "population_cwns",
                    "has_manual_match",
                ]
            ].to_string(index=False)
        )

        if len(matched_permits) > 1:
            print(
                f"    Multiple permits for {matched_permits.iloc[0]['PERMIT_NUMBER']}"
            )
        print()

    # Aggregate CWNS data: group by PERMIT_NUMBER, sum population
    cwns_agg = []
    for cwns_id, group in cwns_df.groupby("CWNS_ID"):
        if len(group) > 1:
            # Check which PERMIT_NUMBERs have manual matches
            has_match = group["PERMIT_NUMBER"].isin(all_manual_permits)
            if has_match.any():
                # Keep the first one that has a match
                group = group[has_match].iloc[:1]
            else:
                # Keep first if no matches
                group = group.iloc[:1]
        cwns_agg.append(group)

    cwns_df = pd.concat(cwns_agg, ignore_index=True)
    print(f"After dropping duplicate CWNS_IDs: {len(cwns_df)} rows")

    # Apply manual permit number mappings before merges
    manual_map = (
        manual_matches[["PERMIT_NUMBER", "PERMIT_NO_clean"]]
        .dropna(subset=["PERMIT_NO_clean"])
        .drop_duplicates()
        .set_index("PERMIT_NUMBER")["PERMIT_NO_clean"]
    )
    if len(manual_map) > 0:
        # Update PERMIT_NUMBER for facilities with manual mappings
        mask = cwns_df["PERMIT_NUMBER"].isin(manual_map.index)
        cwns_df.loc[mask, "PERMIT_NUMBER"] = cwns_df.loc[mask, "PERMIT_NUMBER"].map(
            manual_map
        )
        print(f"Applied {len(manual_map)} manual permit number mappings")

    # Merge COVID surveillance then SSO questionnaire population data
    merged_df = cwns_df.copy()
    kwargs = {"left_on": "PERMIT_NUMBER", "how": "left"}
    merged_df = merged_df.merge(covid_data, right_on="epaid", **kwargs)
    merged_df = merged_df.merge(sso_data, right_on="permit_number", **kwargs)

    # Classify facilities by data source
    has_cwns = merged_df["population_cwns"].notna()
    has_covid = merged_df["population_covid"].notna()
    has_sso = merged_df["population_sso"].notna()

    def get_source(row_idx):
        sources = []
        if has_cwns.iloc[row_idx]:
            sources.append("CWNS")
        if has_covid.iloc[row_idx]:
            sources.append("COVID")
        if has_sso.iloc[row_idx]:
            sources.append("SSO")
        return "+".join(sources) if sources else "Unmatched"

    merged_df["source"] = [get_source(i) for i in range(len(merged_df))]
    pie_data = merged_df["source"].value_counts().to_dict()

    # Create pie chart
    fig, ax = setup_fig(figsize=(10, 8))
    ax.pie(pie_data.values(), labels=pie_data.keys(), autopct="%1.1f%%")
    plt.title("Population Data Sources for Facilities")
    save_fig("figures_py/population_source_comparison.png", 2)

    # Calculate statistics and identify discrepancies
    pop_columns = [col for col in merged_df.columns if "population" in col]
    merged_df["Population Served"] = merged_df[pop_columns].mean(axis=1)
    merged_df["pop_std"] = merged_df[pop_columns].std(axis=1)

    # Population Histogram
    fig, ax = setup_fig()
    plt.hist(merged_df["Population Served"].dropna(), bins=50)
    plt.xlabel("Population Served")
    plt.ylabel("Number of Facilities")
    save_fig("figures_py/population_distribution.png", 2)

    # Save merged population data
    merged_df.to_csv(f"{STEP_DIRS[2]}/merged_population_data_py.csv", index=False)


if __name__ == "__main__":
    main()
