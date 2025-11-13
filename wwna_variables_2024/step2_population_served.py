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

    # Get permit numbers for prioritization
    manual_permit_numbers = set(manual_matches["PERMIT_NUMBER"].dropna().unique())
    manual_permit_no_clean = set(manual_matches["PERMIT_NO_clean"].dropna().unique())
    all_manual_permits = manual_permit_numbers | manual_permit_no_clean

    # Create mapping dictionary from PERMIT_NUMBER to PERMIT_NO_clean
    manual_map = (
        manual_matches[["PERMIT_NUMBER", "PERMIT_NO_clean"]]
        .dropna(subset=["PERMIT_NO_clean"])
        .drop_duplicates()
        .set_index("PERMIT_NUMBER")["PERMIT_NO_clean"]
        .to_dict()
    )

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
    if len(manual_map) > 0:
        # Apply mapping where PERMIT_NUMBER != mapped value (permit number is updated)
        for permit, cleaned in manual_map.items():
            if permit != cleaned:
                cwns_df.loc[cwns_df["PERMIT_NUMBER"] == permit, "PERMIT_NUMBER"] = (
                    cleaned
                )
        non_identity_mappings = sum(1 for k, v in manual_map.items() if k != v)
        if non_identity_mappings > 0:
            print(f"Applied {non_identity_mappings} manual permit number mappings")

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
    save_fig("population_source_comparison.png", 2)

    # Calculate statistics and identify discrepancies
    pop_columns = [col for col in merged_df.columns if "population" in col]
    merged_df["Population Served"] = merged_df[pop_columns].mean(axis=1)
    merged_df["pop_std"] = merged_df[pop_columns].std(axis=1).round(2)

    # Calculate annualized population growth rate from 2022 to 2042 (20-year period)
    # Using compound annual growth rate (CAGR): ((end/start)^(1/years) - 1) * 100
    from_cwns = merged_df["source"].str.contains("CWNS", na=False)
    years = 20  # 2022 to 2042
    merged_df.loc[from_cwns, "population_growth_rate"] = (
        (
            (
                merged_df.loc[from_cwns, "population_cwns_2042"]
                / merged_df.loc[from_cwns, "population_cwns"]
            )
            ** (1 / years)
            - 1
        )
        * 100
    ).round(2)

    # Population Histogram
    fig, ax = setup_fig()
    plt.hist(merged_df["Population Served"].dropna(), bins=50)
    plt.xlabel("Population Served")
    plt.ylabel("Number of Facilities")
    save_fig("population_distribution.png", 2)

    # Deduplicate by PERMIT_NUMBER before saving
    # (some facilities have multiple CWNS records after merging)
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["PERMIT_NUMBER"], keep="first")
    if initial_rows != len(merged_df):
        print(f"Deduplicated population data: {initial_rows} -> {len(merged_df)} rows")

    # Save merged population data
    merged_df.to_csv(f"{STEP_DIRS[2]}/merged_population_data_py.csv", index=False)


if __name__ == "__main__":
    main()
