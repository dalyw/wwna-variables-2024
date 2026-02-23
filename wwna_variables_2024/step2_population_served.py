import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import load_data, save_fig, setup_fig, STEP_DIRS, WWNA_LIST


def main():
    """Estimate population served per WWNA facility using priority: CWNS > SSO > COVID.

    Starts from the full WWNA facility list and merges three population sources:
    - CWNS: matched via NPDES # CA# and ORDER # (from manual matching table)
    - SSO: matched via WDID (SSOQ_TO_WDID_1 = receiving treatment plant)
    - COVID wastewater surveillance: matched via NPDES # CA# (epaid)
    """

    # Load population data sources
    cwns_df = load_data("CWNS")
    covid_data = load_data("WW_SURVEILLANCE")
    sso_raw = load_data("SSO")
    sso_data = sso_raw.groupby("WDID")["population_sso"].sum().reset_index()
    print(f"Loaded {len(cwns_df)} CWNS, {len(covid_data)} COVID, {len(sso_data)} SSO records")

    # Load manual CWNS-to-WWNA matching table
    manual_matches = pd.read_csv("data/manual_updates/cwns_facilities_match_manual.csv")
    cwns_to_wwna = manual_matches[
        ["PERMIT_NUMBER", "NPDES # CA#", "ORDER #"]
    ].drop_duplicates(subset=["PERMIT_NUMBER"])
    cwns_df = cwns_df.merge(cwns_to_wwna, on="PERMIT_NUMBER", how="left")

    # Start from full WWNA facility list
    merged_df = WWNA_LIST[["NPDES # CA#", "ORDER #", "WDID", "FACILITY NAME"]].copy()
    print(f"WWNA facilities: {len(merged_df)}")

    # --- CWNS: two-pass merge (NPDES first, then ORDER for unmatched) ---
    cwns_cols = ["population_cwns", "population_cwns_2042"]
    cwns_npdes = cwns_df[cwns_df["NPDES # CA#"].notna()][["NPDES # CA#"] + cwns_cols]
    cwns_order = cwns_df[cwns_df["ORDER #"].notna()][["ORDER #"] + cwns_cols]

    m_npdes = merged_df.merge(cwns_npdes, on="NPDES # CA#", how="left")
    m_order = merged_df.merge(cwns_order, on="ORDER #", how="left")
    for col in cwns_cols:
        merged_df[col] = m_npdes[col].combine_first(m_order[col])

    # --- SSO: merge via WDID ---
    merged_df = merged_df.merge(sso_data, on="WDID", how="left")

    # --- COVID: merge via NPDES (epaid) ---
    # Uppercase epaid to match WWNA NPDES format (COVID data has mixed case)
    covid_data["epaid"] = covid_data["epaid"].str.upper()
    covid_data = covid_data.drop_duplicates(subset=["epaid"])
    merged_df = merged_df.merge(
        covid_data, left_on="NPDES # CA#", right_on="epaid", how="left"
    )

    # Label each facility with which sources provided population data
    source_flags = {
        "CWNS": merged_df["population_cwns"].notna(),
        "SSO": merged_df["population_sso"].notna(),
        "COVID": merged_df["population_covid"].notna(),
    }
    merged_df["source"] = [
        "+".join(name for name, flag in source_flags.items() if flag.iloc[i])
        or "No data"
        for i in range(len(merged_df))
    ]

    # Population priority: CWNS > SSO > COVID
    merged_df["Population Served"] = (
        merged_df["population_cwns"]
        .combine_first(merged_df["population_sso"])
        .combine_first(merged_df["population_covid"])
    )

    n_pop = merged_df["Population Served"].notna().sum()
    print(f"Facilities with population data: {n_pop} of {len(merged_df)}")
    print(merged_df["source"].value_counts().to_string())

    # Side-by-side pie charts: NPDES facilities vs ALL facilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    npdes_mask = merged_df["NPDES # CA#"].notna()
    for ax, mask, title in [
        (ax1, npdes_mask, f"NPDES Facilities (n={npdes_mask.sum()})"),
        (ax2, pd.Series(True, index=merged_df.index), f"All Facilities (n={len(merged_df)})"),
    ]:
        pie_data = merged_df.loc[mask, "source"].value_counts()
        ax.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%")
        ax.set_title(title)

    plt.suptitle("Population Data Sources for WWNA Facilities")
    plt.tight_layout()
    save_fig("population_source_comparison.png", 2)

    # Compute annualized population growth rate (CAGR)
    # CWNS provides a 2042 projection and 2022 current values
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

    # Histogram of population served across all facilities
    fig, ax = setup_fig()
    plt.hist(merged_df["Population Served"].dropna(), bins=50, log=True)
    plt.xlabel("Population Served")
    plt.ylabel("Number of Facilities")
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,}"))
    save_fig("population_distribution.png", 2)

    merged_df.to_csv(f"{STEP_DIRS[2]}/merged_population_data_py.csv", index=False)


if __name__ == "__main__":
    main()
