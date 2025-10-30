import pandas as pd
from collections import defaultdict
from helper_functions import (
    load_data,
    plot_barh,
    plot_map,
    aggregate_flagged_params,
    STEP_DIRS,
    WWNA_LIST,
    AGG_STRINGS,
    FILE_CONFIGS,
)


def check_impaired(x, water_bodies):
    """Check if watershed contains impaired water body and return matching IDs."""
    if pd.isna(x):
        return False, set()
    x_str = str(x)
    matching = {wb for wb in water_bodies if wb in x_str}
    return len(matching) > 0, matching


def main():

    # Load data
    limits_2024 = load_data("LIMITS", 2024)
    parameter_reference = pd.read_csv(f"{STEP_DIRS[1]}/ref_parameter_merged_py.csv")

    # Merge parameter categories into NPDES limits
    limits_2024["PARAMETER_CODE_CLEAN"] = limits_2024["PARAMETER_CODE"].str.lstrip("0")
    limits_2024 = limits_2024.merge(
        parameter_reference[
            ["PARAMETER_CODE_CLEAN", "PARENT_CATEGORY", "SUB_CATEGORY"]
        ],
        on="PARAMETER_CODE_CLEAN",
        how="left",
    )

    # Filter to limits with valid numerical LIMIT_VALUE_STANDARD_UNITS
    limits_with_values = limits_2024[
        (limits_2024["LIMIT_VALUE_STANDARD_UNITS"].notna())
        & (limits_2024["LIMIT_VALUE_STANDARD_UNITS"] != "")
    ]

    # Extract sub-categories that appear in CA LIMITS data with valid limits
    # Only analyze IR data for numerical limits (not monitoring-only)
    sub_categories = [
        cat for cat in limits_with_values["SUB_CATEGORY"].unique() if pd.notna(cat)
    ]
    print(f"{len(sub_categories)} categories in CA NPDES: {sorted(sub_categories)}")

    # Log unmapped parameters for categories that appear in LIMITS (with valid limits)
    limits_with_categories = limits_with_values[
        limits_with_values["SUB_CATEGORY"].notna()
    ]
    unmapped = limits_with_categories[limits_with_categories["PARENT_CATEGORY"].isna()][
        "PARAMETER_DESC"
    ].unique()
    if len(unmapped) > 0:
        print(f"Unmapped parameters in LIMITS: {unmapped}")

    # Load IR data (Integrated Report 303(d) lists)
    # Compares years from config to identify newly impaired water bodies
    ir_parameter_df = pd.read_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv")
    ir_303d = {}
    ir_keys = sorted(int(y) for y in FILE_CONFIGS["IR"]["year_config"].keys())
    ir_years = [ir_keys[0], ir_keys[-1]]
    for year in ir_years:
        df_year = load_data("IR", year=year)
        df_year = df_year.merge(
            ir_parameter_df[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
            left_on="Pollutant",
            right_on="IR_PARAMETER_DESC",
            how="left",
        )
        # Only keep pollutants in regulated categories (those that appear in LIMITS)
        df_year = df_year[df_year["SUB_CATEGORY"].isin(sub_categories)]
        unmapped = df_year[df_year["PARENT_CATEGORY"].isna()]["Pollutant"].unique()
        ir_303d[year] = df_year
        if len(unmapped) > 0:
            print(f"Unmapped pollutants in {year} data: {unmapped}")

    # Analyze impaired waters
    facilities = WWNA_LIST.copy()
    # Create dictionaries for impaired water bodies
    newly_impaired_bodies = defaultdict(set)
    impaired_water_bodies = defaultdict(set)
    impaired_sets = {}
    first_year = ir_years[0]
    last_year = ir_years[-1]
    for category in sub_categories:
        for year in ir_years:
            impaired_sets[year] = set(
                ir_303d[year].loc[
                    ir_303d[year]["SUB_CATEGORY"] == category,
                    "Water Body CALWNUMS",
                ]
            )
        newly_impaired_bodies[category] = (
            impaired_sets[last_year] - impaired_sets[first_year]
        )
        impaired_water_bodies[category] = impaired_sets[last_year]

    # Find facilities discharging into newly impaired waters that are not yet limited
    FLAGGED_STEP4_LIST = []
    for category in sub_categories:
        # Check each facility for newly impaired waterbodies in this category
        for idx in facilities.index:
            watershed_name = facilities.loc[idx, "CAL WATERSHED NAME"]
            matches, water_body_ids = check_impaired(
                watershed_name, newly_impaired_bodies[category]
            )

            if not matches:
                continue

            npdes = facilities.loc[idx, "NPDES # CA#"]
            sub_limits = limits_2024[limits_2024["EXTERNAL_PERMIT_NMBR"] == npdes]

            # Check if facility monitors parameters in this category
            has_params_in_category = any(sub_limits["SUB_CATEGORY"] == category)
            # Check if they have limits for those parameters
            has_limit = any(
                (sub_limits["SUB_CATEGORY"] == category)
                & (sub_limits["LIMIT_VALUE_STANDARD_UNITS"].notna())
                & (sub_limits["LIMIT_VALUE_STANDARD_UNITS"] != "")
            )

            # Flag facilities that monitor parameters in category but lack limits
            if has_params_in_category and not has_limit:
                FLAGGED_STEP4_LIST.append(
                    {
                        "NPDES # CA#": npdes,
                        "SUB_CATEGORY": category,
                        "Water Body ID": ", ".join(sorted(water_body_ids)),
                    }
                )

    # Create DataFrame and aggregate flagged facilities
    flagged_df = pd.DataFrame(FLAGGED_STEP4_LIST)

    # Aggregate categories per facility
    aggregated = aggregate_flagged_params(
        flagged_df, "NPDES # CA#", "SUB_CATEGORY", AGG_STRINGS["4"]
    )

    # Aggregate water body IDs per facility
    def combine_water_bodies(series):
        """Combine multiple water body ID strings, removing duplicates."""
        all_ids = []
        for water_body_str in series:
            if pd.notna(water_body_str) and water_body_str:
                all_ids.extend([wb.strip() for wb in water_body_str.split(", ")])
        return ", ".join(sorted(set(all_ids)))

    water_body_agg = (
        flagged_df.groupby("NPDES # CA#")["Water Body ID"]
        .apply(combine_water_bodies)
        .reset_index()
    )
    water_body_agg.columns = ["NPDES # CA#", "Water Body ID"]

    # Merge water body IDs into aggregated results
    aggregated = aggregated.merge(water_body_agg, on="NPDES # CA#", how="left")
    aggregated["Water Body ID"] = aggregated["Water Body ID"].fillna("")

    # Save aggregated results for RUN_ALL merge
    aggregated.to_csv(f"{STEP_DIRS[4]}/flagged_facilities_step4_py.csv", index=False)

    # Merge with WWNA_LIST for full facility data needed for visualizations
    flagged_facilities = WWNA_LIST[
        WWNA_LIST["NPDES # CA#"].isin(aggregated["NPDES # CA#"])
    ].copy()
    flagged_facilities = flagged_facilities.merge(
        aggregated, on="NPDES # CA#", how="inner"
    )

    # Bar plot of category counts
    all_categories = []
    for categories_str in flagged_facilities[AGG_STRINGS["4"]["PARAM"]]:
        all_categories.extend(cat.strip() for cat in categories_str.split(", "))
    category_counts = pd.Series(all_categories).value_counts().to_frame("count")
    category_counts.reset_index(inplace=True)
    category_counts.columns = ["Category", "count"]
    plot_barh(
        category_counts,
        x_col="count",
        y_col="Category",
        xlabel="Number of Facilities with Possible Future Limits",
        path="category_summary.png",
        step=4,
    )

    # Map of facilities with parameter counts
    param_counts = {
        npdes: num
        for npdes, num in zip(
            flagged_facilities["NPDES # CA#"],
            flagged_facilities[AGG_STRINGS["4"]["COUNT"]],
        )
    }
    plot_map(param_counts, 6, 4)


if __name__ == "__main__":
    main()
