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
)


def check_impaired(x, water_bodies):
    return any(wb in str(x) for wb in water_bodies) if pd.notna(x) else False


def main():

    # Load data
    limits_2023 = load_data("LIMITS", 2023)
    parameter_reference = pd.read_csv(f"{STEP_DIRS[1]}/ref_parameter_merged_py.csv")

    # Merge parameter categories into NPDES limits
    limits_2023["PARAMETER_CODE_CLEAN"] = limits_2023["PARAMETER_CODE"].str.lstrip("0")
    limits_2023 = limits_2023.merge(
        parameter_reference[
            ["PARAMETER_CODE_CLEAN", "PARENT_CATEGORY", "SUB_CATEGORY"]
        ],
        on="PARAMETER_CODE_CLEAN",
        how="left",
    )

    sub_categories = [
        cat for cat in parameter_reference["SUB_CATEGORY"].unique() if pd.notna(cat)
    ]

    # Load categories to exclude from future limits analysis
    exclude_df = pd.read_csv(
        "data/manual_updates/categories_to_exclude_from_future_limits.csv"
    )
    exclude_categories = set(exclude_df["SUB_CATEGORY"].values)
    sub_categories = [cat for cat in sub_categories if cat not in exclude_categories]
    if len(exclude_categories) > 0:
        print(
            f"Excluding {len(exclude_categories)} categories: {sorted(exclude_categories)}"
        )

    # Filter out excluded categories before checking unmapped parameters
    limits_filtered = limits_2023[~limits_2023["SUB_CATEGORY"].isin(exclude_categories)]

    # Log unmapped parameters for included categories
    unmapped_params = limits_filtered[limits_filtered["PARENT_CATEGORY"].isna()][
        "PARAMETER_DESC"
    ].unique()
    if len(unmapped_params) > 0:
        print(f"Unmapped parameters: {unmapped_params}")

    # Load IR data (Integrated Report 303(d) lists)
    # Compares 2018 vs 2024 to identify newly impaired water bodies
    ir_parameter_df = pd.read_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv")
    ir_303d = {}
    for year in [2018, 2024]:
        df_year = load_data("IR", year=year)
        df_year = df_year.merge(
            ir_parameter_df[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
            left_on="Pollutant",
            right_on="IR_PARAMETER_DESC",
            how="left",
        )
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
    for category in sub_categories:
        for year in [2018, 2024]:
            impaired_sets[year] = set(
                ir_303d[year].loc[
                    ir_303d[year]["SUB_CATEGORY"] == category,
                    "Water Body CALWNUMS",
                ]
            )
        newly_impaired_bodies[category] = impaired_sets[2024] - impaired_sets[2018]
        impaired_water_bodies[category] = impaired_sets[2024]

    # Find facilities discharging into newly impaired waters that are not yet limited
    FLAGGED_STEP4_LIST = []
    for category in sub_categories:
        # Filter to facilities discharging into newly impaired waterbodies for this category
        newly_impaired_mask = facilities["CAL WATERSHED NAME"].apply(
            check_impaired, water_bodies=newly_impaired_bodies[category]
        )

        # Check each affected facility
        for idx in facilities[newly_impaired_mask].index:
            npdes = facilities.loc[idx, "NPDES # CA#"]
            sub_limits = limits_2023[limits_2023["EXTERNAL_PERMIT_NMBR"] == npdes]

            # Check if facility monitors parameters in this category
            has_params_in_category = any(sub_limits["SUB_CATEGORY"] == category)
            # Check if they have limits for those parameters
            has_limit = any(
                (sub_limits["SUB_CATEGORY"] == category)
                & (sub_limits["LIMIT_VALUE_NMBR"].notna())
                & (sub_limits["LIMIT_VALUE_NMBR"] != "")
            )

            # Flag facilities that monitor parameters in category but lack limits
            if has_params_in_category and not has_limit:
                FLAGGED_STEP4_LIST.append(
                    {"NPDES # CA#": npdes, "SUB_CATEGORY": category}
                )

    # Create DataFrame and aggregate flagged facilities
    flagged_df = pd.DataFrame(FLAGGED_STEP4_LIST)
    aggregated = aggregate_flagged_params(
        flagged_df, "NPDES # CA#", "SUB_CATEGORY", AGG_STRINGS["4"]
    )

    # Save aggregated results for RUN_ALL merge
    aggregated.to_csv(f"{STEP_DIRS[4]}/flagged_facilities_step4_py.csv", index=False)

    # Merge with WWNA_LIST for full facility data needed for visualizations
    flagged_facilities = WWNA_LIST[
        WWNA_LIST["NPDES # CA#"].isin(aggregated["NPDES # CA#"])
    ].copy()
    flagged_facilities = flagged_facilities.merge(
        aggregated, on="NPDES # CA#", how="inner"
    )

    # Generate visualizations
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
        xlabel="Number of Facilities",
        title="Facilities Needing Limits by Category",
        path="figures_py/category_summary.png",
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
