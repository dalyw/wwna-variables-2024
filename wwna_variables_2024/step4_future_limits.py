import pandas as pd
from collections import defaultdict
from helper_functions import (
    load_data,
    plot_barh,
    plot_map,
    aggregate_flags,
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


def combine_water_bodies(series):
    all_ids = []
    for wb_str in series:
        if pd.notna(wb_str) and wb_str:
            all_ids.extend([wb.strip() for wb in wb_str.split(", ")])
    return ", ".join(sorted(set(all_ids)))


def main():
    # Load and merge parameter categories into NPDES limits
    limits_2024 = load_data("LIMITS", 2024, dropna=False)
    param_ref = pd.read_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv")
    limits_2024 = limits_2024.merge(
        param_ref[["PARAMETER_CODE", "PARENT_CATEGORY", "SUB_CATEGORY"]],
        on="PARAMETER_CODE",
        how="left",
    )

    # Extract categories with valid limits (for IR filtering)
    limits_with_values = limits_2024[
        limits_2024["LIMIT_VALUE_STANDARD_UNITS"].notna()
        & (limits_2024["LIMIT_VALUE_STANDARD_UNITS"] != "")
    ]
    sub_categories = limits_with_values["SUB_CATEGORY"].dropna().unique().tolist()
    print(f"{len(sub_categories)} categories in CA NPDES: {sorted(sub_categories)}")

    # Log unmapped parameters
    unmapped = limits_with_values[limits_with_values["PARENT_CATEGORY"].isna()][
        "PARAMETER_DESC"
    ].unique()
    if len(unmapped) > 0:
        print(f"Unmapped parameters in LIMITS: {unmapped}")

    # Load IR data and merge categories
    ir_param_df = pd.read_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv")
    ir_keys = sorted(int(y) for y in FILE_CONFIGS["IR"]["year_config"].keys())
    ir_years = [ir_keys[0], ir_keys[-1]]
    ir_303d = {}
    for year in ir_years:
        df_year = load_data("IR", year=year, rename=False).merge(
            ir_param_df[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
            left_on="Pollutant",
            right_on="IR_PARAMETER_DESC",
            how="left",
        )
        df_year = df_year[df_year["SUB_CATEGORY"].isin(sub_categories)]
        ir_303d[year] = df_year
        unmapped = df_year[df_year["PARENT_CATEGORY"].isna()]["Pollutant"].unique()
        if len(unmapped) > 0:
            print(f"Unmapped pollutants in {year} data: {unmapped}")

    # Identify newly impaired water bodies
    facilities = WWNA_LIST.copy()
    newly_impaired_bodies = defaultdict(set)
    first_year, last_year = ir_years[0], ir_years[-1]
    for category in sub_categories:
        impaired_first = set(
            ir_303d[first_year].loc[
                ir_303d[first_year]["SUB_CATEGORY"] == category, "Water Body CALWNUMS"
            ]
        )
        impaired_last = set(
            ir_303d[last_year].loc[
                ir_303d[last_year]["SUB_CATEGORY"] == category, "Water Body CALWNUMS"
            ]
        )
        newly_impaired_bodies[category] = impaired_last - impaired_first

    # Find facilities discharging into newly impaired waters that are not yet limited
    FLAGGED_STEP4_LIST = []
    for category in sub_categories:
        impaired_set = newly_impaired_bodies[category]
        for idx in facilities.index:
            watershed_name = facilities.loc[idx, "CAL WATERSHED NAME"]
            matches, water_body_ids = check_impaired(watershed_name, impaired_set)
            if not matches:
                continue

            npdes = facilities.loc[idx, "NPDES # CA#"]
            facility_limits = limits_2024[limits_2024["EXTERNAL_PERMIT_NMBR"] == npdes]
            if (facility_limits["SUB_CATEGORY"] == category).any():
                FLAGGED_STEP4_LIST.append(
                    {
                        "NPDES # CA#": npdes,
                        "SUB_CATEGORY": category,
                        "Water Body ID": ", ".join(sorted(water_body_ids)),
                    }
                )

    # Aggregate flagged facilities
    flagged_df = pd.DataFrame(FLAGGED_STEP4_LIST)

    # Aggregate water body IDs
    water_body_agg = (
        flagged_df.groupby("NPDES # CA#")["Water Body ID"]
        .apply(combine_water_bodies)
        .reset_index()
    )
    water_body_agg.columns = ["NPDES # CA#", "Water Body ID"]

    # Aggregate flagged params
    aggregated = aggregate_flags(flagged_df, "NPDES # CA#", "SUB_CATEGORY", 4)
    aggregated = aggregated.merge(water_body_agg, on="NPDES # CA#", how="left")
    aggregated["Water Body ID"] = aggregated["Water Body ID"].fillna("")
    aggregated.to_csv(f"{STEP_DIRS[4]}/flagged_facilities_step4_py.csv", index=False)

    # Merge with WWNA_LIST for full facility data needed for visualizations
    flagged_facilities = WWNA_LIST[
        WWNA_LIST["NPDES # CA#"].isin(aggregated["NPDES # CA#"])
    ].merge(aggregated, on="NPDES # CA#", how="inner")

    # Bar plot of category counts
    all_categories = [
        cat.strip()
        for cats_str in flagged_facilities[AGG_STRINGS["4"]["PARAM"]]
        for cat in cats_str.split(", ")
    ]
    category_counts = pd.Series(all_categories).value_counts().reset_index()
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
    plot_map(
        dict(
            zip(
                flagged_facilities["NPDES # CA#"],
                flagged_facilities[AGG_STRINGS["4"]["COUNT"]],
            )
        ),
        6,
        4,
    )


if __name__ == "__main__":
    main()
