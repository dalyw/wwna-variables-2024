import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from helper_functions import (
    read_limits,
    load_data,
    load_facilities_list,
    plot_facilities_map,
    save_and_close,
    aggregate_by_group,
    ref_parameter,
)

count_string = ("Discharges to Impaired and Not Limited: Number of Parameters",)
parameter_string = (
    "Parameters Discharged into Newly Impaired Water Body and Not Yet Limited"
)


def analyze_impaired_waters(
    facilities_list, limits_2023, impaired_303d, sub_categories
):
    """Analyze impaired waters and identify facilities requiring future limits."""

    # Create dictionaries for impaired water bodies
    newly_impaired_bodies = defaultdict(set)
    impaired_water_bodies = defaultdict(set)

    impaired_sets = {}
    for category in sub_categories:
        for year in [2018, 2024]:
            impaired_sets[year] = set(
                impaired_303d[year].loc[
                    impaired_303d[year]["SUB_CATEGORY"] == category,
                    "Water Body CALWNUMS",
                ]
            )
        newly_impaired_bodies[category] = impaired_sets[2024] - impaired_sets[2018]
        impaired_water_bodies[category] = impaired_sets[2024]

    # Find facilities discharging into newly impaired waters that are not yet limited
    flagged_facilities_list = []

    # Calculate masks for all categories
    def check_impaired(x, water_bodies):
        return any(wb in str(x) for wb in water_bodies) if pd.notna(x) else False

    for category in sub_categories:
        newly_impaired_mask = facilities_list["CAL WATERSHED NAME"].apply(
            check_impaired, water_bodies=newly_impaired_bodies[category]
        )

        # Check if facilities have limits for this category
        for idx in facilities_list[newly_impaired_mask].index:
            npdes = facilities_list.loc[idx, "NPDES # CA#"]
            sub_limits = limits_2023[limits_2023["EXTERNAL_PERMIT_NMBR"] == npdes]
            has_limit = any(
                (sub_limits["SUB_CATEGORY"] == category)
                & (sub_limits["LIMIT_VALUE_NMBR"].notna())
                & (sub_limits["LIMIT_VALUE_NMBR"] != "")
            )

            if not has_limit:
                # Note this facility needs limits for this category
                flagged_facilities_list.append(
                    {"NPDES # CA#": npdes, "SUB_CATEGORY": category}
                )

    # Create dataframe of flagged facilities
    if flagged_facilities_list:
        flagged_df = pd.DataFrame(flagged_facilities_list)
    else:
        flagged_df = pd.DataFrame(columns=["NPDES # CA#", "SUB_CATEGORY"])

    if len(flagged_df) > 0:
        aggregated = aggregate_by_group(
            flagged_df, "NPDES # CA#", "SUB_CATEGORY", count_string, parameter_string
        )

        # Merge aggregated results back to facilities_list
        facilities_list = facilities_list.merge(
            aggregated,
            left_on="NPDES # CA#",
            right_on="NPDES # CA#",
            how="left",
            suffixes=("", "_agg"),
        )

        # Fill NaN values
        facilities_list[parameter_string] = facilities_list[parameter_string].fillna("")
        facilities_list[count_string] = (
            facilities_list[count_string].fillna(0).astype(int)
        )

    return facilities_list


def generate_facility_plots(facilities_list, limits_2023):
    """Generate detailed plots for each facility."""
    impaired_facilities = facilities_list[facilities_list[parameter_string] != ""]

    for _, facility in impaired_facilities.iterrows():
        npdes_code = facility["NPDES # CA#"]
        categories_str = facility[parameter_string]

        if not categories_str:
            continue

        categories = [cat.strip() for cat in categories_str.split(", ")]

        n_params = len(categories)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols)

        for idx, category in enumerate(categories):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            param_data = limits_2023[
                (limits_2023["EXTERNAL_PERMIT_NMBR"] == npdes_code)
                & (limits_2023["SUB_CATEGORY"] == category)
            ]
            if len(param_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No {category} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(category, fontsize=10)
                continue

            for _, param_row in param_data.iterrows():
                param_desc = param_row["PARAMETER_DESC"]
                limit_value = param_row["LIMIT_VALUE_STANDARD_UNITS"]

                ax.axhline(y=limit_value, color="r", linestyle="--", alpha=0.5)
                for y1, y2, color in [
                    (0, limit_value, "lightgreen"),
                    (limit_value, limit_value * 2, "lightcoral"),
                ]:
                    ax.fill_between([-1, 1], [y1, y1], [y2, y2], color=color, alpha=0.3)

                ax.set_title(f"{param_desc}\n{category}", fontsize=10)
                ax.set_ylabel(param_row["STANDARD_UNIT_DESC"])

        save_and_close(f"figures_py/{npdes_code}_parameters.png", 4)


def generate_visualizations(facilities_list, limits_2023):
    """Generate visualizations for step4 results."""

    # Generate facility plots
    generate_facility_plots(facilities_list, limits_2023)

    # Create map visualization
    num_params_per_facility = {
        npdes: num
        for npdes, num in zip(
            facilities_list["NPDES # CA#"],
            facilities_list[count_string],
        )
        if num >= 1
    }

    plot_facilities_map(
        num_params_per_facility,
        "# of Parameters with\nPossible Future Limits",
        6,
    )

    # Create simple scatter plot of facilities
    facilities_with_coords = facilities_list[
        ["NPDES # CA#", "LATITUDE DECIMAL DEGREES", "LONGITUDE DECIMAL DEGREES"]
    ].copy()
    facilities_with_coords = facilities_with_coords.rename(
        columns={
            "LATITUDE DECIMAL DEGREES": "LATITUDE",
            "LONGITUDE DECIMAL DEGREES": "LONGITUDE",
        }
    )

    # Add parameter counts
    facilities_with_coords["Parameters"] = facilities_with_coords["NPDES # CA#"].map(
        lambda x: num_params_per_facility.get(x, 0)
    )

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        facilities_with_coords["LONGITUDE"],
        facilities_with_coords["LATITUDE"],
        c=facilities_with_coords["Parameters"],
        cmap="viridis",
        alpha=0.6,
    )
    plt.colorbar(label="Number of Parameters")
    plt.title("Facilities by Number of Parameters\nwith Possible Future Limits")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(
        f"STEP_DIRS{4}/figures_py/facilities_summary_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():

    # Load data
    facilities_list = load_facilities_list()
    limits_2023 = read_limits(2023)
    ir_parameter_df = pd.read_csv(f"STEP_DIRS{1}/ir_parameter_df.csv")

    #  Merge categories into limits df
    limits_2023["PARAMETER_CODE_CLEAN"] = limits_2023["PARAMETER_CODE"].str.lstrip("0")
    limits_2023 = limits_2023.merge(
        ref_parameter[["PARAMETER_CODE", "PARAMETER_DESC"]].rename(
            columns={"PARAMETER_DESC": "PARAMETER_DESC_CLEAN"}
        ),
        left_on="PARAMETER_CODE_CLEAN",
        right_on="PARAMETER_CODE",
        how="left",
        suffixes=("", "_ref"),
    )
    limits_2023 = limits_2023.merge(
        ir_parameter_df[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
        left_on="PARAMETER_DESC_CLEAN",
        right_on="IR_PARAMETER_DESC",
        how="left",
    )

    # Log unmapped parameters
    unmapped_params = limits_2023[limits_2023["PARENT_CATEGORY"].isna()][
        "PARAMETER_DESC"
    ].unique()
    if len(unmapped_params) > 0:
        print(f"Unmapped parameters: {unmapped_params}")

    # Get unique categories
    parent_categories = [
        cat for cat in ir_parameter_df["PARENT_CATEGORY"].unique() if pd.notna(cat)
    ]
    sub_categories = [
        cat for cat in ir_parameter_df["SUB_CATEGORY"].unique() if pd.notna(cat)
    ]
    print(f"{len(parent_categories)} parent, {len(sub_categories)} subcategories")

    # Load IR data
    impaired_303d = {}
    for year in [2018, 2024]:
        impaired_303d[year] = load_data("IR", year=year)
        impaired_303d[year] = impaired_303d[year].merge(
            ir_parameter_df[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
            left_on="Pollutant",
            right_on="IR_PARAMETER_DESC",
            how="left",
        )
        unmapped_pollutants = impaired_303d[year][
            impaired_303d[year]["PARENT_CATEGORY"].isna()
        ]["Pollutant"].unique()
        if len(unmapped_pollutants) > 0:
            print(f"Unmapped pollutants in {year} data: {unmapped_pollutants}")

    # Analyze impaired waters with updated parameters
    facilities_list = analyze_impaired_waters(
        facilities_list, limits_2023, impaired_303d, sub_categories
    )

    # Save results
    facilities_list.to_csv(f"STEP_DIRS{4}/flagged_facilities_step4.csv", index=False)
    generate_visualizations(facilities_list, limits_2023)


if __name__ == "__main__":
    main()
