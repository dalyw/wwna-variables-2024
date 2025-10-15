import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import read_limits, ref_parameter
import json
from collections import defaultdict
import numpy as np
import os
from wwna_variables_2024.plotting_functions import (
    generate_facility_plots,
    plot_facilities_map,
)

# Create output directory if it doesn't exist
os.makedirs("processed_data/step4", exist_ok=True)


def load_and_process_data():
    """Load and process all required data."""
    # Load facilities list
    facilities_list = pd.read_csv(
        "data/facilities_list/NPDES+WDR Facilities List_20240906.csv"
    )

    # Load and process 2023 limits
    limits_2023 = read_limits(2023)

    # Load IR parameter data and create category maps
    ir_parameter_df = pd.read_csv("processed_data/step1/ir_parameter_df.csv")

    # Map categories to limits using parameter descriptions
    parameter_desc_map = dict(
        zip(
            ref_parameter["PARAMETER_CODE"].str.lstrip("0"),
            ref_parameter["PARAMETER_DESC"],
        )
    )
    desc_to_parent = dict(
        zip(
            ir_parameter_df["IR_PARAMETER_DESC"],
            ir_parameter_df["PARENT_CATEGORY"],
        )
    )
    desc_to_sub = dict(
        zip(
            ir_parameter_df["IR_PARAMETER_DESC"],
            ir_parameter_df["SUB_CATEGORY"],
        )
    )

    # Apply mappings to limits
    limits_2023["PARAMETER_DESC_CLEAN"] = (
        limits_2023["PARAMETER_CODE"].str.lstrip("0").map(parameter_desc_map)
    )
    limits_2023["PARENT_CATEGORY"] = limits_2023["PARAMETER_DESC_CLEAN"].map(
        desc_to_parent
    )
    limits_2023["SUB_CATEGORY"] = limits_2023["PARAMETER_DESC_CLEAN"].map(
        desc_to_sub
    )

    # Log unmapped parameters
    unmapped_params = limits_2023[limits_2023["PARENT_CATEGORY"].isna()][
        "PARAMETER_DESC"
    ].unique()
    if len(unmapped_params) > 0:
        print(f"Unmapped parameters: {unmapped_params}")

    # Get unique categories, removing any None values
    parent_categories = [
        cat
        for cat in ir_parameter_df["PARENT_CATEGORY"].unique()
        if pd.notna(cat)
    ]
    sub_categories = [
        cat
        for cat in ir_parameter_df["SUB_CATEGORY"].unique()
        if pd.notna(cat)
    ]

    print(
        f"{len(parent_categories)} parent, {len(sub_categories)} subcategories"
    )

    # Load 303d lists
    columns_to_keep = [
        "Water Body CALWNUMS",
        "Pollutant",
        "Pollutant Category",
        "Decision Status",
        "TMDL Requirement Status",
        "Sources",
        "Expected TMDL Completion Date",
        "Expected Attainment Date",
    ]

    impaired_303d = {}
    for year, rows_to_skip in [(2018, 2), (2024, 1)]:
        impaired_303d[year] = pd.read_csv(
            f"data/ir/{year}-303d.csv", skiprows=rows_to_skip
        )[columns_to_keep].dropna(subset=["Water Body CALWNUMS"])
        impaired_303d[year]["PARENT_CATEGORY"] = impaired_303d[year][
            "Pollutant"
        ].map(desc_to_parent)
        impaired_303d[year]["SUB_CATEGORY"] = impaired_303d[year][
            "Pollutant"
        ].map(desc_to_sub)
        unmapped_pollutants = set(impaired_303d[year]["Pollutant"]) - set(
            desc_to_parent.keys()
        )
        if unmapped_pollutants:
            print(f"Unmapped pollutants in {year} data: {unmapped_pollutants}")

    with open("data/manual_updates/parameter_sorting_dict.json", "r") as f:
        parameter_sorting_dict = json.load(f)
    return (
        facilities_list,
        limits_2023,
        impaired_303d,
        parameter_sorting_dict,
        parent_categories,
        sub_categories,
    )


def analyze_impaired_waters(
    facilities_list,
    limits_2023,
    impaired_303d,
    parameter_sorting_dict,
    parent_categories,
    sub_categories,
):
    """Analyze impaired waters and
    identify facilities requiring future limits."""

    # Create dictionaries for impaired water bodies
    newly_impaired_water_bodies = defaultdict(set)
    impaired_water_bodies = defaultdict(set)

    for category in sub_categories:
        impaired_set_2018 = set(
            impaired_303d[2018].loc[
                impaired_303d[2018]["SUB_CATEGORY"] == category,
                "Water Body CALWNUMS",
            ]
        )
        impaired_set_2024 = set(
            impaired_303d[2024].loc[
                impaired_303d[2024]["SUB_CATEGORY"] == category,
                "Water Body CALWNUMS",
            ]
        )
        newly_impaired_water_bodies[category] = (
            impaired_set_2024 - impaired_set_2018
        )
        impaired_water_bodies[category] = impaired_set_2024

    # Create all column names first
    all_categories = parent_categories + sub_categories
    column_names = []
    for category in all_categories:
        column_names.extend(
            [
                f"Discharges to Newly {category} Impaired",
                f"Discharges to {category} Impaired",
                f"Discharges to Newly {category} Impaired and Not Limited",
            ]
        )
    column_names.extend(
        [
            "Discharges to Impaired Water Bodies and Not Limited",
            "Discharges to Impaired and Not Limited: Number of Parameters",
        ]
    )

    # Pre-allocate all columns with zeros/empty strings
    new_data = pd.DataFrame(
        0, index=facilities_list.index, columns=column_names
    )
    new_data["Discharges to Impaired Water Bodies and Not Limited"] = ""

    # Process all categories at once
    for category in all_categories:
        # Calculate masks for the whole dataset at once
        def check_impaired(x, water_bodies):
            return (
                any(wb in str(x) for wb in water_bodies)
                if pd.notna(x)
                else False
            )

        newly_impaired_mask = facilities_list["CAL WATERSHED NAME"].apply(
            check_impaired, water_bodies=newly_impaired_water_bodies[category]
        )
        impaired_mask = facilities_list["CAL WATERSHED NAME"].apply(
            check_impaired, water_bodies=impaired_water_bodies[category]
        )

        # Update columns using masks
        new_data[f"Discharges to Newly {category} Impaired"] = (
            newly_impaired_mask.astype(int)
        )
        new_data[f"Discharges to {category} Impaired"] = impaired_mask.astype(
            int
        )

        # Check limits for facilities with newly impaired waters
        for index in facilities_list[newly_impaired_mask].index:
            npdes = facilities_list.loc[index, "NPDES # CA#"]
            sub_limits = limits_2023[
                limits_2023["EXTERNAL_PERMIT_NMBR"] == npdes
            ]
            has_limit = any(
                (sub_limits["SUB_CATEGORY"] == category)
                & (sub_limits["LIMIT_VALUE_NMBR"].notna())
                & (sub_limits["LIMIT_VALUE_NMBR"] != "")
                & (sub_limits["LIMIT_VALUE_NMBR"] != "nan")
            )
            if not has_limit:
                new_data.loc[
                    index,
                    f"Discharges to Newly {category} Impaired and Not Limited",
                ] = 1

    # Calculate summary columns
    impaired_categories = []
    for category in sub_categories:
        col_name = f"Discharges to Newly {category} Impaired and Not Limited"
        total = new_data[col_name].sum()

        # Handle different types of total values
        if isinstance(total, pd.Series):
            total = total.iloc[0] if len(total) == 1 else total.sum()
        elif isinstance(total, (np.ndarray, np.generic)):
            total = total.item() if total.size == 1 else total.sum()

        if total > 0:
            impaired_categories.append(category)

    # Update summary columns using vectorized operations
    def get_impaired_categories(row):
        categories = [
            cat
            for cat in sub_categories
            if row[f"Discharges to Newly {cat} Impaired and Not Limited"] > 0
        ]
        return ", ".join(categories)

    new_data["Discharges to Impaired Water Bodies and Not Limited"] = (
        new_data.apply(get_impaired_categories, axis=1)
    )

    # Calculate total parameters per facility
    parameter_cols = [
        f"Discharges to Newly {cat} Impaired and Not Limited"
        for cat in sub_categories
    ]

    # Ensure all columns are numeric before summing
    for col in parameter_cols:
        if col in new_data.columns:
            try:
                new_data[col] = new_data[col].astype(float)
            except Exception:
                new_data[col] = (
                    new_data[col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .astype(float)
                )

    new_data[
        "Discharges to Impaired and Not Limited: Number of Parameters"
    ] = new_data[parameter_cols].sum(axis=1)

    # Combine original data with new columns efficiently
    facilities_list = pd.concat([facilities_list, new_data], axis=1)

    return facilities_list, newly_impaired_water_bodies


def generate_visualizations(
    facilities_list,
    newly_impaired_water_bodies,
    sub_categories,
    parent_categories,
    limits_2023,
):
    """Generate visualizations of the analysis results."""
    # Create bar plot
    all_categories = sub_categories + parent_categories
    data = {}
    for category in all_categories:
        if len(newly_impaired_water_bodies[category]) > 0:
            data[category] = {
                "Discharges to Listed": facilities_list[
                    f"Discharges to {category} Impaired"
                ].sum(),
                "Newly Listed and Not Yet Limited": facilities_list[
                    f"Discharges to Newly {category} Impaired and Not Limited"
                ].sum(),
            }

    if not data:
        return

    df = pd.DataFrame(data).T
    df_sorted = df.sort_values(by=df.columns.tolist(), ascending=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_sorted.index))
    width = 0.35

    # Use colormaps directly instead of get_cmap
    ax.bar(
        x - width / 2,
        df_sorted["Discharges to Listed"],
        width,
        label="Discharging to Listed\nWater Body",
        color=plt.colormaps["viridis"](0.2),
    )
    ax.bar(
        x + width / 2,
        df_sorted["Newly Listed and Not Yet Limited"],
        width,
        label="Discharging to Newly Listed\nWater Body and\nNot Yet Limited",
        color=plt.colormaps["viridis"](0.8),
    )

    plt.ylabel("Number of Facilities", fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.xticks(x, df_sorted.index, rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(df_sorted["Discharges to Listed"]):
        ax.text(i - width / 2, v, str(int(v)), ha="center", va="bottom")
    for i, v in enumerate(df_sorted["Newly Listed and Not Yet Limited"]):
        ax.text(i + width / 2, v, str(int(v)), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(
        "processed_data/step4/facilities_with_future_limits_efficient.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Generate facility plots
    os.makedirs("processed_data/step4/facility_plots", exist_ok=True)
    generate_facility_plots(
        facilities_list,
        newly_impaired_water_bodies,
        limits_2023,
        sub_categories,
    )

    num_parameters_per_facility = dict(
        zip(
            facilities_list["NPDES # CA#"],
            facilities_list[
                "Discharges to Impaired and Not Limited: Number of Parameters"
            ],
        )
    )
    num_parameters_per_facility = {
        k: v for k, v in num_parameters_per_facility.items() if v >= 1
    }

    plot_facilities_map(
        num_parameters_per_facility,
        "# of Parameters with\nPossible Future Limits",
        6,
    )

    # Create simple scatter plot of facilities
    facilities_with_coords = facilities_list[
        [
            "NPDES # CA#",
            "LATITUDE DECIMAL DEGREES",
            "LONGITUDE DECIMAL DEGREES",
        ]
    ].copy()
    facilities_with_coords = facilities_with_coords.rename(
        columns={
            "NPDES # CA#": "NPDES_CODE",
            "LATITUDE DECIMAL DEGREES": "LATITUDE",
            "LONGITUDE DECIMAL DEGREES": "LONGITUDE",
        }
    )

    # Add parameter counts
    facilities_with_coords["Parameters"] = facilities_with_coords[
        "NPDES_CODE"
    ].map(lambda x: num_parameters_per_facility.get(x, 0))

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
    plt.title(
        "Facilities by Number of Parameters\nwith Possible Future Limits"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(
        "processed_data/step4/figures_py/facilities_summary_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(generate_plots=True):
    # Load and process data with updated return values
    (
        facilities_list,
        limits_2023,
        impaired_303d,
        parameter_sorting_dict,
        parent_categories,
        sub_categories,
    ) = load_and_process_data()

    # Analyze impaired waters with updated parameters
    facilities_list, newly_impaired_water_bodies = analyze_impaired_waters(
        facilities_list,
        limits_2023,
        impaired_303d,
        parameter_sorting_dict,
        parent_categories,
        sub_categories,
    )

    # Save results
    facilities_list.to_csv(
        "processed_data/step4/facilities_with_future_limits.csv", index=False
    )

    # Generate visualizations if requested
    if generate_plots:
        generate_visualizations(
            facilities_list,
            newly_impaired_water_bodies,
            sub_categories,
            parent_categories,
            limits_2023,
        )


if __name__ == "__main__":
    main(generate_plots=True)
