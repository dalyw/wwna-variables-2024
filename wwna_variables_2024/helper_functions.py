import pandas as pd
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd

# IMPORT DMR AND ESMR DATA
analysis_range = range(2014, 2024)
save = False
load = True

# Data download configuration
ESMR_RESOURCE_IDS = {
    2014: "c0f64b3f-d921-4eb9-aa95-af1827e5033e",
    2015: "81c399d4-f661-4808-8e6b-8e543281f1c9",
    2016: "aacfe728-f063-452c-9dca-63482cc994ad",
    2017: "44d1f39c-f21b-4060-8225-c175eaea129d",
    2018: "bb3b3d85-44eb-4813-bbf9-ea3a0e623bb7",
    2019: "2eaa2d55-9024-431e-b902-9676db949174",
    2020: "4fa56f3f-7dca-4dbd-bec4-fe53d5823905",
    2021: "28d3a164-7cec-4baf-9b11-7a9322544cd6",
    2022: "8c6296f7-e226-42b7-9605-235cd33cdee2",
    2023: "65eb7023-86b6-4960-b714-5f6574d43556",
    2024: "7adb8aea-62fb-412f-9e67-d13b0729222f",
    2025: "176a58bf-6f5d-4e3f-9ed9-592a509870eb",
}

# Path constants for processed data directories
STEP_DIRS = {}
for i in range(4):
    STEP_DIRS[i + 1] = f"processed_data/step{i+1}"

# Facilities list path
FACILITIES_LIST_PATH = "data/facilities_list/NPDES+WDR Facilities List_20240906.csv"

# Import list of NPDES codes for permits
# in the filtered full facilities flat file
facilities_list = pd.read_csv(FACILITIES_LIST_PATH)
npdes_from_facilities_list = (
    facilities_list[facilities_list["NPDES # CA#"].notna()]["NPDES # CA#"]
    .unique()
    .tolist()
)

ref_parameter = pd.read_csv("data/dmr/REF_PARAMETER.csv")
ref_parameter.set_index("PARAMETER_CODE")


with open("wwna_variables_2024/file_configs.json", "r") as f:
    FILE_CONFIGS = json.load(f)
# Convert dtype strings to actual Python types
dtype_map = {
    "str": str,
    "float": float,
    "int": int,
    "bool": bool,
}
for data_type in FILE_CONFIGS:
    dtype_dict = {}
    for col, dtype_str in FILE_CONFIGS[data_type].get("dtype", {}).items():
        dtype_dict[col] = dtype_map.get(dtype_str, str)
    FILE_CONFIGS[data_type]["dtype"] = dtype_dict


def get_cols_to_keep(data_type):
    """Get columns to keep for a data type from FILE_CONFIGS."""
    return list(FILE_CONFIGS[data_type]["dtype"].keys())


def load_data(data_type, year=None, file_path=None):
    """
    Generic function to load data based on file_configs.json.

    Args:
        data_type: Type of data to load (e.g., "DMR", "ESMR", "CWNS")
        year: Optional year for time-series data
        file_path: Optional explicit file path

    Returns:
        Loaded and processed DataFrame
    """
    config = FILE_CONFIGS[data_type]

    # Determine file path
    if file_path is None:
        # Check if config has explicit file path
        if "file" in config:
            file_path = config["file"]
        else:
            file_path = str(get_data_file_path(data_type, year))

    # Handle year-specific skiprows  #TODO clean up
    skiprows = config.get("skiprows", 0)
    if (
        "year_skiprows" in config
        and year is not None
        and str(year) in config["year_skiprows"]
    ):
        skiprows = config["year_skiprows"][str(year)]

    # Read data with configured columns and dtypes
    data = pd.read_csv(
        file_path,
        usecols=(
            list(config["dtype"].keys()) + config.get("parse_dates", [])
            if config["dtype"]
            else None
        ),
        dtype=config["dtype"] if config["dtype"] else None,
        parse_dates=config.get("parse_dates", []),
        skiprows=skiprows,
        sep=config.get("separator", ","),
        low_memory=False,
    )

    # dropna: drop rows where column IS NA
    dropna_cols = [col for col in config.get("dropna", []) if col in data.columns]
    if dropna_cols:
        data = data.dropna(subset=dropna_cols)

    # drop_notna: drop rows where column IS NOT NA
    for col in config.get("drop_notna", []):
        if col in data.columns:
            data = data[data[col].isna()]

    for col, filter_config in config.get("filters", {}).items():
        if col not in data.columns:
            continue
        if "isin" in filter_config:
            data = data[data[col].isin(filter_config["isin"])]
        elif filter_config.get("isin_filter_list"):  # TODO: make this more clean
            data = data[data[col].isin(npdes_from_facilities_list)]

    # Apply transformations from config
    if config.get("strip_leading_zeros"):
        transform = config["strip_leading_zeros"]
        data[transform] = data[transform].str.lstrip("0")

    elif config.get("mark_toxicity"):
        transform = config["mark_toxicity"]
        if transform["pattern"] == "startswith":
            mask = data[transform["column"]].str.startswith(tuple(transform["values"]))
            data.loc[mask, transform["set_column"]] = transform["set_value"]

    # Apply post-processing (explode, drop_duplicates)
    for col in config.get("explode", []):
        if col in data.columns and data[col].apply(lambda x: isinstance(x, list)).any():
            data = data.explode(col)

    drop_dup_config = config.get("drop_duplicates", {})
    if drop_dup_config:
        data = data.drop_duplicates(subset=drop_dup_config)

    # Apply renames from config
    rename_map = config.get("rename", {})
    if rename_map:
        data = data.rename(columns=rename_map)

    return data


def read_data_year(year, type, drop_toxicity=False):
    """Reads the CA DMR or ESMR data for the given year"""
    data = load_data(type, year=year)

    if type == "ESMR":
        return data

    # Apply DMR-specific transformations

    # Filter data relevant to WWNA facilities list
    data = data[data["EXTERNAL_PERMIT_NMBR"].isin(npdes_from_facilities_list)]

    if drop_toxicity and "PARAMETER_DESC" in data.columns:
        data = data[~data["PARAMETER_DESC"].str.contains("Toxicity")]
    data["POLLUTANT_CODE"] = data["PARAMETER_CODE"].map(ref_parameter["POLLUTANT_CODE"])

    mped = "MONITORING_PERIOD_END_DATE"
    data[f"{mped}_NUMERIC"] = (
        data[mped].dt.year + data[mped].dt.month / 12 + data[mped].dt.day / 365
    )

    unique = data["EXTERNAL_PERMIT_NMBR"].nunique()
    print(f"{year} {type}: {len(data)} records, {unique} facilities")

    return data


def read_data_by_type(data_type, year_range, save=False, drop_toxicity=False):
    """Uses read_data_year to read all the CA DMR data the analysis range"""
    data_dict = {}

    for year in year_range:
        if data_type == "DMR":
            data = read_data_year(year, "DMR", drop_toxicity)
        elif data_type == "ESMR":
            data = read_data_year(year, "ESMR")
        data_dict[year] = data

    if save and data_dict:
        # Concatenate all years and save as CSV
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        filename = f"processed_data/step3/{data_type.lower()}_all_years.csv"
        all_data.to_csv(filename, index=False)
        print(f"Saved {len(all_data)} records from {len(data_dict)} years")

    return data_dict


def read_limits(year):
    """
    Reads the CA DMR limits data for the given year.
    Uses load_data which applies config-based filtering.
    """
    dir_path = get_data_dir_path("DMR", year)
    file_path = dir_path / f"CA_FY{year}_NPDES_LIMITS.csv"

    # Use load_data which handles the facilities list filter via config
    data = load_data("LIMITS", year=year, file_path=str(file_path))

    unique = data["EXTERNAL_PERMIT_NMBR"].nunique()
    print(f"{year} has {len(data)} limits, {unique} unique permits")

    return data


# CATEGORIZE PARAMETERS
with open("data/manual_updates/parameter_sorting_dict.json", "r") as f:
    parameter_sorting_dict = json.load(f)


def categorize_parameters(df, parameter_sorting_dict, desc_column):
    """
    Categorize parameters in a dataframe based on a sorting dictionary.

    Args:
    df (pd.DataFrame): The dataframe containing parameters to categorize.
    parameter_sorting_dict (dict): Dictionary containing categories and
    their associated keywords.
    desc_column (str): Name of column containing parameter descriptions.

    Returns:
    pd.DataFrame: The input dataframe with additional
    'PARENT_CATEGORY' and 'SUB_CATEGORY' columns.
    """
    df["PARENT_CATEGORY"] = "Uncategorized"
    df["SUB_CATEGORY"] = "Uncategorized"

    def apply_categories(d, parent=None):
        """Recursively apply categories from the sorting dictionary."""
        for key, value in d.items():
            if isinstance(value, dict) and "values" in value:
                # Leaf node: has "values" key
                mask = df[desc_column].str.contains(
                    "|".join(map(re.escape, value["values"])),
                    case=value.get("case", False),
                )
                if parent:
                    df.loc[mask, "PARENT_CATEGORY"] = parent
                    df.loc[mask, "SUB_CATEGORY"] = key
                else:
                    df.loc[mask, "PARENT_CATEGORY"] = key
                    df.loc[mask, "SUB_CATEGORY"] = key
            elif isinstance(value, dict):
                # Branch node: contains subcategories
                apply_categories(value, parent=key)

    apply_categories(parameter_sorting_dict)
    return df


def normalize_param_desc(desc):
    """
    Normalize the parameter description by removing commas, brackets,
    spaces, apostrophes, and dots,
    converting to lowercase, and removing "sum" and "total"
    """
    to_remove = [",", " ", "'", "."]
    words_to_remove = ["sum", "total", "tot."]
    for word in words_to_remove:
        to_remove.extend([f", {word}", f", {word.capitalize()}"])

    # Build replacements dict (items to remove to "") and apply replacements
    replacements = {old: "" for old in to_remove}
    replacements.update({"[": "(", "]": ")", "&": "and"})
    for old, new in replacements.items():
        desc = desc.replace(old, new)

    return desc.lower()


def aggregate_by_group(data, group_col, value_col, count_col_name, list_col_name):
    """Aggregate data by group with count and comma-separated list."""
    agg_data = (
        data.groupby(group_col)
        .agg(
            count=(value_col, "count"),
            values=(value_col, lambda x: ", ".join(x.unique())),
        )
        .reset_index()
    )
    agg_data.columns = [group_col, count_col_name, list_col_name]
    return agg_data


def get_data_filename(data_type, year=None):
    """Get expected filename for a data type and optional year."""
    if data_type == "DMR":
        return f"CA_FY{year}_NPDES_DMRS.csv"
    elif data_type == "ESMR":
        return f"esmr-analytical-export_year-{year}.csv"
    elif data_type == "IR":
        return f"{year}-303d.csv"
    elif data_type == "SSO":
        return "Questionnaire.csv"
    elif data_type == "TOXICS":
        return "criteria_for_toxics.csv"
    elif data_type == "CWNS":
        return "facilities_merged.csv"
    return None


def get_data_dir_path(data_type, year=None):
    """Get path to data directory or specific subdirectory."""
    base_dir = Path(f"data/{data_type.lower()}")
    if data_type == "DMR" and year:
        return base_dir / f"CA_FY{year}_NPDES_DMRS_LIMITS"
    return base_dir


def get_data_file_path(data_type, year=None):
    """Get full file path for a data type and optional year."""
    dir_path = get_data_dir_path(data_type, year)
    filename = get_data_filename(data_type, year)
    return dir_path / filename if filename else dir_path


def load_facilities_list(facilities_list_path=None):
    path = facilities_list_path or FACILITIES_LIST_PATH
    return pd.read_csv(path)


# Common plotting settings
FIGURE_DPI = 300
DEFAULT_CMAP = "viridis"


def setup_figure(figsize=(10, 6)):
    """Create and setup a new figure with common settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_and_close(path, step=None, dpi=FIGURE_DPI):
    """Save figure and close it."""
    full_path = f"processed_data/step{step}/{path}" if step else path
    plt.tight_layout()
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_facilities_map(num_params_per_facility, legend_label, label_threshold):
    """
    Plot facilities on a map of CA.

    Args:
        num_params_per_facility: Dictionary mapping
        facility IDs to parameter counts
        legend_label: Label for the legend
        label_threshold: Threshold for labeling facilities
    """
    ca_counties = gpd.read_file("data/ca_counties/CA_Counties.shp")

    # Filter out invalid geometries
    # ca_counties = ca_counties[ca_counties.geometry.is_valid]
    # ca_counties = ca_counties[~ca_counties.geometry.is_empty]

    facilities_list = load_facilities_list()

    # Prepare facilities data
    facilities_with_coords = facilities_list[
        [
            "NPDES # CA#",
            "LATITUDE DECIMAL DEGREES",
            "LONGITUDE DECIMAL DEGREES",
        ]
    ].rename(
        columns={
            "LATITUDE DECIMAL DEGREES": "LATITUDE",
            "LONGITUDE DECIMAL DEGREES": "LONGITUDE",
        }
    )

    # Create DataFrame with facility IDs and merge
    facilities_with_coords_merged = pd.DataFrame(
        {"NPDES # CA#": list(num_params_per_facility.keys())}
    ).merge(facilities_with_coords, on="NPDES # CA#", how="left")

    # Filter out facilities without coordinates
    facilities_with_coords_merged = facilities_with_coords_merged[
        facilities_with_coords_merged["LATITUDE"].notna()
        & facilities_with_coords_merged["LONGITUDE"].notna()
    ]

    facilities_gdf = gpd.GeoDataFrame(
        facilities_with_coords_merged,
        geometry=gpd.points_from_xy(
            facilities_with_coords_merged["LONGITUDE"],
            facilities_with_coords_merged["LATITUDE"],
        ),
        crs="EPSG:4326",
    )

    # Convert both to a projected CRS for better plotting
    ca_counties = ca_counties.set_crs("EPSG:4326")
    target_crs = "EPSG:3310"  # NAD83 California Albers
    facilities_gdf_proj = facilities_gdf.to_crs(target_crs)
    ca_counties_proj = ca_counties.to_crs(target_crs)

    # Create plot
    fig, ax = setup_figure(figsize=(8, 5))
    ca_counties_proj.plot(ax=ax, color="lightgray")
    facilities_gdf_proj["num_parameters"] = facilities_gdf_proj["NPDES # CA#"].map(
        num_params_per_facility
    )

    # Setup colormap
    norm = plt.Normalize(
        vmin=facilities_gdf_proj["num_parameters"].min(),
        vmax=facilities_gdf_proj["num_parameters"].max(),
    )
    cmap = plt.cm.get_cmap(DEFAULT_CMAP)

    facilities_gdf_proj.plot(
        ax=ax, column="num_parameters", cmap=cmap, norm=norm, markersize=10
    )

    # Add facility labels
    top_facilities = (
        facilities_gdf_proj[facilities_gdf_proj["num_parameters"] >= label_threshold]
        .sort_values("num_parameters", ascending=False)
        .head(10)
    )

    # Sort by projected y coordinates for vertical ordering
    top_facilities_sorted = top_facilities.copy()
    top_facilities_sorted["geometry_y"] = top_facilities_sorted.geometry.y
    top_facilities_sorted = top_facilities_sorted.sort_values(
        "geometry_y", ascending=False
    )

    # Calculate label positions
    label_x = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    label_y_start = ax.get_ylim()[1] - 0.57 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    label_y_step = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for idx, (_, row) in enumerate(top_facilities_sorted.iterrows()):
        if not row.geometry.is_empty:
            label_y = label_y_start - idx * label_y_step

            ax.annotate(
                f"{row['NPDES # CA#']}",
                xy=(label_x, label_y),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="center",
            )

            ax.plot(
                [row.geometry.x, label_x + 2.5 * 1e5],
                [row.geometry.y, label_y],
                color="k",
                linewidth=0.5,
            )

    # Add legend
    unique_params = sorted(facilities_gdf_proj["num_parameters"].unique())
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(int(value)),
            markerfacecolor=cmap(norm(value)),
            markersize=10,
        )
        for value in unique_params
    ]
    ax.legend(
        handles=legend_elements,
        title=legend_label,
        loc="upper right",
        frameon=False,
    )

    ax.set_xlabel(""), ax.set_ylabel(""), ax.set_xticks([]), ax.set_yticks([])
    save_and_close("figures_py/facilities_map.png", 3)
