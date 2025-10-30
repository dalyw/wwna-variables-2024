import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing

# Load analysis configuration
with open("wwna_variables_2024/analysis_config.json", "r") as f:
    ANALYSIS_CONFIG = json.load(f)

# IMPORT DMR AND ESMR DATA
year_range_config = ANALYSIS_CONFIG["year_range"]
analysis_range = range(year_range_config[0], year_range_config[1] + 1)
save = False
load = True
DEFAULT_CMAP = "viridis"

# WWNA FACILITY LIST
WWNA_LIST_PATH = ANALYSIS_CONFIG["wwna_list_path"]
WWNA_LIST = pd.read_csv(WWNA_LIST_PATH)
NPDES_FROM_WWNA_LIST = (
    WWNA_LIST[WWNA_LIST["NPDES # CA#"].notna()]["NPDES # CA#"].unique().tolist()
)
if multiprocessing.current_process().name == "MainProcess":
    print(f"{len(NPDES_FROM_WWNA_LIST)} of {len(WWNA_LIST)} WWNA facilities have NPDES")

# Column name constants for aggregated results
AGG_STRINGS = {
    "3": {
        "COUNT": "Parameters with Slope and Near Exceedance: Number of Parameters",
        "PARAM": "Parameters with Slope and Near Exceedance: List of Parameters",
    },
    "4": {
        "COUNT": "Discharges to Impaired and Not Limited: Number of Parameters",
        "PARAM": "Discharges to Impaired and Not Limited: List of Parameters",
    },
}
agg_columns = []
for step in ["3", "4"]:
    for key in ["COUNT", "PARAM"]:
        agg_columns.append(AGG_STRINGS[step][key])


# Path constants for processed data directories
STEP_DIRS = {}
for i in range(4):
    STEP_DIRS[i + 1] = f"processed_data/step{i+1}"

SCRIPTS = [
    "step0_download_data.py",
    "step1_parameter_categorization.py",
    "step2_population_served.py",
    "step3_near_exceedance.py",
    "step4_future_limits.py",
]


with open("wwna_variables_2024/file_configs.json", "r") as f:
    FILE_CONFIGS = json.load(f)

# Data download configuration
ESMR_RESOURCE_IDS = {int(k): v for k, v in FILE_CONFIGS["ESMR"]["year_config"].items()}

# Convert dtype strings to Python types
dtype_map = {"str": str, "float": float, "int": int, "bool": bool}
for config in FILE_CONFIGS.values():
    if "dtype" in config:
        config["dtype"] = {
            col: dtype_map.get(dtype_str, str)
            for col, dtype_str in config["dtype"].items()
        }


def load_data(data_type, year=None, drop_toxicity=False):
    """
    Generic function to load data based on file_configs.json.

    Args:
        data_type: Type of data to load (e.g., "DMR", "ESMR", "CWNS")
        year: Optional year for time-series data
        file_path: Optional explicit file path
        drop_toxicity: If True, drop rows containing "Toxicity" in PARAMETER_DESC

    Returns:
        Loaded and processed DataFrame
    """
    config = FILE_CONFIGS[data_type]
    file_path = str(get_data_file_path(data_type, year))

    # Read data with configured columns and dtypes
    skiprows = config.get("skiprows", 0)
    if isinstance(skiprows, dict) and year is not None:
        skiprows = skiprows.get(str(year), 0)
    data = pd.read_csv(
        file_path,
        usecols=list(config["dtype"].keys()) + config.get("parse_dates", []),
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

    for col, filter_values in config.get("filters", {}).items():
        if col not in data.columns:
            continue
        if isinstance(filter_values, list):
            # Filter by list of values
            data = data[data[col].isin(filter_values)]

    # Apply transformations from config
    if config.get("strip_leading_zeros"):
        transform = config["strip_leading_zeros"]
        data[transform] = data[transform].str.lstrip("0")

    elif config.get("mark_toxicity"):
        transform = config["mark_toxicity"]
        if transform["pattern"] == "startswith":
            mask = data[transform["column"]].str.startswith(tuple(transform["values"]))
            data.loc[mask, transform["column"]] = transform["set_value"]

    # Apply post-processing (explode, drop_duplicates)
    for col in config.get("explode", []):
        if col in data.columns and data[col].apply(lambda x: isinstance(x, list)).any():
            data = data.explode(col)

    drop_dup_cols = config.get("drop_duplicates")
    if drop_dup_cols:
        data = data.drop_duplicates(subset=drop_dup_cols)

    # Apply renames from config
    rename_map = config.get("rename", {})
    if rename_map:
        data = data.rename(columns=rename_map)

    if data_type == "DMR":  # Apply DMR-specific transformations
        if drop_toxicity and "PARAMETER_DESC" in data.columns:
            data = data[~data["PARAMETER_DESC"].str.contains("Toxicity")]

        mped = "MONITORING_PERIOD_END_DATE"
        data[f"{mped}_NUMERIC"] = (
            data[mped].dt.year + data[mped].dt.month / 12 + data[mped].dt.day / 365
        )

        unique = data["EXTERNAL_PERMIT_NMBR"].nunique()
        print(f"{year} {data_type}: {len(data)} records, {unique} facilities")

    # Apply WWNA facilities list filter for DMR and LIMITS
    if data_type in ["DMR", "LIMITS"]:
        data = data[data["EXTERNAL_PERMIT_NMBR"].isin(NPDES_FROM_WWNA_LIST)]

    if data_type == "LIMITS":
        unique = data["EXTERNAL_PERMIT_NMBR"].nunique()
        print(f"{year} has {len(data)} limits, {unique} unique permits")

    return data


def aggregate_flagged_params(data, group_col, value_col, col_names):
    """
    Aggregate flagged parameters by facility with count and comma-separated list.

    Args:
        data: DataFrame to aggregate
        group_col: Column to group by (facility identifier)
        value_col: Column to aggregate (parameter identifier)
        col_names: Dict with "COUNT" and "PARAM" keys for column names

    Returns:
        Aggregated DataFrame
    """
    # Handle empty dataframe
    if len(data) == 0:
        return pd.DataFrame(columns=[group_col, col_names["COUNT"], col_names["PARAM"]])

    agg_data = (
        data.groupby(group_col)
        .agg(
            count=(value_col, "count"),
            values=(value_col, lambda x: ", ".join(x.unique())),
        )
        .reset_index()
    )
    agg_data.columns = [group_col, col_names["COUNT"], col_names["PARAM"]]

    # Fill NA values with defaults
    agg_data[col_names["COUNT"]] = agg_data[col_names["COUNT"]].fillna(0).astype(int)
    agg_data[col_names["PARAM"]] = agg_data[col_names["PARAM"]].fillna("")

    return agg_data


def get_data_file_path(data_type, year=None):
    """Get full file path for a data type and optional year."""
    config = FILE_CONFIGS[data_type]

    # Use base_dir_type if specified, otherwise use data_type
    base_dir = config.get("base_dir_type", data_type)
    dir_path = Path(f"data/{base_dir.lower()}")

    if "subdir_pattern" in config and year:
        dir_path = dir_path / config["subdir_pattern"].format(year=year)

    # Filename pattern
    if "file_pattern" in config:
        filename = (
            config["file_pattern"].format(year=year) if year else config["file_pattern"]
        )
        return dir_path / filename

    return dir_path


def setup_fig(figsize=(10, 6)):
    """Create and setup a new figure with common settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(path, step):
    """Save figure and close it."""
    full_path = f"processed_data/step{step}/figures_py/{path}"
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches="tight")
    plt.close()


def plot_barh(data, x_col, y_col, xlabel, figsize=(12, 6), path=None, step=None):
    """Create a horizontal bar chart with labels."""
    fig, ax = setup_fig(figsize=figsize)
    bars = ax.barh(data[y_col], data[x_col])
    ax.set_xlabel(xlabel)

    # Add labels to bars (inline instead of separate function)
    fmt_kwargs = {"ha": "left", "va": "center", "fontsize": 8}
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width, bar.get_y() + bar.get_height() / 2, f"{int(width)}", **fmt_kwargs
        )

    save_fig(path, step)


def plot_map(num_params_per_facility, label_threshold, step=3):
    """
    Plot facilities on a map of CA.

    Args:
        num_params_per_facility: Dictionary mapping facility IDs to parameter counts
        legend_label: Label for the legend
        label_threshold: Threshold for labeling facilities
        step: Step number for saving the plot
    """
    source_crs = "EPSG:3857"
    target_crs = "EPSG:3310"
    ca_counties = gpd.read_file("data/ca_counties/CA_Counties.shp")

    # Create DataFrame with facility IDs and merge
    facilities_with_coords_merged = pd.DataFrame(
        {"NPDES # CA#": list(num_params_per_facility.keys())}
    ).merge(WWNA_LIST.copy(), on="NPDES # CA#", how="left")

    facilities_gdf = gpd.GeoDataFrame(
        facilities_with_coords_merged,
        geometry=gpd.points_from_xy(
            facilities_with_coords_merged["LONGITUDE DECIMAL DEGREES"],
            facilities_with_coords_merged["LATITUDE DECIMAL DEGREES"],
        ),
        crs="EPSG:4326",
    )

    # Counties are in source CRS, transform to target CRS
    ca_counties_source = ca_counties.set_crs(source_crs, allow_override=True)
    ca_counties_proj = ca_counties_source.to_crs(target_crs)

    # Convert facilities to projected CRS for plotting
    facilities_gdf_proj = facilities_gdf.to_crs(target_crs)

    # Create plot
    fig, ax = setup_fig(figsize=(8, 5))
    ca_counties_proj.plot(
        ax=ax,
        color="lightgray",
        zorder=1,
        edgecolor="white",
        linewidth=0.5,
    )
    facilities_gdf_proj["param_count"] = facilities_gdf_proj["NPDES # CA#"].map(
        num_params_per_facility
    )

    # Set axis limits to show CA extent
    ax.set_xlim(ca_counties_proj.bounds.minx.min(), ca_counties_proj.bounds.maxx.max())
    ax.set_ylim(ca_counties_proj.bounds.miny.min(), ca_counties_proj.bounds.maxy.max())

    # Setup colormap
    norm = plt.Normalize(
        vmin=facilities_gdf_proj["param_count"].min(),
        vmax=facilities_gdf_proj["param_count"].max(),
    )
    cmap = plt.cm.get_cmap(DEFAULT_CMAP)

    facilities_gdf_proj.plot(
        ax=ax,
        column="param_count",
        cmap=cmap,
        norm=norm,
        zorder=2,
    )

    # Add facility labels
    top_facilities = (
        facilities_gdf_proj[facilities_gdf_proj["param_count"] >= label_threshold]
        .sort_values("param_count", ascending=False)
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
    unique_params = sorted(facilities_gdf_proj["param_count"].unique())
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(int(value)),
            markerfacecolor=cmap(norm(value)),
        )
        for value in unique_params
    ]
    ax.legend(
        handles=legend_elements,
        title="# Parameters Flagged",
        loc="upper right",
        frameon=False,
    )

    ax.set_xlabel(""), ax.set_ylabel(""), ax.set_xticks([]), ax.set_yticks([])
    save_fig("facilities_map.png", step)
