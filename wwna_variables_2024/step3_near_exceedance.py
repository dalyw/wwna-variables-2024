import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from helper_functions import (
    analysis_range,
    aggregate_by_group,
    read_data_by_type,
    STEP_DIRS,
    save_and_close,
    setup_figure,
    plot_facilities_map,
    DEFAULT_CMAP,
)

# Grouping columns (DMR dataframe column names)
GROUP_COLS = [
    "EXTERNAL_PERMIT_NMBR",
    "PARAMETER_CODE",
    "STANDARD_UNIT_DESC",
    "MONITORING_LOCATION_CODE",
]

# Output columns
OUTPUT_COLS = GROUP_COLS + ["LIMIT_SET_SCHEDULE_ID", "LIMIT_VALUE_TYPE_CODE"]


def plot_facilities_summary(num_params_per_facility):
    """Plot summary of facilities without geographic data."""
    df = pd.DataFrame(
        list(num_params_per_facility.items()),
        columns=["Facility", "Parameters"],
    ).sort_values("Parameters", ascending=True)

    fig, ax = setup_figure(figsize=(12, 6))
    bars = plt.barh(df["Facility"], df["Parameters"])
    plt.xlabel("Number of Parameters with Slope and Near-Exceedance")
    plt.ylabel("Facility ID")
    plt.title("Facilities with Exceedances")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=8,
        )

    save_and_close("figures_py/facilities_summary.png", 3)


def plot_future_limits_summary(df_sorted):
    """Plot summary of facilities with future limits."""
    fig, ax = setup_figure()
    x = np.arange(len(df_sorted.index))
    width = 0.35

    colors = [plt.colormaps[DEFAULT_CMAP](val) for val in [0.2, 0.8]]
    ax.bar(
        x - width / 2,
        df_sorted["Discharges to Listed"],
        width,
        label="Discharging to Listed\nWater Body",
        color=colors[0],
    )
    ax.bar(
        x + width / 2,
        df_sorted["Newly Listed and Not Yet Limited"],
        width,
        label="Discharging to Newly Listed\nWater Body and\nNot Yet Limited",
        color=colors[1],
    )

    plt.ylabel("Number of Facilities", fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.xticks(x, df_sorted.index, rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(df_sorted["Discharges to Listed"]):
        ax.text(i - width / 2, v, str(int(v)), ha="center", va="bottom")
    for i, v in enumerate(df_sorted["Newly Listed and Not Yet Limited"]):
        ax.text(i + width / 2, v, str(int(v)), ha="center", va="bottom")

    save_and_close("figures_py/flagged_facilities_step4.png", 4)


def get_flagged_facilities(facility_records, slope_threshold=0.05, limit_threshold=0.1):
    """Calculate facilities with significant slope and near exceedance."""
    flagged_slope = []
    flagged_near_exceedance = []

    for rec in facility_records:
        # Skip if no valid limit
        if np.isnan(rec["latest_limit"]):
            continue

        # Check for facilities with significant slope
        if rec["qualifier"] in ["<=", "<"] and rec["slope"] > slope_threshold:
            near_exceedance = rec["Q3"] > (1 - limit_threshold) * rec["latest_limit"]
            has_slope = rec["slope"] > slope_threshold
        elif rec["qualifier"] in [">=", ">"] and rec["slope"] < -slope_threshold:
            near_exceedance = rec["Q1"] < (1 + limit_threshold) * rec["latest_limit"]
            has_slope = rec["slope"] < -slope_threshold
        else:
            has_slope, near_exceedance = False, False

        facility_tuple = tuple(rec[col] for col in OUTPUT_COLS)
        if has_slope:
            flagged_slope.append(facility_tuple)
        if near_exceedance:
            flagged_near_exceedance.append(facility_tuple)

    flagged_all = list(set(flagged_slope) & set(flagged_near_exceedance))

    print(f"{len(flagged_slope)} w/ slope>slope_threshold")
    print(f"{len(flagged_near_exceedance)} pairs with Q1/Q3 > {limit_threshold}")
    print(f"{len(flagged_all)} pairs with both")
    print(f"{len(set(f[0] for f in flagged_all))} " f"facilities affected")

    return flagged_all


def create_facility_parameter_plot(
    npdes_code, param_desc, data, legend_elements, histogram_legend_elements
):
    """Create individual plot for a facility-parameter combination."""

    # Extract data
    dates = data["MONITORING_PERIOD_END_DATE_NUMERIC"]
    values = data["DMR_VALUE_STANDARD_UNITS"]
    limits = data["LIMIT_VALUE_NMBR"]

    # Calculate statistics
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    limit_value = limits.iloc[0] if len(limits) > 0 else np.nan

    # Create figure with subplots
    plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Time series plot
    ax1 = plt.subplot(gs[0])
    ax1.scatter(dates, values, alpha=0.7, s=30, color="blue", label="Data")

    # Add trend line
    if len(dates) > 1:
        z = np.polyfit(dates, values, 1)
        p = np.poly1d(z)
        ax1.plot(dates, p(dates), "k--", alpha=0.8, linewidth=2, label="Trend")

    # Add compliance zones
    if not np.isnan(limit_value):
        y_lim = ax1.get_ylim()[1]
        for y1, y2, color, label in [
            (0, limit_value, "lightgreen", "In Compliance"),
            (limit_value, y_lim, "lightcoral", "Out of Compliance"),
        ]:
            ax1.axhspan(y1, y2, alpha=0.3, color=color, label=label)
        ax1.axhline(
            y=limit_value,
            color="gray",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label="Limit",
        )

    # Mark outliers (values > Q3 + 1.5*IQR)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = values > outlier_threshold
    ax1.scatter(
        dates[outliers],
        values[outliers],
        marker="*",
        s=100,
        color="red",
        label="Outliers",
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Concentration (mg/L)")
    ax1.set_title(f"{param_desc}\nFacility: {npdes_code}")
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Histogram plot
    ax2 = plt.subplot(gs[1])
    ax2.hist(values, bins=20, alpha=0.7, color="blue", edgecolor="black")

    # Add statistical markers
    for val, color, style, label in [
        (q1, "red", "--", "Q1"),
        (q3, "orange", "--", "Q3"),
        (
            (limit_value, "gray", "-", "Limit")
            if not np.isnan(limit_value)
            else (None, None, None, None)
        ),
    ]:
        if val is not None:
            ax2.axvline(val, color=color, linestyle=style, linewidth=2, label=label)

    ax2.set_xlabel("Concentration (mg/L)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution")
    ax2.legend(handles=histogram_legend_elements, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Save the plot
    filename = (
        f"{npdes_code}_"
        f"{param_desc.replace(' ', '_').replace(',', '').replace('[', '').replace(']', '')}"  # noqa: E501
        f".png"
    )
    save_and_close(f"figures_py/{filename}", 3)


def step3_plotting(flagged_data):
    """Generate exceedance analysis visualizations."""

    noline = {"linestyle": "None"}
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="b", label="Data", **noline),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Trend"),
        plt.Line2D([0], [0], marker="*", color="r", label="Outliers", **noline),
        plt.Rectangle((0, 0), 1, 1, fc="lightgreen", alpha=0.3, label="In Compliance"),
        plt.Rectangle(
            (0, 0), 1, 1, fc="lightcoral", alpha=0.3, label="Out of Compliance"
        ),
    ]
    histogram_legend_elements = [
        plt.Line2D([0], [0], color=c, linestyle=l, label=lbl)
        for c, l, lbl in [
            ("red", "--", "Q1"),
            ("orange", "--", "Q3"),
            ("grey", "-", "Limit"),
        ]
    ]

    # Generate plots for facilities with slope and near exceedance
    facilities_grouped = flagged_data.groupby("EXTERNAL_PERMIT_NMBR")

    print(f"Generating plots for {len(facilities_grouped)} facilities")

    for npdes_code, facility_data in facilities_grouped:
        # Group by parameter description
        param_groups = facility_data.groupby("PARAMETER_DESC")

        for param_desc, param_data in param_groups:
            create_facility_parameter_plot(
                npdes_code,
                param_desc,
                param_data,
                legend_elements,
                histogram_legend_elements,
            )


def process_facility_group(args):
    """Process a single facility-parameter group in parallel."""
    key_tuple, group = args

    # Convert data types and handle missing values
    dates = pd.to_numeric(group["MONITORING_PERIOD_END_DATE_NUMERIC"], errors="coerce")
    values = pd.to_numeric(group["DMR_VALUE_STANDARD_UNITS"], errors="coerce")

    # Remove NaN values and ensure unique x values
    mask = ~(np.isnan(dates) | np.isnan(values))
    dates = dates[mask]
    values = values[mask]

    # Get unique x values and their corresponding y means
    unique_dates = np.unique(dates)
    if len(unique_dates) < 3:  # Need at least 3 unique points for trend
        return None

    unique_vals = [np.mean(values[dates == d]) for d in unique_dates]

    # Calculate quartiles
    Q1_percentile, Q3_percentile = np.percentile(values, [25, 75])

    # Center and scale the data, then get linear regression
    dates_norm = (unique_dates - unique_dates.mean()) / unique_dates.std()
    vals_mean, vals_std = np.mean(unique_vals), np.std(unique_vals)
    values_norm = (unique_vals - vals_mean) / vals_std if vals_std != 0 else unique_vals
    slope, _, r_value, p_value, _ = stats.linregress(dates_norm, values_norm)

    # Convert slope back to original scale and reset if poor fit
    if r_value**2 < 0.1 or p_value > 0.05:
        slope = 0

    # Get the most recent limit value
    limits = pd.to_numeric(group["LIMIT_VALUE_STANDARD_UNITS"], errors="coerce")
    latest_limit = limits.iloc[-1] if not limits.empty else np.nan

    return {
        "slope": slope * (vals_std / unique_dates.std()),
        "latest_limit": latest_limit,
        "qualifier": group["LIMIT_VALUE_QUALIFIER_CODE"].values[0],
        "Q1": Q1_percentile,
        "Q3": Q3_percentile,
        "LIMIT_VALUE_TYPE_CODE": group["LIMIT_VALUE_TYPE_CODE"].values[0],
        "LIMIT_SET_SCHEDULE_ID": group["LIMIT_SET_SCHEDULE_ID"].values[0],
        **{col: val for col, val in zip(GROUP_COLS, key_tuple)},
    }


def main():
    # Load unique parameter codes from step1 output
    unique_param_codes = pd.read_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping.csv")[
        "PARAMETER_CODE"
    ].unique()

    # Load and filter DMR data
    data_dict = read_data_by_type(
        "DMR", analysis_range, save=False, drop_toxicity=False
    )

    filtered_data = pd.concat(
        data_dict[y][data_dict[y]["PARAMETER_CODE"].isin(unique_param_codes)]
        for y in analysis_range
    )

    # Group by facility and parameter
    grouped_data = filtered_data.groupby(GROUP_COLS)

    # Parallel processing
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(process_facility_group, grouped_data)

    # Filter out None results and process
    facility_records = [r for r in results if r is not None]

    # Create DataFrame of results
    flagged_facilities = get_flagged_facilities(facility_records)
    flagged_facilities_df = pd.DataFrame(flagged_facilities, columns=OUTPUT_COLS)

    # Count parameters per facility
    param_counts = {}
    for facility, param, *_ in flagged_facilities:
        param_counts.setdefault(facility, set()).add(param)
    flagged_param_counts = {f: len(s) for f, s in param_counts.items()}

    # Merge flagged facilities with actual data for plotting
    flagged_data = filtered_data.merge(
        flagged_facilities_df, on=GROUP_COLS, how="inner"
    )

    # Create facilities visualization
    plot_facilities_map(
        flagged_param_counts,
        "# of Parameters\nwith Slope and\nNear-Exceedance",
        4,
    )
    plot_facilities_summary(flagged_param_counts)
    step3_plotting(flagged_data)

    # Save detailed results
    flagged_facilities_df.to_csv(f"{STEP_DIRS[3]}/flagged_facilities.csv", index=False)

    # Save aggregated results
    aggregated_df = aggregate_by_group(
        flagged_facilities_df,
        "EXTERNAL_PERMIT_NMBR",
        "PARAMETER_CODE",
        "Number of Parameters with Slope and Near Exceedance",
        "Parameters with Slope and Near Exceedance",
    )
    aggregated_df.to_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3.csv", index=False)


if __name__ == "__main__":
    main()
