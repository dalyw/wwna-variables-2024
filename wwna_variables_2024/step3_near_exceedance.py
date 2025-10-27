import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from helper_functions import (
    analysis_range,
    aggregate_flagged_params,
    STEP_DIRS,
    save_fig,
    plot_barh,
    plot_map,
    load_data,
    AGG_STRINGS,
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

        if has_slope:
            flagged_slope.append(rec)
        if near_exceedance:
            flagged_near_exceedance.append(rec)

    # Find records in both lists
    slope_tuples = {tuple(rec[col] for col in OUTPUT_COLS): rec for rec in flagged_slope}
    exceedance_tuples = {tuple(rec[col] for col in OUTPUT_COLS): rec for rec in flagged_near_exceedance}
    flagged_keys = set(slope_tuples.keys()) & set(exceedance_tuples.keys())
    flagged_all = [slope_tuples[k] for k in flagged_keys]

    print(f"{len(flagged_slope)} w/ slope>slope_threshold")
    print(f"{len(flagged_near_exceedance)} pairs with Q1/Q3 > {limit_threshold}")
    print(f"{len(flagged_all)} pairs with both")
    print(f"{len(set(rec['EXTERNAL_PERMIT_NMBR'] for rec in flagged_all))} facilities affected")

    return flagged_all


def _step3_facility_param_plot(
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
    save_fig(f"figures_py/{filename}", 3)


def step3_plotting(flagged_data, flagged_param_counts):
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
            _step3_facility_param_plot(
                npdes_code,
                param_desc,
                param_data,
                legend_elements,
                histogram_legend_elements,
            )

    df = pd.DataFrame(
        list(flagged_param_counts.items()),
        columns=["Facility", "Parameters"],
    ).sort_values("Parameters", ascending=True)

    plot_barh(
        df,
        x_col="Parameters",
        y_col="Facility",
        xlabel="Number of Parameters with Slope and Near-Exceedance",
        title="Facilities with Exceedances",
        path="figures_py/facilities_summary.png",
        step=3,
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
    if len(unique_dates) < 3:  # At least 3 unique points for trend
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


def main(save=False, drop_toxicity=False):
    # Load unique parameter codes from step1 output
    unique_param_codes = pd.read_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv")[
        "PARAMETER_CODE"
    ].unique()

    # Load and filter DMR data
    data_dict = {}

    for year in analysis_range:
        data = load_data("DMR", year=year, drop_toxicity=drop_toxicity)
        data_dict[year] = data

    if save:
        # Concatenate all years and save as CSV
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        filename = f"processed_data/step3/{"DMR".lower()}_all_years_py.csv"
        all_data.to_csv(filename, index=False)
        print(f"Saved {len(all_data)} records from {len(data_dict)} years")

    filtered_data = pd.concat(
        data_dict[y][data_dict[y]["PARAMETER_CODE"].isin(unique_param_codes)]
        for y in analysis_range
    )
    grouped_data = filtered_data.groupby(GROUP_COLS)

    # Parallel processing
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(process_facility_group, grouped_data)

    # Create DataFrame of results (filter out None results)
    facility_records = [r for r in results if r is not None]
    flagged_facilities = get_flagged_facilities(facility_records)
    flagged_facilities_df = pd.DataFrame(flagged_facilities)

    # Count parameters per facility
    param_counts = {}
    for rec in flagged_facilities:
        facility = rec["EXTERNAL_PERMIT_NMBR"]
        param = rec["PARAMETER_CODE"]
        param_counts.setdefault(facility, set()).add(param)
    flagged_param_counts = {f: len(s) for f, s in param_counts.items()}

    # Merge flagged facilities with actual data for plotting
    flagged_data = filtered_data.merge(
        flagged_facilities_df, on=GROUP_COLS, how="inner"
    )

    # Create facilities visualization
    plot_map(flagged_param_counts, 4)
    step3_plotting(flagged_data, flagged_param_counts)

    # Save detailed results (only selected columns)
    flagged_facilities_df[
        [
            "EXTERNAL_PERMIT_NMBR",
            "PARAMETER_CODE",
            "slope",
            "latest_limit",
            "qualifier",
            "Q1",
            "Q3",
        ]
    ].to_csv(f"{STEP_DIRS[3]}/flagged_facilities_py.csv", index=False)

    # Save aggregated results
    aggregated_df = aggregate_flagged_params(
        flagged_facilities_df,
        "EXTERNAL_PERMIT_NMBR",
        "PARAMETER_CODE",
        AGG_STRINGS["3"],
    )
    aggregated_df.to_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3_py.csv", index=False)


if __name__ == "__main__":
    main()
