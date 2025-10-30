import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from helper_functions import (
    analysis_range,
    aggregate_flagged_params,
    save_fig,
    plot_barh,
    plot_map,
    load_data,
    STEP_DIRS,
    AGG_STRINGS,
    ANALYSIS_CONFIG,
)

# Grouping columns (DMR dataframe column names) and output columns
GROUP_COLS = ANALYSIS_CONFIG["step3"]["grouping_columns"]
OUTPUT_COLS = GROUP_COLS + ["LIMIT_SET_SCHEDULE_ID", "LIMIT_VALUE_TYPE_CODE"]

# Thresholds from config
TIME_TO_LIMIT_YEARS = ANALYSIS_CONFIG["step3"]["time_to_limit_years"]
LIMIT_THRESHOLD = ANALYSIS_CONFIG["step3"]["limit_threshold"]


def _step3_facility_param_plot(npdes_code, param_desc, data):
    """Create individual plot for a facility-parameter combination."""

    dates = data["MONITORING_PERIOD_END_DATE_NUMERIC"]
    values = data["DMR_VALUE_STANDARD_UNITS"]
    limits = data["LIMIT_VALUE_STANDARD_UNITS"]
    is_minimum_limit = bool(data["is_minimum_limit"].iloc[0])
    unit_desc = data["STANDARD_UNIT_DESC"].iloc[0]

    # Calculate statistics and latest limit
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    # Pick latest non-null limit aligned to latest monitoring date
    order_idx = np.argsort(dates)
    limits_sorted = limits.iloc[order_idx]
    limits_non_null = limits_sorted.dropna()
    limit_value = limits_non_null.iloc[-1] if len(limits_non_null) > 0 else np.nan

    # Create figure with subplots
    plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Time series plot
    ax1 = plt.subplot(gs[0])
    ax1.scatter(dates, values, alpha=0.7, s=30, color="blue", label="Data")

    # Trendline
    m = float(data["trend_slope"].iloc[0])
    b = float(data["trend_intercept"].iloc[0])
    ax1.plot(dates, m * dates + b, "k--", linewidth=2, label="Trend", zorder=6)

    # Add compliance zones based on limit type
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    low_buf = 0.95 if is_minimum_limit else 0.90
    high_buf = 1.05 if is_minimum_limit else 1.10
    y_min = min(y_min, limit_value * low_buf, y_min * 0.95)
    y_max = max(y_max, max(y_max * 1.05, limit_value * high_buf))
    green_zone = (limit_value, y_max) if is_minimum_limit else (y_min, limit_value)
    red_zone = (y_min, limit_value) if is_minimum_limit else (limit_value, y_max)
    zones = [
        (*green_zone, "lightgreen", "In Compliance"),
        (*red_zone, "lightcoral", "Out of Compliance"),
    ]
    ax1.set_ylim(y_min, y_max)
    for y1, y2, color, label in zones:
        ax1.axhspan(y1, y2, color=color, alpha=0.3, label=label, zorder=1)
    ax1.axhline(y=limit_value, color="gray", linestyle="-", label="Limit")

    # Mark outliers (two-sided IQR rule)
    iqr = q3 - q1
    iqr_multiplier = ANALYSIS_CONFIG["step3"]["iqr_multiplier"]
    lower_thr = q1 - iqr_multiplier * iqr
    upper_thr = q3 + iqr_multiplier * iqr
    outliers = (values < lower_thr) | (values > upper_thr)
    ax1.scatter(
        dates[outliers], values[outliers], marker="*", color="r", label="Outliers"
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel(unit_desc)
    ax1.set_title(f"{param_desc}\nFacility: {npdes_code}")
    ax1.legend(loc="upper right", fontsize=8)

    # Histogram plot
    ax2 = plt.subplot(gs[1])
    ax2.hist(values, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax2.axvline(q1, color="red", label="Q1")
    ax2.axvline(q3, color="orange", label="Q3")
    ax2.axvline(limit_value, color="gray", label="Limit")
    ax2.set_xlabel(unit_desc)
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)

    # Save the plot
    filename = f"{npdes_code}_{param_desc}.png"
    for ch in [" ", ",", "[", "]", "%", "/", ":"]:
        filename = filename.replace(ch, "_")
    save_fig(f"{filename}", 3)


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

    # Early return if no data
    if len(values) == 0:
        return None

    # Quartiles and two-sided outlier filtering with intraquartile range
    Q1_percentile, Q3_percentile = np.percentile(values, [25, 75])
    iqr = Q3_percentile - Q1_percentile
    iqr_mult = ANALYSIS_CONFIG["step3"]["iqr_multiplier"]
    lower_thr = Q1_percentile - iqr_mult * iqr
    upper_thr = Q3_percentile + iqr_mult * iqr
    mask = (values >= lower_thr) & (values <= upper_thr)
    dates_filtered, values_filtered = dates[mask], values[mask]

    # Unique date means and linear trend
    unique_dates = np.unique(dates_filtered)
    if len(unique_dates) < 3:
        return None

    unique_vals = np.array(
        [values_filtered[dates_filtered == d].mean() for d in unique_dates]
    )
    trend_slope, trend_intercept = np.polyfit(unique_dates, unique_vals, 1)
    dates_norm = (unique_dates - unique_dates.mean()) / unique_dates.std()
    vals_mean, vals_std = unique_vals.mean(), unique_vals.std()
    values_norm = (unique_vals - vals_mean) / vals_std if vals_std != 0 else unique_vals
    slope, _, r_value, p_value, _ = stats.linregress(dates_norm, values_norm)

    # Pick the most recent non-null limit and its qualifier (simple, explicit)
    limits = pd.to_numeric(group["LIMIT_VALUE_STANDARD_UNITS"], errors="coerce")
    quals = group["LIMIT_VALUE_QUALIFIER_CODE"]
    types = group.get("LIMIT_VALUE_TYPE_CODE", pd.Series([None] * len(group)))
    dates = pd.to_numeric(group["MONITORING_PERIOD_END_DATE_NUMERIC"], errors="coerce")
    order_idx = np.argsort(dates.values)
    limits_sorted = limits.iloc[order_idx]
    valid_mask = ~limits_sorted.isna()
    if not valid_mask.any():
        raise ValueError(f"No valid limits for group {dict(zip(GROUP_COLS, key_tuple))}")
    last_pos = np.where(valid_mask.to_numpy())[0][-1]
    chosen_limit = float(limits_sorted.iloc[last_pos])
    chosen_qual = quals.iloc[order_idx].iloc[last_pos]
    chosen_type = types.iloc[order_idx].iloc[last_pos]

    return {
        "slope": slope * (vals_std / unique_dates.std()),
        "trend_slope": trend_slope,
        "trend_intercept": trend_intercept,
        "median": float(np.median(values_filtered)),
        "limit": chosen_limit,
        "qualifier": chosen_qual,
        "Q1": Q1_percentile,
        "Q3": Q3_percentile,
        "LIMIT_VALUE_TYPE_CODE": chosen_type,
        "LIMIT_SET_SCHEDULE_ID": group["LIMIT_SET_SCHEDULE_ID"].values[0],
        "PARAMETER_DESC": group["PARAMETER_DESC"].values[0],
        **{col: val for col, val in zip(GROUP_COLS, key_tuple)},
    }


def main(drop_toxicity=False):
    # Load unique parameter codes from step1 output
    unique_param_codes = pd.read_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv")[
        "PARAMETER_CODE"
    ].unique()

    # Load and filter DMR data
    data_dict = {}
    for year in analysis_range:
        data = load_data("DMR", year=year, drop_toxicity=drop_toxicity)
        data_dict[year] = data

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
    flagged_all = []
    time_to_limit_count = 0
    near_count = 0

    for rec in facility_records:
        # Skip if no valid limit
        if np.isnan(rec["limit"]):
            continue

        # Check for facilities with significant slope moving TOWARD non-compliance
        qualifier = rec["qualifier"]
        is_minimum_limit = qualifier in [">=", ">"]
        near_cutoff = (
            (1 - LIMIT_THRESHOLD) * rec["limit"]
            if qualifier in ["<=", "<"]
            else (
                (1 + LIMIT_THRESHOLD) * rec["limit"]
                if qualifier in [">=", ">"]
                else np.nan
            )
        )
        near_exceedance = (
            rec["Q3"] > near_cutoff
            if qualifier in ["<=", "<"]
            else rec["Q1"] < near_cutoff if qualifier in [">=", ">"] else False
        )


        # Tally individual conditions
        distance = (
            rec["limit"] - rec["median"]
            if qualifier in ["<=", "<"]
            else rec["median"] - rec["limit"]
        )
        if not np.isnan(distance) and rec["slope"] != 0:
            time_to_limit = abs(distance) / abs(rec["slope"])
        has_time_to_limit = time_to_limit <= TIME_TO_LIMIT_YEARS
        if has_time_to_limit:
            time_to_limit_count += 1
        if near_exceedance:
            near_count += 1

        # Time-to-limit (years) using median distance; only meaningful if moving toward limit
        time_to_limit = np.inf

        # Flag based on near-exceedance AND time-to-limit
        if near_exceedance and has_time_to_limit:
            rec["is_minimum_limit"] = is_minimum_limit
            rec["near_cutoff"] = near_cutoff
            rec["time_to_limit_years"] = time_to_limit
            flagged_all.append(rec)

    print(f"{near_count} pairs with Q1/Q3 > {LIMIT_THRESHOLD}")
    print(f"{len(flagged_all)} pairs with both (and within {TIME_TO_LIMIT_YEARS} yrs)")
    print(f"{len(set(rec['EXTERNAL_PERMIT_NMBR'] for rec in flagged_all))} facilities")

    flagged_all_df = pd.DataFrame(flagged_all)

    desc_map = (
        filtered_data[GROUP_COLS + ["PARAMETER_DESC"]]
        .drop_duplicates(subset=GROUP_COLS)
    )
    flagged_all_df = flagged_all_df.merge(desc_map, on=GROUP_COLS, how="left")

    # Merge flagged facilities with actual data for plotting
    flagged_data = filtered_data.merge(flagged_all_df, on=GROUP_COLS, how="inner")

    # Count parameters per facility
    flagged_param_counts = (
        flagged_all_df.groupby("EXTERNAL_PERMIT_NMBR")["PARAMETER_CODE"]
        .nunique()
        .to_dict()
    )

    # Map of counts
    plot_map(flagged_param_counts, 4)

    # Generate plots for facilities with slope and near exceedance
    facilities_grouped = flagged_data.groupby("EXTERNAL_PERMIT_NMBR")
    # Use GROUP_COLS minus facility id to avoid mixing within-facility plots,
    plot_group_cols = [c for c in GROUP_COLS if c != "EXTERNAL_PERMIT_NMBR"]
    plot_group_cols.append("PARAMETER_DESC")
    for npdes_code, facility_data in facilities_grouped:
        for _, param_data in facility_data.groupby(plot_group_cols):
            param_desc = (
                param_data["PARAMETER_DESC"].iloc[0]
                if "PARAMETER_DESC" in param_data.columns
                else param_data["PARAMETER_CODE"].iloc[0]
            )
            _step3_facility_param_plot(npdes_code, param_desc, param_data)

    # Bar plot of counts
    df = pd.DataFrame(
        list(flagged_param_counts.items()), columns=["Facility", "Parameters"]
    )
    plot_barh(
        df.sort_values("Parameters", ascending=True),
        x_col="Parameters",
        y_col="Facility",
        xlabel="Number of Parameters with Slope and Near-Exceedance",
        path="facilities_summary.png",
        step=3,
    )

    # Save aggregated results
    aggregated_df = aggregate_flagged_params(
        flagged_all_df,
        "EXTERNAL_PERMIT_NMBR",
        "PARAMETER_CODE",
        AGG_STRINGS["3"],
    )
    aggregated_df.to_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3_py.csv", index=False)


if __name__ == "__main__":
    main()
