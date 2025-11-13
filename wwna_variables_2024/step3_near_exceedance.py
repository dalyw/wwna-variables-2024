import argparse
import gc
import json
import random
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pint import UnitRegistry
from helper_functions import (
    analysis_range,
    aggregate_flags,
    save_fig,
    plot_barh,
    plot_map,
    load_data,
    STEP_DIRS,
    STEP3_CONFIG,
    WWNA_LIST,
)

# Load unit aliases from CSV (for pint parsing)
ALIASES_DF = pd.read_csv(
    Path("data/manual_updates/unit_aliases.csv"), keep_default_na=False
)
UNIT_ALIASES = dict(zip(ALIASES_DF["unit_from"], ALIASES_DF["unit_to_py"]))
_ALIAS_LOOKUP = {k.lower(): v for k, v in UNIT_ALIASES.items()}
_ALIAS_LOOKUP.update({v.lower(): v for v in UNIT_ALIASES.values()})

param_mapping = pd.read_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv")

# Load statistical base code mapping for ESMR
with open("data/manual_updates/statistical_base_code_mapping.json", "r") as f:
    stat_base_mapping = json.load(f)

stat_patterns = sorted(
    [
        (pattern.lower(), code)
        for code, patterns in stat_base_mapping.items()
        for pattern in patterns
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Base column names (single string constants)
permit_col = "EXTERNAL_PERMIT_NMBR"
param_code_col = "PARAMETER_CODE"
unit_desc_col = "STANDARD_UNIT_DESC"
location_col = "MONITORING_LOCATION_CODE"
perm_feature_nmbr_col = "PERM_FEATURE_NMBR"
monitor_date_col = "MONITORING_PERIOD_END_DATE_NUMERIC"
limit_begin_col = "LIMIT_BEGIN_DATE_NUMERIC"
limit_end_col = "LIMIT_END_DATE_NUMERIC"
qualifier_col = "LIMIT_VALUE_QUALIFIER_CODE"
limit_val_col = "LIMIT_VALUE_STANDARD_UNITS"
dmr_val_col = "DMR_VALUE_STANDARD_UNITS"
stat_base_col = "STATISTICAL_BASE_CODE"
limit_type_code_col = "LIMIT_VALUE_TYPE_CODE"
limit_value_id_col = "LIMIT_VALUE_ID"
limit_set_schedule_col = "LIMIT_SET_SCHEDULE_ID"
unit_base_col = "STANDARD_UNIT_BASE"

# UNIQUE_LIMIT_COLS uniquely identify a single row in LIMITS, for merging into DMR
UNIQUE_LIMIT_COLS = [limit_set_schedule_col, limit_value_id_col, limit_type_code_col]
# LIMIT_GROUP_COLS are the identify similarly-monitored data across different permits
# (e.g. 7-day average hourly NH4 concentration in mg/L measured at effluent)
LIMIT_GROUP_COLS = [
    permit_col,
    param_code_col,
    location_col,
    perm_feature_nmbr_col,
    stat_base_col,
    unit_desc_col,
    limit_type_code_col,
]

# Thresholds from config
LIMIT_THRESHOLD = STEP3_CONFIG["limit_threshold"]
RECENT_VIOLATION_YEARS = STEP3_CONFIG["recent_violation_years"]
IQR_MULTIPLIER = STEP3_CONFIG["iqr_multiplier"]
TIME_TO_LIMIT_YEARS = STEP3_CONFIG["time_to_limit_years"]

# Initialize pint UnitRegistry for unit conversions
_ureg = UnitRegistry()
_ureg.formatter.default_format = "~P"  # Use abbreviated unit names
_ureg.define("percent = 1e-2 * dimensionless")
_ureg.define("permille = 1e-3 * dimensionless")


# Pre-compute reference dimensionalities and create lookup dict
_CONC_DIM = _ureg.Quantity(1, "kg/m**3").to_base_units().dimensionality
_FLOW_DIM = _ureg.Quantity(1, "kg/s").to_base_units().dimensionality
_TEMP_DIM = _ureg.Quantity(1, "K").to_base_units().dimensionality

_DIM_TO_LIMIT_TYPE = {
    _ureg.dimensionless: "C",  # Dimensionless concentration (pH, etc.)
    _CONC_DIM: "C",  # Concentration
    _FLOW_DIM: "Q",  # Flow/quantity
    _TEMP_DIM: "Q",  # Temperature (mapped to Q type)
}


def convert_to_base_units(df, convert_cols):

    def normalize_unit_str(unit):
        if unit is None or (isinstance(unit, float) and np.isnan(unit)):
            return "dimensionless"
        unit_str = str(unit).strip()
        unit_str = unit_str.replace("μ", "u").replace("µ", "u")
        unit_str = re.sub(r"\s*/\s*", "/", unit_str)
        unit_str = re.sub(r"\s+", " ", unit_str).strip()
        return _ALIAS_LOOKUP.get(unit_str.lower(), unit_str)

    # Normalize original units (e.g., "mg/L", "lb/day") but keep them as-is
    normalized_units = df[unit_desc_col].apply(normalize_unit_str)
    df[unit_desc_col] = normalized_units

    factor_map = {"dimensionless": 1.0}
    base_unit_map = {"dimensionless": "dimensionless"}
    for unit in normalized_units.unique():
        if unit in factor_map:  # e.g. dimensionless already assigned
            continue
        qty = _ureg.Quantity(1, unit).to_base_units()
        base_unit_str = str(qty.units)
        # map dimensionless to "dimensionless" instead of empty string
        if base_unit_str.strip() == "":
            base_unit_str = "dimensionless"
        factor_map[unit] = qty.magnitude
        base_unit_map[unit] = base_unit_str

    df[unit_base_col] = normalized_units.map(base_unit_map)

    # Convert values to base units
    for col in convert_cols:
        base_col = col.replace("STANDARD", "BASE")
        df[base_col] = df[col] * normalized_units.map(factor_map).astype(float)
        df[col] = df[base_col]
    return df


def main(drop_toxicity=False, exclude_noncompliant=False):
    # LOAD PARAMETER MAPPING FOR ESMR AND DMR PARAMETER DESCRIPTIONS
    param_desc_df = param_mapping[[param_code_col, "DMR_PARAMETER_DESC"]]
    param_desc_lookup = dict(
        zip(param_desc_df[param_code_col], param_desc_df["DMR_PARAMETER_DESC"])
    )

    # LOAD DMR HISTORY
    # Load DMR and LIMITS data for all years, then concatenate and merge
    dmr_parts = []
    limits_parts = []
    for y in analysis_range:
        # Load DMR data
        dmr_year = load_data("DMR", year=y)
        # Drop MONITORING_LOCATION_CODE from DMR - it will come from LIMITS after merge
        dmr_year = dmr_year.drop(columns=["MONITORING_LOCATION_CODE"], errors="ignore")
        dmr_parts.append(dmr_year)

        # Load LIMITS data
        limits_year = load_data("LIMITS", year=y, drop_toxicity=drop_toxicity)
        limits_year = limits_year.drop(columns=["PARAMETER_DESC"], errors="ignore")

        # Normalize MONITORING_LOCATION_CODE for concatenation/deduplication
        location_str = limits_year[location_col].astype(str).str.strip()
        effluent_codes = {"1", "2", "EG", "Y", "K"}
        limits_year.loc[location_str.isin(effluent_codes), location_col] = "1"

        limits_parts.append(limits_year)

    # Combine all DMR years into single DataFrame
    dmr_all = pd.concat(dmr_parts, ignore_index=True)
    print(f"  Total DMR records: {len(dmr_all):,}")

    # Truncate LIMIT_VALUE_TYPE_CODE to first character (C, Q, etc.) BEFORE merging
    # both DMR and LIMITS must be truncated before merge
    dmr_all[limit_type_code_col] = dmr_all[limit_type_code_col].astype(str).str[0]

    # Combine all LIMITS years and deduplicate on UNIQUE_LIMIT_COLS
    limits_all = pd.concat(limits_parts, ignore_index=True)
    # Truncate LIMIT_VALUE_TYPE_CODE to first character BEFORE deduplication
    limits_all[limit_type_code_col] = limits_all[limit_type_code_col].astype(str).str[0]
    limits_all = limits_all.drop_duplicates(subset=UNIQUE_LIMIT_COLS, keep="first")

    # Merge all DMR data with deduplicated LIMITS
    # DMR rows will match LIMITS rows if they share the same UNIQUE_LIMIT_COLS
    dmr_all = dmr_all.merge(limits_all, on=UNIQUE_LIMIT_COLS, how="inner")
    print(f"  Total merged records: {len(dmr_all):,}")

    # Convert DMR values and limit values to base units for comparison
    dmr_all = convert_to_base_units(dmr_all, [dmr_val_col, limit_val_col])

    # Combine similar statistical base codes to enable use of historical data
    stat_base_str = dmr_all[stat_base_col].astype(str).str.strip()
    dmr_all.loc[stat_base_str == "IA", stat_base_col] = "MB"
    dmr_all.loc[stat_base_str == "IB", stat_base_col] = "ME"

    # Normalize PERM_FEATURE_NMBR to combine historical and recent naming conventions
    # '001', '002', etc. -> 'EFF1', 'EFF2', etc.
    feature_nmbr_str = dmr_all[perm_feature_nmbr_col].astype(str).str.strip()
    numeric_pattern = feature_nmbr_str.str.match(r"^0*(\d+)$")
    numeric_features = feature_nmbr_str[numeric_pattern]
    if len(numeric_features) > 0:
        # Extract the numeric part and convert to EFF format
        numeric_values = numeric_features.str.extract(r"^0*(\d+)$")[0]
        dmr_all.loc[numeric_pattern, perm_feature_nmbr_col] = "EFF" + numeric_values
    # Normalize 'INF' to 'INF1' for consistency
    dmr_all.loc[feature_nmbr_str == "INF", perm_feature_nmbr_col] = "INF1"

    # Filter to groups with at least one non-NA limit_value entry after 2024
    # and at least 3 years of data span
    recent_year_threshold = float(max(analysis_range)) - 1  # 2025 - 1 = 2024
    groups_with_recent_limit = set(
        dmr_all.loc[
            (dmr_all[monitor_date_col] >= recent_year_threshold)
            & dmr_all[limit_val_col].notna()
            & dmr_all[qualifier_col].notna(),
            LIMIT_GROUP_COLS,
        ]
        .drop_duplicates()
        .apply(lambda row: tuple(row[col] for col in LIMIT_GROUP_COLS), axis=1)
    )

    def should_include_group(group):
        group_key = tuple(group[LIMIT_GROUP_COLS].iloc[0].values)
        return (
            group_key in groups_with_recent_limit  # noqa: F821
            and group[monitor_date_col].max() - group[monitor_date_col].min() >= 3.0
        )

    dmr_filtered = dmr_all.groupby(LIMIT_GROUP_COLS).filter(should_include_group)

    # Get most recent valid (non-NA) limit per LIMIT_GROUP_COLS for lookup
    # This is only for the lookup dictionary - dmr_filtered still has all rows
    most_recent_limit_filtered = (
        dmr_filtered.dropna(subset=[limit_val_col, qualifier_col])
        .sort_values(by=monitor_date_col, ascending=False, na_position="last")
        .groupby(LIMIT_GROUP_COLS, as_index=False)
        .first()  # First row = most recent valid limit
    )

    # OPTIONALLY DROP RECENT NONCOMPLIANT GROUPS
    # TO avoid double-counting with violations data for risk assessment
    if exclude_noncompliant:
        violation_years = sorted(list(analysis_range))[-RECENT_VIOLATION_YEARS:]
        # Filter to recent years when we care about violations
        recent_violation_data = dmr_all[
            dmr_all[monitor_date_col].apply(lambda x: int(x) in violation_years)
        ]
        # Identify exceedances and get noncompliant LIMIT_GROUP_COLS combinations
        violations = recent_violation_data[
            (recent_violation_data["REPORTED_EXCURSION_NMBR"] > 0)  # Exceedance
            | (recent_violation_data["VIOLATION_CODE"] == "E90")  # Effluent Violation
            | (recent_violation_data["EXCEEDENCE_PCT"] > 0)  # Effluent Violation
        ]
        noncompliant = violations[LIMIT_GROUP_COLS].drop_duplicates()
        print(f" {len(noncompliant)} noncompliant facility+parameter combos")
        # Remove already-noncompliant groups from dmr_filtered
        noncompliant_tuples = set(
            noncompliant.apply(
                lambda row: tuple(row[col] for col in LIMIT_GROUP_COLS), axis=1
            )
        )
        dmr_filtered = dmr_filtered[
            ~dmr_filtered.apply(
                lambda row: tuple(row[col] for col in LIMIT_GROUP_COLS), axis=1
            ).isin(noncompliant_tuples)
        ].copy()

    # Sort by monitoring date. Identify unique LIMIT_GROUP_COLS for ESMR matching
    dmr_filtered = dmr_filtered.sort_values(monitor_date_col)
    dmr_group_filter = dmr_filtered[LIMIT_GROUP_COLS].drop_duplicates()
    print(f"  DMR LIMIT_GROUP_COLS combos: {len(dmr_group_filter):,}")

    # Free memory - delete large DataFrames that are no longer needed
    del (dmr_parts, limits_parts, limits_all, dmr_all, groups_with_recent_limit)
    if exclude_noncompliant:
        del noncompliant, noncompliant_tuples, recent_violation_data, violations
    gc.collect()

    # REFERENCE DATA FOR FACILITY + PARAMETER DESCRIPTIONS
    wwna_facilities = (
        WWNA_LIST[["FACILITY ID", "NPDES # CA#"]]
        .rename(columns={"FACILITY ID": "facility_place_id", "NPDES # CA#": permit_col})
        .astype({"facility_place_id": str})
        .dropna(subset=[permit_col])
    )
    # Create set of matching facility_place_id values for early ESMR filtering
    wwna_facility_ids = set(wwna_facilities["facility_place_id"])

    # LOAD ESMR DATA WITH NORMALIZATION + METADATA
    esmr_dataframes = []

    for y in analysis_range:
        # Load ESMR data for this year and filter to only WWNA facilities
        esmr_year = load_data("ESMR", year=y)
        print(f"  Loaded {len(esmr_year):,} records")
        esmr_year = esmr_year[esmr_year["facility_place_id"].isin(wwna_facility_ids)]

        # Convert ESMR values to base units for comparison with DMR limits
        esmr_year = convert_to_base_units(esmr_year, [dmr_val_col])
        print(f"  After unit conversion: {len(esmr_year):,} records")

        # Map ESMR calculated_method to DMR STATISTICAL_BASE_CODE
        calc_normalized = (
            esmr_year["calculated_method"]
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        esmr_year[stat_base_col] = calc_normalized.apply(
            lambda calc: next(
                (code for pattern, code in stat_patterns if pattern in calc), None
            )
        )
        esmr_year = esmr_year.dropna(subset=[stat_base_col])
        print(f"  After stat base mapping: {len(esmr_year):,} records")

        # Map unit dimensionality to LIMIT_VALUE_TYPE_CODE prefix (C, Q, etc.)
        # Use STANDARD_UNIT_DESC (original units) to determine C vs Q type
        unit_dims = {
            unit: _ureg.Quantity(1, unit).to_base_units().dimensionality
            for unit in esmr_year[unit_desc_col].unique()
        }
        esmr_year[limit_type_code_col] = (
            esmr_year[unit_desc_col].map(unit_dims).map(_DIM_TO_LIMIT_TYPE)
        )
        esmr_year = esmr_year.dropna(subset=[limit_type_code_col])

        # Add permit number (EXTERNAL_PERMIT_NMBR) needed for merging with DMR data
        esmr_year = esmr_year.merge(
            wwna_facilities[[permit_col, "facility_place_id"]],
            on="facility_place_id",
            how="inner",
        )
        print(f"  After permit join: {len(esmr_year):,} records")

        # Map ESMR parameter names to DMR PARAMETER_CODE
        esmr_year = esmr_year.merge(
            param_mapping[[param_code_col, "ESMR_PARAMETER_DESC"]],
            left_on="PARAMETER_DESC",
            right_on="ESMR_PARAMETER_DESC",
            how="inner",
        )
        esmr_year = esmr_year.drop(columns=["ESMR_PARAMETER_DESC"])
        print(f"  After parameter mapping: {len(esmr_year):,} records")

        # Parse location codes:
        # "EFF-001" -> MONITORING_LOCATION_CODE="1", PERM_FEATURE_NMBR="001"
        location_upper = esmr_year["location"].str.upper()

        def parse_location(loc):
            """Parse ESMR location (e.g. 'EFF-001') into loc code and feature number."""
            loc_code = "0" if loc.startswith("INF") else "1"
            if " " in loc:
                loc = loc.split(" ")[0]
            if "-" in loc:
                return loc_code, loc.split("-", 1)[1].replace("-", "")
            return loc_code, ""

        parsed = location_upper.apply(parse_location)
        esmr_year[location_col] = [p[0] for p in parsed]
        esmr_year[perm_feature_nmbr_col] = [p[1] for p in parsed]

        # Drop rows where we couldn't parse the location
        esmr_year = esmr_year.dropna(subset=[location_col, perm_feature_nmbr_col])
        print(f"  After location parsing: {len(esmr_year):,} records")

        # Filter to only ESMR records that match existing DMR LIMIT_GROUP_COLS combos
        esmr_year = esmr_year.merge(dmr_group_filter, on=LIMIT_GROUP_COLS, how="inner")
        print(f"  After DMR group join: {len(esmr_year):,} records")

        esmr_dataframes.append(esmr_year)
        print(f" {len(esmr_year):,} records after filtering for {y}")
        del esmr_year, location_upper, parsed  # Free memory after appending
        gc.collect()

    esmr_data = pd.concat(esmr_dataframes, ignore_index=True)  # Concatenate
    del esmr_dataframes  # Free memory after concatenation
    gc.collect()

    # COMBINE DMR + ESMR INTO A SINGLE TABLE W/ SOURCE
    dmr_filtered["_DATA_SOURCE"] = "DMR"
    esmr_data["_DATA_SOURCE"] = "ESMR"
    data = pd.concat([dmr_filtered, esmr_data], ignore_index=True)
    data = data.sort_values(by=[monitor_date_col], kind="stable").reset_index(drop=True)
    print(f"Combined {len(dmr_filtered):,} DMR + {len(esmr_data):,} ESMR")

    # Create lookup dictionary indexed by LIMIT_GROUP_COLS tuple for fast access
    recent_limit_lookup = {
        tuple(row[col] for col in LIMIT_GROUP_COLS): row
        for _, row in most_recent_limit_filtered.iterrows()
    }
    del most_recent_limit_filtered, esmr_data  # Free memory
    gc.collect()

    # Process groups sequentially
    grouped_data = data.groupby(LIMIT_GROUP_COLS)
    flagged_records = []
    non_flagged_count = 0  # Track how many non-flagged examples we've plotted

    for key_tuple, group in grouped_data:
        dates = group[monitor_date_col]
        values = group[dmr_val_col]

        # Quartiles and two-sided outlier filtering with intraquartile range
        Q1_pct = np.percentile(values, 25)
        Q3_pct = np.percentile(values, 75)
        lower_thr = Q1_pct - IQR_MULTIPLIER * (Q3_pct - Q1_pct)
        upper_thr = Q3_pct + IQR_MULTIPLIER * (Q3_pct - Q1_pct)
        outlier_mask = (values >= lower_thr) & (values <= upper_thr)
        date_array = dates[outlier_mask].to_numpy(dtype=float)
        value_array = values[outlier_mask].to_numpy(dtype=float)

        if np.unique(date_array).size < 8:  # fewer than 8 unique dates
            continue  # Don't analyze

        trend_slope, trend_intercept = np.polyfit(date_array, value_array, 1)

        # Get current limit values from recent_limit_lookup for flagging
        current_limit_row = recent_limit_lookup.get(key_tuple)
        if current_limit_row is None:
            continue  # No current limit found, skip

        current_limit_val = current_limit_row[limit_val_col]
        is_maximum_limit = current_limit_row[qualifier_col] in {"<=", "<"}

        # Flagging logic
        # Use time_to_limit based on median + slope (no Q3/Q1 check needed)
        # This captures cases where values are trending toward limits
        # even if current Q3 isn't near threshold
        slope_toward = (trend_slope > 0) if is_maximum_limit else (trend_slope < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = abs(np.median(value_array) - current_limit_val)
            # Set time_to_limit to inf if already at limit or slope is zero/no trend
            if distance == 0 or not (slope_toward and abs(trend_slope) > 0):
                time_to_limit = np.inf
            else:
                time_to_limit = distance / abs(trend_slope)

        # Check if flagged:  time_to_limit check (removed Q3/Q1 near-exceedance check)
        is_flagged = time_to_limit <= TIME_TO_LIMIT_YEARS

        # Determine if we should plot this group
        should_plot = False
        is_non_flagged_example = False

        if is_flagged:
            should_plot = True
        elif (
            non_flagged_count < 5 and random.random() < 0.1
        ):  # ~10% chance per non-flagged group
            should_plot = True
            is_non_flagged_example = True
            non_flagged_count += 1

        if not should_plot:
            continue  # Skip plotting

        # Generate plot (for flagged or selected non-flagged)
        # Get the corresponding monitoring data from data DataFrame
        param_data = data[
            data[LIMIT_GROUP_COLS].apply(lambda row: tuple(row) == key_tuple, axis=1)
        ].copy()

        # Extract values for plotting
        first_data_row = param_data.iloc[0]
        param_code = key_tuple[LIMIT_GROUP_COLS.index(param_code_col)]
        param_desc = param_desc_lookup.get(param_code, f"Parameter {param_code}")
        dates = param_data[monitor_date_col]
        values = param_data[dmr_val_col]

        plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        # Time series plot
        ax1 = plt.subplot(gs[0])
        outliers = (values < lower_thr) | (values > upper_thr)
        ax1.scatter(
            dates[~outliers], values[~outliers], s=30, color="blue", label="Data"
        )
        ax1.scatter(
            dates[outliers], values[outliers], marker="*", color="r", label="Outliers"
        )
        ax1.plot(
            dates, trend_slope * dates + trend_intercept, "k--", label="Trend", zorder=6
        )

        # Calculate y-axis limits
        param_data_sorted = param_data.reset_index(drop=True)
        limits_sorted = param_data_sorted[limit_val_col].dropna()
        val_min, val_max = values[~outliers].min(), values[~outliers].max()
        limit_min, limit_max = limits_sorted.min(), limits_sorted.max()
        low_buf = 0.95 if is_maximum_limit else 0.90
        high_buf = 1.05 if is_maximum_limit else 1.10
        y_min = np.nanmin([val_min, limit_min, val_min * 0.95, limit_min * low_buf])
        y_max = np.nanmax([val_max, limit_max, val_max * 1.05, limit_max * high_buf])

        # Add outlier text annotation
        if outliers.sum() > 0:
            outlier_text = "Outliers:\n"
            for d, v in zip(dates[outliers], values[outliers]):
                date_str = f"{int(d)}-{int(round((d - int(d)) * 12)) + 1:02d}"
                outlier_text += f"{date_str}: {v:.4e}\n"
            ax1.text(
                0.02,
                0.98,
                outlier_text,
                transform=ax1.transAxes,
                fontsize=7,
                verticalalignment="top",
            )

        # Create compliance zones
        dmr_data_for_limits = param_data_sorted[
            param_data_sorted["_DATA_SOURCE"] == "DMR"
        ].copy()

        # Create segments based on actual monitoring dates where limit values change
        # Extract monitoring date and limit value, sort by date, collapse consecutive same values
        limit_segments = (
            dmr_data_for_limits[[monitor_date_col, limit_val_col]]
            .dropna(subset=[limit_val_col])
            .sort_values(by=monitor_date_col, kind="stable")
            .reset_index(drop=True)
        )

        if len(limit_segments) > 0:
            # Identify where limit value changes between consecutive rows
            limit_segments["limit_changed"] = limit_segments[limit_val_col].ne(
                limit_segments[limit_val_col].shift()
            )
            limit_segments["segment"] = limit_segments["limit_changed"].cumsum()

            # Group by segment and get first row's date and limit value
            segment_df = (
                limit_segments.groupby("segment", sort=False)
                .agg(
                    start=(monitor_date_col, "first"),
                    limit=(limit_val_col, "first"),
                )
                .astype({col: float for col in ["start", "limit"]})
            )

            # Set end date: next segment's start date (or end of plot range for last segment)
            segment_df["end"] = segment_df["start"].shift(-1).fillna(2026.0)

            # Ensure end doesn't exceed plot range
            segment_df["end"] = segment_df["end"].clip(upper=2026.0)
        else:
            # No limit data available
            segment_df = pd.DataFrame(columns=["start", "end", "limit"])

        zone_colors = {"In Compliance": "#d4f8d4", "Out of Compliance": "#fad8d8"}
        for _, seg_row in segment_df.iterrows():
            if is_maximum_limit:
                compliant_bounds = (y_min, seg_row["limit"])
                noncompliant_bounds = (seg_row["limit"], y_max)
            else:
                compliant_bounds = (seg_row["limit"], y_max)
                noncompliant_bounds = (y_min, seg_row["limit"])
            ax1.fill_between(
                (seg_row["start"], seg_row["end"]),
                *compliant_bounds,
                color=zone_colors["In Compliance"],
                zorder=0,
            )
            ax1.fill_between(
                (seg_row["start"], seg_row["end"]),
                *noncompliant_bounds,
                color=zone_colors["Out of Compliance"],
                zorder=0,
            )
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(2015, 2026)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(current_limit_row[unit_base_col])
        ax1.set_title(f"{param_desc}\n{key_tuple[LIMIT_GROUP_COLS.index(permit_col)]}")
        handles, _ = ax1.get_legend_handles_labels()
        handles.extend(
            Patch(facecolor=color, edgecolor=color, label=label)
            for label, color in zone_colors.items()
        )
        ax1.legend(handles, [h.get_label() for h in handles])
        ax1.set_xticks(range(2015, 2026))
        ax1.set_xticklabels([str(y) for y in range(2015, 2026)])

        # Histogram plot
        ax2 = plt.subplot(gs[1])
        ax2.hist(values[~outliers], bins=20, color="blue", edgecolor="black")
        ax2.axvline(Q1_pct, color="red", label="Q1")
        ax2.axvline(Q3_pct, color="orange", label="Q3")
        ax2.axvline(current_limit_val, color="gray", label="Limit")
        ax2.set_xlabel(current_limit_row[unit_base_col])
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)

        # Save figure
        permit_code = key_tuple[LIMIT_GROUP_COLS.index(permit_col)]
        unit = first_data_row[unit_desc_col]
        stat_base = first_data_row[stat_base_col]
        location = first_data_row[location_col]
        filename = f"{permit_code}_{param_desc}_{unit}_{stat_base}_Loc{location}.png"
        for ch in [" ", ",", "[", "]", "%", "/", ":"]:
            filename = filename.replace(ch, "_")

        if is_non_flagged_example:
            # Save to subfolder for non-flagged examples
            subfolder_path = Path(f"{STEP_DIRS[3]}/figures_py/not_flagged_examples")
            subfolder_path.mkdir(parents=True, exist_ok=True)
            full_path = subfolder_path / filename
            plt.tight_layout()
            plt.savefig(full_path, bbox_inches="tight")
            plt.close()
        else:
            # Save normally for flagged groups
            save_fig(f"{filename}", 3)

            # Save CSV with all columns for this flagged group
            csv_filename = filename.replace(".png", ".csv")
            csv_path = Path(f"{STEP_DIRS[3]}/csvs_py") / csv_filename
            # Save the full group data with all columns
            group.to_csv(csv_path, index=False)

            # Store the key tuple for flagged groups
            flagged_records.append(
                {col: val for col, val in zip(LIMIT_GROUP_COLS, key_tuple)}
            )

    flagged_df = pd.DataFrame(flagged_records)
    print(f"{len(flagged_df)} pairs with both near-exceedance and time-to-limit")
    print(f"{flagged_df[permit_col].nunique()} unique facilities")
    print("Generated facility-parameter plots")

    # Map and bar plot of flagged parameter counts per facility
    flagged_counts = flagged_df.groupby(permit_col)[param_code_col].nunique().to_dict()
    df = pd.DataFrame(flagged_counts.items(), columns=["Facility", "Parameters"])
    plot_map(flagged_counts, 4)
    plot_barh(
        df.sort_values("Parameters", ascending=True),
        x_col="Parameters",
        y_col="Facility",
        xlabel="Number of Parameters with Slope and Near-Exceedance",
        path="facilities_summary.png",
        step=3,
    )

    # Save aggregated results
    aggregated_df = aggregate_flags(flagged_df, permit_col, param_code_col, 3)
    aggregated_df.to_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3_py.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-toxicity", action="store_true")
    parser.add_argument("--exclude-noncompliant", action="store_true")
    args = parser.parse_args()

    main(args.drop_toxicity, args.exclude_noncompliant)
