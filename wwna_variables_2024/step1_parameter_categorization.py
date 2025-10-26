import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import (
    normalize_param_desc,
    read_data_year,
    categorize_parameters,
    parameter_sorting_dict,
    get_data_file_path,
    FILE_CONFIGS,
    save_and_close,
)


def match_param_desc(row, target_df, target_desc_column):
    """Match the parameter description in the target df to the current row."""
    normalized_desc = normalize_param_desc(str(row["PARAMETER_DESC"]))
    match = target_df[target_df["normalized_desc"] == normalized_desc]
    return match[target_desc_column].iloc[0] if len(match) > 0 else ""


def plot_pie_counts(df, title):
    """
    Plot pie chart of parameter categories.

    Args:
        df: DataFrame containing PARENT_CATEGORY column
        title: Title for the plot
    """
    category_counts = df["PARENT_CATEGORY"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(
        category_counts,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
        startangle=140,
    )
    plt.title(title)
    plt.legend(category_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    save_and_close(f'{title.lower().replace(" ", "_")}.png', 1)


def main():
    # Load and process each data source
    dataframes = {}
    for key in ["DMR", "ESMR", "IR", "TOXICS"]:
        if key == "DMR":
            data = read_data_year(2023, key)
            processed = (
                data[["PARAMETER_CODE", "PARAMETER_DESC", "POLLUTANT_CODE"]]
                .drop_duplicates(subset=["PARAMETER_CODE"])
                .reset_index(drop=True)
            )
            print(f"{len(processed)} unique parameters in DMR 2023 data")
            dataframes[key] = processed
            continue

        if key in FILE_CONFIGS and "step1" in FILE_CONFIGS[key]:
            cfg = FILE_CONFIGS[key]["step1"]

            # Load CSV data
            if key == "ESMR":
                data = read_data_year(2023, key)
                processed = pd.DataFrame(
                    {cfg["desc_col"]: data[cfg["column"]].unique()}
                )
                dataframes[key] = processed
                continue
            else:
                # Get file path from get_data_file_path
                file_path = str(get_data_file_path(key, cfg.get("year")))
                print(file_path)
                data = pd.read_csv(file_path, skiprows=cfg.get("skiprows", 0))

            # Process data
            # Extract column and rename
            processed = data[[cfg["column"]]].drop_duplicates().reset_index(drop=True)
            processed.rename(
                columns={processed.columns[0]: cfg["desc_col"]}, inplace=True
            )

            # Apply post-processing if specified
            post_proc = cfg.get("post_process", {})
            if post_proc.get("strip_prefix"):
                pattern = post_proc["strip_prefix"]["pattern"]
                processed[cfg["desc_col"]] = processed[cfg["desc_col"]].str.replace(
                    pattern, "", regex=post_proc["strip_prefix"].get("regex", False)
                )

            dataframes[key] = processed

    # Categorize parameters
    category_cols = {
        "DMR": "PARAMETER_DESC",
        "IR": "IR_PARAMETER_DESC",
        "ESMR": "ESMR_PARAMETER_DESC",
        "TOXICS": "TOXICS_PARAMETER_DESC",
    }
    for key, desc_col in category_cols.items():
        categorize_parameters(dataframes[key], parameter_sorting_dict, desc_col)

    # Save ir_parameter_df
    dataframes["IR"].to_csv(f"STEP_DIRS{1}/ir_parameter_df.csv", index=False)

    # Plot category distributions
    for key in category_cols.keys():
        plot_pie_counts(dataframes[key], f"{key} Categories")

    # Parameter name matching
    for key in ["ESMR", "TOXICS"]:
        target_df = dataframes[key]

        # Normalize target descriptions
        target_df["normalized_desc"] = target_df[f"{key}_PARAMETER_DESC"].apply(
            normalize_param_desc
        )

        # Match DMR to target
        matched_col = f"{key}_PARAMETER_DESC_MATCHED"
        dataframes["DMR"][matched_col] = dataframes["DMR"].apply(
            lambda row: match_param_desc(row, target_df, f"{key}_PARAMETER_DESC"),
            axis=1,
        )

        # Print match statistics
        unique_matched = len(dataframes["DMR"][matched_col].unique()) - 1
        print(
            f"{unique_matched} out of {len(dataframes['DMR'])} "
            f"auto matched to {key} PARAMETER_DESC"
        )

    # Add manual mappings for ESMR
    manual_mapping = pd.read_csv(
        "data/manual_updates/dmr_esmr_mapping_manual.csv"
    ).set_index("PARAMETER_CODE")["ESMR_PARAMETER_DESC_MANUAL"]

    dataframes["DMR"]["ESMR_PARAMETER_DESC_MANUAL"] = (
        dataframes["DMR"]["PARAMETER_CODE"].map(manual_mapping).fillna("")
    )
    dataframes["DMR"]["ESMR_PARAMETER_DESC"] = (
        dataframes["DMR"]["ESMR_PARAMETER_DESC_MATCHED"]
        .fillna(dataframes["DMR"]["ESMR_PARAMETER_DESC_MANUAL"])
        .fillna("No Match (unconfirmed)")
    )

    # Final cleanup and save
    dataframes["DMR"] = (
        dataframes["DMR"]
        .drop(columns=["ESMR_PARAMETER_DESC_MATCHED", "ESMR_PARAMETER_DESC_MANUAL"])
        .rename(columns={"PARAMETER_DESC": "DMR_PARAMETER_DESC"})
    )
    dataframes["DMR"].to_csv(f"STEP_DIRS{1}/dmr_esmr_mapping.csv", index=False)


if __name__ == "__main__":
    main()
