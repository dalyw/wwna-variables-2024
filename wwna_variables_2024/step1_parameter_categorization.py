import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from helper_functions import load_data, FILE_CONFIGS, STEP_DIRS, save_fig

# CATEGORIZE PARAMETERS
with open("data/manual_updates/parameter_sorting_dict.json", "r") as f:
    parameter_sorting_dict = json.load(f)

ref_parameter = pd.read_csv("data/dmr/REF_PARAMETER.csv")


def match_param_desc(row, target_df, target_desc_column):
    """Match the parameter description in the target df to the current row."""
    normalized_desc = normalize_param_desc(str(row["PARAMETER_DESC"]))
    match = target_df[target_df["normalized_desc"] == normalized_desc]
    return match[target_desc_column].iloc[0] if len(match) > 0 else ""


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
                # Leaf node: apply category
                mask = df[desc_column].str.contains(
                    "|".join(map(re.escape, value["values"])),
                    case=value.get("case", False),
                )
                df.loc[mask, "PARENT_CATEGORY"] = parent or key
                df.loc[mask, "SUB_CATEGORY"] = key
            elif isinstance(value, dict):
                # Branch node: recurse
                apply_categories(value, parent=key)

    apply_categories(parameter_sorting_dict)
    return df


def plot_pie_counts(df, title):
    """Plot pie chart of parameter categories."""
    category_counts = df["PARENT_CATEGORY"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(
        category_counts,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
        startangle=140,
    )
    plt.title(title)
    plt.legend(category_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    save_fig(f'{title.lower().replace(" ", "_")}.png', 1)


def main():
    # Load and process each data source
    dataframes = {}
    for key in ["DMR", "ESMR", "IR", "TOXICS"]:
        if key == "DMR":
            data = load_data(key, 2023)
            # Add POLLUTANT_CODE from ref_parameter for step1 processing
            data["PARAMETER_CODE_CLEAN"] = data["PARAMETER_CODE"].str.lstrip("0")
            data = data.merge(
                ref_parameter[["PARAMETER_CODE", "POLLUTANT_CODE"]].rename(
                    columns={"PARAMETER_CODE": "PARAMETER_CODE_CLEAN"}
                ),
                on="PARAMETER_CODE_CLEAN",
                how="left",
            )
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
                data = load_data(key, 2023)
            else:
                data = load_data(key, cfg.get("year"))

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
    # Merge with manually added parameters if they exist
    manual_params_path = "data/manual_updates/parameters_manual_additions.csv"
    dataframes["IR"].to_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv", index=False)

    manual_params = pd.read_csv(manual_params_path)
    # Append manual parameters that aren't already in the file
    existing_params = set(dataframes["IR"]["IR_PARAMETER_DESC"].values)
    new_params = manual_params[
        ~manual_params["IR_PARAMETER_DESC"].isin(existing_params)
    ]
    print(f"Adding {len(new_params)} manually added parameters to ir_parameter_df")
    combined = pd.concat(
        [
            dataframes["IR"],
            new_params[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
        ],
        ignore_index=True,
    )

    # Add unmapped DMR parameters to ir_parameter_df
    dmr_params = (
        dataframes["DMR"][["PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]]
        .rename(columns={"PARAMETER_DESC": "IR_PARAMETER_DESC"})
        .dropna(subset=["IR_PARAMETER_DESC"])
    )

    existing_dmr = set(combined["IR_PARAMETER_DESC"].values)
    new_dmr_params = dmr_params[~dmr_params["IR_PARAMETER_DESC"].isin(existing_dmr)]
    new_dmr_params = new_dmr_params.copy()

    # # Apply keyword-based mapping before Uncommon fallback
    # # Map toxicity-related parameters
    # toxicity_keywords = ["static renewal", "static", "toxicity", "tu ", "pass/fail"]
    # for keyword in toxicity_keywords:
    #     mask = new_dmr_params["IR_PARAMETER_DESC"].str.contains(
    #         keyword, case=False, na=False
    #     )
    #     new_dmr_params.loc[mask, "PARENT_CATEGORY"] = "Toxicity"
    #     new_dmr_params.loc[mask, "SUB_CATEGORY"] = "Toxicity"

    new_dmr_params["PARENT_CATEGORY"] = new_dmr_params["PARENT_CATEGORY"].fillna(
        "Uncommon"
    )
    new_dmr_params["SUB_CATEGORY"] = new_dmr_params["SUB_CATEGORY"].fillna("Uncommon")

    if len(new_dmr_params) > 0:
        print(f"Adding {len(new_dmr_params)} DMR parameters to ir_parameter_df")
        combined = pd.concat([combined, new_dmr_params], ignore_index=True)

    # Add LIMITS parameters that aren't in DMR or IR
    # TODO: see if we can only use DMRs and not LIMITS
    limits_data = load_data("LIMITS", 2023)
    limits_params = pd.DataFrame(
        {"IR_PARAMETER_DESC": limits_data["PARAMETER_DESC"].unique()}
    )

    # Apply keyword-based mapping
    limits_params["PARENT_CATEGORY"] = "Uncommon"
    limits_params["SUB_CATEGORY"] = "Uncommon"

    # # Map toxicity-related parameters
    # toxicity_keywords = ["static renewal", "static", "toxicity", "tu ", "pass/fail"]
    # for keyword in toxicity_keywords:
    #     mask = limits_params["IR_PARAMETER_DESC"].str.contains(
    #         keyword, case=False, na=False
    #     )
    #     limits_params.loc[mask, "PARENT_CATEGORY"] = "Toxicity"
    #     limits_params.loc[mask, "SUB_CATEGORY"] = "Toxicity"

    existing_combined = set(combined["IR_PARAMETER_DESC"].values)
    new_limits_params = limits_params[
        ~limits_params["IR_PARAMETER_DESC"].isin(existing_combined)
    ]

    if len(new_limits_params) > 0:
        print(f"Adding {len(new_limits_params)} LIMITS parameters to ir_parameter_df")
        combined = pd.concat([combined, new_limits_params], ignore_index=True)

    combined.to_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv", index=False)

    # Create parameter reference by merging ref_parameter with combined categories
    ref_parameter_merged = ref_parameter.merge(
        combined[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
        left_on="PARAMETER_DESC",
        right_on="IR_PARAMETER_DESC",
        how="left",
    )

    # Clean PARAMETER_CODE for matching
    ref_parameter_merged["PARAMETER_CODE_CLEAN"] = ref_parameter_merged[
        "PARAMETER_CODE"
    ].str.lstrip("0")

    # Save consolidated reference
    ref_parameter_merged.to_csv(
        f"{STEP_DIRS[1]}/ref_parameter_merged_py.csv", index=False
    )
    print(f"Saved parameter reference with {len(ref_parameter_merged)} parameters")

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
        unique = len(dataframes["DMR"][matched_col].unique()) - 1
        print(f"{unique} of {len(dataframes['DMR'])} auto matched to {key}")

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
    dataframes["DMR"].to_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv", index=False)


if __name__ == "__main__":
    main()
