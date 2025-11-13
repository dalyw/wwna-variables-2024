import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from difflib import SequenceMatcher
from helper_functions import load_data, STEP_DIRS, save_fig, YEAR_RANGE


with open("data/manual_updates/parameter_sorting_dict.json", "r") as f:
    PARAMETER_SORTING_DICT = json.load(f)


def clean_param_code(series):
    """Normalize parameter codes by converting to string and stripping leading zeros."""
    return series.astype(str).str.lstrip("0")


ref_parameter = pd.read_csv(
    "data/dmr/REF_Parameter.csv", dtype={"POLLUTANT_CODE": "Int64"}
).assign(PARAMETER_CODE=lambda df: clean_param_code(df["PARAMETER_CODE"]))
ref_parameter_no_desc = ref_parameter.copy().drop(
    columns=["PARAMETER_DESC"], errors="ignore"
)


def _iter_category_rules(tree, parent=None):
    """Recursive function to help apply categories and sub-categories"""
    for key, node in tree.items():
        if not isinstance(node, dict):
            continue
        if "values" in node:
            yield parent or key, key, node["values"], node.get("case", False)
        else:
            yield from _iter_category_rules(node, parent=key)


MANUAL_MAPPING = (
    pd.read_csv("data/manual_updates/dmr_esmr_mapping_manual.csv")
    .assign(PARAMETER_CODE=lambda df: clean_param_code(df["PARAMETER_CODE"]))
    .set_index("PARAMETER_CODE")["ESMR_PARAMETER_DESC_MANUAL"]
)

CATEGORY_RULES = [
    (parent, child, "|".join(map(re.escape, values)), case)
    for parent, child, values, case in _iter_category_rules(PARAMETER_SORTING_DICT)
    if values
]


def normalize_param_desc(desc):
    desc = str(desc)
    replacements = {",": "", " ": "", "'": "", ".": "", "[": "(", "]": ")", "&": "and"}
    for word in ("sum", "total", "tot."):
        replacements[f", {word}"] = ""
        replacements[f", {word.capitalize()}"] = ""
    for old, new in replacements.items():
        desc = desc.replace(old, new)
    return desc.lower()


def load_and_apply_categories(name, year):
    column = f"{name}_PARAMETER_DESC"
    if name == "DMR":
        # For DMR, use all parameter codes from REF_Parameter.csv
        df = ref_parameter[
            ["PARAMETER_CODE", "PARAMETER_DESC", "POLLUTANT_CODE"]
        ].copy()
        df = df.rename(columns={"PARAMETER_DESC": column})
    else:
        df = load_data(name, year)
    df = df.rename(columns={"PARAMETER_DESC": column})
    df = df.drop_duplicates(subset=column)
    if name == "TOXICS":
        df[column] = df[column].str.replace(r"^\d+\.\s*", "", regex=True)
    df = df.reset_index(drop=True)
    df = df.assign(
        **{
            column: df[column].fillna(""),
            "PARENT_CATEGORY": "Uncategorized",
            "SUB_CATEGORY": "Uncategorized",
        }
    )
    descriptions = df[column].astype(str)
    for parent, child, pattern, case in CATEGORY_RULES:
        mask = descriptions.str.contains(pattern, case=case, na=False)
        if mask.any():
            df.loc[mask, ["PARENT_CATEGORY", "SUB_CATEGORY"]] = parent, child
    return df


def main():
    recent_year = YEAR_RANGE[-1]

    # Load and categorize all data sources
    dataframes = {
        "DMR": load_and_apply_categories("DMR", recent_year),
        "ESMR": load_and_apply_categories("ESMR", recent_year),
        "IR": load_and_apply_categories("IR", 2024),
        "TOXICS": load_and_apply_categories("TOXICS", recent_year),
    }

    dmr = dataframes["DMR"]
    manual_params = pd.read_csv("data/manual_updates/parameters_manual_additions.csv")

    # Build IR parameter catalog from multiple sources
    ir_df = dataframes["IR"].copy()
    ir_df = (
        pd.concat(
            [
                ir_df,
                manual_params[["IR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]],
                dmr[["DMR_PARAMETER_DESC", "PARENT_CATEGORY", "SUB_CATEGORY"]]
                .rename(columns={"DMR_PARAMETER_DESC": "IR_PARAMETER_DESC"})
                .assign(
                    PARENT_CATEGORY=lambda df: df["PARENT_CATEGORY"].fillna("Uncommon"),
                    SUB_CATEGORY=lambda df: df["SUB_CATEGORY"].fillna("Uncommon"),
                ),
                pd.concat(
                    [
                        load_data("LIMITS", year)[["PARAMETER_DESC"]]
                        for year in range(YEAR_RANGE[0], YEAR_RANGE[1])
                    ],
                    ignore_index=True,
                )
                .drop_duplicates()
                .dropna()
                .rename(columns={"PARAMETER_DESC": "IR_PARAMETER_DESC"})
                .assign(PARENT_CATEGORY="Uncommon", SUB_CATEGORY="Uncommon"),
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset="IR_PARAMETER_DESC", keep="first")
        .reset_index(drop=True)
    )
    ir_df.to_csv(f"{STEP_DIRS[1]}/ir_parameter_df_py.csv", index=False)
    dataframes["IR"] = ir_df

    # Plot pie charts
    for key, df in dataframes.items():
        category_counts = df["PARENT_CATEGORY"].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(category_counts, autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "")
        plt.title(f"{key} Categories")
        plt.legend(
            category_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
        )
        save_fig(f'{key.lower().replace(" ", "_")}.png', 1)

    # Match DMR parameters to ESMR and TOXICS using normalized descriptions
    for source_name in ("ESMR", "TOXICS"):
        source_col = f"{source_name}_PARAMETER_DESC"
        normalized = (
            dataframes[source_name][source_col].fillna("").map(normalize_param_desc)
        )
        lookup = pd.Series(
            dataframes[source_name][source_col].fillna("").values,
            index=normalized,
        )
        lookup = lookup[lookup.index.str.len() > 0].drop_duplicates(keep="first")
        normalized_dmr = dmr["DMR_PARAMETER_DESC"].map(normalize_param_desc)
        matched = normalized_dmr.map(lookup).fillna("")
        dmr[f"{source_name}_PARAMETER_DESC_MATCHED"] = matched
        print(
            f"{matched[matched != ''].nunique()} of {len(dmr)} exact matched to {source_name}"
        )

    # Apply manual mappings first, then exact normalized matches
    dmr["ESMR_PARAMETER_DESC"] = (
        dmr["PARAMETER_CODE"]
        .map(MANUAL_MAPPING)
        .fillna(dmr["ESMR_PARAMETER_DESC_MATCHED"].replace("", pd.NA))
    )
    
    # Apply similarity-based matching for remaining unmatched (>0.9 similarity)
    # Only for parameters not already matched through exact or manual
    print("Running similarity-based matching")
    unmatched_mask = dmr["ESMR_PARAMETER_DESC"].isna()
    
    # Get ESMR descriptions already mapped (by manual or exact)
    already_mapped_esmr = set(
        dmr[dmr["ESMR_PARAMETER_DESC"].notna()]["ESMR_PARAMETER_DESC"].unique()
    )
    if unmatched_mask.any():
        esmr_descs = dataframes["ESMR"]["ESMR_PARAMETER_DESC"].fillna("").unique()
        esmr_descs = [d for d in esmr_descs if d and d not in already_mapped_esmr]  # Exclude already mapped
        
        similarity_matches = []
        for idx in dmr[unmatched_mask].index:
            dmr_desc = dmr.loc[idx, "DMR_PARAMETER_DESC"]
            dmr_normalized = normalize_param_desc(dmr_desc)
            
            # Find best match by similarity
            best_match = None
            best_sim = 0.0
            for esmr_desc in esmr_descs:
                esmr_normalized = normalize_param_desc(esmr_desc)
                sim = SequenceMatcher(None, dmr_normalized, esmr_normalized).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_match = esmr_desc
            
            # Only use if similarity >0.9
            if best_sim > 0.9:
                similarity_matches.append((idx, best_match, best_sim))
        
        # Apply similarity matches
        if similarity_matches:
            for idx, esmr_desc, sim in similarity_matches:
                dmr.loc[idx, "ESMR_PARAMETER_DESC"] = esmr_desc
            print(f"Applied {len(similarity_matches)} similarity-based matches (>0.9)")
    
    dmr = dmr.drop(columns=["ESMR_PARAMETER_DESC_MATCHED"])

    # Remove ambiguous auto-matched mappings (multiple DMR codes -> same ESMR desc)
    # This catches any remaining conflicts (e.g., multiple similarity matches to same ESMR)
    is_auto_matched = dmr["ESMR_PARAMETER_DESC"].notna() & ~dmr["PARAMETER_CODE"].isin(
        MANUAL_MAPPING.index
    )
    esmr_counts = dmr.loc[is_auto_matched, "ESMR_PARAMETER_DESC"].value_counts()
    ambiguous_esmr = esmr_counts[esmr_counts > 1].index
    if len(ambiguous_esmr) > 0:
        dmr.loc[
            is_auto_matched & dmr["ESMR_PARAMETER_DESC"].isin(ambiguous_esmr),
            "ESMR_PARAMETER_DESC",
        ] = pd.NA
        print(f"Removed {len(ambiguous_esmr)} ambiguous auto-matched ESMR mappings")

    # Fill remaining NAs
    dmr["ESMR_PARAMETER_DESC"] = dmr["ESMR_PARAMETER_DESC"].fillna(
        "No Match (unconfirmed)"
    )

    # Save
    dmr.to_csv(f"{STEP_DIRS[1]}/dmr_esmr_mapping_py.csv", index=False)
    dataframes["DMR"] = dmr


if __name__ == "__main__":
    main()
