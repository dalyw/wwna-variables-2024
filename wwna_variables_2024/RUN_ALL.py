import os
import sys
import pandas as pd
from wwna_variables_2024.helper_functions import (
    WWNA_LIST,
    STEP_DIRS,
    SCRIPTS,
    AGG_STRINGS,
    agg_columns,
)


def run_script(script_name):
    """Run a Python script and check its exit code."""
    print(f"Running {script_name}")
    result = os.system(f"python {script_name}")
    if result != 0:
        print(f"Error running {script_name}")
        sys.exit(1)


def main(skip_steps=None):
    WWNA_LIST_FINAL = WWNA_LIST.copy()

    # Create processed_data directory and subdirectories if they don't exist
    os.makedirs("processed_data", exist_ok=True)
    for i in range(1, 5):
        os.makedirs(f"{STEP_DIRS[i]}", exist_ok=True)
        os.makedirs(f"{STEP_DIRS[i]}/figures_py", exist_ok=True)
        os.makedirs(f"{STEP_DIRS[i]}/figures_R", exist_ok=True)

    # Run each script in sequence, unless skipped
    steps_to_skip = set(skip_steps) if skip_steps else set()
    for i, script in enumerate(SCRIPTS, 1):
        if str(i) in steps_to_skip:
            print(f"Skipping step {i} ({script})")
            continue
        run_script(f"wwna_variables_2024/{script}")

    # Load results from previous steps (use _py suffix for Python outputs)
    population = pd.read_csv(f"{STEP_DIRS[2]}/merged_population_data_py.csv")
    exceedance = pd.read_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3_py.csv")
    future_limits = pd.read_csv(f"{STEP_DIRS[4]}/flagged_facilities_step4_py.csv")

    print(f"Original WWNA_LIST length: {len(WWNA_LIST_FINAL)}")

    # Merge additional data sources
    for df, right_on in [
        (population, "PERMIT_NUMBER"),
        (exceedance, "EXTERNAL_PERMIT_NMBR"),
        (
            future_limits[
                ["NPDES # CA#", AGG_STRINGS["4"]["COUNT"], AGG_STRINGS["4"]["PARAM"]]
            ],
            "NPDES # CA#",
        ),
    ]:
        print(f"\nMerging: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"  - WWNA_LIST_FINAL: {len(WWNA_LIST_FINAL)} rows")
        
        # TODO: move to step2
        # Deduplicate population data before merging (some facilities have multiple CWNS records)
        if right_on == "PERMIT_NUMBER":
            # For population, keep first non-null value for each permit
            df = df.drop_duplicates(subset=[right_on], keep='first')
            print(f"  - After deduplication: {df.shape[0]} rows")
        
        WWNA_LIST_FINAL = WWNA_LIST_FINAL.merge(
            df, left_on="NPDES # CA#", right_on=right_on, how="left"
        )
        print(f"  - After merge: {len(WWNA_LIST_FINAL)} rows")

    fill_na = {
        col: 0 if "Number" in col else ""
        for col in agg_columns
        if col in WWNA_LIST_FINAL.columns
    }
    WWNA_LIST_FINAL = WWNA_LIST_FINAL.fillna(fill_na)

    # Remove duplicates
    initial_count = len(WWNA_LIST_FINAL)
    WWNA_LIST_FINAL = WWNA_LIST_FINAL.drop_duplicates(
        subset=["FACILITY ID"], keep="first"
    )
    if initial_count != len(WWNA_LIST_FINAL):
        print(f"Removed {initial_count - len(WWNA_LIST_FINAL)} duplicates")

    # Save
    WWNA_LIST_FINAL.to_csv("processed_data/WWNA_LIST_FINAL_updated_py.csv", index=False)
    print("Saved updated facilities list")


if __name__ == "__main__":
    # To skip steps 1 and 3, call main(['1', '3'])
    # main(["1", "2", "4"])
    main()
