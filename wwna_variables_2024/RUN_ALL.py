import os
import sys
import pandas as pd
from wwna_variables_2024.helper_functions import (
    WWNA_LIST,
    STEP_DIRS,
    AGG_STRINGS,
    NPDES_FROM_WWNA_LIST,
)


def run_script(script_name, script_args=None):
    """Run a Python script and check its exit code."""
    print(f"Running {script_name}")
    cmd = f"python {script_name}"
    if script_args:
        for key, value in script_args.items():
            arg_key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd += f" --{arg_key}"
            else:
                cmd += f" --{arg_key} {value}"

    if os.system(cmd) != 0:
        print(f"Error running {script_name}")
        sys.exit(1)


def main(skip_steps=None, num_processes=4, exclude_noncompliant=True):
    WWNA_LIST_FINAL = WWNA_LIST.copy()
    print(f"{len(NPDES_FROM_WWNA_LIST)} of {len(WWNA_LIST)} WWNA facilities have NPDES")

    # Create processed_data directory and subdirectories if they don't exist
    os.makedirs("processed_data", exist_ok=True)
    for i in range(1, 5):
        os.makedirs(f"{STEP_DIRS[i]}", exist_ok=True)
        os.makedirs(f"{STEP_DIRS[i]}/figures_py", exist_ok=True)
    os.makedirs(f"{STEP_DIRS[3]}/csvs_py", exist_ok=True)

    # Run each script in sequence, unless skipped
    steps_to_skip = set(skip_steps) if skip_steps else set()
    SCRIPTS = [
        "step0_download_data.py",
        "step1_parameter_categorization.py",
        "step2_population_served.py",
        "step3_near_exceedance.py",
        "step4_future_limits.py",
    ]
    for i, script in enumerate(SCRIPTS, 1):
        if str(i) in steps_to_skip:
            print(f"Skipping step {i} ({script})")
            continue
        # Pass arguments for step3
        script_args = None
        if script == "step3_near_exceedance.py":
            script_args = {
                "exclude_noncompliant": exclude_noncompliant,
            }
        run_script(f"wwna_variables_2024/{script}", script_args=script_args)

    # Load results from previous steps (use _py suffix for Python outputs)
    population = pd.read_csv(f"{STEP_DIRS[2]}/merged_population_data_py.csv")
    exceedance = pd.read_csv(f"{STEP_DIRS[3]}/flagged_facilities_step3_py.csv")
    future_limits = pd.read_csv(f"{STEP_DIRS[4]}/flagged_facilities_step4_py.csv")
    future_limits_merge_cols = [
        "NPDES # CA#",
        AGG_STRINGS["4"]["COUNT"],
        AGG_STRINGS["4"]["PARAM"],
    ]

    # Merge additional data sources
    print(f"Original WWNA_LIST length: {len(WWNA_LIST_FINAL)}")

    # Population: step2 output is already keyed to WWNA facilities (same row order)
    # Just add the population columns directly
    pop_cols = [c for c in population.columns if c not in WWNA_LIST_FINAL.columns]
    for col in pop_cols:
        WWNA_LIST_FINAL[col] = population[col].values
    n_pop = WWNA_LIST_FINAL["Population Served"].notna().sum()
    print(f" After population merge: {n_pop} with population data")

    # Exceedance and future limits: NPDES-only merge
    for df, right_on in [
        (exceedance, "EXTERNAL_PERMIT_NMBR"),
        (future_limits[future_limits_merge_cols], "NPDES # CA#"),
    ]:
        WWNA_LIST_FINAL = WWNA_LIST_FINAL.merge(
            df, left_on="NPDES # CA#", right_on=right_on, how="left"
        )
        print(f" After merge: {len(WWNA_LIST_FINAL)} rows")

    # Save
    WWNA_LIST_FINAL.to_csv("processed_data/WWNA_LIST_updated_py.csv", index=False)
    print("Saved updated facilities list")


if __name__ == "__main__":
    # To skip e.g. steps 1 and 3, call main(['1', '3'])
    # main(["1", "2"])
    main()
