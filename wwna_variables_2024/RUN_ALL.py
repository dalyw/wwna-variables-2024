import os
import sys
import pandas as pd
from wwna_variables_2024.helper_functions import (
    FACILITIES_LIST_PATH,
    load_facilities_list,
)


def run_script(script_name):
    """Run a Python script and check its exit code."""
    print(f"Starting {script_name}")
    result = os.system(f"python {script_name}")
    if result != 0:
        print(f"Error running {script_name}")
        sys.exit(1)


def generate_final_facilities_list():
    """Generate final facilities list with all risk factors."""

    # Load results from previous steps
    facilities_list = load_facilities_list(FACILITIES_LIST_PATH)
    population_data = pd.read_csv(f"STEP_DIRS{2}/merged_population_data.csv")
    exceedance_data = pd.read_csv(f"STEP_DIRS{3}/flagged_facilities_step3.csv")
    future_limits_data = pd.read_csv(f"STEP_DIRS{4}/flagged_facilities_step4.csv")

    # Add population data
    facilities_list = facilities_list.merge(
        population_data, left_on="FACILITY ID", right_on="CWNS_ID", how="left"
    )

    # Add exceedance data (already aggregated)
    facilities_list = facilities_list.merge(
        exceedance_data,
        left_on="NPDES # CA#",
        right_on="EXTERNAL_PERMIT_NMBR",
        how="left",
    )

    # Add future limits columns
    future_cols = [
        "NPDES # CA#",
        "Parameters Discharged into Newly Impaired Water Body and Not Yet Limited",
        "Discharges to Impaired and Not Limited: Number of Parameters",
    ]
    facilities_list = facilities_list.merge(
        future_limits_data[future_cols], on="NPDES # CA#", how="left"
    )

    # Fill NA and drop columns
    fill_na = {
        "Number of Parameters with Slope and Near Exceedance": 0,
        "Parameters with Slope and Near Exceedance": "",
        "Parameters Discharged into Newly Impaired Water Body and Not Yet Limited": "",
        "Discharges to Impaired and Not Limited: Number of Parameters": 0,
    }
    facilities_list = facilities_list.fillna(fill_na)

    # Remove duplicates
    initial_count = len(facilities_list)
    facilities_list = facilities_list.drop_duplicates(
        subset=["FACILITY ID"], keep="first"
    )
    if initial_count != len(facilities_list):
        print(f"Removed {initial_count - len(facilities_list)} duplicates")

    # Save
    facilities_list.to_csv("processed_data/facilities_list_updated.csv", index=False)
    print("Saved updated facilities list")


def main(skip_steps=None):
    # Create processed_data directory and subdirectories if they don't exist
    os.makedirs("processed_data", exist_ok=True)
    for i in range(1, 5):
        os.makedirs(f"STEP_DIRS{i}", exist_ok=True)
        os.makedirs(f"STEP_DIRS{i}/figures_py", exist_ok=True)

    # List of scripts to run in order
    scripts = [
        "step0_download_data.py",
        "step1_parameter_categorization.py",
        "step2_population_served.py",
        "step3_near_exceedance.py",
        "step4_future_limits.py",
    ]

    # Run each script in sequence, unless skipped
    steps_to_skip = set(skip_steps) if skip_steps else set()
    for i, script in enumerate(scripts, 1):
        if str(i) in steps_to_skip:
            print(f"Skipping step {i} ({script})")
            continue

        script_path = f"wwna_variables_2024/{script}"
        if not os.path.exists(script_path):
            print(f"Error: {script} not found at {script_path}")
            sys.exit(1)

        print(f"Running step {i-1}: {script}")
        run_script(script_path)

    # Generate final facilities list
    generate_final_facilities_list()


if __name__ == "__main__":
    # To skip steps 1 and 3, call main(['1', '3'])
    # main(['1','2','3'])
    main()
