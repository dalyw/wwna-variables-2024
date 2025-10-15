import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"processed_data/logging/"
            f"run_log_"
            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            f".log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_script(script_name):
    """Run a Python script and check its exit code."""
    logger.info(f"Starting {script_name}")
    result = os.system(f"python {script_name}")

    if result != 0:
        logger.error(f"Error running {script_name}")
        sys.exit(1)

    logger.info(f"Completed {script_name} successfully")


def generate_final_facilities_list():
    """Generate final facilities list with all risk factors."""
    logger.info("Generating final facilities list...")

    # Load original facilities list
    facilities_list = pd.read_csv(
        "data/facilities_list/NPDES+WDR Facilities List_20240906.csv"
    )

    proc_data = "processed_data"
    # Load results from previous steps
    population_data = pd.read_csv(
        proc_data + "/step2/merged_population_data.csv"
    )
    exceedance_data = pd.read_csv(
        proc_data + "/step3/facilities_with_slope_and_near_exceedance.csv"
    )
    future_limits_data = pd.read_csv(
        proc_data + "/step4/facilities_with_future_limits.csv"
    )

    # Add population data
    logger.info("Adding population data...")
    facilities_list = facilities_list.merge(
        population_data[["CWNS_ID", "population_mean", "population_cv"]],
        left_on="FACILITY ID",
        right_on="CWNS_ID",
        how="left",
    ).rename(
        columns={
            "population_mean": "Population Served",
            "population_cv": "Population Data Coefficient of Variation",
        }
    )

    # Add exceedance data
    logger.info("Adding exceedance data...")
    exceedance_counts = (
        exceedance_data.groupby("NPDES_CODE")
        .size()
        .reset_index(
            name="Number of Parameters with Slope and Near Exceedance"
        )
    )
    facilities_list = facilities_list.merge(
        exceedance_counts,
        left_on="NPDES # CA#",
        right_on="NPDES_CODE",
        how="left",
    )

    # Add future limits data
    logger.info("Adding future limits data...")
    facilities_list = facilities_list.merge(
        future_limits_data[
            [
                "NPDES # CA#",
                "Discharges to Impaired Water Bodies and Not Limited",
            ]
        ],
        on="NPDES # CA#",
        how="left",
    )

    # Fill NA values and drop unnecessary columns
    fill_na = {
        "Number of Parameters with Slope and Near Exceedance": 0,
        "Discharges to Impaired Water Bodies and Not Limited": "",
    }
    facilities_list = facilities_list.fillna(fill_na).drop(
        columns=[
            c
            for c in ["NPDES_CODE", "CWNS_ID"]
            if c in facilities_list.columns
        ]
    )

    # Save final output
    facilities_list.to_csv(
        "processed_data/facilities_list_updated.csv", index=False
    )
    logger.info("Final facilities list generated successfully")


def main(skip_steps=None):
    # Create processed_data directory and subdirectories if they don't exist
    [
        os.makedirs(f"processed_data/step{i}", exist_ok=True)
        for i in range(1, 5)
    ]

    # List of scripts to run in order
    scripts = [
        "step1_parameter_categorization.py",
        "step2_population_served.py",
        "step3_near_exceedance.py",
        "step4_future_limits.py",
    ]

    # Determine which steps to skip
    steps_to_skip = set(skip_steps) if skip_steps else set()

    # Run each script in sequence, unless skipped
    for i, script in enumerate(scripts, 1):
        if str(i) in steps_to_skip:
            logger.info(f"Skipping step {i} ({script}) as requested")
            continue

        if not os.path.exists(f"wwna_variables_2024/{script}"):
            logger.error(f"Script {script} not found!")
            sys.exit(1)

        logger.info(f"Running step {i}: {script}")
        run_script(f"wwna_variables_2024/{script}")

    # Generate final facilities list
    generate_final_facilities_list()

    logger.info("All analyses completed successfully!")


if __name__ == "__main__":
    # To skip steps 1 and 3, call main(['1', '3'])
    # main(['1','2','3'])
    main()
