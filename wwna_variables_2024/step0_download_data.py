import zipfile
import requests
import pandas as pd
from pathlib import Path
from wwna_variables_2024.helper_functions import (
    analysis_range,
    ESMR_RESOURCE_IDS,
    get_data_filename,
    get_data_dir_path,
    get_data_file_path,
)

BASE_DIRS = {
    "DMR": {
        "url": "https://echo.epa.gov/files/echodownloads/NPDES_by_state_year",
        "size_threshold": 1_000_000,
    },
    "ESMR": {
        "url": "https://data.ca.gov/dataset/203e5d1f-ec9d-4d07-93aa-d8b74d3fe71f/resource",
        "size_threshold": 10_000_000,
    },
    "IR": {
        "url": "https://www.waterboards.ca.gov/water_issues/programs/tmdl",
        "size_threshold": 1_000_000,
    },
    "SSO": {
        "url": "https://www.waterboards.ca.gov/water_issues/programs/sso/docs/data_files/Questionnaire.txt",
        "size_threshold": 100_000,
    },
    "TOXICS": {
        "url": "https://data.ca.gov/dataset/7b2b5d26-9407-4368-8744-d5b659024dd7/resource/0d417a2b-6559-4725-820f-add7c57a8bc9/download/oehha-toxicity-criteria-database-20250408.csv",
        "size_threshold": 1_000,
    },
    "CWNS": {
        "url": "https://raw.githubusercontent.com/dalyw/us-sewersheds/refs/heads/main/processed_data/facilities_merged.csv",
        "size_threshold": 100_000,
    },
}
for data_type in ["DMR", "ESMR", "IR", "SSO", "TOXICS", "CWNS"]:
    BASE_DIRS[data_type]["dir"] = Path(f"data/{data_type.lower()}")
    BASE_DIRS[data_type]["dir"].mkdir(parents=True, exist_ok=True)

IR_303D_PATHS = {
    2018: "2018state_ir_reports_final/app_a_2018303d.xlsx",
    2022: "2020_2022state_ir_reports_revised_final/apx-a-303d-list.xlsx",
    2024: "2023_2024state_ir_reports/apx-a-2024-303d-list-final.xlsx",
}


def download_file(url, file_path):
    """Helper to download a file from URL to path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_dmr_year(year):
    """Download and extract one year of DMR data."""
    zipname = f"CA_FY{year}_NPDES_DMRS_LIMITS.zip"
    url = f"{BASE_DIRS['DMR']['url']}/{zipname}"
    zip_path = BASE_DIRS["DMR"]["dir"] / zipname
    print(f"Downloading DMR for {year}")
    download_file(url, zip_path)

    target_folder = get_data_dir_path("DMR", year)
    target_folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            dest = target_folder / Path(file).name
            with zip_ref.open(file) as source, open(dest, "wb") as target:
                target.write(source.read())

    # Delete the zip file
    zip_path.unlink()


def download_esmr_year(year):
    """Download one year of eSMR data."""
    print(f"Downloading eSMR for {year}")
    resource_id = ESMR_RESOURCE_IDS[year]
    filename = get_data_filename("ESMR", year)
    url = f"{BASE_DIRS['ESMR']['url']}/{resource_id}/download/{filename.rstrip('.csv')}_2025-10-06.csv"
    file_path = get_data_file_path("ESMR", year)
    download_file(url, file_path)


def download_ir_year(year):
    """Download IR data as xlsx and convert to csv."""
    print(f"Downloading IR for {year}")
    xlsx_filename = IR_303D_PATHS[year]
    url = f"{BASE_DIRS['IR']['url']}/{xlsx_filename}"

    temp_xlsx = BASE_DIRS["IR"]["dir"] / f"temp_{year}-303d.xlsx"
    download_file(url, temp_xlsx)

    # Save as CSV in the correct location
    df = pd.read_excel(temp_xlsx)
    csv_path = get_data_file_path("IR", year)
    df.to_csv(csv_path, index=False)

    # Delete temporary xlsx
    temp_xlsx.unlink()


def download_sso():
    """Download SSO data from txt and convert to csv."""
    print("Downloading SSO")
    url = BASE_DIRS["SSO"]["url"]
    csv_path = BASE_DIRS["SSO"]["dir"] / "Questionnaire.csv"

    # Download txt file
    txt_path = BASE_DIRS["SSO"]["dir"] / "Questionnaire.txt"
    download_file(url, txt_path)

    # Convert to CSV
    print("Converting SSO from txt to csv")
    df = pd.read_csv(txt_path, sep="\t", low_memory=False)

    # Convert numeric columns that have comma-separated values
    for col in df.columns:
        if col.startswith("SSOq") and "Population" in col:
            # Remove commas and convert to float
            df[col] = df[col].astype(str).str.replace(",", "").replace("", None)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(csv_path, index=False)


def download_data_by_type(data_type, year_range="Base"):
    """Download data for the given type."""
    for item in year_range:
        # Get paths using helper functions
        if data_type == "DMR":
            base_dir_item = get_data_dir_path(data_type, item)
        else:
            base_dir_item = get_data_dir_path(data_type)
        filename = get_data_filename(data_type, item)
        path_to_check = base_dir_item / filename

        if path_to_check.exists():
            if path_to_check.stat().st_size > BASE_DIRS[data_type]["size_threshold"]:
                continue

        print(f"{data_type} {item} missing or corrupted")
        if path_to_check.exists():
            path_to_check.unlink()
        try:
            if data_type == "DMR":
                download_dmr_year(item)
            elif data_type == "ESMR":
                download_esmr_year(item)
            elif data_type == "IR":
                download_ir_year(item)
            elif data_type == "SSO":
                download_sso()
            else:  # TOXICS or CWNS
                print(f"Downloading {data_type}")
                download_file(BASE_DIRS[data_type]["url"], path_to_check)
        except Exception as e:
            print(f"Error processing {data_type} {item}: {e}")


if __name__ == "__main__":
    download_data_by_type("DMR", year_range=analysis_range)
    download_data_by_type("ESMR", year_range=analysis_range)
    download_data_by_type("IR", year_range=[2018, 2022, 2024])
    download_data_by_type("SSO")
    download_data_by_type("TOXICS")
    download_data_by_type("CWNS")
