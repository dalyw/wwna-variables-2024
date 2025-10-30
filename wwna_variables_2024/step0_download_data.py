import zipfile
import requests
import pandas as pd
from pathlib import Path
from wwna_variables_2024.helper_functions import (
    analysis_range,
    FILE_CONFIGS,
    get_data_file_path,
)

# Create data directories
for data_type in ["DMR", "ESMR", "IR", "SSO", "TOXICS", "CWNS"]:
    data_dir = Path(f"data/{data_type.lower()}")
    data_dir.mkdir(parents=True, exist_ok=True)


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
    url = f"{FILE_CONFIGS['DMR']['download']['url']}/{zipname}"
    zip_path = Path("data/dmr") / zipname
    download_file(url, zip_path)

    # Extract the zip file then delete it
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            dest = get_data_file_path("DMR", year) / Path(file).name
            with zip_ref.open(file) as source, open(dest, "wb") as target:
                target.write(source.read())
    zip_path.unlink()


def download_esmr_year(year):
    """Download one year of eSMR data."""
    resource_id = FILE_CONFIGS["ESMR"]["year_config"][str(year)]
    file_path = get_data_file_path("ESMR", year)
    # Use datastore dump url
    url = f"https://data.ca.gov/datastore/dump/{resource_id}?bom=True"
    download_file(url, file_path)


def download_ir_year(year):
    """Download IR data as xlsx and convert to csv."""
    xlsx_filename = FILE_CONFIGS["IR"]["year_config"][str(year)]
    url = f"{FILE_CONFIGS['IR']['download']['url']}/{xlsx_filename}"
    temp_xlsx = Path("data/ir") / f"temp_{year}-303d.xlsx"
    download_file(url, temp_xlsx)

    # Save as CSV in the correct location then delete xlsx
    df = pd.read_excel(temp_xlsx)
    csv_path = get_data_file_path("IR", year)
    df.to_csv(csv_path, index=False)
    temp_xlsx.unlink()


def download_sso():
    """Download SSO data from txt and convert to csv."""
    url = FILE_CONFIGS["SSO"]["download"]["url"]
    csv_path = Path("data/sso") / "Questionnaire.csv"

    # Download txt file and convert to csv
    txt_path = data_dir / "Questionnaire.txt"
    download_file(url, txt_path)
    df = pd.read_csv(txt_path, sep="\t", low_memory=False)

    # Convert numeric columns that have comma-separated values
    for col in df.columns:
        if col.startswith("SSOq") and "Population" in col:
            # Remove commas and convert to float
            df[col] = df[col].astype(str).str.replace(",", "").replace("", None)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(csv_path, index=False)


def download_data_by_type(data_type, year_range=None):
    """Download data for the given type."""
    items = year_range if year_range is not None else [None]
    for item in items:

        # Get paths to check if file already is downloaded
        path_to_check = get_data_file_path(data_type, item)
        if path_to_check.exists() and path_to_check.is_file():
            size_threshold = FILE_CONFIGS[data_type]["download"]["size_threshold"]
            if path_to_check.stat().st_size > size_threshold:
                continue

        if path_to_check.exists() and path_to_check.is_file():
            print(f"{data_type} {item} corrupted. Re-downloading")
            path_to_check.unlink()  # Delete corrupted file

        try:
            print(f"Downloading {data_type}")
            if data_type == "DMR":
                download_dmr_year(item)
            elif data_type == "ESMR":
                download_esmr_year(item)
            elif data_type == "IR":
                download_ir_year(item)
            elif data_type == "SSO":
                download_sso()
            else:  # TOXICS or CWNS
                download_file(FILE_CONFIGS[data_type]["download"]["url"], path_to_check)
        except Exception as e:
            print(f"Error processing {data_type} {item}: {e}")


if __name__ == "__main__":
    download_data_by_type("DMR", year_range=analysis_range)
    download_data_by_type("ESMR", year_range=[analysis_range[-1]])
    download_data_by_type("IR", year_range=[2018, 2022, 2024])
    download_data_by_type("SSO")
    download_data_by_type("TOXICS")
    download_data_by_type("CWNS")
