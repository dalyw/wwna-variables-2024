# Configuration settings for WWNA Variables 2024 analysis

# Analysis Settings
ANALYSIS_RANGE = range(2014, 2025)
SAVE_DEFAULT = True
LOAD_DEFAULT = True

# File Paths
FACILITIES_LIST_PATH = "data/facilities_list/NPDES+WDR Facilities List_20240906.csv"
IR_PARAMETER_PATH = "data/ir/ir_parameter_list.csv"
REF_PARAMETER_PATH = "data/dmrs/REF_Parameter.csv"
REF_FREQUENCY_PATH = "data/dmrs/REF_FREQUENCY_OF_ANALYSIS.csv"
ESMR_DATA_DICT_PATH = "data/esmr/esmr_data_dictionary.csv"
TOXICS_CRITERIA_PATH = "data/toxics/criteria_for_toxics.csv"

# DMR Data Columns to Keep
COLUMNS_TO_KEEP_DMR = [
    "EXTERNAL_PERMIT_NMBR",
    "PERMIT_COMPONENT_TYPE_CODE",
    "PERMIT_COMPONENT_TYPE_DESC",
    "MONITORING_PERIOD_END_DATE",
    "MONITORING_PERIOD_END_DATE_NUMERIC",
    "PARAMETER_CODE",
    "PARAMETER_DESC",
    "LIMIT_VALUE_NMBR",
    "LIMIT_VALUE_STANDARD_UNITS",
    "DMR_VALUE_NMBR",
    "DMR_VALUE_STANDARD_UNITS",
    "VIOLATION_CODE",
    "VIOLATION_DESC",
    "SIGNIFICANT_VIOLATION",
    "SIGNIFICANT_VIOLATION_DESC",
]

# Plotting Settings
FIGURE_SIZE = (12, 8)
DPI = 300
FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 10

# Color Schemes
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Data Processing Settings
CHUNK_SIZE = 10000
MAX_WORKERS = 4
MEMORY_LIMIT_GB = 8

# Output Settings
OUTPUT_DIR = "processed_data"
FIGURES_DIR = "figures_py"
CSV_DIR = "csv"

# Validation Settings
MIN_DATA_POINTS = 5
OUTLIER_THRESHOLD = 3.0
CORRELATION_THRESHOLD = 0.7

# Geographic Settings
DEFAULT_CRS = "EPSG:4326"
CALIFORNIA_CRS = "EPSG:32610"
CALIFORNIA_BOUNDS = {
    "min_lon": -124.5,
    "max_lon": -114.0,
    "min_lat": 32.5,
    "max_lat": 42.0,
}

# Parameter Categories
PARAMETER_CATEGORIES = {
    "Pathogens": ["E. coli", "Fecal Coliform", "Total Coliform"],
    "Mercury": ["Mercury", "Total Mercury"],
    "Toxic Organics": ["PCBs", "Dioxins", "Pesticides"],
    "Disinfectants": ["Chlorine", "Chloramine", "Ozone"],
    "Nutrients": ["Nitrogen", "Phosphorus", "Ammonia"],
    "Metals": ["Lead", "Copper", "Zinc", "Cadmium"],
    "Uncategorized": [],
}

# Facility Types
FACILITY_TYPES = {
    "Municipal": ["Municipal", "City", "Town", "District"],
    "Industrial": ["Industrial", "Manufacturing", "Processing"],
    "Agricultural": ["Agricultural", "Farm", "Ranch"],
    "Other": [],
}

# Compliance Thresholds
COMPLIANCE_THRESHOLDS = {
    "exceedance": 1.0,
    "near_exceedance": 0.8,
    "trend_threshold": 0.1,
}

# Data Quality Settings
MISSING_DATA_THRESHOLD = 0.1
DUPLICATE_THRESHOLD = 0.05
OUTLIER_DETECTION_METHOD = "iqr"

# Export Settings
EXPORT_FORMATS = ["csv", "xlsx", "json"]
COMPRESSION_FORMATS = ["gzip", "bz2", "xz"]

# Performance Settings
USE_MULTIPROCESSING = True
CACHE_RESULTS = True
PARALLEL_BACKEND = "multiprocessing"

# Debug Settings
DEBUG_MODE = False
VERBOSE_OUTPUT = False
SAVE_INTERMEDIATE_RESULTS = False
