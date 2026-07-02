# config.py
from pathlib import Path

# =====================================================================
# 1. RUN REGIME
# =====================================================================
RUN_YR = 2026
RUN_MO = 6
DATASET_TYPES = ["ersst", "hadoisst"]

# =====================================================================
# 2. OUTPUT PATHS
# =====================================================================
BASE_DIR = Path("/cpc/home/mgoss/project/ca_dev")
OUT_DATA_DIR = BASE_DIR / "data"
TMP_DIR = BASE_DIR / "tmp"

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Create run-mode specific output directories
OUT_DIR_SEASONAL = OUT_DATA_DIR / "seasonal"
OUT_DIR_MONTHLY = OUT_DATA_DIR / "monthly"

OUT_DIR_SEASONAL.mkdir(parents=True, exist_ok=True)
OUT_DIR_MONTHLY.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 3. ABSOLUTE SYSTEM DATA INPUT SOURCES
# =====================================================================
DATA_SOURCES = {
    "ersst": {
        "binary_path": Path("/cpc/home/wd52ml/enso/oni.v5/data/ersst.v5/ersst.v5.1854.pres.gr"),
        "nx": 180, "ny": 89,
        "archive_start_date": "1854-01-15",
        "lat_edges": (-89.0, 87.0), "lon_edges": (0.0, 358.0),
        "dtype": "<f4",
        "is_kelvin": False
    },
    "had_historical": {
        "binary_path": Path("/cpc/GODAS/bjha/MERGEDSST/ncar.SST.HAD187001-198110.OI198111-201003.gr"),
        "nx": 360, "ny": 180,
        "archive_start_date": "1870-01-15",
        "lat_edges": (-89.5, 89.5), "lon_edges": (0.5, 359.5),
        "dtype": "<f4",         # FIXED: Corrected to Little-Endian
        "is_kelvin": True       # FIXED: Flag for Kelvin-to-Celsius conversion
    },
    "oi_operational": {
        "dir_path": Path("/cpc/GODAS/OISSTv2.1/DATA/month"),
        "template": "mnth.OISSTv2.1_1x1.{year:04d}{month:02d}.gr",
        "nx": 360, "ny": 181,
        "lat_edges": (-90.0, 90.0), "lon_edges": (0.0, 359.0),
        "dtype": "<f4",         # FIXED: Corrected to Little-Endian
        "is_kelvin": False
    }
}

# =====================================================================
# 4. CLIMATE & SPECTRAL DOMAIN CONFIGURATIONS
# =====================================================================
CLIM_START_YR, CLIM_END_YR = 1991, 2020
EOF_RANGE = "tp_ml"
LAT_BOUNDS = slice(-45.5, 45.5)

IC_OFFSETS = [0, 3, 6, 9]           
IC_NUMS = [4, 3, 2, 1]              

TARGET_NX, TARGET_NY = 360, 180
TARGET_LAT_EDGES = (-89.5, 89.5)
TARGET_LON_EDGES = (0.5, 359.5)

UNDEF_FLAGS = [-9.99e8, -999.0, -999000000.0, -99.99]
VALID_SST_MIN = -5.0  
VALID_SST_MAX = 45.0  
KELVIN_OFFSET = 273.15

# High-Performance EOF Controls
MAX_EOF_MODES = 40                 
EVAL_EOF_MODES = [15, 25, 40]       
APPLY_SVD_AREA_WEIGHTING = True     

# Operational Timeline Alignment
ANALOG_EXCLUSION_RADIUS = 1         

# Adaptive Ridge Regression Solver Hyperparameters
RIDGE_PENALTY = 0.01                
ADAPTIVE_RIDGE_STEP = 0.001         
ADAPTIVE_RIDGE_MAX_POWER = 0.5      
ADAPTIVE_RIDGE_MAX_VALUE = 2.0      
ADAPTIVE_RIDGE_MAX_ITER = 500       

# Verification settings
SKILL_VERIF_START_YR = 1981
SKILL_CALC_MODE = "legacy" # Options: "legacy", "ensemble_mean"

# Forecast Lead Horizon Dimensions
FORECAST_LEAD_START = 0             
FORECAST_LEAD_END = 17              

RUN_MODES = {
    "monthly": {"lag_months": 1, "apply_smoothing": False, "plot_lead_offset": 1},
    "seasonal": {"lag_months": 2, "apply_smoothing": True, "plot_lead_offset": 3}
}

# Regional Diagnostic Index Bounding Coordinates
NINO34_LAT_BOUNDS = slice(-4.5, 4.5)
NINO34_LON_BOUNDS = slice(190.0, 240.0)

# RONI parameters and customizable area-weighting toggle
ZONAL_LON_BOUNDS = slice(0.5, 359.5)
RONI_BASE_START_YR = 1950
RONI_BASE_END_YR = 2020
APPLY_INDEX_AREA_WEIGHTING = False
