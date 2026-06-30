# config.py
from pathlib import Path

# =====================================================================
# 1. RUN REGIME
# =====================================================================
RUN_YR = 2026
RUN_MO = 6
DATASET_TYPES = ["ersst", "hadoisst"]

# =====================================================================
# 2. SANDBOXED OUTPUT PATHS
# =====================================================================
BASE_DIR = Path("/cpc/home/mgoss/project/ca_dev")
OUT_DATA_DIR = BASE_DIR / "data"
TMP_DIR = BASE_DIR / "tmp"

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 3. ABSOLUTE SYSTEM DATA INPUT SOURCES
# =====================================================================
DATA_SOURCES = {
    "ersst": {
        "binary_path": Path("/cpc/home/wd52ml/enso/oni.v5/data/ersst.v5/ersst.v5.1854.pres.gr"),
        "nx": 180, "ny": 89,
        "archive_start_date": "1854-01-15",
        "lat_edges": (-89.0, 87.0), "lon_edges": (0.0, 358.0),
    },
    "had_historical": {
        "binary_path": Path("/cpc/GODAS/bjha/MERGEDSST/ncar.SST.HAD187001-198110.OI198111-201003.gr"),
        "nx": 360, "ny": 180,
        "archive_start_date": "1870-01-15",
        "lat_edges": (-89.5, 89.5), "lon_edges": (0.5, 359.5),
    },
    "oi_operational": {
        "dir_path": Path("/cpc/GODAS/OISSTv2.1/DATA/month"),
        "template": "mnth.OISSTv2.1_1x1.{year:04d}{month:02d}.gr",
        "nx": 360, 
        "ny": 181,                          # Fixed: 181 lat lines
        "lat_edges": (-90.0, 90.0),         # Fixed: Runs from -90 to 90
        "lon_edges": (0.0, 359.0),          # Fixed: Runs from 0 to 359
    }
}

# =====================================================================
# 4. CLIMATE & EVALUATION GRID STANDARDS
# =====================================================================
CLIM_START_YR, CLIM_END_YR = 1991, 2020
EOF_RANGE = "tp_ml"
LAT_BOUNDS = slice(-45.5, 45.5)

IC_OFFSETS = [0, 3, 6, 9]
IC_NUMS = [4, 3, 2, 1]

TARGET_NX, TARGET_NY = 360, 180
TARGET_LAT_EDGES = (-89.5, 89.5)
TARGET_LON_EDGES = (0.5, 359.5)
UNDEF_FLAGS = [-9.99e8, -999.0, -999000000.0]

RUN_MODES = {
    "monthly": {"lag_months": 1, "apply_smoothing": False},
    "seasonal": {"lag_months": 2, "apply_smoothing": True}
}
