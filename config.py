import os
from pathlib import Path

# =====================================================================
# 1. CORE RUN TARGETS (The only things you change each month)
# =====================================================================
RUN_YR = 2026
RUN_MO = 6  # June

# =====================================================================
# 2. DYNAMIC DATE MATH (Calculated automatically on import!)
# =====================================================================
# Replaces those massive blocks of legacy "if curmo = 06; then..."
MONTH_NAMES_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAMES_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Calculate your initial condition target offsets natively
IC_MONTH_NUM = 12 if RUN_MO == 1 else RUN_MO - 1
IC_YEAR = RUN_YR - 1 if IC_MONTH_NUM == 12 else RUN_YR
IC_MONTH_STR = f"{IC_MONTH_NUM:02d}"
IC_MONTH_NAME_SHORT = MONTH_NAMES_SHORT[IC_MONTH_NUM - 1]

# =====================================================================
# 3. STANDARDIZED PATHS (Using Python's modern Pathlib module)
# =====================================================================
BASE_DIR = Path("/cpc/home/mgoss")
DATA_DIR = BASE_DIR / "data/ca_prd"
LOG_DIR = BASE_DIR / "project/ca_prd/sst/opr/0Logs"

# Target output directory dynamically injects your run dates
TARGET_OUT_DIR = BASE_DIR / f"data/season_fcst/ca/{RUN_YR}/{IC_MONTH_STR}"

# Metadata strings
SIGNATURE = "Michael Goss NOAA/NWS/NCEP/CPC"
BASE_PERIOD = "Base Period 1991-2020"
