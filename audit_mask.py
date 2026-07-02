# audit_mask.py
import xarray as xr
import numpy as np
import config

def audit_matrix_nans():
    print("====================================================")
    print("SURGICAL AUDIT: Inspecting NaN Distribution in Anomaly Grid")
    print("====================================================")
    
    anom_file = config.OUT_DATA_DIR / "ersst.seasonal.1948-curr.1x1.nc"
    da = xr.open_dataset(anom_file)["sst"].sel(lat=config.LAT_BOUNDS)
    
    print(f"Raw Input Shape (Time, Lat, Lon): {da.shape}")
    
    # 1. Check for completely blank time steps
    valid_cells_per_month = da.notnull().sum(dim=["lat", "lon"]).values
    min_cells = valid_cells_per_month.min()
    max_cells = valid_cells_per_month.max()
    
    print(f"\n--- TEMPORAL STABILITY ---")
    print(f"Max valid cells in a single month: {max_cells}")
    print(f"Min valid cells in a single month: {min_cells}")
    
    if min_cells == 0:
        blank_months = da.time.values[valid_cells_per_month == 0]
        print(f"CRITICAL: Found {len(blank_months)} completely empty months!")
        print(f"Empty months: {blank_months[:5]} ...")
    
    # 2. Check Spatial Persistence
    print(f"\n--- SPATIAL PERSISTENCE ---")
    mask_all = da.notnull().all(dim="time")
    mask_any = da.notnull().any(dim="time")
    
    print(f"Cells with 100% valid data (0 NaNs across all time): {int(mask_all.sum())}")
    print(f"Cells with AT LEAST ONE valid data point: {int(mask_any.sum())}")
    
    # 3. Where are the missing values?
    total_months = da.shape[0]
    valid_counts = da.notnull().sum(dim="time").values.flatten()
    valid_counts = valid_counts[valid_counts > 0]  # Look only at ocean cells
    
    perfect_cells = (valid_counts == total_months).sum()
    missing_1_to_5 = ((valid_counts >= total_months - 5) & (valid_counts < total_months)).sum()
    missing_more = (valid_counts < total_months - 5).sum()
    
    print(f"\n--- MISSING DATA BREAKDOWN ---")
    print(f"Ocean Cells with Perfect Attendance: {perfect_cells}")
    print(f"Ocean Cells missing 1 to 5 months:   {missing_1_to_5}")
    print(f"Ocean Cells missing > 5 months:      {missing_more}")
    print("====================================================")

if __name__ == "__main__":
    audit_matrix_nans()
