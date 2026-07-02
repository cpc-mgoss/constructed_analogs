# core/plot_spatial.py
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
import config

def format_target_date(center_date, is_seasonal=True):
    """Generates proper operational date strings (e.g., MAM2026, DJF2026/2027 or MAY2026)"""
    if not is_seasonal:
        return center_date.strftime('%b%Y').upper()
    
    # Seasonal logic: uses the center date to find the 3-month block
    m1 = center_date - pd.DateOffset(months=1)
    m2 = center_date
    m3 = center_date + pd.DateOffset(months=1)
    
    # Extract first letters (e.g., M, A, M)
    season_str = m1.strftime('%b')[0] + m2.strftime('%b')[0] + m3.strftime('%b')[0]
    
    # Handle year crossing (e.g., Dec-Jan-Feb)
    if m1.year != m3.year:
        year_str = f"{m1.year}/{m3.year}"
    else:
        year_str = f"{m2.year}"
        
    return f"{season_str}{year_str}"

def generate_spatial_maps(run_mode="seasonal"):
    print("====================================================")
    print(f"GENERATING CA SPATIAL SST ANOMALY MAPS ({run_mode.upper()})")
    print("====================================================")
    
    is_seasonal = (run_mode == "seasonal")
    
    # 1. Load the Forecast NetCDFs
    ersst_files = [config.OUT_DATA_DIR / f"ersst.{run_mode}.forecast_modes_{m}.nc" for m in config.EVAL_EOF_MODES]
    had_files = [config.OUT_DATA_DIR / f"hadoisst.{run_mode}.forecast_modes_{m}.nc" for m in config.EVAL_EOF_MODES]
    
    if not ersst_files[0].exists() or not had_files[0].exists():
        print(f"[ERROR]: Forecast NetCDF files not found for {run_mode}. Run pipeline first.")
        return

    # 2. Combine and Average all 24 Members
    ds_ersst = xr.concat([xr.open_dataset(f) for f in ersst_files], dim="truncation")
    ds_had = xr.concat([xr.open_dataset(f) for f in had_files], dim="truncation")
    
    combined = xr.concat([ds_ersst, ds_had], dim="dataset")
    grand_mean = combined["reconstructed_sst"].mean(dim=["dataset", "truncation", "ic_num"])
    
    # 3. Setup Plotting Parameters
    levels = [-3, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 3]
    cmap = plt.get_cmap('RdBu_r')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    
    # Target date formatting setup
    lag = config.RUN_MODES[run_mode]["lag_months"]
    base_target_date = pd.Timestamp(year=config.RUN_YR, month=config.RUN_MO, day=1) - pd.DateOffset(months=int(lag))
    
    # For seasonal, the trailing month of the IC is 1 month ahead of the center date
    ic_thru_date = base_target_date + pd.DateOffset(months=1) if is_seasonal else base_target_date
    ic_label = ic_thru_date.strftime('%b%Y')

    # 4. Generate Maps for Each Lead Time
    for lead in grand_mean.lead.values:
        
        # Calculate operational lead label based on pipeline type
        op_lead = int(lead) - 3 if is_seasonal else int(lead) - 1
        
        print(f"  -> Plotting Map for Operational Lead {op_lead}...")
        
        target_date = base_target_date + pd.DateOffset(months=int(lead))
        target_label = format_target_date(target_date, is_seasonal)
        
        data_slice = grand_mean.sel(lead=lead)
        
        # Create Map
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=100)
        
        cf = ax.contourf(data_slice.lon, data_slice.lat, data_slice.values, 
                         levels=levels, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), extend='both')
        
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08, aspect=40, shrink=0.7)
        cbar.set_label('SST Anomaly (K)', fontsize=12)
        
        # Update title to use the operational lead
        title = f"CA SST Prd for {target_label}, ICs through {ic_label} (K), Lead {op_lead}"
        ax.set_title(title, fontsize=14, pad=15)
        
        # Footers
        fig.text(0.1, 0.02, "NOAA/NWS/NCEP/CPC (Modernized)", fontsize=10, color='blue', ha='left')
        fig.text(0.9, 0.02, config.CAbp if hasattr(config, 'CAbp') else "Base Period 1991-2020", fontsize=10, color='blue', ha='right')
        
        # Save Output using the operational lead
        out_dir = config.OUT_DIR_SEASONAL if is_seasonal else config.OUT_DIR_MONTHLY
        out_file = out_dir / f"caSST_anom.{op_lead}.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
 
    print("SUCCESS: All spatial maps generated.")
    print("====================================================")

if __name__ == "__main__":
    # Can be run manually for testing
    generate_spatial_maps("seasonal")
