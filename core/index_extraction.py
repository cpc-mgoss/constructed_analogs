# core/index_extraction.py
import xarray as xr
import numpy as np
import pandas as pd
import config

def extract_forecast_indices(forecast_da, anom_ds, ic_ds, dataset_name, run_mode):
    """
    Extracts Niño 3.4 and RONI indices from the forecast matrices.
    Calculates dynamic RONI scalars based on the historical anomaly base period (1950-2020).
    Extracts the baseline climatology for text table Total SST reconstruction.
    """
    # 1. Spatial bounds (Matching Legacy Fortran 1x1 indices: js=86,je=95 / is=190,ie=240)
    # On a -89.5 to 89.5 / 0.5 to 359.5 grid:
    # Lat index 85 to 94 = -4.5 to 4.5
    # Lon index 189 to 239 = 189.5 to 239.5
    lat_slice = slice(-4.5, 4.5)
    lon_slice = slice(189.5, 239.5)

    # 2. Reconstruct Target Months to map to Climatology/Scalars
    lag = config.RUN_MODES[run_mode]["lag_months"]
    base_target_date = pd.Timestamp(year=config.RUN_YR, month=config.RUN_MO, day=1) - pd.DateOffset(months=int(lag))
    
    leads = forecast_da.lead.values
    target_dates = [base_target_date + pd.DateOffset(months=int(l)) for l in leads]
    target_months = [d.month for d in target_dates]

    # 3. Load Climatology and extract Niño 3.4 Baseline Temperatures
    clim_file = config.OUT_DATA_DIR / f"{dataset_name}.{run_mode}.climatology.nc"
    if clim_file.exists():
        clim_ds = xr.open_dataset(clim_file)
        # Pull the primary variable, squeezing out any extraneous dims
        clim_da = clim_ds[list(clim_ds.data_vars)[0]].squeeze() 
        
        # Spatial mean over the Niño 3.4 box
        clim_nino34 = clim_da.sel(lat=lat_slice, lon=lon_slice).mean(dim=["lat", "lon"])
        
        # Extract only the 17 specific months we need for this forecast track
        clim_vals = [clim_nino34.sel(month=m).values.item() for m in target_months]
    else:
        print(f"  [WARNING] Climatology file not found: {clim_file}. Using 0.0 for Total SST.")
        clim_vals = np.zeros(len(leads))

    clim_da_aligned = xr.DataArray(clim_vals, coords=[leads], dims=["lead"])

    # 4. Calculate RONI Base Period Variance Scalars (1950-2020)
    # (Matches your `get_base_period_scalars` logic exactly)
    base_anom = anom_ds.sel(time=slice("1950-01-01", "2020-12-31"))
    
    hist_nino = base_anom.sel(lat=lat_slice, lon=lon_slice).mean(dim=["lat", "lon"])
    hist_zonal = base_anom.sel(lat=lat_slice).mean(dim=["lat", "lon"])
    hist_roni = hist_nino - hist_zonal

    # Standard Deviations grouped by calendar month
    nino_std = hist_nino.groupby("time.month").std(dim="time")
    roni_std = hist_roni.groupby("time.month").std(dim="time")

    # Safe division (fallback to 1.0 if RONI std is zero)
    scalars = xr.where(roni_std > 1e-6, nino_std / roni_std, 1.0)
    
    # Align the 12 calendar month scalars to our 17 forecast leads
    scalar_vals = [scalars.sel(month=m).values.item() for m in target_months]
    scalar_da_aligned = xr.DataArray(scalar_vals, coords=[leads], dims=["lead"])

    # 5. Extract Forecast Anomalies
    fcst_nino = forecast_da.sel(lat=lat_slice, lon=lon_slice).mean(dim=["lat", "lon"])
    fcst_zonal = forecast_da.sel(lat=lat_slice).mean(dim=["lat", "lon"])
    fcst_roni_raw = fcst_nino - fcst_zonal

    # Apply the variance adjustment scalar
    fcst_roni_scaled = fcst_roni_raw * scalar_da_aligned

    # 6. Package Output
    indices_ds = xr.Dataset(
        {
            "nino34_anomaly": fcst_nino,
            "roni_anomaly": fcst_roni_scaled,
            "nino34_climatology": clim_da_aligned
        }
    )

    return indices_ds
