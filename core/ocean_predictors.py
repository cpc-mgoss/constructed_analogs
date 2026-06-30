# core/ocean_predictors.py
import numpy as np
import xarray as xr
import pandas as pd
import sys
import config
from utils import read_grads_binary, read_templated_grads_binary

def load_and_regrid_sst(dataset_type, mode):
    """Ingests central repository tracks and conforms them to the standard evaluation grid."""
    mode_settings = config.RUN_MODES[mode]
    
    run_date = pd.to_datetime(f"{config.RUN_YR}-{config.RUN_MO:02d}-15")
    target_date = run_date - pd.DateOffset(months=mode_settings["lag_months"])
    
    target_lat = np.linspace(config.TARGET_LAT_EDGES[0], config.TARGET_LAT_EDGES[1], config.TARGET_NY)
    target_lon = np.linspace(config.TARGET_LON_EDGES[0], config.TARGET_LON_EDGES[1], config.TARGET_NX)
    
    if dataset_type == "ersst":
        meta = config.DATA_SOURCES["ersst"]
        print(f"[INGESTION]: Parsing ERSST master file through target horizon: {target_date.strftime('%Y-%m')}")
        
        full_da = read_grads_binary(
            file_path=meta["binary_path"], nx=meta["nx"], ny=meta["ny"],
            start_date_str=meta["archive_start_date"], lat_edges=meta["lat_edges"], lon_edges=meta["lon_edges"]
        )
        
        if target_date not in full_da.time:
            print(f"[DATA ERROR]: ERSST archive does not contain records reaching {target_date.strftime('%Y-%m')} yet.")
            sys.exit(1)
            
        sliced_da = full_da.sel(time=slice("1948-02-15", target_date))
        print("Vectorized Regridding: Linearly interpolating coarse ERSST matrix to 1x1 grid...")
        return sliced_da.interp(lat=target_lat, lon=target_lon, method="linear")
        
    else:
        print(f"[INGESTION]: Executing HadISST/OISST split merge engine through target horizon: {target_date.strftime('%Y-%m')}")
        had_meta = config.DATA_SOURCES["had_historical"]
        oi_meta = config.DATA_SOURCES["oi_operational"]
        
        # 1. HadISST Chunk: Already natively maps to the standard 360x180 target coordinates
        had_full = read_grads_binary(
            file_path=had_meta["binary_path"], nx=had_meta["nx"], ny=had_meta["ny"],
            start_date_str=had_meta["archive_start_date"], lat_edges=had_meta["lat_edges"], lon_edges=had_meta["lon_edges"]
        )
        had_chunk = had_full.sel(time=slice("1948-01-01", "1981-08-31"))
        
        # 2. OISST Chunk: Ingest separate monthly files at their native 360x181 grid spacing
        oi_dates = pd.date_range(start="1981-09-01", end=target_date, freq="MS") + pd.Timedelta(days=14)
        print(f"[STITCHING]: Assembling {len(oi_dates)} templated OISST monthly files ({oi_meta['ny']}x{oi_meta['nx']}) into memory...")
        
        oi_raw = read_templated_grads_binary(
            dir_path=oi_meta["dir_path"], template=oi_meta["template"],
            nx=oi_meta["nx"], ny=oi_meta["ny"], date_axis=oi_dates,
            lat_edges=oi_meta["lat_edges"], lon_edges=oi_meta["lon_edges"]
        )
        
        # Replicate legacy math: add 273.16 constant to the data stream
        oi_raw = oi_raw + 273.16
        
        # Regrid the OISST component to conform perfectly to the standard target matrix axes
        print("Vectorized Alignment: Interpolating OISST grid from 181 to 180 latitude coordinates...")
        oi_chunk = oi_raw.interp(lat=target_lat, lon=target_lon, method="linear")
        
        # Join the aligned data structures seamlessly across the timeline axis
        return xr.concat([had_chunk, oi_chunk], dim="time")

def process_anomalies(sst_da):
    """Generates departures against the designated climate baseline window."""
    base_window = sst_da.sel(time=slice(f"{config.CLIM_START_YR}-01-01", f"{config.CLIM_END_YR}-12-31"))
    climatology = base_window.groupby("time.month").mean("time")
    return (sst_da.groupby("time.month") - climatology).drop_vars("month")

def run_ocean_predictor_pipeline(target_dataset, mode):
    """Coordinates data extraction and generates sandboxed netCDF files."""
    dtype = target_dataset
    
    sst_standardized = load_and_regrid_sst(dtype, mode)
    anomalies = process_anomalies(sst_standardized)
    
    if config.RUN_MODES[mode]["apply_smoothing"]:
        print("Computing centered 3-month rolling averages...")
        processed_fields = anomalies.rolling(time=3, center=True, min_periods=3).mean()
    else:
        print("Preserving standard monthly steps (no rolling smoothing applied)...")
        processed_fields = anomalies
    
    print("Extracting 4-tier Initial Condition snapshot coordinates...")
    ic_snapshots = [processed_fields.isel(time=-1 - offset) for offset in config.IC_OFFSETS]
    ic_dataset = xr.concat(ic_snapshots[::-1], dim="ic_num").assign_coords(ic_num=config.IC_NUMS)
    
    master_out = config.OUT_DATA_DIR / f"{dtype}.{mode}.1948-curr.1x1.nc"
    ic_out = config.OUT_DATA_DIR / f"{dtype}.{mode}.msic.ic.nc"
    
    processed_fields.to_netcdf(master_out)
    ic_dataset.to_netcdf(ic_out)
    print(f"SUCCESS: {dtype.upper()} ({mode.upper()}) NetCDF files written out to development workspace.")
