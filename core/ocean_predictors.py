# core/ocean_predictors.py
import numpy as np
import xarray as xr
import pandas as pd
import struct
import config

def run_ocean_predictor_pipeline(target_dataset, mode="seasonal"):
    """
    Ingests raw operational binaries, handles regridding, anomaly calculations,
    and structures initial condition snapshots using the Month-Start (Day 1) standard.
    """
    print(f"\n====================================================")
    print(f"[INGESTION]: Processing {target_dataset.upper()} ({mode.upper()} track)...")
    print(f"====================================================")

    base_run_date = pd.Timestamp(year=config.RUN_YR, month=config.RUN_MO, day=1)
    lag_months = config.RUN_MODES[mode]["lag_months"]
    apply_smoothing = config.RUN_MODES[mode]["apply_smoothing"]

    target_horizon = base_run_date - pd.DateOffset(months=1)
    print(f"Target data ingestion horizon: {target_horizon.strftime('%Y-%m')}")

    # 1. Parse Raw Binary Data Streams
    if target_dataset == "ersst":
        src = config.DATA_SOURCES["ersst"]
        available_da = _parse_sequential_binary(
            src["binary_path"], src["nx"], src["ny"], src["archive_start_date"],
            src["lat_edges"], src["lon_edges"], target_horizon, src["dtype"], src["is_kelvin"]
        )
        monthly_sst = available_da.interp(lat=np.arange(-89.5, 90.5, 1.0), lon=np.arange(0.5, 360.5, 1.0), method="linear")

    elif target_dataset == "hadoisst":
        hist_src = config.DATA_SOURCES["had_historical"]
        historical_da = _parse_sequential_binary(
            hist_src["binary_path"], hist_src["nx"], hist_src["ny"], hist_src["archive_start_date"],
            hist_src["lat_edges"], hist_src["lon_edges"], target_horizon, hist_src["dtype"], hist_src["is_kelvin"]
        )

        oper_da_list = []
        start_oper_date = pd.Timestamp("2010-04-01")
        oper_months = pd.date_range(start=start_oper_date, end=target_horizon, freq="MS")

        for m in oper_months:
            file_name = config.DATA_SOURCES["oi_operational"]["template"].format(year=m.year, month=m.month)
            f_path = config.DATA_SOURCES["oi_operational"]["dir_path"] / file_name
            src_oi = config.DATA_SOURCES["oi_operational"]
            m_da = _parse_single_month_binary(
                f_path, src_oi["nx"], src_oi["ny"], m,
                src_oi["lat_edges"], src_oi["lon_edges"],
                src_oi["dtype"], src_oi["is_kelvin"]
            )
            m_da = m_da.interp(lat=np.arange(-89.5, 89.5 + 1.0, 1.0), lon=np.arange(0.5, 360.5, 1.0), method="linear")
            oper_da_list.append(m_da)

        operational_da = xr.concat(oper_da_list, dim="time")
        monthly_sst = xr.concat([historical_da.sel(time=slice(None, "2010-03-01")), operational_da], dim="time")

    # 2. Handle Track-Specific Temporal Smoothing FIRST
    if apply_smoothing:
        print("Computing centered 3-month rolling averages...")
        processed_sst = monthly_sst.rolling(time=3, center=True).mean()
    else:
        processed_sst = monthly_sst

    # 3. Enforce the Standardized Base Climatology Period (1991-2020)
    clim_pool = processed_sst.sel(time=slice(f"{config.CLIM_START_YR}-01-01", f"{config.CLIM_END_YR}-12-31"))
    climatology = clim_pool.groupby("time.month").mean(dim="time")
    processed_anom = processed_sst.groupby("time.month") - climatology

    # --> NEW: Export Climatology for Text Tables
    out_clim_file = config.OUT_DATA_DIR / f"{target_dataset}.{mode}.climatology.nc"
    climatology.to_netcdf(out_clim_file)
    print(f"  -> Climatology saved to {out_clim_file.name}")

    out_anom_file = config.OUT_DATA_DIR / f"{target_dataset}.{mode}.1948-curr.1x1.nc"
    processed_anom.to_netcdf(out_anom_file)

    # 4. Extract Initial Conditions
    print("Extracting multi-scale Initial Condition snapshot coordinates...")
    ic_snapshots = []

    for tier_num, offset in zip(config.IC_NUMS, config.IC_OFFSETS):
        ic_target_date = base_run_date - pd.DateOffset(months=int(lag_months)) - pd.DateOffset(months=int(offset))
        print(f"  -> Compiling Tier [ic_num={tier_num}] Target Date: {ic_target_date.strftime('%Y-%m-%d')}")

        ic_slice = processed_anom.sel(time=ic_target_date)
        ic_slice = ic_slice.expand_dims(ic_num=[tier_num])
        ic_snapshots.append(ic_slice)

    ic_master_da = xr.concat(ic_snapshots, dim="ic_num")
    out_ic_file = config.OUT_DATA_DIR / f"{target_dataset}.{mode}.msic.ic.nc"
    ic_master_da.to_netcdf(out_ic_file)
    print(f"SUCCESS: {target_dataset.upper()} ({mode.upper()}) pipeline files written to disk.\n")


def _parse_sequential_binary(file_path, nx, ny, start_date_str, lat_edges, lon_edges, end_horizon, target_dtype, is_kelvin):
    """Parses binary files into Xarray objects using clean Month-Start frequencies."""
    bytes_per_record = nx * ny * 4
    file_size = file_path.stat().st_size
    total_months = file_size // bytes_per_record
    
    start_date = pd.Timestamp(start_date_str).replace(day=1)
    time_axis = pd.date_range(start=start_date, periods=total_months, freq="MS")
    
    lat_axis = np.linspace(lat_edges[0], lat_edges[1], ny)
    lon_axis = np.linspace(lon_edges[0], lon_edges[1], nx)
    
    with open(file_path, "rb") as f:
        raw_data = f.read()
        
    parsed_array = np.frombuffer(raw_data, dtype=target_dtype).reshape(total_months, ny, nx)
    
    # Apply Kelvin offset dynamically from config
    if is_kelvin:
        parsed_array = parsed_array - config.KELVIN_OFFSET
    
    invalid_mask = np.isin(parsed_array, config.UNDEF_FLAGS) | (parsed_array < config.VALID_SST_MIN) | (parsed_array > config.VALID_SST_MAX)
    clean_array = np.where(invalid_mask, np.nan, parsed_array)
    
    da = xr.DataArray(clean_array, coords=[time_axis, lat_axis, lon_axis], dims=["time", "lat", "lon"], name="sst")
    return da.sel(time=slice(None, end_horizon))

def _parse_single_month_binary(file_path, nx, ny, current_timestamp, lat_edges, lon_edges, target_dtype, is_kelvin):
    """Parses a single month binary file and forces a Month-Start coordinate stamp."""
    with open(file_path, "rb") as f:
        raw_data = f.read()
        
    parsed_grid = np.frombuffer(raw_data, dtype=target_dtype).reshape(ny, nx)
    
    # Apply Kelvin offset dynamically from config
    if is_kelvin:
        parsed_grid = parsed_grid - config.KELVIN_OFFSET
        
    invalid_mask = np.isin(parsed_grid, config.UNDEF_FLAGS) | (parsed_grid < config.VALID_SST_MIN) | (parsed_grid > config.VALID_SST_MAX)
    clean_grid = np.where(invalid_mask, np.nan, parsed_grid)
    
    lat_axis = np.linspace(lat_edges[0], lat_edges[1], ny)
    lon_axis = np.linspace(lon_edges[0], lon_edges[1], nx)
    
    clean_timestamp = pd.Timestamp(current_timestamp).replace(day=1)

    da = xr.DataArray(clean_grid, coords=[lat_axis, lon_axis], dims=["lat", "lon"], name="sst")
    return da.expand_dims(time=[clean_timestamp])
