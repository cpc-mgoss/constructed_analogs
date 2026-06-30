# utils.py
import numpy as np
import xarray as xr
import pandas as pd
import sys
import config

def read_grads_binary(file_path, nx, ny, start_date_str, lat_edges, lon_edges):
    """Reads a raw monolithic GrADS binary stream and returns a structured Xarray DataArray."""
    if not file_path.exists():
        print(f"\n[CRITICAL FILE MISSING]: {file_path}\n")
        sys.exit(1)
        
    flat_data = np.fromfile(file_path, dtype=np.float32)
    total_months = flat_data.size // (ny * nx)
    
    time_axis = pd.date_range(start=start_date_str, periods=total_months, freq="MS") + pd.Timedelta(days=14)
    lat_axis = np.linspace(lat_edges[0], lat_edges[1], ny)
    lon_axis = np.linspace(lon_edges[0], lon_edges[1], nx)
    
    da = xr.DataArray(
        flat_data.reshape((total_months, ny, nx)),
        coords=[time_axis, lat_axis, lon_axis],
        dims=["time", "lat", "lon"],
        name="sst"
    )
    
    for flag in config.UNDEF_FLAGS:
        da = da.where(da != flag)
    return da.where(da > -9.99e8)


def read_templated_grads_binary(dir_path, template, nx, ny, date_axis, lat_edges, lon_edges):
    """Loops over a target date index, ingesting separate monthly binary files into one unified cube."""
    nt = len(date_axis)
    compiled_data = np.zeros((nt, ny, nx), dtype=np.float32)
    
    for idx, dt in enumerate(date_axis):
        filename = template.format(year=dt.year, month=dt.month)
        file_path = dir_path / filename
        
        if not file_path.exists():
            print(f"\n" + "="*70)
            print(f"CRITICAL DATA AVAILABILITY ERROR")
            print("="*70)
            print(f"Target Monthly File Missing: {filename}")
            print(f"Directory Root: {dir_path}")
            print(f"Status: The upstream operational stream has not written this month's data yet.")
            print("="*70 + "\n")
            sys.exit(1)
            
        month_data = np.fromfile(file_path, dtype=np.float32)
        if month_data.size != (ny * nx):
            print(f"[DATA CORRUPTION ERROR]: File {filename} size does not match expected {ny}x{nx} grid dimensions.")
            sys.exit(1)
            
        compiled_data[idx, :, :] = month_data.reshape((ny, nx))
        
    # Clean missing value structures in bulk
    for flag in config.UNDEF_FLAGS:
        compiled_data = np.where(compiled_data == flag, np.nan, compiled_data)
    compiled_data = np.where(compiled_data <= -9.99e8, np.nan, compiled_data)
    
    lat_axis = np.linspace(lat_edges[0], lat_edges[1], ny)
    lon_axis = np.linspace(lon_edges[0], lon_edges[1], nx)
    
    return xr.DataArray(
        compiled_data,
        coords=[date_axis, lat_axis, lon_axis],
        dims=["time", "lat", "lon"],
        name="sst"
    )
