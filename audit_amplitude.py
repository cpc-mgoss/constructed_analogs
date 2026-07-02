# audit_amplitude.py
import config
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg as la
from core.analog_engine import calculate_eof_modes

def audit_ca_amplitude():
    print("====================================================")
    print("SURGICAL AUDIT: Tracking Amplitude Loss in CA Engine")
    print("====================================================")
    
    # 1. Load the compiled datasets
    anom_ds = xr.open_dataset(config.OUT_DATA_DIR / "ersst.seasonal.1948-curr.1x1.nc")["sst"]
    ic_ds = xr.open_dataset(config.OUT_DATA_DIR / "ersst.seasonal.msic.ic.nc")["sst"]
    
    # Isolate IC Tier 4 (Most recent month)
    ic4 = ic_ds.sel(ic_num=4)
    target_date = pd.Timestamp(ic4.time.values)
    print(f"Target IC Date: {target_date.strftime('%Y-%m')}")
    
    # Define Nino3.4 spatial mean helper
    def get_nino34(da):
        return da.sel(lat=config.NINO34_LAT_BOUNDS, lon=config.NINO34_LON_BOUNDS).mean(dim=["lat", "lon"]).values

    # Step 1: Baseline Truth
    raw_nino = get_nino34(ic4)
    print(f"\n1. BASELINE GROUND TRUTH")
    print(f"  -> Raw IC Tier 4 Nino 3.4 Anomaly: {float(raw_nino):.3f} K")
    
    # Step 2: EOF Truncation Check
    master_eof_ds = calculate_eof_modes(anom_ds, max_modes=40)
    eofs = master_eof_ds["eofs"]
    pcs = master_eof_ds["pcs"]
    
    ocean_mask = anom_ds.sel(lat=config.LAT_BOUNDS).dropna(dim="time", how="all").notnull().all(dim="time")
    ic_clipped = ic4.sel(lat=config.LAT_BOUNDS)
    cos_lat_weights = np.cos(np.radians(ic_clipped.lat))
    
    # Project IC onto EOFs
    numerator = (eofs * ic_clipped * cos_lat_weights).where(ocean_mask).sum(dim=["lat", "lon"], skipna=True)
    denominator = ((eofs ** 2) * cos_lat_weights).where(ocean_mask).sum(dim=["lat", "lon"], skipna=True)
    Y = (numerator / denominator).values  # Shape: (40,)
    
    # Reconstruct IC purely from the 40 EOF modes
    ic_reconstructed_from_eofs = (eofs * xr.DataArray(Y, dims=["mode"], coords=[eofs.mode])).sum(dim="mode")
    trunc_nino = get_nino34(ic_reconstructed_from_eofs)
    
    print(f"\n2. EOF TRUNCATION FIDELITY (40 Modes)")
    print(f"  -> Truncated IC Nino 3.4 Anomaly:  {float(trunc_nino):.3f} K")
    print(f"  -> Amplitude preserved through EOF: {(trunc_nino/raw_nino)*100:.1f}%")
    
    # Step 3: Solver Integrity Check
    clean_pcs = pcs.drop_sel(time=[target_date], errors="ignore")
    Z = clean_pcs.values  # Shape: (months, 40)
    
    # Current Code Math (nt x nt)
    A = np.dot(Z, Z.T)    
    B = np.dot(Z, Y)      
    avg_diag = np.mean(np.diag(A))
    
    # Run exact loop from codebase
    current_ridge = config.RIDGE_PENALTY
    for iteration in range(config.ADAPTIVE_RIDGE_MAX_ITER):
        A_ridge = A + np.eye(Z.shape[0]) * (avg_diag * current_ridge)
        raw_weights = la.solve(A_ridge, B, assume_a='sym')
        if np.all(np.sum(raw_weights ** 2, axis=0) <= config.ADAPTIVE_RIDGE_MAX_POWER) or current_ridge > config.ADAPTIVE_RIDGE_MAX_VALUE:
            break
        current_ridge += config.ADAPTIVE_RIDGE_STEP

    # How well do these weights reconstruct the target PC vector (Y)?
    # Constructed Y = Z^T * W
    Y_constructed = np.dot(Z.T, raw_weights)
    
    # Calculate vector magnitude ratio
    mag_Y = np.linalg.norm(Y)
    mag_Y_constructed = np.linalg.norm(Y_constructed)
    
    print(f"\n3. RIDGE SOLVER FIDELITY")
    print(f"  -> Target PC Vector Magnitude:      {mag_Y:.3f}")
    print(f"  -> Constructed PC Magnitude:        {mag_Y_constructed:.3f}")
    print(f"  -> Amplitude preserved by Solver:   {(mag_Y_constructed/mag_Y)*100:.1f}%")
    print(f"  -> Final Ridge Penalty Applied:     {current_ridge:.4f}")
    print(f"  -> Sum of calculated weights:       {np.sum(raw_weights):.4f}")
    
    # Step 4: Final Spatial Reconstruction
    weights_da = xr.DataArray(raw_weights, coords=[clean_pcs.time], dims=["time"])
    lead_anomalies = anom_ds.sel(lat=config.LAT_BOUNDS, time=clean_pcs.time.values)
    final_forecast_grid = lead_anomalies.dot(weights_da, dims="time")
    
    final_nino = get_nino34(final_forecast_grid)
    print(f"\n4. FINAL RECONSTRUCTION OUTPUT (Lead 0)")
    print(f"  -> Reconstructed Lead 0 Nino 3.4:   {float(final_nino):.3f} K")
    print(f"  -> Total Pipeline Amplitude Yield:  {(final_nino/raw_nino)*100:.1f}%")
    print("====================================================")

if __name__ == "__main__":
    audit_ca_amplitude()
