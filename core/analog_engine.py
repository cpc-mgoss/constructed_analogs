# core/analog_engine.py
import numpy as np
import xarray as xr
import scipy.linalg as la
import config

def calculate_eof_modes(anomalies_da, max_modes=40):
    """
    Computes Empirical Orthogonal Functions (EOFs) and Principal Components (PCs)
    using optimized LAPACK SVD. Accounts for rolling window edge truncation.
    """
    print(f"--- Launching High-Performance SVD EOF Engine (Target Modes: {max_modes}) ---")
    
    # 1. Spatial Domain Clipping (e.g., tp_ml: 45S to 45N)
    domain_da = anomalies_da.sel(lat=config.LAT_BOUNDS)
    
    # GUARDRAIL: Drop time steps that are completely NaN due to rolling window edge effects
    # This cleans out the untruncated boundary months seamlessly
    domain_da = domain_da.dropna(dim="time", how="all")
    
    # Preserve coordinates for downstream re-packing
    time_coords = domain_da.time
    lat_coords = domain_da.lat
    lon_coords = domain_da.lon
    
    # 2. Extract an Operational Land/Sea Mask
    # Identify cells that contain valid data across ALL remaining operational timesteps
    sea_mask = domain_da.notnull().all(dim="time")
    
    # Flatten spatial grid and drop land cells to maximize linear algebra speed
    flat_matrix = domain_da.values.reshape(len(time_coords), -1)
    flat_mask = sea_mask.values.flatten()
    ocean_matrix = flat_matrix[:, flat_mask]
    
    print(f"Matrix Dimension: {ocean_matrix.shape[0]} months x {ocean_matrix.shape[1]} active ocean cells")
    
    if ocean_matrix.shape[1] == 0:
        raise ValueError("CRITICAL: Zero active ocean cells found. Check input fields or coordinate bounds.")
    
    # 3. Execute LAPACK Economy SVD (Divide-and-Conquer Algorithm)
    U, s, Vt = la.svd(ocean_matrix, full_matrices=False)
    
    # 4. Truncate to the requested maximum modes
    U = U[:, :max_modes]
    s = s[:max_modes]
    Vt = Vt[:max_modes, :]
    
    # 5. Calculate Explained Variance Ratio
    eigenvalues = s ** 2
    explained_variance_ratio = eigenvalues / np.sum(ocean_matrix ** 2)
    
    print(f"Top 3 Modes Variance Contribution: {explained_variance_ratio[:3] * 100}%")
    
    # 6. Reconstruct Spatial EOF Maps (Map 1D ocean vectors back to 2D latitude/longitude)
    eof_spatial_vectors = np.zeros((max_modes, len(lat_coords) * len(lon_coords)))
    eof_spatial_vectors[:, flat_mask] = Vt
    eof_spatial_maps = eof_spatial_vectors.reshape(max_modes, len(lat_coords), len(lon_coords))
    
    # 7. Package Outputs into Structured Xarray Containers
    mode_axis = np.arange(1, max_modes + 1)
    
    eofs_ds = xr.DataArray(
        eof_spatial_maps,
        coords=[mode_axis, lat_coords, lon_coords],
        dims=["mode", "lat", "lon"],
        name="eofs"
    )
    
    pcs_ds = xr.DataArray(
        U * s,  # Scale left singular vectors by singular values to get amplitude PCs
        coords=[time_coords, mode_axis],
        dims=["time", "mode"],
        name="pcs"
    )
    
    variance_ds = xr.DataArray(
        explained_variance_ratio,
        coords=[mode_axis],
        dims=["mode"],
        name="variance_fraction"
    )
    
    return xr.merge([eofs_ds, pcs_ds, variance_ds])
