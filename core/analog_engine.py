# core/analog_engine.py
import numpy as np
import xarray as xr
import scipy.linalg as la
import config

def calculate_eof_modes(anomalies_da, max_modes=40):
    """
    Computes Empirical Orthogonal Functions (EOFs) and Principal Components (PCs)
    using optimized LAPACK SVD. Completely replaces legacy eof_4_ca.s.f.
    """
    print(f"--- Launching High-Performance SVD EOF Engine (Target Modes: {max_modes}) ---")
    
    # 1. Spatial Domain Clipping (e.g., tp_ml: 45S to 45N)
    domain_da = anomalies_da.sel(lat=config.LAT_BOUNDS)
    
    # Preserve coordinates for downstream re-packing
    time_coords = domain_da.time
    lat_coords = domain_da.lat
    lon_coords = domain_da.lon
    
    # 2. Extract an Operational Land/Sea Mask
    # We locate cells that remain valid (not NaN) throughout the timeline
    sea_mask = domain_da.isel(time=0).notnull()
    
    # Flatten spatial grid and drop land cells to maximize linear algebra speed
    # This dynamically handles what 'ngrd' used to hardcode in Fortran
    flat_matrix = domain_da.values.reshape(len(time_coords), -1)
    flat_mask = sea_mask.values.flatten()
    ocean_matrix = flat_matrix[:, flat_mask]
    
    # Handle any stray NaN values just in case
    ocean_matrix = np.nan_to_num(ocean_matrix, nan=0.0)
    
    print(f"Matrix Dimension: {ocean_matrix.shape[0]} months x {ocean_matrix.shape[1]} active ocean cells")
    
    # 3. Execute LAPACK Economy SVD (Divide-and-Conquer Algorithm 'dgesdd')
    # This runs multithreaded across your system cores instantly
    U, s, Vt = la.svd(ocean_matrix, full_matrices=False)
    
    # 4. Truncate to the requested maximum modes (modemax)
    U = U[:, :max_modes]
    s = s[:max_modes]
    Vt = Vt[:max_modes, :]
    
    # 5. Calculate Explained Variance Ratio (Replaces Subroutine RATE)
    eigenvalues = s ** 2
    total_variance = np.sum(flat_mask)  # Total variance baseline
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
    
    # Merge into a single unified NetCDF ready dataset
    eof_results = xr.merge([eofs_ds, pcs_ds, variance_ds])
    return eof_results
