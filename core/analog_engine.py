# core/analog_engine.py
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg as la
import config
import warnings
import json

def calculate_eof_modes(anomalies_da, max_modes=40):
    """Computes universal historical EOFs using cos(lat) weighting."""
    print(f"--- Launching High-Performance SVD EOF Engine (Target Modes: {max_modes}) ---")

    # 1. Strip the blank temporal boundaries caused by the rolling mean
    trimmed_anom = anomalies_da.sel(lat=config.LAT_BOUNDS).dropna(dim="time", how="all")

    # 2. Define the true ocean mask on the clean timeline
    ocean_mask = trimmed_anom.notnull().all(dim="time")

    # 3. Apply mask and weights
    active_anom = trimmed_anom.where(ocean_mask, drop=True)
    cos_lat_weights = np.cos(np.radians(active_anom.lat))
    weighted_anom = active_anom * np.sqrt(cos_lat_weights)

    # 4. Stack safely into 1D (all remaining cells now have 100% attendance)
    stacked_anom = weighted_anom.stack(spatial=["lat", "lon"]).dropna(dim="spatial")

    print(f"Matrix Dimension: {stacked_anom.shape[0]} months x {stacked_anom.shape[1]} active ocean cells")

    U, S, Vt = la.svd(stacked_anom.values, full_matrices=False)

    variance_explained = (S ** 2) / np.sum(S ** 2) * 100
    print(f"Top 3 Modes Variance Contribution: {variance_explained[:3]}%")

    pcs = U[:, :max_modes] * S[:max_modes]
    eofs_flat = Vt[:max_modes, :]

    pcs_da = xr.DataArray(pcs, coords=[stacked_anom.time, np.arange(1, max_modes + 1)], dims=["time", "mode"])

    eofs_da = xr.DataArray(
        eofs_flat,
        coords=[np.arange(1, max_modes + 1), stacked_anom.spatial],
        dims=["mode", "spatial"]
    ).unstack("spatial").reindex_like(ocean_mask)

    eofs_da = eofs_da / np.sqrt(cos_lat_weights)
    return xr.Dataset({"eofs": eofs_da, "pcs": pcs_da})


def _project_grid(grid_da, eofs_da):
    """Projects a single 2D spatial grid onto the pre-computed EOF modes."""
    cos_lat = np.cos(np.radians(grid_da.lat))
    numerator = (eofs_da * grid_da * cos_lat).sum(dim=["lat", "lon"], skipna=True)
    denominator = ((eofs_da ** 2) * cos_lat).sum(dim=["lat", "lon"], skipna=True)
    return numerator / denominator


def generate_constructed_analog_forecast(anomalies_da, master_eof_ds, ic_da, target_modes, run_mode, dataset_name):
    """
    Constructs distinct MSIC ensemble members by stacking target IC Principal Components
    into progressively deeper trajectory matrices and solving for the optimal historical weights.
    """
    print(f"\n--- Running MSIC Trajectory Forecast Assembly for Truncation Level: {target_modes} Modes ---")

    eofs = master_eof_ds["eofs"].sel(mode=slice(1, target_modes))
    base_pcs = master_eof_ds["pcs"].sel(mode=slice(1, target_modes))

    ensemble_forecasts = []
    weights_storage = [] 
    ac_storage = []
    rms_storage = []

    # Extract the target calendar month from the IC
    target_time = pd.Timestamp(ic_da.sel(ic_num=config.IC_NUMS[0]).time.values)
    target_month = target_time.month

    # Iterate over the trajectory depths dynamically based on config definitions
    for msic in range(1, len(config.IC_NUMS) + 1):
        print(f"  -> Building Trajectory Vector for MSIC={msic} (Depth: {msic} ICs, State Vector Length: {target_modes * msic})")

        # 1. Determine Valid Dates
        valid_dates = pd.DatetimeIndex(base_pcs.time.values)
        valid_dates = valid_dates[valid_dates.month == target_month]

        for i in range(msic):
            req_hist = valid_dates - pd.DateOffset(months=config.IC_OFFSETS[i])
            valid_dates = valid_dates[np.isin(req_hist, base_pcs.time.values)]

        req_leads = valid_dates + pd.DateOffset(months=config.FORECAST_LEAD_END - 1)
        valid_dates = valid_dates[np.isin(req_leads, anomalies_da.time.values)]

        print(f"     Historical library restricted to {len(valid_dates)} matching seasons.")

        if len(valid_dates) == 0:
            raise ValueError(f"No valid historical dates found for MSIC={msic}.")

        # 2. Stack the State Vectors
        Z_blocks = []
        Y_blocks = []

        for i in range(msic):
            offset = config.IC_OFFSETS[i]
            ic_val = config.IC_NUMS[i]

            # Project the Target IC
            ic_grid = ic_da.sel(ic_num=ic_val).sel(lat=config.LAT_BOUNDS)
            y_pc = _project_grid(ic_grid, eofs).values
            Y_blocks.append(y_pc)

            # Extract the corresponding shifted historical PCs
            hist_dates = valid_dates - pd.DateOffset(months=offset)
            z_pc = base_pcs.sel(time=hist_dates).values
            Z_blocks.append(z_pc)

        Y_stacked = np.concatenate(Y_blocks)              # Shape: (modes * msic,)
        Z_stacked = np.concatenate(Z_blocks, axis=1)      # Shape: (time, modes * msic)

        # 3. Solve for Analog Weights using Adaptive Ridge Regression
        A = np.dot(Z_stacked, Z_stacked.T)
        B = np.dot(Z_stacked, Y_stacked)
        avg_diag = np.mean(np.diag(A))
        nt = Z_stacked.shape[0]

        current_ridge = config.RIDGE_PENALTY
        for iteration in range(config.ADAPTIVE_RIDGE_MAX_ITER):
            A_ridge = A + np.eye(nt) * (avg_diag * current_ridge)
            raw_weights = la.solve(A_ridge, B, assume_a='sym')

            # "Weinalys" power check: Matrix is stable if sum(weights^2) <= 0.5
            if np.sum(raw_weights ** 2) <= config.ADAPTIVE_RIDGE_MAX_POWER or current_ridge > config.ADAPTIVE_RIDGE_MAX_VALUE:
                break
            current_ridge += config.ADAPTIVE_RIDGE_STEP

        print(f"     Convergence achieved at Ridge: {current_ridge:.4f} (Sum of weights^2: {np.sum(raw_weights**2):.3f})")

        # Create a flat 1D array for the matrix math
        weights_1d = xr.DataArray(raw_weights, coords=[valid_dates], dims=["time"])
        weights_storage.append(weights_1d.expand_dims(ic_num=[msic]))

        # 4. Final Forward Forecast Projection
        lead_forecasts = []
        for lead in range(config.FORECAST_LEAD_START, config.FORECAST_LEAD_END):
            target_hist_dates = valid_dates + pd.DateOffset(months=lead)
            lead_anoms = anomalies_da.sel(time=target_hist_dates)
            lead_anoms = lead_anoms.assign_coords(time=valid_dates)

            fcst = lead_anoms.dot(weights_1d, dims="time")
            fcst = fcst.expand_dims(lead=[lead])
            lead_forecasts.append(fcst)

        msic_fcst = xr.concat(lead_forecasts, dim="lead")
        msic_fcst = msic_fcst.expand_dims(ic_num=[msic])
        ensemble_forecasts.append(msic_fcst)

        # 5. Compute Hindcast Skill for this MSIC depth
        print(f"     -> Fetching/Updating Hindcasts for {config.SKILL_VERIF_START_YR}-{config.RUN_YR-1}...")
        
        # Get the cached or newly minted hindcast/obs pairs
        hindcast_ds = get_or_build_hindcasts(
            anomalies_da=anomalies_da,
            Z_stacked=Z_stacked,
            valid_dates=valid_dates,
            ridge_alpha=current_ridge,
            dataset_name=dataset_name,
            target_month=target_month,
            run_mode=run_mode,
            msic=msic,
            target_modes=target_modes
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # The modern way: we calculate AC/RMS directly against the output hindcasts here.
            # If you want to replicate legacy exactly, you compute it per MSIC:
            ac_da = xr.corr(hindcast_ds["hindcast"], hindcast_ds["observation"], dim="verif_time")
            rms_da = np.sqrt(((hindcast_ds["hindcast"] - hindcast_ds["observation"])**2).mean(dim="verif_time"))

        ac_storage.append(ac_da.expand_dims(ic_num=[msic]))
        rms_storage.append(rms_da.expand_dims(ic_num=[msic]))

    # Merge everything for output
    forecast_da = xr.concat(ensemble_forecasts, dim="ic_num")
    forecast_da.name = "reconstructed_sst"

    weights_out = xr.concat(weights_storage, dim="ic_num")
    weights_out.name = "analog_weights"

    ac_out = xr.concat(ac_storage, dim="ic_num")
    ac_out.name = "ac"

    rms_out = xr.concat(rms_storage, dim="ic_num")
    rms_out.name = "rms"

    return xr.merge([forecast_da, weights_out, ac_out, rms_out])


def _get_physics_state_string():
    """Compiles a JSON string of all config parameters that alter the CA physics."""
    critical_state = {
        "lat_bounds": str(config.LAT_BOUNDS),
        "target_modes": config.EVAL_EOF_MODES,
        "ic_offsets": config.IC_OFFSETS,
        "ic_nums": config.IC_NUMS,
        "ridge_penalty": config.RIDGE_PENALTY,
        "ridge_step": config.ADAPTIVE_RIDGE_STEP,
        "exclusion_radius": config.ANALOG_EXCLUSION_RADIUS
    }
    return json.dumps(critical_state, sort_keys=True)

def get_or_build_hindcasts(anomalies_da, Z_stacked, valid_dates, ridge_alpha, dataset_name, target_month, run_mode, msic, target_modes):
    """
    State-aware caching engine for historical hindcasts. 
    Checks config integrity and only calculates missing years.
    """
    current_state_str = _get_physics_state_string()
    var_name = anomalies_da.name if anomalies_da.name else "unknown_var"
    cache_file = config.CACHE_DIR / f"hindcasts_{var_name}_{dataset_name}_{run_mode}_month_{target_month:02d}_msic_{msic}_modes_{target_modes}.nc"
 
    verif_years = list(range(config.SKILL_VERIF_START_YR, config.RUN_YR))
    leads = range(config.FORECAST_LEAD_START, config.FORECAST_LEAD_END)
    
    cached_ds = None
    missing_years = verif_years.copy()

    # 1. Check for valid cache
    if cache_file.exists():
        try:
            cached_ds = xr.open_dataset(cache_file).load()
            saved_state = cached_ds.attrs.get("config_state", "")
            
            if saved_state == current_state_str:
                existing_years = cached_ds.verif_time.dt.year.values
                missing_years = [y for y in verif_years if y not in existing_years]
                print(f"     [CACHE HIT]: Valid config detected. {len(missing_years)} missing years to append.")
            else:
                print("     [CACHE INVALIDATED]: Config change detected. Rebuilding hindcasts from scratch.")
                cached_ds = None
        except Exception as e:
            print(f"     [CACHE ERROR]: Could not read cache ({e}). Rebuilding.")
            cached_ds = None

    if not missing_years:
        return cached_ds # Fast exit, we have everything!

    # 2. Compute missing years
    all_hindcasts = {lead: [] for lead in leads}
    all_obs = {lead: [] for lead in leads}

    for v_year in missing_years:
        target_date = pd.Timestamp(year=v_year, month=target_month, day=1)
        if target_date not in valid_dates:
            continue

        target_idx = np.where(valid_dates == target_date)[0][0]
        target_ic = Z_stacked[target_idx, :]

        # Exclusion Radius
        exclusion_start = target_date - pd.DateOffset(years=config.ANALOG_EXCLUSION_RADIUS)
        exclusion_end = target_date + pd.DateOffset(years=config.ANALOG_EXCLUSION_RADIUS)
        train_mask = (valid_dates < exclusion_start) | (valid_dates > exclusion_end)
        train_dates = valid_dates[train_mask]
        
        X = Z_stacked[train_mask, :]
        y = target_ic

        # Fast Ridge Solver
        import scipy.linalg as la
        A = np.dot(X, X.T)
        B = np.dot(X, y)
        avg_diag = np.mean(np.diag(A))
        A_ridge = A + np.eye(X.shape[0]) * (avg_diag * ridge_alpha)
        weights = la.solve(A_ridge, B, assume_a='sym')
        weights_1d = xr.DataArray(weights, coords=[train_dates], dims=["time"])

        # Project Forward
        for lead in leads:
            target_hist_dates = train_dates + pd.DateOffset(months=lead)
            lead_anoms = anomalies_da.sel(time=target_hist_dates).assign_coords(time=train_dates)
            hindcast = lead_anoms.dot(weights_1d, dims="time")

            actual_obs_date = target_date + pd.DateOffset(months=lead)
            if actual_obs_date in anomalies_da.time:
                actual_obs = anomalies_da.sel(time=actual_obs_date)
                
                # Assign a 'verif_time' coordinate to stamp the year
                hindcast = hindcast.assign_coords(verif_time=target_date)
                actual_obs = actual_obs.assign_coords(verif_time=target_date)
                
                all_hindcasts[lead].append(hindcast)
                all_obs[lead].append(actual_obs)

    # 3. Compile new data
    new_hc_grids, new_ob_grids = [], []
    for lead in leads:
        if all_hindcasts[lead]:
            new_hc_grids.append(xr.concat(all_hindcasts[lead], dim="verif_time").expand_dims(lead=[lead]))
            new_ob_grids.append(xr.concat(all_obs[lead], dim="verif_time").expand_dims(lead=[lead]))

    if new_hc_grids:
        new_ds = xr.Dataset({
            "hindcast": xr.concat(new_hc_grids, dim="lead"),
            "observation": xr.concat(new_ob_grids, dim="lead")
        })
        
        # 4. Merge with existing cache if applicable
        final_ds = xr.concat([cached_ds, new_ds], dim="verif_time").sortby("verif_time") if cached_ds else new_ds
        
        # 5. Save updated cache with config state
        final_ds.attrs["config_state"] = current_state_str
        final_ds.to_netcdf(cache_file)
        return final_ds
        
    return cached_ds

