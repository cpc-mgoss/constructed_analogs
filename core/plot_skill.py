# core/plot_skill.py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config

def generate_skill_maps(run_mode="seasonal"):
    print("====================================================")
    print(f"GENERATING SKILL MAPS ({run_mode.upper()} - {config.SKILL_CALC_MODE.upper()} MODE)")
    print("====================================================")
    
    is_seasonal = (run_mode == "seasonal")
    out_dir = config.OUT_DIR_SEASONAL if is_seasonal else config.OUT_DIR_MONTHLY
    
    # ---------------------------------------------------------
    # MODE 1: LEGACY (Average the pre-computed ACs)
    # ---------------------------------------------------------
    if config.SKILL_CALC_MODE == "legacy":
        ac_grids = []
        for dtype in config.DATASET_TYPES:
            for modes in config.EVAL_EOF_MODES:
                fcst_file = config.OUT_DATA_DIR / f"{dtype}.{run_mode}.forecast_modes_{modes}.nc"
                if not fcst_file.exists():
                    print(f"  [WARNING] Missing {fcst_file.name}, skipping for skill mean...")
                    continue
                
                ds = xr.open_dataset(fcst_file)
                # ds["ac"] is shape (ic_num, lead, lat, lon)
                ac_grids.append(ds["ac"])
                
        # Stack all 6 datasets (2 dtypes x 3 truncations) and average across them and the ICs
        master_ac = xr.concat(ac_grids, dim="ensemble_group").mean(dim=["ensemble_group", "ic_num"])

    # ---------------------------------------------------------
    # MODE 2: MODERN (Skill of the Ensemble Mean Hindcast)
    # ---------------------------------------------------------
    elif config.SKILL_CALC_MODE == "ensemble_mean":
        hindcast_grids = []
        
        # 1. Load all 24 cached hindcasts
        print("  -> Loading physical hindcasts from Cache...")
        for dtype in config.DATASET_TYPES:
            for modes in config.EVAL_EOF_MODES:
                for msic in config.IC_NUMS:
                    # e.g., hindcasts_sst_ersst_seasonal_month_04_msic_1_modes_15.nc
                    cache_name = f"hindcasts_sst_{dtype}_{run_mode}_month_{config.RUN_MO:02d}_msic_{msic}_modes_{modes}.nc"
                    cache_file = config.CACHE_DIR / cache_name
                    
                    if cache_file.exists():
                        h_ds = xr.open_dataset(cache_file)
                        hindcast_grids.append(h_ds["reconstructed_sst"])
                    else:
                        print(f"  [WARNING] Cache miss on {cache_name}")
                        
        if not hindcast_grids:
            print("  [ERROR] No cache files found to compute ensemble mean skill.")
            return

        # 2. Average the physical hindcasts together (filtering the noise BEFORE the correlation)
        super_ensemble_hindcast = xr.concat(hindcast_grids, dim="ensemble").mean(dim="ensemble")
        
        # 3. Load the verification dataset (ERSST Observations acting as "Truth")
        truth_file = config.OUT_DATA_DIR / f"ersst.{run_mode}.1948-curr.1x1.nc"
        truth_ds = xr.open_dataset(truth_file)
        truth_anom = truth_ds["sst"]
        
        print("  -> Correlating Super-Ensemble against Observations...")
        
        # 4. Calculate correlation Lead-by-Lead
        # (Using a loop ensures dimensional alignment between the hindcast valid_dates and obs time)
        ac_leads = []
        for lead_idx, lead_val in enumerate(super_ensemble_hindcast.lead.values):
            # Extract the hindcast for this lead (shape: year, lat, lon)
            lead_hcst = super_ensemble_hindcast.isel(lead=lead_idx)
            
            # Use Xarray's built-in Pearson correlation over the hindcast years
            # It will automatically align the 'time'/'valid_date' coords with the truth_anom!
            lead_ac = xr.corr(lead_hcst, truth_anom, dim="year") 
            ac_leads.append(lead_ac)
            
        # Reconstruct into a single (lead, lat, lon) array
        master_ac = xr.concat(ac_leads, dim="lead")
        master_ac["lead"] = super_ensemble_hindcast.lead.values

    else:
        print(f"  [ERROR] Unknown SKILL_CALC_MODE: {config.SKILL_CALC_MODE}")
        return

    # ---------------------------------------------------------
    # PLOTTING THE MAPS
    # ---------------------------------------------------------
    print("  -> Rendering Skill Graphics...")
    leads = master_ac.lead.values
    
    # Loop through the leads and generate the legacy-style PNGs
    for i, lead in enumerate(leads):
        # In operations, they usually only plot leads 1 through 9, but we can plot them all.
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Extract the 2D AC grid for this specific lead
        ac_grid = master_ac.isel(lead=i)
        
        # Simple contour plot (you can easily upgrade this to Cartopy if desired)
        # Using a colormap that highlights high skill (red) and negative skill (blue)
        plot = ac_grid.plot.contourf(
            ax=ax, 
            levels=np.arange(-1.0, 1.1, 0.1), 
            cmap="RdBu_r", 
            add_colorbar=True,
            extend="both"
        )
        
        ax.set_title(f"Constructed Analog SST Anomaly Correlation\nRun Mode: {run_mode.upper()} | Lead: {lead} Months")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Route output to the correct isolated directory
        out_img = out_dir / f"casst.ac.{lead}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close(fig)
        
    print(f"SUCCESS: Skill maps generated in {out_dir.name}/")
    print("====================================================")

if __name__ == "__main__":
    generate_skill_maps("seasonal")
