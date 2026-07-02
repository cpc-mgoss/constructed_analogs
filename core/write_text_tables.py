import xarray as xr
import numpy as np
import config

def write_forecast_tables(run_mode="seasonal"):
    print("====================================================")
    print(f"GENERATING TEXT TABLES ({run_mode.upper()})")
    print("====================================================")
    
    out_dir = config.OUT_DIR_SEASONAL if run_mode == "seasonal" else config.OUT_DIR_MONTHLY

    for dataset in ["ersst", "hadoisst"]:
        nc_file = config.OUT_DATA_DIR / f"{dataset}.{run_mode}.master_indices.nc"
        if not nc_file.exists():
            continue
            
        ds = xr.open_dataset(nc_file)
        leads = ds.lead.values
        nld = len(leads)
        
        # Determine Dataset Label
        ds_label = "ERSST.v4" if dataset == "ersst" else "HAD-OISST"
        
        # Flatten the 4 MSICs and 3 Truncations into the 12 legacy members
        # Legacy order: loops through MSIC, then loops through Modes (or vice versa depending on the bash loop)
        # We will iterate exactly as the bash script did: msic 1->4, modes 15->40
        members_nino_anom = []
        members_roni_anom = []
        
        for msic in ds.ic_num.values:
            for trunc in ds.truncation.values:
                members_nino_anom.append(ds["nino34_anomaly"].sel(ic_num=msic, truncation=trunc).values)
                members_roni_anom.append(ds["roni_anomaly"].sel(ic_num=msic, truncation=trunc).values)
                
        members_nino_anom = np.array(members_nino_anom) # Shape: (12, 17)
        members_roni_anom = np.array(members_roni_anom)
        
        # Fetch Climatology (assuming we added it to the NetCDF in index_extraction.py)
        # Fallback to zeros if it's not there yet so the script doesn't crash
        if "nino34_climatology" in ds:
            c1d_nino = ds["nino34_climatology"].values
        else:
            c1d_nino = np.zeros(nld)
            
        # Write files for both indices
        for index_type, anom_data in [("nino34", members_nino_anom), ("roni", members_roni_anom)]:
            
            # Calculate Totals and Means
            tot_data = anom_data + c1d_nino
            anom_avg = np.mean(anom_data, axis=0)
            tot_avg = np.mean(tot_data, axis=0)
            
            out_file = out_dir / f"ca.{dataset}.{index_type}.out"
            
            with open(out_file, "w") as f:
                f.write(f" CA {index_type.upper()} based on {ds_label}\n")
                
                # Header: 17I7, A12
                header = "".join([f"{l:7d}" for l in leads]) + f"{'lead(mon)':>12s}\n"
                f.write(header)
                
                # Members: Totals then Anomalies
                for im in range(12):
                    tot_str = "".join([f"{tot_data[im, ld]:7.2f}" for ld in range(nld)]) + f"{im+1:4d}\n"
                    f.write(tot_str)
                    anom_str = "".join([f"{anom_data[im, ld]:7.2f}" for ld in range(nld)]) + f"{im+1:4d}\n"
                    f.write(anom_str)
                    
                f.write(" ensemble average\n")
                f.write("".join([f"{tot_avg[ld]:7.2f}" for ld in range(nld)]) + "\n")
                f.write("".join([f"{anom_avg[ld]:7.2f}" for ld in range(nld)]) + "\n")
                f.write(" climatology\n")
                f.write("".join([f"{c1d_nino[ld]:7.2f}" for ld in range(nld)]) + "\n\n\n")
                
            print(f"SUCCESS: Wrote {out_file.name}")

if __name__ == "__main__":
    write_forecast_tables("seasonal")
