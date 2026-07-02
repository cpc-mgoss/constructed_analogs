# weight_avg.py
import xarray as xr
import numpy as np
import pandas as pd
import config

def generate_weight_tables():
    print("====================================================")
    print("GENERATING CA ENSEMBLE WEIGHT TABLES")
    print("====================================================")
    
    for dataset in ["ersst", "hadoisst"]:
        print(f"\n Ensemble Averaged Weights(%) for {dataset.upper()}")
        print("\n")
        
        all_weights = []
        for modes in config.EVAL_EOF_MODES:
            file_path = config.OUT_DATA_DIR / f"{dataset}.seasonal.forecast_modes_{modes}.nc"
            if not file_path.exists():
                print(f"[ERROR]: Missing file {file_path.name}. Run pipeline first.")
                return
            
            ds = xr.open_dataset(file_path)
            all_weights.append(ds["analog_weights"])
            
        # Stack the 3 mode arrays together, then mean across modes and the 4 IC tiers
        combined_weights = xr.concat(all_weights, dim="truncation")
        mean_weights = combined_weights.mean(dim=["truncation", "ic_num"])
        
        # Convert fractional weights to percentages and round to nearest integer
        weights_pct = np.round(mean_weights.values * 100).astype(int)
        years = pd.DatetimeIndex(mean_weights.time.values).year.values
        
        # Output in chunks of 10 to exactly match Fortran format(10I6)
        for i in range(0, len(years), 10):
            chunk_years = years[i:i+10]
            chunk_weights = weights_pct[i:i+10]
            
            # Format each number to be exactly 6 characters wide, right-aligned
            year_str = "".join([f"{y:>6}" for y in chunk_years])
            weight_str = "".join([f"{w:>6}" for w in chunk_weights])
            
            print(year_str)
            print(weight_str)

if __name__ == "__main__":
    generate_weight_tables()
