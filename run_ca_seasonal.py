# run_ca_seasonal.py
import config
import xarray as xr
from core.ocean_predictors import run_ocean_predictor_pipeline
from core.analog_engine import calculate_eof_modes

def main():
    print("====================================================")
    print(f"Kicking off SEASONAL Constructed Analog Run")
    print("====================================================")
    
    for dtype in config.DATASET_TYPES:
        # Step 1: Ingest raw system files and generate anomalies
        run_ocean_predictor_pipeline(target_dataset=dtype, mode="seasonal")
        
        # Step 2: Load the processed anomalies back into memory
        anomaly_file = config.OUT_DATA_DIR / f"{dtype}.seasonal.1948-curr.1x1.nc"
        print(f"Loading anomalies for EOF analysis: {anomaly_file.name}")
        anom_ds = xr.open_dataset(anomaly_file)["sst"]
        
        # Step 3: Run the multi-mode loops (15, 25, 40) as dictated by the legacy setup
        for modes in [15, 25, 40]:
            eof_outputs = calculate_eof_modes(anom_ds, max_modes=modes)
            
            # Save out clean spectral checkpoints to your sandboxed playground
            out_filename = config.OUT_DATA_DIR / f"{dtype}.seasonal.eof_modes_{modes}.nc"
            eof_outputs.to_netcdf(out_filename)
            print(f"SUCCESS: Spectral matrix saved to {out_filename.name}\n")

if __name__ == "__main__":
    main()
