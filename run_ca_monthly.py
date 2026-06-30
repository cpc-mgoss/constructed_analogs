# run_ca_monthly.py
import config
import xarray as xr
from core.ocean_predictors import run_ocean_predictor_pipeline
from core.analog_engine import calculate_eof_modes

def main():
    print("====================================================")
    print("Kicking off MONTHLY Constructed Analog Run")
    print("====================================================")
    
    for dtype in config.DATASET_TYPES:
        # Step 1: Ingest raw system files and generate anomalies (No temporal smoothing)
        run_ocean_predictor_pipeline(target_dataset=dtype, mode="monthly")
        
        # Step 2: Load the processed anomalies back into memory
        anomaly_file = config.OUT_DATA_DIR / f"{dtype}.monthly.1948-curr.1x1.nc"
        print(f"Loading anomalies for EOF analysis: {anomaly_file.name}")
        anom_ds = xr.open_dataset(anomaly_file)["sst"]
        
        # Step 3: Run the high-performance SVD once at the max boundary
        master_eof_outputs = calculate_eof_modes(anom_ds, max_modes=config.MAX_EOF_MODES)
        
        # Step 4: Save exactly ONE master file containing the full spectrum
        out_filename = config.OUT_DATA_DIR / f"{dtype}.monthly.eof_modes_master.nc"
        master_eof_outputs.to_netcdf(out_filename)
        
        print(f"SUCCESS: Master spectral matrix written to {out_filename.name}")
        print(f"--- Completed Monthly Track for {dtype.upper()} ---\n")

if __name__ == "__main__":
    main()
