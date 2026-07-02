# run_ca_seasonal.py
import config
from core.ocean_predictors import run_ocean_predictor_pipeline
from core.analog_engine import calculate_eof_modes, generate_constructed_analog_forecast
from core.index_extraction import extract_forecast_indices
from core.plot_nino34 import plot_ensemble_forecast
from core.plot_spatial import generate_spatial_maps
from core.plot_skill import generate_skill_maps
from core.write_text_tables import write_forecast_tables
import xarray as xr

def main():
    print("====================================================")
    print("Kicking off SEASONAL Constructed Analog Production Run")
    print("====================================================")
    
    RUN_MODE = "seasonal"

    for dtype in config.DATASET_TYPES:
        # 1. Run Ingestion Pipeline
        run_ocean_predictor_pipeline(target_dataset=dtype, mode=RUN_MODE)
        
        anom_file = config.OUT_DATA_DIR / f"{dtype}.{RUN_MODE}.1948-curr.1x1.nc"
        ic_file = config.OUT_DATA_DIR / f"{dtype}.{RUN_MODE}.msic.ic.nc"
        
        anom_ds = xr.open_dataset(anom_file)["sst"]
        ic_ds = xr.open_dataset(ic_file)["sst"]
        
        # 2. Run EOFs
        master_eof_ds = calculate_eof_modes(anom_ds, max_modes=config.MAX_EOF_MODES)
        
        # 3. Generate Forecast Matrices & Indices
        index_storage = []
        clim_da_storage = None
        for modes in config.EVAL_EOF_MODES:
            forecast_production_ds = generate_constructed_analog_forecast(
                anomalies_da=anom_ds,
                master_eof_ds=master_eof_ds,
                ic_da=ic_ds,
                target_modes=modes,
                run_mode=RUN_MODE,
                dataset_name=dtype
            )
            
            out_path = config.OUT_DATA_DIR / f"{dtype}.{RUN_MODE}.forecast_modes_{modes}.nc"
            forecast_production_ds.to_netcdf(out_path)
            print(f"SUCCESS: Production Forecast Asset written to {out_path.name}")
            
            # Extract Indices for this Truncation Mode
            print(f"Extracting Niño 3.4 and RONI indices for {modes}-Mode subset...")
            indices_ds = extract_forecast_indices(
                forecast_da=forecast_production_ds["reconstructed_sst"], 
                anom_ds=anom_ds, 
                ic_ds=ic_ds,
                dataset_name=dtype,
                run_mode=RUN_MODE
            )

            clim_da_storage = indices_ds["nino34_climatology"]
            indices_ds = indices_ds.drop_vars("nino34_climatology")
            
            indices_ds = indices_ds.expand_dims(truncation=[modes])
            index_storage.append(indices_ds)
 
        # 4. Compile the 12-Member Ensemble (4 ICs x 3 Truncations)
        master_index_ds = xr.concat(index_storage, dim="truncation")
        
        # Add a simple ensemble mean across both tracking dimensions
        master_index_ds["nino34_ensemble_mean"] = master_index_ds["nino34_anomaly"].mean(dim=["ic_num", "truncation"])
        master_index_ds["roni_ensemble_mean"] = master_index_ds["roni_anomaly"].mean(dim=["ic_num", "truncation"])
       
        master_index_ds["nino34_climatology"] = clim_da_storage
 
        final_index_path = config.OUT_DATA_DIR / f"{dtype}.{RUN_MODE}.master_indices.nc"
        master_index_ds.to_netcdf(final_index_path)
        print(f"--- Completed {RUN_MODE} forecast run for {dtype.upper()} ---")

    print("\n[POST-PROCESSING]: Writing Text Tables...")
    write_forecast_tables(RUN_MODE)
 
    print("\n[POST-PROCESSING]: Generating Plots...")
    plot_ensemble_forecast(RUN_MODE)
    generate_spatial_maps(RUN_MODE)
    generate_skill_maps(RUN_MODE)

if __name__ == "__main__":
    main()
