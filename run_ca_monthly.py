# run_ca_monthly.py
import config
from core.ocean_predictors import run_ocean_predictor_pipeline

def main():
    print("====================================================")
    print(f"Kicking off MONTHLY Constructed Analog Run")
    print("====================================================")
    
    for dtype in config.DATASET_TYPES:
        run_ocean_predictor_pipeline(target_dataset=dtype, mode="monthly")

if __name__ == "__main__":
    main()
