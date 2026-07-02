# core/plot_nino34.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd
import numpy as np
import config

def plot_ensemble_forecast(run_mode="seasonal"):
    print("====================================================")
    print(f"GENERATING NINO 3.4 ENSEMBLE FORECAST PLOT ({run_mode.upper()})")
    print("====================================================")
    
    is_seasonal = (run_mode == "seasonal")

    # 1. Load the Master Indices dynamically based on run_mode
    ersst_file = config.OUT_DATA_DIR / f"ersst.{run_mode}.master_indices.nc"
    had_file = config.OUT_DATA_DIR / f"hadoisst.{run_mode}.master_indices.nc"

    if not ersst_file.exists() or not had_file.exists():
        print(f"[ERROR]: Master index files not found for {run_mode}. Run the pipeline first.")
        return

    ds_ersst = xr.open_dataset(ersst_file)
    ds_had = xr.open_dataset(had_file)

    # 2. Reconstruct the true Target Date timeline for the X-axis
    lag = config.RUN_MODES[run_mode]["lag_months"]
    base_target_date = pd.Timestamp(year=config.RUN_YR, month=config.RUN_MO, day=1) - pd.DateOffset(months=int(lag))

    leads = ds_ersst.lead.values
    target_dates = [base_target_date + pd.DateOffset(months=int(l)) for l in leads]

    # 3. Setup the Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Color mapping mimicking the legacy plot's grouping
    ersst_colors = {1: 'black', 2: 'red', 3: 'limegreen', 4: 'blue'}
    had_colors = {1: 'cyan', 2: 'magenta', 3: 'goldenrod', 4: 'darkorange'}

    # 4. Plot Individual Ensemble Members (Spaghetti)
    for trunc in ds_ersst.truncation.values:
        for ic in ds_ersst.ic_num.values:
            # ERSST Members
            ersst_vals = ds_ersst["nino34_anomaly"].sel(truncation=trunc, ic_num=ic).values
            ax.plot(target_dates, ersst_vals, color=ersst_colors[ic], linewidth=0.7, alpha=0.8)

            # HADOISST Members
            had_vals = ds_had["nino34_anomaly"].sel(truncation=trunc, ic_num=ic).values
            ax.plot(target_dates, had_vals, color=had_colors[ic], linewidth=0.7, alpha=0.8)

    # 5. Calculate and Plot the Grand Ensemble Mean
    combined_anoms = xr.concat([ds_ersst["nino34_anomaly"], ds_had["nino34_anomaly"]], dim="dataset")
    grand_mean = combined_anoms.mean(dim=["dataset", "truncation", "ic_num"]).values

    ax.plot(target_dates, grand_mean, color='black', linewidth=2.5, marker='s', markersize=6,
            markerfacecolor='black', markeredgecolor='white', zorder=10, label="ENSEMBLE MEAN")

    # 6. Add Custom Text Labels
    text_x = target_dates[2] # Offset slightly to the left
    ax.text(text_x, 2.15, 'ER-SST, IC-1   OI-SST, IC-1', color='black', fontsize=10, ha='left')
    ax.text(text_x, 2.00, 'ER-SST, IC-2   OI-SST, IC-2', color='red', fontsize=10, ha='left')
    ax.text(text_x, 1.85, 'ER-SST, IC-3   OI-SST, IC-3', color='limegreen', fontsize=10, ha='left')
    ax.text(text_x, 1.70, 'ER-SST, IC-4   OI-SST, IC-4', color='blue', fontsize=10, ha='left')
    ax.text(text_x, 1.55, 'ENSEMBLE MEAN', color='black', fontsize=11, fontweight='bold', ha='left')

    # 7. Formatting limits and axes
    ax.set_ylim(-1.5, 2.6)
    ax.set_xlim(target_dates[0], target_dates[-1])

    ax.set_ylabel("SST-anomaly (K)", fontsize=14)
    xlabel_text = "central month of season" if is_seasonal else "target forecast month"
    ax.set_xlabel(xlabel_text, fontsize=14)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    def custom_date_formatter(x, pos):
        dt = mdates.num2date(x)
        if dt.month == 1 or pos == 0:
            return dt.strftime('%b\n%Y').upper()
        else:
            return dt.strftime('%b').upper()

    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))

    # Calculate "data thru" based on true observation availability
    # (Current Run Month minus 1)
    data_thru = pd.Timestamp(year=config.RUN_YR, month=config.RUN_MO, day=1) - pd.DateOffset(months=1)
    
    title = f"CA Forecast for Nino3.4 SST Index\n24 members, data thru {data_thru.strftime('%b%Y')}"
    ax.set_title(title, fontsize=14)

    # Footer text
    fig.text(0.1, 0.02, "NOAA/NWS/NCEP/CPC (Modernized Python Engine)", fontsize=10, color='blue', ha='left')
    fig.text(0.9, 0.02, "Base Period 1991-2020", fontsize=10, color='blue', ha='right')

    # 8. Save output dynamically to the correct directory!
    out_dir = config.OUT_DIR_SEASONAL if is_seasonal else config.OUT_DIR_MONTHLY
    out_img = out_dir / "nino34_ensemble_forecast.png"
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_img, dpi=150)
    print(f"SUCCESS: Plot saved to {out_img}")
    print("====================================================")
    plt.close(fig)

if __name__ == "__main__":
    plot_ensemble_forecast("seasonal")
