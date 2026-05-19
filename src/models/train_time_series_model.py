import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------------
# Paths
# -----------------------------------

INPUT_PATH = "data/processed/grain_pricing_feature_engineered.csv"

OUTPUT_RESULTS = "reports/tables/time_series_results.csv"

OUTPUT_FIGURE = "reports/figures/arima_forecast.png"

# -----------------------------------
# Load data
# -----------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

# -----------------------------------
# Aggregate weekly demand
# -----------------------------------

ts_df = (
    df.groupby("date")["volume_tons"]
    .sum()
    .reset_index()
)

ts_df = ts_df.sort_values("date")

# -----------------------------------
# Train/Test Split
# -----------------------------------

train_size = int(len(ts_df) * 0.80)

train = ts_df.iloc[:train_size]
test = ts_df.iloc[train_size:]

# -----------------------------------
# Fit ARIMA
# -----------------------------------

print("\nTraining ARIMA model...\n")

model = ARIMA(
    train["volume_tons"],
    order=(2, 1, 2)
)

model_fit = model.fit()

# -----------------------------------
# Forecast
# -----------------------------------

forecast = model_fit.forecast(
    steps=len(test)
)

# -----------------------------------
# Metrics
# -----------------------------------

mae = mean_absolute_error(
    test["volume_tons"],
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test["volume_tons"],
        forecast
    )
)

mape = np.mean(
    np.abs(
        (
            test["volume_tons"] - forecast
        ) / test["volume_tons"]
    )
) * 100

# -----------------------------------
# Save results
# -----------------------------------

results_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE"],
    "Value": [
        round(mae, 2),
        round(rmse, 2),
        round(mape, 2)
    ]
})

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

# -----------------------------------
# Plot Forecast
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    train["date"],
    train["volume_tons"],
    label="Train"
)

plt.plot(
    test["date"],
    test["volume_tons"],
    label="Actual"
)

plt.plot(
    test["date"],
    forecast,
    label="Forecast"
)

plt.title("ARIMA Demand Forecast")

plt.xlabel("Date")
plt.ylabel("Demand Volume")

plt.legend()

os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)

plt.savefig(OUTPUT_FIGURE)

print("ARIMA model completed.")

print("\nTime Series Results")
print(results_df)

print(f"\nSaved results to: {OUTPUT_RESULTS}")
print(f"Saved figure to: {OUTPUT_FIGURE}")