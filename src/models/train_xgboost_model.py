import pandas as pd
import numpy as np
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# -----------------------------------
# Paths
# -----------------------------------

INPUT_PATH = "data/processed/grain_pricing_feature_engineered.csv"

OUTPUT_RESULTS = "reports/tables/xgboost_results.csv"

# -----------------------------------
# Load data
# -----------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date")

# -----------------------------------
# Define target
# -----------------------------------

target = "volume_tons"

# -----------------------------------
# Feature selection
# -----------------------------------

feature_cols = [
    "product_type",
    "region",
    "customer_segment",
    "market_index_price",
    "competitor_price",
    "discount_pct",
    "final_price",
    "cost_per_ton",
    "weather_index",
    "inventory_level",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "price_gap_vs_competitor",
    "discount_amount",
    "margin_per_ton",
    "inventory_demand_ratio",
    "lag_1_demand",
    "lag_4_demand",
    "rolling_4w_avg_demand",
    "rolling_8w_avg_demand",
    "relative_price_position"
]

X = df[feature_cols]
y = df[target]

# -----------------------------------
# Categorical preprocessing
# -----------------------------------

categorical_cols = [
    "product_type",
    "region",
    "customer_segment"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# -----------------------------------
# XGBoost model
# -----------------------------------

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.80,
    colsample_bytree=0.80,
    random_state=42,
    objective="reg:squarederror"
)

# -----------------------------------
# Pipeline
# -----------------------------------

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])

# -----------------------------------
# Time Series Cross Validation
# -----------------------------------

tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
rmse_scores = []
mape_scores = []

fold = 1

for train_idx, test_idx in tscv.split(X):

    print(f"\nTraining fold {fold}...")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mape = np.mean(
        np.abs((y_test - predictions) / y_test)
    ) * 100

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    mape_scores.append(mape)

    print(f"Fold {fold} completed.")
    print(f"MAPE: {round(mape, 2)}%")

    fold += 1

# -----------------------------------
# Aggregate Results
# -----------------------------------

results_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE"],
    "Average": [
        round(np.mean(mae_scores), 2),
        round(np.mean(rmse_scores), 2),
        round(np.mean(mape_scores), 2)
    ]
})

# -----------------------------------
# Save results
# -----------------------------------

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)

results_df.to_csv(OUTPUT_RESULTS, index=False)

print("\nXGBoost Results")
print(results_df)

print(f"\nSaved results to: {OUTPUT_RESULTS}")