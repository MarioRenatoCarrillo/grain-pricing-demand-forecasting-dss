import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------------
# Paths
# -----------------------------------

INPUT_PATH = "data/processed/grain_pricing_feature_engineered.csv"

OUTPUT_PATH = "reports/tables/regression_model_results.csv"

# -----------------------------------
# Load data
# -----------------------------------

df = pd.read_csv(INPUT_PATH)

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
# Categorical and numeric columns
# -----------------------------------

categorical_cols = [
    "product_type",
    "region",
    "customer_segment"
]

numeric_cols = [
    col for col in feature_cols
    if col not in categorical_cols
]

# -----------------------------------
# Preprocessing
# -----------------------------------

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
# Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------------
# Models
# -----------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# -----------------------------------
# Training loop
# -----------------------------------

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

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

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    cv_mae = abs(cv_scores.mean())

    results.append({
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "CV_MAE": round(cv_mae, 2)
    })

    print(f"{model_name} completed.")

# -----------------------------------
# Results table
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAPE"
)

# -----------------------------------
# Save results
# -----------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nModel Comparison Results")
print(results_df)

print(f"\nSaved results to: {OUTPUT_PATH}")