import pandas as pd
import os


REGRESSION_RESULTS = "reports/tables/regression_model_results.csv"
XGBOOST_RESULTS = "reports/tables/xgboost_results.csv"
TIME_SERIES_RESULTS = "reports/tables/time_series_results.csv"

OUTPUT_PATH = "reports/tables/final_model_comparison.csv"


def load_regression_results():
    df = pd.read_csv(REGRESSION_RESULTS)
    df["model_family"] = "Regression / ML Baseline"
    return df.rename(columns={
        "MAE": "mae",
        "RMSE": "rmse",
        "MAPE": "mape",
        "CV_MAE": "cv_mae"
    })


def load_xgboost_results():
    df = pd.read_csv(XGBOOST_RESULTS)

    metrics = {
        row["Metric"]: row["Average"]
        for _, row in df.iterrows()
    }

    return pd.DataFrame([{
        "model": "XGBoost",
        "mae": metrics.get("MAE"),
        "rmse": metrics.get("RMSE"),
        "mape": metrics.get("MAPE"),
        "cv_mae": metrics.get("MAE"),
        "model_family": "Advanced ML"
    }])


def load_time_series_results():
    df = pd.read_csv(TIME_SERIES_RESULTS)

    metrics = {
        row["Metric"]: row["Value"]
        for _, row in df.iterrows()
    }

    return pd.DataFrame([{
        "model": "ARIMA",
        "mae": metrics.get("MAE"),
        "rmse": metrics.get("RMSE"),
        "mape": metrics.get("MAPE"),
        "cv_mae": None,
        "model_family": "Classical Time Series"
    }])


def add_business_evaluation(df):
    interpretability = {
        "Linear Regression": "High",
        "Ridge Regression": "High",
        "Lasso Regression": "High",
        "Random Forest": "Medium",
        "XGBoost": "Medium",
        "ARIMA": "Medium"
    }

    production_fit = {
        "Linear Regression": "Good baseline",
        "Ridge Regression": "Good baseline",
        "Lasso Regression": "Good baseline",
        "Random Forest": "Strong nonlinear baseline",
        "XGBoost": "Best for pricing + demand drivers",
        "ARIMA": "Good time-series benchmark"
    }

    df["interpretability"] = df["model"].map(interpretability)
    df["production_fit"] = df["model"].map(production_fit)

    return df


if __name__ == "__main__":

    regression_df = load_regression_results()
    xgboost_df = load_xgboost_results()
    time_series_df = load_time_series_results()

    comparison_df = pd.concat(
        [regression_df, xgboost_df, time_series_df],
        ignore_index=True
    )

    comparison_df = add_business_evaluation(comparison_df)

    comparison_df = comparison_df.sort_values("mape")

    best_model = comparison_df.iloc[0]["model"]

    comparison_df["selected_model"] = comparison_df["model"] == best_model

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal Model Comparison")
    print(comparison_df)

    print(f"\nBest model selected: {best_model}")
    print(f"Saved comparison to: {OUTPUT_PATH}")