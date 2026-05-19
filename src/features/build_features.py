import pandas as pd
import numpy as np
import os


INPUT_PATH = "data/processed/grain_pricing_cleaned.csv"
OUTPUT_PATH = "data/processed/grain_pricing_feature_engineered.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    print("\nStarting feature engineering...\n")

    # -----------------------------------
    # Convert date
    # -----------------------------------

    df["date"] = pd.to_datetime(df["date"])

    # Sort data for lag calculations
    df = df.sort_values(
        by=["product_type", "region", "date"]
    )

    # -----------------------------------
    # Time Features
    # -----------------------------------

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    print("Created time-based features.")

    # -----------------------------------
    # Price Features
    # -----------------------------------

    df["price_gap_vs_competitor"] = (
        df["final_price"] - df["competitor_price"]
    )

    df["discount_amount"] = (
        df["market_index_price"] - df["final_price"]
    )

    df["margin_per_ton"] = (
        df["final_price"] - df["cost_per_ton"]
    )

    print("Created pricing and margin features.")

    # -----------------------------------
    # Inventory Features
    # -----------------------------------

    df["inventory_demand_ratio"] = (
        df["inventory_level"] / df["volume_tons"]
    )

    print("Created inventory features.")

    # -----------------------------------
    # Lag Features
    # -----------------------------------

    group_cols = ["product_type", "region"]

    df["lag_1_demand"] = (
        df.groupby(group_cols)["volume_tons"]
        .shift(1)
    )

    df["lag_4_demand"] = (
        df.groupby(group_cols)["volume_tons"]
        .shift(4)
    )

    print("Created lag demand features.")

    # -----------------------------------
    # Rolling Window Features
    # -----------------------------------

    df["rolling_4w_avg_demand"] = (
        df.groupby(group_cols)["volume_tons"]
        .transform(lambda x: x.rolling(4).mean())
    )

    df["rolling_8w_avg_demand"] = (
        df.groupby(group_cols)["volume_tons"]
        .transform(lambda x: x.rolling(8).mean())
    )

    print("Created rolling demand features.")

    # -----------------------------------
    # Price Elasticity Proxy
    # -----------------------------------

    df["relative_price_position"] = (
        df["final_price"] / df["competitor_price"]
    )

    print("Created pricing elasticity proxy.")

    # -----------------------------------
    # Handle lag/rolling nulls
    # -----------------------------------

    numeric_cols = [
        "lag_1_demand",
        "lag_4_demand",
        "rolling_4w_avg_demand",
        "rolling_8w_avg_demand"
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Filled lag feature null values.")

    print("\nFeature engineering completed.")

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    feature_df = build_features(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFeature dataset saved to: {OUTPUT_PATH}")