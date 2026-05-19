import pandas as pd
import numpy as np
import os


INPUT_PATH = "data/raw/grain_pricing_demand_synthetic.csv"
OUTPUT_PATH = "data/processed/grain_pricing_cleaned.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\nStarting data cleaning pipeline...\n")

    initial_rows = len(df)

    # -----------------------------------
    # 1. Remove duplicate rows
    # -----------------------------------

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"Removed duplicate rows: {duplicates}")

    # -----------------------------------
    # 2. Fix missing customer segment
    # -----------------------------------

    missing_segment = df["customer_segment"].isna().sum()

    df["customer_segment"] = df["customer_segment"].fillna("Unknown")

    print(f"Filled missing customer segments: {missing_segment}")

    # -----------------------------------
    # 3. Impute missing final_price
    # -----------------------------------

    missing_price = df["final_price"].isna().sum()

    median_prices = (
        df.groupby("product_type")["final_price"]
        .transform("median")
    )

    df["final_price"] = df["final_price"].fillna(median_prices)

    print(f"Imputed missing final_price values: {missing_price}")

    # -----------------------------------
    # 4. Remove negative demand
    # -----------------------------------

    negative_volume = (df["volume_tons"] < 0).sum()

    df = df[df["volume_tons"] >= 0]

    print(f"Removed negative demand rows: {negative_volume}")

    # -----------------------------------
    # 5. Fix negative inventory
    # -----------------------------------

    negative_inventory = (df["inventory_level"] < 0).sum()

    df.loc[df["inventory_level"] < 0, "inventory_level"] = np.nan

    inventory_median = df["inventory_level"].median()

    df["inventory_level"] = df["inventory_level"].fillna(inventory_median)

    print(f"Fixed negative inventory rows: {negative_inventory}")

    # -----------------------------------
    # 6. Cap extreme price outliers
    # -----------------------------------

    upper_limit = df["final_price"].quantile(0.99)

    outliers = (df["final_price"] > upper_limit).sum()

    df["final_price"] = np.where(
        df["final_price"] > upper_limit,
        upper_limit,
        df["final_price"]
    )

    print(f"Capped price outliers: {outliers}")

    # -----------------------------------
    # 7. Recalculate revenue
    # -----------------------------------

    df["revenue"] = (
        df["final_price"] * df["volume_tons"]
    ).round(2)

    # -----------------------------------
    # 8. Recalculate margin
    # -----------------------------------

    df["margin"] = (
        (df["final_price"] - df["cost_per_ton"])
        * df["volume_tons"]
    ).round(2)

    # -----------------------------------
    # 9. Convert date column
    # -----------------------------------

    df["date"] = pd.to_datetime(df["date"])

    final_rows = len(df)

    print("\nCleaning pipeline completed.")

    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {final_rows}")

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    cleaned_df = clean_data(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")