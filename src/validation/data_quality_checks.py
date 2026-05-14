import pandas as pd
import os


INPUT_PATH = "data/raw/grain_pricing_demand_synthetic.csv"
OUTPUT_PATH = "reports/tables/data_quality_report.csv"


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    checks.append({
        "check_name": "total_rows",
        "issue_count": len(df),
        "description": "Total number of rows in dataset"
    })

    checks.append({
        "check_name": "duplicate_rows",
        "issue_count": df.duplicated().sum(),
        "description": "Rows that are exact duplicates"
    })

    checks.append({
        "check_name": "missing_final_price",
        "issue_count": df["final_price"].isna().sum(),
        "description": "Missing final price values"
    })

    checks.append({
        "check_name": "missing_customer_segment",
        "issue_count": df["customer_segment"].isna().sum(),
        "description": "Missing customer segment values"
    })

    checks.append({
        "check_name": "negative_volume",
        "issue_count": (df["volume_tons"] < 0).sum(),
        "description": "Demand volume cannot be negative"
    })

    checks.append({
        "check_name": "negative_inventory",
        "issue_count": (df["inventory_level"] < 0).sum(),
        "description": "Inventory level cannot be negative"
    })

    checks.append({
        "check_name": "price_outliers",
        "issue_count": (df["final_price"] > df["final_price"].quantile(0.99)).sum(),
        "description": "Extreme high price values above 99th percentile"
    })

    expected_revenue = df["final_price"] * df["volume_tons"]

    checks.append({
        "check_name": "invalid_revenue",
        "issue_count": ((df["revenue"] - expected_revenue).abs() > 1).sum(),
        "description": "Revenue should approximately equal final_price * volume_tons"
    })

    return pd.DataFrame(checks)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    report = run_quality_checks(df)

    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save report
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nData Quality Report")
    print(report)
    print(f"\nSaved report to: {OUTPUT_PATH}")