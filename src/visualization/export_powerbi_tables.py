import pandas as pd
import os


FEATURE_DATA = "data/processed/grain_pricing_feature_engineered.csv"
MODEL_COMPARISON = "reports/tables/final_model_comparison.csv"
SCENARIO_RESULTS = "reports/tables/pricing_scenario_results.csv"
DATA_QUALITY = "reports/tables/data_quality_report.csv"

OUTPUT_DIR = "dashboards/powerbi_exports"


def export_powerbi_tables():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA)
    df["date"] = pd.to_datetime(df["date"])

    # Main business KPI table
    kpi_table = (
        df.groupby(["date", "product_type", "region", "customer_segment"])
        .agg(
            total_volume_tons=("volume_tons", "sum"),
            total_revenue=("revenue", "sum"),
            total_margin=("margin", "sum"),
            avg_final_price=("final_price", "mean"),
            avg_competitor_price=("competitor_price", "mean"),
            avg_inventory=("inventory_level", "mean"),
            avg_weather_index=("weather_index", "mean"),
        )
        .reset_index()
    )

    kpi_table["margin_rate"] = (
        kpi_table["total_margin"] / kpi_table["total_revenue"]
    )

    kpi_table.to_csv(
        f"{OUTPUT_DIR}/business_kpi_table.csv",
        index=False
    )

    # Model comparison table
    model_comparison = pd.read_csv(MODEL_COMPARISON)
    model_comparison.to_csv(
        f"{OUTPUT_DIR}/model_comparison_table.csv",
        index=False
    )

    # Pricing scenario table
    scenario_results = pd.read_csv(SCENARIO_RESULTS)
    scenario_results.to_csv(
        f"{OUTPUT_DIR}/pricing_scenario_table.csv",
        index=False
    )

    # Data quality table
    data_quality = pd.read_csv(DATA_QUALITY)
    data_quality.to_csv(
        f"{OUTPUT_DIR}/data_quality_table.csv",
        index=False
    )

    print("Power BI export tables created successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    export_powerbi_tables()