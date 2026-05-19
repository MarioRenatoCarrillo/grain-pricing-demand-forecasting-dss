import pandas as pd
import numpy as np
import os


INPUT_PATH = "data/processed/grain_pricing_feature_engineered.csv"
OUTPUT_PATH = "reports/tables/pricing_scenario_results.csv"


def simulate_pricing_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []

    price_changes = [-0.10, -0.05, 0.00, 0.05, 0.10]

    latest_df = df[df["date"] == df["date"].max()].copy()

    for change in price_changes:
        scenario_df = latest_df.copy()

        scenario_df["scenario"] = f"{int(change * 100)}% price change"

        scenario_df["scenario_price"] = (
            scenario_df["final_price"] * (1 + change)
        )

        # Simple demand response assumption:
        # if price decreases, demand increases
        # if price increases, demand decreases
        elasticity = -1.2

        scenario_df["predicted_demand"] = (
            scenario_df["volume_tons"]
            * (1 + elasticity * change)
        )

        scenario_df["predicted_revenue"] = (
            scenario_df["scenario_price"]
            * scenario_df["predicted_demand"]
        )

        scenario_df["predicted_margin"] = (
            (scenario_df["scenario_price"] - scenario_df["cost_per_ton"])
            * scenario_df["predicted_demand"]
        )

        scenario_df["inventory_risk_flag"] = np.where(
            scenario_df["predicted_demand"] > scenario_df["inventory_level"],
            "High Risk",
            "Normal"
        )

        scenarios.append(scenario_df)

    results = pd.concat(scenarios, ignore_index=True)

    summary = (
        results.groupby("scenario")
        .agg(
            predicted_demand=("predicted_demand", "sum"),
            predicted_revenue=("predicted_revenue", "sum"),
            predicted_margin=("predicted_margin", "sum"),
            high_inventory_risk_count=(
                "inventory_risk_flag",
                lambda x: (x == "High Risk").sum()
            )
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    scenario_results = simulate_pricing_scenarios(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    scenario_results.to_csv(OUTPUT_PATH, index=False)

    print("\nPricing Scenario Results")
    print(scenario_results)

    print(f"\nSaved scenario results to: {OUTPUT_PATH}")