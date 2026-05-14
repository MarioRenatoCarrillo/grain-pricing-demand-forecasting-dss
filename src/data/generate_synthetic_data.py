import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------
# 1. Create base dimensions
# -----------------------------

dates = pd.date_range(start="2022-01-01", end="2025-12-31", freq="W")

products = ["Corn", "Soybeans", "Wheat", "Canola"]
regions = ["Midwest", "South", "West", "Northeast"]
customer_segments = ["Commercial", "Cooperative", "Exporter", "Food Processor"]

records = []

# -----------------------------
# 2. Business assumptions
# -----------------------------

base_prices = {
    "Corn": 210,
    "Soybeans": 480,
    "Wheat": 260,
    "Canola": 520
}

base_demand = {
    "Corn": 1200,
    "Soybeans": 900,
    "Wheat": 700,
    "Canola": 500
}

region_multiplier = {
    "Midwest": 1.20,
    "South": 0.90,
    "West": 0.75,
    "Northeast": 0.65
}

segment_multiplier = {
    "Commercial": 1.00,
    "Cooperative": 1.15,
    "Exporter": 1.30,
    "Food Processor": 0.85
}

# -----------------------------
# 3. Generate clean synthetic data
# -----------------------------

for date in dates:
    month = date.month

    # Seasonal demand effect
    if month in [3, 4, 5]:
        seasonal_factor = 1.15
    elif month in [9, 10, 11]:
        seasonal_factor = 1.25
    else:
        seasonal_factor = 0.95

    for product in products:
        for region in regions:
            for segment in customer_segments:

                base_price = base_prices[product]
                market_index_price = base_price * np.random.normal(1.00, 0.05)
                competitor_price = market_index_price * np.random.normal(1.00, 0.04)

                discount_pct = np.random.choice(
                    [0, 0.02, 0.03, 0.05, 0.07],
                    p=[0.40, 0.20, 0.20, 0.15, 0.05]
                )

                final_price = market_index_price * (1 - discount_pct)

                cost_per_ton = final_price * np.random.uniform(0.70, 0.85)

                weather_index = np.random.normal(1.0, 0.12)

                inventory_level = np.random.normal(
                    base_demand[product] * region_multiplier[region] * 2,
                    150
                )

                # Price sensitivity logic
                price_gap = final_price - competitor_price
                price_effect = -2.5 * price_gap

                demand = (
                    base_demand[product]
                    * region_multiplier[region]
                    * segment_multiplier[segment]
                    * seasonal_factor
                    * weather_index
                    + price_effect
                    + np.random.normal(0, 80)
                )

                demand = max(demand, 50)

                revenue = final_price * demand
                margin = (final_price - cost_per_ton) * demand

                records.append({
                    "date": date,
                    "product_type": product,
                    "region": region,
                    "customer_segment": segment,
                    "market_index_price": round(market_index_price, 2),
                    "competitor_price": round(competitor_price, 2),
                    "discount_pct": discount_pct,
                    "final_price": round(final_price, 2),
                    "cost_per_ton": round(cost_per_ton, 2),
                    "weather_index": round(weather_index, 2),
                    "inventory_level": round(inventory_level, 0),
                    "volume_tons": round(demand, 0),
                    "revenue": round(revenue, 2),
                    "margin": round(margin, 2)
                })

df = pd.DataFrame(records)

# -----------------------------
# 4. Add intentional data issues
# -----------------------------

# Missing prices
missing_price_idx = df.sample(frac=0.01, random_state=1).index
df.loc[missing_price_idx, "final_price"] = np.nan

# Missing customer segments
missing_segment_idx = df.sample(frac=0.01, random_state=2).index
df.loc[missing_segment_idx, "customer_segment"] = np.nan

# Negative demand
negative_demand_idx = df.sample(frac=0.005, random_state=3).index
df.loc[negative_demand_idx, "volume_tons"] = -abs(df.loc[negative_demand_idx, "volume_tons"])

# Extreme price outliers
outlier_price_idx = df.sample(frac=0.005, random_state=4).index
df.loc[outlier_price_idx, "final_price"] = df.loc[outlier_price_idx, "final_price"] * 5

# Invalid revenue
invalid_revenue_idx = df.sample(frac=0.01, random_state=5).index
df.loc[invalid_revenue_idx, "revenue"] = df.loc[invalid_revenue_idx, "revenue"] * 0.4

# Negative inventory
negative_inventory_idx = df.sample(frac=0.005, random_state=6).index
df.loc[negative_inventory_idx, "inventory_level"] = -abs(df.loc[negative_inventory_idx, "inventory_level"])

# Duplicate records
duplicates = df.sample(frac=0.01, random_state=7)
df = pd.concat([df, duplicates], ignore_index=True)

# -----------------------------
# 5. Save dataset
# -----------------------------

# Save dataset
output_path = "data/raw/grain_pricing_demand_synthetic.csv"

df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")

print("Synthetic dataset created successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())