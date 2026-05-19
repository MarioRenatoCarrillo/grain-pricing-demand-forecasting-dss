import streamlit as st
import pandas as pd

st.set_page_config(page_title="Grain Pricing DSS", layout="wide")

st.title("Grain Pricing & Demand Forecasting DSS")

kpi = pd.read_csv("dashboards/powerbi_exports/business_kpi_table.csv")
models = pd.read_csv("dashboards/powerbi_exports/model_comparison_table.csv")
scenarios = pd.read_csv("dashboards/powerbi_exports/pricing_scenario_table.csv")
quality = pd.read_csv("dashboards/powerbi_exports/data_quality_table.csv")

page = st.sidebar.selectbox(
    "Select Page",
    ["Executive Overview", "Model Performance", "Pricing Scenarios", "Data Quality"]
)

if page == "Executive Overview":
    st.header("Executive Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${kpi['total_revenue'].sum():,.0f}")
    col2.metric("Total Margin", f"${kpi['total_margin'].sum():,.0f}")
    col3.metric("Total Volume Tons", f"{kpi['total_volume_tons'].sum():,.0f}")

    st.line_chart(kpi.groupby("date")[["total_revenue", "total_margin"]].sum())
    st.bar_chart(kpi.groupby("product_type")["total_volume_tons"].sum())

elif page == "Model Performance":
    st.header("Model Performance")
    st.dataframe(models)
    st.bar_chart(models.set_index("model")["mape"])

elif page == "Pricing Scenarios":
    st.header("Pricing Scenario Simulator")
    st.dataframe(scenarios)
    st.bar_chart(scenarios.set_index("scenario")[["predicted_revenue", "predicted_margin"]])

elif page == "Data Quality":
    st.header("Data Quality Monitoring")
    st.dataframe(quality)
    st.bar_chart(quality.set_index("check_name")["issue_count"])