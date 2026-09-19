"""
Retail Sales Forecasting — Interactive Demo
SafeX Solutions Internship — Week 4: Advanced Predictive Model

Lets a user pick an existing Store/Product/Date combination from the dataset,
see what the model predicted vs. what actually happened, and tweak "what-if"
inputs (price, discount, weather, holiday flag, etc.) to see how the
prediction changes. Uses the same Ridge Regression (alpha=200) pipeline
built and tuned in retail_sales_forecasting.ipynb.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge

st.set_page_config(page_title = "Retail Sales Forecasting Demo", layout = "wide")

DATA_PATH = "retail_store_inventory.csv"

FEATURE_COLUMNS = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing", "Seasonality",
    "Year", "Month", "Day", "DayOfWeek", "IsWeekend",
    "Lag_1", "Lag_7", "Rolling_7",
]


# ----------------------------------------------------------------------------
# Data loading + feature engineering (mirrors the notebook exactly)
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_engineer():
    df = pd.read_csv(DATA_PATH)

    # --- Clean (same checks as the notebook; dataset is already clean) ---
    df["Date"] = pd.to_datetime(df["Date"], errors = "coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Units Sold"] >= 0]
    df = df.drop_duplicates()

    # --- Calendar features ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)


    # --- Lag / rolling features, per store-product series ---
    group_cols = ["Store ID", "Product ID"]
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)
    df["Lag_1"] = df.groupby(group_cols)["Units Sold"].shift(1)
    df["Lag_7"] = df.groupby(group_cols)["Units Sold"].shift(7)
    df["Rolling_7"] = (
        df.groupby(group_cols)["Units Sold"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df_model = df.dropna(subset=["Lag_1", "Lag_7", "Rolling_7"]).copy()
    df_model = df_model.sort_values("Date").reset_index(drop=True)
    return df_model


@st.cache_resource
def train_model(df_model: pd.DataFrame):
    X = df_model[FEATURE_COLUMNS].copy()
    y = df_model["Units Sold"].copy()

    X = pd.get_dummies(X, drop_first=True)
    train_columns = X.columns

    model = Ridge(alpha=200.0, random_state=42)
    model.fit(X, y)
    return model, train_columns


def predict_row(model, train_columns, row: pd.Series) -> float:
    X_new = pd.DataFrame([row[FEATURE_COLUMNS]])
    X_new = pd.get_dummies(X_new, drop_first=True)
    X_new = X_new.reindex(columns=train_columns, fill_value=0)
    return float(model.predict(X_new)[0])


# ----------------------------------------------------------------------------
#                                     App
# ----------------------------------------------------------------------------
st.title("📦 Retail Sales Forecasting — Demo")
st.caption(
    "Ridge Regression (alpha = 200), tuned with RandomizedSearchCV + TimeSeriesSplit. "
    "Test-set performance: MAE ≈ 68.6 · RMSE ≈ 87.6 · R² ≈ 0.34"
)

with st.spinner("Loading data and training model..."):
    df_model = load_and_engineer()
    model, train_columns = train_model(df_model)

st.sidebar.header("Pick a record")

store_ids = sorted(df_model["Store ID"].unique())
selected_store = st.sidebar.selectbox("Store ID", store_ids)

product_ids = sorted(
    df_model.loc[df_model["Store ID"] == selected_store, "Product ID"].unique()
)
selected_product = st.sidebar.selectbox("Product ID", product_ids)

subset = df_model[
    (df_model["Store ID"] == selected_store)
    & (df_model["Product ID"] == selected_product)
].sort_values("Date")

selected_date = st.sidebar.select_slider(
    "Date",
    options=list(subset["Date"].dt.date),
    value=list(subset["Date"].dt.date)[-1],
)

row = subset[subset["Date"].dt.date == selected_date].iloc[0].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Calendar and lag/rolling features come from the actual history for this "
    "store-product series and aren't user-editable — only the day-of inputs below are."
)

# ----------------------------------------------------------------------------
#                                What-if inputs
# ----------------------------------------------------------------------------
st.subheader(f"Store {selected_store} · Product {selected_product} · {selected_date}")

col1, col2, col3 = st.columns(3)

with col1:
    price = st.number_input("Price", value=float(row["Price"]), step=1.0)
    discount = st.number_input("Discount (%)", value=float(row["Discount"]), step=1.0)
    inventory = st.number_input(
        "Inventory Level", value=float(row["Inventory Level"]), step=1.0
    )

with col2:
    units_ordered = st.number_input(
        "Units Ordered", value=float(row["Units Ordered"]), step=1.0
    )
    competitor_price = st.number_input(
        "Competitor Pricing", value=float(row["Competitor Pricing"]), step=1.0
    )
    weather = st.selectbox(
        "Weather Condition",
        sorted(df_model["Weather Condition"].unique()),
        index=sorted(df_model["Weather Condition"].unique()).index(row["Weather Condition"]),
    )

with col3:
    holiday = st.selectbox(
        "Holiday/Promotion",
        sorted(df_model["Holiday/Promotion"].unique()),
        index=sorted(df_model["Holiday/Promotion"].unique()).index(row["Holiday/Promotion"]),
    )
    seasonality = st.selectbox(
        "Seasonality",
        sorted(df_model["Seasonality"].unique()),
        index=sorted(df_model["Seasonality"].unique()).index(row["Seasonality"]),
    )

# apply what-if edits on top of the real historical row
row["Price"] = price
row["Discount"] = discount
row["Inventory Level"] = inventory
row["Units Ordered"] = units_ordered
row["Competitor Pricing"] = competitor_price
row["Weather Condition"] = weather
row["Holiday/Promotion"] = holiday
row["Seasonality"] = seasonality

predicted = predict_row(model, train_columns, row)
actual = float(row["Units Sold"])

st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted Units Sold", f"{predicted:.1f}")
m2.metric("Actual Units Sold (that day)", f"{actual:.0f}")
m3.metric("Difference", f"{predicted - actual:+.1f}")

st.caption(
    "Note: R² ≈ 0.34 on the test set — the model captures general patterns "
    "(product/store identity, weekend effect, inventory level) but daily demand "
    "in this dataset has a lot of noise the features here don't explain. "
    "See Documentation.docx for the full write-up."
)

with st.expander("Model comparison (from the notebook)"):
    st.markdown(
        """
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression (tuned) | 68.62 | 87.60 | 0.344 |
| Gradient Boosting | 68.67 | 87.72 | 0.342 |
| Random Forest | 69.03 | 88.48 | 0.331 |
"""
    )
