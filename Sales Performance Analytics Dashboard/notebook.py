import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("sales_data.csv")

df.head()
df.info()


df["Date"] = pd.to_datetime(df["Date"])
df.dtypes

# -----------------------
# Checking Missing Values
#------------------------

df.isnull().sum()


# -------------------
# Checking Duplicates
# -------------------

df.duplicated().sum()
df["Sale_ID"].duplicated().sum()


# -----------------------------
# Checking Numerical Statistics
# -----------------------------


df[["Units_Sold", "Revenue", "Target"]].describe()

# ---------------------------
# Checking Categorical Values
# ---------------------------

df["Sales_Rep"].unique()
df["Region"].unique()
df["Product"].unique()
df["Customer_Type"].unique()
df["Deal_Status"].unique()

# -------------------------
# Creating Business Metrics
# -------------------------
is_won = df["Deal_Status"] == "Won"

#
# --- Won-only Revenue used for business metrics ---
#
df["Won_Revenue"] = np.where(is_won, df["Revenue"], 0)


#
# --- Variance ---
# 
df["Variance"] = df["Revenue"] - df["Target"]

# 
# --- Achievement % ---
#
df["Achievement_%"] = np.where(df["Target"] != 0, (df["Revenue"] / df["Target"] * 100).round(2), 0)

# --------------
# Creating Month
# --------------

df["Month"] = df["Date"].dt.month_name()


# ---------------------
# Creating Month Number
# This will be useful for correctly sorting months later in Power BI.
# ---------------------
df["Month_Number"] = df["Date"].dt.month


# ----------------
# Creating Quarter
# ----------------
df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)


# -------------
# Creating Year
# -------------
df["Year"] = df["Date"].dt.year


# --------------------
# Checking New Dataset
# --------------------
df.head()

# ---------------------------------
# Checking Calculations are correct
# ---------------------------------

df[["Revenue", "Target", "Variance", "Achievement_%"]].head(10)


# -------------------------
# Creating Business Summary
# -------------------------
#
# --- Total Revenue ---
#
total_revenue = df["Revenue"].sum()
print(f"Total Revenue: ${total_revenue:.2f}")

#
# --- Total Target ---
# 
total_target = df["Target"].sum()
print(f"Total Target: ${total_target:.2f}")

#
# --- Overall Achievement ---
#
overall_achievement = (total_revenue / total_target * 100)
print(f"Overall Achievement: {overall_achievement:.2f}%")

#
# --- Total Units Sold ---
#
total_units = df["Units_Sold"].sum()
print(f"Total Units Sold: {total_units:,}")


#
# --- Win Rate --- (new: how much of pipeline actually closed)
#
win_rate = (is_won.sum() / len(df) * 100)
print(f"Win Rate: {win_rate:.2f}% ({is_won.sum()} of {len(df)} deals)")
print(f"Lost Revenue (excluded from totals above): ${df.loc[~is_won, 'Revenue'].sum():.2f}")


# -------------------------------
# Revenue by Sales Representative
# -------------------------------

revenue_by_rep = (df.groupby("Sales_Rep")["Revenue"].sum().sort_values(ascending = False))
print(revenue_by_rep)

# -----------------
# Revenue by Region
# -----------------

revenue_by_region = (df.groupby("Region")["Revenue"].sum().sort_values(ascending = False))
print(revenue_by_region)


# ------------------
# Revenue by Product
# ------------------

revenue_by_product = (df.groupby("Product")["Revenue"].sum().sort_values(ascending = False))
print(revenue_by_product)


# -------------
# Revenue Trend
# -------------

monthly_revenue = (df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum())
print(monthly_revenue)


# ------------------------
# Save the cleaned Dataset
# ------------------------

df.to_csv("clean_sales_data.csv", index = False)
print("Clean dataset saved successfully!")


# ----------------
# Final Validation
# ----------------

print("Rows: ", len(df))
print("Columns: ", len(df.columns))
print("Missing Values: ", df.isnull().sum().sum())
print("Duplicate Rows: ", df.duplicated().sum())
