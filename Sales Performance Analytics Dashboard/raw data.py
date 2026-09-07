import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of sales records
n_records = 1000

# ---------------------------
# Lists used to generate data
# ---------------------------

sales_reps = [
    "Ahmed Khan",
    "Sara Ali",
    "Usman Malik",
    "Ayesha Noor",
    "Hamza Sheikh",
    "Fatima Ahmed",
    "Bilal Hussain",
    "Hira Shah",
    "Omar Farooq",
    "Mariam Khan"
]

regions = ["North", "South", "East", "West"]

products = [
    "Managed Security Services",
    "Penetration Testing",
    "Security Audit",
    "Cloud Security",
    "Incident Response"
]

customer_types = ["Enterprise", "SMB", "Startup"]

deal_statuses = ["Won", "Won", "Won", "Won", "Lost"]

# ---------------
# Generate dates
# ---------------

dates = pd.date_range(
    start = "2025-01-01",
    end = "2025-12-31",
    periods = n_records
)

# --------------------
# Generate random data
# --------------------

data = {
    "Sale_ID": range(1, n_records + 1),

    "Date": dates,

    "Sales_Rep": np.random.choice(sales_reps, size = n_records),

    "Region": np.random.choice(regions, size = n_records),

    "Product": np.random.choice(products, size = n_records),

    "Customer_Type": np.random.choice(customer_types, size = n_records, p = [0.40, 0.40, 0.20]),

    "Units_Sold": np.random.randint(1, 6, size = n_records),

    "Deal_Status": np.random.choice(deal_statuses, size = n_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# ---------------
# Product pricing
# ---------------

product_prices = {
    "Managed Security Services": 12000,
    "Penetration Testing": 8000,
    "Security Audit": 6000,
    "Cloud Security": 10000,
    "Incident Response": 15000
}

# Assign base price
df["Base_Price"] = df["Product"].map(product_prices)

# Add realistic price variation
price_variation = np.random.uniform(0.85, 1.15, size = n_records)

df["Revenue"] = (df["Base_Price"]* df["Units_Sold"]* price_variation).round(2)

# --------------------
# Create sales targets
# --------------------

df["Target"] = (df["Revenue"]* np.random.uniform(0.85, 1.20, size = n_records)).round(2)

# -----------------------------
# Remove helper column
# -----------------------------

df.drop(columns = ["Base_Price"], inplace = True)

# ------------
# Sort by date
# ------------

df.sort_values(by = "Date", inplace = True)

# Reset index
df.reset_index(drop = True, inplace = True)

# --------
# Save CSV
# --------

output_file = "sales_data.csv"

# Make sure Date contains only the date
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

output_file = "sales_data.csv"

df.to_csv(output_file, index = False)

print("Sales dataset created successfully!")
print(f"Records: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved as: {output_file}")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary:")
print(df.describe())
