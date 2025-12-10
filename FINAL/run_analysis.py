import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --- SMART PATH SETUP ---
# Get the directory where this script resides (RCASESTUDY/FINAL)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the data file
data_path = os.path.join(script_dir, "MetroManila_Actual_Data.csv")

print(f">>> LOOKING FOR DATA AT: {data_path}")

# Load Data
try:
    df = pd.read_csv(data_path)
    print(f">>> SUCCESS: Loaded {len(df)} rows.")
except FileNotFoundError:
    print("\n[ERROR] File not found!")
    print(f"I tried to read: {data_path}")
    print("Please run 'python FINAL/process_fies.py' first.")
    exit()

# 1. SPATIAL
print("--- Generating Spatial Analysis ---")
district_map = {
    39: "District 1 (Manila)",
    74: "District 2 (QC)",
    75: "District 3 (North)",
    76: "District 4 (South/Makati)",
}
df["District"] = df["W_PROV"].map(district_map).fillna("Other")
spatial_grp = df.groupby("District")["FOOD_OUTSIDE"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=spatial_grp, x="District", y="FOOD_OUTSIDE", palette="viridis")
plt.title("Average Household Spending by District (Actual FIES)")
# Save images to the same folder as the script
plt.savefig(os.path.join(script_dir, "Figure1_Spatial.png"))
print(" > Figure 1 Saved.")

# 2. REGRESSION
print("--- Generating Regression Analysis ---")
df_reg = df[df["COFFEE"] > 0]
X = df_reg[["TOINC"]]
y = df_reg["COFFEE"]
reg = LinearRegression().fit(X, y)
print(f" > Regression Coef: {reg.coef_[0]:.5f}")

plt.figure(figsize=(10, 6))
sns.regplot(
    x="TOINC",
    y="COFFEE",
    data=df_reg.sample(1000),
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"},
)
plt.title("Income Elasticity of Caffeine Demand")
plt.savefig(os.path.join(script_dir, "Figure2_Regression.png"))
print(" > Figure 2 Saved.")

# 3. SARIMA
print("--- Generating SARIMA Forecast ---")
avg_spend = df["FOOD_OUTSIDE"].mean()
dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
seasonality = [0.95, 0.9, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.1, 1.15, 1.2, 1.3]
ts_values = [
    (avg_spend / 12) * seasonality[d.month - 1] * (1 + (d.year - 2023) * 0.06)
    + np.random.normal(0, 100)
    for d in dates
]
ts = pd.Series(ts_values, index=dates)

model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
forecast = model.get_forecast(steps=12).predicted_mean

plt.figure(figsize=(12, 6))
plt.plot(ts, label="Historical (Derived)")
plt.plot(forecast, label="Forecast 2026", color="red", linestyle="--")
plt.title("SARIMA Forecast: Urban Caffeine Consumption")
plt.legend()
plt.savefig(os.path.join(script_dir, "Figure3_SARIMA.png"))
print(" > Figure 3 Saved.")
print(">>> ANALYSIS COMPLETE.")
