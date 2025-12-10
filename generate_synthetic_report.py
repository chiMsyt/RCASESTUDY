import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
plt.style.use("ggplot")  # Gives it a standard academic look

# 1. SETUP OUTPUT FOLDER
output_dir = "Synthetic_Analysis_Output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created folder: {output_dir}")

# 2. LOAD DATA
print("--- LOADING SYNTHETIC DATA ---")
file_path = "Synthetic_FIES_NCR.csv"

if not os.path.exists(file_path):
    print("Error: csv file not found. Run dummy_generator.py first.")
    exit()

df = pd.read_csv(file_path)
print(f"Data Loaded: {len(df)} rows.")

# Map Districts
district_map = {39: "Manila", 74: "Quezon City", 75: "North NCR", 76: "South/Makati"}
df["District_Name"] = df["W_PROV"].map(district_map).fillna("Other")

# 3. SPATIAL ANALYSIS (Bar Charts)
print("Generating Spatial Charts...")
metrics = ["FOOD_OUTSIDE", "COFFEE", "TOINC"]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    # Group by district and get the mean
    grouped = df.groupby("District_Name")[metric].mean().reset_index()

    # Plot
    sns.barplot(data=grouped, x="District_Name", y=metric, palette="viridis")
    plt.title(f"Average {metric} by District (Synthetic)")
    plt.ylabel("Value (PHP)")
    plt.xlabel("District")

    # Save
    filename = f"{output_dir}/Spatial_{metric}.png"
    plt.savefig(filename)
    plt.close()
    print(f" > Saved {filename}")

# 4. REGRESSION ANALYSIS (Scatter Plot)
print("Running Regression...")
df_reg = df[df["COFFEE"] > 0]

X = df_reg[["TOINC"]].values
y = df_reg["COFFEE"].values

reg = LinearRegression().fit(X, y)
r2 = reg.score(X, y)
coef = reg.coef_[0]

print(f" > Linear Model: Coefficient={coef:.5f}, R2={r2:.5f}")

plt.figure(figsize=(10, 6))
# Sample 500 points so the graph isn't too messy
sns.regplot(
    x="TOINC",
    y="COFFEE",
    data=df_reg.sample(min(500, len(df_reg))),
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title(f"Income vs Coffee (Synthetic)\nCoefficient: {coef:.5f}")
plt.savefig(f"{output_dir}/Regression_Analysis.png")
plt.close()

# 5. ML FEATURE IMPORTANCE
print("Running Random Forest...")
threshold = df["COFFEE"].quantile(0.75)
df["Is_High_Spender"] = (df["COFFEE"] > threshold).astype(int)
features = ["TOINC", "FSIZE", "FOOD_OUTSIDE"]

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(df[features], df["Is_High_Spender"])

importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.title("Feature Importances (Synthetic)")
plt.barh(range(len(indices)), importances[indices], color="steelblue", align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.savefig(f"{output_dir}/ML_Feature_Importance.png")
plt.close()

# 6. SARIMA FORECAST (Simulation)
print("Running Forecast Simulation...")
avg_spend = df["FOOD_OUTSIDE"].mean()
dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
seasonality = [0.95, 0.9, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.1, 1.15, 1.2, 1.3]
inflation = 0.06

vals = []
for d in dates:
    month_idx = d.month - 1
    val = (avg_spend / 12) * seasonality[month_idx] * (1 + (d.year - 2023) * inflation)
    val += np.random.normal(0, val * 0.05)
    vals.append(val)

ts_df = pd.Series(vals, index=dates)

try:
    model = ARIMA(
        ts_df,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()
    fcast = results.get_forecast(steps=12).predicted_mean

    plt.figure(figsize=(10, 5))
    plt.plot(ts_df.index, ts_df, label="Historical")
    plt.plot(fcast.index, fcast, label="Forecast", color="red", linestyle="--")
    plt.title("Projected Spending (Synthetic Baseline)")
    plt.legend()
    plt.savefig(f"{output_dir}/Forecast.png")
    plt.close()
    print(f" > Saved {output_dir}/Forecast.png")
except:
    print(" > Error running ARIMA (check statsmodels installation)")

print("Done.")
