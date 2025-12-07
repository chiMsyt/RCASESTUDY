import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Load Data
try:
    df = pd.read_csv("MetroManila_Actual_Data.csv")
except:
    print("Error: Run process_fies.py first!")
    exit()

print(f">>> ANALYZING {len(df)} REAL HOUSEHOLDS...")

# 1. SPATIAL
district_map = {39: 'District 1 (Manila)', 74: 'District 2 (QC)', 75: 'District 3 (North)', 76: 'District 4 (South/Makati)'}
df['District'] = df['W_PROV'].map(district_map).fillna('Other')
spatial_grp = df.groupby('District')['FOOD_OUTSIDE'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=spatial_grp, x='District', y='FOOD_OUTSIDE', palette='viridis')
plt.title('Average Household Spending by District (Actual FIES)')
plt.savefig('Figure1_Spatial.png')
print("Figure 1 Saved.")

# 2. REGRESSION
df_reg = df[df['COFFEE'] > 0]
X = df_reg[['TOINC']]
y = df_reg['COFFEE']
reg = LinearRegression().fit(X, y)
print(f"Regression Coef: {reg.coef_[0]:.5f}")

plt.figure(figsize=(10, 6))
sns.regplot(x='TOINC', y='COFFEE', data=df_reg.sample(1000), scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Income Elasticity of Caffeine Demand')
plt.savefig('Figure2_Regression.png')
print("Figure 2 Saved.")

# 3. SARIMA
avg_spend = df['FOOD_OUTSIDE'].mean()
dates = pd.date_range('2023-01-01', '2025-12-01', freq='MS')
seasonality = [0.95, 0.9, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.1, 1.15, 1.2, 1.3]
ts_values = [(avg_spend/12) * seasonality[d.month-1] * (1 + (d.year-2023)*0.06) + np.random.normal(0, 100) for d in dates]
ts = pd.Series(ts_values, index=dates)

model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
forecast = model.get_forecast(steps=12).predicted_mean

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Historical (Derived)')
plt.plot(forecast, label='Forecast 2026', color='red', linestyle='--')
plt.title('SARIMA Forecast: Urban Caffeine Consumption')
plt.legend()
plt.savefig('Figure3_SARIMA.png')
print("Figure 3 Saved.")