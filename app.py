import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
# We use the ARIMA class with seasonal parameters -> This IS SARIMA
from statsmodels.tsa.arima.model import ARIMA 

# --- 1. SILENCE ALL WARNINGS (Terminal Hygiene) ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Metro Manila Caffeine Analysis", layout="wide")

st.title("☕ Econometric Analysis of Urban Caffeine Consumption")
st.markdown("**Project:** Modeling Demand using FIES Micro-Data and Macro-Economic Shocks")

# --- LOAD SYNTHETIC DATA ONLY ---
@st.cache_data
def load_data():
    file_path = "Synthetic_FIES_NCR.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

df = load_data()

if df is None:
    st.error("⚠️ Synthetic Data not found! Please run 'python dummy_generator.py' first.")
    st.stop()

# Map Districts for readability
district_map = {39: 'Manila', 74: 'Quezon City', 75: 'North NCR', 76: 'South/Makati'}
df['District_Name'] = df['W_PROV'].map(district_map).fillna('Other')

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Spatial Analysis", "Regression Model", "ML Profiling", "SARIMA Forecast"])

# --- TAB 1: SPATIAL ---
with tab1:
    st.subheader("Spatial Analysis: Spending by District")
    st.caption("Comparing residential density vs. commercial spending.")
    
    metric = st.selectbox("Select Metric", ["FOOD_OUTSIDE", "COFFEE", "TOINC"], index=0)
    
    grouped = df.groupby('District_Name')[metric].mean().reset_index()
    
    fig = px.bar(grouped, x='District_Name', y=metric, color='District_Name', 
                 color_discrete_sequence=px.colors.qualitative.Bold)
    
    # FIXED: Updated syntax for 2025 compatibility
    st.plotly_chart(fig, width="stretch")

# --- TAB 2: REGRESSION ---
with tab2:
    st.subheader("Income Elasticity of Demand")
    st.caption("Does higher income guarantee higher coffee spending?")
    
    # Filter > 0 to show active consumers
    df_reg = df[df['COFFEE'] > 0].sample(1000) 
    
    X = df_reg[['TOINC']].values
    y = df_reg['COFFEE'].values
    reg = LinearRegression().fit(X, y)
    
    col1, col2 = st.columns(2)
    col1.metric("Elasticity Coefficient", f"{reg.coef_[0]:.4f}")
    col2.metric("R-Squared Correlation", f"{reg.score(X, y):.4f}")
    
    fig = px.scatter(df_reg, x='TOINC', y='COFFEE', trendline='ols', opacity=0.5,
                     title="Regression Analysis (Sample N=1000)")
    
    # FIXED: Updated syntax for 2025 compatibility
    st.plotly_chart(fig, width="stretch")

# --- TAB 3: ML PROFILING ---
with tab3:
    st.subheader("Random Forest Classifier")
    st.caption("Profiling 'High Spenders' (Top 25%) based on demographics.")
    
    threshold = df['COFFEE'].quantile(0.75)
    df['Is_High_Spender'] = (df['COFFEE'] > threshold).astype(int)
    
    features = ['TOINC', 'FSIZE', 'FOOD_OUTSIDE']
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(df[features], df['Is_High_Spender'])
    
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
    
    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance')
    
    # FIXED: Updated syntax for 2025 compatibility
    st.plotly_chart(fig, width="stretch")

# --- TAB 4: SARIMA FORECAST ---
with tab4:
    st.subheader("Hybrid SARIMA Forecast (2024-2026)")
    st.caption("Projecting FIES baseline using Seasonal Indices and Inflation Factors.")
    
    inflation = st.slider("Simulate Inflation Impact (%)", 2, 10, 6) / 100
    
    # Generate Time Series Data
    avg_spend = df['FOOD_OUTSIDE'].mean()
    dates = pd.date_range('2023-01-01', '2025-12-01', freq='MS')
    
    seasonality = [0.95, 0.9, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.1, 1.15, 1.2, 1.3]
    
    vals = []
    for d in dates:
        month_idx = d.month - 1
        val = (avg_spend/12) * seasonality[month_idx] * (1 + (d.year-2023)*inflation) 
        val += np.random.normal(0, val * 0.05) 
        vals.append(val)
    
    ts_df = pd.Series(vals, index=dates)
    
    with st.spinner("Calculating Forecast..."):
        try:
            # METHOD: ARIMA with seasonal_order = SARIMA
            # We use enforce_stationarity=False to prevent mathematical warnings on dummy data
            model = ARIMA(ts_df, order=(1,1,1), seasonal_order=(1,1,1,12),
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()
            fcast = results.get_forecast(steps=12).predicted_mean
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df, name='Historical Data', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=fcast.index, y=fcast, name='2026 Forecast', 
                                     line=dict(color='red', dash='dash', width=3)))
            
            fig.update_layout(title="Projected Consumption Volume", xaxis_title="Date", yaxis_title="Avg Spend (PHP)")
            
            # FIXED: Updated syntax for 2025 compatibility
            st.plotly_chart(fig, width="stretch")
            
        except Exception as e:
            st.error(f"Model Error: {e}")