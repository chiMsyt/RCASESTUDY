import pandas as pd
import numpy as np

# Configuration
np.random.seed(42)
n_rows = 5000

# --- GENERATE SYNTHETIC COLUMNS ---
# 1. Region (Always 13 for NCR)
w_regn = [13] * n_rows

# 2. Province/District (Matches FIES Codes: 39=Manila, 74=QC, 75=North, 76=South)
districts = [39, 74, 75, 76]
weights = [0.20, 0.40, 0.20, 0.20]
w_prov = np.random.choice(districts, n_rows, p=weights)

# 3. Family Size
fsize = np.random.poisson(4, n_rows)
fsize = np.clip(fsize, 1, 12)

# 4. Total Income
toinc = np.random.lognormal(mean=12.5, sigma=0.6, size=n_rows)
toinc = np.round(toinc, 2)

# 5. Total Expenditure
totex = toinc * np.random.uniform(0.70, 0.95, n_rows)

# 6. Coffee Expenditure (Correlated with Income)
coffee_base = (toinc * 0.015) + np.random.normal(0, 500, n_rows)
coffee = np.clip(coffee_base, 0, 20000)

# 7. Food Outside (Correlated with District)
district_multiplier = np.where(w_prov == 76, 1.2, 1.0)
food_outside = (toinc * 0.05 * district_multiplier) + np.random.normal(0, 2000, n_rows)
food_outside = np.clip(food_outside, 0, None)

# Create DataFrame
df = pd.DataFrame({
    'W_REGN': w_regn,
    'W_PROV': w_prov,
    'FSIZE': fsize,
    'TOINC': toinc,
    'TOTEX': totex,
    'COFFEE': coffee,
    'FOOD_OUTSIDE': food_outside,
    'COCOA': coffee * 0.1
})
df['SEQ_NO'] = range(1, n_rows + 1)

# Save to Root
df.to_csv("Synthetic_FIES_NCR.csv", index=False)
print(">>> Synthetic Data Generated in Root.")