import os

import pandas as pd

# --- SMART PATH SETUP ---
# Get the absolute path of THIS script (process_fies.py)
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../RCASESTUDY/FINAL
# Go up one level to get the Project Root
project_root = os.path.dirname(script_dir)  # .../RCASESTUDY
# Construct path to Data Folder
data_folder = os.path.join(project_root, "DATA_FIES")
# Construct path to Output File (Keep it in FINAL folder)
output_file = os.path.join(script_dir, "MetroManila_Actual_Data.csv")

file_vol1 = "FIES PUF 2023 Volume1.CSV"
path_vol1 = os.path.join(data_folder, file_vol1)

print(f">>> LOOKING FOR DATA IN: {path_vol1}")

try:
    # 1. Load Data
    df = pd.read_csv(path_vol1, encoding="latin1")

    # 2. Filter for NCR (Region 13)
    print("Filter for Region 13 (NCR)...")
    df_ncr = df[df["W_REGN"] == 13].copy()
    print(f" > Found {len(df_ncr)} households.")

    # 3. Extract Columns
    cols = [
        "W_REGN",
        "W_PROV",
        "SEQ_NO",
        "FSIZE",
        "TOINC",
        "COFFEE",
        "COCOA",
        "FOOD_OUTSIDE",
        "TOTEX",
    ]
    df_clean = df_ncr[cols].copy()

    # 4. Save to FINAL folder
    df_clean.to_csv(output_file, index=False)
    print(f">>> SUCCESS: Clean data saved to: {output_file}")

except FileNotFoundError:
    print("\n[ERROR] File not found!")
    print(f"Checked path: {path_vol1}")
    print("Please check that the CSV file name in DATA_FIES matches exactly.")
except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")
