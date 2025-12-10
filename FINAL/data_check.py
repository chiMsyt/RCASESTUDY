import os

import pandas as pd

# NOTE: Update this path to match your system
data_path = "/home/tim/Desktop/RCASESTUDY/DATA_FIES/FIES PUF 2023 Volume1.CSV"

try:
    # 1. Load the raw, uncleaned national data
    # I use 'latin1' encoding as it's common for large survey datasets
    df_raw = pd.read_csv(data_path, encoding="latin1")
except FileNotFoundError:
    print("Error: File not found at the specified path.")
    exit()

# 2. Print necessary technical details
print("--- RAW FIES DATA SNAPSHOT ---")
print(f"Initial National Sample Size (N): {len(df_raw)}")
print(f"Total Columns (Fields): {df_raw.shape[1]}")
print("\nFirst 5 Rows and Header:")
print(df_raw.head().to_string())
