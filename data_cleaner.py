import pandas as pd
import numpy as np
import re
import os

print("1. Loading raw dataset (this might take a minute for 400k rows)...")
# Load the dataset
input_file = r"c:\Users\harsh\OneDrive\Desktop\Cloud Project\dataset_raw.csv"
output_file = r"c:\Users\harsh\OneDrive\Desktop\Cloud Project\dataset_clean.csv"

# Load with low_memory=False to prevent mixed-type warnings on messy columns
df = pd.read_csv(input_file, low_memory=False)

# Drop the weird unnamed index column if it exists (caused by the leading comma in the header)
unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

print(f"Original shape: {df.shape}")
if 'failed' in df.columns:
    print(f"-> Crucial SLA Violation (failed) rows present: {df['failed'].sum()}")

print("\n2. Parsing ugly dictionary values into distinct numeric columns...")
# Columns containing python dictionary strings
dict_cols = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']

for col in dict_cols:
    if col in df.columns:
        print(f"   -> Unpacking {col}...")
        # Use fast string regex extraction instead of slow eval()
        df[f'{col}_cpus'] = df[col].astype(str).str.extract(r"'cpus':\s*([0-9.]+|None)", expand=False)
        df[f'{col}_memory'] = df[col].astype(str).str.extract(r"'memory':\s*([0-9.]+|None)", expand=False)
        
        # Convert to numeric, turning 'None' into proper Pandas NaN
        df[f'{col}_cpus'] = pd.to_numeric(df[f'{col}_cpus'].replace('None', np.nan), errors='coerce')
        df[f'{col}_memory'] = pd.to_numeric(df[f'{col}_memory'].replace('None', np.nan), errors='coerce')
        
        # Drop the original messy dictionary string column
        df = df.drop(columns=[col])

print("\n3. Handling missing values without dropping rows...")
# To absolutely guarantee we don't lose any violation signals, we will impute (fill) NaNs instead of dropping rows.
# For numerical columns, we fill with 0 (since a missing metric like CPU or cycles usually means zero activity logged)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(0)

# For any string/object columns with NaNs, fill with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Unknown')

print("\n4. Removing messy/chunky array format strings...")
# Columns that are essentially gigantic unparsed arrays (like "[0.00314 0.00381...]") 
# These bloat the RAM heavily and are redundant since we just extracted average/max usage.
chunky_cols = ['cpu_usage_distribution', 'tail_cpu_usage_distribution', 'constraint', 'start_after_collection_ids']
cols_to_drop = [c for c in chunky_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)

print(f"\nCleaned shape: {df.shape}")
if 'failed' in df.columns:
    print(f"-> Final Crucial SLA Violation (failed) rows correctly preserved: {df['failed'].sum()}")

print(f"\n5. Saving to {output_file}...")
df.to_csv(output_file, index=False)
print("Done! Cleaned dataset saved.")
