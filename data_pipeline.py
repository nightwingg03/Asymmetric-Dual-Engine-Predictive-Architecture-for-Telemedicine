import pandas as pd
import numpy as np
import os

# Define paths
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\14564935"
MACHINE_USAGE_PATH = os.path.join(DATA_DIR, "machine_usage_days_1_to_8_grouped_300_seconds.csv")

print("1. Loading dataset...")
# Load Machine Usage Data
# Columns available: cpu_util_percent, mem_util_percent, net_in, net_out, disk_io_percent
df_machine = pd.read_csv(MACHINE_USAGE_PATH)

print("2. Mapping Cloud Metrics to Telemedicine Features...")
# Map network load to 'SessionLoad'
df_machine['SessionLoad'] = df_machine['net_in'] + df_machine['net_out']

# Proxy Active Consultations based on memory and cpu usage
df_machine['ActiveConsultations'] = (df_machine['mem_util_percent'] * 0.5 + df_machine['cpu_util_percent'] * 0.5).astype(int)

# Simulate Patient Priority (1.0 = emergency, 0.3 = routine)
np.random.seed(42)
df_machine['PatientPriority'] = np.random.choice([0.3, 0.5, 0.8, 1.0], size=len(df_machine), p=[0.5, 0.3, 0.15, 0.05])

# Simulate Latency based on net_in and disk_io
df_machine['Latency'] = df_machine['net_in'] * 0.1 + df_machine['disk_io_percent'] * 0.2 + np.random.normal(0, 0.5, len(df_machine))
df_machine['Latency'] = df_machine['Latency'].clip(lower=0)


print("3. Engineering SLA Violation Labels...")
# Define SLA breach: cpu_util > 90% AND latency > threshold OR random network drop
LATENCY_THRESHOLD = df_machine['Latency'].quantile(0.95) # Top 5% latency
CPU_THRESHOLD = 85.0 # Adjusted slightly for the dataset distribution

df_machine['SLA_Violation'] = np.where(
    (df_machine['cpu_util_percent'] > CPU_THRESHOLD) & (df_machine['Latency'] > LATENCY_THRESHOLD), 
    1, 
    0
)

# Infuse ~0.5% random task failure SLA violations to match rare event distribution
random_failures = np.random.choice([0, 1], size=len(df_machine), p=[0.995, 0.005])
df_machine['SLA_Violation'] = np.maximum(df_machine['SLA_Violation'], random_failures)

violation_count = df_machine['SLA_Violation'].sum()
print(f"-> Total SLA Violations: {violation_count} out of {len(df_machine)} ({violation_count/len(df_machine)*100:.2f}%)")


print("4. Applying Time-Series Feature Engineering...")
# The dataset is already grouped by 300 seconds (5 min windows)
# Rolling statistics using a 3-row window (15 mins)
df_machine['CPU_Rolling_Mean'] = df_machine['cpu_util_percent'].rolling(window=3, min_periods=1).mean()
df_machine['CPU_Rolling_Std'] = df_machine['cpu_util_percent'].rolling(window=3, min_periods=1).std().fillna(0)

# Latency Trend Slope (Difference from previous row)
df_machine['Latency_Trend_Slope'] = df_machine['Latency'].diff().fillna(0)

# Session Density Ratio
df_machine['Session_Density_Ratio'] = df_machine['ActiveConsultations'] / (df_machine['SessionLoad'] + 1e-5)

# Lag features (t-1, t-2, t-3)
df_machine['CPU_Lag_1'] = df_machine['cpu_util_percent'].shift(1).fillna(0)
df_machine['CPU_Lag_2'] = df_machine['cpu_util_percent'].shift(2).fillna(0)
df_machine['CPU_Lag_3'] = df_machine['cpu_util_percent'].shift(3).fillna(0)

# Time-of-day encoding (Assuming 288 intervals of 5-min per day)
df_machine['TimeOfDay'] = (df_machine.index % 288) / 288.0

# Drop any accidental NaNs created by rolling windows
df_machine = df_machine.fillna(0)

print("\nData Preprocessing Complete! First 5 rows:")
print(df_machine[['cpu_util_percent', 'Latency', 'PatientPriority', 'SLA_Violation']].head())

# Save the explicitly processed dataset for the modeling stage
OUTPUT_FILE = os.path.join(DATA_DIR, "telemedicine_processed_data.csv")
df_machine.to_csv(OUTPUT_FILE, index=False)
print(f"\nFinal dataset saved to: {OUTPUT_FILE}")
