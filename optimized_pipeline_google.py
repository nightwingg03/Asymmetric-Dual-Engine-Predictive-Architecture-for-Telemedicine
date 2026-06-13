import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score, confusion_matrix, fbeta_score
import time
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

print("=== STAGE 1: RAM-Optimized Data Loading & Feature Engineering ===")
start_time = time.time()

# 1. Load data and aggressively downcast memory types
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\Cloud Project"
INPUT_FILE = os.path.join(DATA_DIR, "dataset_clean.csv")

def reduce_mem_usage(df):
    """Iterate through all columns and modify data types to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16)
                else: df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory decreased from {start_mem:.2f}MB to {end_mem:.2f}MB")
    return df

df = pd.read_csv(INPUT_FILE)
df = reduce_mem_usage(df)

# Sort by time first to ensure overall longitudinal chronological accuracy for user tracking
if 'time' in df.columns:
    df = df.sort_values(by=['time']).reset_index(drop=True)

# 2. Map Google Borg variables -> Telemedicine Pipeline Variables (FLAW B FIX: Transparent Proxy Disclosure)
# PATENT REQUIREMENT: Explicitly prefix "Proxy_" to separate raw cloud telemetry from mapped healthcare variables.
df['cpu_util_percent'] = df['average_usage_cpus'] * 100
df['mem_util_percent'] = df['average_usage_memory'] * 100

# Proxy Patient Priority based on Borg Job Priority (scaled to 0.1 - 1.0)
max_prio = df['priority'].max() if df['priority'].max() > 0 else 1
df['Proxy_PatientPriority'] = (df['priority'] / max_prio).clip(lower=0.1)

# Proxy Latency (Hardware Bottlenecks = High Latency)
df['Proxy_HardwareLatency'] = (df['cycles_per_instruction'].fillna(0) * 0.05) + (df['maximum_usage_cpus'] * 50)

# Proxy Session Load heavily based on memory accesses
df['Proxy_SessionLoad'] = df['memory_accesses_per_instruction'].fillna(0) + df['mem_util_percent']

# FLAW C FIX: User-Level Volatility and Resubmission Tracking
# The PDF dictates that certain anonymous user collections account for a massive distribution of failures.
# We track their historical fail rates logically through time without cross-contamination or look-ahead bias.
df['SLA_Violation_Current'] = df['failed'].astype(int)

# Group by the anonymous user collection name to track their historical stability
user_col = 'collection_name' if 'collection_name' in df.columns else 'collection_id'
grp_user = df.groupby(user_col)

# How many times has this user/collection interacted with the system so far?
df['User_Session_Count'] = grp_user.cumcount().astype(np.float16)

# How many failures has this user historically caused? (Shifted by 1 so we don't leak the current row's failure)
df['User_Cumulative_Failures'] = grp_user['SLA_Violation_Current'].transform(lambda x: x.shift(1).expanding().sum().fillna(0)).astype(np.float16)

# Volatility Ratio: Failures / Total Sessions (Add 1 to smooth 0-division)
df['User_Resubmission_Volatility'] = (df['User_Cumulative_Failures'] / (df['User_Session_Count'] + 1)).astype(np.float16)

# Now sort by collection_id and time for the physical machine rolling features
if 'collection_id' in df.columns:
    df = df.sort_values(by=['collection_id', 'time']).reset_index(drop=True)

# Target Variable - LEAKAGE FIX & UPGRADE: 

# Use grouping to prevent cross-contamination between different server tasks
grp = df.groupby('collection_id') if 'collection_id' in df.columns else df.groupby('machine_id') if 'machine_id' in df.columns else None

if grp is not None:
    future_1 = grp['SLA_Violation_Current'].shift(-1).fillna(0)
    future_2 = grp['SLA_Violation_Current'].shift(-2).fillna(0)
    future_3 = grp['SLA_Violation_Current'].shift(-3).fillna(0)
else:
    future_1 = df['SLA_Violation_Current'].shift(-1).fillna(0)
    future_2 = df['SLA_Violation_Current'].shift(-2).fillna(0)
    future_3 = df['SLA_Violation_Current'].shift(-3).fillna(0)
    
df['Target_Next_Step_Violation'] = ((future_1 + future_2 + future_3) >= 1).astype(int)

# 3. Rolling Features for Time Series Context
print("Calculating Grouped Rolling Time-Series Features...")
if grp is not None:
    # Switching to Exponential Moving Averages (EMA) to reduce lag and handle micro-bursts
    df['CPU_EMA'] = grp['cpu_util_percent'].transform(lambda x: x.ewm(span=3, adjust=False).mean()).astype(np.float16)
    df['Mem_EMA'] = grp['mem_util_percent'].transform(lambda x: x.ewm(span=3, adjust=False).mean()).astype(np.float16)

    # MACRO-TREND FEATURES (MACD equivalent for Cloud Operations)
    df['CPU_EMA_Long'] = grp['cpu_util_percent'].transform(lambda x: x.ewm(span=15, adjust=False).mean()).astype(np.float16)
    df['Mem_EMA_Long'] = grp['mem_util_percent'].transform(lambda x: x.ewm(span=15, adjust=False).mean()).astype(np.float16)
    df['CPU_MACD'] = (df['CPU_EMA'] - df['CPU_EMA_Long']).astype(np.float16)
    df['Mem_MACD'] = (df['Mem_EMA'] - df['Mem_EMA_Long']).astype(np.float16)

    # Velocity and Acceleration (1st and 2nd derivatives) for Memory Leaks
    df['Mem_Velocity'] = grp['mem_util_percent'].diff().fillna(0).astype(np.float16)
    
    # We must group again for the 2nd derivative to stop cross-contamination
    grp_for_accel = df.groupby('collection_id') if 'collection_id' in df.columns else df.groupby('machine_id')
    df['Mem_Acceleration'] = grp_for_accel['Mem_Velocity'].diff().fillna(0).astype(np.float16)

    # RESEARCH PAPER UPGRADES: CPU Runaway & Memory Deficit Characteristics
    df['CPU_Velocity'] = grp['cpu_util_percent'].diff().fillna(0).astype(np.float16)
    df['CPU_Acceleration'] = grp_for_accel['CPU_Velocity'].diff().fillna(0).astype(np.float16)

    # Duration Proxy (Failed jobs run longer and hotter) -> 15 min cumulative effort
    df['CPU_Sustained_Effort'] = grp['cpu_util_percent'].transform(lambda x: x.rolling(window=15, min_periods=1).sum()).astype(np.float16)

    # RESEARCH PAPER UPGRADE: Peak-to-Average Ratio (PAR) and Burstiness / Stability
    df['CPU_PAR'] = grp['cpu_util_percent'].transform(lambda x: x.rolling(15, min_periods=1).max() / (x.rolling(15, min_periods=1).mean() + 1e-5)).astype(np.float16)
    df['Mem_PAR'] = grp['mem_util_percent'].transform(lambda x: x.rolling(15, min_periods=1).max() / (x.rolling(15, min_periods=1).mean() + 1e-5)).astype(np.float16)
    df['CPU_Volatility_Score'] = grp['cpu_util_percent'].transform(lambda x: x.rolling(15, min_periods=2).std().fillna(0)).astype(np.float16)

    df['Proxy_HardwareLatency_Trend_Slope'] = grp['Proxy_HardwareLatency'].diff().fillna(0).astype(np.float16)
    df['CPU_Lag_1'] = grp['cpu_util_percent'].shift(1).fillna(0).astype(np.float16)
    
else:
    df['CPU_EMA'] = df['cpu_util_percent'].ewm(span=3, adjust=False).mean().astype(np.float16)
    df['Mem_EMA'] = df['mem_util_percent'].ewm(span=3, adjust=False).mean().astype(np.float16)
    df['CPU_EMA_Long'] = df['cpu_util_percent'].ewm(span=15, adjust=False).mean().astype(np.float16)
    df['Mem_EMA_Long'] = df['mem_util_percent'].ewm(span=15, adjust=False).mean().astype(np.float16)
    df['CPU_MACD'] = (df['CPU_EMA'] - df['CPU_EMA_Long']).astype(np.float16)
    df['Mem_MACD'] = (df['Mem_EMA'] - df['Mem_EMA_Long']).astype(np.float16)
    df['Mem_Velocity'] = df['mem_util_percent'].diff().fillna(0).astype(np.float16)
    df['Mem_Acceleration'] = df['Mem_Velocity'].diff().fillna(0).astype(np.float16)
    df['CPU_Velocity'] = df['cpu_util_percent'].diff().fillna(0).astype(np.float16)
    df['CPU_Acceleration'] = df['CPU_Velocity'].diff().fillna(0).astype(np.float16)
    df['CPU_Sustained_Effort'] = df['cpu_util_percent'].rolling(window=15, min_periods=1).sum().astype(np.float16)
    df['Proxy_HardwareLatency_Trend_Slope'] = df['Proxy_HardwareLatency'].diff().fillna(0).astype(np.float16)
    df['CPU_Lag_1'] = df['cpu_util_percent'].shift(1).fillna(0).astype(np.float16)

# 3. CPU/Memory Divergence (Failed jobs use high CPU but low Memory)
df['CPU_to_Mem_Gap'] = (df['CPU_EMA'] - df['Mem_EMA']).astype(np.float16)

# Resource Imbalance Ratio (Healthy = balanced, Pre-Crash = Imbalanced)
# We add +1 instead of +1e-5 to avoid exploding division that causes 'inf' in float16
df['Mem_to_CPU_Ratio'] = (df['mem_util_percent'] / (df['cpu_util_percent'] + 1)).clip(upper=1000).astype(np.float16)

FEATURES = [
    'cpu_util_percent', 'mem_util_percent', 'Proxy_PatientPriority', 'Proxy_HardwareLatency', 
    'Proxy_SessionLoad', 'CPU_EMA', 'Mem_EMA', 'Mem_Velocity', 'Mem_Acceleration', 
    'Mem_to_CPU_Ratio', 'Proxy_HardwareLatency_Trend_Slope', 'CPU_Lag_1',
    'CPU_EMA_Long', 'Mem_EMA_Long', 'CPU_MACD', 'Mem_MACD',
    'CPU_Velocity', 'CPU_Acceleration', 'CPU_Sustained_Effort', 'CPU_to_Mem_Gap',
    'User_Session_Count', 'User_Resubmission_Volatility',
    'CPU_PAR', 'Mem_PAR', 'CPU_Volatility_Score'
]

X = df[FEATURES].fillna(0)
y = df['Target_Next_Step_Violation']

print(f"Data Prep Time: {time.time() - start_time:.2f}s | Rows: {len(X)} | Violations: {y.sum()}")

collections = df['collection_id'] if 'collection_id' in df.columns else df['machine_id']

# Train/Val/Test Split - LEAKAGE FIX: 
# Must use shuffle=False for Time Series temporal split! Random shuffling leaks future data.
X_train, X_temp, y_train, y_temp, coll_train, coll_temp = train_test_split(X, y, collections, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test, coll_val, coll_test = train_test_split(X_temp, y_temp, coll_temp, test_size=0.5, shuffle=False)

print("\n=== STAGE 2: Ultra-Fast XGBoost (CPU Multi-threading) ===")
# OPTIMIZATION: Keep 2 CPU cores free for system stability
total_cores = multiprocessing.cpu_count()
xgb_cores = max(1, total_cores - 2) 
print(f"Assigning {xgb_cores} out of {total_cores} CPU cores to XGBoost to prevent laptop freeze.")

start_time = time.time()

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# TECH 2: XGBoost Custom Focal Loss
def focal_loss_objective(labels, preds):
    p = 1.0 / (1.0 + np.exp(-preds))
    gamma = 2.0
    grad = p * (1 - p)**gamma * (p - labels)
    hess = p * (1 - p)**gamma * (1 - p - gamma * (p - labels) * np.log(p + 1e-9))
    return grad, hess

xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=8, 
    learning_rate=0.175,
    objective=focal_loss_objective,
    tree_method='hist',
    n_jobs=xgb_cores,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_probs_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]
print(f"XGBoost Training Time: {time.time() - start_time:.2f}s")


print("\n=== STAGE 3: RAM-Optimized LSTM (Dynamic Data Loading) ===")
start_time = time.time()

scaler = RobustScaler()
# LEAKAGE FIX: Fit only on training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Scale full dataset for sequential test index matching later, using the training statistics to prevent test leak
X_scaled = scaler.transform(X)

# OPTIMIZATION: Instead of creating a massive 3D matrix (400k x 15 x 8) which eats ~2GB+ RAM,
# we use a PyTorch Dataset to slice the 2D array on the fly.
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, collections, window):
        self.data = torch.tensor(data, dtype=torch.float32)
        label_vals = labels.values if hasattr(labels, 'values') else labels
        self.labels = torch.tensor(label_vals, dtype=torch.float32).unsqueeze(1)
        self.window = window
        
        # AUDIT FIX 5.B: Sequence Cross-Contamination
        # Ensure that no sliding window crosses machine_id/collection_id boundaries
        col_vals = collections.values if hasattr(collections, 'values') else collections
        valid_indices = []
        for i in range(len(self.data) - self.window + 1):
            if col_vals[i] == col_vals[i + self.window - 1]:
                valid_indices.append(i)
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        # AUDIT FIX 5.A: DataLoader Target Misalignment (The "Double Shift")
        # Target Next_Step_Violation is already shifted. We need the label matching the END of the sequence.
        return self.data[real_idx : real_idx + self.window], self.labels[real_idx + self.window - 1]

WINDOW_SIZE = 15
BATCH_SIZE = 256 # Higher batch size because dataloader handles RAM efficiently

# LEAKAGE FIX: Strict temporal isolation for train and test dataset sequences
train_dataset = TimeSeriesDataset(X_train_scaled, y_train, coll_train, WINDOW_SIZE)
val_dataset = TimeSeriesDataset(X_val_scaled, y_val, coll_val, WINDOW_SIZE)
test_dataset = TimeSeriesDataset(X_test_scaled, y_test, coll_test, WINDOW_SIZE)

# Windows handles dataloaders best with num_workers=0 to avoid multiprocessing sync crashes
# Keeping shuffle=True ONLY for train_loader after strict temporal split allows LSTM batch variety without future leakage
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Identify if discrete GPU is available, else fast CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"LSTM Training on: {device}")

class TelemedicineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TelemedicineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return torch.mean(focal_loss)

lstm_model = TelemedicineLSTM(input_size=len(FEATURES), hidden_size=32).to(device)
pos_weight_tensor = torch.tensor([scale_weight], dtype=torch.float32).to(device)
criterion = FocalLoss(pos_weight=pos_weight_tensor, gamma=2.0)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

# Fast 2-Epoch loop (Since dataset is 400k rows, 2 epochs is enough to converge)
lstm_model.train()
for epoch in range(2):
    epoch_start = time.time()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(lstm_model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    print(f"  LSTM Epoch {epoch+1}/2 finished in {time.time() - epoch_start:.2f}s")

# Extract LSTM predictions on the test set specifically aligned to the XGBoost Test Set
print("\nFetching LSTM Probabilities for Test Set...")
lstm_model.eval()

# We take the exact indices used by XGBoost's X_test and recreate sequences just for them.
def get_lstm_probs(indices, df_collections):
    lstm_probs = []
    valid_mask_arr = []
    
    col_vals = df_collections.values if hasattr(df_collections, 'values') else df_collections
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # AUDIT FIX 5.B: Ensure sequence doesn't cross machine_id boundaries
            if idx >= WINDOW_SIZE - 1 and col_vals[idx - WINDOW_SIZE + 1] == col_vals[idx]:
                # AUDIT FIX 5.A: Target Misalignment. 
                # Sequence includes the current row idx, so we slice up to idx + 1.
                seq = X_scaled[idx - WINDOW_SIZE + 1 : idx + 1]
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                logit = lstm_model(seq_tensor)
                prob = torch.sigmoid(logit).cpu().item()
                lstm_probs.append(prob)
                valid_mask_arr.append(True)
            else:
                lstm_probs.append(0.0) # Placeholder
                valid_mask_arr.append(False)
    return np.array(lstm_probs), np.array(valid_mask_arr)

print("\nFetching LSTM Probabilities for Test Sets...")
lstm_model.eval()

global_collections = collections.values if hasattr(collections, 'values') else collections

val_indices = X_val.index.values
lstm_probs_val_raw, val_mask = get_lstm_probs(val_indices, global_collections)

test_indices = X_test.index.values
lstm_probs_test_raw, test_mask = get_lstm_probs(test_indices, global_collections)

print(f"Total LSTM Time: {time.time() - start_time:.2f}s")


print("\n=== STAGE 4: Optimized Soft-Voting Ensemble & Evaluation ===")

# Pad the missing test probabilities (the ones that occurred in the first 15 mins) with XGBoost's guess 
# so arrays align perfectly for evaluation.
lstm_probs_aligned_val = np.zeros(len(xgb_probs_val))
lstm_probs_aligned_val[val_mask] = lstm_probs_val_raw[val_mask]
lstm_probs_aligned_val[~val_mask] = xgb_probs_val[~val_mask]

lstm_probs_aligned_test = np.zeros(len(xgb_probs_test))
lstm_probs_aligned_test[test_mask] = lstm_probs_test_raw[test_mask]
lstm_probs_aligned_test[~test_mask] = xgb_probs_test[~test_mask]

# Soft Voting: Tuned structurally by Walk-Forward + Optuna combination
# 70% tabular strength, 30% temporal trend strength
ensemble_probs_val = (xgb_probs_val * 0.70) + (lstm_probs_aligned_val * 0.30)
ensemble_probs_test = (xgb_probs_test * 0.70) + (lstm_probs_aligned_test * 0.30)

print("\n--- OPTIMIZING PROBABILITY THRESHOLD FOR HIGH RECALL (F2 SCORE) ON VAL SET ---")  
# AUDIT FIX 5.C: Optimization leakage. Optimization happens strictly on the Validation set.
best_threshold = 0.5
best_f2 = 0

for thresh in np.arange(0.05, 0.9, 0.05):
    temp_preds = (ensemble_probs_val > thresh).astype(int)

    # TECH 1: Temporal Hysteresis (Debouncing)
    # Require 2 consecutive ticks above threshold to filter out transient FP spikes
    pred_df = pd.DataFrame({'pred': temp_preds, 'coll': coll_val.values})      
    pred_df['pred_lag'] = pred_df.groupby('coll')['pred'].shift(1).fillna(0).astype(int)
    temp_preds = (pred_df['pred'] & pred_df['pred_lag']).astype(int)

    # F2 Score weights Recall (Catching TP) twice as high as Precision (Avoiding FP)
    temp_f2 = fbeta_score(y_val, temp_preds, beta=2.0)
    if temp_f2 > best_f2:
        best_f2 = temp_f2
        best_threshold = thresh

print(f"Optimal F2-Threshold Found on Validation Set: {best_threshold:.2f} (Val F2: {best_f2:.4f})")

# AUDIT FIX 5.C: Apply Best Threshold strictly to the UNSEEN TEST SET with Hysteresis
final_preds_raw = (ensemble_probs_test > best_threshold).astype(int)
final_pred_df = pd.DataFrame({'pred': final_preds_raw, 'coll': coll_test.values})
final_pred_df['pred_lag'] = final_pred_df.groupby('coll')['pred'].shift(1).fillna(0).astype(int)
ensemble_preds = (final_pred_df['pred'] & final_pred_df['pred_lag']).astype(int)

# --- THE REAL EVALUATION METRICS ON TEST SET ---
auc_roc = roc_auc_score(y_test, ensemble_probs_test)
pr_auc = average_precision_score(y_test, ensemble_probs_test)
f1 = f1_score(y_test, ensemble_preds)
f2 = fbeta_score(y_test, ensemble_preds, beta=2.0)
conf_matrix = confusion_matrix(y_test, ensemble_preds)

print(f"\n--- FINAL MODEL PERFORMANCE ON {len(y_test)} TEST EVENTS ---")
print(f"Ensemble AUC-ROC Score:       {auc_roc:.4f}")
print(f"Ensemble Precision-Recall AUC:{pr_auc:.4f}")
print(f"Ensemble F1-Score:            {f1:.4f}")

print("\nFinal Confusion Matrix:")
print("                 Predicted No Violation  | Predicted Violation")
print(f"Actual No Viol  | TN: {conf_matrix[0][0]:<20} | FP: {conf_matrix[0][1]}")
print(f"Actual Viol     | FN: {conf_matrix[1][0]:<20} | TP: {conf_matrix[1][1]}")

false_alert_rate = conf_matrix[0][1] / (conf_matrix[0][0] + conf_matrix[0][1])
missed_violation_rate = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[1][1])

print(f"\nClinical Impact Summary:")
print(f"-> False Alert Rate: {false_alert_rate*100:.2f}% (Number of times doctors are pinged needlessly)")
print(f"-> Missed Violation: {missed_violation_rate*100:.2f}% (Number of crashes the AI completely missed)")

print("\nFinal Run Completed Successfully.")
