import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import fbeta_score, f1_score, roc_auc_score, confusion_matrix
import optuna
import time
import multiprocessing
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. LOAD & PREP (From Current Optimized Pipeline)
# ==========================================
print("Loading and Engineering Features...")
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\Cloud Project"
INPUT_FILE = os.path.join(DATA_DIR, "dataset_clean.csv")

def reduce_mem_usage(df):
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
    return df

df = pd.read_csv(INPUT_FILE)
df = reduce_mem_usage(df)
if 'time' in df.columns:
    df = df.sort_values(by='time').reset_index(drop=True)

df['cpu_util_percent'] = df['average_usage_cpus'] * 100
df['mem_util_percent'] = df['average_usage_memory'] * 100
df['PatientPriority'] = (df['priority'] / (df['priority'].max() if df['priority'].max() > 0 else 1)).clip(lower=0.1)
df['Latency'] = (df['cycles_per_instruction'].fillna(0) * 0.05) + (df['maximum_usage_cpus'] * 50)
df['SessionLoad'] = df['memory_accesses_per_instruction'].fillna(0) + df['mem_util_percent']

df['SLA_Violation_Current'] = df['failed'].astype(int)
fs = [df['SLA_Violation_Current'].shift(-i).fillna(0) for i in range(1, 4)]
df['Target_Next_Step_Violation'] = (sum(fs) >= 1).astype(int)

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
df['CPU_to_Mem_Gap'] = (df['CPU_EMA'] - df['Mem_EMA']).astype(np.float16)

df['Mem_to_CPU_Ratio'] = (df['mem_util_percent'] / (df['cpu_util_percent'] + 1)).clip(upper=1000).astype(np.float16)
df['Latency_Trend_Slope'] = df['Latency'].diff().fillna(0).astype(np.float16)
df['CPU_Lag_1'] = df['cpu_util_percent'].shift(1).fillna(0).astype(np.float16)

FEATURES = ['cpu_util_percent', 'mem_util_percent', 'PatientPriority', 'Latency', 'SessionLoad', 
            'CPU_EMA', 'Mem_EMA', 'Mem_Velocity', 'Mem_Acceleration', 'Mem_to_CPU_Ratio', 'Latency_Trend_Slope', 
            'CPU_Lag_1', 'CPU_EMA_Long', 'Mem_EMA_Long', 'CPU_MACD', 'Mem_MACD', 'CPU_Velocity', 
            'CPU_Acceleration', 'CPU_Sustained_Effort', 'CPU_to_Mem_Gap']

X = df[FEATURES].values
y = df['Target_Next_Step_Violation'].values
total_cores = multiprocessing.cpu_count()
xgb_cores = max(1, total_cores - 2) 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, window=15):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.window = window
    def __len__(self):
        return len(self.data) - self.window
    def __getitem__(self, idx):
        return self.data[idx : idx + self.window], self.labels[idx + self.window]

class TelemedicineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TelemedicineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])

class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        return torch.mean(((1 - pt) ** self.gamma) * bce_loss)

def train_and_eval(train_idx, test_idx, xgb_params=None, xgb_weight=0.6, return_metrics=True):
    X_tr, y_train = X[train_idx], y[train_idx]
    X_te, y_test = X[test_idx], y[test_idx]
    
    scale_weight_val = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    # XGBoost
    params = xgb_params or {'max_depth': 6, 'learning_rate': 0.3}
    model_xgb = xgb.XGBClassifier(
        n_estimators=50, tree_method='hist', n_jobs=xgb_cores, random_state=42, 
        scale_pos_weight=scale_weight_val, **params
    )
    model_xgb.fit(X_tr, y_train)
    xgb_probs = model_xgb.predict_proba(X_te)[:, 1]
    
    # LSTM
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    train_loader = DataLoader(TimeSeriesDataset(X_tr_sc, y_train), batch_size=512, shuffle=True, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm = TelemedicineLSTM(len(FEATURES), 32).to(device)
    criterion = FocalLoss(torch.tensor([scale_weight_val], dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    
    lstm.train()
    for _ in range(1): # 1 epoch for pure speed in experimental loops
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(lstm(bx), by)
            loss.backward()
            optimizer.step()
            
    lstm.eval()
    valid_mask = np.arange(len(y_test)) >= 15
    valid_indices = np.where(valid_mask)[0]
    lstm_probs_list = []
    
    with torch.no_grad():
        for idx in valid_indices:
            seq = X_te_sc[idx - 15 : idx]
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            lstm_probs_list.append(torch.sigmoid(lstm(seq_t)).item())
            
    lstm_probs_aligned = np.zeros(len(xgb_probs))
    lstm_probs_aligned[valid_mask] = lstm_probs_list
    lstm_probs_aligned[~valid_mask] = xgb_probs[~valid_mask] 
    
    ensemble_probs = (xgb_probs * xgb_weight) + (lstm_probs_aligned * (1 - xgb_weight))
    
    # Find best F2 threshold
    best_t, best_f2 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.1):
        preds = (ensemble_probs > t).astype(int)
        f2 = fbeta_score(y_test, preds, beta=2.0, zero_division=0)
        if f2 > best_f2: best_f2, best_t = f2, t
            
    if not return_metrics: return best_f2
    
    final_preds = (ensemble_probs > best_t).astype(int)
    f1 = f1_score(y_test, final_preds)
    roc = roc_auc_score(y_test, ensemble_probs)
    return best_f2, f1, roc, best_t


print("\n--- TEST 1: WALK-FORWARD VALIDATION (2 Splits) ---")
# Proves model generalizes across time instead of a lucky random shuffle.
tscv = TimeSeriesSplit(n_splits=2)
f2_scores = []
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    f2, f1, roc, t = train_and_eval(train_idx, test_idx)
    f2_scores.append(f2)
    print(f"Fold {i+1} | F2: {f2:.4f} | F1: {f1:.4f} | AUC: {roc:.4f} | Threshold: {t:.1f}")
print(f"-> Mean Walk-Forward F2 Score: {np.mean(f2_scores):.4f}")


print("\n--- TEST 2: OPTUNA HYPERPARAM TUNING (Single Temporal Split) ---")
train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, shuffle=False)

def optuna_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5)
    }
    xgb_weight = trial.suggest_float('xgb_weight', 0.4, 0.9)
    # Return F2 from single split evaluation
    return train_and_eval(train_idx, test_idx, params, xgb_weight, return_metrics=False)

study_single = optuna.create_study(direction='maximize')
study_single.optimize(optuna_objective, n_trials=3)
best_params = study_single.best_params
print(f"Best Trial F2: {study_single.best_value:.4f}")
print(f"Found Parameters: {best_params}")


print("\n--- TEST 3: COMBINED (OPTUNA + WALK-FORWARD) ---")
def combined_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25)
    }
    xgb_weight = trial.suggest_float('xgb_weight', 0.5, 0.8)
    
    cv_f2 = []
    # Use smaller split just for speed of Optuna Combined Trial
    tscv_sm = TimeSeriesSplit(n_splits=2) 
    for tr_idx, te_idx in tscv_sm.split(X):
        cv_f2.append(train_and_eval(tr_idx, te_idx, params, xgb_weight, return_metrics=False))
    return np.mean(cv_f2)

study_combined = optuna.create_study(direction='maximize')
study_combined.optimize(combined_objective, n_trials=3)
print(f"Best Combined Walk-Forward F2: {study_combined.best_value:.4f}")
print(f"Robust Parameters Found: {study_combined.best_params}")
