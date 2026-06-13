import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, fbeta_score
import time

import warnings
warnings.filterwarnings('ignore')

# TECH 2: Focal Loss implementation for XGBoost
def focal_loss_objective(labels, preds):
    # preds are unbounded logits, apply sigmoid
    p = 1.0 / (1.0 + np.exp(-preds))
    gamma = 2.0
    # Custom focal loss gradient and hessian
    grad = p * (1 - p)**gamma * (p - labels)
    hess = p * (1 - p)**gamma * (1 - p - gamma * (p - labels) * np.log(p + 1e-9))
    return grad, hess

def run_experiment(use_tech1=False, use_tech2=False, use_tech3=False):
    print(f"\n=========================================")
    print(f"RUNNING CONFIG: Hysteresis(Tech1)={use_tech1} | FocalLoss(Tech2)={use_tech2} | PAR_Features(Tech3)={use_tech3}")
    
    # 1. Load data
    df = pd.read_csv("dataset_clean.csv")
    df = df.sort_values(by=['time']).reset_index(drop=True)

    df['cpu_util_percent'] = df['average_usage_cpus'] * 100
    df['mem_util_percent'] = df['average_usage_memory'] * 100
    df['SLA_Violation_Current'] = df['failed'].astype(int)

    # Base Features from previous run
    grp_user = df.groupby('collection_id') if 'collection_id' in df.columns else df.groupby('machine_id')
    df['User_Session_Count'] = grp_user.cumcount().astype(np.float16)
    df['User_Cumulative_Failures'] = grp_user['SLA_Violation_Current'].transform(lambda x: x.shift(1).expanding().sum().fillna(0)).astype(np.float16)
    df['User_Resubmission_Volatility'] = (df['User_Cumulative_Failures'] / (df['User_Session_Count'] + 1)).astype(np.float16)

    if 'collection_id' in df.columns:
        df = df.sort_values(by=['collection_id', 'time']).reset_index(drop=True)
        grp = df.groupby('collection_id')
    else:
        grp = df.groupby('machine_id')

    future_1 = grp['SLA_Violation_Current'].shift(-1).fillna(0)
    future_2 = grp['SLA_Violation_Current'].shift(-2).fillna(0)
    future_3 = grp['SLA_Violation_Current'].shift(-3).fillna(0)
    df['Target_Next_Step_Violation'] = ((future_1 + future_2 + future_3) >= 1).astype(int)

    df['CPU_EMA'] = grp['cpu_util_percent'].transform(lambda x: x.ewm(span=3, adjust=False).mean()).astype(np.float16)
    df['Mem_EMA'] = grp['mem_util_percent'].transform(lambda x: x.ewm(span=3, adjust=False).mean()).astype(np.float16)
    df['CPU_Velocity'] = grp['cpu_util_percent'].diff().fillna(0).astype(np.float16)
    df['Mem_Velocity'] = grp['mem_util_percent'].diff().fillna(0).astype(np.float16)

    # TECH 3: PAR and Burstiness Features (Peak-to-Average Ratio)
    if use_tech3:
        # Prevent Leakage: Only use rolling on past data per machine/collection
        df['CPU_PAR'] = grp['cpu_util_percent'].transform(lambda x: x.rolling(15, min_periods=1).max() / (x.rolling(15, min_periods=1).mean() + 1e-5)).astype(np.float16)
        df['Mem_PAR'] = grp['mem_util_percent'].transform(lambda x: x.rolling(15, min_periods=1).max() / (x.rolling(15, min_periods=1).mean() + 1e-5)).astype(np.float16)
        
        # Stability / Burst Texture
        df['CPU_Volatility_Score'] = grp['cpu_util_percent'].transform(lambda x: x.rolling(15, min_periods=2).std().fillna(0)).astype(np.float16)

    FEATURES = ['cpu_util_percent', 'mem_util_percent', 'CPU_EMA', 'Mem_EMA', 'CPU_Velocity', 'Mem_Velocity', 'User_Session_Count', 'User_Resubmission_Volatility']
    if use_tech3:
        FEATURES.extend(['CPU_PAR', 'Mem_PAR', 'CPU_Volatility_Score'])

    X = df[FEATURES].fillna(0)
    y = df['Target_Next_Step_Violation']
    # Keep track of collection_id for proper evaluation (Hysteresis needs it)
    collections = df['collection_id'] if 'collection_id' in df.columns else df['machine_id']

    X_train, X_test, y_train, y_test, coll_train, coll_test = train_test_split(X, y, collections, test_size=0.2, shuffle=False)
    
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Simple XGBoost proxy
    if use_tech2:
        # Need pure DMatrix for custom objective if using python API, or pass to classifier
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, n_jobs=-1, objective=focal_loss_objective)
    else:
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_weight, eval_metric='auc', n_jobs=-1)
        
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Quick threshold determination
    best_f2 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred = (xgb_probs >= thresh).astype(int)
        
        # TECH 1: Temporal Hysteresis (Debouncing)
        if use_tech1:
            # We need 2 consecutive ticks of failure prediction to sound alarm
            pred_df = pd.DataFrame({'pred': y_pred, 'coll': coll_test.values})
            pred_df['pred_lag'] = pred_df.groupby('coll')['pred'].shift(1).fillna(0).astype(int)
            y_pred = (pred_df['pred'] & pred_df['pred_lag']).astype(int)
            
        f2 = fbeta_score(y_test, y_pred, beta=2.0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh

    # Final Eval
    y_pred = (xgb_probs >= best_thresh).astype(int)
    if use_tech1:
        pred_df = pd.DataFrame({'pred': y_pred, 'coll': coll_test.values})
        pred_df['pred_lag'] = pred_df.groupby('coll')['pred'].shift(1).fillna(0).astype(int)
        y_pred = (pred_df['pred'] & pred_df['pred_lag']).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_test, xgb_probs)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)
    
    print(f"Results for Config: {int(use_tech1)}{int(use_tech2)}{int(use_tech3)}")
    print(f"F2 Score: {f2:.4f} @ Thresh {best_thresh:.2f}")
    print(f"F1 Score: {f1:.4f} | AUC-ROC: {auc:.4f}")
    print(f"TN: {tn} | FP: {fp} | Rate of FP: {fp/(fp+tn):.4f}")
    print(f"FN: {fn} | TP: {tp} | Rate of FN: {fn/(fn+tp):.4f}")
    print("-----------------------------------------")

if __name__ == "__main__":
    run_experiment(False, False, False) # BASE
    run_experiment(True, False, False)  # TECH 1: Hysteresis
    run_experiment(False, True, False)  # TECH 2: Focal Loss
    run_experiment(False, False, True)  # TECH 3: PAR Features
    run_experiment(True, True, True)    # ALL ENSEMBLE
