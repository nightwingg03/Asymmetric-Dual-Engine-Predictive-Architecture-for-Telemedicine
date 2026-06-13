import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score, confusion_matrix
import shap

import warnings
warnings.filterwarnings('ignore')

# 1. Load the processed dataset
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\14564935"
INPUT_FILE = os.path.join(DATA_DIR, "telemedicine_processed_data.csv")

print("1. Loading Telemedicine metrics data...")
df = pd.read_csv(INPUT_FILE)

# Define Features and Target
TARGET = 'SLA_Violation'
FEATURES = [
    'cpu_util_percent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent',
    'SessionLoad', 'ActiveConsultations', 'PatientPriority', 'Latency',
    'CPU_Rolling_Mean', 'CPU_Rolling_Std', 'Latency_Trend_Slope',
    'Session_Density_Ratio', 'CPU_Lag_1', 'CPU_Lag_2', 'CPU_Lag_3', 'TimeOfDay'
]

X = df[FEATURES]
y = df[TARGET]

# Split into Train and Test (Stratify is important for severe class imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} rows (Violations: {y_train.sum()})")
print(f"Testing set: {X_test.shape[0]} rows (Violations: {y_test.sum()})")

# 2. Handle Class Imbalance using SMOTE (Only on Training Data)
print("\n2. Applying SMOTE to handle rare SLA Violations...")
# To apply SMOTE we need at least k_neighbors=5 by default.
# If violation count is very low (e.g. 8), we dynamically adjust k_neighbors.
n_violations = y_train.sum()
k_neighbors = min(5, n_violations - 1) if n_violations > 1 else 1

if n_violations > 1:
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k_neighbors) # Bring minority class to 10% of majority
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE Training set: {X_train_resampled.shape[0]} rows (Violations: {y_train_resampled.sum()})")
else:
    print("Not enough violations to apply SMOTE correctly. Proceeding with scale_pos_weight alone.")
    X_train_resampled, y_train_resampled = X_train, y_train


# 3. Train Tabular Baseline Model (Stage 1: XGBoost)
print("\n3. Training XGBoost Model...")
# Calculate scale_pos_weight to handle the remaining imbalance
majority_class = (y_train_resampled == 0).sum()
minority_class = (y_train_resampled == 1).sum()
scale_weight = majority_class / minority_class if minority_class > 0 else 1.0

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    objective='binary:logistic',
    eval_metric='aucpr',
    random_state=42
)

xgb_model.fit(X_train_resampled, y_train_resampled)


# 4. Evaluation 
print("\n4. Evaluating Model Metrics...")
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Primary Metrics
auc_roc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"AUC-ROC Score: {auc_roc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nConfusion Matrix:")
print("                 Predicted No Violation  | Predicted Violation")
print(f"Actual No Viol  | TN: {conf_matrix[0][0]:<20} | FP: {conf_matrix[0][1]}")
if len(conf_matrix) > 1 and len(conf_matrix[0]) > 1:
    print(f"Actual Viol     | FN: {conf_matrix[1][0]:<20} | TP: {conf_matrix[1][1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Note: False alert rate = FP / (FP + TN)
false_alert_rate = conf_matrix[0][1] / (conf_matrix[0][0] + conf_matrix[0][1])
missed_violation_rate = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[1][1]) if len(conf_matrix) > 1 else 0

print(f"\nClinical Impact Metrics:")
print(f"False Alert Rate (Disrupts workflow): {false_alert_rate*100:.2f}%")
print(f"Missed Violation Rate (Safety hazard): {missed_violation_rate*100:.2f}%")

print("\n5. Applying SHAP for Interpretability...")
# Get feature importances 
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# Calculate mean absolute SHAP values for feature importance
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({'Feature': FEATURES, 'SHAP Value': mean_shap})
shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)

print("\nTop 5 Drivers of SLA Violations (SHAP Explainability):")
print(shap_df.head())
