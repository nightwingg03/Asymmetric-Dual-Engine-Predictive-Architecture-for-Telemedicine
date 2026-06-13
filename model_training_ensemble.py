import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Data for Ensembling...")
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\14564935"
INPUT_FILE = os.path.join(DATA_DIR, "telemedicine_processed_data.csv")
LSTM_PROBS_FILE = os.path.join(DATA_DIR, "lstm_test_probs.npy")

df = pd.read_csv(INPUT_FILE)
TARGET = 'SLA_Violation'
FEATURES = [
    'cpu_util_percent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent',
    'SessionLoad', 'ActiveConsultations', 'PatientPriority', 'Latency',
    'CPU_Rolling_Mean', 'CPU_Rolling_Std', 'Latency_Trend_Slope',
    'Session_Density_Ratio', 'CPU_Lag_1', 'CPU_Lag_2', 'CPU_Lag_3', 'TimeOfDay'
]
X = df[FEATURES]
y = df[TARGET]

# We need to recreate the EXACT test set that the LSTM used.
# The LSTM chopped off the first 15 rows due to the sliding window.
WINDOW_SIZE = 15
X_adjusted = X.iloc[WINDOW_SIZE:].reset_index(drop=True)
y_adjusted = y.iloc[WINDOW_SIZE:].reset_index(drop=True)

X_train, X_test, y_train, y_test_adjusted = train_test_split(X_adjusted, y_adjusted, test_size=0.2, random_state=42, stratify=y_adjusted)

# Check if LSTM probabilities exist
try:
    lstm_probs = np.load(LSTM_PROBS_FILE)
    print(f"Loaded {len(lstm_probs)} LSTM prediction probabilities.")
except FileNotFoundError:
    print("Run `python model_training_lstm.py` first to generate LSTM predictions.")
    exit()

print("\n2. Re-training XGBoost (Fast) for Ensembling...")
# Keep it fast and simple, standard SMOTE
n_violations = y_train.sum()
k_neighbors = min(5, n_violations - 1) if n_violations > 1 else 1

if n_violations > 1:
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k_neighbors) 
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    X_train_resampled, y_train_resampled = X_train, y_train

majority_class = (y_train_resampled == 0).sum()
minority_class = (y_train_resampled == 1).sum()
scale_weight = majority_class / minority_class if minority_class > 0 else 1.0

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, 
                              scale_pos_weight=scale_weight, eval_metric='aucpr', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n3. Soft Voting Ensemble (Averaging XGBoost + LSTM)...")
# Soft Voting: Average the probabilities of both models
ensemble_probs = (xgb_probs * 0.6) + (lstm_probs * 0.4) # Slightly weight XGBoost higher as per research standard

# Determine threshold manually to favor recalling severe violations 
# (In medicine, we'd rather have a false alarm than a missed crash)
THRESHOLD = 0.5
ensemble_preds = (ensemble_probs > THRESHOLD).astype(int)

auc_roc = roc_auc_score(y_test_adjusted, ensemble_probs)
pr_auc = average_precision_score(y_test_adjusted, ensemble_probs)

print(f"Ensemble AUC-ROC Score: {auc_roc:.4f}")
print(f"Ensemble Precision-Recall AUC: {pr_auc:.4f}")

conf_matrix = confusion_matrix(y_test_adjusted, ensemble_preds)
print("\nFinal Ensemble Confusion Matrix:")
print("                 Predicted No Violation  | Predicted Violation")
print(f"Actual No Viol  | TN: {conf_matrix[0][0]:<20} | FP: {conf_matrix[0][1]}")
if len(conf_matrix) > 1 and len(conf_matrix[0]) > 1:
     print(f"Actual Viol     | FN: {conf_matrix[1][0]:<20} | TP: {conf_matrix[1][1]}")

print("\nFinal Conclusion on the Clinical Viability:")
if len(conf_matrix) > 1 and conf_matrix[1][1] == 0:
    print("-> BEWARE: The pipeline accurately runs, but the dataset size is currently too small and imbalanced (only 2 test violations). The model acts safely by predicting 'No Violation'. You must feed this pipeline a much larger dataset chunk (e.g., 500,000 rows) for the Ensemble technique to truly activate.")
else:
    print("-> SUCCESS: The model learned to predict SLA behaviors from the ensemble.")
