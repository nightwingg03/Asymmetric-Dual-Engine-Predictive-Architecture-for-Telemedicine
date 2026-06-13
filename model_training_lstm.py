import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Configuration and hyperparameters
DATA_DIR = r"c:\Users\harsh\OneDrive\Desktop\14564935"
INPUT_FILE = os.path.join(DATA_DIR, "telemedicine_processed_data.csv")
WINDOW_SIZE = 15 # 15-step sliding window (75 minutes of history)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

print("1. Loading and preparing time-series data for LSTM...")
df = pd.read_csv(INPUT_FILE)

# Define features used in time-series
TARGET = 'SLA_Violation'
FEATURES = [
    'cpu_util_percent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent',
    'SessionLoad', 'ActiveConsultations', 'PatientPriority', 'Latency',
    'CPU_Rolling_Mean', 'CPU_Rolling_Std', 'Latency_Trend_Slope',
    'Session_Density_Ratio'
]

# Scale features (LSTM requires normalized inputs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
y_labels = df[TARGET].values

# 2. Sequencing Data for LSTM (Sliding Window)
def create_sequences(data, labels, window_size):
    seqs, seq_labels = [], []
    for i in range(len(data) - window_size):
        seqs.append(data[i : i + window_size])
        seq_labels.append(labels[i + window_size]) # Predict the very next step's SLA
    return np.array(seqs), np.array(seq_labels)

X_seq, y_seq = create_sequences(X_scaled, y_labels, WINDOW_SIZE)

print(f"Generated {len(X_seq)} sequences of shape {X_seq.shape[1:]} (Window size: {WINDOW_SIZE}, Features: {X_seq.shape[2]})")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# Calculate class weights for PyTorch loss function to handle severe imbalance
num_non_violations = (y_train == 0).sum()
num_violations = (y_train == 1).sum()
pos_weight = torch.tensor([num_non_violations / (num_violations + 1e-5)], dtype=torch.float32)

print(f"Training set: {X_train.shape[0]} sequences (Violations: {y_train.sum()})")
print(f"Testing set: {X_test.shape[0]} sequences (Violations: {y_test.sum()})")

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# 4. Define the LSTM Model
class TelemedicineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TelemedicineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1) # Output for Binary Classification

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, (hn, cn) = self.lstm(x)
        # Get the output from the last time step
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = TelemedicineLSTM(input_size=len(FEATURES), hidden_size=64, num_layers=2)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Handles Imbalance natively via weighting
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
print("\n2. Training LSTM Model...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} -> Time-Series Loss: {epoch_loss/len(train_loader):.4f}")

# 6. Evaluation Loop
print("\n3. Evaluating LSTM Model on Test Sequences...")
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        probs = torch.sigmoid(outputs) # Convert logits to probabilities
        preds = (probs > 0.5).float() # Threshold at 0.5
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())

y_pred = np.array(all_preds).squeeze()
y_pred_probs = np.array(all_probs).squeeze()

auc_roc = roc_auc_score(y_test, y_pred_probs)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"LSTM AUC-ROC Score: {auc_roc:.4f}")
print("\nConfusion Matrix:")
print("                 Predicted No Violation  | Predicted Violation")
print(f"Actual No Viol  | TN: {conf_matrix[0][0]:<20} | FP: {conf_matrix[0][1]}")
if len(conf_matrix) > 1 and len(conf_matrix[0]) > 1:
     print(f"Actual Viol     | FN: {conf_matrix[1][0]:<20} | TP: {conf_matrix[1][1]}")

print("\nClassification Report (Sequential Pattern Learning):")
print(classification_report(y_test, y_pred))

# Save the predictions so we can ensemble them with XGBoost later
print("\n4. Saving LSTM predictions for Ensembling...")
np.save(os.path.join(DATA_DIR, "lstm_test_probs.npy"), y_pred_probs)
print("LSTM processing complete!")