# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import predict

# %%
# === Parameters ===
DATA_PATH = '../../../data/test_label.csv'  # or validation_label.csv
MODEL_TYPE = 'ensemble'  # 'rf', 'nn', or 'ensemble'
CHUNK_SIZE = 500  # Number of samples per chunk
DRIFT_THRESHOLD = 0.7  # Alert if accuracy falls below 70%

# === Load Test Data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label']).values
y_true = df['label'].values

# Load label encoder
label_encoder = joblib.load('../../models/label_encoder.pkl')

# Encode true labels
y_true_encoded = label_encoder.transform(y_true)

# %%
# === Chunk Data and Evaluate ===
accuracy_per_chunk = []
chunk_starts = list(range(0, len(X), CHUNK_SIZE))

for idx, start in enumerate(chunk_starts):
    end = start + CHUNK_SIZE
    X_chunk = X[start:end]
    y_chunk = y_true[start:end]
    
    if len(X_chunk) == 0:
        break

    y_pred_chunk = predict(X_chunk, model_type=MODEL_TYPE)
    
    acc = accuracy_score(y_true_encoded[start:end], y_pred_chunk)
    accuracy_per_chunk.append(acc)

    print(f"Chunk {idx+1}: Accuracy = {acc:.4f}")

    # Drift Detection
    if acc < DRIFT_THRESHOLD:
        print(f"Drift Alert at Chunk {idx+1}: Accuracy {acc:.4f} below threshold {DRIFT_THRESHOLD}")

# %%
# === Plot Accuracy Over Time ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracy_per_chunk)+1), accuracy_per_chunk, marker='o')
plt.axhline(y=DRIFT_THRESHOLD, color='r', linestyle='--', label='Drift Threshold')
plt.xlabel("Chunk (Simulated Time Step)")
plt.ylabel("Accuracy")
plt.title(f"Concept Drift Detection - {MODEL_TYPE.upper()} Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# === Save Chunk Accuracies ===
results_df = pd.DataFrame({
    'chunk': range(1, len(accuracy_per_chunk)+1),
    'accuracy': accuracy_per_chunk
})
results_df.to_csv('../../results/concept_drift_results.csv', index=False)

print("Drift detection results saved.")

# %%
