# %%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
os.chdir(os.path.dirname(__file__))

from predict import predict
import joblib

# %%
# === Paths ===
DATA_DIR = '../../../data'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
# === Files to Evaluate ===
datasets = {
    'clean': 'test_label.csv',
    # 'fgsm': 'adversarial_fgsm.csv',
    # 'pgd': 'adversarial_pgd.csv',
    'clean_smoothed': 'test_label_smoothed.csv',
}

# === Models to Evaluate ===
model_types = ['rf', 'nn', 'ensemble']  # you can later add 'nn_adv' if retrained with adversarial examples

# === Load Label Encoder ===
label_encoder = joblib.load('../../models/label_encoder.pkl')

# %%
# === Evaluation Function ===
def evaluate(X, y_true_encoded, model_type, defence_name, data_type):
    y_pred = predict(X, model_type=model_type)
    
    acc = accuracy_score(y_true_encoded, y_pred)
    f1 = f1_score(y_true_encoded, y_pred, average='weighted')
    
    return {
        'model': model_type,
        'defence': defence_name,
        'data_type': data_type,
        'accuracy': acc,
        'f1_score': f1,
    }

# === Main Evaluation Loop ===
all_results = []

for data_type, filename in datasets.items():
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    X = df.drop(columns=['label']).values
    y_true = df['label'].values
    y_true_encoded = label_encoder.transform(y_true)

    for model_type in model_types:
        # --- 1. Evaluate without smoothing ---
        result = evaluate(X, y_true_encoded, model_type=model_type, defence_name='none', data_type=data_type)
        all_results.append(result)

        # --- 2. Optionally apply smoothing before predicting ---
        if data_type in ['fgsm', 'pgd', 'clean']:  # Only smooth non-smoothed data
            X_smoothed = X + np.random.normal(0, 0.01, size=X.shape)
            result_smooth = evaluate(X_smoothed, y_true_encoded, model_type=model_type, defence_name='feature_smoothing', data_type=data_type+'_smoothed')
            all_results.append(result_smooth)

# === Save All Results ===
results_df = pd.DataFrame(all_results)
results_path = os.path.join(RESULTS_DIR, 'defence_results.csv')
results_df.to_csv(results_path, index=False)

print("Defence evaluations completed and saved.")
# %%
