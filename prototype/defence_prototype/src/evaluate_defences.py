import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from predict import predict
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # /src/
DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))    # /defence_prototype/
PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))

DATA_DIR = os.path.join(PROTOTYPE_DIR, 'data')
RESULTS_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = {
    'clean': 'test_label.csv',
    # 'fgsm': 'adversarial_fgsm.csv',
    # 'pgd': 'adversarial_pgd.csv',
}

model_types = ['rf', 'nn', 'ensemble']

label_encoder = joblib.load(os.path.join(DEFENCE_PROTOTYPE_DIR, 'models', 'label_encoder.pkl'))

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

all_results = []

for data_type, filename in datasets.items():
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    X = df.drop(columns=['label']).values
    y_true = label_encoder.transform(df['label'].values)

    for model_type in model_types:
        result = evaluate(X, y_true, model_type=model_type, defence_name='none', data_type=data_type)
        all_results.append(result)

        if data_type in ['clean', 'fgsm', 'pgd']:
            X_smoothed = X + np.random.normal(0, 0.01, size=X.shape)
            result_smooth = evaluate(X_smoothed, y_true, model_type=model_type, defence_name='feature_smoothing', data_type=f"{data_type}_smoothed")
            all_results.append(result_smooth)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'defence_results.csv'), index=False)

print("Defence evaluations completed and saved.")

print("Saving to:", os.path.join(RESULTS_DIR, 'defence_results.csv'))
print("Results preview:\n", results_df.head())