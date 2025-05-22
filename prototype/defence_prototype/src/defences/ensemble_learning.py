import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../../data/uploaded_dataset.csv')
model_dir = os.path.join(script_dir, '../../models')

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def run_ensemble_evaluation(data_path, model_dir):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['label']).values
    y_true = df['label'].values

    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    y_true_encoded = label_encoder.transform(y_true)

    rf = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
    rf_probs = rf.predict_proba(X)

    input_dim = X.shape[1]
    num_classes = rf_probs.shape[1]

    nn_model = SimpleNN(input_dim, num_classes)
    nn_model.load_state_dict(torch.load(os.path.join(model_dir, 'neural_net.pt')))
    nn_model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        nn_logits = nn_model(X_tensor)
        nn_probs = torch.softmax(nn_logits, dim=1).numpy()

    ensemble_probs = (rf_probs + nn_probs) / 2
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_true_encoded, ensemble_preds)
    f1 = f1_score(y_true_encoded, ensemble_preds, average='weighted')

    return {
        "accuracy": acc,
        "f1_score": f1,
        "message": "✅ Ensemble evaluation complete."
    }

# Optional CLI usage
if __name__ == "__main__":
    result = run_ensemble_evaluation(os.path.normpath(data_path), os.path.normpath(model_dir))
    print(result)
