import os
import joblib
import torch
import torch.nn as nn
import numpy as np

# === Set up clean base paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # /src/
DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))    # /defence_prototype/
MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')

# === Load Random Forest model ===
rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))

# === Define and load the neural network ===
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

input_dim = rf.n_features_in_
num_classes = len(rf.classes_)

nn_model = SimpleNN(input_dim=input_dim, output_dim=num_classes)
nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'neural_net.pt')))
nn_model.eval()

# === Prediction function ===
def predict(X, model_type='ensemble'):
    if model_type == 'rf':
        return rf.predict(X)

    elif model_type == 'nn':
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = nn_model(X_tensor)
            return torch.argmax(outputs, dim=1).numpy()

    elif model_type == 'ensemble':
        rf_probs = rf.predict_proba(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            nn_probs = torch.softmax(nn_model(X_tensor), dim=1).numpy()
        avg_probs = (rf_probs + nn_probs) / 2
        return np.argmax(avg_probs, axis=1)

    else:
        raise ValueError("model_type must be 'rf', 'nn', or 'ensemble'")
