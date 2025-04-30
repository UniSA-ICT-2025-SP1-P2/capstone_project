# %%
import torch
import torch.nn as nn
import joblib
import numpy as np

# %%
# === Define the Same Neural Network Architecture ===
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

# === Load Saved Models ===
rf = joblib.load('../../models/random_forest.pkl')

input_dim = rf.n_features_in_  # Get feature size from Random Forest model
num_classes = len(rf.classes_)  # Get number of classes

nn_model = SimpleNN(input_dim=input_dim, output_dim=num_classes)
nn_model.load_state_dict(torch.load('../../models/neural_net.pt'))
nn_model.eval()

# === Predict Function ===
def predict(X, model_type='ensemble'):
    if model_type == 'rf':
        preds = rf.predict(X)
        return preds

    elif model_type == 'nn':
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = nn_model(X_tensor)
            preds = torch.argmax(outputs, dim=1).numpy()
        return preds

    elif model_type == 'ensemble':
        rf_probs = rf.predict_proba(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            nn_logits = nn_model(X_tensor)
            nn_probs = torch.softmax(nn_logits, dim=1).numpy()

        avg_probs = (rf_probs + nn_probs) / 2
        preds = np.argmax(avg_probs, axis=1)
        return preds

    else:
        raise ValueError("model_type must be 'rf', 'nn', or 'ensemble'")
