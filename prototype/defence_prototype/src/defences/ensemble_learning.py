# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# %%
# Load data
df = pd.read_csv('../../../data/validation_label.csv')
X = df.drop(columns=['label']).values
y_true = df['label'].values

# Load label encoder
label_encoder = joblib.load('../../models/label_encoder.pkl')
y_true_encoded = label_encoder.transform(y_true)

# Load Random Forest
rf = joblib.load('../../models/random_forest.pkl')
rf_probs = rf.predict_proba(X)

# %%
# Define and load Neural Network
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

input_dim = X.shape[1]
num_classes = rf_probs.shape[1]
nn_model = SimpleNN(input_dim, num_classes)
nn_model.load_state_dict(torch.load('../../models/neural_net.pt'))
nn_model.eval()

# %%
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    nn_logits = nn_model(X_tensor)
    nn_probs = torch.softmax(nn_logits, dim=1).numpy()

# Combine probabilities
ensemble_probs = (rf_probs + nn_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Evaluate
acc = accuracy_score(y_true_encoded, ensemble_preds)
f1 = f1_score(y_true_encoded, ensemble_preds, average='weighted')

print(f"Ensemble Accuracy: {acc:.4f}")
print(f"Ensemble F1 Score: {f1:.4f}")

# %%
