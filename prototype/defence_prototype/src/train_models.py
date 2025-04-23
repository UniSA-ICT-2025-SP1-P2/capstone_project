# Import packas

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
DATA_PATH = '../data/train.csv'
MODEL_DIR = '../models'

# Load training data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label']).values
y = df['label'].values

# Split for training NN
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save RF model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.pkl'))


# === 2. Train Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Prepare tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

model = SimpleNN(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'neural_net.pt'))

