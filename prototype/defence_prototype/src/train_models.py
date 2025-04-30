#  %%
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib



#  %%
# === Paths ===
# Get the directory *this file* is in (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                    # /src/defences/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))  # /prototype/

DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'train_label.csv')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Final resolved data path:", DATA_PATH)

# === Load and Preprocess Data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label']).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])  # convert to numeric labels

# Save label encoder for decoding predictions later
joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# === Train/Test Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
num_classes = len(np.unique(y))

#  %%
# === Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.pkl'))

#  %%
# === Define and Train Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # Output layer for multi-class
        )

    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
model = SimpleNN(input_dim=input_dim, output_dim=num_classes)

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)  # long for CrossEntropyLoss

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

# Save trained neural net
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'neural_net.pt'))

print("✅ Models trained and saved.")

# %%
