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

def train_models(data_path, model_dir, test_size=0.2, lr=0.001, epochs=20):
    os.makedirs(model_dir, exist_ok=True)
    print("Final resolved DATA_PATH:", data_path)

    # === Load and Preprocess Data ===
    df = pd.read_csv(data_path)
    X = df.drop(columns=['label']).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    num_classes = len(np.unique(y))

    # === Train Random Forest ===
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(model_dir, 'random_forest.pkl'))

    # === Train Neural Network ===
    input_dim = X.shape[1]
    model = SimpleNN(input_dim=input_dim, output_dim=num_classes)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, 'neural_net.pt'))
    print("✅ Models trained and saved.")
    return True

# CLI fallback
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
    PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))

    DATA_PATH = os.path.join(PROTOTYPE_DIR, 'data', 'train_label.csv')
    MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')

    train_models(DATA_PATH, MODEL_DIR)
