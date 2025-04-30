import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import joblib

# === Path setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                          # /src/defences
DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))         # /src/
PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))    # /prototype/

DATA_DIR = os.path.join(PROTOTYPE_DIR, 'data')
MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load clean training data ===
df_clean = pd.read_csv(os.path.join(DATA_DIR, 'train_label.csv'))
X_clean = df_clean.drop(columns=['label']).values
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
y_clean = label_encoder.transform(df_clean['label'])

# === Load adversarial data ===
df_fgsm = pd.read_csv(os.path.join(DATA_DIR, 'adversarial_fgsm.csv'))
df_pgd = pd.read_csv(os.path.join(DATA_DIR, 'adversarial_pgd.csv'))

X_fgsm = df_fgsm.drop(columns=['label']).values
X_pgd = df_pgd.drop(columns=['label']).values
y_fgsm = label_encoder.transform(df_fgsm['label'])
y_pgd = label_encoder.transform(df_pgd['label'])

# === Combine all training data ===
X_all = np.vstack([X_clean, X_fgsm, X_pgd])
y_all = np.concatenate([y_clean, y_fgsm, y_pgd])

X_tensor = torch.tensor(X_all, dtype=torch.float32)
y_tensor = torch.tensor(y_all, dtype=torch.long)

input_dim = X_all.shape[1]
num_classes = len(np.unique(y_all))

# === Model definition ===
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

# === Load original model ===
model = SimpleNN(input_dim=input_dim, output_dim=num_classes)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'neural_net.pt')))
model.train()

# === Train on combined dataset ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # short fine-tuning
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

# === Save updated model ===
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'neural_net_adv.pt'))
print("Adversarially trained model saved as 'neural_net_adv.pt'")
