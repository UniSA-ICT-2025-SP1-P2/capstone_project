import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import LabelEncoder

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

def run_adversarial_training(data_dir, model_dir, epochs=10, lr=0.001):
    os.makedirs(model_dir, exist_ok=True)

    # Load clean data
    df_clean = pd.read_csv(os.path.join(data_dir, 'train_label.csv'))
    X_clean = df_clean.drop(columns=['label']).values
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    y_clean = label_encoder.transform(df_clean['label'])

    # Load adversarial data
    df_fgsm = pd.read_csv(os.path.join(data_dir, 'adversarial_fgsm.csv'))
    df_pgd = pd.read_csv(os.path.join(data_dir, 'adversarial_pgd.csv'))
    X_fgsm = df_fgsm.drop(columns=['label']).values
    X_pgd = df_pgd.drop(columns=['label']).values
    y_fgsm = label_encoder.transform(df_fgsm['label'])
    y_pgd = label_encoder.transform(df_pgd['label'])

    # Combine all data
    X_all = np.vstack([X_clean, X_fgsm, X_pgd])
    y_all = np.concatenate([y_clean, y_fgsm, y_pgd])
    input_dim = X_all.shape[1]
    num_classes = len(np.unique(y_all))

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)

    # Load and fine-tune model
    model = SimpleNN(input_dim, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'neural_net.pt')))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, 'neural_net_adv.pt'))
    print("✅ Adversarially trained model saved as 'neural_net_adv.pt'")
    return True

# CLI fallback
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))                           # /src/defences/
    DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))          # /src/
    PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))     # /prototype/

    DATA_DIR = os.path.join(PROTOTYPE_DIR, 'data')
    MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')

    run_adversarial_training(DATA_DIR, MODEL_DIR)
