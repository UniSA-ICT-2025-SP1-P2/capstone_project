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

def fix_labels(df, original_path):
    # Rename Category → label if needed
    if 'label' not in df.columns and 'Category' in df.columns:
        df = df.rename(columns={'Category': 'label'})

    if 'label' not in df.columns:
        raise ValueError("No 'label' or 'Category' column found.")

    # Extract second part of label string if it contains a dash
    def clean_label(val):
        val = str(val)
        if '-' in val:
            parts = val.split('-')
            return parts[1] if len(parts) > 1 else None
        else:
            return val

    df['label'] = df['label'].apply(clean_label)

    # Check for missing labels after cleaning
    if df['label'].isnull().any():
        bad_rows = df[df['label'].isnull()]
        print("Invalid label format in the following rows:")
        print(bad_rows.head())
        raise ValueError("Some labels could not be split properly or are missing.")

    # Save cleaned version with _label.csv suffix
    directory = os.path.dirname(original_path)
    base = os.path.basename(original_path).replace('.csv', '')
    output_path = os.path.join(directory, f"{base}_label.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned and saved: {output_path}")

    return df

def run_adversarial_training(data_dir, model_dir, epochs=10, lr=0.001):
    os.makedirs(model_dir, exist_ok=True)

    # Load and clean datasets
    clean_path = os.path.join(data_dir, 'uploaded_dataset.csv')
    fgsm_path = os.path.join(data_dir, 'adversarial_fgsm.csv')
    pgd_path = os.path.join(data_dir, 'adversarial_pgd.csv')

    df_clean = pd.read_csv(clean_path)
    df_clean = fix_labels(df_clean, clean_path)

    df_fgsm = pd.read_csv(fgsm_path)
    df_fgsm = fix_labels(df_fgsm, fgsm_path)

    df_pgd = pd.read_csv(pgd_path)
    df_pgd = fix_labels(df_pgd, pgd_path)

    # Prepare training data
    X_clean = df_clean.drop(columns=['label']).values
    X_fgsm = df_fgsm.drop(columns=['label']).values
    X_pgd = df_pgd.drop(columns=['label']).values

    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    y_clean = label_encoder.transform(df_clean['label'])
    y_fgsm = label_encoder.transform(df_fgsm['label'])
    y_pgd = label_encoder.transform(df_pgd['label'])

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
    print("Adversarially trained model saved as 'neural_net_adv.pt'")
    return True

# CLI fallback
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))                           # /src/defences
    SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))                         # /src
    DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))            # /defence_prototype
    PROJECT_ROOT = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))       # /capstone_project

    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')

    run_adversarial_training(DATA_DIR, MODEL_DIR)
