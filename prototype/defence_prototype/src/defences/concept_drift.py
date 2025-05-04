import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib
from predict import predict


def run_concept_drift(data_path, model_type, chunk_size, threshold, results_dir):
    # === Load Data ===
    df = pd.read_csv(data_path)
    X = df.drop(columns=['label']).values
    y_true = df['label'].values

    # === Load label encoder ===
    label_encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'label_encoder.pkl'))

    print("Resolved label encoder path:", label_encoder_path)
    label_encoder = joblib.load(label_encoder_path)
    y_true_encoded = label_encoder.transform(y_true)

    # === Evaluate in chunks ===
    accuracy_per_chunk = []
    chunk_starts = list(range(0, len(X), chunk_size))

    for idx, start in enumerate(chunk_starts):
        end = start + chunk_size
        X_chunk = X[start:end]
        y_chunk = y_true_encoded[start:end]

        if len(X_chunk) == 0:
            break

        y_pred_chunk = predict(X_chunk, model_type=model_type)
        acc = accuracy_score(y_chunk, y_pred_chunk)
        accuracy_per_chunk.append(acc)

        print(f"Chunk {idx+1}: Accuracy = {acc:.4f}")
        if acc < threshold:
            print(f"Drift Alert at Chunk {idx+1}: Accuracy {acc:.4f} below threshold {threshold}")

    # === Plot Accuracy Over Time ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracy_per_chunk)+1), accuracy_per_chunk, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Drift Threshold')
    plt.xlabel("Chunk (Simulated Time Step)")
    plt.ylabel("Accuracy")
    plt.title(f"Concept Drift Detection - {model_type.upper()} Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Save Results ===
    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'chunk': range(1, len(accuracy_per_chunk)+1),
        'accuracy': accuracy_per_chunk
    })
    results_df.to_csv(os.path.join(results_dir, 'concept_drift_results.csv'), index=False)

    print("Drift detection results saved.")


# === CLI fallback ===
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))                          # /src/defences/
    DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))         # /src/
    PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))    # /defence_prototype/

    data_path = os.path.join(PROTOTYPE_DIR, 'data', 'test_label.csv')
    results_dir = os.path.join(DEFENCE_PROTOTYPE_DIR, 'results')

    run_concept_drift(
        data_path=data_path,
        model_type='ensemble',
        chunk_size=500,
        threshold=0.7,
        results_dir=results_dir
    )
