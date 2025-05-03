import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib

# Import from parent folder
from predict import predict

# === Base directory setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_concept_drift(data_path, model_type='ensemble', chunk_size=500, drift_threshold=0.7, output_path=None):
    # Load Test Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['label']).values
    y_true = df['label'].values

    # Load label encoder
    label_encoder_path = os.path.join(BASE_DIR, '..', 'models', 'label_encoder.pkl')
    label_encoder = joblib.load(label_encoder_path)

    y_true_encoded = label_encoder.transform(y_true)

    # === Chunk Data and Evaluate ===
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
        if acc < drift_threshold:
            print(f"Drift Alert at Chunk {idx+1}: Accuracy {acc:.4f} below threshold {drift_threshold}")

    # === Plot Accuracy Over Time ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracy_per_chunk) + 1), accuracy_per_chunk, marker='o')
    plt.axhline(y=drift_threshold, color='r', linestyle='--', label='Drift Threshold')
    plt.xlabel("Chunk (Simulated Time Step)")
    plt.ylabel("Accuracy")
    plt.title(f"Concept Drift Detection - {model_type.upper()} Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Save Chunk Accuracies ===
    results_df = pd.DataFrame({
        'chunk': range(1, len(accuracy_per_chunk) + 1),
        'accuracy': accuracy_per_chunk
    })

    if not output_path:
        output_path = os.path.join(BASE_DIR, '..', 'results', 'concept_drift_results.csv')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Drift detection results saved to {output_path}")


# === CLI usage ===
if __name__ == "__main__":
    default_data_path = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'test_label.csv'))
    run_concept_drift(
        data_path=default_data_path,
        model_type='ensemble',
        chunk_size=500,
        drift_threshold=0.7
    )
