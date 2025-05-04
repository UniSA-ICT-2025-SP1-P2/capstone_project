import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib
from predict import predict

# Setup constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # /src/
DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))    # /defence_prototype/
PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))
RESULTS_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate(X, y_true_encoded, model_type, defence_name, data_type, label_encoder, results_dir):
    y_pred = predict(X, model_type=model_type)

    # Save numeric predictions
    pred_df = pd.DataFrame({
        'true_label': y_true_encoded,
        'predicted': y_pred
    })
    filename = f"{model_type}_{data_type}_{defence_name}_predictions.csv"
    pred_df.to_csv(os.path.join(results_dir, filename), index=False)

    # Save decoded predictions
    decoded_df = pred_df.copy()
    decoded_df['true_label_decoded'] = label_encoder.inverse_transform(pred_df['true_label'])
    decoded_df['predicted_decoded'] = label_encoder.inverse_transform(pred_df['predicted'])
    decoded_filename = filename.replace(".csv", "_decoded.csv")
    decoded_df.to_csv(os.path.join(results_dir, decoded_filename), index=False)

    acc = accuracy_score(y_true_encoded, y_pred)
    f1 = f1_score(y_true_encoded, y_pred, average='weighted')
    return {
        'model': model_type,
        'defence': defence_name,
        'data_type': data_type,
        'accuracy': acc,
        'f1_score': f1,
    }

def run_evaluation(
    data_files,
    model_types=['rf', 'nn', 'ensemble'],
    defences=['feature_smoothing'],
    results_dir=RESULTS_DIR,
    model_dir=os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')
):
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    all_results = []

    for data_type, path in data_files.items():
        df = pd.read_csv(path)
        X = df.drop(columns=['label']).values
        y_true = label_encoder.transform(df['label'].values)

        for model_type in model_types:
            result = evaluate(X, y_true, model_type, defence_name='none', data_type=data_type, label_encoder=label_encoder, results_dir=results_dir)
            all_results.append(result)

            if 'feature_smoothing' in defences:
                X_smoothed = X + np.random.normal(0, 0.01, size=X.shape)
                result_smooth = evaluate(X_smoothed, y_true, model_type, defence_name='feature_smoothing', data_type=f"{data_type}_smoothed", label_encoder=label_encoder, results_dir=results_dir)
                all_results.append(result_smooth)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, 'defence_results.csv'), index=False)
    print("Saved to:", os.path.join(results_dir, 'defence_results.csv'))
    print(results_df.head())
    return results_df

# Optional CLI entry
if __name__ == "__main__":
    DATA_DIR = os.path.join(PROTOTYPE_DIR, 'data')
    data_files = {
        'clean': os.path.join(DATA_DIR, 'test_label.csv'),
        # 'fgsm': os.path.join(DATA_DIR, 'adversarial_fgsm.csv'),
        # 'pgd': os.path.join(DATA_DIR, 'adversarial_pgd.csv'),
    }
    run_evaluation(data_files)
