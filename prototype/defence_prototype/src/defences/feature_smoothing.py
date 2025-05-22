import os
import numpy as np
import pandas as pd

def apply_feature_smoothing(X, noise_std=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X + noise

def apply_feature_smoothing_path(input_path, noise_std, output_path):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['label']).values
    y = df['label'].values

    X_smooth = apply_feature_smoothing(X, noise_std=noise_std)
    smoothed_df = pd.DataFrame(X_smooth, columns=df.columns[:-1])
    smoothed_df['label'] = y
    smoothed_df.to_csv(output_path, index=False)

    print(f"✅ Smoothed test data saved to: {output_path}")
    return smoothed_df

# Run when script is executed directly
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))  # capstone_project/
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    input_path = os.path.join(DATA_DIR, 'uploaded_dataset.csv')
    output_path = os.path.join(DATA_DIR, 'uploaded_dataset_smoothed.csv')

    apply_feature_smoothing_path(
        input_path=input_path,
        noise_std=0.02,
        output_path=output_path
    )
