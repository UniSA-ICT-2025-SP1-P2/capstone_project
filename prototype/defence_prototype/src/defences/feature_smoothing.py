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

    print(f"Smoothed test data saved to: {output_path}")
    return smoothed_df

# Example usage
if __name__ == "__main__":
    apply_feature_smoothing_path(
        input_path='../../../data/test_label.csv',
        noise_std=0.02,
        output_path='../../../data/test_label_smoothed.csv'
    )
