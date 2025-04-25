# %%
import numpy as np
import pandas as pd

# %%
def apply_feature_smoothing(X, noise_std=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X + noise

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('../../../data/validation_label.csv')
    X = df.drop(columns=['label']).values
    y = df['label'].values

    X_smooth = apply_feature_smoothing(X, noise_std=0.02)
    smoothed_df = pd.DataFrame(X_smooth, columns=df.columns[:-1])
    smoothed_df['label'] = y
    smoothed_df.to_csv('../../../data/validation_label_smoothed.csv', index=False)

    print("Smoothed validation data saved.")

# %%
