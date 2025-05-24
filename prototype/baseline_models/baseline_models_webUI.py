# Run the following code to generate baseline classification models on the CIC-MalMem2022.csv dataset:
import os
import shap
import time
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define base paths relative to prototype directory:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASELINE_DATA_DIR = os.path.join(BASE_DIR, "baseline_models", "baseline_data")
MODEL_DIR = os.path.join(BASE_DIR, "baseline_models", "trained_baseline_models")
SHAP_DIR = os.path.join(BASE_DIR, "baseline_models", "shap_charts")

# Define perturbable features (for adversarial generation):
ADVERSARIAL_FEATURES = [
    "svcscan.nservices", "dlllist.avg_dlls_per_proc", "handles.nsection",
    "svcscan.shared_process_services", "handles.nthread", "handles.nmutant",
    "handles.nkey", "handles.nevent", "ldrmodules.not_in_load_avg",
    "ldrmodules.not_in_load", "ldrmodules.not_in_mem", "handles.nsemaphore",
    "handles.nfile", "dlllist.ndlls", "handles.ntimer",
    "handles.avg_handles_per_proc", "ldrmodules.not_in_init_avg",
    "handles.nhandles", "ldrmodules.not_in_init", "ldrmodules.not_in_mem_avg",
    "svcscan.nactive", "handles.ndirectory", "svcscan.process_services",
    "handles.ndesktop", "modules.nmodules"
]

# Utility functions:
def find_category(file_name):
    return file_name.split("-")[0] if "-" in file_name else file_name

def find_category_name(file_name):
    parts = file_name.split("-")
    return parts[1] if len(parts) > 1 else file_name

def extract_unique_file_id(file_name):
    return file_name.rsplit('-', 1)[0]


def run_all_models(df, selected_models=None, progress_callback=None):
    """
    Train baseline classifiers on the provided DataFrame and save models, SHAP charts, and datasets.
    """
    if selected_models is None:
        selected_models = ['RandomForest', 'KNN', 'LogisticRegression', 'DecisionTree', 'SVM']

    def update_progress(msg):
        if progress_callback:
            progress_callback(msg)

    # Extract category fields
    df["category"] = df["Category"].apply(find_category)
    df["category_name"] = df["Category"].apply(find_category_name)
    df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)

    # Label encoding for categories
    le_catname = LabelEncoder()
    df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])
    df['group_id'] = df.apply(
        lambda row: row['unique_file_id'] if row['Class'] != 'Benign' else f"benign_{row.name}",
        axis=1
    )

    # Create train/validation/test splits using group shuffle
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
    val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
    validation_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    def get_xy(data):
        drop_cols = [
            'Category', 'Class', 'category', 'category_name',
            'unique_file_id', 'group_id', 'category_name_encoded'
        ]
        X = data.drop(columns=drop_cols)
        y = data['category_name_encoded']
        return X, y

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(validation_df)
    X_test, y_test = get_xy(test_df)

    # Define classifiers and parameter grids
    classifiers = {
        'RandomForest': (RandomForestClassifier(), False),
        'KNN': (KNeighborsClassifier(), True),
        'LogisticRegression': (LogisticRegression(max_iter=1000), True),
        'DecisionTree': (DecisionTreeClassifier(), False),
        'SVM': (SVC(probability=True), True)
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 75], 'max_depth': [3, 5, 7],
            'min_samples_split': [4, 6], 'min_samples_leaf': [2, 3]
        },
        'KNN': { 'n_neighbors': [7, 9, 11] },
        'LogisticRegression': { 'C': [0.1, 0.5, 1] },
        'DecisionTree': {
            'max_depth': [3, 5], 'min_samples_split': [6, 8], 'min_samples_leaf': [2, 3]
        },
        'SVM': { 'C': [0.1, 0.5, 1], 'kernel': ['rbf'] }
    }

    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SHAP_DIR, exist_ok=True)
    os.makedirs(BASELINE_DATA_DIR, exist_ok=True)

    # Prepare containers
    results = {}
    shap_features_list = []
    metrics_list = []
    conti_encoded = le_catname.transform(["Conti"])[0]

    # Train each selected model
    for clf_name in selected_models:
        clf_obj, scale_required = classifiers[clf_name]
        update_progress(f"Starting training for {clf_name}...")

        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf_obj)]) if scale_required else clf_obj

        if clf_name in param_grids:
            grid = {f"clf__{k}" if scale_required else k: v for k, v in param_grids[clf_name].items()}  
            search = GridSearchCV(
                pipeline, grid, cv=GroupKFold(n_splits=5), scoring='accuracy', n_jobs=-1
            )
            search.fit(X_train, y_train, groups=train_df['group_id'])
            model = search.best_estimator_
            update_progress(f"Completed GridSearch for {clf_name}.")
        else:
            model = pipeline.fit(X_train, y_train) if scale_required else clf_obj.fit(X_train, y_train)

        # Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{clf_name}_model.pkl")
        joblib.dump(model, model_path)

        # Evaluate on test set and collect metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                metrics_list.append({"Classifier": clf_name, "Class": label, **scores})

        model_result = {"model_path": model_path, "report": report}

        # Generate SHAP explanations
        try:
            model_for_shap = model.named_steps['clf'] if scale_required else model
            explainer = shap.Explainer(model_for_shap, X_train)
            shap_vals = explainer(X_test)
            conti_vals = shap_vals.values[:, :, conti_encoded] if shap_vals.values.ndim == 3 else shap_vals.values
            shap_mean = np.abs(conti_vals).mean(axis=0)
            shap_series = pd.Series(shap_mean, index=X_test.columns).sort_values(ascending=False)
            shap_path = os.path.join(SHAP_DIR, f"{clf_name}_SHAP.png")
            shap_series.plot(kind='bar', title=f"SHAP for {clf_name}")
            plt.ylabel("|SHAP|")
            plt.tight_layout()
            plt.savefig(shap_path)
            plt.close()

            for f, v in shap_series.items():
                shap_features_list.append({"Classifier": clf_name, "Feature": f, "SHAP Importance": v})

            model_result["shap_image"] = shap_path
            update_progress(f"SHAP plot saved for {clf_name}.")
        except Exception as e:
            logging.warning(f"SHAP explanation failed for {clf_name}: {e}")
            model_result["shap_image"] = None

        results[clf_name] = model_result

    # Save summaries and datasets
    pd.DataFrame(metrics_list).to_excel(
        os.path.join(BASELINE_DATA_DIR, "Classifier_Results.xlsx"),
        sheet_name="Metrics", index=False
    )
    pd.DataFrame(shap_features_list).to_excel(
        os.path.join(BASELINE_DATA_DIR, "Classifier_Results.xlsx"),
        sheet_name="SHAP_Features", index=False
    )

    train_df.to_csv(os.path.join(BASELINE_DATA_DIR, "Train_Dataset.csv"), index=False)
    validation_df.to_csv(os.path.join(BASELINE_DATA_DIR, "Validation_Dataset.csv"), index=False)
    # Add 'label' column so downstream code can read category_name
    test_df["label"] = test_df["category_name"]
    test_df.to_csv(os.path.join(BASELINE_DATA_DIR, "Test_Dataset.csv"), index=False)

    pd.DataFrame(ADVERSARIAL_FEATURES, columns=["Feature"]).to_csv(
        os.path.join(BASELINE_DATA_DIR, "Adversarial_Features.csv"), index=False
    )
    joblib.dump(le_catname, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

    update_progress("All models trained and saved.")
    return {
        "status": "complete",
        "message": "Models trained and results saved.",
        "adversarial_features": ADVERSARIAL_FEATURES,
        "results": results
    }

def evaluate_models(df, selected_models=None, progress_callback=None):
    """
    Load saved models and evaluate them on a new DataFrame (e.g., adversarial samples).
    Returns a dict of classification reports for each model.
    """
    if selected_models is None:
        selected_models = ['RandomForest', 'KNN', 'LogisticRegression', 'DecisionTree', 'SVM']

    def update_progress(msg):
        if progress_callback:
            progress_callback(msg)

    # Prepare DataFrame for evaluation
    df["category_name"] = df["Category"].apply(find_category_name)
    le_catname = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    df["category_name_encoded"] = le_catname.transform(df["category_name"])

    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    X_eval = df[feature_names]
    y_eval = df["category_name_encoded"]

    results = {}
    for clf_name in selected_models:
        model_path = os.path.join(MODEL_DIR, f"{clf_name}_model.pkl")
        if not os.path.exists(model_path):
            logging.warning(f"Model not found for {clf_name} at {model_path}")
            continue
        update_progress(f"Evaluating model {clf_name}...")
        model = joblib.load(model_path)
        y_pred = model.predict(X_eval)
        report = classification_report(y_eval, y_pred, output_dict=True)
        results[clf_name] = report
        update_progress(f"Completed evaluation for {clf_name}.")

    return results
