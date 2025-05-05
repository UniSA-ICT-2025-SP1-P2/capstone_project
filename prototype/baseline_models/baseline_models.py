# Run the following code to generate baseline classification models on the CIC-MalMem2022.csv dataset:
# ml_pipeline.py

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
from sklearn.metrics import (classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Set base directory for relative paths (e.g., for Flask context)
app_root = os.path.dirname(os.path.abspath(__file__))

# Define perturbable features (for adversarial usage)
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

def find_category(file_name):
    return file_name.split("-")[0] if "-" in file_name else file_name

def find_category_name(file_name):
    parts = file_name.split("-")
    return parts[1] if len(parts) > 1 else file_name

def extract_unique_file_id(file_name):
    return file_name.rsplit('-', 1)[0]

def run_all_models(df):
    df["category"] = df["Category"].apply(find_category)
    df["category_name"] = df["Category"].apply(find_category_name)
    df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)

    # Label encoding
    le_catname = LabelEncoder()
    df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])
    df['group_id'] = df.apply(lambda row: row['unique_file_id'] if row['Class'] != 'Benign' else f"benign_{row.name}", axis=1)

    # Splits
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

    classifiers = {
        'RandomForest': (RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42), False),
        'KNN': (KNeighborsClassifier(n_neighbors=7, weights='distance'), True),
        'LogisticRegression': (LogisticRegression(penalty='l2', C=0.5, solver='liblinear', max_iter=1000, random_state=42), True),
        'DecisionTree': (DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42), False),
        'SVM': (SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42), True)
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

    results = {}
    shap_features_list = []
    metrics_list = []
    conti_encoded = le_catname.transform(["Conti"])[0]

    for clf_name, (clf_obj, scale_required) in classifiers.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf_obj)]) if scale_required else clf_obj

        if clf_name in param_grids:
            grid = {f"clf__{k}" if scale_required else k: v for k, v in param_grids[clf_name].items()}
            search = GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
            search.fit(X_train, y_train, groups=train_df['group_id'])
            model = search.best_estimator_
        else:
            model = pipeline.fit(X_train, y_train) if scale_required else clf_obj.fit(X_train, y_train)

        os.makedirs(os.path.join(app_root, "models"), exist_ok=True)
        model_path = os.path.join(app_root, "models", f"{clf_name}_model.pkl")
        joblib.dump(model, model_path)

        y_pred = model.predict(X_test)
        y_true = y_test
        report = classification_report(y_true, y_pred, output_dict=True)

        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                metrics_list.append({"Classifier": clf_name, "Class": label, **scores})

        model_result = {
            "model_path": model_path,
            "report": report,
        }

        try:
            model_for_shap = model.named_steps['clf'] if scale_required else model
            explainer = shap.Explainer(model_for_shap, X_train)
            shap_vals = explainer(X_test)

            conti_vals = shap_vals.values[:, :, conti_encoded] if shap_vals.values.ndim == 3 else shap_vals.values
            shap_mean = np.abs(conti_vals).mean(axis=0)
            shap_series = pd.Series(shap_mean, index=X_test.columns).sort_values(ascending=False)

            os.makedirs(os.path.join(app_root, "outputs", "shap_charts"), exist_ok=True)
            shap_path = os.path.join(app_root, "outputs", "shap_charts", f"{clf_name}_SHAP.png")
            shap_series.plot(kind='bar', title=f"SHAP for {clf_name}")
            plt.ylabel("|SHAP|")
            plt.tight_layout()
            plt.savefig(shap_path)
            plt.close()

            for f, v in shap_series.items():
                shap_features_list.append({"Classifier": clf_name, "Feature": f, "SHAP Importance": v})

            model_result["shap_image"] = shap_path

        except Exception as e:
            logging.warning(f"SHAP explanation failed for {clf_name}: {e}")
            model_result["shap_image"] = None

        results[clf_name] = model_result

    # Save metrics and SHAP features
    results_excel_path = os.path.join(app_root, "Classifier_Results.xlsx")
    with pd.ExcelWriter(results_excel_path) as writer:
        pd.DataFrame(metrics_list).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame(shap_features_list).to_excel(writer, sheet_name="SHAP_Features", index=False)

    # Save datasets
    train_df.to_csv(os.path.join(app_root, "Train_Dataset_Malware_Type.csv"), index=False)
    validation_df.to_csv(os.path.join(app_root, "Validation_Dataset_Malware_Type.csv"), index=False)
    test_df.to_csv(os.path.join(app_root, "Test_Dataset_Malware_Type.csv"), index=False)

    # Save adversarial features to CSV
    pd.DataFrame({"Perturbable_Feature": ADVERSARIAL_FEATURES}).to_csv(
        os.path.join(app_root, "Adversarial_Features.csv"), index=False
    )

    return {
        "status": "complete",
        "message": "Models trained and results saved.",
        "adversarial_features": ADVERSARIAL_FEATURES,
        "results": results
    }