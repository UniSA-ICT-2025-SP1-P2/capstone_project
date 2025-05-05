import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the CIC-MalMem-2022 dataset.
    
    Args:
        data_path (str): Path to the dataset CSV file.
        
    Returns:
        tuple: (df, df_category_summary) - Preprocessed DataFrame and category summary.
    """
    try:
        df = pd.read_csv(data_path)
        df.fillna(method="ffill", inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    # Functions to extract category information
    def find_category(file_name):
        return file_name.split("-")[0] if "-" in file_name else file_name

    def find_category_name(file_name):
        parts = file_name.split("-")
        return parts[1] if len(parts) > 1 else file_name

    def extract_unique_file_id(file_name):
        return file_name.rsplit('-', 1)[0]

    # Create new columns
    df["category"] = df["Category"].apply(find_category)
    df["category_name"] = df["Category"].apply(find_category_name)
    df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)

    # Compute category summary
    unique_counts = df.groupby('category_name')['unique_file_id'].nunique()
    total_records = df['category_name'].value_counts()
    df_category_summary = pd.DataFrame({
        'Total_Records': total_records,
        'Unique_File_Counts': unique_counts
    })
    df_category_summary['Percentage'] = (df_category_summary['Total_Records'] / len(df)) * 100
    df_category_summary = df_category_summary.reset_index().rename(columns={'index': 'category_name'})
    
    print("Category Summary:")
    print(df_category_summary)
    df_category_summary.to_csv("df_category_summary.csv", index=False)
    
    return df, df_category_summary

def split_and_encode_data(df):
    """
    Encode labels and split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le_catname,
                train_df, validation_df, test_df, meta_val, meta_test)
    """
    # Encoding
    meta_cols = ['Category', 'category_name', 'unique_file_id']
    df_meta = df[meta_cols].copy()
    
    le_class = LabelEncoder()
    le_category = LabelEncoder()
    le_catname = LabelEncoder()
    
    df['Class_encoded'] = le_class.fit_transform(df['Class'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])
    
    df['group_id'] = df.apply(
        lambda row: row['unique_file_id'] if row['Class'] != 'Benign' else f"benign_{row.name}", axis=1
    )
    
    # Define features and target
    features = df.drop(columns=[
        'Category', 'Class', 'category', 'category_name', 'Class_encoded',
        'category_encoded', 'category_name_encoded', 'unique_file_id', 'group_id'
    ])
    target = df['category_name_encoded']
    
    # Split data
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
    val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
    validation_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    def get_features_and_target(sub_df):
        X = sub_df.drop(columns=[
            'Category', 'Class', 'category', 'category_name', 'Class_encoded',
            'category_encoded', 'category_name_encoded', 'unique_file_id', 'group_id'
        ])
        y = sub_df['category_name_encoded']
        return X, y
    
    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(validation_df)
    X_test, y_test = get_features_and_target(test_df)
    feature_cols = X_train.columns.tolist()
    
    meta_val = validation_df[meta_cols].copy()
    meta_test = test_df[meta_cols].copy()
    
    # Print split summary
    total = len(X_train) + len(X_val) + len(X_test)
    print("Total records:", total)
    print(f"Train: {len(X_train)} ({round((len(X_train)/total)*100, 2)}%)")
    print(f"Test: {len(X_test)} ({round((len(X_test)/total)*100, 2)}%)")
    print(f"Validation: {len(X_val)} ({round((len(X_val)/total)*100, 2)}%)")
    
    # Save datasets
    train_df.to_csv("Train_Dataset_Malware_Type.csv", index=False)
    validation_df.to_csv("Validation_Dataset_Malware_Type.csv", index=False)
    test_df.to_csv("Test_Dataset_Malware_Type.csv", index=False)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le_catname,
            train_df, validation_df, test_df, meta_val, meta_test)

def train_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le_catname, train_df, meta_test):
    """
    Train machine learning models and compute SHAP feature importance.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and test data.
        feature_cols (list): List of feature column names.
        le_catname: LabelEncoder for category names.
        train_df (pd.DataFrame): Training DataFrame with group_id.
        meta_test (pd.DataFrame): Test metadata.
        
    Returns:
        tuple: (trained_models, shap_values_dict, metrics_df, shap_features_df)
    """
    # Classifier configurations
    rf_classifier = RandomForestClassifier(
        n_estimators=50, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42
    )
    logistic_classifier = LogisticRegression(
        penalty='l2', C=0.5, solver='liblinear', max_iter=1000, random_state=42
    )
    tree_classifier = DecisionTreeClassifier(
        max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42
    )
    
    classifiers = {
        'RandomForest': (rf_classifier, False),
        'LogisticRegression': (logistic_classifier, True),
        'DecisionTree': (tree_classifier, False)
    }
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 75], 'max_depth': [3, 5, 7],
            'min_samples_split': [4, 6], 'min_samples_leaf': [2, 3]
        },
        'LogisticRegression': {'C': [0.1, 0.5, 1]},
        'DecisionTree': {
            'max_depth': [3, 5], 'min_samples_split': [6, 8], 'min_samples_leaf': [2, 3]
        }
    }
    
    results_dict = {}
    shap_values_dict = {}
    metrics_list = []
    shap_features_list = []
    train_groups = train_df['group_id']
    conti_encoded = le_catname.transform(['Conti'])[0]
    
    for clf_name, (clf_obj, scale_required) in classifiers.items():
        print(f"\nTraining and evaluating {clf_name}...")
        
        # Build pipeline
        if scale_required:
            pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf_obj)])
            grid = {f'clf__{param}': values for param, values in param_grids[clf_name].items()}
            grid_search = GridSearchCV(
                pipeline, grid, cv=GroupKFold(n_splits=5), scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train, groups=train_groups)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        else:
            grid_search = GridSearchCV(
                clf_obj, param_grids[clf_name], cv=GroupKFold(n_splits=5), scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train, groups=train_groups)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        
        joblib.dump(best_model, f"{clf_name}_AA_model.pkl")
        print(f"Saved model to {clf_name}_AA_model.pkl")
        
        # Predictions and metrics
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        y_val_pred_labels = le_catname.inverse_transform(y_val_pred)
        y_val_labels = le_catname.inverse_transform(y_val)
        y_test_pred_labels = le_catname.inverse_transform(y_test_pred)
        y_test_labels = le_catname.inverse_transform(y_test)
        
        report_dict = classification_report(y_test_labels, y_test_pred_labels, output_dict=True)
        for class_label, scores in report_dict.items():
            if class_label not in ["accuracy", "macro avg", "weighted avg"]:
                metrics_list.append({
                    'Classifier': clf_name, 'Class': class_label,
                    'Precision': scores.get('precision', None),
                    'Recall': scores.get('recall', None),
                    'F1-score': scores.get('f1-score', None),
                    'Support': scores.get('support', None)
                })
        
        # SHAP feature importance
        try:
            model_for_shap = best_model.named_steps['clf'] if scale_required else best_model
            explainer = shap.Explainer(model_for_shap, X_train, feature_names=feature_cols)
            shap_values = explainer(X_test)
            shap_values_dict[clf_name] = shap_values
            
            if len(shap_values.values.shape) == 3:
                conti_shap = shap_values.values[:, :, conti_encoded]
                mean_shap = np.abs(conti_shap).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            shap_importance = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=False)
            print(f"\nSHAP Feature Importance for 'Conti' - {clf_name}:")
            print(shap_importance)
            
            shap_importance.plot(kind='bar', title=f"SHAP Importance for 'Conti' - {clf_name}")
            plt.ylabel('Mean |SHAP value|')
            plt.show()
            
            for feature, shap_val in shap_importance.items():
                shap_features_list.append({
                    'Classifier': clf_name, 'Feature': feature, 'SHAP Importance': shap_val
                })
            
            # SHAP summary plot
            shap_values_array = shap_values.values
            feature_names = shap_values.feature_names
            if shap_values_array.ndim == 3:
                shap_values_array = shap_values_array[:, :, 0]
            X_test_np = X_test.to_numpy()
            plt.figure(figsize=(12, 8))
            plt.title(f"SHAP Summary Plot - {clf_name} Model", fontsize=14, fontweight="bold")
            shap.summary_plot(shap_values_array, X_test_np, feature_names=feature_names, show=False)
            plt.show()
        
        except Exception as e:
            print(f"SHAP explanation failed for {clf_name}: {e}")
        
        # Save test results
        if hasattr(best_model, "predict_proba"):
            test_probs = best_model.predict_proba(X_test)
            predicted_probabilities = [round(prob[label] * 100, 2) for prob, label in zip(test_probs, y_test_pred)]
        else:
            predicted_probabilities = [None] * len(y_test)
        
        results_test_clf = X_test.copy()
        results_test_clf['Actual_Class'] = y_test_labels
        results_test_clf['Predicted_Class'] = y_test_pred_labels
        results_test_clf['Correct'] = results_test_clf['Actual_Class'] == results_test_clf['Predicted_Class']
        results_test_clf['Prediction_Probability'] = predicted_probabilities
        results_test_clf = results_test_clf.merge(meta_test, left_index=True, right_index=True)
        
        csv_filename = f"{clf_name}_Malware_Type_Test_Results.csv"
        results_test_clf.to_csv(csv_filename, index=False)
        
        print(f"\nValidation Set Classification Report for {clf_name}:")
        print(classification_report(y_val_labels, y_val_pred_labels, digits=4))
        print(f"\nTest Set Classification Report for {clf_name}:")
        print(classification_report(y_test_labels, y_test_pred_labels, digits=4))
        
        results_dict[clf_name] = results_test_clf
    
    # Save combined results
    metrics_df = pd.DataFrame(metrics_list)
    shap_features_df = pd.DataFrame(shap_features_list)
    with pd.ExcelWriter("Classifier_Results.xlsx") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        shap_features_df.to_excel(writer, sheet_name="SHAP_Features", index=False)
    print("\nCombined classification metrics and SHAP feature importances saved to 'Classifier_Results.xlsx'.")
    
    return train_models, shap_values_dict, metrics_df, shap_features_df

def fgsm_attack_simulation(X_test, test_df, le_catname, feature_cols, trained_models):
    """
    Perform FGSM attack simulation on Conti samples.
    
    Args:
        X_test: Test features.
        test_df: Test DataFrame.
        le_catname: LabelEncoder for category names.
        feature_cols: List of feature column names.
        trained_models: Dictionary of trained models.
        
    Returns:
        dict: Adversarial samples for each epsilon.
    """
    classifiers = ['LogisticRegression', 'DecisionTree', 'RandomForest']
    epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    source_model = 'LogisticRegression'
    
    # Select 100 random Conti samples
    conti_encoded = le_catname.transform(['Conti'])[0]
    conti_indices = test_df[test_df['category_name_encoded'] == conti_encoded].index
    if len(conti_indices) < 100:
        print(f"Warning: Only {len(conti_indices)} Conti samples available. Using all available samples.")
        selected_indices = conti_indices
    else:
        selected_indices = np.random.choice(conti_indices, 100, replace=False)
    
    X_test_conti = X_test.loc[selected_indices]
    y_test_conti = test_df.loc[selected_indices, 'category_name_encoded']
    original_category = test_df.loc[selected_indices, 'Category']
    original_class = test_df.loc[selected_indices, 'Class']
    original_category_type = test_df.loc[selected_indices, 'category']
    original_category_name = test_df.loc[selected_indices, 'category_name']
    
    # Save original samples
    original_samples_df = X_test_conti.copy()
    original_samples_df['Category'] = original_category.values
    original_samples_df['Class'] = original_class.values
    original_samples_df['category'] = original_category_type.values
    original_samples_df['category_name'] = original_category_name.values
    original_samples_df.to_csv("original_fgsm_samples.csv", index=False)
    
    # Critical features to preserve
    features_not_to_modify = [
        "malfind.uniqueInjections", "malfind.protection", "malfind.commitCharge",
        "pslist.avg_threads", "malfind.ninjections", "pslist.avg_handlers",
        "psxview.not_in_deskthrd_false_avg", "pslist.nppid", "pslist.nproc",
        "svcscan.kernel_drivers", "psxview.not_in_eprocess_pool_false_avg",
        "callbacks.ngeneric", "callbacks.nanonymous", "pslist.nprocs64bit",
        "psxview.not_in_eprocess_pool", "psxview.not_in_ethread_pool",
        "psxview.not_in_pslist", "psxview.not_in_pspcid_list", "psxview.not_in_session",
        "psxview.not_in_pspcid_list_false_avg", "callbacks.ncallbacks",
        "psxview.not_in_session_false_avg", "psxview.not_in_ethread_pool_false_avg",
        "psxview.not_in_deskthrd", "psxview.not_in_csrss_handles_false_avg",
        "psxview.not_in_pslist_false_avg", "psxview.not_in_csrss_handles",
        "pslist.avg_handlers", "psxview.not_in_deskthrd_false_avg", "pslist.nppid"
    ]
    
    missing_features = [feat for feat in features_not_to_modify if feat not in X_test.columns]
    if missing_features:
        print(f"Warning: Features not to modify missing in X_test: {missing_features}")
    
    indices_not_to_modify = [X_test.columns.get_loc(feat) for feat in features_not_to_modify if feat in X_test.columns]
    
    # Prepare model for attack
    scaler = trained_models[source_model].named_steps.get('scaler', None)
    model_clf = trained_models[source_model].named_steps['clf'] if scaler else trained_models[source_model]
    X_test_conti_scaled = scaler.transform(X_test_conti) if scaler else X_test_conti.values
    classifier = SklearnClassifier(model=model_clf)
    benign_encoded = le_catname.transform(['Benign'])[0]
    y_target = np.full(len(X_test_conti_scaled), benign_encoded)
    
    # Generate adversarial samples
    adversarial_samples = {}
    for eps in epsilon_values:
        print(f"Generating adversarial samples with {source_model} model, epsilon={eps}")
        fgsm = FastGradientMethod(estimator=classifier, eps=eps, targeted=True)
        if eps == 0.0:
            X_test_adv_raw = X_test_conti.values
        else:
            X_test_adv_scaled = fgsm.generate(X_test_conti_scaled, y=y_target)
            X_test_adv_final = X_test_adv_scaled.copy()
            X_test_adv_final[:, indices_not_to_modify] = X_test_conti_scaled[:, indices_not_to_modify]
            X_test_adv_raw = scaler.inverse_transform(X_test_adv_final) if scaler else X_test_adv_final
        
        X_test_adv_df = pd.DataFrame(X_test_adv_raw, columns=feature_cols)
        X_test_adv_df.iloc[:, indices_not_to_modify] = X_test_conti.iloc[:, indices_not_to_modify].values
        X_test_adv_df['Category'] = original_category.values
        X_test_adv_df['Class'] = original_class.values
        
        original_features = X_test_conti.iloc[:, indices_not_to_modify].values
        modified_features = X_test_adv_df.iloc[:, indices_not_to_modify].values
        max_diff = np.max(np.abs(original_features - modified_features))
        print(f"Maximum difference in protected features: {max_diff}")
        
        columns_order = ['Category'] + feature_cols + ['Class']
        X_test_adv_df = X_test_adv_df[columns_order]
        csv_filename = f"modified_fgsm_samples_{source_model}_eps_{eps}.csv"
        X_test_adv_df.to_csv(csv_filename, index=False)
        print(f"Saved adversarial samples to '{csv_filename}'")
        adversarial_samples[eps] = X_test_adv_df
    
    # Evaluate adversarial samples
    overall_results = []
    for eps in epsilon_values:
        print(f"\nEvaluating samples (Source: {source_model}, Epsilon: {eps})")
        csv_filename = f"modified_fgsm_samples_{source_model}_eps_{eps}.csv"
        try:
            X_test_adv_df = pd.read_csv(csv_filename)
            X_test_adv_features_only = X_test_adv_df[feature_cols]
        except FileNotFoundError:
            print(f"Samples file not found: {csv_filename}")
            continue
        
        all_labels = np.arange(len(le_catname.classes_))
        misclassification_results = []
        
        for clf_name in classifiers:
            model = trained_models[clf_name]
            y_pred_adv = model.predict(X_test_adv_features_only)
            adv_accuracy = (y_pred_adv == y_test_conti).mean()
            if eps == 0.0:
                clean_accuracy = adv_accuracy
            else:
                clean_result = next((r for r in overall_results if r['Classifier'] == clf_name and r['Epsilon'] == 0.0), None)
                clean_accuracy = clean_result['Adversarial_Accuracy'] if clean_result else adv_accuracy
            accuracy_drop = clean_accuracy - adv_accuracy
            
            print(f"Accuracy for {clf_name} at epsilon={eps}: {adv_accuracy:.4f}")
            if eps > 0:
                print(f"Accuracy Drop for {clf_name}: {accuracy_drop:.4f}")
            
            evasion_count = sum(y_pred_adv != y_test_conti)
            print(f"Total number of evasion samples for {clf_name}: {evasion_count}")
            benign_misclassified = sum(y_pred_adv == benign_encoded)
            print(f"Number of Conti samples misclassified as Benign: {benign_misclassified}")
            
            pred_counts = pd.Series(y_pred_adv).value_counts()
            pred_dist = pd.DataFrame({
                'Variant': le_catname.inverse_transform(pred_counts.index),
                'Count': pred_counts.values
            })
            print(f"\nPrediction Distribution for {clf_name}:")
            print(pred_dist.to_string(index=False))
            
            cm = confusion_matrix(y_test_conti, y_pred_adv, labels=all_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=le_catname.classes_, yticklabels=le_catname.classes_)
            plt.title(f'Confusion Matrix for {clf_name} on Samples (Epsilon={eps})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Variant', y='Count', data=pred_dist)
            plt.title(f'Prediction Distribution for {clf_name} on Samples (Epsilon={eps})')
            plt.xlabel('Predicted Variant')
            plt.ylabel('Count')
            plt.show()
            
            for variant, count in zip(le_catname.inverse_transform(pred_counts.index), pred_counts.values):
                misclassification_results.append({
                    'Source_Model': source_model, 'Epsilon': eps, 'Eval_Model': clf_name,
                    'Variant': variant, 'Count': count, 'Evasion_Count': evasion_count,
                    'Benign_Count': benign_misclassified if variant == 'Benign' else 0,
                    'Clean_Accuracy': clean_accuracy, 'Adversarial_Accuracy': adv_accuracy,
                    'Accuracy_Drop': accuracy_drop
                })
            
            overall_results.append({
                'Epsilon': eps, 'Classifier': clf_name, 'Clean_Accuracy': clean_accuracy,
                'Adversarial_Accuracy': adv_accuracy, 'Accuracy_Drop': accuracy_drop
            })
        
        misclassification_df = pd.DataFrame(misclassification_results)
        csv_filename = f"misclassification_distribution_{source_model}_eps_{eps}.csv"
        misclassification_df.to_csv(csv_filename, index=False)
        print(f"\nMisclassification distribution saved to '{csv_filename}'")
    
    overall_df = pd.DataFrame(overall_results)
    overall_df.to_csv('adversarial_accuracy_vs_epsilon.csv', index=False)
    print("\nOverall accuracy metrics saved to 'adversarial_accuracy_vs_epsilon.csv'")
    
    plt.figure(figsize=(10, 6))
    for clf_name in classifiers:
        clf_data = overall_df[overall_df['Classifier'] == clf_name]
        plt.plot(clf_data['Epsilon'], clf_data['Adversarial_Accuracy'], marker='o', label=clf_name)
    plt.xlabel('Epsilon')
    plt.ylabel('Adversarial Accuracy')
    plt.title('Adversarial Accuracy vs Epsilon')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Vulnerability thresholds
    threshold = 0.5
    vulnerability_thresholds = {}
    for clf_name in classifiers:
        clf_data = overall_df[(overall_df['Classifier'] == clf_name) & (overall_df['Epsilon'] > 0)].sort_values('Epsilon')
        below_threshold = clf_data[clf_data['Adversarial_Accuracy'] < threshold]
        vulnerability_thresholds[clf_name] = below_threshold['Epsilon'].min() if not below_threshold.empty else None
    
    print("\nVulnerability Thresholds (first epsilon where accuracy < 0.5):")
    for clf_name, eps in vulnerability_thresholds.items():
        print(f"{clf_name}: epsilon = {eps if eps is not None else 'N/A (accuracy remained ≥ 0.5)'}")
    
    drop_threshold = 0.05
    vulnerability_thresholds_drop = {}
    for clf_name in classifiers:
        clf_data = overall_df[(overall_df['Classifier'] == clf_name) & (overall_df['Epsilon'] > 0)].sort_values('Epsilon')
        significant_drop = clf_data[clf_data['Accuracy_Drop'] > drop_threshold]
        vulnerability_thresholds_drop[clf_name] = significant_drop['Epsilon'].min() if not significant_drop.empty else None
    
    print("\nVulnerability Thresholds (first epsilon where accuracy drop > 0.05):")
    for clf_name, eps in vulnerability_thresholds_drop.items():
        print(f"{clf_name}: epsilon = {eps if eps is not None else 'N/A (drop ≤ 0.05)'}")
    
    # SHAP validation
    model = trained_models['LogisticRegression']
    scaler = model.named_steps.get('scaler', None)
    model_clf = model.named_steps['clf'] if scaler else model
    X_train_subset = X_train.sample(100, random_state=42) if len(X_train) > 100 else X_train
    explainer = shap.Explainer(model_clf, X_train_subset, feature_names=feature_cols)
    
    try:
        original_samples_df = pd.read_csv("original_fgsm_samples.csv")
        X_test_conti = original_samples_df[feature_cols]
        shap_values_original = explainer(X_test_conti)
    except Exception as e:
        print(f"SHAP computation failed for original samples: {e}")
        return adversarial_samples
    
    if len(shap_values_original.values.shape) == 3:
        shap_original_conti = shap_values_original.values[:, :, conti_encoded]
    else:
        shap_original_conti = shap_values_original.values
    
    mean_shap_original = np.abs(shap_original_conti).mean(axis=0)
    comparison_dfs = []
    
    for eps in epsilon_values:
        csv_filename = f"modified_fgsm_samples_{source_model}_eps_{eps}.csv"
        try:
            X_test_adv_df = pd.read_csv(csv_filename)
            X_test_adv = X_test_adv_df[feature_cols]
        except FileNotFoundError:
            print(f"Adversarial samples file not found: {csv_filename}")
            continue
        
        try:
            shap_values_adversarial = explainer(X_test_adv)
        except Exception as e:
            print(f"SHAP computation failed for adversarial samples (eps={eps}): {e}")
            continue
        
        if len(shap_values_adversarial.values.shape) == 3:
            shap_adversarial_conti = shap_values_adversarial.values[:, :, conti_encoded]
        else:
            shap_adversarial_conti = shap_values_adversarial.values
        
        mean_shap_adversarial = np.abs(shap_adversarial_conti).mean(axis=0)
        comparison_df = pd.DataFrame({
            'Feature': feature_cols,
            'Mean_SHAP_Original': mean_shap_original,
            f'Mean_SHAP_Adversarial_eps_{eps}': mean_shap_adversarial
        })
        comparison_df = comparison_df.sort_values(by='Mean_SHAP_Original', ascending=False)
        comparison_df.to_csv(f'shap_comparison_original_vs_adversarial_eps_{eps}.csv', index=False)
        print(f"\nSHAP Value Comparison for epsilon={eps} (Sorted by Mean SHAP Original):")
        print(comparison_df.to_string(index=False))
        
        top_n = 10
        top_features_df = comparison_df.head(top_n)
        plt.figure(figsize=(12, 6))
        plt.plot(top_features_df['Feature'], top_features_df['Mean_SHAP_Original'], marker='o', label='Original')
        plt.plot(top_features_df['Feature'], top_features_df[f'Mean_SHAP_Adversarial_eps_{eps}'], marker='x', label=f'Adversarial (eps={eps})')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Mean Absolute SHAP Values: Original vs Adversarial Samples (Epsilon={eps})')
        plt.ylabel('Mean |SHAP|')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        comparison_dfs.append(comparison_df)
    
    return adversarial_samples

def generate_fgsm_adversarial_samples(X_train, y_train, le_catname, feature_cols, source_model='LogisticRegression', epsilon=0.1, n_samples=5000):
    """
    Generate FGSM adversarial samples for retraining.
    
    Args:
        X_train, y_train: Training data.
        le_catname: LabelEncoder for category names.
        feature_cols: List of feature column names.
        source_model (str): Model to use for attack.
        epsilon (float): Perturbation strength.
        n_samples (int): Number of adversarial samples to generate.
        
    Returns:
        pd.DataFrame: Adversarial samples.
    """
    try:
        model = joblib.load(f"{source_model}_AA_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{source_model}_AA_model.pkl' not found.")
    
    scaler = model.named_steps.get('scaler', None)
    model_clf = model.named_steps['clf'] if scaler else model
    X_train_scaled = scaler.transform(X_train) if scaler else X_train.values
    classifier = SklearnClassifier(model=model_clf)
    
    benign_encoded = le_catname.transform(['Benign'])[0]
    malware_mask = y_train != benign_encoded
    X_train_malware = X_train[malware_mask]
    X_train_malware_scaled = X_train_scaled[malware_mask]
    y_train_malware = y_train[malware_mask]
    
    if len(X_train_malware) != len(X_train_malware_scaled) or len(X_train_malware) != len(y_train_malware):
        raise ValueError(f"Mismatch in sample counts: X_train_malware ({len(X_train_malware)}), "
                         f"X_train_malware_scaled ({len(X_train_malware_scaled)}), y_train_malware ({len(y_train_malware)})")
    
    if len(X_train_malware) < n_samples:
        print(f"Warning: Only {len(X_train_malware)} malware samples available. Using all available samples.")
        selected_indices = X_train_malware.index
    else:
        selected_indices = np.random.choice(X_train_malware.index, n_samples, replace=False)
    
    X_train_subset = X_train_malware.loc[selected_indices]
    y_train_subset = y_train_malware.loc[selected_indices]
    positional_indices = [np.where(X_train_malware.index == idx)[0][0] for idx in selected_indices]
    X_train_subset_scaled = X_train_malware_scaled[positional_indices]
    
    # No feature preservation (as per original code)
    features_not_to_modify = []
    indices_not_to_modify = [X_train.columns.get_loc(feat) for feat in features_not_to_modify if feat in X_train.columns]
    
    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=True)
    X_train_adv_scaled = fgsm.generate(X_train_subset_scaled, y=np.full(len(X_train_subset_scaled), benign_encoded))
    X_train_adv_scaled[:, indices_not_to_modify] = X_train_subset_scaled[:, indices_not_to_modify]
    X_train_adv = scaler.inverse_transform(X_train_adv_scaled) if scaler else X_train_adv_scaled
    
    X_train_adv_df = pd.DataFrame(X_train_adv, columns=feature_cols)
    X_train_adv_df['category_name_encoded'] = y_train_subset.values
    
    output_path = "../defence_prototype/fgsm_adversarial_samples.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_train_adv_df.to_csv(output_path, index=False)
    print(f"Saved {len(X_train_adv_df)} adversarial training samples to '{output_path}'")
    
    return X_train_adv_df

def pgd_attack_simulation(X_test, test_df, le_catname, feature_cols, trained_models):
    """
    Perform PGD attack simulation on Conti samples.
    
    Args:
        X_test: Test features.
        test_df: Test DataFrame.
        le_catname: LabelEncoder for category names.
        feature_cols: List of feature column names.
        trained_models: Dictionary of trained models.
        
    Returns:
        dict: Adversarial samples for each epsilon.
    """
    classifiers = ['LogisticRegression', 'DecisionTree', 'RandomForest']
    epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    source_model = 'LogisticRegression'
    
    conti_encoded = le_catname.transform(['Conti'])[0]
    conti_indices = test_df[test_df['category_name_encoded'] == conti_encoded].index
    if len(conti_indices) < 100:
        print(f"Warning: Only {len(conti_indices)} Conti samples available. Using all available samples.")
        selected_indices = conti_indices
    else:
        selected_indices = np.random.choice(conti_indices, 100, replace=False)
    
    X_test_conti = X_test.loc[selected_indices]
    y_test_conti = test_df.loc[selected_indices, 'category_name_encoded']
    original_samples_df = X_test_conti.copy()
    original_samples_df['Category'] = test_df.loc[selected_indices, 'Category']
    original_samples_df['Class'] = test_df.loc[selected_indices, 'Class']
    original_samples_df['category'] = test_df.loc[selected_indices, 'category']
    original_samples_df['category_name'] = test_df.loc[selected_indices, 'category_name']
    original_samples_df.to_csv("original_pgd_samples.csv", index=False)
    
    features_not_to_modify = [
        "malfind.uniqueInjections", "malfind.protection", "malfind.commitCharge",
        "pslist.avg_threads", "malfind.ninjections", "pslist.avg_handlers",
        "psxview.not_in_deskthrd_false_avg", "pslist.nppid", "pslist.nproc",
        "svcscan.kernel_drivers", "psxview.not_in_eprocess_pool_false_avg",
        "callbacks.ngeneric", "callbacks.nanonymous", "pslist.nprocs64bit",
        "psxview.not_in_eprocess_pool", "psxview.not_in_ethread_pool",
        "psxview.not_in_pslist", "psxview.not_in_pspcid_list", "psxview.not_in_session",
        "psxview.not_in_pspcid_list_false_avg", "callbacks.ncallbacks",
        "psxview.not_in_session_false_avg", "psxview.not_in_ethread_pool_false_avg",
        "psxview.not_in_deskthrd", "psxview.not_in_csrss_handles_false_avg",
        "psxview.not_in_pslist_false_avg", "psxview.not_in_csrss_handles",
        "pslist.avg_handlers", "psxview.not_in_deskthrd_false_avg", "pslist.nppid"
    ]
    
    missing_features = [feat for feat in features_not_to_modify if feat not in X_test.columns]
    if missing_features:
        print(f"Warning: Features not to modify missing in X_test: {missing_features}")
    
    indices_not_to_modify = [X_test.columns.get_loc(feat) for feat in features_not_to_modify if feat in X_test.columns]
    
    scaler = trained_models[source_model].named_steps.get('scaler', None)
    model_clf = trained_models[source_model].named_steps['clf'] if scaler else trained_models[source_model]
    X_test_conti_scaled = scaler.transform(X_test_conti) if scaler else X_test_conti.values
    classifier = SklearnClassifier(model=model_clf)
    benign_encoded = le_catname.transform(['Benign'])[0]
    y_target = np.full(len(X_test_conti_scaled), benign_encoded)
    
    adversarial_samples_pgd = {}
    for eps in epsilon_values:
        print(f"Generating PGD adversarial samples with {source_model} model, epsilon={eps}")
        pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=0.01, max_iter=10, targeted=True)
        if eps == 0.0:
            X_test_adv_raw = X_test_conti.values
        else:
            X_test_adv_scaled = pgd.generate(X_test_conti_scaled, y=y_target)
            X_test_adv_final = X_test_adv_scaled.copy()
            X_test_adv_final[:, indices_not_to_modify] = X_test_conti_scaled[:, indices_not_to_modify]
            X_test_adv_raw = scaler.inverse_transform(X_test_adv_final) if scaler else X_test_adv_final
        
        X_test_adv_df = pd.DataFrame(X_test_adv_raw, columns=feature_cols)
        X_test_adv_df.iloc[:, indices_not_to_modify] = X_test_conti.iloc[:, indices_not_to_modify].values
        X_test_adv_df['Category'] = original_samples_df['Category'].values
        X_test_adv_df['Class'] = original_samples_df['Class'].values
        
        max_diff = np.max(np.abs(X_test_conti.iloc[:, indices_not_to_modify].values - X_test_adv_df.iloc[:, indices_not_to_modify].values))
        print(f"Max difference in protected features: {max_diff}")
        
        columns_order = ['Category'] + feature_cols + ['Class']
        X_test_adv_df = X_test_adv_df[columns_order]
        csv_filename = f"modified_pgd_samples_{source_model}_eps_{eps}.csv"
        X_test_adv_df.to_csv(csv_filename, index=False)
        print(f"Saved PGD adversarial samples to '{csv_filename}'")
        adversarial_samples_pgd[eps] = X_test_adv_df
    
    overall_results_pgd = []
    for eps in epsilon_values:
        print(f"\nEvaluating PGD samples (Source: {source_model}, Epsilon: {eps})")
        csv_filename = f"modified_pgd_samples_{source_model}_eps_{eps}.csv"
        try:
            X_test_adv_df = pd.read_csv(csv_filename)
            X_test_adv_features_only = X_test_adv_df[feature_cols]
        except FileNotFoundError:
            print(f"Samples file not found: {csv_filename}")
            continue
        
        all_labels = np.arange(len(le_catname.classes_))
        for clf_name in classifiers:
            model = trained_models[clf_name]
            y_pred_adv = model.predict(X_test_adv_features_only)
            adv_accuracy = (y_pred_adv == y_test_conti).mean()
            if eps == 0.0:
                clean_accuracy = adv_accuracy
            else:
                clean_result = next((r for r in overall_results_pgd if r['Classifier'] == clf_name and r['Epsilon'] == 0.0), None)
                clean_accuracy = clean_result['Adversarial_Accuracy'] if clean_result else adv_accuracy
            accuracy_drop = clean_accuracy - adv_accuracy
            
            print(f"Accuracy for {clf_name} at epsilon={eps}: {adv_accuracy:.4f}")
            if eps > 0:
                print(f"Accuracy Drop for {clf_name}: {accuracy_drop:.4f}")
            
            cm = confusion_matrix(y_test_conti, y_pred_adv, labels=all_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=le_catname.classes_, yticklabels=le_catname.classes_)
            plt.title(f'PGD Confusion Matrix for {clf_name} (Epsilon={eps})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            
            overall_results_pgd.append({
                'Epsilon': eps, 'Classifier': clf_name, 'Clean_Accuracy': clean_accuracy,
                'Adversarial_Accuracy': adv_accuracy, 'Accuracy_Drop': accuracy_drop
            })
    
    overall_df_pgd = pd.DataFrame(overall_results_pgd)
    overall_df_pgd.to_csv('adversarial_accuracy_vs_epsilon_pgd.csv', index=False)
    print("\nPGD adversarial accuracy metrics saved to 'adversarial_accuracy_vs_epsilon_pgd.csv'")
    
    plt.figure(figsize=(10, 6))
    for clf_name in classifiers:
        clf_data = overall_df_pgd[overall_df_pgd['Classifier'] == clf_name]
        plt.plot(clf_data['Epsilon'], clf_data['Adversarial_Accuracy'], marker='o', label=clf_name)
    plt.xlabel('Epsilon')
    plt.ylabel('Adversarial Accuracy')
    plt.title('PGD Adversarial Accuracy vs Epsilon')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return adversarial_samples_pgd

def generate_pgd_adversarial_samples(X_train, y_train, le_catname, feature_cols, source_model='LogisticRegression', epsilon=0.1, n_samples=5000):
    """
    Generate PGD adversarial samples for retraining.
    
    Args:
        X_train, y_train: Training data.
        le_catname: LabelEncoder for category names.
        feature_cols: List of feature column names.
        source_model (str): Model to use for attack.
        epsilon (float): Perturbation strength.
        n_samples (int): Number of adversarial samples to generate.
        
    Returns:
        pd.DataFrame: Adversarial samples.
    """
    try:
        model = joblib.load(f"{source_model}_AA_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{source_model}_AA_model.pkl' not found.")
    
    scaler = model.named_steps.get('scaler', None)
    model_clf = model.named_steps['clf'] if scaler else model
    X_train_scaled = scaler.transform(X_train) if scaler else X_train.values
    classifier = SklearnClassifier(model=model_clf)
    
    benign_encoded = le_catname.transform(['Benign'])[0]
    malware_mask = y_train != benign_encoded
    X_train_malware = X_train[malware_mask]
    X_train_malware_scaled = X_train_scaled[malware_mask]
    y_train_malware = y_train[malware_mask]
    
    if len(X_train_malware) < n_samples:
        print(f"Warning: Only {len(X_train_malware)} malware samples available. Using all available samples.")
        selected_indices = X_train_malware.index
    else:
        selected_indices = np.random.choice(X_train_malware.index, n_samples, replace=False)
    
    X_train_subset = X_train_malware.loc[selected_indices]
    y_train_subset = y_train_malware.loc[selected_indices]
    positional_indices = [np.where(X_train_malware.index == idx)[0][0] for idx in selected_indices]
    X_train_subset_scaled = X_train_malware_scaled[positional_indices]
    
    pgd = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.01, max_iter=10, targeted=True)
    X_train_adv_scaled = pgd.generate(X_train_subset_scaled, y=np.full(len(X_train_subset_scaled), benign_encoded))
    X_train_adv = scaler.inverse_transform(X_train_adv_scaled) if scaler else X_train_adv_scaled
    
    X_train_adv_df = pd.DataFrame(X_train_adv, columns=feature_cols)
    X_train_adv_df['category_name_encoded'] = y_train_subset.values
    
    output_path = "../defence_prototype/pgd_adversarial_samples.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_train_adv_df.to_csv(output_path, index=False)
    print(f"Saved {len(X_train_adv_df)} PGD adversarial training samples to '{output_path}'")
    
    return X_train_adv_df

def main():
    """
    Main function to execute the malware analysis pipeline.
    """
    data_path = "data.csv"  # Update with actual path
    df, df_category_summary = load_and_preprocess_data(data_path)
    
    (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le_catname,
     train_df, validation_df, test_df, meta_val, meta_test) = split_and_encode_data(df)
    
    trained_models, shap_values_dict, metrics_df, shap_features_df = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le_catname, train_df, meta_test
    )
    
    fgsm_samples = fgsm_attack_simulation(X_test, test_df, le_catname, feature_cols, trained_models)
    fgsm_adv_samples = generate_fgsm_adversarial_samples(X_train, y_train, le_catname, feature_cols)
    
    pgd_samples = pgd_attack_simulation(X_test, test_df, le_catname, feature_cols, trained_models)
    pgd_adv_samples = generate_pgd_adversarial_samples(X_train, y_train, le_catname, feature_cols)

if __name__ == "__main__":
    main()