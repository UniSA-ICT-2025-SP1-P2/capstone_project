import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
import shap
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from train_models import find_category_name  # Import utility function

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, "baseline_models", "trained_baseline_models")
DATA_DIR = os.path.join(BASE_DIR, "baseline_models", "baseline_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_simulation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_resources():
    """Load necessary resources for attack simulation"""
    le_catname = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return le_catname, feature_names

def compare_shap_values(model, X_original, X_adversarial, class_label="Conti", epsilon=0.1):
    """Compare SHAP values between original and adversarial samples."""
    le_catname, feature_names = load_resources()
    conti_encoded = le_catname.transform([class_label])[0]
    
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "Train_Dataset.csv"))
        X_train = train_df[feature_names]
        X_train_subset = X_train.sample(100, random_state=42) if len(X_train) > 100 else X_train
    except Exception as e:
        print(f"Error loading training data: {e}. Using original data as background.")
        X_train_subset = X_original
    
    try:
        explainer = shap.Explainer(model.predict, X_train_subset) if hasattr(model, 'named_steps') else shap.Explainer(model, X_train_subset)
        shap_values_original = explainer(X_original)
        shap_values_adversarial = explainer(X_adversarial)
        
        shap_original = shap_values_original.values[:, :, conti_encoded] if shap_values_original.values.ndim == 3 else shap_values_original.values
        shap_adversarial = shap_values_adversarial.values[:, :, conti_encoded] if shap_values_adversarial.values.ndim == 3 else shap_values_adversarial.values
        
        mean_shap_original = np.abs(shap_original).mean(axis=0)
        mean_shap_adversarial = np.abs(shap_adversarial).mean(axis=0)
        
        comparison_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP_Original': mean_shap_original,
            'Mean_SHAP_Adversarial': mean_shap_adversarial
        }).sort_values(by='Mean_SHAP_Original', ascending=False)
        
        filename = os.path.join(OUTPUT_DIR, f"shap_comparison_eps_{epsilon}.csv")
        comparison_df.to_csv(filename, index=False)
        
        top_n = 10
        top_features_df = comparison_df.head(top_n)
        plt.figure(figsize=(12, 6))
        plt.plot(top_features_df['Feature'], top_features_df['Mean_SHAP_Original'], marker='o', label='Original')
        plt.plot(top_features_df['Feature'], top_features_df['Mean_SHAP_Adversarial'], marker='x', label='Adversarial')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'SHAP Comparison (Epsilon={epsilon})')
        plt.ylabel('Mean |SHAP|')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_comparison_eps_{epsilon}.png"))
        plt.close()
        
        return comparison_df
        
    except Exception as e:
        print(f"Error in SHAP comparison: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def run_attack_simulation(models, X_test, y_test, epsilon_values, features_not_to_modify, source_model, target_class="Benign"):
    """Run attack simulation with adversarial samples generated for multiple epsilon values using FGSM."""
    
    le_catname, feature_names = load_resources()
    all_results = []
    
    conti_encoded = le_catname.transform(["Conti"])[0]
    benign_encoded = le_catname.transform([target_class])[0]
    
    # Select Conti samples
    conti_indices = y_test.index[y_test == conti_encoded]
    if len(conti_indices) == 0:
        raise ValueError("No Conti samples found in the dataset")
    
    compare_indices = conti_indices[:100] if len(conti_indices) > 100 else conti_indices
    X_conti = X_test.loc[compare_indices]
    
    model = models[source_model]
    scaler = model.named_steps.get('scaler', None) if hasattr(model, 'named_steps') else None
    classifier = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    art_classifier = SklearnClassifier(model=classifier)
    
    for epsilon in epsilon_values:
        print(f"Running FGSM attack simulation for epsilon={epsilon}")
        
        # Generate adversarial samples
        X_conti_scaled = scaler.transform(X_conti) if scaler else X_conti.values
        fgsm = FastGradientMethod(estimator=art_classifier, eps=epsilon, targeted=True)
        X_adv_scaled = fgsm.generate(X_conti_scaled, y=np.full(len(X_conti), benign_encoded))
        X_adv = scaler.inverse_transform(X_adv_scaled) if scaler else X_adv_scaled
        
        # Preserve critical features
        indices_not_to_modify = [feature_names.index(f) for f in features_not_to_modify if f in feature_names]
        X_adv[:, indices_not_to_modify] = X_conti.values[:, indices_not_to_modify]
        
        X_adv_df = pd.DataFrame(X_adv, columns=feature_names, index=compare_indices)
        
        # Save comparison
        comparison_df = pd.concat([X_conti.add_suffix('_original'), X_adv_df.add_suffix('_adversarial')], axis=1)
        comparison_df.to_csv(os.path.join(OUTPUT_DIR, f"comparison_samples_fgsm_eps_{epsilon}.csv"), index=False)
        
        # Add Category and Class
        try:
            test_df = pd.read_csv(os.path.join(DATA_DIR, "Test_Dataset.csv"))
            X_adv_df['Category'] = test_df.loc[compare_indices, 'Category'].values if 'Category' in test_df.columns else "Conti"
            X_adv_df['Class'] = test_df.loc[compare_indices, 'Class'].values if 'Class' in test_df.columns else "Malware"
        except Exception as e:
            print(f"Could not retrieve Category/Class information: {e}")
            X_adv_df['Category'] = "Conti"
            X_adv_df['Class'] = "Malware"
        
        X_adv_df.to_csv(os.path.join(OUTPUT_DIR, f"adversarial_samples_fgsm_eps_{epsilon}.csv"), index=False)
        
        # Compare SHAP values
        compare_shap_values(
            model=model,
            X_original=X_conti,
            X_adversarial=X_adv_df[feature_names],
            class_label="Conti",
            epsilon=epsilon
        )
        
        # Run predictions
        results = []
        X_features = X_adv_df[feature_names]
        all_labels = np.arange(len(le_catname.classes_))
        
        for clf_name, clf_model in models.items():
            try:
                y_pred = clf_model.predict(X_features)
                accuracy = (y_pred == y_test.loc[compare_indices].values).mean()
                
                report = classification_report(y_test.loc[compare_indices], y_pred,
                                              labels=all_labels,
                                              target_names=le_catname.classes_,
                                              output_dict=True,
                                              zero_division=0)
                
                cm = confusion_matrix(y_test.loc[compare_indices], y_pred, labels=all_labels)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=le_catname.classes_,
                           yticklabels=le_catname.classes_)
                plt.title(f'FGSM Confusion Matrix for {clf_name} (Epsilon={epsilon})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"cm_fgsm_{clf_name}_eps_{epsilon}.png"))
                plt.close()
                
                y_pred_labels = le_catname.inverse_transform(y_pred)
                pred_dist = pd.Series(y_pred_labels).value_counts().reset_index()
                pred_dist.columns = ['Variant', 'Count']
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Variant', y='Count', data=pred_dist)
                plt.title(f'FGSM Prediction Distribution for {clf_name} (Epsilon={epsilon})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"dist_fgsm_{clf_name}_eps_{epsilon}.png"))
                plt.close()
                
                results.append({
                    'Classifier': clf_name,
                    'Accuracy': accuracy,
                    'Report': report,
                    'Epsilon': epsilon
                })
            except Exception as e:
                print(f"Error in FGSM attack simulation for {clf_name}: {e}")
                import traceback
                print(traceback.format_exc())
        
        all_results.extend(results)
        
        # Save results for this epsilon
        with open(os.path.join(OUTPUT_DIR, f"attack_results_fgsm_eps_{epsilon}.txt"), 'w') as f:
            for result in results:
                f.write(f"Classifier: {result['Classifier']}\n")
                f.write(f"Accuracy: {result['Accuracy']}\n")
                f.write("Classification Report:\n")
                for class_name, metrics in result['Report'].items():
                    f.write(f"  {class_name}: {metrics}\n")
                f.write("\n" + "="*50 + "\n\n")
    
    # Save summary
    results_df = pd.DataFrame([
        {
            'Classifier': r['Classifier'],
            'Epsilon': r['Epsilon'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Report']['weighted avg']['precision'],
            'Recall': r['Report']['weighted avg']['recall'],
            'F1-score': r['Report']['weighted avg']['f1-score']
        }
        for r in all_results
    ])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "attack_results_fgsm_summary.csv"), index=False)
    
    return all_results

def run_pgd_attack_simulation(models, X_test, y_test, epsilon_values, features_not_to_modify, source_model, target_class="Benign"):
    """Run attack simulation with adversarial samples generated for multiple epsilon values using PGD."""
    le_catname, feature_names = load_resources()
    all_results = []
    
    conti_encoded = le_catname.transform(["Conti"])[0]
    benign_encoded = le_catname.transform([target_class])[0]
    
    # Select Conti samples
    conti_indices = y_test.index[y_test == conti_encoded]
    if len(conti_indices) == 0:
        raise ValueError("No Conti samples found in the dataset")
    
    compare_indices = conti_indices[:100] if len(conti_indices) > 100 else conti_indices
    X_conti = X_test.loc[compare_indices]
    
    model = models[source_model]
    scaler = model.named_steps.get('scaler', None) if hasattr(model, 'named_steps') else None
    classifier = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    art_classifier = SklearnClassifier(model=classifier)
    
    for epsilon in epsilon_values:
        print(f"Running PGD attack simulation for epsilon={epsilon}")
        
        # Generate adversarial samples
        X_conti_scaled = scaler.transform(X_conti) if scaler else X_conti.values
        pgd = ProjectedGradientDescent(estimator=art_classifier, eps=epsilon, eps_step=epsilon/10, max_iter=100, targeted=True)
        X_adv_scaled = X_conti_scaled if epsilon == 0.0 else pgd.generate(X_conti_scaled, y=np.full(len(X_conti), benign_encoded))
        X_adv = scaler.inverse_transform(X_adv_scaled) if scaler else X_adv_scaled
        
        # Preserve critical features
        indices_not_to_modify = [feature_names.index(f) for f in features_not_to_modify if f in feature_names]
        X_adv[:, indices_not_to_modify] = X_conti.values[:, indices_not_to_modify]
        
        X_adv_df = pd.DataFrame(X_adv, columns=feature_names, index=compare_indices)
        
        # Save comparison
        comparison_df = pd.concat([X_conti.add_suffix('_original'), X_adv_df.add_suffix('_adversarial')], axis=1)
        comparison_df.to_csv(os.path.join(OUTPUT_DIR, f"comparison_samples_pgd_eps_{epsilon}.csv"), index=False)
        
        # Add Category and Class
        try:
            test_df = pd.read_csv(os.path.join(DATA_DIR, "Test_Dataset.csv"))
            X_adv_df['Category'] = test_df.loc[compare_indices, 'Category'].values if 'Category' in test_df.columns else "Conti"
            X_adv_df['Class'] = test_df.loc[compare_indices, 'Class'].values if 'Class' in test_df.columns else "Malware"
        except Exception as e:
            print(f"Could not retrieve Category/Class information: {e}")
            X_adv_df['Category'] = "Conti"
            X_adv_df['Class'] = "Malware"
        
        X_adv_df.to_csv(os.path.join(OUTPUT_DIR, f"adversarial_samples_pgd_eps_{epsilon}.csv"), index=False)
        
        # Run predictions
        results = []
        X_features = X_adv_df[feature_names]
        all_labels = np.arange(len(le_catname.classes_))
        
        for clf_name, clf_model in models.items():
            try:
                y_pred = clf_model.predict(X_features)
                accuracy = (y_pred == y_test.loc[compare_indices].values).mean()
                
                report = classification_report(y_test.loc[compare_indices], y_pred,
                                              labels=all_labels,
                                              target_names=le_catname.classes_,
                                              output_dict=True,
                                              zero_division=0)
                
                cm = confusion_matrix(y_test.loc[compare_indices], y_pred, labels=all_labels)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=le_catname.classes_,
                           yticklabels=le_catname.classes_)
                plt.title(f'PGD Confusion Matrix for {clf_name} (Epsilon={epsilon})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"cm_pgd_{clf_name}_eps_{epsilon}.png"))
                plt.close()
                
                y_pred_labels = le_catname.inverse_transform(y_pred)
                pred_dist = pd.Series(y_pred_labels).value_counts().reset_index()
                pred_dist.columns = ['Variant', 'Count']
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Variant', y='Count', data=pred_dist)
                plt.title(f'PGD Prediction Distribution for {clf_name} (Epsilon={epsilon})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"dist_pgd_{clf_name}_eps_{epsilon}.png"))
                plt.close()
                
                results.append({
                    'Classifier': clf_name,
                    'Accuracy': accuracy,
                    'Report': report,
                    'Epsilon': epsilon
                })
            except Exception as e:
                print(f"Error in PGD attack simulation for {clf_name}: {e}")
                import traceback
                print(traceback.format_exc())
        
        all_results.extend(results)
        
        # Save results for this epsilon
        with open(os.path.join(OUTPUT_DIR, f"attack_results_pgd_eps_{epsilon}.txt"), 'w') as f:
            for result in results:
                f.write(f"Classifier: {result['Classifier']}\n")
                f.write(f"Accuracy: {result['Accuracy']}\n")
                f.write("Classification Report:\n")
                for class_name, metrics in result['Report'].items():
                    f.write(f"  {class_name}: {metrics}\n")
                f.write("\n" + "="*50 + "\n\n")
    
    # Save summary
    results_df = pd.DataFrame([
        {
            'Classifier': r['Classifier'],
            'Epsilon': r['Epsilon'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Report']['weighted avg']['precision'],
            'Recall': r['Report']['weighted avg']['recall'],
            'F1-score': r['Report']['weighted avg']['f1-score']
        }
        for r in all_results
    ])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "attack_results_pgd_summary.csv"), index=False)
    
    return all_results

def generate_adversarial_samples_for_retraining(model, X, y, epsilon, n_samples=1000, target_class="Benign"):
    """Generate adversarial samples for retraining from non-Benign samples."""
    le_catname, feature_names = load_resources()
    np.random.seed(42)
    
    try:
        benign_encoded = le_catname.transform([target_class])[0]
    except ValueError:
        print(f"Target class '{target_class}' not found. Using first class as target.")
        benign_encoded = 0
    
    malware_mask = y != benign_encoded
    X_malware = X[malware_mask]
    y_malware = y[malware_mask]
    
    if len(X_malware) == 0:
        raise ValueError(f"No non-{target_class} samples found in the dataset")
    
    if len(X_malware) < n_samples:
        selected_indices = X_malware.index
        print(f"Warning: Requested {n_samples} samples but only {len(X_malware)} are available.")
    else:
        selected_indices = np.random.choice(X_malware.index, n_samples, replace=False)
    
    X_subset = X_malware.loc[selected_indices]
    
    classifier = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    scaler = model.named_steps.get('scaler', None) if hasattr(model, 'named_steps') else None
    X_subset_scaled = scaler.transform(X_subset) if scaler else X_subset.values
    
    art_classifier = SklearnClassifier(model=classifier)
    fgsm = FastGradientMethod(estimator=art_classifier, eps=epsilon, targeted=True)
    X_adv_scaled = fgsm.generate(X_subset_scaled, y=np.full(len(X_subset), benign_encoded))
    
    X_adv = scaler.inverse_transform(X_adv_scaled) if scaler else X_adv_scaled
    X_adv_df = pd.DataFrame(X_adv, columns=feature_names)
    
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "Train_Dataset.csv"))
        if 'Category' in train_df.columns:
            categories = [find_category_name(train_df.loc[idx, 'Category']) if idx in train_df.index else "Unknown"
                          for idx in selected_indices]
            X_adv_df.insert(0, 'Category', categories)
    except Exception as e:
        print(f"Error adding Category information: {e}")
        X_adv_df.insert(0, 'Category', ['Unknown'] * len(X_adv_df))
    
    # Save to original location
    filename = os.path.join(OUTPUT_DIR, f"adversarial_retrain_eps_{epsilon}.csv")
    X_adv_df.to_csv(filename, index=False)
    print(f"Saved retraining samples to {filename}")
    
    # Save to additional 'data' folder with new name
    data_output_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_output_dir, exist_ok=True)
    data_filename = os.path.join(data_output_dir, f"adversarial_fgsm.csv")
    X_adv_df.to_csv(data_filename, index=False)
    print(f"Saved retraining samples to {data_filename}")
    
    return X_adv_df

def generate_pgd_samples_for_retraining(model, X, y, epsilon, n_samples=1000, target_class="Benign"):
    """Generate adversarial samples for retraining using PGD from non-Benign samples."""
    le_catname, feature_names = load_resources()
    np.random.seed(42)
    
    try:
        benign_encoded = le_catname.transform([target_class])[0]
    except ValueError:
        print(f"Target class '{target_class}' not found. Using first class as target.")
        benign_encoded = 0
    
    # Select non-Benign samples
    malware_mask = y != benign_encoded
    X_malware = X[malware_mask]
    y_malware = y[malware_mask]
    
    if len(X_malware) == 0:
        raise ValueError(f"No non-{target_class} samples found in the dataset")
    
    if len(X_malware) < n_samples:
        selected_indices = X_malware.index
        print(f"Warning: Requested {n_samples} samples but only {len(X_malware)} available.")
    else:
        selected_indices = np.random.choice(X_malware.index, n_samples, replace=False)
    
    X_subset = X_malware.loc[selected_indices]
    
    # Prepare classifier and scaler
    classifier = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    scaler = model.named_steps.get('scaler', None) if hasattr(model, 'named_steps') else None
    X_subset_scaled = scaler.transform(X_subset) if scaler else X_subset.values
    
    # Generate PGD adversarial samples
    art_classifier = SklearnClassifier(model=classifier)
    pgd = ProjectedGradientDescent(
        estimator=art_classifier,
        eps=epsilon,
        eps_step=epsilon / 5,  # Dynamic step size
        max_iter=75,          # Increased iterations for better convergence
        targeted=True
    )
    X_adv_scaled = pgd.generate(X_subset_scaled, y=np.full(len(X_subset), benign_encoded))
    
    # Reverse scaling if applicable
    X_adv = scaler.inverse_transform(X_adv_scaled) if scaler else X_adv_scaled
    X_adv_df = pd.DataFrame(X_adv, columns=feature_names)
    
    # Add Category information
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "Train_Dataset.csv"))
        if 'Category' in train_df.columns:
            categories = [find_category_name(train_df.loc[idx, 'Category']) if idx in train_df.index else "Unknown"
                          for idx in selected_indices]
            X_adv_df.insert(0, 'Category', categories)
    except Exception as e:
        print(f"Error adding Category information: {e}")
        X_adv_df.insert(0, 'Category', ['Unknown'] * len(X_adv_df))
    
    # Save samples
    filename = os.path.join(OUTPUT_DIR, f"adversarial_retrain_pgd_eps_{epsilon}.csv")
    X_adv_df.to_csv(filename, index=False)
    print(f"Saved PGD retraining samples to {filename}")
    
    # Save to additional 'data' folder
    data_output_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_output_dir, exist_ok=True)
    data_filename = os.path.join(data_output_dir, f"adversarial_pgd.csv")
    X_adv_df.to_csv(data_filename, index=False)
    print(f"Saved PGD retraining samples to {data_filename}")
    
    return X_adv_df

if __name__ == "__main__":
    print("This module is not intended to be run directly. Import and use its functions instead.")
