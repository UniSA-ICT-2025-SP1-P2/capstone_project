from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import traceback
import joblib
import numpy as np
import shutil

# Custom modules
from prototype.baseline_models.baseline_models_webUI import run_all_models, find_category_name, \
    evaluate_models as evaluate_adversarial_models
from prototype.attack_simulation.attack_sim import run_attack_simulation, run_pgd_attack_simulation, \
    generate_adversarial_samples_for_retraining, generate_pgd_samples_for_retraining
from prototype.results_presentation.concept_drift_presentation import visualize_concept_drift
from prototype.results_presentation.defence_results_presentation import create_advanced_performance_comparison
from prototype.defence_prototype.src.defences import adversarial_training, feature_smoothing, ensemble_learning, \
    concept_drift
from prototype.defence_prototype.src.evaluate_defences import run_evaluation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Base paths - centralized for consistency
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, "baseline_models", "trained_baseline_models")
DATA_DIR = os.path.join(BASE_DIR, "baseline_models", "baseline_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_simulation_results")
DEFENCE_DIR = os.path.join(BASE_DIR, "defence_prototype")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ATTACK_RESULTS_DIR = os.path.join(BASE_DIR, 'attack_simulation', 'attack_simulation_results')
ADVERSARIAL_DATA_DIR = os.path.join(BASE_DIR, "data")

# Create necessary directories
for directory in [OUTPUT_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_models():
    """Load all trained ML models from the model directory"""
    models = {}
    model_names = ["RandomForest", "KNN", "LogisticRegression", "DecisionTree", "SVM"]

    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"✅ Loaded model: {name}")
            except Exception as e:
                print(f"❌ Failed to load model {name}: {str(e)}")

    return models


def load_test_data():
    """Load test dataset and related encoders"""
    test_path = os.path.join(DATA_DIR, "Test_Dataset.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_df = pd.read_csv(test_path)

    # Load encoders and feature names
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if not os.path.exists(label_encoder_path) or not os.path.exists(feature_names_path):
        raise FileNotFoundError("Required model metadata files not found")

    label_encoder = joblib.load(label_encoder_path)
    feature_names = joblib.load(feature_names_path)

    # Prepare features and labels
    X_test = test_df[feature_names]
    y_test = pd.Series(label_encoder.transform(test_df['category_name']), index=test_df.index)
    y_test_original = test_df['category_name']

    return X_test, y_test, y_test_original, label_encoder, feature_names


def load_adversarial_features():
    """Load features that can be modified in adversarial attacks"""
    path = os.path.join(DATA_DIR, "Adversarial_Features.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df['Feature'].tolist()
    print("⚠️ Adversarial features file not found, using empty list")
    return []


@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and model evaluation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file type, please upload a CSV file'}), 400

    filename = os.path.basename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    selected_models = request.form.getlist("models")
    if not selected_models:
        selected_models = ["RandomForest", "KNN", "LogisticRegression", "DecisionTree", "SVM"]  # Default to all

    try:
        df = pd.read_csv(filepath)

        # Save the original uploaded dataset, but drop 'Class' only in the saved file
        saved_df = df.drop(columns=['Class'], errors='ignore')  # Keep 'df' in memory untouched

        # Path 1: Save in baseline_data
        baseline_path = os.path.join(BASE_DIR, "baseline_models", "baseline_data", "uploaded_dataset.csv")
        saved_df.to_csv(baseline_path, index=False)

        # Path 2: Save in attack_simulation_results
        attack_path = os.path.join(OUTPUT_DIR, "uploaded_dataset.csv")
        saved_df.to_csv(attack_path, index=False)
        print(f"📁 Uploaded dataset saved to:\n  - {baseline_path}\n  - {attack_path} (both without 'Class')")

        # Path 3: Save in prototype/data for adversarial training
        adversarial_training_path = os.path.join(BASE_DIR, "data", "uploaded_dataset.csv")
        saved_df.to_csv(adversarial_training_path, index=False)

        results = run_all_models(df, selected_models=selected_models)

        # Get SHAP values if available
        shap_path = os.path.join(DATA_DIR, "Classifier_Results.xlsx")
        shap_df = pd.DataFrame()
        if os.path.exists(shap_path):
            try:
                shap_df = pd.read_excel(shap_path, sheet_name="SHAP_Features")
            except Exception as e:
                print(f"Warning: Could not load SHAP values: {str(e)}")

        transformed = {
            "status": results.get("status"),
            "message": results.get("message"),
            "adversarial_features": results.get("adversarial_features", []),
            "results": {}
        }

        # Process results for each model
        for model_name, data in results.get("results", {}).items():
            report = data.get("report", {})
            model_path = data.get("model_path")
            shap_image = data.get("shap_image")

            # Get feature count
            feature_count = 0
            try:
                model = joblib.load(model_path)
                if hasattr(model, 'named_steps') and hasattr(model.named_steps['clf'], 'n_features_in_'):
                    feature_count = model.named_steps['clf'].n_features_in_
                elif hasattr(model, 'n_features_in_'):
                    feature_count = model.n_features_in_
                else:
                    feature_count = len(results.get('adversarial_features', []))
            except Exception as e:
                print(f"Warning: Could not get feature count for {model_name}: {str(e)}")
                feature_count = len(results.get('adversarial_features', []))

            # Extract SHAP values
            shap_values = {}
            if not shap_df.empty:
                model_shap = shap_df[shap_df["Classifier"] == model_name]
                if not model_shap.empty:
                    top_features = model_shap.nlargest(10, "SHAP Importance")
                    shap_values = dict(zip(top_features["Feature"], top_features["SHAP Importance"]))

            transformed["results"][model_name] = {
                "accuracy": report.get("accuracy"),
                "f1_score": report.get("macro avg", {}).get("f1-score"),
                "precision": report.get("macro avg", {}).get("precision"),
                "recall": report.get("macro avg", {}).get("recall"),
                "model_details": {
                    "type": model_name,
                    "features": f"{feature_count} features used"
                },
                "shap_image": shap_image if shap_image and os.path.exists(shap_image) else None,
                "shap_values": shap_values
            }

        return jsonify(transformed)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/generate_adversarial', methods=['POST'])
def generate_adversarial():
    """Generate adversarial examples using FGSM or PGD attacks"""
    try:
        attack_type = request.form.get('attack_type', 'fgsm')
        epsilon = float(request.form.get('epsilon', 0.1))

        if attack_type not in ['fgsm', 'pgd']:
            return jsonify({'error': 'Invalid attack type. Choose "fgsm" or "pgd".'}), 400
        if epsilon <= 0 or epsilon > 1.0:
            return jsonify({'error': 'Epsilon must be between 0 and 1.0'}), 400

        models = load_models()
        X_test, y_test_encoded, y_test_original, label_encoder, feature_names = load_test_data()
        features_to_lock = load_adversarial_features()

        valid_models = ["LogisticRegression", "SVM"]
        available_valid_models = [m for m in valid_models if m in models]
        if not available_valid_models:
            return jsonify({'error': 'No gradient-based models available for attack generation.'}), 400

        source_model = available_valid_models[0]
        print(f"Running {attack_type.upper()} attack using {source_model}, epsilon={epsilon}")

        # === Generate adversarial samples ===
        sample_path = os.path.join(OUTPUT_DIR, f"adversarial_samples_{attack_type}_eps_{epsilon}.csv")
        if os.path.exists(sample_path):
            os.remove(sample_path)
            print(f"🧹 Removed old adversarial samples at: {sample_path}")

        if attack_type == 'fgsm':
            run_attack_simulation(models, X_test, y_test_encoded, [epsilon], features_to_lock, source_model)
        else:
            run_pgd_attack_simulation(models, X_test, y_test_encoded, [epsilon], features_to_lock, source_model)

        if not os.path.exists(sample_path):
            return jsonify({'error': f'Adversarial samples file not found at {sample_path}'}), 500

        adv_df = pd.read_csv(sample_path)
        print(f"🔍 Successfully generated {len(adv_df)} adversarial samples at epsilon={epsilon}")
        adv_df['source'] = 'adversarial'

        if 'Class' not in adv_df.columns:
            print("Warning: 'Class' column not found in adversarial samples. Adding default class 'Conti'.")
            adv_df['Class'] = 'Conti'

        adv_df = adv_df.drop(columns=[col for col in adv_df.columns if col not in feature_names + ['Class', 'Category', 'source']], errors='ignore')
        for f in feature_names:
            if f not in adv_df.columns:
                adv_df[f] = 0

        # === Load full original dataset ===
        original_path = os.path.join(BASE_DIR, "baseline_models", "baseline_data", "uploaded_dataset.csv")
        if not os.path.exists(original_path):
            return jsonify({'error': 'Uploaded dataset not found at expected location.'}), 400
        original_df = pd.read_csv(original_path)
        original_df["source"] = "original"

        if "category_name" in original_df.columns:
            original_df["label"] = original_df["category_name"]
        elif "Category" in original_df.columns:
            original_df["label"] = original_df["Category"].apply(find_category_name)
        elif "Class" in original_df.columns:
            original_df["label"] = original_df["Class"]
        else:
            original_df["label"] = "Unknown"

        for f in feature_names:
            if f not in original_df.columns:
                original_df[f] = 0
        original_df = original_df[feature_names + ['label', 'source']]

        # === Prepare adversarial data ===
        adv_df = adv_df.rename(columns={"Class": "label"})
        if "Category" in adv_df.columns:
            adv_df["label"] = adv_df["Category"].apply(find_category_name)
        adv_df = adv_df[feature_names + ['label', 'source']]

        original_df.reset_index(drop=True, inplace=True)
        adv_df.reset_index(drop=True, inplace=True)
        combined_df = pd.concat([original_df, adv_df], ignore_index=True)

        print(f"🔢 Combined dataset contains {len(combined_df)} rows ({len(original_df)} original + {len(adv_df)} adversarial)")

        X_combined = combined_df[feature_names].astype(float)
        try:
            y_combined = label_encoder.transform(combined_df["label"])
        except ValueError as e:
            print(f"Warning: Label encoding error: {str(e)}")
            unknown_labels = set(combined_df["label"]) - set(label_encoder.classes_)
            if unknown_labels:
                print(f"Extending label encoder with: {unknown_labels}")
                label_encoder.classes_ = np.concatenate([label_encoder.classes_, list(unknown_labels)])
            y_combined = label_encoder.transform(combined_df["label"])

        # === Save merged dataset ===
        combined_path = os.path.join(OUTPUT_DIR, f"combined_dataset_with_adversarial_eps_{epsilon}.csv")
        latest_merged_path = os.path.join(OUTPUT_DIR, "latest_combined_dataset.csv")
        combined_df.drop(columns=["source"], inplace=True, errors="ignore")
        combined_df.drop(columns=['Class_encoded', 'category_encoded'], inplace=True, errors='ignore')
        combined_df.to_csv(combined_path, index=False)
        combined_df.to_csv(latest_merged_path, index=False)
        print(f"📝 Saved merged dataset as: {latest_merged_path}")

        # In your Flask code, right before calling the generation function, add:
        print("Categories before generation:")
        test_malware_mask = y_test_encoded != 2
        original_test_categories = y_test_original[test_malware_mask]
        print(f"Sample of original malware categories: {original_test_categories.head(10).tolist()}")

        # === Let the function handle filtering and label alignment ===
        try:
            if attack_type == 'fgsm':
                retrain_df = generate_adversarial_samples_for_retraining(
                    model=models[source_model],
                    X=X_test,
                    y=y_test_encoded,
                    epsilon=epsilon,
                    n_samples=100,
                    target_class="Benign"
                )

                # Fix the categories after generation
                malware_mask = y_test_encoded != 2
                original_malware_categories = y_test_original[malware_mask]

                # Get the first 100 malware categories (since n_samples=100)
                correct_categories = original_malware_categories.iloc[:100].tolist()
                retrain_df['Category'] = correct_categories

                # Re-save the corrected CSV
                retrain_df.to_csv(
                    "C:/Users/maxam/PycharmProjects/capstone_project2/prototype/data/adversarial_fgsm.csv", index=False)
                print(f"Fixed categories: {pd.Series(correct_categories).value_counts()}")


            elif attack_type == 'pgd':
                print("🚀 Starting adversarial generation (PGD)...")
                retrain_df = generate_pgd_samples_for_retraining(
                    model=models[source_model],
                    X=X_test,
                    y=y_test_encoded,
                    epsilon=epsilon,
                    n_samples=100,
                    target_class='Benign'
                )
                # Fix the categories after generation
                malware_mask = y_test_encoded != 2
                original_malware_categories = y_test_original[malware_mask]
                # Get the first 100 malware categories (since n_samples=100)
                correct_categories = original_malware_categories.iloc[:100].tolist()
                retrain_df['Category'] = correct_categories
                # Re-save the corrected CSV

                retrain_df.to_csv(
                    "C:/Users/maxam/PycharmProjects/capstone_project2/prototype/data/adversarial_pgd.csv", index=False)
                print(f"✅ PGD generation completed! Generated {len(retrain_df)} samples")
                print(f"📦 Fixed categories: {pd.Series(correct_categories).value_counts()}")

        except Exception as e:
            print(f"⚠️ Failed to save adversarial training file: {str(e)}")

        # === Evaluate models ===
        combined_df["Category"] = combined_df["label"]
        raw_results = evaluate_adversarial_models(combined_df)
        evaluation_reports = {}
        for model_name, report in raw_results.items():
            evaluation_reports[model_name] = {
                "accuracy": report.get("accuracy"),
                "f1_score": report.get("macro avg", {}).get("f1-score"),
                "precision": report.get("macro avg", {}).get("precision"),
                "recall": report.get("macro avg", {}).get("recall"),
                "model_details": {
                    "type": model_name,
                    "features": f"{len(combined_df.columns) - 2} features used"
                },
                "shap_image": None,
                "shap_values": {}
            }

        return jsonify({
            'message': f'{attack_type.upper()} adversarial attack completed successfully.',
            'evaluation': evaluation_reports,
            'output_file': combined_path
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/apply_defences', methods=['POST'])
def apply_defences():
    """Apply selected defense mechanisms to the models/data"""
    try:
        selected = request.form.getlist('defences')
        if not selected:
            return jsonify({'message': 'No defenses selected.'}), 400

        responses = []
        evaluation_reports = {}

        # === Run Baseline Evaluation (None Defence) ===
        try:
            clean_data_path = os.path.join(DATA_DIR, 'Test_Dataset.csv')
            if os.path.exists(clean_data_path):
                df = pd.read_csv(clean_data_path)
                label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
                feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

                y_true = label_encoder.transform(df['category_name'])  # adjust if needed
                X = df[feature_names].values

                from prototype.defence_prototype.src.evaluate_defences import evaluate
                results = []
                for model_type in ['rf', 'nn', 'nn_adv', 'ensemble']:
                    result = evaluate(
                        X, y_true, model_type, defence_name='none', data_type='none',
                        label_encoder=label_encoder, results_dir=OUTPUT_DIR
                    )
                    results.append(result)
                    print(
                        f"📊 Baseline Evaluation: {model_type} - Acc: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")

                # Save results
                results_csv_path = os.path.join(OUTPUT_DIR, 'defence_results.csv')
                df_results = pd.DataFrame(results)
                if os.path.exists(results_csv_path):
                    existing = pd.read_csv(results_csv_path)
                    updated = pd.concat([existing, df_results], ignore_index=True)
                else:
                    updated = df_results
                updated.to_csv(results_csv_path, index=False)

                import shutil
                shutil.copyfile(
                    results_csv_path,
                    os.path.join(BASE_DIR, 'defence_prototype', 'results', 'defence_results.csv')
                )

        except Exception as e:
            print(f"❌ Failed to run baseline evaluation: {str(e)}")
            responses.append(f"❌ Baseline evaluation failed: {str(e)}")

        def evaluate_all_models_for_defence(data_path, defence_name, data_type, label_col='label'):
            try:
                df = pd.read_csv(data_path)
                label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
                feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

                if label_col not in df.columns:
                    raise ValueError(f"Expected label column '{label_col}' not found in dataset.")

                y_true = label_encoder.transform(df[label_col])
                X = df[feature_names].values

                from prototype.defence_prototype.src.evaluate_defences import evaluate
                results = []
                for model_type in ['rf', 'nn', 'nn_adv', 'ensemble']:
                    result = evaluate(
                        X, y_true, model_type, defence_name, data_type, label_encoder, OUTPUT_DIR
                    )
                    results.append(result)

                    print(f"📊 {defence_name.title()} Evaluation: {result['model']} - "
                          f"Acc: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")

                results_csv_path = os.path.join(OUTPUT_DIR, 'defence_results.csv')
                results_df = pd.DataFrame(results)
                if os.path.exists(results_csv_path):
                    existing = pd.read_csv(results_csv_path)
                    updated = pd.concat([existing, results_df], ignore_index=True)
                else:
                    updated = results_df
                updated.to_csv(results_csv_path, index=False)

                import shutil
                shutil.copyfile(
                    results_csv_path,
                    os.path.join(BASE_DIR, 'defence_prototype', 'results', 'defence_results.csv')
                )

                return []
            except Exception as e:
                print(f"⚠️ Evaluation for {defence_name} failed: {str(e)}")
                return []

        if 'feature_smoothing' in selected:
            try:
                noise_std = float(request.form.get('noise_std', 0.02))
                input_path = os.path.join(OUTPUT_DIR, 'latest_combined_dataset.csv')
                output_path = os.path.join(OUTPUT_DIR, 'latest_combined_dataset_smoothed.csv')

                feature_smoothing.apply_feature_smoothing_path(
                    input_path=input_path,
                    noise_std=noise_std,
                    output_path=output_path
                )
                responses.append(f'✅ Feature Smoothing applied with noise level {noise_std}')

                eval_msgs = evaluate_all_models_for_defence(
                    data_path=output_path,
                    defence_name='feature_smoothing',
                    data_type='clean_smoothed'
                )
                responses.extend(eval_msgs)

            except Exception as e:
                print(f"Error applying feature smoothing: {str(e)}")
                responses.append(f'❌ Feature Smoothing failed: {str(e)}')

        if 'adversarial_training' in selected:
            try:
                epochs = int(request.form.get('epochs', 10))
                lr = float(request.form.get('lr', 0.001))

                adversarial_training.run_adversarial_training(
                    data_dir=ADVERSARIAL_DATA_DIR,
                    model_dir=MODEL_DIR,
                    epochs=epochs,
                    lr=lr
                )
                responses.append(f'✅ Adversarial Training completed with {epochs} epochs and model updated.')

                full_path = os.path.join(OUTPUT_DIR, 'latest_combined_dataset.csv')
                responses += evaluate_all_models_for_defence(
                    data_path=full_path,
                    defence_name='adversarial_training',
                    data_type='clean'
                )

            except Exception as e:
                print(f"Error during adversarial training: {str(e)}")
                responses.append(f'❌ Adversarial Training failed: {str(e)}')

        if 'concept_drift' in selected:
            try:
                chunk_size = int(request.form.get('chunk_size', 500))
                threshold = float(request.form.get('threshold', 0.7))
                input_path = os.path.join(DATA_DIR, 'uploaded_dataset_label.csv')

                feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
                df = pd.read_csv(input_path)

                if 'label' not in df.columns:
                    raise ValueError("Expected 'label' column not found in dataset.")

                df_filtered = df[feature_names + ['label']]

                filtered_path = os.path.join(OUTPUT_DIR, 'concept_drift_cleaned.csv')
                df_filtered.to_csv(filtered_path, index=False)

                concept_drift.run_concept_drift(
                    data_path=filtered_path,
                    model_type='ensemble',
                    chunk_size=chunk_size,
                    threshold=threshold,
                    results_dir=OUTPUT_DIR
                )

                import shutil
                src = os.path.join(OUTPUT_DIR, 'concept_drift_results.csv')
                dst = os.path.join(BASE_DIR, 'defence_prototype', 'results', 'concept_drift_results.csv')
                shutil.copyfile(src, dst)

                responses.append(f'🧠 Concept Drift applied with chunk size {chunk_size} and threshold {threshold}.')
                responses += evaluate_all_models_for_defence(
                    data_path=filtered_path,
                    defence_name='concept_drift',
                    data_type='concept_drift'
                )
            except Exception as e:
                print(f"Error during concept drift: {str(e)}")
                responses.append(f'❌ Concept Drift failed: {str(e)}')

        if 'ensemble_learning' in selected:
            try:
                original_data_path = os.path.join(
                    BASE_DIR, 'baseline_models', 'baseline_data', 'uploaded_dataset_label.csv'
                )

                df = pd.read_csv(original_data_path)

                if 'label' not in df.columns:
                    raise ValueError("Expected 'label' column not found in dataset.")

                if df['label'].dtype != object:
                    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
                    df['label'] = label_encoder.inverse_transform(df['label'])

                temp_path = os.path.join(OUTPUT_DIR, 'uploaded_dataset_temp.csv')
                df.to_csv(temp_path, index=False)

                ensemble_result = ensemble_learning.run_ensemble_evaluation(
                    data_path=temp_path,
                    model_dir=MODEL_DIR
                )

                acc = ensemble_result.get("accuracy", 0)
                responses.append(f'🧩 Ensemble Learning applied (Accuracy: {acc:.2f}).')
                responses += evaluate_all_models_for_defence(
                    data_path=temp_path,
                    defence_name='ensemble_learning',
                    data_type='ensemble'
                )

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"Error during ensemble learning: {str(e)}")
                responses.append(f'❌ Ensemble Learning failed: {str(e)}')

        allowed_models = ["rf", "nn", "nn_adv", "ensemble"]
        filtered_eval = {k: v for k, v in evaluation_reports.items() if k in allowed_models}

        return jsonify({
            'message': responses
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_models', methods=['POST'])
def evaluate_models_route():
    """Evaluate models on different datasets (clean/adversarial/smoothed)"""
    try:
        # Define all available datasets
        data_files = {
            'clean': os.path.join(DATA_DIR, 'uploaded_dataset.csv'),
            'fgsm': os.path.join(DATA_DIR, 'adversarial_fgsm_label.csv'),
            'pgd': os.path.join(DATA_DIR, 'adversarial_pgd_label.csv'),
        }

        # Print which datasets exist
        for key, path in data_files.items():
            if os.path.exists(path):
                print(f"✅ Found dataset: {key} → {path}")
            else:
                print(f"⚠️ Missing dataset: {key} → {path}")

        # Filter only those that actually exist
        data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}

        if not data_files:
            return jsonify({'error': 'No valid datasets found for evaluation'}), 400

        # Run evaluation
        results = run_evaluation(
            data_files=data_files,
            model_types=['rf', 'nn', 'nn_adv', 'ensemble'],
            defences=['none', 'feature_smoothing']
        )

        # Return results as JSON
        results_json = results.to_dict(orient='records')
        return jsonify({
            'message': f'Evaluated models on {len(data_files)} datasets',
            'datasets_used': list(data_files.keys()),
            'evaluation': results_json
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/run_concept_drift', methods=['GET'])
def run_concept_drift():
    """Generate and return concept drift visualization"""
    try:
        # 1. Generate the image
        visualize_concept_drift()  # saves it to known location

        # 2. Copy it to static dir (Flask can serve it easily)
        src = os.path.join(BASE_DIR, 'defence_prototype', 'results', 'concept_drift_analysis.png')
        dst = os.path.join(STATIC_DIR, 'concept_drift_analysis.png')
        if not os.path.exists(src):
            return jsonify({'error': 'Image generation failed'}), 500

        import shutil
        shutil.copyfile(src, dst)

        # 3. Return the static path as JSON
        return jsonify({'image_path': f'/static/concept_drift_analysis.png'})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/run_defence_results', methods=['GET'])
def run_defence_results():
    """Generate and return defence results summary chart"""
    try:
        full_path = os.path.join(BASE_DIR, 'defence_prototype', 'results', 'defence_results.csv')
        if not os.path.exists(full_path):
            return jsonify({'error': 'Defence results CSV not found'}), 500

        # Step 1: Load full CSV and filter to defences that are safe to compare
        df = pd.read_csv(full_path)
        valid_defences = ['none', 'feature_smoothing']
        df_filtered = df[df['defence'].isin(valid_defences)]

        # Step 2: Save filtered CSV temporarily
        temp_filtered_path = os.path.join(BASE_DIR, 'defence_prototype', 'results', 'defence_results_filtered.csv')
        df_filtered.to_csv(temp_filtered_path, index=False)

        # Step 3: Monkey patch load_csv_data to return filtered data
        import prototype.results_presentation.defence_results_presentation as drp
        original_loader = drp.load_csv_data

        def filtered_loader():
            return pd.read_csv(temp_filtered_path)

        drp.load_csv_data = filtered_loader

        try:
            # Step 4: Run visualisation
            fig = drp.create_advanced_performance_comparison()

            # Step 5: Copy image output to static folder
            src = os.path.join(BASE_DIR, 'defence_prototype', 'results', 'defence_results.png')
            dst = os.path.join(STATIC_DIR, 'final_defence_summary.png')
            if not os.path.exists(src):
                return jsonify({'error': 'Defence results image was not generated'}), 500

            import shutil
            shutil.copyfile(src, dst)

        finally:
            # Step 6: Restore original loader
            drp.load_csv_data = original_loader

        return jsonify({'image_path': '/static/final_defence_summary.png'})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/run_model_results', methods=['GET'])
def run_model_results():
    """Generate and return model results summary chart without displaying via plt"""
    try:
        import os
        import pandas as pd
        import traceback
        from results_presentation import model_results

        # Define expected file paths
        fgsm_path = os.path.join(BASE_DIR, "attack_simulation_results", "attack_results_fgsm_summary.csv")
        pgd_path = os.path.join(BASE_DIR, "attack_simulation_results", "attack_results_pgd_summary.csv")

        # Check file existence
        if not os.path.exists(fgsm_path):
            return jsonify({'error': f'FGSM file not found at: {fgsm_path}'}), 400
        if not os.path.exists(pgd_path):
            return jsonify({'error': f'PGD file not found at: {pgd_path}'}), 400

        # Patch load_csv_data to return the correct DataFrames
        def patched_load_csv_data():
            try:
                df_fgsm = pd.read_csv(fgsm_path)
                df_pgd = pd.read_csv(pgd_path)
                return df_fgsm, df_pgd
            except Exception as e:
                print(f"Error loading CSV data: {e}")
                return None, None

        # Temporarily override the original
        original_loader = model_results.load_csv_data
        model_results.load_csv_data = patched_load_csv_data

        try:
            # Generate the matplotlib figure
            fig = model_results.create_model_results_table()

            if fig is None:
                return jsonify({'error': 'Failed to generate the figure'}), 500

            # Save the figure as an image file (skip plt.show() and plt.close())
            output_path = os.path.join(STATIC_DIR, 'model_results_summary.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')

            return jsonify({'image_path': '/static/model_results_summary.png'})

        finally:
            # Restore original function
            model_results.load_csv_data = original_loader

    except Exception as e:
        print("Detailed error:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if required directories exist
    for path in [MODEL_DIR, DATA_DIR]:
        if not os.path.exists(path):
            print(f"Warning: Required directory not found: {path}")

    # Start Flask app
    app.run(debug=True)
