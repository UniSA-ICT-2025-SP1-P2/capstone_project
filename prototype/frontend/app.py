from flask import Flask, request, jsonify, render_template
from sklearn.metrics import classification_report
import os
import pandas as pd
import traceback
import joblib

# Custom modules
from prototype.baseline_models.baseline_models_webUI import run_all_models
from prototype.attack_simulation.attack_sim import run_attack_simulation, run_pgd_attack_simulation
from prototype.results_presentation.concept_drift_presentation import visualize_concept_drift
from prototype.results_presentation.defence_results_presentation import visualise_defence_results
from prototype.results_presentation.model_results import load_csv_data, create_model_results_table
from prototype.defence_prototype.src.defences import adversarial_training, feature_smoothing, ensemble_learning
from prototype.defence_prototype.src.evaluate_defences import run_evaluation
from prototype.baseline_models.baseline_models_webUI import find_category_name

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, "baseline_models", "trained_baseline_models")
DATA_DIR = os.path.join(BASE_DIR, "baseline_models", "baseline_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_simulation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    models = {}
    for name in ["RandomForest", "KNN", "LogisticRegression", "DecisionTree", "SVM"]:
        path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

def load_test_data():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Test_Dataset.csv"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    X_test = test_df[feature_names]
    y_test = pd.Series(label_encoder.transform(test_df['category_name']), index=test_df.index)
    return X_test, y_test, label_encoder, feature_names

def load_adversarial_features():
    path = os.path.join(DATA_DIR, "Adversarial_Features.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df['Feature'].tolist()
    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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

    try:
        df = pd.read_csv(filepath)
        results = run_all_models(df, selected_models=selected_models)

        shap_path = os.path.join(DATA_DIR, "Classifier_Results.xlsx")
        shap_df = pd.read_excel(shap_path, sheet_name="SHAP_Features") if os.path.exists(shap_path) else pd.DataFrame()

        transformed = {
            "status": results.get("status"),
            "message": results.get("message"),
            "adversarial_features": results.get("adversarial_features", []),
            "results": {}
        }

        for model_name, data in results.get("results", {}).items():
            report = data.get("report", {})
            model_path = data.get("model_path")
            shap_image = data.get("shap_image")

            try:
                model = joblib.load(model_path)
                feature_count = model.named_steps['clf'].n_features_in_ if hasattr(model, 'named_steps') else model.n_features_in_
            except:
                feature_count = len(results.get('adversarial_features', []))

            shap_values = {}
            if not shap_df.empty:
                model_shap = shap_df[shap_df["Classifier"] == model_name]
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
    try:
        attack_type = request.form.get('attack_type')
        epsilon = float(request.form.get('epsilon', 0.1))

        models = load_models()
        X_test, y_test, label_encoder, feature_names = load_test_data()
        features_to_lock = load_adversarial_features()

        valid_models = ["LogisticRegression", "SVM"]
        available_valid_models = [m for m in valid_models if m in models]

        if not available_valid_models:
            return jsonify({'error': 'No gradient-based models available.'}), 400

        source_model = available_valid_models[0]
        print(f"Running {attack_type.upper()} using {source_model}, epsilon={epsilon}")

        # Run attack
        if attack_type == 'fgsm':
            sample_path = os.path.join(OUTPUT_DIR, f"adversarial_samples_fgsm_eps_{epsilon}.csv")
            run_attack_simulation(models, X_test, y_test, [epsilon], features_to_lock, source_model)
        elif attack_type == 'pgd':
            sample_path = os.path.join(OUTPUT_DIR, f"adversarial_samples_pgd_eps_{epsilon}.csv")
            run_pgd_attack_simulation(models, X_test, y_test, [epsilon], features_to_lock, source_model)
        else:
            return jsonify({'error': 'Invalid attack type selected.'}), 400

        if not os.path.exists(sample_path):
            return jsonify({'error': 'Adversarial samples file not found.'}), 500

        adv_df = pd.read_csv(sample_path)
        if 'Class' not in adv_df.columns:
            return jsonify({'error': 'Missing "Class" column in adversarial samples.'}), 500

        # Keep only relevant columns
        adv_df = adv_df.drop(columns=[col for col in adv_df.columns if col not in feature_names + ['Class']], errors='ignore')
        for f in feature_names:
            if f not in adv_df.columns:
                adv_df[f] = 0
        adv_df = adv_df[feature_names + ['Class']]

        # Load original uploaded data
        uploaded_file = os.listdir(app.config['UPLOAD_FOLDER'])[0]
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
        original_df = pd.read_csv(original_path)

        # Replace generic "Malware" with actual category using find_category_name
        if 'Category' in original_df.columns:
            original_df["category_name"] = original_df["Category"].apply(find_category_name)
        else:
            return jsonify({'error': 'Missing "Category" column for mapping.'}), 500

        original_df["Class"] = original_df.apply(
            lambda row: row["category_name"] if row["Class"] == "Malware" else row["Class"], axis=1
        )

        original_df = original_df[feature_names + ['Class']]

        # Merge original and adversarial
        combined_df = pd.concat([original_df, adv_df], ignore_index=True)

        # Final fix: map 'Malware' → actual category if any rows slipped through
        combined_df["Class"] = combined_df["Class"].replace("Malware", "Conti")  # fallback only

        # Encode using trained label encoder (category_name-style)
        X_combined = combined_df[feature_names].astype(float)
        y_combined = label_encoder.transform(combined_df["Class"])

        model = models[source_model]
        y_pred = model.predict(X_combined)
        combined_df["category_name"] = label_encoder.inverse_transform(y_pred)

        combined_df = combined_df.rename(columns={"Class": "label"})

        # Save files
        combined_path = os.path.join(OUTPUT_DIR, f"combined_dataset_with_adversarial_eps_{epsilon}.csv")
        combined_df.to_csv(combined_path, index=False)
        latest_merged_path = os.path.join(OUTPUT_DIR, "latest_combined_dataset.csv")
        combined_df.to_csv(latest_merged_path, index=False)
        print(f"📝 Also saved merged dataset as: {latest_merged_path}")

        report = classification_report(y_combined, y_pred, output_dict=True, zero_division=0)
        evaluation = {
            'accuracy': (y_pred == y_combined).mean(),
            'f1_score': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }

        return jsonify({
            'message': f'{attack_type.upper()} adversarial samples merged and evaluated.',
            'evaluation': evaluation,
            'output_file': combined_path
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/apply_defences', methods=['POST'])
def apply_defences():
    try:
        responses = []
        selected = request.form.getlist('defences')
        X_test, y_test, label_encoder, feature_names = load_test_data()

        # === Feature Smoothing: save smoothed test dataset ===
        if 'feature_smoothing' in selected:
            original_path = os.path.join(DATA_DIR, 'Test_Dataset.csv')
            smoothed_path = os.path.join(DATA_DIR, 'Test_Dataset_smoothed.csv')

            feature_smoothing.apply_feature_smoothing_path(
                input_path=original_path,
                noise_std=0.02,
                output_path=smoothed_path
            )
            responses.append('✅ Feature Smoothing applied and saved.')

        # === Adversarial Training: retrain and overwrite neural_net.pt ===
        if 'adversarial_training' in selected:
            from prototype.defence_prototype.src.defences.adversarial_training import run_adversarial_training
            run_adversarial_training(data_dir=DATA_DIR, model_dir=MODEL_DIR, epochs=10, lr=0.001)
            responses.append('✅ Adversarial Training completed and model updated.')

        # === Concept Drift: generate visual only ===
        if 'concept_drift' in selected:
            visualize_concept_drift(save_path='static/concept_drift_analysis.png')
            responses.append('🧠 Concept Drift visualised (no model impact).')

        # === Ensemble Learning: overwrite ensemble models if needed ===
        if 'ensemble_learning' in selected:
            from prototype.defence_prototype.src.defences.ensemble_learning import run_ensemble_evaluation
            ensemble_result = run_ensemble_evaluation(
                data_path=os.path.join(DATA_DIR, 'Test_Dataset.csv'),  # or smoothed path if needed
                model_dir=MODEL_DIR
            )
            responses.append(f'🧩 Ensemble Learning applied (Acc: {ensemble_result["accuracy"]:.2f}).')

        return jsonify({'message': responses})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_models', methods=['POST'])
def evaluate_models():
    try:
        data_files = {}

        combined_path = os.path.join(OUTPUT_DIR, 'latest_combined_dataset.csv')
        smoothed_path = os.path.join(DATA_DIR, 'Test_Dataset_smoothed.csv')
        default_path = os.path.join(DATA_DIR, 'Test_Dataset.csv')

        # Priority: Combined (adversarial) > Smoothed > Default
        if os.path.exists(combined_path):
            data_files['clean'] = combined_path
        elif os.path.exists(smoothed_path):
            data_files['clean'] = smoothed_path
        else:
            data_files['clean'] = default_path

        results = run_evaluation(data_files=data_files)
        return jsonify({'evaluation': results.to_dict(orient='records')})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/run_concept_drift', methods=['GET'])
def run_concept_drift():
    try:
        img_path = 'static/concept_drift_analysis.png'
        visualize_concept_drift(save_path=img_path)
        return jsonify({'image_path': '/' + img_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_defence_results', methods=['GET'])
def run_defence_results():
    try:
        img_path = 'static/defence_results.png'
        visualise_defence_results(save_path=img_path)
        return jsonify({'image_path': '/' + img_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_model_results', methods=['GET'])
def run_model_results():
    try:
        df = load_csv_data()
        img = create_model_results_table(df)
        output_path = 'static/model_results_table.png'
        img.save(output_path)
        return jsonify({'image_path': '/' + output_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
