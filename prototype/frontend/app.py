from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from werkzeug.utils import secure_filename
import traceback

# Custom modules (update paths if necessary)
from prototype.baseline_models.baseline_models_webUI import run_all_models
from prototype.results_presentation.concept_drift_presentation import visualize_concept_drift
from prototype.results_presentation.defence_results_presentation import visualise_defence_results
from prototype.results_presentation.model_results import load_csv_data, create_model_results_table
from prototype.defence_prototype.src.defences import adversarial_training, feature_smoothing, ensemble_learning
from prototype.defence_prototype.src.evaluate_defences import run_evaluation
from prototype.attack_simulation.attack_sim import fgsm_attack_simulation, pgd_attack_simulation
from prototype.defence_prototype.src.train_models import train_models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded dataset with shape {df.shape}")
        results = run_all_models(df)

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

            # Determine number of input features
            feature_count = len(results.get('adversarial_features', []))  # fallback
            try:
                from joblib import load
                model = load(model_path)
                if hasattr(model, 'named_steps'):
                    feature_count = model.named_steps['clf'].n_features_in_
                elif hasattr(model, 'n_features_in_'):
                    feature_count = model.n_features_in_
            except Exception as e:
                print(f"[WARN] Could not determine feature count for {model_name}: {e}")

            model_result = {
                "accuracy": report.get("accuracy"),
                "f1_score": report.get("macro avg", {}).get("f1-score"),
                "precision": report.get("macro avg", {}).get("precision"),
                "recall": report.get("macro avg", {}).get("recall"),
                "model_details": {
                    "type": model_name,
                    "features": f"{feature_count} features used"
                },
                "shap_values": {},  # optional
                "shap_image": shap_image
            }

            transformed["results"][model_name] = model_result

        print(f"[INFO] Transformed result ready.")
        return jsonify(transformed)

    except Exception as e:
        print("[ERROR] Exception occurred during processing:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/train_models', methods=['POST'])
def trigger_training():
    try:
        # Define paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, 'prototype', 'defence_prototype'))
        PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))
        DATA_PATH = os.path.join(PROTOTYPE_DIR, 'data', 'train_label.csv')
        MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')

        # Call the training function
        train_models(DATA_PATH, MODEL_DIR)

        return jsonify({'message': 'Model training initiated successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_adversarial', methods=['POST'])
def generate_adversarial():
    try:
        attack_type = request.form.get('attack_type')
        epsilon = float(request.form.get('epsilon', 0.1))

        # Assuming your data and model loading is handled elsewhere or inside the simulation functions
        if attack_type == 'fgsm':
            fgsm_attack_simulation(epsilon=epsilon)
        elif attack_type == 'pgd':
            pgd_attack_simulation(epsilon=epsilon)
        else:
            return jsonify({'error': 'Invalid attack type'}), 400

        return jsonify({'message': f'{attack_type.upper()} adversarial samples generated successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_defences', methods=['POST'])
def apply_defences():
    try:
        responses = []
        selected = request.form.getlist('defences')

        if 'feature_smoothing' in selected:
            feature_smoothing.apply_feature_smoothing()
            responses.append('Feature Smoothing applied.')

        if 'adversarial_training' in selected:
            adversarial_training.adversarial_train()
            responses.append('Adversarial Training completed.')

        if 'concept_drift' in selected:
            visualize_concept_drift(save_path='static/concept_drift_analysis.png')
            responses.append('Concept Drift visualised.')

        if 'ensemble_learning' in selected:
            ensemble_learning.apply_ensemble_learning()
            responses.append('Ensemble Learning applied.')

        return jsonify({'message': responses})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_models', methods=['POST'])
def evaluate_models():
    try:
        results = run_evaluation()
        return jsonify({'evaluation': results})
    except Exception as e:
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
