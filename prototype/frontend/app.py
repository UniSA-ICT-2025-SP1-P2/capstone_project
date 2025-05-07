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

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)


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
        raw_results = run_all_models(df)
        print(f"[INFO] Model training completed.")

        def safe_metric(val, decimal_places=4):
            return round(val, decimal_places) if isinstance(val, (int, float)) else 0

        def clean_model_data(model_data):
            return {
                "accuracy": safe_metric(model_data.get("accuracy")),
                "f1_score": safe_metric(model_data.get("f1_score")),
                "precision": safe_metric(model_data.get("precision")),
                "recall": safe_metric(model_data.get("recall")),
                "training_time": str(model_data.get("training_time", "N/A")),
                "model_details": model_data.get("model_details", {}),
                "shap_values": model_data.get("shap_values", {})
            }

        # Filter only dict results and clean them
        cleaned_results = {}
        for model_name, model_data in raw_results.items():
            if isinstance(model_data, dict):
                cleaned_results[model_name] = clean_model_data(model_data)
            else:
                print(f"[WARN] Skipping {model_name} - not a dict: {type(model_data)}")

        return jsonify(cleaned_results)
    except Exception as e:
        print("[ERROR] Exception during processing:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/run_concept_drift')
def run_concept_drift():
    try:
        output_path = os.path.join(app.config['STATIC_FOLDER'], 'concept_drift_analysis.png')
        visualize_concept_drift()

        if not os.path.exists(output_path):
            print("[ERROR] Concept drift image not found.")
            return jsonify({'error': 'Image generation failed'}), 500

        return jsonify({'image_path': '/static/concept_drift_analysis.png'})
    except Exception as e:
        print("[ERROR] Concept Drift:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/run_defence_results')
def run_defence_results():
    try:
        output_path = os.path.join(app.config['STATIC_FOLDER'], 'results_table_simple.png')
        visualise_defence_results(output_file=output_path)

        if not os.path.exists(output_path):
            return jsonify({'error': 'Image generation failed'}), 500

        return jsonify({'image_path': '/static/results_table_simple.png'})
    except Exception as e:
        print("[ERROR] Defence Results:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/run_model_results')
def run_model_results():
    try:
        df = load_csv_data("model_results.csv")
        if df is None:
            return jsonify({'error': 'model_results.csv not found'}), 404

        img = create_model_results_table(df)
        output_path = os.path.join(app.config['STATIC_FOLDER'], 'model_results_table.png')
        img.save(output_path)

        if not os.path.exists(output_path):
            return jsonify({'error': 'Image save failed'}), 500

        return jsonify({'image_path': '/static/model_results_table.png'})
    except Exception as e:
        print("[ERROR] Model Results Table:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
