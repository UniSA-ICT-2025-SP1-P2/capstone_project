from flask import Flask, request, render_template_string, send_file, redirect, url_for
import os
import shutil
from werkzeug.utils import secure_filename

# === Adjusted Imports Based on Folder Structure ===
from src.train_models import train_models
from src.evaluate_defences import run_evaluation
from src.predict import predict
from src.defences.adversarial_training import run_adversarial_training
from src.defences.feature_smoothing import apply_feature_smoothing_path
from src.defences.concept_drift import run_concept_drift
from src.defences.ensemble_learning import run_ensemble_evaluation

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
RESULT_FOLDER = 'results'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_messages = []

    if request.method == 'POST':
        # === Handle File Uploads ===
        uploaded_dataset = request.files.get('dataset')
        fgsm = request.files.get('fgsm')
        pgd = request.files.get('pgd')

        dataset_path = None
        fgsm_path = None
        pgd_path = None

        if uploaded_dataset:
            filename = secure_filename(uploaded_dataset.filename)
            dataset_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_dataset.save(dataset_path)

        if fgsm:
            filename = secure_filename(fgsm.filename)
            fgsm_path = os.path.join(UPLOAD_FOLDER, filename)
            fgsm.save(fgsm_path)

        if pgd:
            filename = secure_filename(pgd.filename)
            pgd_path = os.path.join(UPLOAD_FOLDER, filename)
            pgd.save(pgd_path)

        # === Get Parameters ===
        epochs = int(request.form.get('epochs', 10))
        lr = float(request.form.get('lr', 0.001))
        noise_std = float(request.form.get('noise_std', 0.02))
        chunk_size = int(request.form.get('chunk_size', 500))
        threshold = float(request.form.get('threshold', 0.7))

        # === Run Train Models ===
        if dataset_path:
            train_models(dataset_path, MODEL_FOLDER, epochs=epochs, lr=lr)
            result_messages.append("✅ Models trained.")

        # === Run Adversarial Training ===
        if dataset_path:
            run_adversarial_training(UPLOAD_FOLDER, MODEL_FOLDER, epochs=epochs, lr=lr)
            result_messages.append("✅ Adversarial training complete.")

        # === Feature Smoothing ===
        if dataset_path:
            smoothed_path = os.path.join(RESULT_FOLDER, 'smoothed_dataset.csv')
            apply_feature_smoothing_path(dataset_path, noise_std, smoothed_path)
            result_messages.append(f"✅ Smoothed data saved: {smoothed_path}")

        # === Concept Drift ===
        if dataset_path:
            run_concept_drift(dataset_path, model_type='ensemble', chunk_size=chunk_size, threshold=threshold, results_dir=RESULT_FOLDER)
            result_messages.append("✅ Concept drift detection complete.")

        # === Evaluate Defences ===
        data_files = {'clean': dataset_path}
        if fgsm_path: data_files['fgsm'] = fgsm_path
        if pgd_path: data_files['pgd'] = pgd_path
        run_evaluation(data_files, results_dir=RESULT_FOLDER, model_dir=MODEL_FOLDER)
        result_messages.append("✅ Defence evaluation complete.")

        # === Ensemble Evaluation ===
        result = run_ensemble_evaluation(dataset_path, MODEL_FOLDER)
        result_messages.append(f"✅ Ensemble Evaluation: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")

    return render_template_string(TEMPLATE, result_messages=result_messages)

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

TEMPLATE = '''
<!doctype html>
<title>Defence Evaluation App</title>
<h2>Run ML Defences & Evaluation</h2>
<form method=post enctype=multipart/form-data>
  <label>Clean Dataset CSV:</label><br>
  <input type=file name=dataset><br><br>
  <label>FGSM Dataset CSV:</label><br>
  <input type=file name=fgsm><br><br>
  <label>PGD Dataset CSV:</label><br>
  <input type=file name=pgd><br><br>
  <label>Epochs: <input type=number name=epochs value=10></label><br>
  <label>Learning Rate: <input type=text name=lr value=0.001></label><br>
  <label>Noise STD (for smoothing): <input type=text name=noise_std value=0.02></label><br>
  <label>Chunk Size (drift): <input type=number name=chunk_size value=500></label><br>
  <label>Drift Threshold: <input type=text name=threshold value=0.7></label><br><br>
  <input type=submit value="Run Everything">
</form>

{% if result_messages %}
  <h3>Results:</h3>
  <ul>
  {% for msg in result_messages %}
    <li>{{ msg }}</li>
  {% endfor %}
  </ul>
  <a href="/download/concept_drift_results.csv">Download Drift Results</a><br>
  <a href="/download/defence_results.csv">Download Defence Results</a><br>
  <a href="/download/smoothed_dataset.csv">Download Smoothed Dataset</a>
{% endif %}
'''

if __name__ == '__main__':
    app.run(debug=True)
