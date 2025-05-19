# README File for Webpage UI - Training Baseline Malware Classifiers:

This module (`baseline_models_webUI.py`) supports model training and SHAP-based explanation through a Python script that can be integrated into a **Flask-based web application**. It uses the CIC-MalMem-2022 dataset and includes adversarially mutable features to support reproducibility and robustness.

---

## Repository Structure:

```
baseline_models/
├── baseline_data/
│   ├── Adversarial_Features.csv          # Mutable features selected for FGSM/PGD
│   ├── Classifier_Results.xlsx           # Classification metrics and SHAP values
│   ├── Train_Dataset.csv / Validation_Dataset.csv / Test_Dataset.csv
│
├── shap_charts/
│   ├── *_SHAP.png                        # Visualized SHAP importance per model
│
├── trained_baseline_models/
│   ├── *_model.pkl                       # Saved classifiers (e.g., RandomForest, KNN)
│   ├── label_encoder.pkl                 # Saved label encoder for inverse-transform
│   ├── feature_names.pkl                 # Feature list for input reconstruction
│
├── data/
│   └── CIC-MalMem2022.csv                # Full memory dataset (preprocessed)
├── Baseline_Models_Analysis.ipynb        # Jupyter notebook for standalone model development
├── baseline_models_webUI.py              # This script: for model execution via interface
├── README.md
```

---

## Purpose:

This script enables modular execution of **baseline malware classifiers**:
- Accepts **user-specified model selection**
- Shows **progress updates** via callback (e.g., to a Flask web frontend)
- Stores **evaluation metrics and SHAP values**
- Exports train/validation/test splits for traceability
- Supports safe adversarial feature identification

---

## ML Classification Models:

- `RandomForest`
- `K-Nearest Neighbors (KNN)`
- `Logistic Regression`
- `Support Vector Machine (SVM)`
- `Decision Tree`

---

## Features:

- GridSearchCV for hyperparameter tuning
- SHAP analysis focused on **Conti** class
- Dynamic progress updates using `progress_callback`
- Label encoding & Group-aware data splitting
- Adversarial feature export for explainability experiments

---

## Function: `run_all_models(df, selected_models=None, progress_callback=None)`

### Parameters:
- `df`: Pandas DataFrame loaded from `CIC-MalMem2022.csv`
- `selected_models`: list of model names to train (optional)
- `progress_callback`: function to stream log/progress to frontend (optional)

### Returns:
A dictionary with:
- `status`: `"complete"`
- `message`: final update
- `adversarial_features`: features safe to perturb
- `results`: dictionary of model paths, SHAP image paths, and classification reports

---

## Requirements
Install required packages:
```bash
pip install scikit-learn pandas numpy shap matplotlib joblib
```

---

## Integration Example:
This script is designed to be called from a Flask web interface or CLI wrapper. For web usage, you can call:
```python
result = run_all_models(df, selected_models=["RandomForest", "SVM"], progress_callback=log_fn)
```

---

## Notes:
- Data splitting is **group-aware** to prevent sample leakage.
- SHAP analysis may fail for certain models under specific configurations (handled via exception).
- The label encoder (`label_encoder.pkl`) and feature names (`feature_names.pkl`) are saved for consistent inference.

---

