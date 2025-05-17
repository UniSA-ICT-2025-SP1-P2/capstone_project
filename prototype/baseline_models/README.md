# README File for baseline classification models - utilising the CIC-MalMem-2022 dataset

This repository contains baseline machine learning classifiers developed to detect and categorise malware using memory forensic features derived from the **CIC-MalMem-2022** dataset. The core focus of this notebook is to train, evaluate, and interpret multiple classification models with **SHAP-based feature importance** for the **Conti** ransomware family. 

---

## Repository Structure:

---
baseline_models/
├── baseline_data/
│   ├── Adversarial_Features.csv          # List of selected mutable features for adversarial attacks
│   ├── Classifier_Results.xlsx           # Evaluation metrics and SHAP values
│   ├── Train_Dataset.csv                 # Training dataset
│   ├── Validation_Dataset.csv            # Validation dataset
│   ├── Test_Dataset.csv                  # Test dataset
│
├── shap_charts/
│   ├── RandomForest_SHAP.png             # SHAP bar plot (Random Forest)
│   ├── LogisticRegression_SHAP.png       # SHAP bar plot (Logistic Regression)
│   ├── DecisionTree_SHAP.png             # SHAP bar plot (Decision Tree)
│
├── trained_baseline_models/
│   ├── *.pkl                              # Saved model binaries for each classifier
│
├── data/
│   └── CIC-MalMem2022.csv                # Full dataset file (source: CIC-MalMem-2022)

---

## Notebook: Baseline_Models_Analysis.ipynb

### Purpose:
Train, evaluate, and interpret 5 supervised classifiers to identify malware families from memory snapshot data, with a special focus on **Conti ransomware**.

# ML Classification Models Trained:
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

### Code Workflow Summary:
1. **Data Loading and Cleaning**
   - Reads from ../data/CIC-MalMem2022.csv
   - Handles missing values via forward-fill
2. **Feature Engineering**
   - Extracts malware family, unique identifiers, and encodes target labels
3. **Train/Test/Validation Splits**
   - Uses `GroupShuffleSplit` to prevent data leakage between related samples
4. **Model Training & Hyperparameter Tuning**
   - Applies `GridSearchCV` where applicable
   - Stores trained models as .pkl files
5. **Evaluation Metrics**
   - Classification Report (Precision, Recall, F1-score)
   - Model predictions and probabilities saved per sample
6. **SHAP Analysis**
   - Computes SHAP values per model
   - Outputs bar plots of SHAP importance for the "Conti" class
7. **Output Files**
   - Evaluation metrics saved to Classifier_Results.xlsx
   - Visualizations saved to shap_charts/
   - Partitioned datasets exported for reuse

---

### Dataset: [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html)
- Source: Canadian Institute for Cybersecurity
- Contains Windows memory dumps for 14 malware families + benign samples
- Used after transformation into numerical features via Volatility

---

### Notes:
- This notebook is primarily for initial experimentation and analysis of ML classification and feature importance.
- A separate script (baseline_models_webUI.py) supports training and inference via a Flask web interface — a separate README will accompany that file.
- The Adversarial_Features.csv file contains features pre-selected for safe perturbation during adversarial testing.

---

### Requirements:
- Python ≥ 3.8
- scikit-learn, pandas, numpy, shap, matplotlib, joblib

To install all required libraries:
```bash
pip install -r requirements.txt

