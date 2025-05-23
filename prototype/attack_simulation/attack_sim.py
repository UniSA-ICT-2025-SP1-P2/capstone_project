import os
import pandas as pd
from train_models import run_all_models
from attack_sim import (
    compare_shap_values,
    run_attack_simulation,
    run_pgd_attack_simulation,
    generate_adversarial_samples_for_retraining,
    generate_pgd_samples_for_retraining
)
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "baseline_models", "trained_baseline_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_simulation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    try:
        # Step 1: Train models
        logging.info("Starting model training...")
        df = pd.read_csv(DATASET_PATH)
        selected_models = ['LogisticRegression', 'DecisionTree']
        
        training_result = run_all_models(df, selected_models=selected_models, progress_callback=logging.info)
        
        if training_result['status'] != 'complete':
            logging.error("Model training failed.")
            return
        
        logging.info("Model training completed successfully.")

        # Step 2: Load necessary objects
        le_catname = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        models = {clf_name: joblib.load(os.path.join(MODEL_DIR, f"{clf_name}_model.pkl"))
                  for clf_name in selected_models}
        
        # Load test and train datasets
        test_df = pd.read_csv(os.path.join(BASE_DIR, "baseline_models", "baseline_data", "Test_Dataset.csv"))
        train_df = pd.read_csv(os.path.join(BASE_DIR, "baseline_models", "baseline_data", "Train_Dataset.csv"))
        X_test = test_df[feature_names]
        y_test = test_df['category_name_encoded']
        X_train = train_df[feature_names]
        y_train = train_df['category_name_encoded']

        # Configuration for attack simulation
        epsilon_values = [0.05, 0.1, 0.15, 0.2]
        features_not_to_modify = [
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
        source_model = 'LogisticRegression'

        # Step 3: Run FGSM attack simulation
        logging.info("Running FGSM attack simulation...")
        attack_results = run_attack_simulation(
            models=models,
            X_test=X_test,
            y_test=y_test,
            epsilon_values=epsilon_values,
            features_not_to_modify=features_not_to_modify,
            source_model=source_model
        )
        logging.info("FGSM attack simulation completed. Results saved.")

        # Step 4: Run PGD attack simulation
        logging.info("Running PGD attack simulation...")
        pgd_attack_results = run_pgd_attack_simulation(
            models=models,
            X_test=X_test,
            y_test=y_test,
            epsilon_values=epsilon_values,
            features_not_to_modify=features_not_to_modify,
            source_model=source_model
        )
        logging.info("PGD attack simulation completed. Results saved.")

        # Step 5: Generate adversarial samples for retraining via FGSM
        logging.info("Generating adversarial samples for retraining...")
        for epsilon in epsilon_values:
            generate_adversarial_samples_for_retraining(
                model=models[source_model],
                X=X_train,
                y=y_train,
                epsilon=epsilon,
                n_samples=1000,
                target_class="Benign"
            )
        logging.info("Adversarial samples for retraining generated and saved.")

        logging.info("Generating PGD adversarial samples for retraining...")
        for epsilon in epsilon_values:
            generate_pgd_samples_for_retraining(
                model=models[source_model],
                X=X_train,
                y=y_train,
                epsilon=epsilon,
                n_samples=1000,
                target_class="Benign"
            )
        logging.info("PGD adversarial samples for retraining generated and saved.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
