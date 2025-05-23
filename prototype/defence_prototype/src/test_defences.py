import os
from defences import feature_smoothing, concept_drift
from defences import adversarial_training  

# === Define project paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DEFENCE_PROTOTYPE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
PROTOTYPE_DIR = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(DEFENCE_PROTOTYPE_DIR, '..'))  # /prototype/

DATA_DIR = os.path.join(PROJECT_ROOT, 'data') 
MODEL_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'models')
RESULTS_DIR = os.path.join(DEFENCE_PROTOTYPE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# === Run a test ===

# Example 1: Feature Smoothing
def test_feature_smoothing():
    print("\nRunning Feature Smoothing...")
    feature_smoothing_path = os.path.join(DATA_DIR, 'test_label.csv')
    feature_smoothing.df = feature_smoothing.apply_feature_smoothing_path(
        feature_smoothing_path,
        noise_std=0.02,
        output_path=os.path.join(DATA_DIR, 'test_label_smoothed.csv')
    )
    print("✅ Feature smoothing complete.\n")

# Example 2: Concept Drift
def test_concept_drift():
    print("\nRunning Concept Drift Detection...")
    concept_drift.run_concept_drift(
        data_path=os.path.join(DATA_DIR, 'test_label.csv'),
        model_type='ensemble',
        chunk_size=500,
        threshold=0.7,
        results_dir=RESULTS_DIR
    )
    print("✅ Concept drift detection complete.\n")

# Example 3: Adversarial Training
def test_adversarial_training():
    print("\nRunning Adversarial Training...")
    adversarial_training.run_adversarial_training(
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        epochs=10,
        lr=0.001
    )
    print("✅ Adversarial training complete.\n")

# === Choose which to run ===
if __name__ == "__main__":
    # Uncomment the ones you want to test
    test_feature_smoothing()
    test_concept_drift()
    # test_adversarial_training()
