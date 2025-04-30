# run_experiment.py
import os
import subprocess
import time

# === Setup: Set working directory to this file's location ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /defence_prototype/
os.chdir(BASE_DIR)
print("📂 Base working directory:", os.getcwd())

# === Define script paths ===
SRC_DIR = os.path.join(BASE_DIR, 'src')                          # /defence_prototype/src
DEFENCES_DIR = os.path.join(SRC_DIR, 'defences')                 # /defence_prototype/src/defences

SCRIPTS = [
    os.path.join(SRC_DIR, 'train_models.py'),
    # os.path.join(SRC_DIR, 'generate_adversarials.py'),  # Uncomment if needed
    os.path.join(DEFENCES_DIR, 'concept_drift.py'),
    os.path.join(SRC_DIR, 'evaluate_defences.py'),
]

# === Confirm scripts before running ===
print("📄 Scripts to run:")
for s in SCRIPTS:
    print(" -", s)

# === Run each script in its own folder ===
for script in SCRIPTS:
    print(f"\n🚀 Running {script}...")
    start_time = time.time()
    
    # Run with working directory set to script's folder
    subprocess.run(["python", script], check=True, cwd=os.path.dirname(script))
    
    elapsed = time.time() - start_time
    print(f"✅ Completed {os.path.basename(script)} in {elapsed:.2f} seconds.")

print("\n🎯 All experiments completed successfully.")
