import subprocess
import time
import os

# === Paths to your scripts ===
SRC_DIR = 'src'
DEFENCES_DIR = os.path.join(SRC_DIR, 'defences')

SCRIPTS = [
    os.path.join(SRC_DIR, 'train_models.py'),
    os.path.join(SRC_DIR, 'generate_adversarials.py'),
    os.path.join(DEFENCES_DIR, 'concept_drift.py'),  # optional: run separately after evaluation
    os.path.join(SRC_DIR, 'evaluate_defences.py'),
]

# === Run each step ===
for script in SCRIPTS:
    print(f"\n Running {script}...")
    start_time = time.time()
    
    subprocess.run(["python", script], check=True)
    
    elapsed = time.time() - start_time
    print(f"Completed {os.path.basename(script)} in {elapsed:.2f} seconds.")

print("\nAll experiments completed successfully.")
