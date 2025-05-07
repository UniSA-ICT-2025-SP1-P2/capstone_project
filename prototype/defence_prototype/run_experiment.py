# NOT USED AT THE MOMENT

# import os
# import subprocess
# import time

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.chdir(BASE_DIR)
# print("Base working directory:", os.getcwd())

# SRC_DIR = os.path.join(BASE_DIR, 'src')
# DEFENCES_DIR = os.path.join(SRC_DIR, 'defences')

# SCRIPTS = [
#     os.path.join(SRC_DIR, 'train_models.py'),
#     # os.path.join(SRC_DIR, 'generate_adversarials.py'),
#     os.path.join(DEFENCES_DIR, 'concept_drift.py'),
#     os.path.join(SRC_DIR, 'evaluate_defences.py'),
# ]

# print("Scripts to run:")
# for s in SCRIPTS:
#     print(" -", s)

# for script in SCRIPTS:
#     print(f"\nRunning {script}...")
#     start_time = time.time()
#     subprocess.run(["python", script], check=True, cwd=os.path.dirname(script))
#     elapsed = time.time() - start_time
#     print(f"Completed {os.path.basename(script)} in {elapsed:.2f} seconds.")

# print("\nAll experiments completed successfully.")
