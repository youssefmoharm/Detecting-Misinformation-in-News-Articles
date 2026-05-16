"""
run_all.py
----------
Runs the full misinformation detection pipeline in order:
  Step 1 → Data Cleaning
  Step 2 → RoBERTa Baseline
  Step 3 → Linguistic Feature Engineering
  Step 4 → Hybrid Model
  Step 5 → Analysis & Comparison

Usage:
    python run_all.py
"""

import subprocess
import sys
import os

STEPS = [
    ("step1_data_cleaning.py",    "Data Cleaning"),
    ("step2_roberta_baseline.py", "RoBERTa Baseline"),
    ("step3_feature_engineering.py", "Feature Engineering"),
    ("step4_hybrid_model.py",     "Hybrid Model"),
    ("step5_analysis.py",         "Analysis & Comparison"),
]

# Output files each step is expected to produce
EXPECTED_OUTPUTS = {
    "step1_data_cleaning.py":       ["cleaned_data.csv"],
    "step2_roberta_baseline.py":    ["baseline_results.json", "baseline_preds.npy", "true_labels.npy"],
    "step3_feature_engineering.py": ["linguistic_features.csv"],
    "step4_hybrid_model.py":        ["hybrid_results.json", "hybrid_preds.npy"],
    "step5_analysis.py":            [],   # produces plots, no file to check
}

def all_outputs_exist(script):
    return all(os.path.exists(f) for f in EXPECTED_OUTPUTS[script])

print("=" * 60)
print("  Misinformation Detection — Full Pipeline Runner")
print("=" * 60)

for script, label in STEPS:
    print(f"\n{'─'*60}")
    print(f"▶  Running: {label}  ({script})")
    print(f"{'─'*60}")

    # Skip if all outputs already exist (except step5 which always runs)
    if script != "step5_analysis.py" and all_outputs_exist(script):
        print(f"⏭  Outputs already exist — skipping {script}")
        continue

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False   # stream output directly to console
    )

    if result.returncode != 0:
        print(f"\n❌  {script} failed (return code {result.returncode}).")
        print("    Fix the error above and re-run run_all.py.")
        sys.exit(result.returncode)

    # Verify expected outputs were created
    missing = [f for f in EXPECTED_OUTPUTS[script] if not os.path.exists(f)]
    if missing:
        print(f"\n⚠️  {script} finished but these expected files are missing:")
        for f in missing:
            print(f"    - {f}")
        print("    Check the script for errors.")
        sys.exit(1)

    print(f"\n✅  {label} completed successfully.")

print("\n" + "=" * 60)
print("  ✅  Full pipeline completed successfully!")
print("=" * 60)
