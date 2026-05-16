# pip install pandas numpy matplotlib seaborn scikit-learn

print("Step 5: Model Comparison & Analysis...\n")

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# GUARD: Run missing steps first
# -----------------------------
missing = []

if not os.path.exists("baseline_results.json") or \
   not os.path.exists("baseline_preds.npy") or \
   not os.path.exists("true_labels.npy"):
    missing.append(("step2_roberta_baseline.py", "Baseline results"))

if not os.path.exists("hybrid_results.json") or \
   not os.path.exists("hybrid_preds.npy"):
    missing.append(("step4_hybrid_model.py", "Hybrid results"))

if missing:
    import subprocess
    for script, label in missing:
        print(f"⚠️  {label} not found. Running {script} first...\n")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False   # let output stream to console
        )
        if result.returncode != 0:
            print(f"❌ {script} failed with return code {result.returncode}.")
            print("Please run it manually before running step5.")
            sys.exit(1)
        print(f"✅ {script} completed.\n")

# -----------------------------
# LOAD RESULTS
# -----------------------------
with open("baseline_results.json") as f:
    baseline = json.load(f)

with open("hybrid_results.json") as f:
    hybrid = json.load(f)

# -----------------------------
# LOAD PREDICTIONS
# -----------------------------
try:
    baseline_preds = np.load("baseline_preds.npy")
    hybrid_preds   = np.load("hybrid_preds.npy")
    true_labels    = np.load("true_labels.npy")
    print("✅ Prediction files loaded successfully.\n")
except FileNotFoundError as e:
    print(f"❌ Could not load prediction file: {e}")
    print("Please ensure step2 and step4 have been run successfully.")
    sys.exit(1)

# -----------------------------
# COMPARISON TABLE
# -----------------------------
print("\nMODEL COMPARISON:\n")

metrics = ["accuracy", "f1", "precision", "recall"]

print(f"{'Metric':<12} | {'Baseline':<10} | {'Hybrid':<10} | {'Delta'}")
print("-"*50)

for m in metrics:
    base_val = baseline[m]
    hybrid_val = hybrid[m]
    delta = hybrid_val - base_val
    
    print(f"{m:<12} | {base_val:.4f}    | {hybrid_val:.4f}   | {delta:+.4f}")

# -----------------------------
# BAR CHART
# -----------------------------
labels = ["Accuracy", "F1", "Precision", "Recall"]

baseline_vals = [
    baseline["accuracy"], baseline["f1"],
    baseline["precision"], baseline["recall"]
]

hybrid_vals = [
    hybrid["accuracy"], hybrid["f1"],
    hybrid["precision"], hybrid["recall"]
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, baseline_vals, width, label="Baseline")
plt.bar(x + width/2, hybrid_vals, width, label="Hybrid")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Baseline vs Hybrid Performance")
plt.legend()
plt.show()

# -----------------------------
# CONFUSION MATRICES
# -----------------------------
cm_base = confusion_matrix(true_labels, baseline_preds)
cm_hybrid = confusion_matrix(true_labels, hybrid_preds)

fig, axes = plt.subplots(1,2, figsize=(12,5))

sns.heatmap(cm_base, annot=True, fmt='d', ax=axes[0])
axes[0].set_title("Baseline Confusion Matrix")

sns.heatmap(cm_hybrid, annot=True, fmt='d', ax=axes[1])
axes[1].set_title("Hybrid Confusion Matrix")

plt.show()

# -----------------------------
# ERROR ANALYSIS
# -----------------------------
df = pd.read_csv("cleaned_data.csv").sample(len(true_labels), random_state=42).reset_index(drop=True)

print("\nERROR ANALYSIS:\n")

print("\n1. Baseline WRONG → Hybrid CORRECT:\n")

count = 0
for i in range(len(true_labels)):
    if (baseline_preds[i] != true_labels[i]) and (hybrid_preds[i] == true_labels[i]):
        print("TEXT:", df['text'][i][:200])
        print("True:", true_labels[i],
              "| Baseline:", baseline_preds[i],
              "| Hybrid:", hybrid_preds[i])
        print("-"*50)
        count += 1
        if count == 5:
            break

print("\n2. BOTH MODELS WRONG:\n")

count = 0
for i in range(len(true_labels)):
    if (baseline_preds[i] != true_labels[i]) and (hybrid_preds[i] != true_labels[i]):
        print("TEXT:", df['text'][i][:200])
        print("True:", true_labels[i],
              "| Baseline:", baseline_preds[i],
              "| Hybrid:", hybrid_preds[i])
        print("-"*50)
        count += 1
        if count == 5:
            break

# -----------------------------
# FEATURE HEATMAP
# -----------------------------
feat_df = pd.read_csv("linguistic_features.csv")

plt.figure(figsize=(10,6))
sns.heatmap(feat_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ Step 5 completed ✅")