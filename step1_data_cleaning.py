# pip install pandas matplotlib seaborn scikit-learn

print("Step 1: Data Exploration & Cleaning started...\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD DATA
# -----------------------------
try:
    df = pd.read_csv("WELFake_Dataset.csv")
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading file:", e)
    exit()

# -----------------------------
# AUTO-DETECT COLUMNS
# -----------------------------
print("Detecting columns...")

text_col = None
label_col = None

for col in df.columns:
    if df[col].dtype == 'object':
        if text_col is None:
            text_col = col
    if df[col].dtype in ['int64', 'float64']:
        if label_col is None:
            label_col = col

# Clean detection manually if known columns exist
if 'text' in df.columns:
    text_col = 'text'
if 'label' in df.columns:
    label_col = 'label'

print(f"Detected TEXT column: {text_col}")
print(f"Detected LABEL column: {label_col}\n")

# -----------------------------
# BASIC INFO
# -----------------------------
print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df[label_col].value_counts())

print("\nNull Values:")
print(df.isnull().sum())

# -----------------------------
# CLEANING
# -----------------------------
print("\nCleaning data...")

df = df.dropna(subset=[text_col, label_col])

# Compute text length after dropping nulls
df['text_length'] = df[text_col].astype(str).apply(len)
print("\nAverage Text Length:", df['text_length'].mean())

# Normalize labels: fake=1, real=0
df[label_col] = df[label_col].apply(lambda x: 1 if str(x).lower() in ['1','fake'] else 0)

# -----------------------------
# CLASS DISTRIBUTION PLOT
# -----------------------------
print("\nPlotting class distribution...")

plt.figure(figsize=(6,4))
df[label_col].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Label (0=Real, 1=Fake)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# SAMPLE ROWS
# -----------------------------
print("\nSample FAKE articles:\n")
print(df[df[label_col]==1].head(3)[[text_col]])

print("\nSample REAL articles:\n")
print(df[df[label_col]==0].head(3)[[text_col]])

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
df.to_csv("cleaned_data.csv", index=False)

print("\n✅ Cleaned data saved as 'cleaned_data.csv'")
print("Step 1 completed ✅")