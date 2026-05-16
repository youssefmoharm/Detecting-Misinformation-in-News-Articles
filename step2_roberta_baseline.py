# pip install transformers torch datasets scikit-learn pandas matplotlib seaborn

print("Step 2: Training RoBERTa baseline model...\n")

# ✅ REPRODUCIBILITY FIX
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import json

# -----------------------------
# LOAD CLEANED DATA — 5000 samples for statistical validity
# -----------------------------
try:
    df = pd.read_csv("cleaned_data.csv").sample(5000, random_state=42)
    print(f"Loaded cleaned_data.csv — {len(df)} articles\n")
except Exception as e:
    print("Error loading cleaned data:", e)
    exit()

# -----------------------------
# SPLIT DATA (70/15/15)
# -----------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print("Data split completed:")
print(len(train_df), len(val_df), len(test_df))

# -----------------------------
# TOKENIZATION
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=256
    )

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)
test_ds  = Dataset.from_pandas(test_df)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds   = val_ds.map(tokenize_function, batched=True)
test_ds  = test_ds.map(tokenize_function, batched=True)

train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# -----------------------------
# MODEL
# -----------------------------
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)

# -----------------------------
# TRAINING CONFIG
# -----------------------------
training_args = TrainingArguments(
    output_dir="./baseline_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1
)

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted")
    }

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining model...")
trainer.train()

# -----------------------------
# EVALUATE
# -----------------------------
print("\nEvaluating model on test set...")

predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, average='weighted')
rec  = recall_score(labels, preds, average='weighted')
f1   = f1_score(labels, preds, average='weighted')

print("\nClassification Report:\n")
print(classification_report(labels, preds))

# ✅ SAVE PREDICTIONS (CRITICAL FIX)
np.save("baseline_preds.npy", preds)
np.save("true_labels.npy", labels)

# -----------------------------
# SAVE RESULTS
# -----------------------------
results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
}

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# SAVE MODEL
# -----------------------------
trainer.save_model("./baseline_model")

print("\n✅ Baseline model saved in ./baseline_model/")
print("✅ Results saved as baseline_results.json")
print("✅ Predictions saved (.npy files)")
print("Step 2 completed ✅")
