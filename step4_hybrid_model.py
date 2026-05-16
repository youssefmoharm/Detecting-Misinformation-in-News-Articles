# pip install transformers torch datasets pandas scikit-learn

print("Step 4: Training Hybrid Model with Gated Fusion...\n")

# ✅ REPRODUCIBILITY
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# IMPORTS
# -----------------------------
import torch.nn as nn
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import json
import os

# -----------------------------
# LOAD DATA — 5000 samples for statistical validity
# -----------------------------
df_text = pd.read_csv("cleaned_data.csv").sample(5000, random_state=42)
df_feat = pd.read_csv("linguistic_features.csv").loc[df_text.index]

df = df_text.copy()
for col in df_feat.columns:
    if col != 'label':
        df[col] = df_feat[col]

df = df.reset_index(drop=True)

print(f"Dataset size: {len(df)} articles")
print(f"Class distribution:\n{df['label'].value_counts()}\n")

# -----------------------------
# SPLIT DATA (70/15/15)
# -----------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -----------------------------
# FEATURE SCALING
# -----------------------------
feature_cols = [
    "sensationalism_score", "exclamation_ratio", "question_ratio",
    "caps_ratio", "avg_word_length", "type_token_ratio",
    "avg_sentence_length", "text_length"
]

scaler = StandardScaler()
train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# -----------------------------
# TRAINING CONFIG
# -----------------------------
NUM_FEATURES  = len(feature_cols)   # 8
NUM_EPOCHS    = 3
BATCH_SIZE    = 8
MAX_LEN       = 256

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# -----------------------------
# DATASET
# -----------------------------
class HybridDataset(Dataset):
    def __init__(self, dataframe):
        self.texts    = dataframe['text'].tolist()
        self.labels   = dataframe['label'].values
        self.features = dataframe[feature_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features':       torch.tensor(self.features[idx]),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = HybridDataset(train_df)
val_dataset   = HybridDataset(val_df)
test_dataset  = HybridDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# -----------------------------
# GATED FUSION MODEL
# Innovation: instead of simple concatenation, a learned sigmoid gate
# controls how much linguistic signal is mixed into the semantic stream.
# Gate = sigmoid(W_g * [semantic; linguistic]) ∈ R^256
# Output = gate * semantic_proj + (1-gate) * linguistic_proj
# This lets the model learn per-sample how much to trust each source.
# -----------------------------
class GatedFusionModel(nn.Module):
    def __init__(self, hidden_dim=256, num_features=NUM_FEATURES):
        super(GatedFusionModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Project RoBERTa CLS → hidden_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Project linguistic features → hidden_dim
        self.linguistic_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Gating network: takes both projections, outputs per-dim gate ∈ (0,1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, features):
        # Semantic stream
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = roberta_out.last_hidden_state[:, 0, :]          # (B, 768)
        s = self.semantic_proj(cls)                            # (B, 256)

        # Linguistic stream
        l = self.linguistic_proj(features)                     # (B, 256)

        # Gated fusion
        gate_input = torch.cat([s, l], dim=1)                  # (B, 512)
        g = self.gate(gate_input)                              # (B, 256)
        fused = g * s + (1.0 - g) * l                         # (B, 256)

        return self.classifier(fused)

model = GatedFusionModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

# -----------------------------
# OPTIMIZER + LR SCHEDULER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW([
    {"params": model.roberta.parameters(),       "lr": 1e-5, "weight_decay": 0.01},
    {"params": model.semantic_proj.parameters(), "lr": 1e-3, "weight_decay": 0.01},
    {"params": model.linguistic_proj.parameters(),"lr": 1e-3, "weight_decay": 0.01},
    {"params": model.gate.parameters(),          "lr": 1e-3, "weight_decay": 0.01},
    {"params": model.classifier.parameters(),    "lr": 1e-3, "weight_decay": 0.01},
])

total_steps   = len(train_loader) * NUM_EPOCHS
warmup_steps  = int(0.1 * total_steps)   # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Training for {NUM_EPOCHS} epochs | {total_steps} total steps | {warmup_steps} warmup steps\n")

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_f1   = 0.0
history   = []

for epoch in range(NUM_EPOCHS):
    print(f"{'='*50}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*50}")

    # --- TRAIN ---
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features       = batch['features'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, features)
        loss    = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Train Loss: {avg_loss:.4f}")

    # --- VALIDATE ---
    model.eval()
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features       = batch['features'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, features)
            pred    = torch.argmax(outputs, dim=1)

            val_preds.extend(pred.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_f1  = f1_score(val_true, val_preds, average='weighted')
    val_acc = accuracy_score(val_true, val_preds)
    print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    history.append({"epoch": epoch+1, "loss": avg_loss, "val_f1": val_f1})

    if val_f1 > best_f1:
        best_f1 = val_f1
        os.makedirs("hybrid_model", exist_ok=True)
        torch.save(model.state_dict(), "hybrid_model/best_model.pt")
        print(f"  ✅ Best model saved (F1={best_f1:.4f})")

# -----------------------------
# TEST EVALUATION
# -----------------------------
print(f"\n{'='*50}")
print("Evaluating best model on test set...")
print(f"{'='*50}\n")

model.load_state_dict(torch.load("hybrid_model/best_model.pt", map_location=device))
model.eval()

preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features       = batch['features'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, features)
        pred    = torch.argmax(outputs, dim=1)

        preds.extend(pred.cpu().numpy())
        true.extend(labels.cpu().numpy())

print(classification_report(true, preds, target_names=["Real", "Fake"]))

# Save predictions
np.save("hybrid_preds.npy",  np.array(preds))
np.save("true_labels.npy",   np.array(true))   # overwrite with 5k-sample labels

# Save results
results = {
    "accuracy":  accuracy_score(true, preds),
    "precision": precision_score(true, preds, average='weighted'),
    "recall":    recall_score(true, preds, average='weighted'),
    "f1":        f1_score(true, preds, average='weighted'),
    "training_history": history
}

with open("hybrid_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Hybrid results saved  → hybrid_results.json")
print("✅ Predictions saved     → hybrid_preds.npy")
print("Step 4 completed ✅")
