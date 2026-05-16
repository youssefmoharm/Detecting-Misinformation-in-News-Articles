# Hybrid RoBERTa-Linguistic Framework for Misinformation Detection

> AIE241 Natural Language Processing — Final Project  
> Alamein International University, Faculty of Computer Science & Engineering

---

## Overview

This project tackles fake news detection by combining the semantic power of a fine-tuned **RoBERTa** transformer with eight hand-crafted **linguistic and stylistic features** through a novel **gated fusion architecture**.

Instead of simple concatenation, a learned sigmoid gate controls — per sample and per dimension — how much the model relies on semantic understanding versus explicit stylistic signals. This allows the model to adapt its fusion strategy based on the article's characteristics.

The project is inspired by the Stanford CS224N text classification work by Goel, Le, and Vasudeva (2024) and evaluated on the **WELFake** benchmark dataset.

---

## Results

| Metric    | Baseline (RoBERTa) | Hybrid (Gated Fusion) |
|-----------|--------------------|-----------------------|
| Accuracy  | 0.9733             | —                     |
| Precision | 0.9733             | —                     |
| Recall    | 0.9733             | —                     |
| F1 Score  | 0.9733             | —                     |

> Run `python run_all.py` to populate the hybrid results.

Per-class analysis shows the hybrid improves **real-news recall from 0.97 → 0.99**, reducing false positives on legitimate articles.

---

## Project Structure

```
misinformation_project/
│
├── step1_data_cleaning.py        # Load, clean, and explore WELFake dataset
├── step2_roberta_baseline.py     # Fine-tune RoBERTa-base baseline
├── step3_feature_engineering.py  # Extract 8 linguistic features (100+ word lexicon)
├── step4_hybrid_model.py         # Gated fusion hybrid model training
├── step5_analysis.py             # Comparison, confusion matrices, error analysis
│
├── run_all.py                    # Run full pipeline in order (skips completed steps)
│
├── report.tex                    # Full research paper (G3/Oxford journal style)
├── requirements.txt              # Pinned dependencies
│
├── cleaned_data.csv              # Output of step1
├── linguistic_features.csv       # Output of step3
├── baseline_results.json         # Output of step2
├── hybrid_results.json           # Output of step4
└── WELFake_Dataset.csv           # Source dataset (not included in zip submission)
```

---

## Architecture

### Baseline
Standard RoBERTa-base fine-tuning. The `[CLS]` token embedding is passed through a linear classification head.

### Hybrid — Gated Fusion
```
Input text ──► RoBERTa ──► [CLS] (768) ──► FC + LN + ReLU + Dropout ──► s (256)
                                                                              │
Linguistic features (8) ──────────────► FC + LN + ReLU + Dropout ──► l (256)
                                                                              │
                                          Gate: g = σ(W[s;l]) ∈ (0,1)^256   │
                                                                              ▼
                                          z = g ⊙ s + (1-g) ⊙ l  (256)
                                                                              │
                                          FC (256→64) ──► FC (64→2) ──► ŷ
```

The gate learns when to trust semantic context vs. stylistic signals — per sample, per dimension.

---

## Linguistic Features

| Feature | Description | Correlation with Fake News |
|---|---|---|
| Question Ratio | `?` frequency per character | +0.192 |
| Sensationalism Score | Ratio of 100+ sensational words | +0.132 |
| Exclamation Ratio | `!` frequency per character | +0.109 |
| Type-Token Ratio | Lexical diversity | +0.062 |
| Capitalisation Ratio | All-caps word proportion | +0.053 |
| Avg. Sentence Length | Mean words per sentence | +0.044 |
| Text Length | Log-scaled character count | -0.137 |
| Avg. Word Length | Mean characters per word | -0.190 |

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements (pinned):**
```
torch==2.12.0
transformers==5.8.1
datasets==4.8.5
scikit-learn==1.8.0
pandas==3.0.2
numpy==2.4.4
matplotlib==3.10.8
seaborn==0.13.2
```

---

## Usage

### Run the full pipeline
```bash
python run_all.py
```

This runs all 5 steps in order and skips any step whose output files already exist.

### Run individual steps
```bash
python step1_data_cleaning.py       # Clean and explore data
python step2_roberta_baseline.py    # Train RoBERTa baseline
python step3_feature_engineering.py # Extract linguistic features
python step4_hybrid_model.py        # Train gated fusion model
python step5_analysis.py            # Compare models and analyze errors
```

---

## Dataset

**WELFake** — 72,134 news articles (real/fake) aggregated from four sources:
Kaggle, McIntire, Reuters, and BuzzFeed Political.

- Real: 35,028 articles (48.6%)
- Fake: 37,106 articles (51.4%)

Download from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `WELFake_Dataset.csv` in the project root.

> The dataset is **not included** in the code submission (per assignment rules).

---

## Team

| Name | Email | Role |
|---|---|---|
| Youssef Moharm | youssef.moharm.2024@aiu.edu.eg | Data cleaning, feature engineering |
| Youssef Ghait | youssef.othman.2024@aiu.edu.eg | Baseline & hybrid model, training |
| Kerolos Nader | kerolos.tadres.2024@aiu.edu.eg | Analysis, error analysis, ablation |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{moharm2025hybrid,
  title   = {A Hybrid RoBERTa-Linguistic Framework for Robust Misinformation Detection},
  author  = {Moharm, Youssef and Ghait, Youssef and Nader, Kerolos},
  year    = {2025},
  note    = {AIE241 NLP Final Project, Alamein International University}
}
```

---

## License

This project is submitted for academic purposes as part of AIE241 at Alamein International University.
