# pip install pandas numpy seaborn matplotlib

print("Step 3: Extracting linguistic features...\n")

# ✅ REPRODUCIBILITY FIX
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    df = pd.read_csv("cleaned_data.csv")
    print("Loaded cleaned_data.csv\n")
except Exception as e:
    print("Error loading file:", e)
    exit()

# -----------------------------
# SENSATIONAL WORD LIST (expanded to 100+ words)
# -----------------------------
sensational_words = set([
    # Alarm / urgency
    "shocking","unbelievable","explosive","breaking","urgent","bombshell",
    "alert","warning","danger","crisis","emergency","critical","immediate",
    "alarming","terrifying","horrifying","devastating","catastrophic",
    # Deception / conspiracy
    "hoax","conspiracy","secret","exposed","coverup","cover-up","whistleblower",
    "leaked","classified","hidden","suppressed","censored","banned","forbidden",
    "silenced","blackout","deepstate","deep-state","illuminati","cabal",
    # Outrage / emotion
    "outrage","scandal","lies","truth","corrupt","corruption","betrayal",
    "traitor","treason","criminal","illegal","fraud","rigged","stolen",
    "cheating","manipulation","propaganda","brainwash","indoctrination",
    # Sensational adjectives
    "insane","crazy","unreal","impossible","incredible","miraculous",
    "stunning","jaw-dropping","mind-blowing","eye-opening","groundbreaking",
    "unprecedented","historic","massive","enormous","gigantic","extreme",
    # Clickbait verbs
    "revealed","exposes","destroys","obliterates","annihilates","crushes",
    "slams","blasts","rips","tears","shreds","dismantles","debunks",
    "proves","confirms","admits","confesses","caught","busted","arrested",
    # Fear / threat
    "threat","attack","invasion","takeover","collapse","meltdown","disaster",
    "apocalypse","endgame","final","last","dying","dead","killed","murdered",
    # Political sensationalism
    "fake","hoax","witch-hunt","witch","hunt","deep","state","globalist",
    "elites","establishment","mainstream","narrative","agenda","plot"
])

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def tokenize_words(text):
    return re.findall(r'\b\w+\b', str(text))

def tokenize_sentences(text):
    return re.split(r'[.!?]+', str(text))

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
features = []

for text in df['text'].astype(str):
    
    words = tokenize_words(text)
    sentences = tokenize_sentences(text)
    
    total_words = len(words) if len(words) > 0 else 1
    total_chars = len(text) if len(text) > 0 else 1
    total_sentences = len(sentences) if len(sentences) > 0 else 1
    
    # 1. sensationalism_score
    sens_count = sum(1 for w in words if w.lower() in sensational_words)
    sensationalism_score = sens_count / total_words
    
    # 2. exclamation_ratio
    exclamation_ratio = text.count('!') / total_chars
    
    # 3. question_ratio
    question_ratio = text.count('?') / total_chars
    
    # 4. caps_ratio
    caps_words = sum(1 for w in words if w.isupper())
    caps_ratio = caps_words / total_words
    
    # 5. avg_word_length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # 6. type_token_ratio
    type_token_ratio = len(set(words)) / total_words
    
    # 7. avg_sentence_length
    avg_sentence_length = total_words / total_sentences
    
    # 8. text_length (log scaled)
    text_length = np.log(total_chars + 1)
    
    features.append([
        sensationalism_score,
        exclamation_ratio,
        question_ratio,
        caps_ratio,
        avg_word_length,
        type_token_ratio,
        avg_sentence_length,
        text_length
    ])

# -----------------------------
# CREATE FEATURE DATAFRAME
# -----------------------------
feature_names = [
    "sensationalism_score",
    "exclamation_ratio",
    "question_ratio",
    "caps_ratio",
    "avg_word_length",
    "type_token_ratio",
    "avg_sentence_length",
    "text_length"
]

features_df = pd.DataFrame(features, columns=feature_names)

# Add label
features_df['label'] = df['label'].values

# -----------------------------
# SAVE FEATURES
# -----------------------------
features_df.to_csv("linguistic_features.csv", index=False)

print("✅ Linguistic features saved as linguistic_features.csv\n")

# -----------------------------
# CORRELATION ANALYSIS
# -----------------------------
print("Feature Correlations with Label:\n")

correlations = features_df.corr()['label'].drop('label')
print(correlations.sort_values(ascending=False))

# -----------------------------
# HEATMAP
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(features_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ Step 3 completed ✅")