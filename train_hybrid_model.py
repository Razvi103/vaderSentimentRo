import requests
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import sys
import os

# --- 1. SETUP VADER ---
try:
    from vaderSentimentRoHelpers.vaderSentimentRo import SentimentIntensityAnalyzer
except ImportError:
    try:
        from vaderSentimentRo import SentimentIntensityAnalyzer
    except ImportError:
        print("❌ Critical: vaderSentimentRo.py not found.")
        sys.exit(1)

# Ensure we use the clean/lemmatized lexicon we built
LEXICON_FILE = "../vader_lexicon_ro_clean.txt" 
EMOJI_FILE = "../emoji_utf8_lexicon_ro.txt"


analyzer = SentimentIntensityAnalyzer(lexicon_file=LEXICON_FILE, emoji_lexicon=EMOJI_FILE)

# --- 2. DATA LOADING ---
URL_TRAIN = "https://raw.githubusercontent.com/ancatache/LaRoSeDa/main/data_splitted/laroseda_train.json"
URL_TEST = "https://raw.githubusercontent.com/ancatache/LaRoSeDa/main/data_splitted/laroseda_test.json"

def load_data(url, name):
    print(f"⬇️  Downloading {name} set...")
    resp = requests.get(url)
    data = resp.json()
    reviews = data['reviews'] if 'reviews' in data else data
    
    texts = []
    labels = []
    
    for item in reviews:
        stars = int(item.get("starRating"))
        if stars is None or stars == 3: continue
        
        # 1,2 -> 0 (Neg) | 4,5 -> 1 (Pos)
        label = 1 if stars > 3 else 0
        text = item.get("content") or ""
        
        texts.append(text)
        labels.append(label)
        
    print(f"✅ Loaded {len(texts)} samples for {name}.")
    return texts, labels

# --- 3. FEATURE ENGINEERING ---
def get_vader_features(texts):
    """
    Returns a matrix of shape (N_samples, 4) containing:
    [neg_score, neu_score, pos_score, compound_score]
    """
    print("🤖 Extracting VADER features (this takes time)...")
    features = []
    for text in tqdm(texts):
        # We assume text is already reasonably clean or we pass it raw
        # (For maximum speed, we skip SpaCy lemmatization here, 
        # relying on VADER's exact match + the ML model learning raw word forms)
        scores = analyzer.polarity_scores(text)
        features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
    return np.array(features)

def train_hybrid():
    # A. Load Data
    train_texts, y_train = load_data(URL_TRAIN, "TRAIN")
    test_texts, y_test = load_data(URL_TEST, "TEST")

    # B. VADER Features (The "Expert Opinion")
    print("\n--- Step 1: VADER Scoring ---")
    X_train_vader = get_vader_features(train_texts)
    X_test_vader = get_vader_features(test_texts)

    # C. TF-IDF Features (The "Raw Words")
    print("\n--- Step 2: TF-IDF Vectorization ---")
    # Limit to top 3000 words to keep it fast and prevent overfitting
    # ngram_range=(1,2) catches phrases like "nu recomand" automatically
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=3)
    
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    # D. Combine Features
    print("\n--- Step 3: Hybrid Fusion ---")
    # Stack VADER scores (4 cols) next to TF-IDF scores (3000 cols)
    X_train = sp.hstack((X_train_vader, X_train_tfidf))
    X_test = sp.hstack((X_test_vader, X_test_tfidf))

    # E. Train Model
    print("\n--- Step 4: Training Logistic Regression ---")
    # C=1.0 is standard regularization
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)

    # F. Evaluate
    print("\n--- Step 5: Evaluation ---")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 HYBRID MODEL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Optional: See what the model learned
    # We look at weights. High + weight = Positive, Low - weight = Negative
    print("\n🧐 What did the model learn? (Top features)")
    feature_names = ['V_NEG', 'V_NEU', 'V_POS', 'V_COMP'] + vectorizer.get_feature_names_out().tolist()
    coefs = clf.coef_[0]
    
    # Sort by weight
    top_indices = np.argsort(coefs)
    
    print("\nTop 10 Most NEGATIVE predictors:")
    for idx in top_indices[:10]:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        
    print("\nTop 10 Most POSITIVE predictors:")
    for idx in top_indices[-10:][::-1]:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

if __name__ == "__main__":
    train_hybrid()