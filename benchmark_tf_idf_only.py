import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import sys

# --- CONFIG ---
# Using the standard URLs
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
        label = 1 if stars > 3 else 0
        text = item.get("content") or ""
        texts.append(text)
        labels.append(label)
        
    print(f"✅ Loaded {len(texts)} samples for {name}.")
    return texts, labels

def run_tfidf_benchmark():
    # 1. Load Data
    train_texts, y_train = load_data(URL_TRAIN, "TRAIN")
    test_texts, y_test = load_data(URL_TEST, "TEST")

    # 2. Vectorize (The "Pure ML" Approach)
    # We use ngram_range=(1,2) so it can learn "nu recomand" as a single feature
    print("\n--- TF-IDF Vectorization ---")
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=2)
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")

    # 3. Train
    print("\n--- Training Logistic Regression ---")
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    print("\n--- Evaluation ---")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 PURE TF-IDF ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # 5. Interpretability
    print("\n🧐 What did Pure ML learn?")
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_indices = np.argsort(coefs)
    
    print("\nTop 5 NEGATIVE features:")
    for idx in top_indices[:5]:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

    print("\nTop 5 POSITIVE features:")
    for idx in top_indices[-5:][::-1]:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

if __name__ == "__main__":
    run_tfidf_benchmark()