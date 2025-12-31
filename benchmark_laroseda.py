import json
import requests
import spacy
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# --- 1. IMPORT VADER ADAPTER ---
try:
    from vaderSentimentRoHelpers.vaderSentimentRo import SentimentIntensityAnalyzer
    print("✅ Imported VADER from vaderSentimentRoHelpers")
except ImportError:
    try:
        from vaderSentimentRo import SentimentIntensityAnalyzer
        print("✅ Imported VADER from local folder")
    except ImportError:
        print("❌ Critical Error: Could not find 'vaderSentimentRo.py'.")
        sys.exit(1)

# --- 2. CONFIGURATION ---
# THE CORRECT URL extracted from the Hugging Face script you provided
DATASET_URL = "https://raw.githubusercontent.com/ancatache/LaRoSeDa/main/data_splitted/laroseda_test.json"

LEXICON_FILE = "../vader_lexicon_ro_clean.txt"
EMOJI_FILE = "../emoji_utf8_lexicon_ro.txt"

# --- 3. LOADING RESOURCES ---
print("⏳ Loading SpaCy model (ro_core_news_sm)...")
try:
    nlp = spacy.load("ro_core_news_lg", disable=['parser', 'ner'])
except OSError:
    print("❌ Error: SpaCy model not found. Run: python -m spacy download ro_core_news_sm")
    sys.exit(1)

print("⏳ Loading VADER-Ro Adapter...")


analyzer = SentimentIntensityAnalyzer(
    lexicon_file=LEXICON_FILE, 
    emoji_lexicon=EMOJI_FILE
)

def get_vader_prediction(text):
    """
    Pipeline: Text -> SpaCy Lemma -> VADER -> Label (0/1)
    """
    if not text or not isinstance(text, str):
        return 0
        
    # A. Lemmatize
    doc = nlp(text)
    
    lemma_words = []
    for token in doc:
        t_text = token.text.lower()
        t_lemma = token.lemma_.lower()
        
        # Keep negations and 'fi' (to be) original
        if t_text.startswith('n'): 
            lemma_words.append(t_text)
        elif t_lemma == "fi": 
            lemma_words.append(t_lemma)
        else:
            lemma_words.append(t_lemma)
            
    lemmatized_text = " ".join(lemma_words)
    
    # B. Score
    scores = analyzer.polarity_scores(lemmatized_text)
    compound = scores['compound']
    
    # C. Threshold 
    # Standard is 0.05. If you find VADER is too harsh, try 0.0
    return 1 if compound >= 0.35 else 0

def run_benchmark():
    print(f"⬇️  Downloading LaRoSeDa Test Set from GitHub...")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status() 
        data = response.json()
        
        # The script you shared confirms the structure is: {"reviews": [...]}
        if "reviews" in data:
            reviews = data["reviews"]
        else:
            reviews = data
            
        print(f"✅ Loaded {len(reviews)} reviews.")

    except Exception as e:
        print(f"❌ Download Error: {e}")
        return

    y_true = []
    y_pred = []

    print("🚀 Running Analysis...")
    
    # Using tqdm for progress bar
    for item in tqdm(reviews):
        # 1. Get Label (starRating)
        stars = int(item.get("starRating"))
        
        if stars is None or stars == 3:
            continue
            
        # 1,2 -> Negative (0) | 4,5 -> Positive (1)
        true_label = 1 if stars > 3 else 0
        
        # 2. Get Text
        text = item.get("content")
        
        # 3. Predict
        pred_label = get_vader_prediction(text)
        
        y_true.append(true_label)
        y_pred.append(pred_label)

    # --- REPORTING ---
    print("\n" + "="*50)
    print("🏁 FINAL RESULTS: VADER-RO on LaRoSeDa")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    
    print("\n📉 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Neg: {cm[0][0]}\t| False Pos: {cm[0][1]}")
    print(f"False Neg: {cm[1][0]}\t| True Pos: {cm[1][1]}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()