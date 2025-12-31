import json
import requests
import spacy
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
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
# Using the working URL from your benchmark
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
    Matches the logic in benchmark_laroseda_fixed.py
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
    
    # C. Threshold (Standard 0.05)
    return 1 if compound >= 0.05 else 0

def analyze_mistakes():
    print(f"⬇️  Downloading LaRoSeDa Test Set...")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status() 
        data = response.json()
        
        if "reviews" in data:
            reviews = data["reviews"]
        else:
            reviews = data
            
        print(f"✅ Loaded {len(reviews)} reviews.")

    except Exception as e:
        print(f"❌ Download Error: {e}")
        return

    false_positives = [] # VADER said Positive, Truth was Negative
    false_negatives = [] # VADER said Negative, Truth was Positive

    print("🚀 Scanning for mistakes...")
    
    for item in tqdm(reviews):
        stars = int(item.get("starRating"))
        if stars is None or stars == 3: continue
            
        true_label = 1 if stars > 3 else 0
        text = item.get("content")
        
        # Get Prediction
        pred_label = get_vader_prediction(text)
        
        # Categorize Error
        if pred_label == 1 and true_label == 0:
            false_positives.append(text)
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(text)

    print("\n" + "="*50)
    print("📊 ERROR ANALYSIS REPORT")
    print("="*50)
    print(f"False Positives (The 'Optimist' Bug): {len(false_positives)}")
    print(f"False Negatives (The 'Pessimist' Bug): {len(false_negatives)}")
    print("-" * 50)
    
    # Analyze the False Positives (The bigger problem)
    # We want to know: Which words appear often in negative reviews that we mistakenly called positive?
    
    # Custom Stop Words to clean up the list
    ro_stop_words = [
        'si', 'sa', 'se', 'de', 'la', 'in', 'nu', 'o', 'un', 'pe', 'care', 'mai', 'fi', 'am', 'eu', 
        'foarte', 'pentru', 'ca', 'este', 'sunt', 'din', 'cu', 'ce', 'au', 'doar', 'dar', 'al', 'ai',
        'fost', 'era', 'cand', 'dupa', 'prin', 'fara', 'ea', 'el', 'ei', 'ele', 'imi', 'iti', 'isi'
    ]
    
    if false_positives:
        print("\n🔍 TOP WORDS IN FALSE POSITIVES")
        print("(These words appeared in Negative reviews, but VADER gave a Positive score)")
        print("(Look for words like 'scump', 'problema', 'mic' that might need negative scores)")
        
        vec = CountVectorizer(stop_words=ro_stop_words, max_features=30)
        X = vec.fit_transform(false_positives)
        sum_words = X.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

        for i, (word, freq) in enumerate(words_freq[:25]):
            print(f"{i+1}. {word}: {freq}")

if __name__ == "__main__":
    analyze_mistakes()