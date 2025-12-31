import spacy
import sys
import os

# --- CONFIG ---
INPUT_FILE = "vader_lexicon_ro_clean.txt"
OUTPUT_FILE = "vader_lexicon_ro_lemmatized.txt"

# Load SpaCy (Must be the exact same model you use in the benchmark)
print("⏳ Loading SpaCy model...")
try:
    nlp = spacy.load("ro_core_news_lg", disable=['parser', 'ner'])
except OSError:
    print("❌ Error: ro_core_news_sm not found.")
    sys.exit(1)

def lemmatize_dictionary():
    print(f"📖 Reading {INPUT_FILE}...")
    
    # Store entries as a dictionary to handle duplicates automatically
    # Key = Word, Value = Score
    new_lexicon = {}
    
    # 1. Read existing words
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ Input file not found!")
        return

    print(f"⚙️  Lemmatizing {len(lines)} entries... (This takes 10-20s)")
    
    count_added = 0
    count_modified = 0

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        
        original_word = parts[0]
        try:
            score = float(parts[1])
        except ValueError: continue

        # A. Keep the original word (Safety net)
        # We keep the original because sometimes SpaCy makes mistakes or returns the same word
        if original_word not in new_lexicon:
            new_lexicon[original_word] = score

        # B. Generate the Lemma
        # We process the word in isolation to get its base form
        doc = nlp(original_word)
        lemma = doc[0].lemma_.lower()

        # C. Add the Lemma (if it's different)
        if lemma != original_word:
            # If the lemma isn't already in our list, add it with the SAME score
            if lemma not in new_lexicon:
                new_lexicon[lemma] = score
                count_added += 1
            else:
                # OPTIONAL: If lemma exists, could average the scores? 
                # For now, we trust the existing score or the first one we saw.
                pass

    # 2. Save the new "Super Lexicon"
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for word in sorted(new_lexicon.keys()):
            f.write(f"{word}\t{new_lexicon[word]}\n")

    print("\n✅ DONE!")
    print(f"Original entries: {len(lines)}")
    print(f"New lemmas added: {count_added}")
    print(f"Total size: {len(new_lexicon)}")
    print(f"Use '{OUTPUT_FILE}' in your benchmark script now!")

if __name__ == "__main__":
    lemmatize_dictionary()