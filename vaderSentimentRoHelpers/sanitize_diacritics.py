import pandas as pd
import unicodedata

# Input/Output filenames
INPUT_FILE = 'vader_lexicon_ro.txt'
OUTPUT_FILE = 'vader_lexicon_ro_final.txt'

def remove_diacritics(text):
    """
    Replaces Romanian special characters with their ASCII equivalents.
    Handles both standard (comma-below) and legacy (cedilla) versions.
    """
    replacements = {
        'ă': 'a', 'Ă': 'A',
        'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I',
        'ș': 's', 'Ș': 'S', 'ş': 's', 'Ş': 'S', # Both comma and cedilla
        'ț': 't', 'Ț': 'T', 'ţ': 't', 'Ţ': 'T'  # Both comma and cedilla
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def has_diacritics(text):
    """Checks if the text contains any Romanian diacritics."""
    special_chars = set("ăâîșțşţĂÂÎȘȚŞŢ")
    return any(char in special_chars for char in text)

# 1. Load the translated lexicon
# We use 'header=None' because VADER lexicons don't usually have headers
# The columns are: Token | Score | Std Dev | Ratings
try:
    df = pd.read_csv(INPUT_FILE, sep='\t', header=None, names=['token', 'score', 'std', 'ratings'])
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Make sure you ran the translation script first.")
    exit()

final_lexicon = []
count_new_entries = 0

print("Processing lexicon...")

for index, row in df.iterrows():
    token = str(row['token'])
    score = row['score']
    
    # Add the original entry (ALWAYS keep the original)
    final_lexicon.append({'token': token, 'score': score})
    
    # Check if we need to create a non-diacritic twin
    if has_diacritics(token):
        clean_token = remove_diacritics(token)
        
        # Only add if the clean version is actually different
        if clean_token != token:
            final_lexicon.append({'token': clean_token, 'score': score})
            count_new_entries += 1

# 2. Create DataFrame
df_final = pd.DataFrame(final_lexicon)

# 3. Deduplicate
# It is possible that 'rau' was already in the file from the translation
# We keep the first occurrence.
initial_count = len(df_final)
df_final.drop_duplicates(subset=['token'], inplace=True)
final_count = len(df_final)

# 4. Save
# Important: Ensure UTF-8 encoding
df_final.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False, encoding='utf-8')

print("--- SUMMARY ---")
print(f"Original entries processed: {len(df)}")
print(f"New non-diacritic entries generated: {count_new_entries}")
print(f"Duplicates removed: {initial_count - final_count}")
print(f"Final Lexicon Size: {final_count}")
print(f"Saved to: {OUTPUT_FILE}")