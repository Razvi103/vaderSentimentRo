import pandas as pd
from deep_translator import GoogleTranslator
import re
import time

# Load the file
df = pd.read_csv('./vaderSentiment/vader_lexicon.txt', sep='\t', header=None, names=['token', 'score', 'std', 'ratings'])

translator = GoogleTranslator(source='en', target='ro')
ro_lexicon = []

print("Starting processing...")

for index, row in df.iterrows():
    token = str(row['token'])
    score = row['score']
    
    # CHECK: Does the token contain any letters?
    # If NO (it's ":-)", "$:", "0_0"), keep it as is.
    if not re.search('[a-zA-Z]', token):
        ro_lexicon.append({'token': token, 'score': score})
        continue # Skip the translation part
    
    # If we are here, it's likely a word (e.g., "amazing")
    try:
        # Translate
        ro_word = translator.translate(token)
        
        if ro_word:
            ro_word = ro_word.lower()
            ro_lexicon.append({'token': ro_word, 'score': score})
            
            # CRITICAL: Also keep the original English word?
            # Romanians use "cool", "wow", "party", "weekend" often.
            # It is safer to keep the English version too.
            ro_lexicon.append({'token': token, 'score': score})
            
    except Exception as e:
        print(f"Error translating {token}: {e}")
        # If translation fails, keep the original to be safe
        ro_lexicon.append({'token': token, 'score': score})

    # Progress tracker
    if index % 500 == 0:
        print(f"Processed {index} rows...")

# Save
df_ro = pd.DataFrame(ro_lexicon)
df_ro.drop_duplicates(subset=['token'], inplace=True)
df_ro.to_csv('vader_lexicon_ro.txt', sep='\t', index=False, header=False)

print("Done! Emoticons preserved, words translated.")