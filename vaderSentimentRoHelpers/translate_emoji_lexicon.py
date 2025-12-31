import pandas as pd
from deep_translator import GoogleTranslator

# 1. Load the Emoji Lexicon
# (Assuming the file is named 'emoji_utf8_lexicon.txt')
# It usually has no header, just: Emoji \t Description
df = pd.read_csv('./vaderSentiment/emoji_utf8_lexicon.txt', sep='\t', header=None, names=['emoji', 'description'])

translator = GoogleTranslator(source='en', target='ro')

print("Translating emoji descriptions... (approx 3000 items)")
ro_emoji_map = []

for index, row in df.iterrows():
    emoji_char = row['emoji']
    en_desc = row['description']
    
    try:
        # Translate the description (e.g., "grinning face" -> "față rânjind")
        ro_desc = translator.translate(en_desc)
        
        if ro_desc:
            ro_emoji_map.append({'emoji': emoji_char, 'description': ro_desc.lower()})
        else:
            # Fallback
            ro_emoji_map.append({'emoji': emoji_char, 'description': en_desc})
            
    except Exception as e:
        print(f"Error at index {index}: {e}")
        ro_emoji_map.append({'emoji': emoji_char, 'description': en_desc})

    if index % 100 == 0:
        print(f"Processed {index} emojis...")

# 2. Save the new Romanian Emoji Lexicon
df_ro = pd.DataFrame(ro_emoji_map)
df_ro.to_csv('emoji_utf8_lexicon_ro.txt', sep='\t', index=False, header=False)

print("Emoji translation complete!")