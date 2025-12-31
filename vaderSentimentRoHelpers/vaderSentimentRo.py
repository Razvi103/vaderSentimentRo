# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import os
import re
import spacy
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

# ROMANIAN NEGATION TERMS
# Includes: nu, n- (n-am), nici, deloc, etc. + variations without diacritics
NEGATE = \
    ["nu", "n", "n-", "nici", "niciodata", "niciodată", "nicidecum", 
     "deloc", "ba", "nimic", "nimeni", "nu-i", "nu-s", "n-ai", "n-are", 
     "n-au", "n-am", "ioc", "tman", "taman", "catusi"]

# ROMANIAN BOOSTER WORDS (Intensifiers)
# Words that increase intensity (Foarte bun) or decrease it (Cam bun)
BOOSTER_DICT = \
    {
     # INCREASERS (The "Very" equivalents)
     "foarte": B_INCR, "extrem": B_INCR, "absolut": B_INCR, 
     "complet": B_INCR, "total": B_INCR, "mult": B_INCR, 
     "prea": B_INCR, "super": B_INCR, "extra": B_INCR, 
     "ultra": B_INCR, "mega": B_INCR, "nemaipomenit": B_INCR,
     "incredibil": B_INCR, "fara margini": B_INCR, "maxim": B_INCR,
     "enorm": B_INCR, "teribil": B_INCR, "grozav": B_INCR,
     "strasnic": B_INCR, "strașnic": B_INCR, "tare": B_INCR,
     "beton": B_INCR, "si mai": B_INCR, "și mai": B_INCR,
     
     # DECREASERS (The "Kinda" equivalents)
     "cam": B_DECR, "oarecum": B_DECR, "putin": B_DECR, "puțin": B_DECR,
     "oleaca": B_DECR, "oleacă": B_DECR, "un pic": B_DECR,
     "vag": B_DECR, "aprox": B_DECR, "aproximativ": B_DECR,
     "aproape": B_DECR, "abia": B_DECR, "cat de cat": B_DECR,
     "cât de cât": B_DECR, "relativ": B_DECR, "partial": B_DECR, "parțial": B_DECR
    }

# ROMANIAN IDIOMS (Phrases that don't translate word-for-word)
# Future work: You can expand this list based on data you see.
SENTIMENT_LADEN_IDIOMS = {
    "floare la ureche": 2,  # Piece of cake
    "taie frunza la caini": -2, # Lazy (cuts leaves for dogs)
    "taie frunză la câini": -2,
    "frec menta": -2, # Lazy (rubbing mint)
    "fraca menta": -2,
    "freacă menta"
    "frecat menta": -2,
    "calca pe nervi": -2, # Gets on nerves
    "calcă pe nervi": -2,
    "cu capul in nori": -1, # Head in clouds (unfocused)
    "mana cereasca": 2, # Godsend
    "mână cerească": 2
}

# SLANG AND SPECIAL CASES
# Words that might be neutral in dictionary but have specific slang meaning
SPECIAL_CASES = {
    "varza": -2.0, "varză": -2.0,      # "Cabbage" -> Mess/Bad
    "beton": 2.5,                      # "Concrete" -> Awesome/Solid
    "marfa": 2.0, "marfă": 2.0,        # "Merchandise" -> Cool
    "misto": 2.0, "mișto": 2.0,        # Cool
    "nasol": -2.0, "naspa": -2.0,      # Bad/Nasty
    "tare": 1.5,                       # "Hard" -> Cool/Strong
    "blana": 2.0, "blană": 2.0,        # "Fur" -> Cool/Awesome
    "bomba": 2.5, "bombă": 2.5,        # "Bomb" -> Awesome
    "jale": -2.5,                      # "Mourning" -> Disaster
    "praf": -2.0,                      # "Dust" -> Useless/Destroyed
    "brici": 2.0,                      # "Razor" -> Sharp/Excellent
    "smecher": 2.0, "șmecher": 2.0,    # Sly/Cool
    "aiurea": -1.5,                    # Nonsense/Bad
    "horror": -2.5                     # Often used in RO as slang
}


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    Adapted for Romanian: checks NEGATE list and specific 'n-' prefixes.
    """
    input_words = [str(w).lower() for w in input_words]
    for word in NEGATE:
        if word in input_words:
            return True
    
    # Check for Romanian hyphenated negations (n-am, n-ai, n-au)
    # The tokenizer might split "n-am" into "n" and "am" or keep "n-am"
    if include_nt:
        for word in input_words:
            if word.startswith("n-"): 
                return True
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence 
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no") \
               or (i > 1 and words_and_emoticons[i - 2].lower() == "no") \
               or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"] ):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        # Check for negation case using "least" equivalents
        # English: "at least" -> Romanian: "macar", "cel putin", "barim"
        # If someone says "macar e bun" (at least it's good), it implies a constraint but usually positive.
        # If someone says "cel mai putin bun" (the least good), it negates.
        
        # This is complex in Romanian. A simple heuristic is checking for "putin" (little/less).
        
        if i > 0:
            prev = words_and_emoticons[i - 1].lower()
            
            # Logic for "cel mai putin" (the least) -> Negates
            if prev == "putin" or prev == "puțin":
                if i > 1:
                    prev2 = words_and_emoticons[i - 2].lower()
                    if prev2 == "mai" or prev2 == "cel":
                        valence = valence * N_SCALAR
                        
        return valence

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        # Romanian equivalents: dar, însă, ci, totuși
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        
        but_words = ["dar", "insa", "însă", "ci", "totusi", "totuși"]
        
        # Find the first occurrence of ANY "but" word
        bi = -1
        for b_word in but_words:
            if b_word in words_and_emoticons_lower:
                bi = words_and_emoticons_lower.index(b_word)
                break # Found one, stop looking
        
        if bi >= 0:
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASES:
                valence = SPECIAL_CASES[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "never" and \
                    (words_and_emoticons_lower[i - 1] == "so" or
                     words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "without" and \
                    words_and_emoticons_lower[i - 1] == "doubt":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "never" and \
                    (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "this") or \
                    (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "without" and \
                    (words_and_emoticons_lower[i - 2] == "doubt" or words_and_emoticons_lower[i - 1] == "doubt"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


if __name__ == '__main__':
    # 1. ÎNCARCĂ SPACY (Lematizatorul)
    print("⏳ Se încarcă modelul spaCy pentru limba română...")
    try:
        nlp = spacy.load("ro_core_news_sm")
    except OSError:
        print("EROARE: Trebuie să rulezi: python -m spacy download ro_core_news_sm")
        exit()

    # 2. INIȚIALIZEAZĂ VADER
    # Folosește fișierul CURĂȚAT la Pasul 2
    analyzer = SentimentIntensityAnalyzer(
        lexicon_file="../vader_lexicon_ro_clean.txt", 
        emoji_lexicon="../emoji_utf8_lexicon_ro.txt"
    )

    print("\n----------------------------------------------------")
    print("ROMANIAN VADER (CU LEMATIZARE) - TEST SUITE")
    print("----------------------------------------------------")

    test_sentences = [
        "Mâncarea este excelentă.",        
        "Filmul a fost oribil.",           
        "Nu este bine.",                   
        "Nu e rău deloc.",                
        "N-am nicio problemă.",            
        "Ești foarte deștept.",            
        "Ești extrem de deștept!",         
        "E puțin plictisitor.",            
        "Ești varză la jocul ăsta.",       
        "Mașina asta e beton.",           
        "A fost o experiență mișto.",      
        "Ideea e bună, dar execuția e proastă.",
        "Vremea este insuportabilă.",      
        "Mă simt minunat! 😀"
    ]

    # --- FUNCȚIA MAGICĂ: PREPROCESARE ---
    def get_vader_score_ro(text):
        # A. Lematizare cu spaCy
        doc = nlp(text)
        
        # B. Reconstruim propoziția folosind LEMMA (forma de bază)
        # Ex: "Mâncarea este excelentă" -> "mâncare fi excelent"
        # Păstrăm pronumele și negatorii exact cum sunt, lematizăm doar adjective/substantive/verbe
        lemma_words = []
        for token in doc:
            # Truc: Dacă e 'nu', 'n-', păstrează-l original (pentru detectarea negației)
            if token.text.lower().startswith('n'):
                lemma_words.append(token.text.lower())
            elif token.lemma_ == "fi": # Verbul 'a fi' poate deruta uneori, dar e ok de obicei
                lemma_words.append(token.lemma_)
            else:
                lemma_words.append(token.lemma_)
                
        text_lemmatized = " ".join(lemma_words)
        
        # C. Calculăm scorul pe textul lematizat
        scores = analyzer.polarity_scores(text_lemmatized)
        return text_lemmatized, scores

    # --- RULARE TESTE ---
    for sentence in test_sentences:
        lemma_text, vs = get_vader_score_ro(sentence)
        print(f"INPUT ORIGINAL:  {sentence}")
        print(f"VADER VEDE:      {lemma_text}")
        print(f"SCOR:            {str(vs)}")
        print("-" * 60)