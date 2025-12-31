import os

# --- CONFIGURARE ---
INPUT_FILE = "vader_lexicon_ro_final.txt"
OUTPUT_FILE = "vader_lexicon_ro_clean.txt"

# 1. LISTA NEAGRĂ (Cuvinte de șters din dicționarul .txt)
# Le ștergem pentru că le gestionezi tu prin cod (NEGATE, BOOSTER)
BLOCKLIST = {
    # Negatori (din lista ta)
    "nu", "n", "n-", "nici", "niciodata", "niciodată", "nicidecum", 
    "deloc", "ba", "nimic", "nimeni", "nu-i", "nu-s", "n-ai", "n-are", 
    "n-au", "n-am", "ioc", "tman", "taman", "catusi",
    
    # Boostere (din lista ta)
    "foarte", "extrem", "absolut", "complet", "total", "mult", 
    "prea", "super", "extra", "ultra", "mega", "nemaipomenit",
    "incredibil", "fara margini", "maxim", "enorm", "teribil", "grozav",
    "strasnic", "strașnic", "tare", "beton", "si mai", "și mai",
    "cam", "oarecum", "putin", "puțin", "oleaca", "oleacă", "un pic",
    "vag", "aprox", "aproximativ", "aproape", "abia", "cat de cat",
    "cât de cât", "relativ", "partial", "parțial"
}

# 2. LISTA DE ADAUGĂRI (Injectăm exact SPECIAL_CASES ale tale)
INJECTIONS = {
    "varza": -2.0, "varză": -2.0,
    "beton": 2.5,
    "marfa": 2.0, "marfă": 2.0,
    "misto": 2.0, "mișto": 2.0,
    "nasol": -2.0, "naspa": -2.0, "nașpa": -2.0,
    "tare": 1.5,
    "blana": 2.0, "blană": 2.0,
    "bomba": 2.5, "bombă": 2.5,
    "jale": -2.5,
    "praf": -2.0,
    "brici": 2.0,
    "smecher": 2.0, "șmecher": 2.0,
    "aiurea": -1.5,
    "horror": -2.5,
    # Adăugiri de siguranță pentru Lematizare SpaCy
    "proast": -2.0, "insuportabil": -2.0
}

def clean_and_inject():
    print(f"🔧 Procesez fișierul {INPUT_FILE} folosind listele tale...")
    
    final_lexicon = {}
    
    # A. CITIREA
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 2: continue
                
                word = parts[0].lower()
                try:
                    score = float(parts[1])
                except ValueError: continue

                # Dacă e în lista ta de logică, îl scoatem din fișierul de scoruri
                if word in BLOCKLIST:
                    continue
                
                final_lexicon[word] = score
                
    except FileNotFoundError:
        print(f"❌ Nu găsesc {INPUT_FILE}")
        return

    # B. INJECTAREA ARGOULUI TĂU
    print("💉 Injectez lista ta de 'Special Cases'...")
    for word, score in INJECTIONS.items():
        final_lexicon[word] = score

    # C. SALVAREA
    print(f"💾 Salvez în {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for word in sorted(final_lexicon.keys()):
            f.write(f"{word}\t{final_lexicon[word]}\n")

    print("✅ Gata. Lexiconul este sincronizat cu codul tău.")

if __name__ == "__main__":
    clean_and_inject()