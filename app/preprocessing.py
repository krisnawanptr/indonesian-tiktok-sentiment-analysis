# app/preprocessing.py

import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_word_elongation

# --- Cleaning Functions ---
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else text

def remove_html(text):
    return re.sub(r'<.*?>', '', text) if isinstance(text, str) else text

def remove_emoji(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF" u"\U0001FA70-\U0001FAFF" u"\U0001F000-\U0001F02F"
            u"\U0001F0A0-\U0001F0FF" u"\U0001F650-\U0001F67F" u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF" u"\U00002600-\U000026FF" u"\U00002B00-\U00002BFF"
            u"\u2600-\u26FF" u"\u2700-\u27BF" u"\uFE0F" u"\u3030"
        "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    return text

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text) if isinstance(text, str) else text

def remove_number(text):
    return re.sub(r'\d+', '', text) if isinstance(text, str) else text

def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def normalize_elongated_words(text):
    return replace_word_elongation(text) if isinstance(text, str) else text

# --- Load Kamus Kata Baku ---
kamus_df = pd.read_excel('kamuskatabaku.xlsx')
kamus_dict = dict(zip(kamus_df['tidak_baku'], kamus_df['kata_baku']))

def normalize_words(text, kamus=kamus_dict):
    if isinstance(text, str):
        return " ".join([kamus.get(w, w) for w in text.split()])
    return text

def tokenize(text):
    return text.split() if isinstance(text, str) else []

# --- Stopword Removal ---
stop_factory = StopWordRemoverFactory()
default_stopwords = set(stop_factory.get_stop_words())
keep_words = {'tidak', 'bukan', 'belum', 'jangan'}
custom_stopwords = default_stopwords - keep_words

def stopwords_removal(tokens):
    return [w for w in tokens if w.lower() not in custom_stopwords] if isinstance(tokens, list) else []

# --- Stemming ---
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming(tokens):
    return [stemmer.stem(w) for w in tokens] if isinstance(tokens, list) else []

# --- Final Preprocessing Pipeline ---
def full_preprocess(text):
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_number(text)
    text = case_folding(text)
    text = normalize_elongated_words(text)
    text = normalize_words(text)
    tokens = tokenize(text)
    tokens = stopwords_removal(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)
