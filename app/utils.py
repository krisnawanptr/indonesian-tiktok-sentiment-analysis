from wordcloud import WordCloud
from collections import Counter
import pandas as pd

def generate_wordcloud(text):
    if not text.strip():
        return None
    return WordCloud(width=800, height=400, background_color='white').generate(text)

def get_top_words(text, n=10):
    words = text.split()
    word_freq = Counter(words).most_common(n)
    return pd.DataFrame(word_freq, columns=['Kata', 'Frekuensi'])
