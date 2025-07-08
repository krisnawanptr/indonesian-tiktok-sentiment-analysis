import os
import sys

# Tambahkan path project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from google_play_scraper import reviews, Sort

from preprocessing import full_preprocess
from utils import generate_wordcloud, get_top_words

# ==================== Load Model & Vectorizer ====================
@st.cache_resource
def load_model():
    model = joblib.load('../models/svm_model.pkl')
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ==================== Streamlit Config ====================
st.set_page_config(page_title="Analisis Sentimen Real-Time", layout="wide")
st.title("📊 Real-Time Analisis Sentimen Komentar TikTok")

# ==================== Input Jumlah Komentar ====================
jumlah = st.number_input(
    "Masukkan jumlah komentar terbaru yang ingin dianalisis:",
    min_value=10, max_value=5000, value=100, step=10
)

if st.button("Ambil dan Analisis Komentar"):
    with st.spinner("🔄 Mengambil komentar dari Google Play Store..."):
        try:
            result, _ = reviews(
                'com.ss.android.ugc.trill',  # TikTok Indonesia
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=int(jumlah),
                filter_score_with=None
            )
            df = pd.DataFrame(result)
        except Exception as e:
            st.error(f"Gagal mengambil data: {e}")
            st.stop()

    if df.empty or 'content' not in df.columns:
        st.error("❌ Data tidak valid atau kosong.")
        st.stop()

    # ==================== Tampilkan Data Asli ====================
    st.subheader("📝 Data Komentar Asli")
    st.dataframe(df[['content']].head())

    # ==================== Preprocessing & Prediksi ====================
    with st.spinner("🔧 Melakukan preprocessing dan prediksi..."):
        df['cleaned'] = df['content'].astype(str).apply(full_preprocess)
        X = vectorizer.transform(df['cleaned'])
        df['sentimen'] = model.predict(X)

    st.success("✅ Analisis Sentimen Berhasil!")

    # ==================== Hasil Analisis ====================
    st.subheader("📄 Hasil Analisis Sentimen")
    st.dataframe(df[['content', 'sentimen']])

    st.markdown("### 📊 Distribusi Sentimen")
    distribusi = df['sentimen'].value_counts().reset_index()
    distribusi.columns = ['Sentimen', 'Jumlah']
    st.table(distribusi)

    # ==================== Visualisasi WordCloud ====================
    st.markdown("## 🔍 Visualisasi WordCloud & Kata Terbanyak")

    label_order = ['Negative', 'Positive', 'Neutral']
    label_colors = {'Negative': 'red', 'Positive': 'green', 'Neutral': 'gray'}

    for label in label_order:
        subset = df[df['sentimen'] == label]
        if subset.empty:
            continue

        all_text = " ".join(subset['cleaned'].dropna().astype(str))

        if not all_text.strip():
            st.warning(f"Tidak cukup data bersih untuk WordCloud '{label}'.")
            continue

        st.markdown(f"### 💬 Sentimen: <span style='color:{label_colors[label]}'><b>{label}</b></span>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ☁️ WordCloud")
            wc = generate_wordcloud(all_text)
            if wc:
                st.image(wc.to_array(), use_container_width=True)
            else:
                st.warning("WordCloud tidak bisa ditampilkan.")

        with col2:
            st.markdown("#### 📈 Top 10 Kata")
            top_words_df = get_top_words(all_text, n=10)
            if not top_words_df.empty:
                st.table(top_words_df)
            else:
                st.warning("Tidak ada kata dominan.")

    # ==================== Unduh Hasil ====================
    csv = df[['content', 'sentimen']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Analisis", csv, "hasil_sentimen.csv", "text/csv")
