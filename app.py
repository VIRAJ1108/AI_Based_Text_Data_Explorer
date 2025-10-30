import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import spacy
from nltk.corpus import stopwords
from nltk import word_tokenize, download

# --- Setup ---
download('stopwords')
download('punkt')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# --- Helper functions ---
def preprocess_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(tokens)

def get_frequent_words(texts, n=20):
    all_words = " ".join(texts).split()
    return Counter(all_words).most_common(n)

def plot_frequent_words(freq_words):
    if not freq_words:
        st.warning("No words to display!")
        return
    words, counts = zip(*freq_words)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=list(words), y=list(counts), ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def generate_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("🧠 AI-powered Text Data Explorer")

st.write("Upload a CSV file containing text data (e.g., tweets, articles).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

    st.write("### Select the text column:")
    text_column = st.selectbox("Choose column", data.columns)

    if st.button("Start Analysis"):
        texts = data[text_column].dropna().tolist()[:1000]
        st.write("Preprocessing text data...")
        cleaned_texts = [preprocess_text(t) for t in texts]

        st.subheader("🔠 Most Frequent Words")
        freq_words = get_frequent_words(cleaned_texts)
        plot_frequent_words(freq_words)

        st.subheader("☁️ Word Cloud")
        generate_wordcloud(cleaned_texts)

        st.success("Analysis complete!")

else:
    st.info("Please upload a CSV file to begin.")
