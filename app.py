import streamlit as st
import joblib
import re
import nltk
import requests
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load ML model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set page config and custom background
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #ffe4b5;
        }
        .card {
            background-color: #fff5e6;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .black-warning {
            background-color: #f9f9f9;
            color: black;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-weight: bold;
            margin-top: 15px;
        }
        .stTextArea label {
            font-weight: bold;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: black;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #444;'>Check if the news is Real or Fake </h4>", unsafe_allow_html=True)
st.write("")  # Spacer

# API Key
NEWS_API_KEY = "dccdc2e9-0420-4d6a-9ee8-518939c911bf"

# ---------- Text Cleaner ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------- ML Prediction ----------
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)
    return "Real News" if result[0] == 1 else "Fake News"

# ---------- NewsAPI Check ----------
def is_news_real_with_newsapi(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
            return "Real News"
        else:
            return "Fake or Not Found"
    except Exception as e:
        return f"Error: {e}"

# ---------- Session State ----------
if "prediction" not in st.session_state:
    st.session_state["prediction"] = ""
if "ai_result" not in st.session_state:
    st.session_state["ai_result"] = ""
if "api_result" not in st.session_state:
    st.session_state["api_result"] = ""

# Special workaround to reset input
clear_trigger = st.session_state.get("clear_trigger", False)
default_text = "" if clear_trigger else st.session_state.get("news_text", "")

# ---------- Main Card ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Text Area
news_input = st.text_area(
    "Enter News Content (Full or Headline):",
    value=default_text,
    key="news_text",
    height=200
)

# Buttons under text input
col1, col2 = st.columns(2)
with col1:
    check_btn = st.button("Verify News")
with col2:
    clear_btn = st.button("Clear Input")

# Clear Button Logic
if clear_btn:
    st.session_state.clear()
    st.session_state["clear_trigger"] = True
    st.rerun()

# ---------- Prediction Logic ----------
if check_btn:
    if not news_input.strip():
        st.markdown('<div class="black-warning">⚠️ Please enter some news content.</div>', unsafe_allow_html=True)
    else:
        ai_result = predict_news(news_input)
        api_result = is_news_real_with_newsapi(news_input, NEWS_API_KEY)

        st.session_state["ai_result"] = ai_result
        st.session_state["api_result"] = api_result

        # Final Verdict
        if ai_result == "Real News" or api_result == "Real News":
            st.session_state["prediction"] = "✅ Final Verdict: Real News"
        elif "Error" in api_result or "Invalid" in api_result:
            st.session_state["prediction"] = f"⚠️ NewsAPI Issue: {api_result}\n🤖 AI Model Prediction: {ai_result}"
        else:
            st.session_state["prediction"] = "❌ Final Verdict: Fake News"

# ---------- Output ----------
if st.session_state.get("prediction"):
    st.markdown(f'<div class="black-warning">{st.session_state["prediction"]}</div>', unsafe_allow_html=True)

# ---------- Details ----------
if st.session_state.get("ai_result") and st.session_state.get("api_result"):
    with st.expander("🔍 See Detailed Results"):
        st.markdown(f"**AI Model Prediction:** {st.session_state['ai_result']}")
        st.markdown(f"**NewsAPI Check:** {st.session_state['api_result']}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    "<div style='text-align: center; color: #555; margin-top: 30px;'>"
    "Built with Streamlit, ML & NewsAPI | © 2025 Jyothi"
    "</div>", unsafe_allow_html=True
)
