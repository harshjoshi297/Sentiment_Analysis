import streamlit as st
import joblib
import re

st.set_page_config(page_title="Sentiment Analysis App")
st.write("✅ App file loaded")

st.title("📝 Sentiment Analysis App")

# ---- Load model safely ----
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model failed to load")
    st.exception(e)
    st.stop()

# ---- Cleaning ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---- UI ----
user_input = st.text_area("Enter a product review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")
