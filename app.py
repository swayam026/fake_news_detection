# app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection System")

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.success("Model loaded successfully!")

# --- Optional: show training performance ---
st.subheader("📊 Model Evaluation Summary")
try:
    cm = pd.read_csv("confusion_matrix.csv")  # optional if you save cm as csv in train.py
except:
    cm = None

col1, col2 = st.columns(2)
with col1:
    st.image("confusion_matrix.png", caption="Confusion Matrix (from training)", use_container_width=True)

with col2:
    st.write("**This model uses TF-IDF + Logistic Regression.**")
    st.write("- Dataset: True.csv + Fake.csv")
    st.write("- Evaluation done on 20% test data split.")
    st.write("- Accuracy shown above from training script.")

st.markdown("---")

# --- Prediction UI ---
st.subheader("🧠 Try It Yourself")
news_text = st.text_area("Enter News Headline or Article Text:", height=200, placeholder="Type or paste news here...")

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text!")
    else:
        prediction = model.predict([news_text])[0]
        proba = model.predict_proba([news_text])[0]
        st.markdown("### 🔍 Prediction Result")
        if prediction == "FAKE":
            st.error(f"Fake News Detected! (Confidence: {proba[0]*100:.2f}%)")
        else:
            st.success(f"Real News Detected! (Confidence: {proba[1]*100:.2f}%)")

# --- Footer Credit ---
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made by <b>Swayam Agarwal</b>
    </div>
    """,
    unsafe_allow_html=True
)
