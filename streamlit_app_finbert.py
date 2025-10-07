import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Page setup ---
st.set_page_config(page_title="Headline Analyzer (FinBERT)", page_icon="🧠", layout="centered")
st.title("🧠 AI Headline Analyzer (FinBERT Edition)")
st.caption("Paste a market or financial headline to see instant Bullish / Bearish / Neutral sentiment.")

# --- Load model ---
@st.cache_resource
def load_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- Input ---
headline = st.text_input("Enter a financial headline:", placeholder="e.g., Bank of England warns against rolling back regulations")

if st.button("Analyze Sentiment") and headline:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        labels = ['Bearish', 'Neutral', 'Bullish']
        pred_idx = torch.argmax(probs, dim=1).item()
        sentiment = labels[pred_idx]
        confidence = probs[0][pred_idx].item() * 100

    st.subheader("📊 Result")
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")

    if sentiment == "Bullish":
        st.success("Market tone appears optimistic 📈")
    elif sentiment == "Bearish":
        st.error("Market tone appears cautious 📉")
    else:
        st.info("Market tone appears neutral ⚖️")

