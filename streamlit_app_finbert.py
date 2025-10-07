# streamlit_app_finbert.py
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Page ----------
st.set_page_config(page_title="Headline Analyzer (FinBERT)", page_icon="🧠", layout="centered")
st.title("🧠 AI Headline Analyzer — FinBERT + Sector Tag")
st.caption("Paste any market/financial headline → AI returns Bullish/Bearish/Neutral, confidence, and sector tag.")

# ---------- Load FinBERT (cached) ----------
@st.cache_resource(show_spinner=True)
def load_finbert():
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tok, mdl

tokenizer, model = load_finbert()
id2label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

# ---------- Sector detection (simple keyword map) ----------
SECTOR_MAP = {
    "Tech":        [r"\bAI\b", r"\bchip(s)?\b", r"\bsemiconductor(s)?\b", r"\bGPU\b", r"\bcloud\b", r"\bsoftware\b", r"\bsmartphone(s)?\b"],
    "Consumer":    [r"\bretail\b", r"\bconsumer\b", r"\bstore(s)?\b", r"\bapparel\b", r"\bgrocery\b", r"\be-commerce\b"],
    "Auto":        [r"\bEV\b", r"\bvehicle(s)?\b", r"\bautomaker(s)?\b", r"\bdealership(s)?\b"],
    "Energy":      [r"\boil\b", r"\bWTI\b", r"\bnatural gas\b", r"\bOPEC\b", r"\brefiner(y|ies)\b"],
    "Financials":  [r"\bbank(s)?\b", r"\binsurer(s)?\b", r"\bcredit\b", r"\bfintech\b", r"\bcentral bank\b", r"\bBank of England\b", r"\bFederal Reserve\b", r"\bECB\b"],
    "Healthcare":  [r"\bFDA\b", r"\btrial\b", r"\bphase (II|III)\b", r"\bdrug\b", r"\btherapy\b", r"\bbiotech\b"],
    "Industrial":  [r"\bmanufactur(e|ing)\b", r"\bsupply chain\b", r"\blogistics\b", r"\bfreight\b"],
    "Communications": [r"\bmedia\b", r"\bstreaming\b", r"\badvertising\b", r"\bsocial\b", r"\btelecom\b"],
}

def detect_sector(text: str) -> str:
    t = text.lower()
    for sector, patterns in SECTOR_MAP.items():
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                return sector
    return "General"

# ---------- Tiny finance-aware hints (optional nudge) ----------
BEARISH_HINTS = [
    r"\bwarn(s|ed|ing)?\b", r"\bcut(s|ting)?\b", r"\bguidance cut(s|ting)?\b", r"\bprobe(s|d)?\b",
    r"\binvestigat(e|ion|ed|es)\b", r"\brecall(s|ed|ing)?\b", r"\blayoff(s)?\b", r"\bplunge(s|d)?\b"
]
BULLISH_HINTS = [
    r"\braise(s|d|ing)?\b", r"\bbeat(s|en)?\b", r"\bboost(s|ed|ing)?\b", r"\bupgrade(s|d)?\b",
    r"\bstrong demand\b", r"\brecord\b", r"\baccelerat(e|es|ed|ing)\b"
]

def rule_bias(text: str) -> float:
    # small nudge: +0.12 for bullish phrase, -0.12 for bearish phrase (kept tiny so FinBERT leads)
    bias = 0.0
    if any(re.search(p, text, re.I) for p in BULLISH_HINTS): bias += 0.12
    if any(re.search(p, text, re.I) for p in BEARISH_HINTS): bias -= 0.12
    return bias

# ---------- Inference ----------
def analyze(headline: str):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    base_conf = float(probs[pred_idx].item())  # 0..1
    sentiment = id2label[pred_idx]

    # apply tiny rule bias by nudging the confidence toward bullish/bearish
    bias = rule_bias(headline)
    if sentiment == "Bullish" and bias < 0:
        adj = max(0.0, base_conf + bias)  # reduce a bit
    elif sentiment == "Bearish" and bias > 0:
        adj = max(0.0, base_conf - bias)  # reduce a bit
    else:
        adj = min(1.0, max(0.0, base_conf + (bias if sentiment != "Neutral" else 0)))

    confidence_pct = round(adj * 100, 1)
    sector = detect_sector(headline)

    return sentiment, confidence_pct, sector, bias

# ---------- Smart explanation blocks ----------
def render_explanation(sentiment: str, sector: str, confidence: float, bias: float, headline: str):
    # Headline echo
    st.write(f"**Headline:** {headline}")

    # Richer explanation text
    if sentiment == "Bullish":
        st.success("Market tone appears optimistic 📈")
        st.write(
            "This headline communicates positive expectations or improving conditions. "
            "Phrases like *raise, beat, strong demand, upgrade* tend to attract risk-on flows, "
            "supporting momentum or follow-through buying if price and volume confirm. "
            f"In the **{sector}** sector, optimism often translates to stronger short-term appetite for growth exposure."
        )
        st.write(
            "**Suggested lens:** Watch for opening drive strength, pullback-to-VWAP entries, "
            "or relative strength vs. sector peers. Manage risk around previous day highs and key levels."
        )
    elif sentiment == "Bearish":
        st.error("Market tone appears cautious 📉")
        st.write(
            "Language such as *warns, cuts, probe, decline* signals uncertainty or risk aversion. "
            "Headlines like this can prompt de-risking, mean-reversion fades, or rotation into defensives, "
            f"especially within **{sector}** where policy, regulation, or funding conditions matter. "
            "Expect choppier tape unless other catalysts offset the tone."
        )
        st.write(
            "**Suggested lens:** Consider lower-highs/back-test failures, VWAP rejections, or put hedges. "
            "Be mindful of oversold snapbacks if the move extends too quickly."
        )
    else:
        st.info("Market tone appears neutral ⚖️")
        st.write(
            "Tone appears balanced — not clearly risk-on or risk-off. "
            "Neutral headlines often reflect scheduled events, in-line data, or non-committal language. "
            "Price usually waits for additional catalysts to set direction."
        )
        st.write(
            "**Suggested lens:** Let price lead. Focus on range edges, liquidity zones, or confirmation from volume/market breadth."
        )

    # Tiny transparency note about rules
    if abs(bias) > 0:
        st.caption("ℹ️ Note: A small rules-based nudge was applied for finance phrases (e.g., 'raises guidance', 'warns'). FinBERT remains the primary signal.")

    # Placeholder historical insight (MVP-friendly)
    st.caption(
        "📈 Historical pattern insight (preview): Similar-toned headlines have *sometimes* been followed by "
        "short-term continuation when volume confirms. A full backtested view of analogous headlines "
        "is coming soon."
    )

# ---------- UI ----------
with st.form("analyze"):
    headline = st.text_input("Headline", placeholder="e.g., NVIDIA raises revenue guidance on strong AI demand")
    submitted = st.form_submit_button("Analyze")

if submitted and headline.strip():
    sentiment, conf, sector, bias = analyze(headline.strip())

    st.subheader("📊 Result")
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Confidence:** {conf:.1f}%")
    st.markdown(f"**Sector:** {sector}")

    render_explanation(sentiment, sector, conf, bias, headline.strip())

st.caption("Educational tool — not financial advice.")
